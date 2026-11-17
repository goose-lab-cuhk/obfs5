package tunnel

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	mathrand "math/rand/v2"
	"net"
	"sync"
	"time"

	"obfsgen/config"
	"obfsgen/engine"
	"obfsgen/scheduler"
)

type ClientConfig struct {
	Server     string         // host:port
	ServerName string         // TLS SNI; also verified unless Pin is set
	Pin        string         // hex SHA-256 of the server certificate (self-signed servers)
	PSK        []byte         // pre-shared key, authenticates the hello
	Conns      int            // tunnels (TLS connections) open at the same time (default 16)
	Sync       bool           // synchronous mode instead of asynchronous
	Active     func() bool    // whether the layers above have open streams (may be nil)
	Models     *engine.Models // the models and the opener distribution
	Config     *config.Config // the settings (config.yaml)
	Logf       func(format string, args ...any)
}

// Client: a UDP socket to KCP.
//
//	send:    KCP ==WriteTo==> out (buffer) ==pull==> tunnel ==records==> TLS ==> server
//	receive: KCP <==ReadFrom== in (chan) <==onPacket== tunnel <==records== TLS <== server
//
// Conns workers each do:
//
//	+--> wait until there is something to carry (a packet, or open streams)
//	|    dial TCP ==> TLS handshake (= RTT) ==> tunnel.run: one schedule ==> close
//	+--- repeat
type Client struct {
	cfg       ClientConfig
	sessionID uint64      // tells the server which tunnels belong together
	out       *buffer     // packets waiting for a record
	joiner    *joiner     // joins cut segments from the server
	in        chan []byte // packets from the server, read by KCP

	mu       sync.Mutex
	lastRecv time.Time // when the last packet came from the server

	ctx    context.Context // cancelled by Close
	cancel context.CancelFunc
}

func NewClient(cfg ClientConfig) *Client {
	if cfg.Conns <= 0 {
		cfg.Conns = 16
	}
	if cfg.Logf == nil {
		cfg.Logf = func(string, ...any) {}
	}
	var id [8]byte
	rand.Read(id[:])

	c := &Client{
		cfg:       cfg,
		sessionID: binary.LittleEndian.Uint64(id[:]),
		out:       newBuffer(),
		joiner:    newJoiner(),
		in:        make(chan []byte, 1024),
	}
	c.ctx, c.cancel = context.WithCancel(context.Background())
	for i := 0; i < cfg.Conns; i++ {
		go c.worker(i)
	}
	return c
}

func (c *Client) worker(i int) {
	for c.ctx.Err() == nil {
		wakeup := c.out.waitForPush() // before the check: no push is missed

		streamsOpen := c.cfg.Active != nil && c.cfg.Active()
		if c.out.isEmpty() && !streamsOpen {
			// idle: sleep until KCP writes (a new stream writes its SYN)
			select {
			case <-wakeup:
			case <-c.ctx.Done():
			}
			continue
		}

		err := c.runTunnel()
		if err != nil && c.ctx.Err() == nil {
			c.cfg.Logf("tunnel %d: %v", i, err)
			time.Sleep(time.Second)
		}
	}
}

func (c *Client) runTunnel() error {
	// 1. TCP + TLS. The TLS 1.3 handshake is one round trip end to end, so
	// it gives R even where a middlebox or relay answers the TCP handshake.
	dialCtx, cancel := context.WithTimeout(c.ctx, c.cfg.Config.Tunnel.HandshakeTimeout.Duration())
	defer cancel()
	raw, err := (&net.Dialer{}).DialContext(dialCtx, "tcp", c.cfg.Server)
	if err != nil {
		return err
	}
	defer raw.Close()
	raw.SetDeadline(time.Now().Add(c.cfg.Config.Tunnel.MaxLife.Duration())) // a stuck tunnel ends here

	conn := tls.Client(raw, c.tlsConfig())
	t0 := time.Now()
	if err := conn.HandshakeContext(dialCtx); err != nil {
		return err
	}
	rtt := time.Since(t0)

	// 2. opener: drawn from the seed, big enough for the hello
	random := mathrand.New(mathrand.NewPCG(mathrand.Uint64(), mathrand.Uint64()))
	m := c.cfg.Models
	seed := random.Uint64()
	opener := m.Opener.Draw(seed, minOpener)
	h := hello{sessionID: c.sessionID, seed: seed, rttMicros: uint32(rtt.Microseconds())}

	t := &tunnel{
		conn:    conn,
		out:     c.out,
		deliver: c.onPacket,
		joiner:  c.joiner,
		hello:   h.encode(c.cfg.PSK),
	}

	// 3. scheduler
	if c.cfg.Sync {
		// sync: the server samples the same plan (opener and plan from the seed)
		t.sync = scheduler.NewSync(scheduler.NewPlan(m.Sync, m.SyncTok, opener, seed), true)
		t.next = t.sync.First()
	} else {
		// async
		t.async = scheduler.NewAsync(m.Async, m.AsyncTok, scheduler.ClientRole, rtt.Seconds(), random, c.cfg.Config.Scheduler)
		t.next = t.async.Open(opener)
	}

	// 4. play, then close
	t.start = time.Now()
	done := make(chan struct{})
	defer close(done)
	return t.run(c.ctx, readRecords(conn, done))
}

func (c *Client) tlsConfig() *tls.Config {
	cfg := &tls.Config{
		ServerName:                  c.cfg.ServerName,
		MinVersion:                  tls.VersionTLS13,
		NextProtos:                  []string{"h2", "http/1.1"},
		DynamicRecordSizingDisabled: true, // one Write = one record of our size
	}
	if c.cfg.Pin != "" {
		// self-signed: check the certificate hash
		want, _ := hex.DecodeString(c.cfg.Pin)
		cfg.InsecureSkipVerify = true
		cfg.VerifyConnection = func(cs tls.ConnectionState) error {
			if len(cs.PeerCertificates) == 0 {
				return errors.New("no certificate")
			}
			got := sha256.Sum256(cs.PeerCertificates[0].Raw)
			if !bytes.Equal(got[:], want) {
				return fmt.Errorf("certificate pin mismatch: %x", got)
			}
			return nil
		}
	}
	return cfg
}

func (c *Client) onPacket(p []byte) {
	c.mu.Lock()
	c.lastRecv = time.Now()
	c.mu.Unlock()

	select {
	case c.in <- append([]byte(nil), p...):
	default: // KCP is behind: drop it, KCP resends
	}
}

func (c *Client) LastRecv() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastRecv
}

// ---- net.PacketConn, for KCP ----

// sessionAddr: a client session id
type sessionAddr uint64

func (a sessionAddr) Network() string { return "obfsgen" }
func (a sessionAddr) String() string  { return fmt.Sprintf("%016x", uint64(a)) }

func (c *Client) RemoteAddr() net.Addr { return sessionAddr(0) }

func (c *Client) ReadFrom(b []byte) (int, net.Addr, error) {
	select {
	case p := <-c.in:
		return copy(b, p), sessionAddr(0), nil
	case <-c.ctx.Done():
		return 0, nil, net.ErrClosed
	}
}

func (c *Client) WriteTo(b []byte, _ net.Addr) (int, error) {
	if c.ctx.Err() != nil {
		return 0, net.ErrClosed
	}
	c.out.push(b)
	return len(b), nil
}

func (c *Client) Close() error                     { c.cancel(); return nil }
func (c *Client) LocalAddr() net.Addr              { return sessionAddr(c.sessionID) }
func (c *Client) SetDeadline(time.Time) error      { return nil }
func (c *Client) SetReadDeadline(time.Time) error  { return nil }
func (c *Client) SetWriteDeadline(time.Time) error { return nil }
