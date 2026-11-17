// Package session is the session layer:
//
//	socks <==streams==> smux <==bytes==> KCP <==packets==> tunnel (a UDP socket to KCP)
//	                                   seq + ACK + resend
//
//	send:    KCP ==WriteTo==>  tunnel buffer ==(record due)==> TLS
//	receive: KCP <==ReadFrom== tunnel        <================== TLS
//
// KCP packet: [conv 4|cmd 1|frg 1|wnd 2|ts 4|sn 4|una 4|len 4|data] [next segment]...
// Lost packets (buffer full, tunnel closed mid-packet): KCP resends.
package session

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"net"
	"sync"
	"time"

	"github.com/xtaci/kcp-go/v5"
	"github.com/xtaci/smux"

	"obfsgen/config"
	"obfsgen/tunnel"
)

func tuneKCP(s *kcp.UDPSession, k config.KCP) {
	s.SetNoDelay(1, 20, 2, 1)           // min RTO 30 ms, tick 20 ms, fast resend 2, no congestion control (the model paces)
	s.SetWindowSize(k.Window, k.Window) // ACKs come back only in the peer's scheduled records: a small window stalls the sender
	s.SetMtu(k.MTU)
	s.SetStreamMode(true)
	s.SetWriteDelay(false)
	s.SetACKNoDelay(true)
}

// smux keepalives are off: they would make the client poll the server
// forever. A dead server is caught by the client's watchdog instead.
func smuxConfig() *smux.Config {
	c := smux.DefaultConfig()
	c.Version = 2
	c.KeepAliveDisabled = true
	c.MaxReceiveBuffer = 4 << 20
	c.MaxStreamBuffer = 128 << 10 // data one stream may queue ahead of the others: a bulk download must not hold up a page
	return c
}

// ---------------------------------------------------------------- client --

// Client opens streams to the server over one KCP session, rebuilt when the
// server stops answering.
type Client struct {
	cfg tunnel.ClientConfig
	mu  sync.Mutex
	ob  *tunnel.Client
	sm  *smux.Session
}

func NewClient(cfg tunnel.ClientConfig) *Client { return &Client{cfg: cfg} }

// session returns the current smux session, making a new one when needed.
func (c *Client) session() (*smux.Session, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sm != nil && !c.sm.IsClosed() && !c.stale() {
		return c.sm, nil
	}
	if c.ob != nil {
		c.cfg.Logf("session: rebuilding")
		c.sm.Close()
		c.ob.Close()
	}
	// the tunnels ask whether streams are open (then it keeps tunnels open so
	// the server can send).
	cfg := c.cfg
	cfg.Active = c.streamsOpen
	ob := tunnel.NewClient(cfg)
	var b [4]byte
	rand.Read(b[:])
	ks, err := kcp.NewConn3(binary.LittleEndian.Uint32(b[:]), ob.RemoteAddr(), nil, 0, 0, ob)
	if err != nil {
		ob.Close()
		return nil, err
	}
	tuneKCP(ks, c.cfg.Config.KCP)
	sm, err := smux.Client(ks, smuxConfig())
	if err != nil {
		ks.Close()
		ob.Close()
		return nil, err
	}
	c.ob, c.sm = ob, sm
	return sm, nil
}

// stale: streams are open, and the server has said nothing for too long.
func (c *Client) stale() bool {
	if c.sm.NumStreams() == 0 {
		return false
	}
	last := c.ob.LastRecv()
	return !last.IsZero() && time.Since(last) > c.cfg.Config.Client.Watchdog.Duration()
}

// streamsOpen reports whether the current session has open streams.
func (c *Client) streamsOpen() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sm != nil && c.sm.NumStreams() > 0
}

// Open opens a new stream to the server.
func (c *Client) Open() (net.Conn, error) {
	sm, err := c.session()
	if err != nil {
		return nil, err
	}
	return sm.OpenStream()
}

func (c *Client) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sm != nil {
		c.sm.Close()
		c.ob.Close()
	}
}

// ---------------------------------------------------------------- server --

// Serve runs the server's session layer until ctx ends, calling handle in a
// new goroutine for every stream a client opens.
func Serve(ctx context.Context, cfg tunnel.ServerConfig, handle func(net.Conn)) error {
	var mu sync.Mutex
	conns := map[string]*kcp.UDPSession{}
	userExpired := cfg.Expired
	cfg.Expired = func(a net.Addr) {
		mu.Lock()
		ks := conns[a.String()]
		delete(conns, a.String())
		mu.Unlock()
		if ks != nil {
			ks.Close()
		}
		if userExpired != nil {
			userExpired(a)
		}
	}
	srv, err := tunnel.Listen(cfg)
	if err != nil {
		return err
	}
	defer srv.Close()
	lis, err := kcp.ServeConn(nil, 0, 0, srv)
	if err != nil {
		return err
	}
	go func() { <-ctx.Done(); lis.Close(); srv.Close() }()
	for {
		ks, err := lis.AcceptKCP()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		tuneKCP(ks, cfg.Config.KCP)
		mu.Lock()
		if old := conns[ks.RemoteAddr().String()]; old != nil {
			old.Close()
		}
		conns[ks.RemoteAddr().String()] = ks
		mu.Unlock()
		sm, err := smux.Server(ks, smuxConfig())
		if err != nil {
			ks.Close()
			continue
		}
		go func() {
			for {
				st, err := sm.AcceptStream()
				if err != nil {
					return
				}
				go handle(st)
			}
		}()
	}
}
