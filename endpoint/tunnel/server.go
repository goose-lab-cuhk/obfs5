package tunnel

import (
	"context"
	"crypto/tls"
	mathrand "math/rand/v2"
	"net"
	"sync"
	"time"

	"obfsgen/config"
	"obfsgen/engine"
	"obfsgen/scheduler"
)

type ServerConfig struct {
	Listen      string          // TCP address to accept tunnels on
	Certificate tls.Certificate // the TLS certificate
	PSK         []byte          // pre-shared key, checks the client's hello
	Models      *engine.Models  // the models
	Config      *config.Config  // the settings (config.yaml)
	Logf        func(format string, args ...any)

	Expired func(net.Addr) // a session had no tunnel for server.session_idle: close its KCP
}

type clientSession struct {
	out      *buffer   // packets for this client; all its tunnels pull from it
	joiner   *joiner   // joins cut segments from this client
	tunnels  int       // tunnels open now
	lastSeen time.Time // when a tunnel last opened or closed
}

type packetFrom struct {
	data []byte
	from sessionAddr
}

// Server: a UDP socket to KCP.
//
//	send:    KCP ==WriteTo(id)==> sessions[id].out ==pull==> tunnel ==records==> TLS ==> client id
//	receive: KCP <==ReadFrom== in (all clients) <==onPacket== tunnel <==records== TLS <== client id
//
// Per connection:
//
//	accept ==> TLS handshake ==> 1st record: check hello ==> tunnel.run: one schedule ==> close
//
// same session id ==> same buffer, same KCP session
type Server struct {
	cfg ServerConfig
	ln  net.Listener
	in  chan packetFrom // packets from all clients, read by KCP

	mu       sync.Mutex
	sessions map[uint64]*clientSession

	ctx    context.Context // cancelled by Close
	cancel context.CancelFunc
}

func Listen(cfg ServerConfig) (*Server, error) {
	if cfg.Logf == nil {
		cfg.Logf = func(string, ...any) {}
	}
	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return nil, err
	}
	s := &Server{
		cfg:      cfg,
		ln:       ln,
		in:       make(chan packetFrom, 4096),
		sessions: map[uint64]*clientSession{},
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	go s.acceptLoop()
	go s.expireLoop()
	return s, nil
}

func (s *Server) Addr() net.Addr { return s.ln.Addr() }

func (s *Server) tlsConfig() *tls.Config {
	return &tls.Config{
		Certificates:                []tls.Certificate{s.cfg.Certificate},
		MinVersion:                  tls.VersionTLS13,
		NextProtos:                  []string{"h2", "http/1.1"},
		SessionTicketsDisabled:      true, // no records outside the model after the handshake
		DynamicRecordSizingDisabled: true, // one Write = one record of our size
	}
}

func (s *Server) acceptLoop() {
	for {
		raw, err := s.ln.Accept()
		if err != nil {
			return // closed
		}
		go s.serveTunnel(raw)
	}
}

func (s *Server) serveTunnel(raw net.Conn) {
	defer raw.Close()
	raw.SetDeadline(time.Now().Add(s.cfg.Config.Tunnel.MaxLife.Duration()))

	// 1. TLS
	conn := tls.Server(raw, s.tlsConfig())
	ctx, cancel := context.WithTimeout(s.ctx, s.cfg.Config.Tunnel.HandshakeTimeout.Duration())
	err := conn.HandshakeContext(ctx)
	cancel()
	if err != nil {
		return
	}

	// 2. the opener must carry a valid hello
	done := make(chan struct{})
	defer close(done)
	records := readRecords(conn, done)
	var first received
	select {
	case first = <-records:
	case <-time.After(s.cfg.Config.Tunnel.HandshakeTimeout.Duration()):
		return
	case <-s.ctx.Done():
		return
	}
	if first.err != nil {
		return
	}
	rec, err := removePadding(first.plaintext)
	if err != nil {
		return
	}
	h, ok := decodeHello(rec.hello, s.cfg.PSK)
	if !ok {
		s.cfg.Logf("rejected %v: authentication failed", raw.RemoteAddr())
		return
	}
	sess := s.tunnelOpened(h.sessionID)
	defer s.tunnelClosed(h.sessionID)

	t := &tunnel{
		conn:    conn,
		out:     sess.out,
		deliver: func(p []byte) { s.onPacket(p, h.sessionID) },
		joiner:  sess.joiner,
		start:   first.at, // the schedule starts when the opener arrives
	}

	// 3. scheduler
	m := s.cfg.Models
	if rec.flags&flagSync != 0 {
		// sync: the client's plan, from the seed; the opener must be the seed's
		if rec.size != m.Opener.Draw(h.seed, minOpener) {
			s.cfg.Logf("rejected %v: the first record does not match its seed (another parameter folder?)", raw.RemoteAddr())
			return
		}
		t.sync = scheduler.NewSync(scheduler.NewPlan(m.Sync, m.SyncTok, rec.size, h.seed), false)
	} else {
		// async, with the client's RTT
		random := mathrand.New(mathrand.NewPCG(mathrand.Uint64(), mathrand.Uint64()))
		rtt := float64(h.rttMicros) / 1e6
		t.async = scheduler.NewAsync(m.Async, m.AsyncTok, scheduler.ServerRole, rtt, random, s.cfg.Config.Scheduler)
		t.async.Listen()
	}

	if err := t.receiveRecord(first); err != nil {
		return
	}

	// 4. play, then close
	err = t.run(s.ctx, records)
	if err != nil && s.ctx.Err() == nil {
		s.cfg.Logf("tunnel from %v: %v", raw.RemoteAddr(), err)
	}
}

func (s *Server) tunnelOpened(id uint64) *clientSession {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess := s.sessions[id]
	if sess == nil {
		sess = &clientSession{out: newBuffer(), joiner: newJoiner()}
		s.sessions[id] = sess
	}
	sess.tunnels++
	sess.lastSeen = time.Now()
	return sess
}

func (s *Server) tunnelClosed(id uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess := s.sessions[id]
	if sess != nil {
		sess.tunnels--
		sess.lastSeen = time.Now()
	}
}

func (s *Server) onPacket(p []byte, id uint64) {
	select {
	case s.in <- packetFrom{data: append([]byte(nil), p...), from: sessionAddr(id)}:
	default: // KCP is behind: drop it, the client resends
	}
}

func (s *Server) expireLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
		}

		var expired []uint64
		s.mu.Lock()
		for id, sess := range s.sessions {
			if sess.tunnels == 0 && time.Since(sess.lastSeen) > s.cfg.Config.Server.SessionIdle.Duration() {
				delete(s.sessions, id)
				expired = append(expired, id)
			}
		}
		s.mu.Unlock()

		// outside the lock: the callback closes KCP
		for _, id := range expired {
			if s.cfg.Expired != nil {
				s.cfg.Expired(sessionAddr(id))
			}
		}
	}
}

// ---- net.PacketConn, for KCP ----

func (s *Server) ReadFrom(b []byte) (int, net.Addr, error) {
	select {
	case p := <-s.in:
		return copy(b, p.data), p.from, nil
	case <-s.ctx.Done():
		return 0, nil, net.ErrClosed
	}
}

func (s *Server) WriteTo(b []byte, addr net.Addr) (int, error) {
	a, ok := addr.(sessionAddr)
	if !ok {
		return 0, net.InvalidAddrError("not an obfsgen session")
	}
	s.mu.Lock()
	sess := s.sessions[uint64(a)]
	s.mu.Unlock()
	if sess != nil {
		sess.out.push(b)
	}
	return len(b), nil
}

func (s *Server) Close() error {
	s.cancel()
	return s.ln.Close()
}

func (s *Server) LocalAddr() net.Addr              { return s.ln.Addr() }
func (s *Server) SetDeadline(time.Time) error      { return nil }
func (s *Server) SetReadDeadline(time.Time) error  { return nil }
func (s *Server) SetWriteDeadline(time.Time) error { return nil }
