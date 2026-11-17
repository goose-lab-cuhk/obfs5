// Package socks is the application layer:
//
//	client: browser ==SOCKS5==> stream: [len]["host:port"][data...] ==> session
//	server: session ==> stream: read "host:port", dial it ==TCP==> target
package socks

import (
	"context"
	"errors"
	"io"
	"net"
	"sync"
	"time"

	"obfsgen/session"
	"obfsgen/tunnel"
)

// A stream starts with its target: one length byte, then "host:port".
func writeTarget(w io.Writer, target string) error {
	if len(target) == 0 || len(target) > 255 {
		return errors.New("socks: bad target")
	}
	_, err := w.Write(append([]byte{byte(len(target))}, target...))
	return err
}

func readTarget(r io.Reader) (string, error) {
	var n [1]byte
	if _, err := io.ReadFull(r, n[:]); err != nil {
		return "", err
	}
	b := make([]byte, n[0])
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

// relay copies both ways and closes both ends when either side finishes.
func relay(a, b io.ReadWriteCloser) {
	var once sync.Once
	done := func() { once.Do(func() { a.Close(); b.Close() }) }
	go func() { io.Copy(a, b); done() }()
	io.Copy(b, a)
	done()
}

// ---------------------------------------------------------------- client --

// dial opens a stream to target through the server.
func dial(c *session.Client, target string) (net.Conn, error) {
	st, err := c.Open()
	if err != nil {
		return nil, err
	}
	if err := writeTarget(st, target); err != nil {
		st.Close()
		return nil, err
	}
	return st, nil
}

// Serve accepts SOCKS5 clients on ln until ctx ends (the client).
func Serve(ctx context.Context, c *session.Client, ln net.Listener) error {
	go func() { <-ctx.Done(); ln.Close() }()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go func() {
			target, err := socksHandshake(conn)
			if err != nil {
				conn.Close()
				return
			}
			st, err := dial(c, target)
			if err != nil {
				conn.Close()
				return
			}
			relay(conn, st)
		}()
	}
}

// ---------------------------------------------------------------- server --

// DialTargets runs the server until ctx ends: each stream is dialled to its
// target.
func DialTargets(ctx context.Context, cfg tunnel.ServerConfig) error {
	return session.Serve(ctx, cfg, func(st net.Conn) { serveStream(st, cfg.Logf) })
}

func serveStream(st net.Conn, logf func(string, ...any)) {
	st.SetReadDeadline(time.Now().Add(30 * time.Second))
	target, err := readTarget(st)
	if err != nil {
		st.Close()
		return
	}
	st.SetReadDeadline(time.Time{})
	out, err := net.DialTimeout("tcp", target, 10*time.Second)
	if err != nil {
		if logf != nil {
			logf("dial %s: %v", target, err)
		}
		st.Close()
		return
	}
	relay(st, out)
}
