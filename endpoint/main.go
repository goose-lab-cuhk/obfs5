// Command obfsgen runs the proxy: a SOCKS5 client and a server whose TLS records follow the model.
//
//	 client                                                        server
//	 browser                                                       target site
//	    ║ SOCKS5                                                      ║ TCP
//	 ===╬=== APPLICATION  socks ======================================╬===
//	    ║ streams                                                     ║
//	 ===╬=== SESSION      session: smux + KCP ========================╬===
//	    ║ packets                                                     ║
//	 ===╬=== TUNNEL       tunnel: buffer ==> records ==> padding =====╬===
//	    ║                 scheduler <== engine (C) <== parameters     ║
//	    ║ records, each of exactly the model's size                   ║
//	 ===╬=== TLS 1.3      one connection per schedule ================╬===
//	    ╚══════════════════════════ wire ═════════════════════════════╝
//
//	obfsgen gencert -name example.com -dir /etc/obfsgen     # self-signed cert, prints its pin
//	obfsgen server  -listen :443 -cert cert.pem -key key.pem -psk SECRET [-params DIR] [-config FILE]
//	obfsgen client  -server host:443 -sni example.com -pin HEX -psk SECRET -socks 127.0.0.1:1080 [-params DIR] [-config FILE]
//
// -params is the parameter folder written by model/export.py (both ends must
// use the same one). The settings (connections, timeouts, the scheduler's
// deadline, KCP) are in config.yaml, built into the binary; -config reads
// changes from a copy.
package main

import (
	"context"
	"crypto/tls"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"obfsgen/certs"
	"obfsgen/config"
	"obfsgen/engine"
	"obfsgen/session"
	"obfsgen/socks"
	"obfsgen/tunnel"
)

// The settings; -config FILE changes them.
//
//go:embed config.yaml
var builtinConfig []byte

const defaultParams = "params/example/campus-gateway"

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var err error
	switch os.Args[1] {
	case "server":
		err = server(ctx, os.Args[2:])
	case "client":
		err = client(ctx, os.Args[2:])
	case "gencert":
		err = gencert(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		log.Fatal(err)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: obfsgen server|client|gencert [flags]  (-h for the flags)")
	os.Exit(2)
}

func server(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("server", flag.ExitOnError)
	listen := fs.String("listen", ":443", "TCP address to accept tunnels on")
	certFile := fs.String("cert", "cert.pem", "certificate (PEM)")
	keyFile := fs.String("key", "key.pem", "private key (PEM)")
	psk := fs.String("psk", "", "pre-shared key")
	pskFile := fs.String("psk-file", "", "read the pre-shared key from this file")
	params := fs.String("params", defaultParams, "parameter folder (model/export.py)")
	configFile := fs.String("config", "", "settings to change (a copy of config.yaml)")
	quiet := fs.Bool("quiet", false, "log errors only")
	_ = fs.Parse(args)
	if err := readPSK(psk, *pskFile); err != nil {
		return err
	}
	if *psk == "" {
		return fmt.Errorf("server: -psk or -psk-file is required")
	}
	cfg, err := config.Load(builtinConfig, *configFile)
	if err != nil {
		return err
	}
	cert, err := tls.LoadX509KeyPair(*certFile, *keyFile)
	if err != nil {
		return err
	}
	m, err := engine.LoadModels(*params)
	if err != nil {
		return err
	}
	logf := log.Printf
	if *quiet {
		logf = func(string, ...any) {}
	}
	log.Printf("obfsgen server on %s, kernels %s, certificate pin %s", *listen, simd(), certs.Pin(cert))
	return socks.DialTargets(ctx, tunnel.ServerConfig{Listen: *listen, Certificate: cert, PSK: []byte(*psk), Models: m, Config: cfg, Logf: logf})
}

func client(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("client", flag.ExitOnError)
	srv := fs.String("server", "", "server host:port (required)")
	sni := fs.String("sni", "", "TLS server name (default: the server's host)")
	pin := fs.String("pin", "", "SHA-256 pin of a self-signed server certificate (hex)")
	psk := fs.String("psk", "", "pre-shared key")
	pskFile := fs.String("psk-file", "", "read the pre-shared key from this file")
	socksAddr := fs.String("socks", "127.0.0.1:1080", "local SOCKS5 address")
	params := fs.String("params", defaultParams, "parameter folder (model/export.py)")
	configFile := fs.String("config", "", "settings to change (a copy of config.yaml)")
	quiet := fs.Bool("quiet", false, "log errors only")
	_ = fs.Parse(args)
	if err := readPSK(psk, *pskFile); err != nil {
		return err
	}
	if *srv == "" || *psk == "" {
		return fmt.Errorf("client: -server and -psk (or -psk-file) are required")
	}
	cfg, err := config.Load(builtinConfig, *configFile)
	if err != nil {
		return err
	}
	if *sni == "" {
		host, _, err := net.SplitHostPort(*srv)
		if err != nil {
			return err
		}
		*sni = host
	}
	m, err := engine.LoadModels(*params)
	if err != nil {
		return err
	}
	logf := log.Printf
	if *quiet {
		logf = func(string, ...any) {}
	}
	c := session.NewClient(tunnel.ClientConfig{
		Server: *srv, ServerName: *sni, Pin: *pin, PSK: []byte(*psk),
		Conns: cfg.Client.Conns, Sync: cfg.Client.Mode == "sync", Models: m, Config: cfg, Logf: logf,
	})
	defer c.Close()
	ln, err := net.Listen("tcp", *socksAddr)
	if err != nil {
		return err
	}
	log.Printf("obfsgen client: SOCKS5 on %s -> %s (%s, %d connections, kernels %s)", ln.Addr(), *srv, cfg.Client.Mode, cfg.Client.Conns, simd())
	return socks.Serve(ctx, c, ln)
}

func gencert(args []string) error {
	fs := flag.NewFlagSet("gencert", flag.ExitOnError)
	name := fs.String("name", "example.com", "certificate name")
	dir := fs.String("dir", ".", "where to write cert.pem and key.pem")
	_ = fs.Parse(args)
	c, k, pin, err := certs.SelfSigned(*name)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*dir, "cert.pem"), c, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*dir, "key.pem"), k, 0o600); err != nil {
		return err
	}
	fmt.Println(pin)
	return nil
}

func readPSK(psk *string, file string) error {
	if file == "" {
		return nil
	}
	b, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	*psk = strings.TrimSpace(string(b))
	return nil
}

func simd() string {
	return [...]string{"plain C", "AVX2", "AVX-512", "NEON"}[engine.SIMD()]
}
