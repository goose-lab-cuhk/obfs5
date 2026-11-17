// Package tunnel is the tunnel layer: one TLS connection plays one
// schedule and closes at [EOF]; records carry the session layer's packets
// plus padding.
//
//	KCP ==packets==> buffer ==(record due)==> addPadding: record of the model's size ==> TLS
//	KCP <==packets== unpack <================ removePadding <============================ TLS
package tunnel

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"errors"
)

// The padding unit. A record's size is the model's size, counted as the TLS
// record body (the 5-byte header not included):
//
//	+-----------+-------------------------------------------+---------+--------------+
//	| header 5B | plaintext: size − 17                      | type 1B | AEAD tag 16B |
//	+-----------+-------------------------------------------+---------+--------------+
//	             |
//	             v
//	+-------+-------------------+-----------------------+------------------+
//	| flags | hello             | data                  | padding          |
//	| 1B    | 36B, 1st rec only | 2B length n | n bytes | zeros to the end |
//	+-------+-------------------+-----------------------+------------------+
//	                                     |
//	                                     v
//	                       +-----+--------------+-----+--------------+---
//	                       | 2B  | KCP segments | 2B  | KCP segments | ...
//	                       +-----+--------------+-----+--------------+---
//	                       an entry may also be a piece of a cut segment (buffer.go)
//
//	flags: HELLO (the record has a hello), SYNC (synchronous mode)
//	hello: session id 8B | seed 8B | RTT µs 4B | HMAC-SHA256(psk, first 20B)[:16] 16B
const (
	flagHello = 1
	flagSync  = 2

	tlsOverhead = 17    // TLS adds 1 type byte + 16 bytes AEAD tag
	maxPlain    = 16384 // the largest TLS plaintext
	helloLen    = 36
)

// minOpener: the smallest first record, one that holds the hello.
var minOpener = recordSize(1 + helloLen + 2)

// plaintextSize: how many bytes we write so that TLS sends a record of size.
func plaintextSize(size int) int {
	n := size - tlsOverhead
	if n < 1 {
		n = 1
	}
	if n > maxPlain {
		n = maxPlain
	}
	return n
}

// recordSize: the size of the record TLS sends for n plaintext bytes.
func recordSize(n int) int {
	return n + tlsOverhead
}

// hello is sent in the client's first record of every connection.
type hello struct {
	sessionID uint64 // connections with the same id carry one KCP session
	seed      uint64 // sync mode: both ends sample the plan from this seed
	rttMicros uint32 // the client's RTT, for the server's async scheduler
}

func (h hello) encode(psk []byte) []byte {
	b := make([]byte, helloLen)
	binary.LittleEndian.PutUint64(b[0:8], h.sessionID)
	binary.LittleEndian.PutUint64(b[8:16], h.seed)
	binary.LittleEndian.PutUint32(b[16:20], h.rttMicros)
	copy(b[20:36], mac(psk, b[0:20]))
	return b
}

// decodeHello checks the MAC and reads the fields.
func decodeHello(b []byte, psk []byte) (hello, bool) {
	if len(b) != helloLen {
		return hello{}, false
	}
	if !hmac.Equal(b[20:36], mac(psk, b[0:20])) {
		return hello{}, false // wrong pre-shared key
	}
	h := hello{
		sessionID: binary.LittleEndian.Uint64(b[0:8]),
		seed:      binary.LittleEndian.Uint64(b[8:16]),
		rttMicros: binary.LittleEndian.Uint32(b[16:20]),
	}
	return h, true
}

func mac(psk, msg []byte) []byte {
	h := hmac.New(sha256.New, psk)
	h.Write(msg)
	sum := h.Sum(nil)
	return sum[:16]
}

// record is a received record after removePadding.
type record struct {
	size  int    // size on the wire
	flags byte   // flagHello, flagSync
	hello []byte // nil if the record has no hello
	data  []byte // entries: whole KCP segments or pieces (buffer.go)
}

var errRecord = errors.New("tunnel: malformed record")

// removePadding reads a received record (its TLS plaintext p).
func removePadding(p []byte) (record, error) {
	r := record{size: recordSize(len(p))}
	if len(p) == 0 {
		return r, errRecord
	}

	// flags
	r.flags = p[0]
	p = p[1:]

	// hello
	if r.flags&flagHello != 0 {
		if len(p) < helloLen {
			return r, errRecord
		}
		r.hello = p[:helloLen]
		p = p[helloLen:]
	}

	// data (the rest is padding)
	if len(p) >= 2 {
		n := int(binary.BigEndian.Uint16(p))
		if len(p) < 2+n {
			return r, errRecord
		}
		r.data = p[2 : 2+n]
	}
	return r, nil
}

// addPadding builds a record of exactly size bytes on the wire: flags,
// hello (if not nil), as much data from buf as fits, then zeros.
func addPadding(size int, flags byte, helloBytes []byte, buf *buffer) []byte {
	p := make([]byte, plaintextSize(size)) // all zeros: the padding

	// flags
	if helloBytes != nil {
		flags = flags | flagHello
	}
	p[0] = flags
	off := 1

	// hello
	if helloBytes != nil {
		copy(p[off:], helloBytes)
		off += helloLen
	}

	// data: [2B length n][n bytes]
	if len(p)-off >= 2 {
		n := buf.pull(p[off+2:])
		binary.BigEndian.PutUint16(p[off:], uint16(n))
	}
	return p
}
