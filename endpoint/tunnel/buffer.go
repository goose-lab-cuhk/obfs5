package tunnel

import (
	"encoding/binary"
	"sync"
	"time"
)

// buffer holds the session's outgoing KCP packets until a record is due (send
// side only). When full, it drops; KCP resends.
//
//	KCP ==push==> [ pkt | pkt | pkt | ... ] ==pull(room)==> record (when one is due)
//
// pull fills the room oldest first. Whole KCP segments go in as one entry;
// a segment that does not fit is cut, and its rest stays at the head of the
// queue for the next record of ANY tunnel (the receiver joins the pieces):
//
//	record room:  |<------------------ room ------------------>|
//	buffer:       [ seg seg ][ seg ][ seg ...
//	taken:        [ seg seg ][ se|           "g" waits at the head
//
// Data waits here for the schedule, often longer than KCP's RTO, so KCP
// resends segments whose first copy has not left yet. push drops such
// copies; without this the buffer fills with copies and stalls:
//
//	queued: [ sn 7 | sn 8 ]   KCP resends sn 7 ==push==> dropped (still queued)
type buffer struct {
	mu      sync.Mutex
	packets [][]byte        // KCP packets, oldest first
	queued  map[uint32]bool // sn of the data segments in packets (and in cut)
	wakeup  chan struct{}   // closed at the next push

	cut    []byte // the segment being cut, nil if none
	cutOff int    // bytes of cut already sent
	cutID  uint32 // its piece id
	lastID uint32
}

const bufferMaxPackets = 512

// Record data entries:
//
//	whole:  [2B n]           [n bytes: KCP segments]
//	piece:  [2B n | 0x8000]  [4B id][2B offset][2B total] [n bytes of a segment]
const (
	pieceBit    = 0x8000
	pieceHeader = 8
	pieceMin    = 16 // a smaller room is left as padding
)

func newBuffer() *buffer {
	return &buffer{queued: map[uint32]bool{}, wakeup: make(chan struct{})}
}

// push: from KCP's WriteTo. Keeps the segments not already queued.
func (b *buffer) push(packet []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.packets) >= bufferMaxPackets {
		return // full: drop it
	}

	var keep []byte
	for _, seg := range kcpSegments(packet) {
		if isData(seg) {
			if b.queued[sn(seg)] {
				continue // a copy is still queued
			}
			b.queued[sn(seg)] = true
		}
		keep = append(keep, seg...)
	}
	if len(keep) == 0 {
		return
	}
	b.packets = append(b.packets, keep)

	// wake up waiting workers
	close(b.wakeup)
	b.wakeup = make(chan struct{})
}

// pull fills room with entries, oldest first, and returns the bytes written.
func (b *buffer) pull(room []byte) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	used := 0
	for {
		// 1. the rest of a cut segment: it is the oldest data
		if b.cut != nil {
			n := min(len(room)-used-2-pieceHeader, len(b.cut)-b.cutOff)
			if n <= 0 || n < min(pieceMin, len(b.cut)-b.cutOff) {
				break
			}
			binary.BigEndian.PutUint16(room[used:], uint16(n)|pieceBit)
			binary.BigEndian.PutUint32(room[used+2:], b.cutID)
			binary.BigEndian.PutUint16(room[used+6:], uint16(b.cutOff))
			binary.BigEndian.PutUint16(room[used+8:], uint16(len(b.cut)))
			copy(room[used+2+pieceHeader:], b.cut[b.cutOff:b.cutOff+n])
			used += 2 + pieceHeader + n
			b.cutOff += n
			if b.cutOff < len(b.cut) {
				break // the record is full
			}
			b.sent(b.cut)
			b.cut = nil
			continue
		}
		if len(b.packets) == 0 {
			break
		}

		// 2. whole segments of the oldest packet that fit
		var take []byte
		for _, seg := range kcpSegments(b.packets[0]) {
			if len(take)+len(seg) > len(room)-used-2 {
				break
			}
			take = b.packets[0][:len(take)+len(seg)]
		}
		if len(take) == 0 {
			// 3. not even one fits: start cutting the first segment
			if len(room)-used-2-pieceHeader < pieceMin {
				break
			}
			seg := kcpSegments(b.packets[0])[0]
			b.cut, b.cutOff = append([]byte(nil), seg...), 0
			b.lastID++
			b.cutID = b.lastID
			b.drop(len(seg))
			continue
		}
		binary.BigEndian.PutUint16(room[used:], uint16(len(take)))
		copy(room[used+2:], take)
		used += 2 + len(take)
		b.sent(take)
		b.drop(len(take))
	}
	return used
}

// drop removes n bytes from the head of the oldest packet.
func (b *buffer) drop(n int) {
	b.packets[0] = b.packets[0][n:]
	if len(b.packets[0]) == 0 {
		b.packets = b.packets[1:]
	}
}

// sent: these segments have left; a later copy is a real resend.
func (b *buffer) sent(segs []byte) {
	for _, seg := range kcpSegments(segs) {
		if isData(seg) {
			delete(b.queued, sn(seg))
		}
	}
}

// waitForPush returns a channel that is closed at the next push.
func (b *buffer) waitForPush() <-chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.wakeup
}

// isEmpty reports whether no packet is waiting.
func (b *buffer) isEmpty() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.packets) == 0 && b.cut == nil
}

// ---- receive side ----

// joiner joins the pieces of cut segments. One per session: the pieces of
// one segment may come in records of different tunnels.
type joiner struct {
	mu     sync.Mutex
	pieces map[uint32]*joining
}

type joining struct {
	buf   []byte
	got   int
	start time.Time
}

func newJoiner() *joiner { return &joiner{pieces: map[uint32]*joining{}} }

// add returns the whole segment once its last piece is in, else nil.
func (j *joiner) add(id uint32, off, total int, piece []byte) []byte {
	j.mu.Lock()
	defer j.mu.Unlock()
	p := j.pieces[id]
	if p == nil {
		if len(j.pieces) > 1024 { // pieces lost with a tunnel: KCP resends
			for k, q := range j.pieces {
				if time.Since(q.start) > 30*time.Second {
					delete(j.pieces, k)
				}
			}
		}
		p = &joining{buf: make([]byte, total), start: time.Now()}
		j.pieces[id] = p
	}
	if total != len(p.buf) || off+len(piece) > total {
		delete(j.pieces, id)
		return nil
	}
	copy(p.buf[off:], piece)
	p.got += len(piece)
	if p.got < total {
		return nil
	}
	delete(j.pieces, id)
	return p.buf
}

// unpack splits a record's data back into KCP packets: the reverse of pull.
func unpack(data []byte, j *joiner, deliver func(packet []byte)) error {
	for len(data) > 0 {
		if len(data) < 2 {
			return errRecord
		}
		n := int(binary.BigEndian.Uint16(data))
		data = data[2:]
		if n&pieceBit == 0 {
			if len(data) < n {
				return errRecord
			}
			deliver(data[:n])
			data = data[n:]
			continue
		}
		n &^= pieceBit
		if len(data) < pieceHeader+n {
			return errRecord
		}
		id := binary.BigEndian.Uint32(data)
		off := int(binary.BigEndian.Uint16(data[4:]))
		total := int(binary.BigEndian.Uint16(data[6:]))
		if seg := j.add(id, off, total, data[pieceHeader:pieceHeader+n]); seg != nil {
			deliver(seg)
		}
		data = data[pieceHeader+n:]
	}
	return nil
}

// ---- KCP segments (see the session package for the header) ----
//
//	packet: [conv 4|cmd 1|frg 1|wnd 2|ts 4|sn 4|una 4|len 4|data len] [next segment]...

const (
	kcpHeader  = 24
	kcpCmdData = 81 // IKCP_CMD_PUSH
)

// kcpSegments splits a KCP packet into its segments. Bytes that are not a
// whole segment are returned as one last piece, unchanged.
func kcpSegments(p []byte) [][]byte {
	var segs [][]byte
	for len(p) > 0 {
		n := len(p)
		if len(p) >= kcpHeader {
			n = kcpHeader + int(binary.LittleEndian.Uint32(p[20:24]))
		}
		if n > len(p) {
			n = len(p)
		}
		segs = append(segs, p[:n])
		p = p[n:]
	}
	return segs
}

func isData(seg []byte) bool { return len(seg) >= kcpHeader && seg[4] == kcpCmdData }
func sn(seg []byte) uint32   { return binary.LittleEndian.Uint32(seg[12:16]) }
