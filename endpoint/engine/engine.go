// Package engine runs the model: the Go binding of the C engine (obfs.h and
// the .c files, compiled with the package), the tokenizer tables that turn
// tokens into bytes and seconds (tokenizer.go), and the loading of a
// parameter folder (params.go).
//
// Every C function is reentrant: a Model is read-only and may be shared by
// any number of goroutines; a Stream belongs to one goroutine at a time.
package engine

/*
#cgo CFLAGS: -O3 -std=gnu11
#cgo LDFLAGS: -lm
#include "obfs.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Model is a loaded weight file.
type Model struct {
	m       *C.obfs_model
	DModel  int
	Bs, Bt  int // regular size / time bins
	SizeOut int // Bs + 1 ([EOF] is the last slot)
	MaxSeq  int // positions available: pos must stay below this
	Roles   int
	convLen int
	ssmLen  int
}

// loadModel parses an OBFSW001 weight file held in memory.
func loadModel(b []byte) (*Model, error) {
	if len(b) == 0 {
		return nil, errors.New("engine: empty weight file")
	}
	var errbuf [256]C.char
	m := C.obfs_load_mem(unsafe.Pointer(&b[0]), C.size_t(len(b)), &errbuf[0], C.int(len(errbuf)))
	if m == nil {
		return nil, errors.New("engine: " + C.GoString(&errbuf[0]))
	}
	md := &Model{
		m:       m,
		DModel:  int(m.d_model),
		Bs:      int(m.B_s),
		Bt:      int(m.B_t),
		SizeOut: int(m.size_out),
		MaxSeq:  int(m.max_seq_len),
		Roles:   int(m.n_roles),
		convLen: int(C.obfs_conv_state_floats(m)),
		ssmLen:  int(C.obfs_ssm_state_floats(m)),
	}
	return md, nil
}

// SIMD reports the kernels compiled in: 0 plain C, 1 AVX2, 2 AVX-512, 3 NEON.
func SIMD() int { return int(C.obfs_simd()) }

// Stream is one token stream's recurrent state plus its latest outputs.
type Stream struct {
	m      *Model
	role   int64 // -1: no role (sync model)
	conv   []float32
	ssm    []float32
	H      []float32 // hidden vector after the last token
	Logits []float32 // size logits for the next token
	Pos    int       // tokens fed so far
}

// NewStream returns a zeroed stream. role is 0 client / 1 server, or -1 for
// the sync model.
func (m *Model) NewStream(role int) *Stream {
	return &Stream{
		m:      m,
		role:   int64(role),
		conv:   make([]float32, m.convLen),
		ssm:    make([]float32, m.ssmLen),
		H:      make([]float32, m.DModel),
		Logits: make([]float32, m.SizeOut),
	}
}

// Step feeds one token. It panics past the model's last position: the C code
// does not bound-check pos.
func (s *Stream) Step(sizeBin, timeBin int) {
	if s.Pos >= s.m.MaxSeq {
		panic("engine: stream past max_seq_len")
	}
	tok := [2]int64{int64(sizeBin), int64(timeBin)}
	pos := int64(s.Pos)
	var roles *C.int64_t
	if s.role >= 0 {
		roles = (*C.int64_t)(unsafe.Pointer(&s.role))
	}
	C.obfs_step(s.m.m, 1, (*C.int64_t)(unsafe.Pointer(&tok[0])), (*C.int64_t)(unsafe.Pointer(&pos)),
		roles, f32p(s.conv), f32p(s.ssm), f32p(s.H), f32p(s.Logits))
	s.Pos++
}

// DecideSize draws the next size bin; Bs means [EOF]. q holds SizeOut Exp(1)
// draws. sizeMask (SizeOut) restricts the size bins; without it, timeMask
// (Bt) conditions the size on a time bin in the mask. Nil means no mask.
func (s *Stream) DecideSize(sizeMask, timeMask []uint8, q []float32) int {
	return int(C.obfs_decide_size(s.m.m, f32p(s.H), f32p(s.Logits), u8p(sizeMask), u8p(timeMask), f32p(q)))
}

// DecideTime draws the time bin for a regular size bin; q holds Bt Exp(1)
// draws, timeMask (Bt) may be nil.
func (s *Stream) DecideTime(sizeBin int, timeMask []uint8, q []float32) int {
	return int(C.obfs_decide_time(s.m.m, f32p(s.H), C.int(sizeBin), u8p(timeMask), f32p(q)))
}

// ReceiveGapPMF fills pmf (Bt) with the distribution of the gap before the
// next received record and returns the receive bins' total probability.
func (s *Stream) ReceiveGapPMF(minTimeBin int, recv []uint8, pmf []float64) float32 {
	return float32(C.obfs_receive_gap_pmf(s.m.m, f32p(s.H), f32p(s.Logits), C.int(minTimeBin),
		u8p(recv), (*C.double)(unsafe.Pointer(&pmf[0]))))
}

func f32p(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func u8p(x []uint8) *C.uint8_t {
	if len(x) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&x[0]))
}
