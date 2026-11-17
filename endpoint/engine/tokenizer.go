// The tokenizer: converts records to tokens and back with the tokenizer tables
// (params/<name>/{async,sync}/tokenizer.json).

package engine

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// Tables is one tokenizer. The JSON fields read here (the file has more):
//
//	B_s                 number of size bins ([EOF] is size bin B_s)
//	B_t                 number of time bins
//	S_norm              16384: size value = direction × bytes / S_norm
//	epsilon             1e-8: time value = ln(gap seconds + epsilon)
//	max_records         500: a flow stops at this many records
//	size_edges          B_s+1 bin edges on the size value
//	time_edges          B_t+1 bin edges on the time value
//	size_bin_direction  per size bin: +1 or −1 (async: sent / received by the
//	                    endpoint; sync: client→server / server→client)
//	size_intra          per size bin: {values: observed size values, probs: their weights}
//	time_intra          per time bin: observed time values, one drawn uniformly
type Tables struct {
	Bs         int       `json:"B_s"`
	Bt         int       `json:"B_t"`
	SNorm      float64   `json:"S_norm"`
	Epsilon    float64   `json:"epsilon"`
	MaxRecords int       `json:"max_records"`
	SizeEdges  []float64 `json:"size_edges"`
	TimeEdges  []float64 `json:"time_edges"`
	SizeDir    []int     `json:"size_bin_direction"`
	SizeIntra  []struct {
		Values []float64 `json:"values"`
		Probs  []float64 `json:"probs"`
	} `json:"size_intra"`
	TimeIntra [][]float64 `json:"time_intra"`

	cum      [][]float64 // per size bin: cumsum(probs) / sum(probs)
	SendBins []uint8     // Bs: 1 for a send (or client→server) bin
	RecvBins []uint8     // Bs: 1 for a receive (or server→client) bin
}

// parseTables loads a tokenizer JSON file.
func parseTables(b []byte) (*Tables, error) {
	t := &Tables{}
	if err := json.Unmarshal(b, t); err != nil {
		return nil, err
	}
	if len(t.SizeEdges) != t.Bs+1 || len(t.TimeEdges) != t.Bt+1 || len(t.SizeDir) != t.Bs ||
		len(t.SizeIntra) != t.Bs || len(t.TimeIntra) != t.Bt {
		return nil, fmt.Errorf("engine: inconsistent tokenizer tables")
	}
	t.cum = make([][]float64, t.Bs)
	for i, e := range t.SizeIntra {
		var s float64
		c := make([]float64, len(e.Probs))
		for k, p := range e.Probs {
			s += p
			c[k] = s
		}
		for k := range c {
			c[k] /= s
		}
		t.cum[i] = c
	}
	t.SendBins = make([]uint8, t.Bs)
	t.RecvBins = make([]uint8, t.Bs)
	for i, d := range t.SizeDir {
		if d > 0 {
			t.SendBins[i] = 1
		} else {
			t.RecvBins[i] = 1
		}
	}
	return t, nil
}

// assign: the number of interior edges ≤ v.
func assign(edges []float64, v float64) int {
	in := edges[1 : len(edges)-1]
	return sort.Search(len(in), func(i int) bool { return in[i] > v })
}

// SizeBin of a record of size bytes; dir +1 sent (client→server), −1 received.
// Values are rounded to float32, as in training.
func (t *Tables) SizeBin(size float64, dir int) int {
	return assign(t.SizeEdges, float64(float32(float64(dir)*size/t.SNorm)))
}

// TimeBin of a gap in seconds.
func (t *Tables) TimeBin(gap float64) int {
	return assign(t.TimeEdges, float64(float32(math.Log(math.Max(gap, 0)+t.Epsilon))))
}

// Encode a record to its token.
func (t *Tables) Encode(size, gap float64, dir int) (int, int) {
	return t.SizeBin(size, dir), t.TimeBin(gap)
}

// SizeOf draws a size in bytes inside a size bin from a uniform u ∈ [0,1).
// The sign of the stored value gives the direction.
func (t *Tables) SizeOf(bin int, u float64) (bytes int, dir int) {
	c := t.cum[bin]
	k := sort.Search(len(c), func(i int) bool { return c[i] > u })
	if k == len(c) {
		k = len(c) - 1
	}
	v := t.SizeIntra[bin].Values[k]
	dir = 1
	if v < 0 {
		dir = -1
	}
	return int(math.Round(math.Abs(v) * t.SNorm)), dir
}

// GapOf returns sample index i of a time bin as seconds.
func (t *Tables) GapOf(bin, i int) float64 {
	return math.Max(math.Exp(t.TimeIntra[bin][i])-t.Epsilon, 0)
}

// DrawSize and DrawGap use r for the intra-bin draw.
func (t *Tables) DrawSize(bin int, r *rand.Rand) (int, int) { return t.SizeOf(bin, r.Float64()) }

func (t *Tables) DrawGap(bin int, r *rand.Rand) float64 {
	return t.GapOf(bin, r.IntN(len(t.TimeIntra[bin])))
}

// WaitedMask: the time bins at or above the bin holding waited.
func (t *Tables) WaitedMask(waited float64) []uint8 {
	m := make([]uint8, t.Bt)
	for i := t.TimeBin(waited); i < t.Bt; i++ {
		m[i] = 1
	}
	return m
}

// GapFromQuantile inverts quantile u of a time-bin pmf to seconds.
func (t *Tables) GapFromQuantile(pmf []float64, u float64, r *rand.Rand) (float64, int) {
	var c float64
	k := len(pmf) - 1
	for i, p := range pmf {
		c += p
		if c >= u {
			k = i
			break
		}
	}
	return t.DrawGap(k, r), k
}
