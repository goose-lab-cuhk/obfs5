package engine

import (
	"encoding/json"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
)

// Models are the parameters of a parameter folder (both ends load all of them).
type Models struct {
	Async    *Model
	AsyncTok *Tables
	Sync     *Model
	SyncTok  *Tables
	Opener   *Opener
}

// Opener is the distribution of the size of the first record of the training
// flows that the client opens (opener.json): sizes and their counts.
type Opener struct {
	Sizes  []int    `json:"sizes"`
	Counts []uint64 `json:"counts"`
}

// Draw returns the size of a connection's first record, at least minSize
// bytes, drawn from the distribution with a generator seeded by seed. Both
// ends get the same size from the same seed (integer arithmetic only).
func (o *Opener) Draw(seed uint64, minSize int) int {
	var total uint64
	for i, s := range o.Sizes {
		if s >= minSize {
			total += o.Counts[i]
		}
	}
	if total == 0 {
		return minSize // no first record that large: the smallest that holds the hello
	}
	k := rand.New(rand.NewPCG(seed, 1)).Uint64N(total) // stream 0 is the sync plan's
	for i, s := range o.Sizes {
		if s < minSize {
			continue
		}
		if k < o.Counts[i] {
			return s
		}
		k -= o.Counts[i]
	}
	panic("unreachable")
}

// LoadModels reads a parameter folder written by model/export.py, such as
// params/example/campus-gateway: {async,sync}/{weights.obfw,tokenizer.json}
// and opener.json.
func LoadModels(dir string) (*Models, error) {
	var m Models
	var err error
	if m.Async, m.AsyncTok, err = loadForm(dir, "async"); err != nil {
		return nil, err
	}
	if m.Sync, m.SyncTok, err = loadForm(dir, "sync"); err != nil {
		return nil, err
	}
	b, err := os.ReadFile(filepath.Join(dir, "opener.json"))
	if err != nil {
		return nil, err
	}
	m.Opener = &Opener{}
	if err = json.Unmarshal(b, m.Opener); err != nil {
		return nil, err
	}
	if len(m.Opener.Sizes) == 0 || len(m.Opener.Sizes) != len(m.Opener.Counts) {
		return nil, errors.New("engine: opener.json holds no distribution")
	}
	return &m, nil
}

func loadForm(dir, form string) (*Model, *Tables, error) {
	w, err := os.ReadFile(filepath.Join(dir, form, "weights.obfw"))
	if err != nil {
		return nil, nil, err
	}
	t, err := os.ReadFile(filepath.Join(dir, form, "tokenizer.json"))
	if err != nil {
		return nil, nil, err
	}
	m, err := loadModel(w)
	if err != nil {
		return nil, nil, err
	}
	tables, err := parseTables(t)
	return m, tables, err
}
