package scheduler

import (
	"math"
	"math/rand/v2"

	"obfsgen/engine"
)

// Synchronous mode. Both ends sample the same plan:
//
//	client                                   server
//	  opener size, seed  ===== 1st record ====>  opener size (its wire size), seed
//	  plan = NewPlan(opener, seed)               plan = NewPlan(opener, seed)
//	  walk the plan  <======= records =======>   walk the plan
//
// Walking: my record leaves its Gap after my last event; the peer's record
// is awaited without a deadline.

// PlannedRecord is one record of a plan.
type PlannedRecord struct {
	FromClient bool    // sent by the client
	Size       int     // bytes
	Gap        float64 // seconds after the sender's last event
}

// Plan samples a plan: x = opener; repeat y ~ p(·|x),
// append y, until [EOF]. Everything random comes from seed, so both ends get
// the same plan. It samples on demand, one record ahead of the walk, so the
// first records need not wait for the whole plan.
type Plan struct {
	m       *engine.Model
	t       *engine.Tables
	random  *rand.Rand
	stream  *engine.Stream
	noise   []float32
	records []PlannedRecord
	max     int
	done    bool // [EOF] sampled, or max records
}

func NewPlan(m *engine.Model, t *engine.Tables, opener int, seed uint64) *Plan {
	p := &Plan{
		m: m, t: t,
		random: rand.New(rand.NewPCG(seed, 0)),
		stream: m.NewStream(-1), // the sync model has no role
		noise:  make([]float32, max(m.SizeOut, m.Bt)),
		max:    min(t.MaxRecords, m.MaxSeq-1),
	}
	// x = the opener
	sizeBin, timeBin := t.Encode(float64(opener), 0, +1)
	p.stream.Step(sizeBin, timeBin)
	p.records = []PlannedRecord{{FromClient: true, Size: opener}}
	p.done = len(p.records) >= p.max
	return p
}

// At returns record i, sampling up to it; false past the end of the plan.
func (p *Plan) At(i int) (PlannedRecord, bool) {
	for len(p.records) <= i && !p.done {
		p.sample()
	}
	if i < len(p.records) {
		return p.records[i], true
	}
	return PlannedRecord{}, false
}

// sample: y ~ p(·|x), append y to x.
func (p *Plan) sample() {
	m, t := p.m, p.t
	fillNoise(p.noise[:m.SizeOut], p.random)
	sizeBin := p.stream.DecideSize(nil, nil, p.noise[:m.SizeOut])
	if sizeBin == m.Bs {
		p.done = true // [EOF]
		return
	}
	fillNoise(p.noise[:m.Bt], p.random)
	timeBin := p.stream.DecideTime(sizeBin, nil, p.noise[:m.Bt])

	size, dir := t.DrawSize(sizeBin, p.random)
	gap := t.DrawGap(timeBin, p.random)
	p.records = append(p.records, PlannedRecord{FromClient: dir > 0, Size: size, Gap: gap})
	p.stream.Step(sizeBin, timeBin)
	p.done = len(p.records) >= p.max
}

// fillNoise fills buf with Exp(1) values, which the engine uses for its draws.
func fillNoise(buf []float32, random *rand.Rand) {
	for i := range buf {
		buf[i] = float32(random.ExpFloat64())
	}
}

// Sync walks a plan on one side.
type Sync struct {
	plan     *Plan
	isClient bool
	next     int     // index of the next record
	clock    float64 // the time of my last event
}

func NewSync(plan *Plan, isClient bool) *Sync {
	return &Sync{plan: plan, isClient: isClient}
}

// First is the first step: the client sends the opener, the server waits.
func (s *Sync) First() Step {
	return s.current()
}

// Sent: my record left at now.
func (s *Sync) Sent(now float64) Step {
	s.clock = math.Max(s.clock, now)
	s.next++
	return s.current()
}

// Received: the peer's record arrived at now.
func (s *Sync) Received(now float64) Step {
	s.clock = math.Max(s.clock, now)
	s.next++
	return s.current()
}

func (s *Sync) current() Step {
	r, ok := s.plan.At(s.next)
	if !ok {
		return Step{Kind: End} // the plan is over: [EOF]
	}
	if r.FromClient != s.isClient {
		return Step{Kind: Wait, At: -1} // the peer's record: wait, no deadline
	}
	at := s.clock + r.Gap
	if s.next == 0 {
		at = 0 // the opener goes at once
	}
	return Step{Kind: Send, At: at, Size: r.Size}
}
