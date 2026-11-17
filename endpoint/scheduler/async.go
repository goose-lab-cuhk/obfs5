// Package scheduler decides the size and time of every record. No I/O:
//
//	tunnel ==Sent / Received / Timeout (seconds since start)==> scheduler ==> next Step
package scheduler

import (
	"math"
	"math/rand/v2"

	"obfsgen/config"
	"obfsgen/engine"
)

type Step struct {
	Kind Kind
	At   float64 // Send: when to send. Wait: when to give up waiting (−1: never)
	Size int     // Send: the record's size in bytes
}

type Kind int

const (
	Send Kind = iota // send a record of Size bytes at At
	Wait             // wait for the peer's record until At
	End              // [EOF]: close the connection
)

const (
	ClientRole = 0
	ServerRole = 1
)

const (
	noEvent = iota
	sentEvent
	receivedEvent
)

// Async: one endpoint in asynchronous mode. Its stream = its sends + the arrivals it sees.
//
//	Open (client) / Listen (server)
//	      |
//	      v
//	+-> decide: sample y ==[EOF]==> End
//	|     | send                 | receive
//	|     v                      v
//	|   Send at κ + Δ          Wait until deadline
//	|     |                      |   Timeout: k+1, sample with time bins ≥ w
//	|     |                      |   (k > 4: no receive) ==> Send / Wait / End
//	|     | Sent                 | Received (also cancels a pending Send)
//	+-----+----------------------+
type Async struct {
	model  *engine.Model
	tables *engine.Tables
	stream *engine.Stream // the model's state over this endpoint's records
	random *rand.Rand
	rtt    float64          // R, seconds
	p      config.Scheduler // the deadline κ + α·quantile(λ) + R + β, and the rounds

	clock      float64 // κ: the time of my last event
	lastEvent  int     // noEvent, sentEvent or receivedEvent
	records    int     // records in my stream
	maxRecords int

	state Kind // Send: a send is pending. Wait: waiting. End: done
	send  struct {
		size             int
		sizeBin, timeBin int
		at               float64
		isOpener         bool // the opener is never cancelled
	}
	deadline float64 // when Wait gives up

	rounds          int       // k: deadlines passed in a row
	onlySendsOrEOF  []uint8   // size-bin mask: send bins and [EOF]
	noise           []float32 // Exp(1) noise for the model's draws
	receiveGapProbs []float64 // per time bin
}

func NewAsync(m *engine.Model, t *engine.Tables, role int, rtt float64, random *rand.Rand, p config.Scheduler) *Async {
	a := &Async{
		model:  m,
		tables: t,
		stream: m.NewStream(role),
		random: random,
		rtt:    rtt,
		p:      p,
		state:  Wait,
	}
	a.maxRecords = min(t.MaxRecords, m.MaxSeq-1)

	a.onlySendsOrEOF = make([]uint8, m.SizeOut)
	copy(a.onlySendsOrEOF, t.SendBins)
	a.onlySendsOrEOF[m.Bs] = 1 // size bin Bs is [EOF]

	a.noise = make([]float32, max(m.SizeOut, m.Bt))
	a.receiveGapProbs = make([]float64, m.Bt)
	return a
}

// Open: the client starts with the opener.
func (a *Async) Open(size int) Step {
	sizeBin, timeBin := a.tables.Encode(float64(size), 0, +1)
	a.planSend(size, sizeBin, timeBin, 0)
	a.send.isOpener = true
	return a.current()
}

// Listen: the server waits for the opener.
func (a *Async) Listen() Step {
	a.state = Wait
	return a.current()
}

func (a *Async) Sent(now float64) Step {
	if a.state != Send {
		return a.current()
	}
	a.clock = math.Max(a.clock, now)
	a.stream.Step(a.send.sizeBin, a.send.timeBin)
	a.records++
	a.lastEvent = sentEvent
	return a.decide()
}

// Received cancels a pending send (except the opener).
func (a *Async) Received(size int, now float64) Step {
	if a.state == End {
		return a.current()
	}
	if a.records >= a.maxRecords {
		a.state = End
		return a.current()
	}

	// observed gap = max(0, w − R·[my last event was a send])
	gap := 0.0
	if a.lastEvent != noEvent {
		waited := now - a.clock
		if a.lastEvent == sentEvent {
			waited = waited - a.rtt
		}
		gap = math.Max(waited, 0)
	}
	a.clock = math.Max(a.clock, now)

	sizeBin, timeBin := a.tables.Encode(float64(size), gap, -1)
	a.stream.Step(sizeBin, timeBin)
	a.records++
	a.lastEvent = receivedEvent

	if a.state == Send && a.send.isOpener {
		return a.current() // keep the opener
	}
	return a.decide()
}

// Timeout: k ← k+1, sample again with time bins ≥ w.
func (a *Async) Timeout(now float64) Step {
	if a.state != Wait {
		return a.current()
	}
	a.rounds++
	waited := math.Max(now-a.clock, 0)
	waitedBin := a.tables.TimeBin(waited)
	timeMask := a.tables.WaitedMask(waited) // time bins ≥ waited

	var sizeMask []uint8 // nil: the size is conditioned on timeMask
	if waitedBin >= a.tables.Bt-1 || a.rounds > a.p.MaxRounds {
		sizeMask = a.onlySendsOrEOF // no receive: send or [EOF]
	}
	sizeBin := a.stream.DecideSize(sizeMask, timeMask, a.drawNoise(a.model.SizeOut))
	if sizeBin == a.model.Bs {
		a.state = End // [EOF]
		return a.current()
	}
	timeBin := a.stream.DecideTime(sizeBin, timeMask, a.drawNoise(a.model.Bt))
	size, dir := a.tables.DrawSize(sizeBin, a.random)
	gap := a.tables.DrawGap(timeBin, a.random)

	if dir < 0 {
		return a.waitUntilDeadline(now, waitedBin)
	}

	at := math.Max(a.clock+gap, now)
	sizeBin, timeBin = a.tables.Encode(float64(size), at-a.clock, +1)
	a.planSend(size, sizeBin, timeBin, at)
	return a.current()
}

// decide: sample the next record (k = 0).
func (a *Async) decide() Step {
	if a.records >= a.maxRecords {
		a.state = End
		return a.current()
	}
	sizeBin := a.stream.DecideSize(nil, nil, a.drawNoise(a.model.SizeOut))
	if sizeBin == a.model.Bs {
		a.state = End // [EOF]
		return a.current()
	}
	timeBin := a.stream.DecideTime(sizeBin, nil, a.drawNoise(a.model.Bt))
	size, dir := a.tables.DrawSize(sizeBin, a.random)
	gap := a.tables.DrawGap(timeBin, a.random)

	if dir > 0 {
		a.planSend(size, sizeBin, timeBin, a.clock+gap) // κ' = κ + Δ
		return a.current()
	}
	a.rounds = 0
	return a.waitUntilDeadline(a.clock, 0)
}

func (a *Async) planSend(size, sizeBin, timeBin int, at float64) {
	a.state = Send
	a.send.size = size
	a.send.sizeBin = sizeBin
	a.send.timeBin = timeBin
	a.send.at = at
	a.send.isOpener = false
}

// waitUntilDeadline: κ + α·quantile(λ) + R + β, quantile over time bins ≥ minBin.
func (a *Async) waitUntilDeadline(now float64, minBin int) Step {
	a.stream.ReceiveGapPMF(minBin, a.tables.RecvBins, a.receiveGapProbs)
	lambda := a.p.Lambda[0] + (a.p.Lambda[1]-a.p.Lambda[0])*a.random.Float64()
	quantile, _ := a.tables.GapFromQuantile(a.receiveGapProbs, lambda, a.random)

	a.state = Wait
	a.deadline = math.Max(a.clock+a.p.Alpha*quantile+a.rtt+a.p.Beta, now)
	return a.current()
}

func (a *Async) current() Step {
	switch a.state {
	case Send:
		return Step{Kind: Send, At: a.send.at, Size: a.send.size}
	case Wait:
		if a.records == 0 {
			return Step{Kind: Wait, At: -1} // the server waits for the opener
		}
		return Step{Kind: Wait, At: a.deadline}
	}
	return Step{Kind: End}
}

// drawNoise: Exp(1) values for the engine's draws.
func (a *Async) drawNoise(n int) []float32 {
	for i := 0; i < n; i++ {
		a.noise[i] = float32(a.random.ExpFloat64())
	}
	return a.noise[:n]
}
