package tunnel

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"time"

	"obfsgen/scheduler"
)

// tunnel plays one schedule on one TLS connection (either side):
//
//	+--> wait for the first of:
//	|      (a) my send is due        ==> sendRecord():    buffer ==> record ==> TLS
//	|      (b) a record arrives      ==> receiveRecord(): record ==> packets ==> KCP
//	|      (c) async deadline passed ==> the scheduler decides again
//	|      (d) the peer closed       ==> its [EOF]: stop
//	|    the scheduler gives the next step
//	+--- until the step is [EOF] (the caller then closes the connection)
type tunnel struct {
	conn    *tls.Conn
	out     *buffer             // KCP packets to send
	deliver func(packet []byte) // hands a received KCP packet to KCP
	joiner  *joiner             // joins cut segments (shared by the session's tunnels)
	hello   []byte              // client: sent in the first record, then nil

	// The scheduler: exactly one of the two is set.
	async *scheduler.Async // asynchronous mode
	sync  *scheduler.Sync  // synchronous mode
	next  scheduler.Step   // what to do next

	start time.Time // time 0 of the schedule
}

// received is one record read from the connection, or the error that ended it.
type received struct {
	plaintext []byte
	at        time.Time
	err       error
}

// readRecords reads records in a goroutine and sends them on the returned
// channel. The goroutine stops when the connection ends or done is closed.
func readRecords(conn *tls.Conn, done <-chan struct{}) <-chan received {
	ch := make(chan received, 64)
	go func() {
		for {
			p := make([]byte, maxPlain+1)
			n, err := conn.Read(p) // one record per Read
			var r received
			if n > 0 {
				r = received{plaintext: p[:n], at: time.Now()}
			} else {
				r = received{err: err}
			}
			select {
			case ch <- r:
			case <-done: // nobody reads any more
				return
			}
			if n == 0 {
				return // the error was sent
			}
		}
	}()
	return ch
}

func (t *tunnel) run(ctx context.Context, records <-chan received) error {
	for t.next.Kind != scheduler.End {
		// The timer of the next step: a send time, or an async deadline.
		// Wait with no deadline: no timer.
		var timer <-chan time.Time
		if t.next.At >= 0 {
			timer = time.After(time.Until(t.timeAt(t.next.At)))
		}

		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-timer:
			// A record that is already in comes first: it may cancel the send.
			select {
			case r := <-records:
				if err := t.handle(r); err != nil || r.err != nil {
					return err
				}
				continue
			default:
			}
			if t.next.Kind == scheduler.Send {
				// (a) the schedule says: send now
				if err := t.sendRecord(); err != nil {
					return err
				}
			} else {
				// (c) async: waited long enough, decide again
				t.next = t.async.Timeout(t.now())
			}

		case r := <-records:
			if err := t.handle(r); err != nil || r.err != nil {
				return err
			}
		}
	}
	return nil
}

// handle: (b) a record from the peer, or (d) the connection's end. After
// r.err != nil the tunnel is over; the error is nil for the peer's [EOF].
func (t *tunnel) handle(r received) error {
	if errors.Is(r.err, io.EOF) {
		return nil // (d) the peer closed: its [EOF]
	}
	if r.err != nil {
		return r.err
	}
	return t.receiveRecord(r) // (b)
}

// now: seconds since the schedule started.
func (t *tunnel) now() float64 {
	return time.Since(t.start).Seconds()
}

// timeAt: the clock time of a schedule time.
func (t *tunnel) timeAt(seconds float64) time.Time {
	return t.start.Add(time.Duration(seconds * float64(time.Second)))
}

func (t *tunnel) sendRecord() error {
	now := t.now()

	var flags byte
	if t.hello != nil && t.sync != nil {
		flags = flagSync // tells the server to use synchronous mode
	}
	p := addPadding(t.next.Size, flags, t.hello, t.out)
	t.hello = nil // only in the first record

	if _, err := t.conn.Write(p); err != nil {
		return err
	}
	if t.sync != nil {
		t.next = t.sync.Sent(now)
	} else {
		t.next = t.async.Sent(now)
	}
	return nil
}

func (t *tunnel) receiveRecord(r received) error {
	rec, err := removePadding(r.plaintext)
	if err != nil {
		return err
	}
	if err := unpack(rec.data, t.joiner, t.deliver); err != nil {
		return err
	}

	at := r.at.Sub(t.start).Seconds()
	if t.sync != nil {
		t.next = t.sync.Received(at)
	} else {
		t.next = t.async.Received(rec.size, at)
	}
	return nil
}
