"""A flow's three sequences of (signed size, gap).

The client and server streams (async model) are the records as each endpoint
saw them, ordered by the ack walk; a size is + for the endpoint's own sends.
The sync stream (sync model) is the records as the capture point saw them; a
size is + for client to server. A gap runs from the previous record's last
byte to this record's first, minus the part of the handshake RTT the record
waited for: R after an endpoint's own send (endpoint streams), R_s or R_c when
the direction changes (sync stream). Gaps are at least 0; the first is 0.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .flows import Flow

Sequence_ = Tuple[np.ndarray, np.ndarray]  # (signed sizes int32, gaps float64)


def endpoint_streams(flow: Flow) -> Tuple[Sequence_, Sequence_]:
    """The client stream and the server stream."""
    base = {side: _first_data_seq(flow, side) for side in (1, -1)}
    client = _endpoint(flow, 1, flow.rtt_client, base[-1])
    server = _endpoint(flow, -1, flow.rtt_server, base[1])
    return client, server


def sync_stream(flow: Flow) -> Sequence_:
    """The sync stream: the records in capture-point order."""
    order = np.lexsort((flow.rec_off, flow.rec_dir, flow.rec_head))
    heads = flow.ts[flow.rec_head[order]]
    tails = flow.ts[flow.rec_tail[order]]
    dirs = flow.rec_dir[order]
    sizes = (dirs * flow.rec_len[order]).astype(np.int32)
    gaps = np.zeros(len(order))
    for i in range(1, len(order)):
        g = heads[i] - tails[i - 1]
        if dirs[i - 1] == 1 and dirs[i] == -1:
            g -= flow.rtt_server
        elif dirs[i - 1] == -1 and dirs[i] == 1:
            g -= flow.rtt_client
        gaps[i] = max(0.0, g)
    return sizes, gaps


def _first_data_seq(flow: Flow, side: int) -> int:
    data = (flow.dir == side) & (flow.length > 0)
    return int(flow.seq[data].min()) if data.any() else 0


def _endpoint(flow: Flow, side: int, own_rtt: float, peer_base: int) -> Sequence_:
    """One endpoint's stream; ``side`` is the direction of its sends."""
    idx = np.arange(len(flow.rec_dir))
    sends = idx[flow.rec_dir == side]
    recvs = idx[flow.rec_dir == -side]
    sends = sends[np.argsort(flow.rec_off[sends], kind="stable")]
    recvs = recvs[np.argsort(flow.rec_off[recvs], kind="stable")]
    recv_end = flow.rec_off[recvs] + 5 + flow.rec_len[recvs]

    # Ack walk: before each send come the received records its ack covers.
    events: List[Tuple[int, bool]] = []  # (record, is_send)
    r = 0
    for s in sends:
        acked = (int(flow.ack[flow.rec_head[s]]) - peer_base) % (1 << 32)
        while r < len(recvs) and recv_end[r] <= acked:
            events.append((recvs[r], False))
            r += 1
        events.append((s, True))
    events.extend((recvs[k], False) for k in range(r, len(recvs)))

    shift = own_rtt / 2.0  # one-way delay between the capture point and the endpoint
    sizes = np.empty(len(events), dtype=np.int32)
    gaps = np.zeros(len(events))
    prev_tail, prev_send = 0.0, False
    for i, (k, is_send) in enumerate(events):
        moved = -shift if is_send else shift
        head = flow.ts[flow.rec_head[k]] + moved
        tail = flow.ts[flow.rec_tail[k]] + moved
        if i > 0:
            g = head - prev_tail
            if prev_send and not is_send:
                g -= flow.rtt
            gaps[i] = max(0.0, g)
        sizes[i] = flow.rec_len[k] if is_send else -flow.rec_len[k]
        prev_tail, prev_send = tail, is_send
    return sizes, gaps
