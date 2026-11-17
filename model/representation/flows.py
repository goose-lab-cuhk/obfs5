"""One connection's Zeek rows to a checked flow: packets in time order, TLS
application-data records joined to the packets that carried them, handshake RTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
_FLAG_BITS = {"F": FIN, "S": SYN, "R": RST, "P": PSH, "A": ACK, "U": URG}

# The checks, in the order build_flow applies them.
NOT_TCP = "not_tcp"
TOO_MANY_PACKETS = "too_many_packets"
INCOMPLETE_LOGS = "incomplete_logs"  # the rows found differ from Zeek's own counts
NO_TLS = "no_tls_established"
NO_SYN_ACK = "no_syn_ack"
NO_HANDSHAKE = "no_full_handshake"
EMPTY = "empty"
SEQ_UNDERFLOW = "seq_underflow"  # data before the first byte: the offsets cannot be trusted
PREFIX_MISSING = "stream_prefix_missing"  # a direction's first data byte was not captured
JOIN_FAILED = "record_join_failed"  # a record byte that no packet carried
NO_CLIENT_DATA = "no_upstream_data"
NO_SERVER_DATA = "no_downstream_data"
DAMAGED = "reassembly_damage"  # Zeek missed bytes or saw a content gap


def flag_bits(flags: Optional[str]) -> int:
    """Zeek's flag letters ("ShAD" style) → a TCP flag bitmask."""
    bits = 0
    for ch in flags or "":
        bits |= _FLAG_BITS.get(ch, 0)
    return bits


@dataclass
class Flow:
    """A checked connection. Packets are in time order; ``dir`` is +1 for the
    originator (the client). Records are in the order their first bytes
    appeared; ``head`` / ``tail`` index the packets that carried their first /
    last byte, and ``off`` is the record's offset in its direction."""

    ts: np.ndarray
    dir: np.ndarray
    flags: np.ndarray
    seq: np.ndarray
    ack: np.ndarray
    length: np.ndarray
    rec_dir: np.ndarray
    rec_len: np.ndarray
    rec_off: np.ndarray
    rec_head: np.ndarray
    rec_tail: np.ndarray
    rtt: float  # handshake RTT R = t(client ACK) - t(SYN)
    rtt_server: float  # R_s = t(SYN-ACK) - t(SYN): capture point <-> server
    rtt_client: float  # R_c = t(client ACK) - t(SYN-ACK): capture point <-> client
    closed: bool  # a FIN or RST was seen


def build_flow(
    row: dict,
    pkt: Optional[Dict[str, np.ndarray]],
    rec: Optional[Dict[str, np.ndarray]],
    n_rec_all: int,
    max_packets: int,
) -> Tuple[Optional[Flow], Optional[str]]:
    """One connection's obf_flow row, packet columns and application-data record
    columns (log order) → (Flow, None), or (None, the first check it fails)."""
    n_pkt = int(row.get("n_pkt") or 0)
    if row.get("proto") != "tcp":
        return None, NOT_TCP
    if n_pkt > max_packets:
        return None, TOO_MANY_PACKETS
    n_have = 0 if pkt is None else len(pkt["ts"])
    if n_have != n_pkt or n_rec_all != int(row.get("n_rec") or 0):
        return None, INCOMPLETE_LOGS
    if not row.get("tls_estab"):
        return None, NO_TLS
    if row.get("t_synack") is None:
        return None, NO_SYN_ACK
    if row.get("t_syn") is None or row.get("t_ack") is None:
        return None, NO_HANDSHAKE
    if n_have == 0:
        return None, EMPTY

    order = np.argsort(pkt["ts"], kind="stable")  # log order breaks timestamp ties
    ts = pkt["ts"][order]
    d = np.where(pkt["is_orig"][order], 1, -1)
    flags = pkt["flags"][order]
    seq, ack, length = pkt["seq"][order], pkt["ack"][order], pkt["len"][order]

    joined = _join_records(d, seq, length, rec)
    if isinstance(joined, str):
        return None, joined
    rec_dir, rec_len, rec_off, rec_head, rec_tail = joined

    rtt = _handshake_rtt(d, flags, ts)
    if rtt is None:
        return None, NO_HANDSHAKE if _saw_syn_ack(d, flags) else NO_SYN_ACK
    if not np.any((d == 1) & (length > 0)):
        return None, NO_CLIENT_DATA
    if not np.any((d == -1) & (length > 0)):
        return None, NO_SERVER_DATA
    if rec_dir.size == 0:
        return None, NO_TLS
    history = row.get("history") or ""
    if int(row.get("missed") or 0) > 0 or "g" in history.lower():
        return None, DAMAGED

    return (
        Flow(
            ts,
            d,
            flags,
            seq,
            ack,
            length,
            rec_dir,
            rec_len,
            rec_off,
            rec_head,
            rec_tail,
            rtt[0],
            rtt[1],
            rtt[2],
            bool(np.any(flags & (FIN | RST))),
        ),
        None,
    )


def _join_records(d, seq, length, rec):
    """Join records to the packets carrying their first and last byte. Relative
    seq numbers start at 1 and offsets at 0, so a record spans off + 1 to
    off + 5 + length; a retransmitted segment counts at its first arrival."""
    spans = {}
    for side in (1, -1):
        idx = np.flatnonzero((d == side) & (length > 0))
        s_seq, first = np.unique(seq[idx], return_index=True)
        spans[side] = (s_seq, length[idx[first]], idx[first])
        if s_seq.size and s_seq[0] < 1:
            return SEQ_UNDERFLOW

    parts = []
    for side in (1, -1):
        if rec is None:
            break
        mine = np.flatnonzero(np.where(rec["is_orig"], 1, -1) == side)
        if mine.size == 0:
            continue
        s_seq, s_len, s_pkt = spans[side]
        if s_seq.size == 0:
            return JOIN_FAILED
        if s_seq[0] != 1:
            return PREFIX_MISSING
        mine = mine[np.argsort(rec["off"][mine], kind="stable")]
        off, rlen = rec["off"][mine], rec["length"][mine]
        located = []
        for target in (off + 1, off + 5 + rlen):
            k = np.searchsorted(s_seq, target, side="right") - 1
            if np.any(target >= s_seq[k] + s_len[k]):
                return JOIN_FAILED
            located.append(s_pkt[k])
        parts.append((np.full(mine.size, side), rlen, off, located[0], located[1]))

    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty, empty
    cols = [np.concatenate(c) for c in zip(*parts)]
    order = np.argsort(cols[3], kind="stable")
    return tuple(c[order] for c in cols)


def _handshake_rtt(d, flags, ts) -> Optional[Tuple[float, float, float]]:
    """(R, R_s, R_c) from the first 10 packets: the latest SYN, a SYN-ACK from the
    other side, then an ACK from the SYN's side. None without all three."""
    syn_t = syn_side = synack_t = None
    for i in range(min(10, len(ts))):
        f = int(flags[i])
        if f & SYN and not f & ACK:
            syn_t, syn_side, synack_t = ts[i], d[i], None
            continue
        if syn_t is None:
            continue
        if synack_t is None and f & SYN and f & ACK:
            if d[i] != syn_side:
                synack_t = ts[i]
            continue
        if synack_t is not None and d[i] == syn_side and f & ACK:
            full = ts[i] - syn_t
            if full > 0:
                return float(full), float(synack_t - syn_t), float(ts[i] - synack_t)
    return None


def _saw_syn_ack(d, flags) -> bool:
    syn_side = None
    for i in range(min(10, len(d))):
        f = int(flags[i])
        if f & SYN and not f & ACK:
            syn_side = d[i]
        elif syn_side is not None and f & SYN and f & ACK and d[i] != syn_side:
            return True
    return False
