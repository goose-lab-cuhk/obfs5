##! Writes obf_flow (connections), obf_rec (TLS records) and obf_pkt (TCP
##! packets) for model/build_dataset.py.
##!
##!     zeek -C -r capture.pcap obfsgen.zeek    # a pcap (pcaps.sh runs this per file)
##!     zeek -C -i eth0 obfsgen.zeek            # a live interface, until Ctrl-C
##!     @load /path/to/obfsgen.zeek             # in zeekctl's local.zeek: long captures

module Obfsgen;

export {
	redef enum Log::ID += { FLOW_LOG, REC_LOG, PKT_LOG };

	## Suppress every other log (conn.log, ssl.log, weird.log, ...).
	const only_own_logs = T &redef;

	## One row per TCP packet.
	type PktInfo: record {
		ts:      time   &log;   ##< the packet's timestamp
		uid:     string &log;   ##< joins to obf_rec.log / obf_flow.log
		is_orig: bool   &log;   ##< sent by the connection originator
		flags:   string &log;   ##< SYN FIN RST ACK PSH URG
		seq:     count  &log;   ##< relative sequence number (first data byte = 1)
		ack:     count  &log;   ##< relative ack, in the peer's sequence space
		len:     count  &log;   ##< TCP payload length
	};

	## One row per TLS record.
	type RecInfo: record {
		tail_ts:     time   &log;  ##< when the reassembler could deliver the record
		uid:         string &log;
		is_orig:     bool   &log;  ##< sent by the connection originator
		ctype:       count  &log;  ##< 20 CCS, 21 alert, 22 handshake, 23 app data
		length:      count  &log;  ##< record body length (wire size is 5 + length)
		off:         count  &log;  ##< 0-based offset of the header in its direction
		encrypted:   bool   &log;  ##< sent after encryption started
		probable_hs: bool   &log;  ##< TLS 1.3 encrypted handshake / NewSessionTicket
	};

	## One row per connection, written when Zeek removes its state.
	type FlowInfo: record {
		ts:          time    &log;              ##< connection start
		uid:         string  &log;
		id:          conn_id &log;              ##< id.orig_h, id.orig_p, ...
		proto:       string  &log &default="";  ##< tcp / udp / icmp
		capture:     string  &log &default="";  ##< pcap path, or interface when live
		service:     string  &log &default="";  ##< Zeek's protocol verdict, e.g. "ssl"
		## Connection history, uppercase = originator: retransmissions (t/T),
		## reassembly gaps (g/G), direction flip (^), FIN (f/F), RST (r/R).
		history:     string  &log &default="";
		t_syn:       time    &log &optional;    ##< originator SYN
		t_synack:    time    &log &optional;    ##< responder SYN-ACK
		t_ack:       time    &log &optional;    ##< originator ACK
		sni:         string  &log &default="";  ##< server name from the ClientHello
		tls_version: count   &log &default=0;   ##< negotiated, e.g. 0x0304 = TLS 1.3
		tls_estab:   bool    &log &default=F;   ##< ssl_established fired
		t_estab:     time    &log &optional;    ##< when it fired
		ssl_history: string  &log &default="";  ##< TLS message sequence
		missed:      count   &log &default=0;   ##< bytes the reassembler never saw
		n_pkt:       count   &log &default=0;   ##< TCP packets seen
		n_rec:       count   &log &default=0;   ##< TLS records seen
	};
}

# Required: by default the SSL analyzer detaches at ssl_established, and no
# application-data record would be logged.
redef SSL::disable_analyzer_after_detection = F;

# Keep an idle long-lived connection as one connection (the default splits it
# after 5 minutes of silence).
redef tcp_inactivity_timeout = 1 hr;

global capture_path = "";

# Veto writes to every other stream, so their files are never created.
# Stats::LOG and CaptureLoss::LOG exist only under zeekctl.
hook Log::log_stream_policy(rec: any, id: Log::ID)
	{
	if ( ! only_own_logs )
		return;
	if ( id == Obfsgen::REC_LOG || id == Obfsgen::FLOW_LOG || id == Obfsgen::PKT_LOG )
		return;
@ifdef ( Stats::LOG )
	if ( id == Stats::LOG )
		return;
@endif
@ifdef ( CaptureLoss::LOG )
	if ( id == CaptureLoss::LOG )
		return;
@endif
	break;
	}

redef record connection += {
	obf:       FlowInfo &optional;
	off_c:     count &default=0;   ##< record-stream offset, TLS client
	off_s:     count &default=0;   ##< ... and TLS server
	pend_hs_c: bool  &default=F;   ##< probable_hs fires just before its record
	pend_hs_s: bool  &default=F;
	flipped:   bool  &default=F;
};

function flow(c: connection): FlowInfo
	{
	if ( ! c?$obf )
		c$obf = FlowInfo($ts=c$start_time, $uid=c$uid, $id=c$id, $capture=capture_path);
	return c$obf;
	}

event zeek_init() &priority=5
	{
	if ( reading_live_traffic() || reading_traces() )
		capture_path = packet_source()$path;

	Log::create_stream(Obfsgen::REC_LOG, Log::Stream($columns=RecInfo, $path="obf_rec"));
	Log::create_stream(Obfsgen::FLOW_LOG, Log::Stream($columns=FlowInfo, $path="obf_flow"));
	Log::create_stream(Obfsgen::PKT_LOG, Log::Stream($columns=PktInfo, $path="obf_pkt"));
	}

event tcp_packet(c: connection, is_orig: bool, flags: string, seq: count,
                 ack: count, len: count, payload: string)
	{
	++flow(c)$n_pkt;
	Log::write(Obfsgen::PKT_LOG, PktInfo($ts=get_current_packet_ts(), $uid=c$uid,
	    $is_orig=is_orig, $flags=flags, $seq=seq, $ack=ack, $len=len));
	}

event connection_SYN_packet(c: connection, pkt: SYN_packet)
	{
	# This event also fires for the responder's SYN-ACK. In a capture that starts
	# at the SYN-ACK, that is the only SYN-bearing packet; latching it would fake
	# a handshake with t_syn == t_synack. Leaving t_syn unset drops such flows.
	if ( ! pkt$is_orig )
		return;

	local f = flow(c);
	if ( ! f?$t_syn )
		f$t_syn = get_current_packet_ts();
	}

event connection_established(c: connection)      # the responder's SYN-ACK
	{
	local f = flow(c);
	if ( ! f?$t_synack )
		f$t_synack = get_current_packet_ts();
	}

event connection_first_ACK(c: connection)        # the originator's ACK
	{
	local f = flow(c);
	if ( ! f?$t_ack )
		f$t_ack = get_current_packet_ts();
	}

function emit_record(c: connection, is_client: bool, ctype: count, length: count,
                     encrypted: bool)
	{
	# The SSL analyzer speaks TLS client/server, the logs speak TCP
	# originator/responder; they differ only after ssl_connection_flipped.
	local is_orig = c$flipped ? ! is_client : is_client;

	local off = is_client ? c$off_c : c$off_s;
	local hs  = is_client ? c$pend_hs_c : c$pend_hs_s;

	if ( is_client )
		{ c$off_c += 5 + length; c$pend_hs_c = F; }
	else
		{ c$off_s += 5 + length; c$pend_hs_s = F; }

	++flow(c)$n_rec;

	Log::write(Obfsgen::REC_LOG, RecInfo(
	    $tail_ts=get_current_packet_ts(), $uid=c$uid, $is_orig=is_orig,
	    $ctype=ctype, $length=length, $off=off, $encrypted=encrypted,
	    $probable_hs=hs));
	}

event ssl_plaintext_data(c: connection, is_client: bool, record_version: count,
                         content_type: count, length: count)
	{
	emit_record(c, is_client, content_type, length, F);
	}

event ssl_encrypted_data(c: connection, is_client: bool, record_version: count,
                         content_type: count, length: count)
	{
	emit_record(c, is_client, content_type, length, T);
	}

# Raised just before the record event it describes, without an offset: latch
# it per direction and consume it in emit_record().
event ssl_probable_encrypted_handshake_message(c: connection, is_client: bool,
                                               length: count)
	{
	if ( is_client )
		c$pend_hs_c = T;
	else
		c$pend_hs_s = T;
	}

event ssl_established(c: connection)
	{
	local f = flow(c);
	f$tls_estab = T;
	f$t_estab = get_current_packet_ts();
	}

event ssl_connection_flipped(c: connection)
	{
	c$flipped = T;
	}

# Priority -5: runs after the base scripts have filled c$conn and c$ssl.
event connection_state_remove(c: connection) &priority=-5
	{
	local f = flow(c);
	f$history = c$conn$history;
	f$proto = fmt("%s", get_port_transport_proto(c$id$resp_p));

	if ( c$conn?$service )
		f$service = c$conn$service;
	if ( c$conn?$missed_bytes )
		f$missed = c$conn$missed_bytes;

	if ( c?$ssl )
		{
		if ( c$ssl?$version_num )
			f$tls_version = c$ssl$version_num;
		if ( c$ssl?$ssl_history )
			f$ssl_history = c$ssl$ssl_history;
		if ( c$ssl?$server_name )
			f$sni = c$ssl$server_name;
		}

	Log::write(Obfsgen::FLOW_LOG, f);
	}
