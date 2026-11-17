#!/bin/sh
# Install the obfsgen server as a systemd service.
#
#   sudo ./install.sh                  # port 443, certificate name example.com
#   sudo PORT=8443 NAME=cdn.example.org ./install.sh
#
# Keeps /etc/obfsgen (certificate, key, PSK) across reinstalls. To use a real
# certificate, put cert.pem and key.pem there and delete /etc/obfsgen/pin. The
# package's parameter folder goes to /usr/local/share/obfsgen.
set -eu
PORT=${PORT:-443}
NAME=${NAME:-example.com}
DIR=/etc/obfsgen
PARAMS=/usr/local/share/obfsgen
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "systemd is required" >&2; exit 1; }

case $(uname -m) in
x86_64)
    if grep -qw avx2 /proc/cpuinfo; then BIN=obfsgen-linux-amd64; else BIN=obfsgen-linux-amd64-portable; fi ;;
aarch64 | arm64) BIN=obfsgen-linux-arm64 ;;
*) echo "unsupported CPU: $(uname -m)" >&2; exit 1 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
install -m 755 "$HERE/bin/$BIN" /usr/local/bin/obfsgen
rm -rf "$PARAMS" && mkdir -p "$PARAMS" && cp -R "$HERE/params/." "$PARAMS/"

mkdir -p "$DIR" && chmod 700 "$DIR"
if [ ! -f "$DIR/cert.pem" ]; then
    /usr/local/bin/obfsgen gencert -name "$NAME" -dir "$DIR" > "$DIR/pin"
fi
if [ ! -f "$DIR/psk" ]; then
    head -c 24 /dev/urandom | base64 | tr -d '/+=\n' > "$DIR/psk"
    chmod 600 "$DIR/psk"
fi

cat > /etc/systemd/system/obfsgen.service <<EOF
[Unit]
Description=obfsgen shaped proxy server
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/obfsgen server -listen :$PORT -cert $DIR/cert.pem -key $DIR/key.pem -psk-file $DIR/psk -params $PARAMS -quiet
Restart=always
RestartSec=2
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable obfsgen >/dev/null 2>&1
systemctl restart obfsgen
sleep 1
systemctl --no-pager --lines=3 status obfsgen || true

IP=$(curl -4 -s --max-time 5 https://api.ipify.org 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
PSK=$(cat "$DIR/psk")
PIN=$(cat "$DIR/pin" 2>/dev/null || true)
echo
echo "obfsgen server ($BIN) is running on port $PORT."
echo "Client settings:"
echo "  server  $IP:$PORT"
echo "  sni     $NAME"
echo "  psk     $PSK"
[ -n "$PIN" ] && echo "  pin     $PIN"
echo
echo "Import link:"
echo "  obfsgen://$PSK@$IP:$PORT?sni=$NAME${PIN:+&pin=$PIN}"
echo
echo "Desktop client (with the same parameter folder as the server):"
echo "  obfsgen client -server $IP:$PORT -sni $NAME -psk '$PSK'${PIN:+ -pin $PIN} -socks 127.0.0.1:1080"
