#!/usr/bin/env bash
# Build the Linux server package: static binaries (cgo + musl via zig cc) for
# amd64 (AVX2 and portable) and arm64 (NEON), plus install.sh and the
# parameter folder PARAMS (default: params/example/campus-gateway).
#
#   ZIG="python3 -m ziglang" scripts/build-server.sh [out-dir]    (in endpoint/)
#
# ZIG is the zig command (default: zig). `pip install ziglang` provides one.
set -euo pipefail
PARAMS=${PARAMS:-$(dirname "$0")/../../params/example/campus-gateway}
[ -f "$PARAMS/async/weights.obfw" ] || { echo "no parameter folder at $PARAMS" >&2; exit 1; }
PARAMS=$(cd "$PARAMS" && pwd)
cd "$(dirname "$0")/.."
OUT=${1:-dist}
ZIG=${ZIG:-zig}
VERSION=$(git describe --always --dirty 2>/dev/null || echo dev)
PKG="$OUT/obfsgen-server-$VERSION"
WRAP=$(mktemp -d)
trap 'rm -rf "$WRAP"' EXIT
rm -rf "$PKG" && mkdir -p "$PKG/bin"

# The C engine picks its kernels from the compiler's target: x86_64_v3 turns
# on AVX2/FMA/F16C, plain x86_64 gives the portable C path, arm64 has NEON.
build() { # goarch zig-target-and-cpu output
    printf '#!/bin/sh\nexec %s cc -target %s "$@"\n' "$ZIG" "$2" > "$WRAP/cc-$3"
    chmod +x "$WRAP/cc-$3"
    CGO_ENABLED=1 GOOS=linux GOARCH="$1" CC="$WRAP/cc-$3" \
        go build -trimpath -ldflags "-s -w -linkmode external -extldflags -static" \
        -o "$PKG/bin/$3" .
    echo "built $3"
}
build amd64 "x86_64-linux-musl -mcpu=x86_64_v3" obfsgen-linux-amd64
build amd64 "x86_64-linux-musl" obfsgen-linux-amd64-portable
build arm64 "aarch64-linux-musl" obfsgen-linux-arm64

# the parameter folder, without the PyTorch checkpoints
for form in async sync; do
    mkdir -p "$PKG/params/$form"
    cp "$PARAMS/$form/weights.obfw" "$PARAMS/$form/tokenizer.json" "$PKG/params/$form/"
done
cp "$PARAMS/opener.json" "$PKG/params/"
cp scripts/install-server.sh "$PKG/install.sh"
chmod +x "$PKG/install.sh"
tar -C "$OUT" -czf "$PKG.tar.gz" "$(basename "$PKG")"
echo "package: $PKG.tar.gz"
