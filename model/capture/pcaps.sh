#!/usr/bin/env bash
# Run Zeek with obfsgen.zeek over pcap files: the logs that model/build_dataset.py reads.
#
#   pcaps.sh -o OUT PCAP_OR_DIR...    one log directory per pcap: OUT/<pcap name>/
# Options:
#   -f FILTER   BPF capture filter (default: "tcp port 443")
#   -j N        pcaps processed at once (default: the number of CPUs)
#
# Zeek is taken from $ZEEK if set, else from PATH. A directory argument is
# searched for *.pcap and *.pcapng files. A pcap whose log directory already
# exists is skipped, so an interrupted run can simply be repeated.
set -euo pipefail

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

script="$(cd "$(dirname "$0")" && pwd)/obfsgen.zeek"
zeek=${ZEEK:-zeek}
filter="tcp port 443"
jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
out=

while getopts "o:f:j:h" opt; do
    case $opt in
        o) out=$OPTARG ;;
        f) filter=$OPTARG ;;
        j) jobs=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ -n $out && $# -gt 0 ]] || usage
command -v "$zeek" >/dev/null || { echo "zeek not found (set ZEEK=/path/to/zeek)" >&2; exit 1; }

# Collect the pcaps; two pcaps with the same name would share a log directory.
pcaps=()
for arg in "$@"; do
    if [[ -d $arg ]]; then
        while IFS= read -r -d '' f; do pcaps+=("$f"); done \
            < <(find -L "$arg" -type f \( -name '*.pcap' -o -name '*.pcapng' \) -print0 | sort -z)
    else
        pcaps+=("$arg")
    fi
done
[[ ${#pcaps[@]} -gt 0 ]] || { echo "no pcaps found in: $*" >&2; exit 1; }
dups=$(for p in "${pcaps[@]}"; do b=${p##*/}; echo "${b%.*}"; done | sort | uniq -d)
[[ -z $dups ]] || { echo "pcaps with the same name: $dups" >&2; exit 1; }

mkdir -p "$out"
out=$(cd "$out" && pwd)

# Zeek writes into OUT/.part/<name>/, which moves to OUT/<name>/ when it ends
# well (build ignores directories whose name starts with a dot).
run_one() {
    local pcap name dir tmp
    pcap="$(cd "$(dirname "$1")" && pwd)/${1##*/}"
    name=${pcap##*/}
    name=${name%.*}
    dir=$out/$name
    tmp=$out/.part/$name
    [[ -d $dir ]] && return 0
    rm -rf "$tmp"
    mkdir -p "$tmp"
    if ! (cd "$tmp" && "$zeek" -C -r "$pcap" -f "$filter" "$script") 2>"$tmp/zeek.err"; then
        echo "zeek failed on $pcap (see $tmp/zeek.err)" >&2
        return 1
    fi
    [[ -s $tmp/zeek.err ]] || rm -f "$tmp/zeek.err"
    mv "$tmp" "$dir"
    echo "$name"
}
export -f run_one
export out zeek filter script

status=0
printf '%s\0' "${pcaps[@]}" | xargs -0 -n 1 -P "$jobs" bash -c 'run_one "$1"' _ || status=$?
rmdir "$out/.part" 2>/dev/null || true
exit $status
