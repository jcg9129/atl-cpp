#!/usr/bin/env bash
# Runs one benchmark several times and records the result in a results directory outside the repository.
#
#   benchmarks/record.sh <area>-<name> [runs]        e.g. ds-segment_tree 3
#
# Builds target benchmarks-<area>-<name> in the build tree, runs the binary
# <runs> times (default 3; the box is noisy, so each row reports the MINIMUM of
# the per-run means and the max/min spread), and prepends a dated record to
# $ATL_BENCH_RESULTS/<area>-<name>.md. See docs/conventions.md, "Benchmarks".
#
# Environment:
#   ATL_BENCH_RESULTS directory the records go to (required). It must exist and
#                   lie outside this repository: records are kept apart from
#                   the published sources.
#   ATL_BENCH_FLAGS extra compiler flags, e.g. "-march=native". When set,
#                   the script configures the build tree with exactly these
#                   flags; each flag set gets its own tree, so they never
#                   share object files.
#   ATL_BENCH_TREE  build tree to use. Default: build, or with
#                   ATL_BENCH_FLAGS, build/bench/<flags>. Without
#                   ATL_BENCH_FLAGS the tree is used as it is configured.
#   ATL_BENCH_NOTE  one line of free text stored with the record
#   ATL_BENCH_TIMEOUT seconds one run may take before it counts as a failure.
#                   Default: 1800
#   ATL_BENCH_SAMPLES Catch2 samples per row. Default: Catch2's own (100); a
#                   row is already the minimum over <runs> runs, so 25 is plenty
#   ATL_BENCH_FILTER  a Catch2 test spec (e.g. one TEST_CASE name, quoted) to run
#                   only the matching test cases (a comma separates specs, so
#                   names must not contain one); the record states it
#
# The record states the flags the benchmark was actually compiled with, read
# back from the build tree.
set -euo pipefail

name="${1:?usage: benchmarks/record.sh <area>-<name> [runs]}"
runs="${2:-3}"
root="$(cd "$(dirname "$0")/.." && pwd)"
flags="${ATL_BENCH_FLAGS-}"
if [ -n "${ATL_BENCH_FLAGS+set}" ]; then
	slug="$(printf '%s' "$flags" | tr -c 'A-Za-z0-9=' '_' | sed 's/^_*//; s/_*$//')"
	tree="${ATL_BENCH_TREE:-$root/build/bench/${slug:-default}}"
else
	tree="${ATL_BENCH_TREE:-$root/build}"
fi
binary="$tree/benchmarks/$name"
source_file="benchmarks/${name/-//}.cpp"
results="${ATL_BENCH_RESULTS:?set ATL_BENCH_RESULTS to the results directory (outside this repository)}"
[ -d "$results" ] || { echo "ATL_BENCH_RESULTS=$results is not a directory" >&2; exit 1; }
results="$(cd "$results" && pwd -P)"
case "$results/" in
"$(cd "$root" && pwd -P)/"*) echo "ATL_BENCH_RESULTS=$results is inside the repository; refusing" >&2; exit 1 ;;
esac
record="$results/$name.md"

[ -f "$root/$source_file" ] || { echo "no such benchmark source: $source_file" >&2; exit 1; }
cache() { sed -n "s/^$1:[A-Z]*=//p" "$tree/CMakeCache.txt" 2>/dev/null | head -1; }
if [ -n "${ATL_BENCH_FLAGS+set}" ]; then
	# configure every time (cheap): a new benchmark source needs its target before
	# the build runs, and the flags must match this tree
	cmake -S "$root" -B "$tree" -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_CXX_FLAGS=$flags" >/dev/null
fi
[ -f "$tree/CMakeCache.txt" ] || { echo "not a configured build tree: $tree" >&2; exit 1; }
cmake --build "$tree" --target "benchmarks-$name" -j4 >/dev/null
build_type="$(cache CMAKE_BUILD_TYPE)"
asserts="off"
case "$(cache ATL_BENCHMARK_ASSERTS)" in ON | on | 1 | TRUE | true) asserts="on" ;; esac
compiler="$("$(cache CMAKE_CXX_COMPILER)" --version | head -1)"
# the flags this target was compiled with, minus warnings, include paths, macros and
# output/dependency switches; compile_commands.json works for every generator
compiled_with="$(python3 -I - "$tree/compile_commands.json" "$(cd "$root" && pwd -P)/$source_file" <<'PY'
import json, shlex, sys
from pathlib import Path
target = Path(sys.argv[2]).resolve()
for entry in json.load(open(sys.argv[1])):
    if Path(entry["directory"], entry["file"]).resolve() != target:
        continue
    args = entry.get("arguments") or shlex.split(entry["command"])
    kept, skip = [], False
    for arg in args[1:]:
        if skip:
            skip = False
        elif arg in ("-o", "-c", "-MF", "-MT", "-MQ", "-isystem", "-I", "-D"):
            skip = True
        elif arg == str(target) or arg == entry["file"] or arg in ("-MD", "-MMD", "-pedantic"):
            pass
        elif arg == "-DNDEBUG":
            kept.append(arg)  # whether assert() is compiled out belongs in the record
        elif not arg.startswith(("-W", "-I", "-D", "-isystem")):
            kept.append(arg)
    print(" ".join(kept))
    break
PY
)"
[ -n "$compiled_with" ] || { echo "no compile command for $source_file in $tree/compile_commands.json" >&2; exit 1; }
cpu="$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
case "$cpu" in "" | -) cpu="model not reported" ;; esac
commit="$(git -C "$root" rev-parse --short HEAD)"
git -C "$root" diff --quiet HEAD -- include experimental benchmarks || commit="$commit + uncommitted changes"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for run in $(seq "$runs"); do
	# a hang is a failure, never a wait
	timeout "${ATL_BENCH_TIMEOUT:-1800}" "$binary" ${ATL_BENCH_FILTER:+"$ATL_BENCH_FILTER"} --reporter xml ${ATL_BENCH_SAMPLES:+--benchmark-samples "$ATL_BENCH_SAMPLES"} >"$work/run$run.xml"
	# one "name<TAB>mean in ns" line per benchmark row
	awk -v run="$run" '
		/<BenchmarkResults / {
			name = $0; sub(/.*<BenchmarkResults name="/, "", name); sub(/".*/, "", name)
			gsub(/&quot;/, "\"", name); gsub(/&lt;/, "<", name); gsub(/&gt;/, ">", name); gsub(/&amp;/, "\\&", name)
			if (seen[name]++) {
				printf "duplicate benchmark row in run %s: %s\n", run, name > "/dev/stderr"
				exit 1
			}
		}
		/<mean value=/ && name != "" {
			mean = $0; sub(/.*<mean value="/, "", mean); sub(/".*/, "", mean)
			printf "%s\t%s\n", name, mean; name = ""
		}' "$work/run$run.xml" >>"$work/rows.tsv"
done

{
	echo "## $(date +%Y-%m-%d) — commit $commit"
	echo
	echo "- machine: $(uname -m), $(uname -sr), CPU $cpu"
	echo "- compiler: $compiler, $build_type, asserts $asserts"
	echo "- flags: \`$compiled_with\`"
	echo "- runs: $runs${ATL_BENCH_SAMPLES:+ of $ATL_BENCH_SAMPLES samples per row}; each row is the minimum of the per-run means, spread = slowest run / fastest run"
	[ -n "${ATL_BENCH_NOTE:-}" ] && echo "- note: $ATL_BENCH_NOTE"
	[ -n "${ATL_BENCH_FILTER:-}" ] && echo "- filter: only the test cases matching \`$ATL_BENCH_FILTER\`"
	echo
	echo "| row | min ms | spread |"
	echo "|---|---|---|"
	awk -F'\t' '
		!($1 in low) { order[++count] = $1; low[$1] = $2 + 0; high[$1] = $2 + 0 }
		{ if ($2 + 0 < low[$1]) low[$1] = $2 + 0; if ($2 + 0 > high[$1]) high[$1] = $2 + 0 }
		END {
			for (i = 1; i <= count; ++i) {
				row = order[i]; ms = low[row] / 1e6; spread = high[row] / low[row]
				# a "|" inside a row name (Catch2 sections like "star | unit hops") would split the table cell
				label = ""
				for (k = 1; k <= length(row); ++k) {
					c = substr(row, k, 1)
					label = label (c == "|" ? "\\|" : c)
				}
				format = ms >= 100 ? "%.0f" : ms >= 1 ? "%.1f" : "%.3f"
				# a bare ">" in a printf argument list is an output redirection
				flag = (spread > 1.5) ? " (unstable)" : ""
				printf("| %s | " format " | %.2fx%s |\n", label, ms, spread, flag)
			}
		}' "$work/rows.tsv"
	echo
} >"$work/section.md"

if [ -f "$record" ]; then
	# keep the three header lines, then the new record, then the older ones
	{ head -n 4 "$record"; cat "$work/section.md"; tail -n +5 "$record"; } >"$work/record.md"
else
	{
		echo "# $name"
		echo
		echo "Source: \`$source_file\`. Newest record first; written by \`benchmarks/record.sh\`."
		echo
		cat "$work/section.md"
	} >"$work/record.md"
fi
cp "$work/record.md" "$record"
echo "recorded $(grep -c . "$work/rows.tsv") measurements in $record"
cat "$work/section.md"
