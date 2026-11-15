#!/usr/bin/env bash
set -euo pipefail
preset=${1:-gcc}
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
if command -v cygpath >/dev/null 2>&1; then
    scratch=$(cygpath -m "$scratch")
fi
cmake --install "build/$preset" --prefix "$scratch/install"
# Read the configured compiler and flags so architecture and standard library
# choices also reach the external consumer. No build-tree include paths leak in.
compiler=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "build/$preset/CMakeCache.txt")
flags=$(sed -n 's/^CMAKE_CXX_FLAGS:STRING=//p' "build/$preset/CMakeCache.txt")
cmake -S tests/consumer -B "$scratch/consumer" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler" \
    -DCMAKE_CXX_FLAGS="$flags" -DCMAKE_PREFIX_PATH="$scratch/install"
cmake --build "$scratch/consumer" --parallel "${JOBS:-2}"
ctest --test-dir "$scratch/consumer" --output-on-failure --no-tests=error
