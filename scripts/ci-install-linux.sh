#!/usr/bin/env bash
# CI only. Never invoked by the local verification procedure.
set -euo pipefail
# Runner networking fails transiently (e.g. a truncated Launchpad response in
# add-apt-repository); retry network steps instead of failing the job.
retry() {
    local attempt
    for attempt in 1 2 3; do
        "$@" && return 0
        echo "attempt $attempt failed: $*" >&2
        sleep $((attempt * 15))
    done
    return 1
}
retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
retry sudo apt-get update
retry sudo apt-get install -y gcc-15 g++-15 ninja-build ccache
case "${1:-gcc}" in
    clang|tools)
        retry curl -fsSL --retry 3 -o /tmp/atl-llvm.key https://apt.llvm.org/llvm-snapshot.gpg.key
        sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/atl-llvm.gpg /tmp/atl-llvm.key
        echo 'deb [signed-by=/usr/share/keyrings/atl-llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-22 main' |
            sudo tee /etc/apt/sources.list.d/atl-llvm.list
        retry sudo apt-get update
        retry sudo apt-get install -y clang-22 clang-format-22 clang-tidy-22
        ;;
    linux32)
        retry sudo apt-get install -y gcc-multilib gcc-15-multilib g++-15-multilib
        ;;
esac
