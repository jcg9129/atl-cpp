#!/usr/bin/env bash
# CI only. Never invoked by the local verification procedure.
set -euo pipefail
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-15 g++-15 ninja-build ccache
case "${1:-gcc}" in
    clang|tools)
        curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key |
            sudo gpg --dearmor -o /usr/share/keyrings/atl-llvm.gpg
        echo 'deb [signed-by=/usr/share/keyrings/atl-llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-22 main' |
            sudo tee /etc/apt/sources.list.d/atl-llvm.list
        sudo apt-get update
        sudo apt-get install -y clang-22 clang-format-22 clang-tidy-22
        ;;
    linux32)
        sudo apt-get install -y gcc-multilib gcc-15-multilib g++-15-multilib
        ;;
esac
