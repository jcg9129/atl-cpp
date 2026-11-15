#!/usr/bin/env bash
# CI-only downloads; this script is never invoked by local CMake configuration.
set -euo pipefail
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Usage: fetch-doc-tools.sh <new-scratch-directory>' >&2
    exit 1
fi
mkdir -p -- "$1"
tools_dir=$(cd -- "$1" && pwd)
curl --fail --location --retry 3 \
    https://github.com/doxygen/doxygen/releases/download/Release_1_15_0/doxygen-1.15.0.linux.bin.tar.gz \
    -o "$tools_dir/doxygen.tar.gz"
curl --fail --location --retry 3 \
    https://github.com/jothepro/doxygen-awesome-css/archive/46483f1e5a70ffb9ecd3b82d0a1cd1b24edf13da.tar.gz \
    -o "$tools_dir/theme.tar.gz"
(
    cd -- "$tools_dir"
    echo '0ec2e5b2c3cd82b7106d19cb42d8466450730b8cb7a9e85af712be38bf4523a1  doxygen.tar.gz' | sha256sum --check --strict -
    echo '65abe1d85bc2adf9d1a1b6ef90ae5e63f72425024196f89ec27b7bdc52d94267  theme.tar.gz' | sha256sum --check --strict -
)
mkdir -- "$tools_dir/doxygen" "$tools_dir/theme"
tar -xzf "$tools_dir/doxygen.tar.gz" -C "$tools_dir/doxygen" --strip-components=1
tar -xzf "$tools_dir/theme.tar.gz" -C "$tools_dir/theme" --strip-components=1
[[ $("$tools_dir/doxygen/bin/doxygen" --version) == 1.15.0 ]]
