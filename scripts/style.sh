#!/usr/bin/env bash
# NUL-delimited tracked paths only; tolerate staged deletions and partial trees.
set -euo pipefail
mode=${1:?expected fmt, fmt-fix, tidy, or tidy-proof}
format=${CLANG_FORMAT:-clang-format-22}
tidy=${CLANG_TIDY:-clang-tidy-22}
case "$mode" in
    fmt|fmt-fix)
        args=(--dry-run --Werror)
        if [[ $mode == fmt-fix ]]; then args=(-i); fi
        git ls-files -z -- include tests benchmarks experimental |
            while IFS= read -r -d '' source; do
                case "$source" in
                    *.hpp|*.cpp) if [[ -f $source ]]; then "$format" "${args[@]}" "$source"; fi ;;
                esac
            done
        ;;
    tidy|tidy-proof)
        scratch=$(mktemp -d)
        trap 'rm -rf -- "$scratch"' EXIT
        gcc_runtime=$(g++-15 -print-libgcc-file-name)
        compiler_args=(-std=c++23 "--gcc-install-dir=${gcc_runtime%/*}")
        if [[ $mode == tidy ]]; then
            git ls-files -z -- include |
                while IFS= read -r -d '' header; do
                    if [[ $header == *.hpp && -f $header ]]; then
                        printf '#include <%s>\n' "${header#include/}"
                    fi
                done > "$scratch/check.cpp"
            printf 'int main() { return 0; }\n' >> "$scratch/check.cpp"
            "$tidy" --config-file=.clang-tidy "$scratch/check.cpp" -- "${compiler_args[@]}" -I"$PWD/include"
        else
            # Put the violation in a header to prove the filter works for both
            # relative and absolute include paths, as on a CI checkout.
            mkdir -p "$scratch/include/atl"
            printf 'inline int DeliberateCamelCase() { return 0; }\n' > "$scratch/include/atl/violation.hpp"
            config="$PWD/.clang-tidy"
            for path_mode in relative absolute; do
                if [[ $path_mode == relative ]]; then
                    printf '#include <atl/violation.hpp>\n' > "$scratch/proof.cpp"
                    diagnostic_path='include/atl/violation.hpp'
                else
                    printf '#include "%s/include/atl/violation.hpp"\n' "$scratch" > "$scratch/proof.cpp"
                    diagnostic_path="$scratch/include/atl/violation.hpp"
                fi
                if (cd "$scratch" && "$tidy" --config-file="$config" proof.cpp -- "${compiler_args[@]}" -Iinclude) > "$scratch/tidy.log" 2>&1; then
                    cat "$scratch/tidy.log"
                    echo 'Expected naming check to fail' >&2
                    exit 1
                fi
                cat "$scratch/tidy.log"
                grep -Eq 'error:.*invalid case style.*\[readability-identifier-naming,-warnings-as-errors\]' "$scratch/tidy.log"
                awk -v prefix="$diagnostic_path:" 'index($0, prefix) == 1 { found = 1 } END { exit !found }' "$scratch/tidy.log"
            done
        fi
        ;;
    *) echo "Unknown style task: $mode" >&2; exit 2 ;;
esac
