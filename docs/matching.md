# `atl::border_lengths`, `atl::common_prefix_lengths`, `atl::occurrences` — string matching

Compute borders, prefix matches and all occurrences with equality comparisons.
Use these linear algorithms for one pattern or static prefix geometry; all
symbols may be tokens rather than characters.

```cpp
#include <atl/strings/matching.hpp>
#include <atl/strings/comparable_text.hpp>
```

\include matching/quick.cpp

## API

`R` must satisfy `comparable_text<R, Proj>`: a sized random-access range
with equality-comparable projected symbols. `Proj` is invoked on range
references, so reference wrappers and capturing mutable lambdas work.
`length_range<R>` requires a sized random-access range of `uint32` values.

```cpp
template<typename R, typename Proj = std::identity>
    requires comparable_text<R, Proj>
[[nodiscard]] constexpr std::vector<uint32> border_lengths(R &&text, Proj proj = {});
template<typename R, typename Proj = std::identity>
    requires comparable_text<R, Proj>
[[nodiscard]] constexpr std::vector<uint32> common_prefix_lengths(R &&text, Proj proj = {});
template<typename R> requires length_range<R>
[[nodiscard]] constexpr std::vector<uint32> to_border_lengths(R &&common_prefix);
template<typename R> requires length_range<R>
[[nodiscard]] constexpr std::vector<uint32> to_common_prefix_lengths(R &&border);
template<typename Text, typename Pattern, typename Borders, typename Proj = std::identity>
    requires comparable_text<Text, Proj> && comparable_text<Pattern, Proj> && length_range<Borders>
        && std::equality_comparable_with<text_symbol_t<Text, Proj>, text_symbol_t<Pattern, Proj>>
[[nodiscard]] constexpr std::vector<uint32> occurrences(Text &&text, Pattern &&pattern, Borders &&borders, Proj proj = {});
```

| Operation | Result |
|---|---|
| `border_lengths(text, proj)` | Entry i is the longest proper border of `text[0..i]`; entry zero is zero. |
| `common_prefix_lengths(text, proj)` | Entry i is the common-prefix length of text and its suffix at i; entry zero is n. |
| `to_border_lengths(common_prefix)` | Corresponding border array, without needing text. |
| `to_common_prefix_lengths(border)` | Corresponding Z-array, without needing text. |
| `occurrences(text, pattern, borders, proj)` | Ascending start positions, including overlaps. The empty pattern matches nowhere. |

## Complexity

All operations take O(n) time, or O(text size + pattern size) for matching.
Projections and comparisons are invoked O(n) times. Results occupy O(n), or
O(matches), storage. Conversion to Z additionally builds an O(n) canonical
text; other operations use O(1) auxiliary storage. All passes are iterative.
No input references are retained.

## Preconditions

Asserted under `ATL_ASSERT`: text, pattern and conversion-array sizes are
below 2^32; the border array has the pattern's size; positive Z entries at
i > 0 are at most n - i; border entry i is at most i.

Documented only: conversion arrays must be valid for some text, and matching
borders must be exactly `border_lengths(pattern, proj)`. Bounds checks do
not validate these structural laws. The projection is stable and equality
is an equivalence relation. Random-access non-contiguous ranges and
`std::string_view` work; linked lists and single-pass ranges do not.
All character types, signed integers and equality-comparable proxy symbols
are compared as values, never used as alphabet indices. Text and pattern symbols
of different integral signedness are compared by mathematical value, so a
negative symbol equals no unsigned one.

## Measured performance

`benchmarks/strings/matching.cpp` measures full-array construction throughput.
The library Z loop skips a comparison known to fail inside the active box;
by box expansion still performs it. Both border loops use repeated projection.
The lowercase projection uses `std::tolower` on an unsigned-char value.

### Z, identity projection

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library -O2 | by box expansion -O2 | library native | by box expansion native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 1.1 (1.02x) | 1.2 (1.03x) | 1.2 (1.08x) | 1.2 (1.09x) |
| abab 10^6 | 1.3 (1.07x) | 2.7 (1.02x) | 1.4 (1.04x) | 2.8 (1.03x) |
| fib 10^6 | 0.966 (1.21x) | 2.8 (1.02x) | 0.911 (1.13x) | 2.9 (1.04x) |
| random2 10^6 | 4.7 (1.03x) | 4.7 (1.02x) | 4.9 (1.03x) | 5 (1.06x) |
| random26 10^6 | 1 (1.21x) | 1.1 (1.13x) | 1.1 (1.26x) | 1.2 (1.07x) |
| all-a 10^7 | 30.1 (1.06x) | 30.7 (1.05x) | 33.3 (1.01x) | 33.2 (1.02x) |
| abab 10^7 | 32.7 (1.03x) | 46.6 (1.02x) | 33.4 (1.09x) | 48.1 (1.04x) |
| fib 10^7 | 29.6 (1.07x) | 48.7 (1.01x) | 30.5 (1.02x) | 50.5 (1.05x) |
| random2 10^7 | 69 (1.02x) | 72.2 (1.01x) | 72.1 (1.00x) | 76.4 (1.03x) |
| random26 10^7 | 30.5 (1.03x) | 31.4 (1.03x) | 32.3 (1.10x) | 32.6 (1.08x) |

### Z, lowercase projection

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library -O2 | by box expansion -O2 | library native | by box expansion native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 3.5 (1.02x) | 3.9 (1.02x) | 3.8 (1.04x) | 3.9 (1.01x) |
| abab 10^6 | 4 (1.01x) | 5.5 (1.02x) | 4.2 (1.07x) | 5.6 (1.06x) |
| fib 10^6 | 3.8 (1.12x) | 6.9 (1.01x) | 4 (1.10x) | 7.1 (1.02x) |
| random2 10^6 | 11.3 (1.01x) | 11.5 (1.03x) | 11.9 (1.05x) | 12 (1.05x) |
| random26 10^6 | 4.3 (1.08x) | 4.3 (1.01x) | 4.6 (1.09x) | 4.5 (1.05x) |
| all-a 10^7 | 54.8 (1.07x) | 56.9 (1.03x) | 59.6 (1.02x) | 59.5 (1.02x) |
| abab 10^7 | 60 (1.03x) | 72.5 (1.06x) | 63.8 (1.02x) | 78.8 (1.02x) |
| fib 10^7 | 60 (1.04x) | 86.5 (1.03x) | 60.1 (1.11x) | 89.9 (1.05x) |
| random2 10^7 | 132 (1.01x) | 144 (1.01x) | 140 (1.03x) | 151 (1.02x) |
| random26 10^7 | 63.2 (1.01x) | 62.7 (1.02x) | 64.8 (1.08x) | 65.5 (1.04x) |

### Borders, lowercase projection

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library -O2 | by repeated projection -O2 | library native | by repeated projection native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 3.1 (1.04x) | 2.8 (1.14x) | 3 (1.05x) | 3.4 (1.07x) |
| abab 10^6 | 3.1 (1.05x) | 2.8 (1.09x) | 3 (1.07x) | 3.2 (1.14x) |
| fib 10^6 | 3.4 (1.01x) | 3.1 (1.10x) | 3.2 (1.14x) | 3.5 (1.07x) |
| random2 10^6 | 10.5 (1.01x) | 10.5 (1.05x) | 11 (1.06x) | 11.3 (1.01x) |
| random26 10^6 | 4 (1.04x) | 4 (1.05x) | 4.5 (1.06x) | 4.3 (1.07x) |
| all-a 10^7 | 50.1 (1.03x) | 47.9 (1.03x) | 50.9 (1.05x) | 53.2 (1.04x) |
| abab 10^7 | 50.8 (1.03x) | 48.1 (1.02x) | 51.2 (1.04x) | 53.8 (1.07x) |
| fib 10^7 | 53.9 (1.03x) | 50.9 (1.01x) | 51.8 (1.08x) | 56.7 (1.03x) |
| random2 10^7 | 125 (1.01x) | 124 (1.01x) | 131 (1.03x) | 131 (1.03x) |
| random26 10^7 | 59.8 (1.03x) | 59.4 (1.05x) | 63.6 (1.06x) | 63.5 (1.03x) |


## Pitfalls and notes

The text's borders are the chain `b[n-1], b[b[n-1]-1], ...`; for nonempty
text, the smallest period is `n - b[n-1]`. Reuse one pattern's borders for
many texts. Changing the projection requires recomputing them. A case-folding
projection must cast char to unsigned char prior to `std::tolower`.

## How it works

KMP follows border links on a mismatch. Every successful extension increases
the matched length once, so total fallback work is linear. Z construction
reuses the rightmost known prefix occurrence; an interior mirrored length
ending strictly inside that box is exact. Z-to-border conversion seeds each
occurrence's endpoint and sweeps decreasing lengths backward. The inverse
copies symbols from border predecessors and assigns fresh integer symbols
at borderless positions, then runs Z on that canonical text.

## See also

[Palindrome radii](palindrome-lengths.md), [Lyndon words](lyndon.md),
[Aho–Corasick](aho-corasick.md), [Suffix automaton](suffix-automaton.md).
