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

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Values are minimum per-run mean milliseconds over three runs of 25 samples;
parentheses give slowest/fastest run spread. Both flag sets are shown.

### Z, identity projection

| Shape | library -O2 | by box expansion -O2 | library native | by box expansion native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 2.2 (1.02x) | 2.0 (1.01x) | 2.4 (1.01x) | 2.7 (1.02x) |
| abab 10^6 | 1.9 (1.02x) | 2.8 (1.02x) | 2.3 (1.01x) | 3.2 (1.02x) |
| fib 10^6 | 1.7 (1.02x) | 3.5 (1.03x) | 2.2 (1.05x) | 3.6 (1.02x) |
| random2 10^6 | 5.5 (1.02x) | 6.0 (1.03x) | 6.9 (1.01x) | 7.3 (1.00x) |
| random26 10^6 | 2.1 (1.02x) | 2.4 (1.02x) | 2.6 (1.02x) | 2.8 (1.00x) |
| all-a 10^7 | 41.1 (1.01x) | 40.1 (1.02x) | 44.4 (1.00x) | 47.6 (1.01x) |
| abab 10^7 | 38.6 (1.02x) | 47.8 (1.01x) | 43.1 (1.04x) | 52.4 (1.01x) |
| fib 10^7 | 38.3 (1.02x) | 58.7 (1.01x) | 42.7 (1.01x) | 59.1 (1.01x) |
| random2 10^7 | 78.1 (1.01x) | 85.3 (1.00x) | 92.1 (1.01x) | 98.3 (1.01x) |
| random26 10^7 | 41.7 (1.03x) | 44.1 (1.02x) | 46.1 (1.05x) | 49.2 (1.02x) |

### Z, lowercase projection

| Shape | library -O2 | by box expansion -O2 | library native | by box expansion native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 6.4 (1.01x) | 6.2 (1.01x) | 5.7 (1.02x) | 6.2 (1.02x) |
| abab 10^6 | 6.1 (1.01x) | 8.8 (1.01x) | 6.0 (1.04x) | 8.3 (1.00x) |
| fib 10^6 | 5.9 (1.01x) | 10.5 (1.01x) | 6.2 (1.01x) | 10.4 (1.01x) |
| random2 10^6 | 14.6 (1.01x) | 15.2 (1.02x) | 14.7 (1.04x) | 15.5 (1.01x) |
| random26 10^6 | 6.6 (1.01x) | 6.6 (1.02x) | 6.3 (1.03x) | 6.9 (1.01x) |
| all-a 10^7 | 85.4 (1.01x) | 82.9 (1.02x) | 79.3 (1.00x) | 82.8 (1.03x) |
| abab 10^7 | 82.2 (1.01x) | 110 (1.01x) | 81.8 (1.01x) | 105 (1.00x) |
| fib 10^7 | 82.2 (1.01x) | 131 (1.01x) | 84.1 (1.01x) | 128 (1.01x) |
| random2 10^7 | 169 (1.01x) | 187 (1.01x) | 167 (1.02x) | 191 (1.01x) |
| random26 10^7 | 88.7 (1.01x) | 89.6 (1.00x) | 87.3 (1.02x) | 91.3 (1.02x) |

### Borders, lowercase projection

| Shape | library -O2 | by repeated projection -O2 | library native | by repeated projection native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 4.5 (1.00x) | 4.7 (1.01x) | 4.9 (1.00x) | 4.9 (1.00x) |
| abab 10^6 | 4.5 (1.06x) | 4.7 (1.01x) | 4.9 (1.01x) | 4.9 (1.01x) |
| fib 10^6 | 4.4 (1.00x) | 4.7 (1.02x) | 4.9 (1.03x) | 4.9 (1.03x) |
| random2 10^6 | 13.9 (1.01x) | 13.9 (1.02x) | 13.8 (1.01x) | 13.9 (1.04x) |
| random26 10^6 | 7.5 (1.00x) | 6.5 (1.01x) | 7.4 (1.01x) | 7.4 (1.01x) |
| all-a 10^7 | 65.1 (1.01x) | 67.9 (1.01x) | 69.9 (1.02x) | 70.1 (1.01x) |
| abab 10^7 | 65.5 (1.00x) | 68.2 (1.01x) | 69.1 (1.01x) | 69.0 (1.01x) |
| fib 10^7 | 65.3 (1.01x) | 68.7 (1.01x) | 70.0 (1.01x) | 70.5 (1.01x) |
| random2 10^7 | 156 (1.00x) | 156 (1.02x) | 159 (1.00x) | 158 (1.01x) |
| random26 10^7 | 95.7 (1.03x) | 85.2 (1.05x) | 95.2 (1.01x) | 95.5 (1.00x) |


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
