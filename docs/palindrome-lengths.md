# `atl::palindrome_lengths`

Build a static oracle for the longest palindrome at each center and for
constant-time substring palindrome checks. Use it for per-center geometry;
use a palindromic tree for distinct palindromes or online updates.

```cpp
#include <atl/strings/palindrome_lengths.hpp>
```

\include palindrome-lengths/quick.cpp

## API

```cpp
palindrome_lengths() = default;
template<typename R, typename Proj = std::identity>
    requires comparable_text<R, Proj>
constexpr explicit palindrome_lengths(R &&text, Proj proj = {});
[[nodiscard]] constexpr uint32 odd_length(uint32 center) const;
[[nodiscard]] constexpr uint32 even_length(uint32 gap) const;
[[nodiscard]] constexpr bool is_palindrome(uint32 l, uint32 r) const;
constexpr std::size_t size() const;
constexpr bool empty() const;
```

`comparable_text<R, Proj>` is a sized random-access range whose projected
symbols are equality-comparable. The constructor reads text in place via
proj; it retains neither text nor projection. All character types, integers,
equality-only tokens, reference wrappers, capturing lambdas and working
proxy ranges are accepted. There is no alphabet bound or character indexing.

| Operation | Result |
|---|---|
| default construction | Empty text. |
| `odd_length(center)` | Longest odd palindrome centered at a symbol, at least one. |
| `even_length(gap)` | Longest even palindrome between gap - 1 and gap, possibly zero. |
| `is_palindrome(l, r)` | Whether `[l, r)` is palindromic, including empty intervals. |
| `size()` / `empty()` | Text size / whether it is zero. |

## Complexity

Build takes O(n) comparisons and projection calls and O(n) storage: one
uint32 length at each of 2n + 1 virtual centers for nonempty text, about
8 bytes per symbol. Queries take O(1). The implementation is iterative,
with O(1) auxiliary stack space. Empty text uses an empty array.

## Preconditions

Asserted: n < 2^31; center < size(); 1 <= gap < size(); and
l <= r <= size(). Storage must also fit the vector's platform-dependent
max_size(). End gaps zero and n are not accepted by even_length.

Documented only: projection results are stable and equality is an equivalence
relation. Non-contiguous random-access ranges and string_view work; a
single-pass range does not.

## Measured performance

`benchmarks/strings/palindrome_lengths.cpp` measures full radius construction.
The library expands virtual positions; by parity step expands real pairs.
The library wins on repetitive shapes, while random26 favors parity steps.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean ms over three runs of 25 samples; parentheses are
slowest/fastest spread.

| Shape | library -O2 | by parity step -O2 | library native | by parity step native |
|---|---:|---:|---:|---:|
| all-a 10^6 | 10.9 (1.00x) | 12.1 (1.00x) | 10.9 (1.00x) | 12.2 (1.00x) |
| abab 10^6 | 10.8 (1.00x) | 12.1 (1.00x) | 10.8 (1.01x) | 12.0 (1.01x) |
| fib 10^6 | 11.5 (1.01x) | 12.7 (1.01x) | 11.6 (1.01x) | 12.7 (1.01x) |
| random2 10^6 | 17.1 (1.01x) | 17.6 (1.00x) | 16.9 (1.01x) | 17.6 (1.01x) |
| random26 10^6 | 6.4 (1.02x) | 4.7 (1.01x) | 5.8 (1.01x) | 4.7 (1.01x) |
| all-a 10^7 | 145 (1.00x) | 158 (1.00x) | 145 (1.01x) | 159 (1.00x) |
| abab 10^7 | 144 (1.00x) | 157 (1.00x) | 144 (1.01x) | 157 (1.00x) |
| fib 10^7 | 153 (1.00x) | 164 (1.00x) | 154 (1.00x) | 164 (1.00x) |
| random2 10^7 | 207 (1.01x) | 212 (1.00x) | 204 (1.00x) | 212 (1.00x) |
| random26 10^7 | 100 (1.01x) | 83.3 (1.01x) | 94.7 (1.00x) | 83.5 (1.00x) |

## Pitfalls and notes

Radii here are real palindrome lengths, not half-lengths. To count positional
palindromes, sum `(odd_length(c) + 1) / 2` over symbols and
`even_length(g) / 2` over interior gaps, using a wide accumulator. To find
the longest palindrome, take the maximum over both kinds of center.
A lowercase projection must pass unsigned-char values to std::tolower.

## How it works

Manacher's algorithm operates on virtual `#s0#s1#...#` coordinates. A length
entry is directly the real palindrome length. Inside the rightmost known
palindrome, reuse its mirrored radius, then expand beyond the certified
region. Separators always compare equal; real pairs invoke the projection.
Expansion boundaries advance only O(n) times in total. Subtraction checks
the remaining right-hand extent without overflowing virtual-coordinate sums.
The virtual center of `[l, r)` is l + r.
Reference: Manacher (1975), “A new linear-time on-line algorithm for finding
the smallest initial palindrome of a string”.

## See also

[Palindromic tree](palindromic-tree.md), [String matching](matching.md).
