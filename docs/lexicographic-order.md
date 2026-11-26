# `atl::lexicographic_order`

Return stable lexicographic word indices without rearranging the input. Use it
for dictionary ordering with byte, wide-character, integer or token symbols.

```cpp
#include <atl/strings/lexicographic_order.hpp>
#include <atl/strings/comparable_text.hpp>
```

\include lexicographic-order/quick.cpp

## API

```cpp
template<typename R, typename Proj = std::identity>
    requires sortable_word_range<R, Proj>
[[nodiscard]] std::vector<uint32> lexicographic_order(R &&words, Proj proj = {});
```

`words` is an input range of sized random-access words. `proj` is invoked on
word element references once per symbol. The result is a permutation of input
word indices, stable among equal projected words. No references are retained.
The operation is runtime-only because it uses `std::ranges::stable_sort`.

The shared vocabulary in `comparable_text.hpp` is:

| Entity | Definition / requirement |
|---|---|
| `text_symbol_t<R, Proj = std::identity>` | `remove_cvref_t<invoke_result_t<Proj &, range_reference_t<R>>>` |
| `word_symbol_t<R, Proj = std::identity>` | `text_symbol_t<range_reference_t<R>, Proj>` |
| `comparable_text<R, Proj = std::identity>` | Sized random-access range, invocable projection, equality-comparable projected symbols. |
| `sortable_text<R, Proj = std::identity>` | Refines `comparable_text`; projected symbols are totally ordered. |
| `sortable_word_range<R, Proj = std::identity>` | Input range whose references are `sortable_text` under the projection. |
| `length_range<R>` | Sized random-access range with value type `uint32`. |

```cpp
struct index_interval {
    uint32 begin;
    uint32 end;
    [[nodiscard]] constexpr bool operator==(index_interval const &) const = default;
};
```

`index_interval` represents [begin, end), with componentwise equality.
The projection may be a capturing lambda, a mutable callable or a reference wrapper.

## Complexity

Let L be total symbols and n word count. Materialization takes O(L+n) time
and storage. Bytes use an iterative MSD radix partition, O(L+n) time for fixed
257-bucket passes and bounded-size comparison sorts. Other symbols use a stable
sort with O(L log n) symbol comparisons and O(n) sorting scratch when its buffer
allocation succeeds; the standard library's low-memory fallback can perform
O(n log² n) word comparisons. Random-access words need not be contiguous; the
outer collection can be single-pass. Explicit heap stacks avoid recursion per
character, including long common prefixes.

## Preconditions

Asserted under `ATL_ASSERT`: fewer than 2^32 words; checked size sums and
appends fit `size_t` and vector capacity limits. Documented only: projection is
stable and ordering is consistent with equality. Comparisons follow the projected
type's `<`, so plain `char` follows platform signedness rather than
`std::string::operator<` byte semantics; project to `unsigned char` for that order.
Wide characters and integers are compared as values. Proxy ranges work when
the projection produces usable ordered symbols. Projected symbols need not be
default constructible or assignable (they are emplaced into the buffer). The
collection and its words are iterated through `std::ranges::begin` and
`std::ranges::end`. Allocation failure propagates as
a standard exception. The input is read during the call; the result owns its indices.

## Measured performance

[The benchmark](../benchmarks/strings/lexicographic_order.cpp) measures full
materialize-and-sort throughput. By stable sort varies bulk versus per-symbol
append and comparison spelling; by unsigned bytes flips signed bytes for memcmp;
by MSD partition stably sorts wide-symbol slices; by LCP merge retains run LCPs.
The latter two live in
`experimental/atl/experimental/strings/lexicographic_order_variants.hpp`;
the tests use a stable comparison sort as the certifying oracle.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| char random26 2e5 | library (MSD radix) | 9.5 | 10.3 |
| char random26 2e5 | by stable sort, bulk append | 34.7 | 36.1 |
| char random26 2e5 | by stable sort, per-symbol append | 35.6 | 36.1 |
| char random26 2e5 | by stable sort, ranges compare | 36.3 | 36.4 |
| char random26 2e5 | by unsigned bytes | 43.1 | 43.6 |
| char random26 2e5 | by LCP merge | 37 | 40.4 |
| char random2 2e5 | library (MSD radix) | 15.3 | 17.1 |
| char random2 2e5 | by stable sort, bulk append | 48 | 50 |
| char random2 2e5 | by stable sort, per-symbol append | 47.2 | 49.7 |
| char random2 2e5 | by stable sort, ranges compare | 51.8 | 56.3 |
| char random2 2e5 | by unsigned bytes | 42.5 | 42.3 |
| char random2 2e5 | by LCP merge | 39.3 | 41.8 |
| char duplicates 2e5 | library (MSD radix) | 5.8 | 7 |
| char duplicates 2e5 | by stable sort, bulk append | 32.2 | 34.1 |
| char duplicates 2e5 | by stable sort, per-symbol append | 32.7 | 34.2 |
| char duplicates 2e5 | by stable sort, ranges compare | 34.8 | 38.3 |
| char duplicates 2e5 | by unsigned bytes | 35.7 | 36.2 |
| char duplicates 2e5 | by LCP merge | 27 | 29.3 |
| char prefix500 2e4 | library (MSD radix) | 4.8 | 4.8 |
| char prefix500 2e4 | by stable sort, bulk append | 78.9 | 86 |
| char prefix500 2e4 | by stable sort, per-symbol append | 73.8 | 89.1 |
| char prefix500 2e4 | by stable sort, ranges compare | 95.2 | 110 |
| char prefix500 2e4 | by unsigned bytes | 12.1 | 14.4 |
| char prefix500 2e4 | by LCP merge | 12 | 13.7 |
| int random26 2e5 | library (stable sort) | 37.2 | 39.8 |
| int random26 2e5 | by stable sort, per-symbol append | 39.7 | 41.5 |
| int random26 2e5 | by MSD partition | 52.4 | 58.7 |
| int random26 2e5 | by LCP merge | 42.1 | 44.8 |
| uint64 random26 2e5 | library (stable sort) | 43.9 | 45.6 |
| uint64 random26 2e5 | by stable sort, per-symbol append | 44.2 | 46.5 |
| uint64 random26 2e5 | by MSD partition | 58.6 | 67.8 |
| uint64 random26 2e5 | by LCP merge | 48.4 | 58.1 |
| int duplicates 2e5 | library (stable sort) | 35.2 | 38.1 |
| int duplicates 2e5 | by stable sort, per-symbol append | 36.8 | 39.7 |
| int duplicates 2e5 | by MSD partition | 41.4 | 49.3 |
| int duplicates 2e5 | by LCP merge | 29 | 30.3 |
| int prefix500 2e4 | library (stable sort) | 103 | 118 |
| int prefix500 2e4 | by stable sort, per-symbol append | 127 | 142 |
| int prefix500 2e4 | by MSD partition | 63.1 | 68.7 |
| int prefix500 2e4 | by LCP merge | 69.6 | 73.7 |
| uint64 prefix500 2e4 | library (stable sort) | 185 | 201 |
| uint64 prefix500 2e4 | by stable sort, per-symbol append | 257 | 258 |
| uint64 prefix500 2e4 | by MSD partition | 118 | 131 |
| uint64 prefix500 2e4 | by LCP merge | 128 | 141 |


MSD radix wins all byte shapes. The wide-symbol contenders lose random words. The library loses int
duplicates to LCP merge (35.2/38.1 versus 29/30.3 ms), and prefix500
to MSD partition for both int (103/118 versus 63.1/68.7 ms) and uint64
(185/201 versus 118/131 ms). No wide-symbol method wins every shape. The shared reserve policy has no separate lexicographic-order
measurement yet; forward ranges reserve total symbols only for trivially copyable
projected types.

Wide-symbol duplicate and uint64 random-word rows include contenders that
vary between runs (spreads up to 1.80x); these minima alone do not establish
stable rankings.

## Pitfalls and notes

Equal words keep their input order, including empty words. A dictionary
constructor still requires you to feed words in the computed order; it does not
consume an argsort directly. Token comparison costs are additional to the symbol
comparison counts above.

## How it works

One flat symbol buffer and end-offset array make comparisons contiguous.
Byte slices skip a shared prefix, then stably distribute into 257 buckets: ended
words first, then the next byte with a signed-byte sign-bit flip when needed.
Slices under 32 words use stable comparison sorting from the known prefix depth.
Other symbol types compare whole materialized words in a stable sort. Pending
radix slices live on a vector rather than the call stack.

## See also

[Sorted dictionary](sorted-dictionary.md), [Suffix array](suffix-array.md).
