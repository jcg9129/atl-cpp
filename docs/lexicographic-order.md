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

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean milliseconds over three runs of 25 samples.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| char random26 2e5 | library (MSD radix) | 13.3 | 13.1 |
| char random26 2e5 | by stable sort, bulk append | 44.4 | 44.9 |
| char random26 2e5 | by stable sort, per-symbol append | 47.5 | 47.1 |
| char random26 2e5 | by stable sort, ranges compare | 46.4 | 45.9 |
| char random26 2e5 | by unsigned bytes | 51.0 | 51.3 |
| char random26 2e5 | by LCP merge | 48.1 | 48.9 |
| char random2 2e5 | library (MSD radix) | 22.2 | 22.2 |
| char random2 2e5 | by stable sort, bulk append | 64.8 | 65.4 |
| char random2 2e5 | by stable sort, per-symbol append | 67.5 | 67.3 |
| char random2 2e5 | by stable sort, ranges compare | 69.5 | 69.4 |
| char random2 2e5 | by unsigned bytes | 49.0 | 49.4 |
| char random2 2e5 | by LCP merge | 50.8 | 49.5 |
| char duplicates 2e5 | library (MSD radix) | 9.4 | 9.3 |
| char duplicates 2e5 | by stable sort, bulk append | 41.5 | 41.8 |
| char duplicates 2e5 | by stable sort, per-symbol append | 45.0 | 44.2 |
| char duplicates 2e5 | by stable sort, ranges compare | 45.3 | 44.4 |
| char duplicates 2e5 | by unsigned bytes | 41.7 | 41.5 |
| char duplicates 2e5 | by LCP merge | 36.0 | 36.1 |
| char prefix500 2e4 | library (MSD radix) | 11.7 | 12.0 |
| char prefix500 2e4 | by stable sort, bulk append | 82.9 | 81.7 |
| char prefix500 2e4 | by stable sort, per-symbol append | 105 | 98.9 |
| char prefix500 2e4 | by stable sort, ranges compare | 123 | 122 |
| char prefix500 2e4 | by unsigned bytes | 18.3 | 18.2 |
| char prefix500 2e4 | by LCP merge | 16.8 | 17.0 |
| int random26 2e5 | library (stable sort) | 51.7 | 51.8 |
| int random26 2e5 | by stable sort, per-symbol append | 53.9 | 53.9 |
| int random26 2e5 | by MSD partition | 76.3 | 76.4 |
| int random26 2e5 | by LCP merge | 58.1 | 58.2 |
| uint64 random26 2e5 | library (stable sort) | 63.3 | 63.0 |
| uint64 random26 2e5 | by stable sort, per-symbol append | 63.4 | 62.2 |
| uint64 random26 2e5 | by MSD partition | 92.5 | 90.6 |
| uint64 random26 2e5 | by LCP merge | 74.2 | 75.6 |
| int duplicates 2e5 | library (stable sort) | 49.4 | 49.9 |
| int duplicates 2e5 | by stable sort, per-symbol append | 50.5 | 50.5 |
| int duplicates 2e5 | by MSD partition | 63.5 | 63.8 |
| int duplicates 2e5 | by LCP merge | 39.6 | 40.5 |
| int prefix500 2e4 | library (stable sort) | 155 | 155 |
| int prefix500 2e4 | by stable sort, per-symbol append | 160 | 160 |
| int prefix500 2e4 | by MSD partition | 67.7 | 66.4 |
| int prefix500 2e4 | by LCP merge | 72.1 | 71.9 |
| uint64 prefix500 2e4 | library (stable sort) | 243 | 242 |
| uint64 prefix500 2e4 | by stable sort, per-symbol append | 251 | 251 |
| uint64 prefix500 2e4 | by MSD partition | 122 | 121 |
| uint64 prefix500 2e4 | by LCP merge | 127 | 127 |


MSD radix wins all byte shapes. The wide-symbol contenders lose random words
but can win duplicate or common-prefix shapes, so general wide symbols retain
stable sort. The shared reserve policy has no separate lexicographic-order
measurement yet; forward ranges reserve total symbols only for trivially copyable
projected types.

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
