# Types and utilities

The foundations the rest of `atl` is built on: fixed-width integer aliases, concepts and
traits that treat native and software 128-bit types as first-class integers, a few algebra helpers, and the
feature-detection macros that make optional features (`std::to_chars`,
…) degrade gracefully. Almost everything here lives in `namespace atl` (traits sit in
`namespace atl::traits`).

```cpp
#include <atl/types/alias.hpp>            // integer / float aliases
#include <atl/types/concepts.hpp>         // integral and modular-integer concepts
#include <atl/types/algebraic.hpp>        // binary_operation, action, mode tags
#include <atl/types/tags.hpp>             // sorted_unique
#include <atl/types/vertex.hpp>           // vertex_id, none, vertex_range
#include <atl/types/traits/integral.hpp>  // is_*_integral, make_signed/unsigned
#include <atl/types/traits/is_one_of.hpp> // is_one_of
#include <atl/types/traits/widen_mul.hpp> // widen_mul_type
#include <atl/types/utils.hpp>            // num_bits
#include <atl/types/forward_like.hpp>     // atl::forward_like (portable)
#include <atl/utils/assert.hpp>           // atl_assert
#include <atl/utils/prefetch.hpp>         // prefetch
#include <atl/config.hpp>                 // feature-detection macros
```

## Integer and float aliases

Fixed-width names for the built-in arithmetic types, from `<atl/types/alias.hpp>`.

| alias | underlying type |
|---|---|
| `atl::int8` / `int16` / `int32` / `int64` | `std::int8_t` … `std::int64_t` |
| `atl::uint8` / `uint16` / `uint32` / `uint64` | `std::uint8_t` … `std::uint64_t` |
| `atl::int128` / `atl::uint128` | Native `__int128_t` / `__uint128_t` or portable software classes; see [128-bit integers](int128.md) |
| `atl::float32` / `float64` | `float` / `double` |

There is deliberately no `float128`: `long double` is 80-bit x87 on x86,
binary128 on AArch64 and plain `double` on MSVC, so no alias could promise the width.

## Concepts

Integer concepts from `<atl/types/concepts.hpp>`, which — unlike `std::integral` —
recognise the 128-bit types and deliberately exclude `bool` and the character types.

| concept | satisfied by |
|---|---|
| `atl::integral<T>` | any supported integer, including native or software `int128`/`uint128` |
| `atl::signed_integral<T>` | any signed integer, including `int128` |
| `atl::unsigned_integral<T>` | any unsigned integer, including `uint128` |
| `atl::modular_integer<Z>` | a modular integer type: `Z::mod()`, `value()`, construction from `int64`/`uint64`, `+ - * /` and their compound forms, negation, `inverse()`, `power(uint64)`; `mod_int` and `dynamic_mod_int` model it |
| `atl::static_mod_int<Z>` | a `modular_integer` whose `Z::mod()` is a constant expression (`mod_int`, not `dynamic_mod_int`) |

From `<atl/types/algebraic.hpp>`. Each checks that the call is well-formed, on a
`const` callable unless noted (the structures call them from const members, so a
`mutable` lambda is rejected at the boundary), and that the result converts to the
stated type:

| concept | call | used by |
|---|---|---|
| `atl::binary_operation<Op, T, R = T>` | `op(a, b)` on two `T`s, result convertible to `R` (default: closed) | every reducing structure |
| `atl::inverse_operation<Inverse, T>` | `inverse(a)`, result convertible to `T` | the group structures (`weighted_dsu`, `difference_fenwick`, …) |
| `atl::action<Mapping, F, S>` | `map(f, s)`: tag `F` on element `S`, result convertible to `S` | the lazy structures |
| `atl::fallible_action<Mapping, F, S>` | `map(f, s)` on `S &` in place, returning exactly `bool` (`false`: the aggregate alone cannot absorb the tag) | `segment_tree_beats` |
| `atl::update_function<Update, T>` | `f(x)` on a `T const &`, result convertible to `T`; checked on `Update &`, so a stateful callable is accepted | the `update(i, f)` members |
| `atl::fill_function<Fill, S>` | `fill(width)` or the position-aware `fill(lo, hi)`, result convertible to `S`; the two-argument form wins when both exist | `dynamic_lazy_segment_tree` |
| `atl::top_vertex_operation`, `atl::top_edge_operation`, `atl::path_action` | the vertex merge, edge merge and two-tag path action of a top tree | `top_tree` |

> **Note.** A concept checks only that the *call* is well-formed. The algebraic laws
> are preconditions on the caller, stated by each structure: a monoid (associative
> with an identity) for `segment_tree`, a commutative monoid for
> `binary_indexed_tree`, an idempotent commutative operation for `sparse_table`, a
> group (`op(a, inverse(a)) == identity`) for the group structures. Lazy structures
> need `action` to distribute over the element operation
> (`map(f, op(a, b)) == op(map(f, a), map(f, b))`) and to respect tag composition.

Two opt-in mode tags make a commutativity promise explicit in the type, so a
structure can exploit it. Both are unchecked, like every algebraic law:

| tag | promise | effect |
|---|---|---|
| `atl::commutative_tags_t` / `atl::commutative_tags` | `compose(f, g) == compose(g, f)` for every pair of tags held (add, xor, chmin, chmax; not assignment or affine maps) | `dual_segment_tree` skips boundary pushes; `persistent_lazy_segment_tree` keeps tags on the nodes they were applied to |
| `atl::commutative_fold_t` / `atl::commutative_fold` | `op(a, b) == op(b, a)` for the element fold | `link_cut_tree` keeps one aggregate per node instead of one per direction |

Pass the value as the first constructor argument (it deduces the mode) or the type as
the trailing template argument; `atl::tag_order<T>` and `atl::fold_order<T>` accept
`void` (no promise, the default) or the tag type.

## Integer traits

From `<atl/types/traits/integral.hpp>`, in `namespace atl::traits`. Each predicate has a
`struct` and a `_v` variable template; each transformation has a `struct` and a `_t`
alias.

| trait | result |
|---|---|
| `is_integral<T>` / `is_integral_v<T>` | `T` is a (cv-qualified) integer — signed or unsigned, `int128` included |
| `is_signed_integral<T>` / `is_signed_integral_v<T>` | `T` is a signed integer |
| `is_unsigned_integral<T>` / `is_unsigned_integral_v<T>` | `T` is an unsigned integer |
| `make_signed<T>` / `make_signed_t<T>` | the signed counterpart of `T` (cv preserved; handles `int128`/`uint128`) |
| `make_unsigned<T>` / `make_unsigned_t<T>` | the unsigned counterpart of `T` (cv preserved; handles `int128`/`uint128`) |

> **Note.** `make_signed`/`make_unsigned` have no `type` for unsupported types (including `bool`, floats,
> classes and enums), so requires-expression probes fail by substitution.
> Convert an enum to an explicit integer first rather than have arithmetic silently run
> on its underlying type.

### Type-list membership

From `<atl/types/traits/is_one_of.hpp>`:

| trait | result |
|---|---|
| `is_one_of<Tp, Types...>` / `is_one_of_v<Tp, Types...>` | whether `Tp` is exactly one of `Types...` |

### Widening for multiplication

From `<atl/types/traits/widen_mul.hpp>`: a type wide enough to hold the product of two
`T` values without overflow (e.g. for modular multiplication).

| trait | result |
|---|---|
| `widen_mul_type<T>` / `widen_mul_type_t<T>` | `int8→int16`, `int16→int32`, `int32→int64`, `int64→int128`, the unsigned analogues, and any other `T` mapped to itself |
| `has_widen_mul_v<T>` | whether `widen_mul_type_t<T>` is strictly wider than `T`, i.e. holds every product exactly; branch on this, never on `sizeof` |

`int64`/`uint64` always widen to native or software 128-bit types. When nothing wider
exists (`int128` itself, non-integers) the trait returns `T`,
which does NOT hold the product: test `has_widen_mul_v<T>` and take a fallback path.

## `atl::forward_like`

A portable spelling of C++23's `std::forward_like`: `using std::forward_like`
when `__cpp_lib_forward_like` is set, an identical hand-rolled cast otherwise.
It exists because `std::forward_like` only landed in libstdc++ 13 / libc++ 16,
and older toolchains that still accept `-std=c++23`
reject the *name* at parse time — so any header that merely mentions
`std::forward_like` fails to compile there, whether or not the offending
function is ever instantiated. Every atl container that perfect-forwards
range elements into storage (the segment trees, `binary_indexed_tree`,
`coordinate_compression`, `wavelet_matrix`, the persistent structures) uses
`atl::forward_like`, so they compile and run on those toolchains. Force the
fallback with `-DATL_NO_STD_FORWARD_LIKE` (mirrors the `ATL_NO_INT128`
pattern); `tests/types/forward_like.cpp` covers either branch, and `ATL_FORCE_FALLBACKS` selects it along with the other portable fallbacks.

## Size and cache helpers

| helper | header | description |
|---|---|---|
| `atl::num_bits<T>` | `<atl/types/utils.hpp>` | number of bits in the integer type `T` (`sizeof(T) * CHAR_BIT`; constrained to `atl::integral`) |
| `atl::prefetch<IsWrite = false>(void const *p)` | `<atl/utils/prefetch.hpp>` | cache hint for the line holding `p`, for a read (or a write with `IsWrite`); never faults, a no-op in constant evaluation and on compilers without `__builtin_prefetch` |

## Argument tags and vertex vocabulary

| entity | header | meaning |
|---|---|---|
| `atl::sorted_unique_t` / `atl::sorted_unique` | `<atl/types/tags.hpp>` | constructor tag: the input range is already strictly increasing under the container's comparator; the order is asserted, not sorted |
| `atl::graph::vertex_id` | `<atl/types/vertex.hpp>` | `uint32`; vertices are dense ids in `[0, vertex_count(g))` |
| `atl::graph::none` | `<atl/types/vertex.hpp>` | `uint32(-1)`: "no vertex / no edge / unreached" in results |
| `atl::graph::vertex_range<R>` | `<atl/types/vertex.hpp>` | an input range whose elements convert to `vertex_id`; a wider element type narrows silently, so its values must already be valid ids |

## Internal utility headers

`<atl/utils/floor_midpoint.hpp>` and `<atl/utils/radix_sort.hpp>` currently
contain only implementation helpers in `atl::detail`. Their contracts remain
in source; callers should use the public algorithms that build on them. They
are included in the coverage map because their files sit outside `detail/`.

## Assertions

From `<atl/utils/assert.hpp>`:

| macro | behaviour |
|---|---|
| `atl_assert(condition)` | with `-DATL_ASSERT`: on a false `condition`, print it with file/line to `stderr` and `std::abort()`; in constant evaluation a failure is a compile error. Without `ATL_ASSERT` it expands to `do { (void)sizeof(bool(condition)); } while (0)`: zero cost and never evaluated (still type-checked), so the condition must not carry side effects the program relies on. Both modes are statements, so expression/comma-operator use is rejected |

## Feature-detection macros

From `<atl/config.hpp>` (which also pulls in `<version>` for the `__cpp_lib_*` macros).

| macro | meaning |
|---|---|
| `ATL_HAS_INT128` | `1` when a native `__int128` is available (`__SIZEOF_INT128__` and neither `ATL_NO_INT128` nor `ATL_FORCE_FALLBACKS`), else `0`. Selects native vs software aliases; both provide the same atl traits |
| `ATL_FORCE_FALLBACKS` | define it to use every portable fallback: software int128 and portable word primitives, the `forward_like` cast, no-op prefetch, and value decomposition for wide floats |
| `ATL_PORTABLE_WIDE_ARITHMETIC` | define it to force portable word primitives at runtime, independently of the aliases |
| `ATL_NO_INT128` | define it to select software 128-bit aliases; must match across translation units |
| `ATL_NO_STD_FORWARD_LIKE` | define it to force the `atl::forward_like` fallback (see above) |

## Hash customization and seed lifetime

`<atl/utils/hash.hpp>` provides `atl::hash<K>`. A user specialization must
support `atl::hash<K>{}(value)`, with a result implicitly convertible to
`atl::uint64`. Construction, invocation, and conversion must be non-throwing
for use inside composite hashes. Equal values must hash equally while the
seed remains unchanged. Composite hashes preserve `noexcept` by rejecting
throwing element customizations at their constraints.

Pairs call the declared `hash<A>` and `hash<B>`, preserving cv-qualifiers;
arrays and vectors likewise use `hash<T>`. Tuples hash each const element
reference's decayed type, so `tuple<int&>` calls `hash<int>`. Integer pair
packing and integer byte-buffer paths never call element hashers and do not
require them. The customization's call must accept the actual element reference.

The random seed is drawn once during static initialization (a plain global, so
hashing pays no guard check: 5-8% faster `flat_hash_set` lookups on x86-64 than
a function-local seed). It stays fixed until `set_hash_seed(uint64)` changes
it. Call that function from `main` or later, before building any table that
uses the seed; no thread may hash or change the seed concurrently with that
call. Changing it invalidates previously computed hashes and table placement.
Do not build seeded tables during static initialization: a hash computed before
the seed's initializer runs uses seed 0 and changes afterwards.

Floating hashes normalize signed zero and NaNs. Known x87 extended layouts
use ten value bytes, excluding the padding in 12/16-byte objects; known IEEE
binary128 layouts use all sixteen bytes. Other wide layouts (and forced-fallback builds) use a value
significand/exponent decomposition. Composite hashes containing `long double`
inherit this correction; stored hash values are not a stable serialization.

## Hashing composite keys

From `<atl/utils/hash.hpp>`: `atl::hash<T>` is a seeded hash functor for
integers, enums, floating-point values, pointers, strings, pairs, tuples,
vectors and arrays. Equal keys hash equally within a process; hash values
may change between runs. `atl::identity_hash` is for already uniform integer
keys. Call `atl::set_hash_seed` (from `main` on) before populating any hashed
containers when reproducible hashes are needed; changing the seed invalidates
existing tables.

This complete example combines a public integer concept with a composite key.
The numeric result is deterministic even though the hash is seeded.

\include types-and-utilities/composite-key.cpp

Expected stdout:

\verbinclude types-and-utilities/composite-key.out

## Default folds

`atl::fill_function<Fill, S>` checks a const callable for the default aggregate
of an untouched interval. If `fill(lo, hi)` is invocable, that form is selected
and its result must convert to `S`; an invalid result rejects the callable even
when `fill(width)` would be valid. Only when the position form is not invocable
does the width form participate.

## See also

- [counting sort](counting-sort.md) — `atl::counting_sort` (`<atl/utils/counting_sort.hpp>`)
