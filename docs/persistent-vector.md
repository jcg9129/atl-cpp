# `atl::persistent_vector`

A growable persistent indexed sequence with concat, slice and insertion. The full vector allows branching; the semi-persistent variant allows mutation only of the newest version.

```cpp
#include <atl/ds/persistent_vector.hpp>
```

\include persistent-vector/basic.cpp

## API

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type |
| `Radix` | trie branching factor, a power of two (default 16 — default; larger values trade wider copies for a shallower trie) |

### Member functions

Both variants provide these:

| function | description | complexity |
|---|---|---|
| `persistent_vector()` | empty vector | O(1) |
| `persistent_vector(Range&& values)` | build from a sized range (moves out of an owning rvalue range; copies from an lvalue or an rvalue span/view) | O(n) |
| `persistent_vector(std::size_t n, T value = {})` | `n` copies of `value` | O(n) |
| `size()` → `std::size_t` | number of elements | O(1) |
| `empty()` → `bool` | whether `size() == 0` | O(1) |
| `operator[](std::size_t i)` → `T const&` | element at `i` | O(log n) |
| `front()` → `T const&` | first element (precond: non-empty) | O(log n) |
| `back()` → `T const&` | last element (precond: non-empty) | O(log n) |
| `begin()` / `end()` | leaf-caching forward iterator (`persistent_vector` only): one root descent per leaf; `std::ranges::` algorithms apply; invalidated like `operator[]` | O(log_Radix n) at a leaf boundary, O(1) otherwise; full traversal O(n + ceil(n/Radix) log_Radix n) |
| `a == b` | element-wise equality (`persistent_vector` only; O(1) for two handles of one version) | O(n + ceil(n/Radix) log_Radix n) |
| `reserve(std::size_t pushes)` | pre-grow the arenas for that many further `push_back`s onto this version (`persistent_vector` only) | O(shared pool entries) if relocated |
| `push_back(T value)` → `persistent_vector` | a new version with `value` appended | O(Radix) amortised |
| `pop_back()` → `persistent_vector` | a new version with the last element removed (precond: non-empty) | O(Radix) amortised |
| `set(std::size_t i, T value)` → `persistent_vector` | a new version with `[i] = value` | O(log n) |
| `emplace_back(Args&&...)` → `persistent_vector` | `push_back` of a `T` constructed in place from `args` | O(Radix) amortised |

`persistent_vector` additionally provides:

| function | description | complexity |
|---|---|---|
| `concat(persistent_vector const& other)` → `persistent_vector` | the concatenation `*this ++ other` | O(log n) if the two share an arena, else O(`other.size()`) |
| `slice(std::size_t from, std::size_t to)` → `persistent_vector` | the subrange `[from, to)` | O(log n) |
| `insert(std::size_t i, T value)` → `persistent_vector` | a copy with `value` inserted before index `i` | O(log n) |
| `emplace(std::size_t i, Args&&...)` → `persistent_vector` | `insert` before `i` of a `T` constructed from `args` | O(log n) |

Versions use shared pool ownership and are runtime-only (`std::shared_ptr`).
Copying a version costs O(1). Version-returning members have `const &` and `&&`
overloads: the latter consumes the handle; the moved-from object may only be
destroyed or assigned to. Allocation failure propagates an exception; mutations
are not transactional if allocation, element operations or callables throw.

## Complexity

Build takes O(n). Indexed operations and shared-arena surgery take O(Radix log_Radix n) time in the worst case; tables suppress the fixed Radix factor. Full-vector tail pushes copy O(Radix) values and spills also copy the trie path; along a linear sequence of pushes this is amortized O(Radix + log_Radix n). Branching can repeat a spill, so that amortized statement does not apply across arbitrary versions. Cross-arena concat deep-copies the right operand in O(other.size()), then copies the boundary path to join it. Full traversal of the leaf-caching iterator costs O(n + ceil(n/Radix) log_Radix n); it does not cache branch paths. Partial-vector fresh growth uses O(1) amortized new nodes per push. Memory is the total shared-pool allocation.

## Preconditions

Asserted under `ATL_ASSERT`: valid indices/ranges, nonempty front/back/pop,
logical size <= 2^32-1, available 32-bit pool ids, and reserve arithmetic.
Partial-vector mutations require the latest version; its signed version stamp
must remain below 2^31-1. `T` is `std::semiregular`; `Radix` is a power of two
from 2 through 65536. Ranges are sized and use the library element-forwarding
rule: lvalues/views copy; owning rvalues and yielded prvalues may move.

## Measured performance

Vector workloads are measured by `benchmarks/ds/persistent_vector.cpp`
(2026-11-22, AMD
EPYC-Milan ccx23, GCC 15.2.0, RelWithDebInfo, assertions off; plain
`-O2` / `-O2 -march=native`, five runs of 25 samples). One million
keep-all pushes take 175/186 ms for full persistence versus 5.9/9.6 ms
for semi-persistence; these offer different version-access contracts.
The list handle-copy comparison is on the [list page](persistent-list.md).

## Pitfalls and notes

References and iterators into a pool can be invalidated by mutations or reserve on any version sharing it. Copy values that must survive an update.

## How it works

The full vector has a tail leaf and a radix trie. Strict nodes use radix
digits; concat and slice introduce relaxed nodes with cumulative child-size
tables. Tail changes copy a leaf, and spills copy the branch path. Concat
rebalances boundary children; slice shares complete interior subtrees. A
separate arena handles each node kind. Partial persistence uses one modification
slot per node and version stamps; fresh territory can be filled in place,
while revisiting a filled slot copies the affected node. Its latest-version
check prevents branching mutations.

## See also

- [`persistent_array`](persistent-array.md) — fixed-size persistent array
- [`persistent_list`](persistent-list.md) — front-oriented persistent sequence
