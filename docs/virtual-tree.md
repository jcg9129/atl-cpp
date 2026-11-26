# `atl::graph::virtual_tree_of`

A virtual tree keeps a key set and the LCAs needed to connect it, hanging each
vertex under its nearest kept ancestor. Use it for many sparse queries whose
total key count is bounded. A whole-tree DP is simpler for one dense query.

```cpp
#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
```

\include virtual-tree/quick.cpp

## API and complexity

```cpp
struct virtual_tree {
    std::vector<vertex_id> vertices;
    std::vector<uint32> parent;
};
```

`vertices` contains original ids in `tree.entry_time` order. `parent[i]` is
a local index into these arrays, or `none` for each root. Parents precede
children; every virtual subtree is contiguous. With k distinct nonempty keys,
there are at most 2k - 1 vertices. An empty range produces empty arrays;
keys across forest components give several roots. Duplicate keys are allowed.

| Signature | Meaning |
|---|---|
| `template<vertex_range R> requires std::convertible_to<std::ranges::range_reference_t<R>, vertex_id> constexpr void virtual_tree_of(lca const &tree, R &&keys, virtual_tree &out)` | replace the result, retaining buffer capacity |
| `template<vertex_range R> requires std::convertible_to<std::ranges::range_reference_t<R>, vertex_id> [[nodiscard]] constexpr virtual_tree virtual_tree_of(lca const &tree, R &&keys)` | return an owning result |
| `inline void virtual_tree_clear()` | release the current thread's scratch |

`keys` may be a non-const-iterable input view or a range of convertible proxies.
The result owns its arrays; retaining references across refill can invalidate
them. It borrows nothing from `tree` once built. Edge lengths are the caller's:
`tree.depth(child) - tree.depth(parent)`, or the difference of precomputed
weighted depths. The result does not mark which vertices were keys.

Construction comparison-sorts in O(k log k) time with at most k - 1 LCA
queries and O(k) scratch. For k >= ceil(n/128), a bitmap takes O(k + n/64)
time and O(k + n/64) scratch words; a sized input with at most 32 keys uses
the fixed stack buffer instead. The runtime overloads reuse thread-local
scratch; once capacities suffice, refill needs no further allocation.
Constant evaluation uses local scratch, including the bitmap path.
`virtual_tree_clear` takes time proportional to released storage and is not
constexpr because it accesses thread-local state. Allocation failures propagate.

## Preconditions

Asserted under `ATL_ASSERT`: every converted key is below tree.size(), and
there are fewer than 2^31 input keys. Values must fit vertex_id when converted.

Documented only: input iteration must not invoke virtual_tree_of or
virtual_tree_clear on the same thread, and keys must not alias output storage.
The LCA and graph must satisfy [their own contracts](lca.md). Callback and
range exceptions propagate; scratch is not a transaction with rollback.

## Measured performance

The alternative builders are defined in
[`virtual_tree_variants.hpp`](../experimental/atl/experimental/graph/virtual_tree_variants.hpp).

[benchmarks/graph/virtual_tree.cpp](../benchmarks/graph/virtual_tree.cpp)
changes key sorting while keeping the stack sweep and query keys fixed.
The counting-dispatch contender comparison-sorts below ceil(n/32), then counts;
pure counting scans the key domain, while bitmap sorting scans words and
collapses duplicates. Cells are minimum milliseconds over five runs of 25
samples, plain / native. LCA construction is excluded.

The counting-dispatch rows measure `virtual_tree_by_sort` with
`counting_dispatch`; the bitmap rows measure the library entry point.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| n | batch | by counting dispatch | library bitmap | by counting | by comparison |
|---|---|---|---|---|---|
| 500000 | 1 query, k=n/2 | 10.9 / 10.8 | 8.6 / 8.9 | 11.5 / 10.9 | 25.4 / 28.3 |
| 500000 | 4 queries, k=n/8 | 12.6 / 12 | 11.3 / 11 | 12.5 / 12 | 25.1 / 24.7 |
| 500000 | 16 queries, k=n/32 | 15.8 / 15.9 | 13 / 13 | 15.8 / 17 | 25.3 / 24.8 |
| 500000 | 64 queries, k=n/128 | 24.8 / 22.8 | 25.2 / 28.2 | 24.4 / 24.3 | 24.3 / 29.8 |
| 500000 | 512 queries, k=n/1024 | 19.5 / 19.5 | 19.5 / 18.9 | 99.6 / 103 | 20.1 / 21 |
| 200000 | 1 query, k=n/2 | 3.9 / 3.8 | 3.2 / 3.1 | 3.8 / 3.7 | 9.4 / 9.6 |
| 200000 | 4 queries, k=n/8 | 4.6 (1.04x) | 4.2 (1.04x) | 4.7 (1.14x) | 9.3 (1.04x) |
| 200000 | 16 queries, k=n/32 | 5.8 (1.02x) | 5 (1.02x) | 5.8 (1.03x) | 9 (1.02x) |
| 200000 | 64 queries, k=n/128 | 8.5 (1.06x) | 8.5 (1.03x) | 9.4 (1.02x) | 8.5 (1.03x) |
| 200000 | 512 queries, k=n/1024 | 6.9 (1.03x) | 6.9 (1.04x) | 36.9 (1.04x) | 6.9 (1.03x) |

The library switches to the bitmap from ceil(n/128), except for the fixed
buffer path. At n/1024, comparison sorting is competitive; the precise
crossover between n/128 and n/1024 is unmeasured. The benchmark compares
one-sort and two-sort algorithms,
packed/projected/entry-time keys, an entry-space sparse-table backend,
depth-ranked frames, a raw stack array and per-query scratch allocation.
At n=500000, k=n/128 the library bitmap loses to counting
dispatch (25.2/28.2 versus 24.8/22.8 ms). It beats that contender for
the three larger key batches in both modes.

All shown n=500000 sorting comparisons include rows that vary between
runs (spreads up to 2.03x). Their minimum-time rankings alone do not
establish stable crossover points.

## Pitfalls and notes

Keep vertex ids distinct from local result indices. For a DP indexed by local
position, initialize exactly result.vertices.size() slots per query. For a DP
indexed by original vertex, reset only kept vertices. Clear caller key marks
by walking the keys so sparse queries do not become O(n).

A reverse pass is a bottom-up DP order. Marked-vertex separation, for example,
counts child subtrees still reaching a marked vertex: a marked vertex requires
cutting each open child path; an unmarked branching vertex can itself be
removed. Check original adjacent marked pairs separately, because a virtual
edge can represent more than one original edge.

\include virtual-tree/separation.cpp

## How it works

Keys pack `uint64(entry_time) << 32 | vertex`, keeping shifts portable on
32-bit targets. Comparison sorting orders these integers directly; bitmap
sorting sets one bit per entry time, then scans set bits through tree.order().
Each scanned word is cleared for the next query.

The sweep runs from largest entry time toward smallest. A monotone stack holds
one root path; joining a new key with the top identifies which frames finish.
Pops form a postorder with reverse sibling order. Reversing the pop sequence
therefore produces the required preorder. Finished children wait in lists
threaded through the output parent array until their parent's index exists.
Every frame is pushed and popped once. Crossing components flushes the stack.
Depth ranks replace entry ranks for large inputs; on one root path they agree.
The sweep uses heap buffers and no recursion per vertex.

## See also

- [Lowest common ancestors](lca.md).
- [Graph vocabulary](graph.md).
- [Counting sort](counting-sort.md).
