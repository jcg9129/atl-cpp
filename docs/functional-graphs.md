# `atl::graph` functional graphs

Static functional graphs have one outgoing edge per vertex. These indices answer repeated
k-step endpoint and ordered edge-fold queries, including walks much longer than the graph.
Use ordinary graph traversal when successors change or vertices can have sinks.

```cpp
#include <atl/graph/functional_graph.hpp>
```

\include functional-graphs/quick.cpp

## API

All entities below are in `atl::graph`, all functions are `constexpr`, and step counts use
`atl::uint64`. `vertex_id` is `atl::uint32`; `none` is the reserved non-vertex sentinel.

```cpp
template<typename Successors>
concept successor_range = vertex_range<Successors> && std::ranges::sized_range<Successors>
    && detail::vertex_integral<std::ranges::range_value_t<Successors>>
    && std::convertible_to<std::ranges::range_reference_t<Successors>, std::ranges::range_value_t<Successors>>;

template<typename Queries>
concept functional_graph_query_range = std::ranges::input_range<Queries> && std::ranges::sized_range<Queries>
    && std::same_as<std::ranges::range_value_t<Queries>, functional_graph_successor_query>
    && std::convertible_to<std::ranges::range_reference_t<Queries>, functional_graph_successor_query>;

template<typename Values, typename Value>
concept functional_graph_value_range = std::ranges::input_range<Values> && std::ranges::sized_range<Values>
    && std::copyable<Value> && std::same_as<std::ranges::range_value_t<Values>, Value>
    && std::constructible_from<Value, atl::detail::forwarded_element_t<Values>>;

inline constexpr uint64 maximum_functional_graph_steps = 1'000'000'000'000'000'000ULL;

struct functional_graph_decomposition {
    std::vector<uint32> cycle_id;
    std::vector<uint32> cycle_position;
    std::vector<uint32> depth_to_cycle;
    std::vector<vertex_id> cycle_root;
    vertex_groups cycles;
};
struct functional_graph_successor_query {
    vertex_id start;
    uint64 steps;
};
template<std::copyable Value>
struct functional_graph_walk_result {
    vertex_id vertex;
    Value fold;
};

template<successor_range Successors>
[[nodiscard]] constexpr functional_graph_decomposition
functional_graph_decomposition_of(Successors &&successors);
```

`successor_range` accepts sized input ranges of integral ids, excluding bool and character
types. The `values` range has exactly `Value` as its range value type. Successor and query references must convert to their range value types; proxies are supported.
Both input ranges are consumed once; values move only from owning rvalues, and copy from spans and views.

`cycle_id[v]` is the eventual cycle, including for tail vertices. Cycles are numbered by
increasing minimum vertex; each list starts at its minimum and follows successor order.
`cycle_position[v]` is the cycle index, or `none` on tails. `cycle_root[v]` is the first cycle
vertex reached, itself on cycles; `depth_to_cycle[v]` counts steps to it. `cycles` contains
exactly the depth-zero vertices. Empty graphs have empty arrays and `cycles.offsets == {0}`.

```cpp
class functional_graph_successor_index {
public:
    template<successor_range Successors>
    constexpr explicit functional_graph_successor_index(Successors &&successors);
    constexpr std::size_t size() const noexcept;
    [[nodiscard]] constexpr functional_graph_decomposition const &decomposition() const noexcept;
    [[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const;
};

template<std::copyable Value, binary_operation<Value> Operation>
    requires std::copyable<Operation>
class functional_graph_walk_index {
public:
    template<successor_range Successors, typename Values>
        requires functional_graph_value_range<Values, Value>
    constexpr functional_graph_walk_index(Successors &&successors, Values &&values,
        Value identity, Operation operation);
    constexpr std::size_t size() const noexcept;
    [[nodiscard]] constexpr functional_graph_decomposition const &decomposition() const noexcept;
    [[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const;
    [[nodiscard]] constexpr functional_graph_walk_result<Value> walk(vertex_id start, uint64 steps) const;
};
```

For `x[0] = start` and `x[i+1] = successors[x[i]]`, `successor(start,k)` returns `x[k]`.
`values[v]` is the value of the step `v -> successors[v]`. `walk(start,k)` returns `x[k]`
and `values[x[0]] op ... op values[x[k-1]]` in that order: the k vertices it leaves, start
included and endpoint excluded, so `walk(v,a+b)` is `walk(v,a)` then `walk(x[a],b)`. Zero steps returns `{start, identity}`.
Supply the walk index's template arguments explicitly; there are no deduction guides. Neither values
nor operations need default constructors. The concepts check callable shape and copyability;
they cannot establish the monoid laws. Operations must be copy-assignable as well as
copy-constructible; use `std::ref` or `std::cref` for capturing lambdas. A mutable callable
can be invoked through `std::ref`, provided its returned values satisfy the stable monoid contract.

Indices own preprocessing state. Copies have independent storage. Metadata references last
until destruction, move or assignment. Queries are const, use no engine scratch allocations,
and the walk index's `successor` consults no values; `Value` and `Operation` may themselves allocate.

## Complexity

Bounds assume constant-size values and constant-time monoid operations; scale by their costs
otherwise. Let n be the vertex count and k the requested step count.

| Operation | Time | Storage |
|---|---|---|
| Decomposition or either index construction | O(n) | O(n) |
| `size`, `decomposition` | O(1) | O(1) auxiliary |
| Either index's `successor` | O(log(n+1)); O(1) when k reaches the cycle | O(1) auxiliary |
| `walk` | O(log²(n+1) + log(k+1)) | O(1) auxiliary values |

The public defaults use indegree peeling and the existing `lca::ancestor` with cycle
arithmetic: linear construction and storage, with constexpr support. Tail folds compose
reversed preorder segment-tree intervals; cycle prefixes/suffixes supply rotated complete
laps. These bounds motivate the defaults; there is no measured performance comparison.

## Preconditions

Asserted with `atl_assert` under `-DATL_ASSERT`:

- At input validation, n < 2^31. Each raw successor must be nonnegative,
  representable as `vertex_id`, and less than n; `none` is invalid.
- Value count equals n. Walk engines using segment trees also require
  `n <= 2^(numeric_limits<size_t>::digits - 2)`, hence n <= 2^30 with 32-bit `size_t`.
  The lifting tables assert that `n × levels` fits the corresponding vector's `max_size()`.
- Each query start is less than n, and each step count is at most 10^18.
  Empty graphs and empty batches are valid; querying an empty graph is invalid.

Documented only: `operation` is associative and stable/pure, `identity` is a two-sided identity,
and **all intermediate folds** are representable in `Value`, including precomputed blocks
that are never directly requested. Commutativity, inverses, idempotence and nonnegative values
are unnecessary.

All traversals use heap buffers and explicit stacks; none recurse per vertex. Allocation
failures and container length errors propagate. Stored ids, positions, indegrees and tail
depths fit `uint32` because n < 2^31; even n + 1 fits 32-bit `size_t`. Step arithmetic stays
in `uint64`: k <= 10^18 < 2^60, and subtracting the tail occurs only when k reaches it.
Cycle-position addition is performed in `uint64` and reduced modulo the cycle length
prior to conversion to `size_t`. Batch links and result indices use `size_t`; no uint32
query-count limit is imposed beyond the containers' allocation limits.

## Measured performance

The comparison is **not measured yet**. The source is [benchmarks/graph/functional_graph.cpp](../benchmarks/graph/functional_graph.cpp).
No benchmark records, timings or measured rankings are available.

The benchmark compares peeling with marking decomposition; online endpoint and walk-fold
lifting against the public engines; heavy-chain upward prefixes; and both offline reverse-DFS
batches. Shapes are a long cycle, a long tail into a self-loop, balanced in-trees into many
short cycles (plus a star extreme), and a relabeled random mapping. Sizes, query/vertex ratios,
small/boundary/large steps, repeated/random starts, scalar xor and noncommutative affine folds
are swept. Each row has a unique workload/engine/phase key. The row suffix records an untimed
census of retained and peak **requested heap bytes** (excluding input arrays, allocator overhead
and stack storage), plus monoid-operation count. Online rows split build, queries, and combined
costs; build and combined rows also include freeing the index they build, like the build rows of
the other atl benchmarks. Batch rows include query materialization and output allocation. Output checksums are
common within each workload. Allocation metering and operation counting are disabled while
timing. The benchmark's allocation hooks apply only to its own executable.

## Pitfalls and notes

Cycle products depend on rotation when composition is noncommutative. A walk that enters in
the middle of a cycle must repeat the lap starting there. A prefix cannot be subtracted from
a fold without an inverse, and the API makes no such assumption.

Experimental contenders live in `<atl/experimental/graph/functional_graph_variants.hpp>`
under `atl::experimental` (add `-Iexperimental`). Every contender is not measured yet:

| Entity | Purpose and theoretical bounds |
|---|---|
| `functional_graph_decomposition_by_marking(successors)` | Canonical iterative path-stack decomposition, O(n) time/space |
| `functional_graph_successor_index_by_lifting` | Same endpoint interface, O(nL) build/storage, O(L) query |
| `functional_graph_walk_index_by_lifting<Value,Operation>` | Same walk interface, O(nL) build/storage, O(L) query |
| `functional_graph_walk_index_by_chain_prefix<Value,Operation>` | Same walk interface, O(n) build/storage, O(log(n+1) + log(k+1)) walk |
| `functional_graph_successors_by_reverse_dfs(successors, queries)` | Plain `std::vector<vertex_id>`, O(n+q) time/peak space |
| `functional_graph_walks_by_reverse_dfs(successors, values, queries, identity, operation)` | Plain `std::vector<functional_graph_walk_result<Value>>`, O((n+q)log(n+1) + sum log(k+1)) time, O(n+q) peak space |

Here L = bit_width(10^18), and query ranges satisfy `functional_graph_query_range`. All are
constexpr, have the same input/query and monoid contracts, and are **not measured yet**. Batches
preserve input query order and duplicates. Their step bound is fixed at 10^18; no constructor
step-limit argument or accessor exists. Batch functions preprocess once for the supplied query set and retain no reusable index.

Alternatives deliberately absent from this comparison:

- Floyd/Brent cycle detection uses constant space for one start but does not build shared metadata for all starts.
- Direct single-start walking costs O(k), unsuitable for the supported 10^18-step queries.
- Disjoint sparse tables require O(n log n) storage for interval folds.
- Rolling doubling rows discard the tables needed for subsequent online queries.
- A separate mirrored-segment-tree baseline duplicates the public walk engine's interval design.
- A new O(1) level-ancestor structure would add preprocessing machinery beyond the reused linear-space LCA engine.
- General SCC is a test cross-check; one-outdegree structure permits the simpler linear peeling implementation.

## How it works

Peeling repeatedly removes indegree-zero vertices. The remaining vertices are exactly the
cycles. Scanning vertices in increasing order discovers each cycle at its minimum and writes
its successor order. Reverse peel order then propagates the cycle id, entry root and depth
through tails. Marking instead records each unfinished successor path, marks a cycle on
reentering that path, and propagates metadata in completion order; it uses the same linear
canonicalization sweep.

Cut every cycle vertex's outgoing edge to produce a forest. Existing `lca` preprocessing gives
ancestors and heavy-first preorder. A short endpoint query uses `ancestor`; a query reaching
the cycle subtracts the tail depth and reduces its remaining steps modulo the cycle length.
The empty graph bypasses `lca`. Tail folds query a segment tree in reversed preorder,
excluding the top endpoint so each step's value is counted exactly once. The heavy-prefix contender
derives chain heads from parent/preorder hooks, caches the upward fold to each head, and needs
a segment-tree query only for its last interval.

Cycles store prefixes and suffixes. A full lap at position p is `suffix[p] op prefix[p]`;
binary powering repeats that rotated lap without storing powers per cycle. At most two ordered
intervals supply the remainder. Tail folds always precede cycle folds. Avoiding an unused final
squaring also avoids creating an unneeded intermediate value.

Lifting stores level-major endpoint and optional fold rows. Offline batches bucket queries by
start and traverse reverse trees iteratively, answering short ancestors from the active depth
stack. Walk-fold batches keep a segment tree in reversed active-depth order and share the cycle
fold machinery. All traversals are iterative; long tails need no recursive call stack.

## See also

- [Graph vocabulary](graph.md)
- [Lowest common ancestors](lca.md)
- [Segment trees](segment-tree.md)
- [Static range folds](sparse-table.md)
