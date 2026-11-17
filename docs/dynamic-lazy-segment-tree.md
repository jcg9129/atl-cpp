# `atl::dynamic_lazy_segment_tree`

Range actions and monoid folds over a large implicit `[0, n)`, with nodes
created along mutation paths. Use it when range boundaries arrive online and
the untouched aggregate can be computed cheaply. Compress boundaries and use
a dense lazy tree when an offline workload permits that.

```cpp
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
```

\include dynamic-lazy-segment-tree/quick-start.cpp

Expected stdout:

\verbinclude dynamic-lazy-segment-tree/quick-start.out

## API and complexity

`dynamic_lazy_segment_tree<Value, Tag, Op, Mapping, Compose, Fill>` requires
copyable values and tags, `binary_operation<Op, Value>`,
`action<Mapping, Tag, Value>`, `binary_operation<Compose, Tag>` and
`fill_function<Fill, Value>`. Callables are checked on const objects.
Aliases are `value_type = Value`, `tag_type = Tag`, `size_type = std::size_t`.
Costs assume constant-time algebra and fill calls.

| signature | behavior | cost |
|---|---|---|
| `constexpr dynamic_lazy_segment_tree()` | empty, value-initialized identities and callables; every type must be default-initializable | O(1) |
| `constexpr dynamic_lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{}, Compose compose = Compose{}, Fill fill = Fill{})` | initialize implicit defaults with `fill` | O(1) |
| `constexpr void apply(std::size_t l, std::size_t r, Tag const& f)` | tag every element in `[l, r)` | O(log n) time and new nodes |
| `constexpr void apply(std::size_t i, Tag const& f)` | tag one element | O(log n) time and new nodes |
| `constexpr void set(std::size_t i, Value x)` | assign one element | O(log n) time and new nodes |
| `constexpr Value fold(std::size_t l, std::size_t r) const` | ordered fold, identity if empty | O(log n), no allocation |
| `constexpr Value const& fold() const` | whole-domain fold | O(1) |
| `constexpr Value at(std::size_t i) const` | value under all pending tags | O(log n), no allocation |
| `template<std::predicate<Value const&> Pred> constexpr std::size_t max_right(std::size_t l, Pred pred) const` | largest accepted right endpoint | O(log n), no allocation |
| `constexpr std::size_t size() const noexcept`, `constexpr bool empty() const noexcept` | domain size and emptiness | O(1) |
| `constexpr std::size_t node_count() const noexcept` | pool entries including the null sentinel | O(1) |
| `constexpr void reserve(std::size_t ops)` | conservative node-capacity hint for more mutations | O(current entries) if reallocated |
| `constexpr void reserve_nodes(std::size_t count)` | reserve total capacity in pool entries | O(current entries) if reallocated |

Memory is O(q log n) after q mutations. Child ids are 32-bit. `fold()` returns
a pool reference that may become invalid after mutation or assignment. All
operations are constexpr when their value, tag and callable types support
constant evaluation.

## Preconditions

Asserted with `ATL_ASSERT`: `i < size()`, `l <= r <= size()`, valid search
endpoints, `pred(identity)`, available 32-bit ids, and reserve arithmetic that fits `size_t`.

Documented only: the element and tag operations are monoids;
`compose(f, g)` applies `g` first, then `f`. Mapping respects
composition and distributes over element folds. Carry width in the aggregate
when a tag needs it, as in the example. Mapping need not preserve element
identity: tags apply only to nonempty domain intervals, and empty query pieces
return the identity before mapping.

If `fill(lo, hi)` is invocable it wins, and its result must convert to `Value`;
an invalid position result rejects the callable even if `fill(width)` works.
The width form is selected only when the position form is not invocable.
For the selected form, empty intervals equal identity and splitting an
interval must agree with combining its fills. Width-only fill obeys
`fill(a+b) == op(fill(a), fill(b))`; position fill obeys
`fill(lo, hi) == op(fill(lo, mid), fill(mid, hi))`.

The domain fits `size_t`, and user arithmetic and reserve estimates must not
overflow. Allocation failure propagates an exception; mutations are not
transactional when an allocation or callable throws.

## Measured performance

`benchmarks/ds/dynamic_lazy_segment_tree.cpp` replays 100,000 affine/sum steps
over a 10^9 domain, including construction. Materialized push creates both
children at every push; offline compression uses width-carrying dense cells.
Point reads compare one tag-composing descent with a singleton range fold.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-15. Milliseconds, plain `-O2` / `-O2 -march=native`, both with
`-g -DNDEBUG -std=c++23`; minimum of three run means.

| mixed workload | time |
|---|---:|
| library | 225 / 229 |
| by materialized push | 222 / 226 |
| by offline compression | 123 / 120 |

| 1,000,000 point reads after 50,000 applies | time |
|---|---:|
| library | 395 / 396 |
| by singleton fold | 549 / 548 |

## Pitfalls and notes

Equality-comparable tags allow identity pushes to skip both mapping and child
materialization. A const fill may maintain a pure cache, but results must depend
only on its arguments. A `mutable` lambda itself is not const-callable.
For periodic implicit data, position fill can use a wrap-aware static RMQ.

## How it works

Each materialized node stores its current aggregate and a pending tag owed to
its children. Missing children are answered by fill. Mutation settles a live
tag onto both children, creates only the needed descent side when no tag is
pending, then pulls using child aggregates or fills. Reads carry composed
ancestor tags and never settle them. Canonical interval decomposition visits
O(log n) nodes, and a point read follows one root-to-leaf path.

## See also

- [Dynamic segment tree](dynamic-segment-tree.md)
- [Lazy segment tree](lazy-segment-tree.md)
- [Range minimum queries](rmq.md)
