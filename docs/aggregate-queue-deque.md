# `atl::aggregate_queue`, `atl::aggregate_deque`

Mutable containers that keep the fold of their live values, from front to back, over a
**monoid**: `aggregate_queue` pushes at the back and pops at the front (a FIFO sliding
window), `aggregate_deque` pushes and pops at both ends. Every update is O(1) amortized and
`fold()` is O(1). The operation need not be commutative or invertible (matrix product, affine
composition, string concatenation, min/max all work). For folds of arbitrary subranges or
indexed updates use a [segment tree](segment-tree.md); for folds of static data a
[sparse table](sparse-table.md) or [`direct_rmq`](rmq.md).

```cpp
#include <atl/ds/aggregate_queue.hpp>
#include <atl/ds/aggregate_deque.hpp>
```

\include aggregate-queue-deque/sliding-window.cpp

Expected stdout:

\verbinclude aggregate-queue-deque/sliding-window.out

## API

Both classes are `template<std::copyable Value, binary_operation<Value> Operation>`.
[`binary_operation<Value>`](../include/atl/types/algebraic.hpp) requires the operation to be
callable as a `const` object on two `Value const &` operands with a result convertible to
`Value`; the engines call it exactly that way and convert the result explicitly, so a narrow
`Value` (`std::int16_t` under `std::plus<>`, which yields `int`) works. `Value` needs neither
default construction nor equality. There is no default constructor and no default algebra:
the identity is always passed, and both template arguments are deduced from the constructor
(`atl::aggregate_deque deque(0, std::plus<>{})`). A capturing lambda, a `std::reference_wrapper`
or a callable with a deleted non-const call operator is accepted.

| member | description | complexity |
|---|---|---|
| `using value_type = Value;`, `using size_type = std::size_t;` | member types | |
| `constexpr explicit aggregate_queue(Value identity, Operation operation = Operation{})` | an empty queue; `fold()` of an empty container is `identity` | O(1) |
| `constexpr explicit aggregate_deque(Value identity, Operation operation = Operation{})` | an empty deque | O(1) |
| `constexpr void push_back(Value value)` | append | O(1) amortized, O(n) worst case |
| `constexpr void push_front(Value value)` (deque) | prepend | O(1) amortized, O(n) worst case |
| `template<typename... Arguments> requires std::constructible_from<Value, Arguments...> constexpr void emplace_back(Arguments &&...arguments)` | append `Value(arguments...)`, built before any storage changes, so the arguments may refer to `front()` or `back()` | O(1) amortized, O(n) worst case |
| `template<typename... Arguments> requires std::constructible_from<Value, Arguments...> constexpr void emplace_front(Arguments &&...arguments)` (deque) | prepend `Value(arguments...)`, likewise | O(1) amortized, O(n) worst case |
| `constexpr void pop_front()` | remove the first value | O(1) amortized, O(n) worst case |
| `constexpr void pop_back()` (deque) | remove the last value | O(1) amortized, O(n) worst case |
| `[[nodiscard]] constexpr Value const &front() const` | the first value | O(1) |
| `[[nodiscard]] constexpr Value const &back() const` | the last value | O(1) |
| `[[nodiscard]] constexpr Value fold() const` | `operation` folded over the live values from front to back, or `identity` when empty | O(1), one operation call |
| `constexpr size_type size() const noexcept` | number of live values | O(1) |
| `constexpr bool empty() const noexcept` | whether `size() == 0` | O(1) |
| `[[nodiscard]] constexpr size_type max_size() const noexcept` | the largest `size()` a push or `reserve` may reach: the smallest element limit of the backing vectors | O(1) |
| `constexpr void clear()` | destroy the live values; keep the allocation, identity and operation | O(n) |
| `constexpr void reserve(size_type elements)` | allocate for `elements` simultaneously live values, transfer and split storage included: no later operation allocates while `size() <= elements` | O(n + elements) |

Everything is `constexpr`. Copies are deep (O(n)); moves are O(1) container work and leave
the source only destructible or assignable. References from `front()` and `back()` stay valid
until the next mutation, `reserve`, move or assignment of the container.

## Complexity

Bounds count value transfers and operation calls; the cost of one call (a string
concatenation, a matrix product) multiplies them.

| operation | `aggregate_queue` | `aggregate_deque` |
|---|---|---|
| push | O(1) amortized, O(n) worst case (vector growth), one call | O(1) amortized, O(n) worst case (vector growth), one call |
| pop | O(1) amortized; O(n) worst case when the back stack is transferred | O(1) amortized; O(n) worst case when the other side is split |
| `front`, `back`, `size`, `empty` | O(1) | O(1) |
| `fold` | O(1), one call, allocates no container storage | O(1), one call, allocates no container storage |
| `clear` | O(n) | O(n) |
| `reserve(elements)` | O(n + elements) | O(n + elements) |

Storage is O(n) live; the allocation retained after pops or `clear()` is O(H + reserved
elements) for the largest occupancy H so far. Each live value is stored twice (raw and as a
partial fold); the deque adds a split buffer that is empty between operations.

## Preconditions

Asserted with `atl_assert` (active under `-DATL_ASSERT`):

- `front`, `back` and every pop need a nonempty container.
- A push needs `size() < max_size()`, and `reserve(elements)` needs `elements <= max_size()`. The checks run before any addition or allocation, so no
  `std::size_t` count overflows, on 32-bit targets included.

Documented only:

- `(Value, Operation, identity)` is a monoid: the operation is associative and `identity` is
  neutral on both sides. Commutativity, inverses and idempotence are not needed.
- Every intermediate fold is valid arithmetic for the user's types (no signed overflow).
  Floating-point addition is not exactly associative, so the result can differ in rounding
  from a left-to-right sum.
- Moves and copies of `Value`, calls of the operation and the conversion of its result to
  `Value` do not throw during a mutation: a throw while a pop transfers or splits values would
  leave them out of order or stranded in the scratch buffer. A failed allocation throws
  `std::bad_alloc` and leaves the contents unchanged: pops allocate their transfer or split
  storage before moving any value, and a copy assignment builds a temporary copy first.

## Measured performance

Measurements: 2026-11-23. The source is
[benchmarks/ds/aggregate_queue_deque.cpp](../benchmarks/ds/aggregate_queue_deque.cpp).
Both flag sets are measured; the public defaults do not win every workload.

The benchmark races the library deque on every shape and, on the FIFO shapes, the library
queue and the three experimental queue contenders below. Shapes are a fixed-width sliding
window, sawtooth fill-and-drain, drains alternating between the deque's ends, and a random
four-ended walk, each at widths 16, 4096 and 131072. Values are xor of random words, 4x4 matrix
products over Z/101Z (noncommutative), and min and max over random, increasing, decreasing and
plateau inputs. Every trace is built before timing, ends empty, and is first checked against
the library deque (fold, size and endpoints after every update). Each row reports end-to-end
replay throughput with the fold read after every update, the same with `reserve(width)`, and a
reserved replay; separate untimed passes report p99, p999 and maximum update latency (clock
overhead included), operation calls per update and per fold, allocation calls and peak
requested heap bytes (allocator overhead excluded).

On AMD EPYC-Milan ccx23, GCC 15.2.0, RelWithDebInfo, assertions off,
plain `-O2` / `-O2 -march=native`, five runs of 25 samples, the fixed FIFO
width=16 xor end-to-end queue loses to DABA-lite ring (0.35/0.336 versus
0.234/0.268 ms). Reserved replay favors the queue (0.109/0.095 versus
0.22/0.235 ms). These are distinct timed phases, not a uniform engine win.

## Pitfalls and notes

- Pick the queue for FIFO windows: its transfer moves each value once, while the deque's
  split can move a value several times.
- The worst case is the bound for a single update: a pop that transfers or splits n values
  makes n operation calls. A latency-sensitive FIFO can use the DABA Lite contenders below,
  which bound the calls per update by a constant.
- `fold()` returns by value; with a heavy `Value` read it once per step.
- There is no iteration, indexed access, range constructor or automatic eviction: the caller
  pops expired values itself.

Experimental contenders live in
[`<atl/experimental/ds/aggregate_queue_variants.hpp>`](../experimental/atl/experimental/ds/aggregate_queue_variants.hpp)
under `atl::experimental` (add `-Iexperimental`). All three mirror the queue's interface:

- `aggregate_queue_by_daba_lite_ring<Value, Operation>`: the DABA Lite schedule on a growable
  ring. At most three operation calls per push, two per pop and one per fold, whatever the
  occupancy; ring growth still relocates O(n) records (none after a sufficient `reserve`).
- `aggregate_queue_by_daba_lite_chunks<Value, Operation>`: the same schedule on fixed-size
  chunks of 32 records that never relocate; chunks emptied at the front are reused at the
  back, so an update does constant container work apart from allocating a chunk (allocator
  latency is not bounded). The chunk size is not tuned.
- `aggregate_queue_by_monotonic_candidates<Value, Compare = std::less<>>`: minimum
  (`std::less<>`) or maximum (`std::greater<>`) only; the second argument is a strict weak
  ordering, not a monoid operation, and the identity must be neutral for the selected extremum: it must not precede any stored value.
  It keeps the values that can still become the extremum, so pops and folds make no
  comparison and a push makes amortized O(1) comparisons. It does not fit general idempotent
  folds (gcd, bitwise and) and has no `pop_back`.

The deque has no experimental contender; the benchmark runs the library deque alone on the
four-ended shapes. Deliberately absent alternatives:

- Full DABA: the same constant worst-case bound per update as DABA Lite, with a more involved
  schedule.
- Two-Stacks Lite: the same amortized transfer as the library queue; it saves storage by
  overwriting values with folds, which `front()` and `back()` need.
- An adaptation of Okasaki's real-time queue: it bounds the worst case like DABA Lite, through
  incremental rotations that keep more state than DABA Lite's four boundaries.
- A four-ended monotonic candidate deque: a `pop_back` would have to restore the candidates its
  push removed; `aggregate_deque` covers min and max at both ends.
- Range-query structures (sparse tables, segment trees, treaps, direct RMQ) as window engines:
  they answer arbitrary ranges at O(log n) or static-build cost, which a window that only moves
  at its ends does not need.

## How it works

**Two stacks.** Both public engines store two stacks of (value, fold) entries. The front stack's
top is the first value, and each entry folds itself and everything below it in logical order
(`op(value, below)`); the back stack's top is the last value, and each entry folds everything
below it and then itself (`op(below, value)`). The container's fold is
`op(front fold, back fold)`: one call, no container storage allocated. Pushes add one entry with one call.

**Queue transfer.** A pop from an empty front stack first moves the whole back stack onto it,
which reverses it and rebuilds every fold in the front direction. Each value crosses once, so
pops are O(1) amortized.

**Deque split.** A pop from an empty side splits the other side: the requested side receives
ceil(n/2) values and the other floor(n/2), both rebuilt in logical order through a scratch
buffer; a side holding a single value is popped directly. The split costs O(n) and leaves the
size difference of the two sides at most one, while every push or pop changes it by one, so
the difference pays for each split and updates stay O(1) amortized. Moving everything to the
requested side instead would make alternating pops at the two ends quadratic. Every
destination (scratch, both stacks) is reserved before the first value moves, which is why a
failed allocation changes nothing.

**DABA Lite.** The ring and chunk contenders follow DABA Lite: the stored values are divided by
four inner boundaries into a folded prefix, a region of local suffix folds, raw values, and
new suffix folds, followed by a back region summarized by one running fold. Each update does
one constant-size repair step (adopt a singleton, start a new cycle, or extend one suffix and
finish one prefix entry), so suffix folds are built incrementally instead of in one transfer.
The ring stores boundaries as offsets from the live front, which never outgrow `size()`; the
chunks store them as cursors into chunks whose addresses never change, and a copy rebinds the
four cursors in one linear walk. The chunk list is destroyed iteratively.

**Monotonic candidates.** Each candidate carries the number of live values that expire through
it; a push merges the spans of the candidates it strictly precedes, and a pop decrements the
first span. Counting instead of arrival indices keeps duplicates apart without equality and
cannot overflow.

Reference: Tangwongsan, Hirzel and Schneider, "In-order sliding-window aggregation in
worst-case constant time", VLDB Journal (2021), sections 3, 6 and 7. The implementations were
written independently from the paper's invariants and schedule.

## See also

- [Segment tree](segment-tree.md): point updates and arbitrary range folds.
- [Sparse tables](sparse-table.md) and [range minimum queries](rmq.md): constant-time folds of static data.
- [Persistent queue and deque](persistent-queue-deque.md): persistent FIFO and deque without folds.
- [Treap sequence](treap-sequence.md): positional surgery with folds.
