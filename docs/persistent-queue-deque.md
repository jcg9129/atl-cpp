# `atl::persistent_queue`, `atl::persistent_deque`

An unbounded persistent FIFO and a fixed-capacity persistent deque. Each mutation returns a new version; use a list when only stack operations are needed.

```cpp
#include <atl/ds/persistent_adapters.hpp>
```

\include persistent-queue-deque/basic.cpp

## API

### `persistent_queue`

Fully persistent, unbounded FIFO. Read the oldest element with `front()`, then drop
it with `pop()`.

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type |

#### Member functions

| function | description | complexity |
|---|---|---|
| `persistent_queue()` | an empty queue | O(1) |
| `size()` → `std::size_t` | number of elements | O(1) |
| `empty()` → `bool` | whether `size() == 0` | O(1) |
| `front()` → `T const&` | the oldest element (precond: non-empty) | O(min(size, log pushes)) |
| `back()` → `T const&` | the most recently pushed element (precond: non-empty) | O(1) |
| `push(T value)` → `persistent_queue` | a new version with `value` at the back | O(1) |
| `emplace(Args&&... args)` → `persistent_queue` | `push` of an element built in place from `args` | O(1) |
| `pop()` → `persistent_queue` | a new version with the front removed (precond: non-empty) | O(1) |

The mutators are `[[nodiscard]]` — they return the new version rather than mutating.

### `persistent_deque`

Persistent double-ended queue over a fixed-capacity circular array. The physical
slot is `index % capacity`, so slots freed at one end are reused at the other and the
deque holds up to `capacity` elements concurrently regardless of the total number of
operations. Persistence is inherited from `Backing`.

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type (default-constructible) |
| `Backing` | the persistent array storing the elements (default `persistent_array<T>`); must satisfy `detail::persistent_ring_backing`: copyable, constructible from a capacity, `size()`, a version-returning `set(i, v)`, and `b[i]` returning `T const&` into the backing (the references `front`/`back`/`[]` return) |

#### Member functions

| function | description | complexity |
|---|---|---|
| `persistent_deque()` | an empty deque of capacity 0 — a placeholder (e.g. a member) to assign a sized deque to later | O(1) |
| `persistent_deque(std::size_t capacity)` | an empty deque holding up to `capacity` elements at once | O(capacity) |
| `size()` → `std::size_t` | number of elements | O(1) |
| `empty()` → `bool` | whether `size() == 0` | O(1) |
| `capacity()` → `std::size_t` | maximum elements held at once | O(1) |
| `front()` → `T const&` | the front element (precond: non-empty) | O(log capacity) |
| `back()` → `T const&` | the back element (precond: non-empty) | O(log capacity) |
| `operator[](std::size_t i)` → `T const&` | element `i`, 0 being the front (precond: `i < size()`) | O(log capacity) |
| `reserve(std::size_t pushes)` | pre-grow the backing's arena for that many further pushes (when `Backing` has `reserve`) | O(pushes) |
| `push_back(T value)` → `persistent_deque` | a new version with `value` at the back (precond: `size() < capacity()`) | O(log capacity) |
| `push_front(T value)` → `persistent_deque` | a new version with `value` at the front (precond: `size() < capacity()`) | O(log capacity) |
| `emplace_back(Args&&... args)` → `persistent_deque` | `push_back` of an element built in place from `args` (precond: `size() < capacity()`) | O(log capacity) |
| `emplace_front(Args&&... args)` → `persistent_deque` | `push_front` of an element built in place from `args` (precond: `size() < capacity()`) | O(log capacity) |
| `pop_back()` → `persistent_deque` | a new version with the back removed (precond: non-empty) | O(1) |
| `pop_front()` → `persistent_deque` | a new version with the front removed (precond: non-empty) | O(1) |

The mutators are `[[nodiscard]]` — they return the new version rather than mutating.

> **Note.** Preconditions (`front`/`back`/`pop*` on an empty container, `push*`/
> `emplace*` past `capacity()`) are checked by `atl_assert` only under
> `-DATL_ASSERT`.

Versions use shared pool ownership and are runtime-only (`std::shared_ptr`).
Copying a version costs O(1). Version-returning members have `const &` and `&&`
overloads: the latter consumes the handle; the moved-from object may only be
destroyed or assigned to. Allocation failure propagates an exception; mutations
are not transactional if allocation, element operations or callables throw.

## Complexity

Queue push/pop/back take O(1); front takes O(min(size, log pushes)), where pushes
counts all values pushed on the lineage. Deque reads and pushes cost the backing's
indexed access/set (O(log capacity) for persistent_array); pops cost O(1).
Storage is all backing nodes allocated across versions. A queue retains popped
values, and a deque pop changes only indices, retaining its backing values.

## Preconditions

Asserted under `ATL_ASSERT`: nonempty front/back/pop; valid indices;
size < capacity on deque pushes. `T` is semiregular. The deque backing must
be copyable, constructible from `size_t`, provide size(), set returning a new
backing, and indexed access returning `T const&`. These expressions are
constrained; backing persistence and reference lifetimes are documented laws.

## Measured performance

Not measured yet.

## Pitfalls and notes

A zero-capacity deque is a placeholder; assign a sized deque before pushing.
Read references have the backing's invalidation rules. Queue histories retain
the entire list of pushes, including elements removed from the logical queue.

## How it works

The queue prepends each pushed value to a skew-binary list and tracks how many
values have been popped. Its live front is at index size()-1 and its back at
index zero. Pop only increments a counter. The deque keeps a head and count
on a persistent circular array; pushes replace a slot, pops change indices.
Circular index arithmetic subtracts before adding when the sum would wrap.

## See also

- [Persistent list](persistent-list.md) — front-oriented sequence and stack.
- [Persistent array](persistent-array.md) — default deque backing.
