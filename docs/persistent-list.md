# `atl::persistent_list`

A persistent front-oriented sequence with O(1) push/pop at the front and O(log n) indexing. Use it for a persistent stack or front growth.

```cpp
#include <atl/ds/persistent_list.hpp>
```

\include persistent-list/basic.cpp

## API

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type |

### Member functions

| function | description | complexity |
|---|---|---|
| `persistent_list()` | empty list | O(1) |
| `persistent_list(Range&& values)` | build from a sized range, preserving order (index 0 is `values[0]`); moves out of an owning rvalue range only, copies from an rvalue span/view | O(n) |
| `persistent_list(std::size_t n, T value = {})` | `n` copies of `value` | O(n) |
| `size()` → `std::size_t` | number of elements | O(1) |
| `empty()` → `bool` | whether `size() == 0` | O(1) |
| `operator[](std::size_t i)` → `T const&` | element at `i` (0 is the front) | O(log n) |
| `front()` → `T const&` | the front element (precond: non-empty) | O(1) |
| `begin()` / `end()` | allocation-free forward iteration front to back (spine, then each tree in preorder, fixed 32-entry stack) — a full read in O(n) instead of n O(log n) lookups | O(1) per step amortised |
| `reserve(std::size_t pushes)` | pre-grow the arenas for that many further `push_front`s | O(shared pool entries) if relocated |
| `push_front(T value)` → `persistent_list` | a new version with `value` prepended | O(1) |
| `pop_front()` → `persistent_list` | a new version with the front removed (precond: non-empty) | O(1) |
| `set(std::size_t i, T value)` → `persistent_list` | a new version with `[i] = value` | O(log n) |
| `emplace_front(Args&&...)` → `persistent_list` | `push_front` of a `T` constructed in place from `args` | O(1) |

Versions use shared pool ownership and are runtime-only (`std::shared_ptr`).
Copying a version costs O(1). Version-returning members have `const &` and `&&`
overloads: the latter consumes the handle; the moved-from object may only be
destroyed or assigned to. Allocation failure propagates an exception; mutations
are not transactional if allocation, element operations or callables throw.

## Complexity

End operations make O(1) new nodes; indexing and set take O(log n). Build and full traversal take O(n). Arena memory is the total allocated by every version.

## Preconditions

Asserted under `ATL_ASSERT`: index < size; nonempty front/pop; size <= 2^32-1
before building and before growth. Pool entries must fit 32-bit ids and
allocation limits. The element type satisfies `std::semiregular`.
Range inputs are sized; lvalues and views are copied even when the view is
an rvalue, while owning rvalues and yielded prvalues may move.

## Measured performance

`benchmarks/ds/persistent_vector.cpp` compares one million chained front pushes
by copying the shared pool handle with moving that handle. Both keep appended
nodes in the shared arena and construct a fresh list for each sample.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| method | time |
|---|---:|
| by copied version handle (`v = v.push_front(x)`) | 5.1 / 6.8 |
| by moved version handle (`v = std::move(v).push_front(x)`) | 4.1 / 4 |

## Pitfalls and notes

References and iterators into a pool can be invalidated by mutations or reserve on any version sharing it. Copy values that must survive an update.

## How it works

A skew-binary spine stores complete trees of sizes 2^k - 1. Equal leading
trees combine below a new front node; popping splits that tree into its two
children. At most one such combine/split happens, giving O(1) end operations.
Indexing skips spine trees and then follows subtree sizes. `set` copies that
path and preceding spine cells. An iterator uses a fixed 32-entry traversal
stack, sufficient because stored sizes are 32-bit. Separate tree/spine arenas
append nodes and retain them until the shared pool dies.

## See also

- [`persistent_array`](persistent-array.md) — fixed-size persistent array
- [`persistent_vector`](persistent-vector.md) — growable persistent sequence
- [`persistent_queue`, `persistent_deque`](persistent-queue-deque.md) — persistent FIFO / double-ended queues
