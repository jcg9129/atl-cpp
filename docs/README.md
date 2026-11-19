# Guide to atl {#atl_guides}

`atl` is a header-only C++23 library. Include a public component with
`#include <atl/…>`; the generated API reference describes its contracts and
constraints. The handwritten guides explain semantics, tradeoffs, examples
and measurements.

The site's **Related Pages** tree discovers all guide pages in this checkout.
The generated **Guide coverage** page lists every public header and the guides
that discuss it, with a complete guide index. Both grow automatically when
components and their pages arrive.

See [Library conventions](conventions.md) for contribution rules, including
where to put guide prose, executable examples and API comments.

- [128-bit integers](int128.md): native aliases, portable software integers and formatting.
- [Bit utilities](bit-utilities.md): lowest set bit and fixed-popcount enumeration.
- [XOR bases](xor-basis.md): reduced spans and stamp-filtered queries.
- [Fast I/O](fast-io.md): buffered readers and writers.
- [Flat hash containers](flat-hash.md): open-addressing `flat_hash_set` and `flat_hash_map`.
- [Static vector](static-vector.md): fixed-capacity `static_vector`.
- [Hierarchical bitset](hierarchical-bitset.md): `hierarchical_bitset` with fast successor/predecessor queries.
- [Coordinate compression](coordinate-compression.md): `coordinate_compression` and `compress`.
- [Disjoint sets](dsu.md): `dsu`, `rollback_dsu` and `persistent_dsu`.
- [Weighted disjoint sets](weighted-dsu.md): `weighted_dsu` and `rollback_weighted_dsu` with group potentials.
- [Persistent arrays](persistent-array.md): `persistent_array` and `semi_persistent_array`.
- [Time divide and conquer](time-divide-and-conquer.md): offline add/remove over time with `time_divide_and_conquer`.
- [Fenwick trees](fenwick-tree.md): point updates and prefix aggregates in one or more dimensions.
- [Difference Fenwick](difference-fenwick.md): commutative group range updates and point reads.
- [Sparse tables](sparse-table.md): constant-time folds of static data.
- [Range minimum queries](rmq.md): borrowed arrays and leftmost argmin.
- [Wavelet matrix](wavelet-matrix.md): static range order statistics.
- [Li Chao trees](li-chao-tree.md): line and segment envelopes over fixed or online domains.
- [Segment tree](segment-tree.md): point updates and range folds.
- [Lazy segment tree](lazy-segment-tree.md): range actions and range folds.
- [Dual segment tree](dual-segment-tree.md): range actions and point reads.
- [Segment tree beats](segment-tree-beats.md): conditional range actions.
- [Dynamic segment tree](dynamic-segment-tree.md): sparse point updates and folds.
- [Dynamic lazy segment tree](dynamic-lazy-segment-tree.md): sparse range actions and folds.
- [Treap map](treap-map.md): ordered keys, order statistics and value folds.
- [Treap sequence](treap-sequence.md): positional surgery, folds, lazy tags and reversal.
- [Splay engine](splay-engine.md): structural algorithms over caller-owned nodes.
- [Splay map](splay-map.md): self-adjusting ordered keys and value folds.
- [Splay sequence](splay-sequence.md): self-adjusting positional surgery and range operations.
- [Link-cut tree](link-cut-tree.md): dynamic path folds and lazy updates.
- [Subtree link-cut tree](subtree-link-cut-tree.md): commutative group subtree folds.
- [Euler tour tree](euler-tour-tree.md): dynamic component and edge-side updates.
- [Top tree](top-tree.md): cluster algebras for mixed path and subtree queries.
- [Meldable heaps](meldable-heap-forest.md): pool-backed priority queues, lazy shifts and handles.
- [Persistent heaps](persistent-heap.md): branching meldable priority queues.
- [Persistent lists](persistent-list.md): front operations and random access.
- [Persistent queue and deque](persistent-queue-deque.md): FIFO and fixed-capacity circular adapters.
- [Persistent vectors](persistent-vector.md): full and partial persistence with indexed growth.
- [Persistent segment tree](persistent-segment-tree.md): point updates, folds and prefix-version descent.
- [Persistent lazy segment tree](persistent-lazy-segment-tree.md): range actions and cross-version range copies.
- [Persistent treap sequence](persistent-treap-sequence.md): branching sequence surgery and folds.
- [Extended gcd](gcd.md): gcds with signed Bézout coefficients.
- [Power](power.md): binary exponentiation and explicit monoid identities.
- [Modular arithmetic](modular-arithmetic.md): modular values, reducers and engines.
- [Sieve](sieve.md): prime lists, factorizations and divisors within a fixed bound.
- [Multiplicative functions](multiplicative-functions.md): totient and divisor tables from prime-power rules.
- [Factorization](factorization.md): deterministic primality and Pollard rho on uint64.
- [CRT](crt.md): linear congruences, class intersections and Garner reconstruction.
- [Discrete logarithms](discrete-log.md): least exponents, unit orders and primitive roots.
- [Floor sums](floor-sum.md): Euclidean sums, staircase products and first residues.
- [Convolution and transforms](convolution.md): modular and exact integer polynomial products, NTT engines and complex FFT.
- [Bitwise convolutions](bitwise-convolution.md): subset transforms and OR, AND, XOR and disjoint-subset products.
- [Matrices and elimination](matrix.md): semiring products, field systems and packed XOR systems.
- [Linear recurrences](linear-recurrence.md): shortest recurrences and distant terms over exact fields.
- [Formal power series](power-series.md): reciprocal, division, logarithm, exponential, powers, roots and polynomial division over prime fields, with factorial tables, batch inverses and modular square roots.
- [Polynomial points](polynomial-points.md): product trees, multipoint evaluation, interpolation, Taylor shifts, chirp-z and sample shifts over prime fields.
- [Series composition](series-composition.md): composition, power projection and compositional inverse of power series over prime fields.
- [Polynomial gcd](polynomial-gcd.md): gcd, extended gcd, modular inverses and Pade approximants of polynomials over prime fields.
- [Online convolution](online-convolution.md): relaxed products of coefficient streams, with feedback, over prime fields.
