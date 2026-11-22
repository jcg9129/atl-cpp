# `atl::geometry::halfplane_intersection`

Intersect closed linear inequalities in the plane, including empty, unbounded
and lower-dimensional answers. Use coefficient half-planes when you need an
exact classification and input-edge identities without choosing an artificial
bounding box. Corner coordinates are rounded constructions.

```cpp
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
```

\include halfplane-intersection/quick.cpp

Output:

\verbinclude halfplane-intersection/quick.out

## API

`T` models `coordinate`: non-character signed integers through 64 bits,
`float` or `double`. `halfplane_coefficient_t<T>` is `int64` for integral T,
`float64` otherwise. `halfplane<T>` is an aggregate with `coordinate_type`,
`coefficient_type`, and public coefficients `p`, `q`, `r`, representing
`p*x + q*y <= r`. Default coefficients are zero and require assignment prior
to geometric use. Equality compares stored coefficients, so scaled equivalent
constraints need not be equal.

| Signature | Meaning |
|---|---|
| `static constexpr halfplane left_of(point<T> a, point<T> b)` | Closed left side of directed a-to-b; floating coefficients are constructions |
| `static constexpr halfplane at_most(coefficient_type p, coefficient_type q, coefficient_type r)` | Construct and validate the coefficient inequality |
| `constexpr containment contains(point<T> v) const` | Exact residual sign: inside, boundary or outside |
| `constexpr line<float64> boundary() const` | Construct the supporting directed line, interior on its left |
| `constexpr containment contains(halfplane<T> const &h, point<T> v, Policy p = {})` | Free form; `Policy` models `sign_policy<T>`, default `exact` |
| `constexpr bool halfplane_sweep_less::operator()(halfplane<T> const &a, halfplane<T> const &b) const` | Exact CCW boundary-direction order from +x, equal directions equivalent |
| `constexpr halfplane_region halfplane_intersection(R const &halfplanes, Policy = {})` | Region classification, boundary input indices and constructed corners |

Result-only functions are `[[nodiscard]]`. `halfplane_range<R>` requires a
sized random-access range of half-planes readable through a const range,
including convertible element proxies. `halfplane_coordinate_t<R>` names its T.
For intersection, `Policy` models `halfplane_intersection_policy`, accepting
only `exact` and `fast`; `approx` is supported by free containment but not
intersection. Default policy is `exact`.

```cpp
struct halfplane_region {
    region_kind kind{region_kind::empty};
    std::vector<uint32> boundary;
    std::vector<point<float64>> vertices;
};
```

`region_kind` has `empty`, `point`, `segment`, `polygon`, `unbounded`.
The boundary layout is part of the contract:

- Polygon edges run CCW, starting at the first direction in the [0, 2 pi)
  sweep. Vertex i is the crossing of edges i-1 and i, cyclically.
- An unbounded chain runs with the region on its left, from the incoming
  infinite edge to the outgoing one; vertex i joins edges i and i+1.
  A single half-plane has one edge and no vertices. A strip or line has two
  antiparallel edges and no vertices. A ray has two coincident antiparallel
  edges and one apex. Empty input represents the whole plane with no edges.
- A segment has two antiparallel carrying constraints and two endpoints,
  oriented along the first constraint's direction.
- A point has two nonparallel carrying constraints and one vertex; the
  choice of carrying pair is unspecified.
- An empty intersection has no edges or vertices.

Equivalent input half-planes use their smallest input index. Zero-length
polygon edges are omitted. Input order is preserved externally; results own
their buffers and retain no views into the input.

## Complexity

Factories, containment, boundary construction and direction comparison take
O(1) time and space. Intersection takes O(n log n) time and O(n) extra memory.
Key sorting is linear apart from exact comparisons within equal-key groups;
thus worst-case time remains O(n log n), even on presorted input. Integral
input with up to eight ascending key runs uses rotation or pairwise merging.
Standard sorting uses O(log n) stack space; other passes are iterative.
Allocation failures propagate.

## Preconditions

Asserted under `ATL_ASSERT`: `(p,q) != (0,0)`; integral coefficients satisfy
`|p|, |q| < 2^31`, `|r| < 2^62`; floating coefficients are finite. Integer
`left_of` requires coordinates of magnitude below `2^30`; its endpoints must
be distinct. Intersection size fits `uint32` and `ptrdiff_t`. Boundary-line construction
alone accepts full-range int64 coefficients, widening their negation; it
requires a nonzero normal and the supporting line construction's finite
normal-product bounds.

Documented only: floating products and expansion intermediates must remain
finite and normal, with zero-factor products allowed; intermediate sums and
constructed corners must remain finite. This includes products of three
coefficients for determinant signs, pairwise products for ordering and
containment products involving the query point. Containment additionally requires
the absolute sum `|r| + |p x| + |q y|` to be finite; this sum and the rounded
residual are asserted under `ATL_ASSERT`. Finite products alone are insufficient
when their expansion sums overflow. Round-to-nearest and no unsafe
reassociation are required; the implementation does not rescale. Excess
precision uses the geometry predicate rounding barriers.

Integral normal minors have magnitude below `2^63`. Each offset-times-minor
term is below `2^125`, and their three-term sum below `3*2^125 < 2^127`,
so native and software `atl::int128` both give exact signs. Containment can
query every value of its integral coordinate type, including full-range int64.

The precision tiers are:

1. Integral coefficients have exact classification and boundary indices.
2. Floating coefficients under `exact` have exact decisions about the stored
   values. `left_of` constructs those coefficients; it need not preserve the
   true line through arbitrary floating points. Integer-valued floating points
   below `2^26` produce exact coefficients. `fast` uses rounded determinant
   signs and may misclassify nearly concurrent or parallel boundaries; its
   direction ordering remains exact.
3. Corners and supporting lines are constructions. Compare them with a chosen
   tolerance; never feed approximate equality into a sort. `contains(h,v,
   approx{eps})` applies a residual dead zone, in coefficient-residual units.

## Measured performance

`benchmarks/geometry/halfplane_intersection.cpp` measures throughput of complete
intersections, including sorting and allocation. The dispatch-size race compares
the library run detector with forced key sort and forced rotated-run sort on
circle tangents. Other rows vary the determinant filter, deque pass, buffer,
sort-key width and radix width, plus two finished circle hulls.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-11-02. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run means over three runs of 25 samples; ms.
Two runs are a circularly rotated sequence; eight and nine are ascending key
runs. Only these dispatch-size rows have supplied measurements.

| n | Key runs | Library -O2 / native | By key sort -O2 / native | By rotated runs -O2 / native |
|---:|---:|---:|---:|---:|
| 1000 | 2 | 0.066 / 0.069 | 0.070 / 0.073 | 0.066 / 0.069 |
| 1000 | 8 | 0.069 / 0.071 | 0.074 / 0.074 | 0.069 / 0.072 |
| 1000 | 9 | 0.072 / 0.074 | 0.071 / 0.075 | — |
| 10000 | 2 | 0.668 / 0.681 | 0.734 / 0.746 | 0.666 / 0.683 |
| 10000 | 8 | 0.674 / 0.694 | 0.727 / 0.735 | 0.677 / 0.692 |
| 10000 | 9 | 0.731 / 0.746 | 0.731 / 0.740 | — |
| 50000 | 2 | 3.4 / 3.6 | 3.8 / 3.9 | 3.4 / 3.6 |
| 50000 | 8 | 3.5 / 3.6 | 3.8 / 3.9 | 3.5 / 3.6 |
| 50000 | 9 | 3.8 / 3.9 | 3.8 / 3.9 | — |
| 500000 | 2 | 35.8 / 37.1 | 41.9 / 42.6 | 36.0 / 36.7 |
| 500000 | 8 | 37.7 / 38.3 | 42.0 / 42.6 | 37.9 / 38.1 |
| 500000 | 9 | 42.5 / 43.0 | 41.7 / 42.8 | — |

At 500000 constraints and two runs, dispatch saves 14.6% with plain -O2
((41.9 - 35.8) / 41.9) and 12.9% with native flags ((42.6 - 37.1) / 42.6).
The nine-run rows use the full-sort fallback.

## Pitfalls and notes

No artificial bounding box is introduced: unboundedness and degeneracies are
classified directly. A line or ray has kind `unbounded`, regardless of area.
Repeated calls with fixed normals can sort once with `halfplane_sweep_less`;
changing offsets preserves direction order, while equal-key groups still need
exact settlement. Shrinking an offset by radius times normal length constructs
an inscribed-circle feasibility problem, whose shifted coefficients are rounded.

`atl/experimental/geometry/halfplane_intersection_variants.hpp` provides
`halfplane_sort` selectors (`key`, `comparator`, `key_std_sort`, `key_radix_16`,
`key_64_std_sort`, `key_64_radix_16`, `rotated_runs`) and the constrained
`halfplane_intersection_by_composition<Determinant, Pass, Sort, UsesStdDeque = false>(R const &)`.
It changes one mechanism at a time: expansion-only or product-permanent
filtering, Dekker minors, every-front or envelope passes, cached integral
corners, standard deque storage and keyed sorts. All use the same result
layout. Rounded determinants carry `fast`'s limitations and are checked against
the public `fast` call: their boundary can differ from `exact` on tangent inputs. The Dekker contender
requires uncontracted split arithmetic; its implementation enforces rounding
stores. Standard-deque compositions are runtime only because that container
is not constexpr. These additional race rows are not measured yet in the
supplied publication record.

## How it works

An exact angular comparison settles ties within monotone slope keys, then
parallel constraints collapse to the tightest, with smallest-index ties. A
cyclic direction gap of at least pi identifies the unbounded-chain case,
built with a monotone stack. Otherwise a deque intersects sorted constraints.
Symbolically shrinking all offsets tests for a positive-area interior;
if that is empty, symbolic growth distinguishes empty, point and segment
limits without choosing a numeric epsilon. Every constraint is pushed and
popped only a constant number of times per pass. Integral decisions use
widened determinants. Floating decisions filter accurately evaluated minors
and use exact expansions when the error bound cannot certify a sign.
Corners are constructed from determinant ratios only at the end. The tests
compare kinds and boundaries with an independent rational all-pairs oracle,
including degenerate and coefficient-bound inputs. Reference: de Berg et al.
(2008), *Computational Geometry: Algorithms and Applications*. The
implementation is independent.

## See also

[Geometry overview](geometry.md), [Geometry primitives](geometry-primitives.md),
[Convex polygon operations](convex-polygon-operations.md).
