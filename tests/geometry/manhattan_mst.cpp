#include <atl/ds/dsu.hpp>
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/experimental/geometry/manhattan_mst_variants.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/manhattan_mst.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::manhattan_mst;
using atl::geometry::point;

namespace {
	using lp = point<std::int64_t>;

	int128 l1(lp const a, lp const b) {
		int128 const dx = int128(a.x) - int128(b.x);
		int128 const dy = int128(a.y) - int128(b.y);
		return (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
	}

	int128 tree_weight(std::vector<lp> const &ps,
					   std::vector<std::array<atl::uint32, 2>> const &edges) {
		int128 total = 0;
		for (auto const &e : edges) total += l1(ps[e[0]], ps[e[1]]);
		return total;
	}

	// O(n^2) Prim on the complete L1 graph — the independent weight oracle
	int128 prim_weight(std::vector<lp> const &ps) {
		std::size_t const n = ps.size();
		int128 const inf = int128(1) << 100; // above any L1 distance
		std::vector<int128> best(n, inf);
		std::vector<char> used(n, 0);
		best[0] = 0;
		int128 total = 0;
		for (std::size_t round = 0; round < n; ++round) {
			std::size_t at = n;
			for (std::size_t i = 0; i < n; ++i)
				if (!used[i] && (at == n || best[i] < best[at])) at = i;
			used[at] = 1;
			total += best[at];
			for (std::size_t i = 0; i < n; ++i)
				if (!used[i]) best[i] = (std::min)(best[i], l1(ps[at], ps[i]));
		}
		return total;
	}

	// spanning + acyclic + weight parity against the oracle
	void check_tree(std::vector<lp> const &ps) {
		auto const expected = prim_weight(ps);
		// the library and the raced Fenwick contender
		for (auto const &edges : {manhattan_mst(ps), atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(ps)}) {
			REQUIRE(edges.size() + 1 == ps.size());
			atl::dsu components(ps.size());
			for (auto const &e : edges) {
				REQUIRE(e[0] < ps.size());
				REQUIRE(e[1] < ps.size());
				REQUIRE(components.unite(e[0], e[1])); // acyclic, hence spanning
			}
			REQUIRE(tree_weight(ps, edges) == expected);
			for (std::size_t k = 1; k < edges.size(); ++k) // Kruskal order
				REQUIRE(l1(ps[edges[k - 1][0]], ps[edges[k - 1][1]]) <= l1(ps[edges[k][0]], ps[edges[k][1]]));
		}
	}
}

TEST_CASE("manhattan_mst: handcrafted", "[geometry]") {
	REQUIRE(manhattan_mst(std::vector<lp>{{3, 4}}).empty());

	auto const pair = manhattan_mst(std::vector<lp>{{0, 0}, {5, -2}});
	REQUIRE(pair.size() == 1);

	// collinear points must chain along the line: weight = span
	std::vector<lp> const collinear{{10, 1}, {0, 1}, {7, 1}, {3, 1}, {-2, 1}};
	REQUIRE(tree_weight(collinear, manhattan_mst(collinear)) == 12);

	// unit grid: any spanning tree of unit steps
	std::vector<lp> grid;
	for (std::int64_t x = 0; x < 4; ++x)
		for (std::int64_t y = 0; y < 3; ++y) grid.push_back({x, y});
	REQUIRE(tree_weight(grid, manhattan_mst(grid)) == std::int64_t(grid.size()) - 1);

	// duplicates connect for free
	std::vector<lp> const dup{{2, 2}, {2, 2}, {2, 2}, {9, -1}, {2, 2}};
	auto const dup_edges = manhattan_mst(dup);
	REQUIRE(dup_edges.size() == 4);
	REQUIRE(tree_weight(dup, dup_edges) == 10);

	// exactness at the FULL int64 range: no coordinate precondition
	std::int64_t const hi = (std::numeric_limits<std::int64_t>::max)();
	std::int64_t const lo = (std::numeric_limits<std::int64_t>::min)();
	std::vector<lp> const giant{{lo, lo}, {hi, hi}, {hi, lo}, {lo, hi}};
	int128 const span = int128(hi) - int128(lo);
	REQUIRE(tree_weight(giant, manhattan_mst(giant)) == 3 * span);
	REQUIRE(tree_weight(giant, atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(giant)) == 3 * span);
}

TEST_CASE("manhattan_mst: random vs the Prim oracle", "[geometry]") {
	auto seed = GENERATE(take(12, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// dense small grids maximize ties, duplicates and octant borders
	for (int it = 0; it < 120; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 40);
		std::int64_t const span = 1 + std::int64_t(rng() % 12);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng() % std::uint64_t(span)),
						  std::int64_t(rng() % std::uint64_t(span))});
		check_tree(ps);
	}
	// full-range coordinates exercise the widened sweep keys
	for (int it = 0; it < 30; ++it) {
		std::size_t const n = static_cast<std::size_t>(2 + rng() % 24);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng()), std::int64_t(rng())});
		check_tree(ps);
	}
}

TEST_CASE("manhattan_mst: symmetry invariance", "[geometry]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0),
										(std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 40; ++it) {
		std::size_t const n = static_cast<std::size_t>(2 + rng() % 30);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng() % 1000) - 500,
						  std::int64_t(rng() % 1000) - 500});
		int128 const base = tree_weight(ps, manhattan_mst(ps));
		auto transformed = ps;
		for (auto &p : transformed) std::swap(p.x, p.y);
		REQUIRE(tree_weight(transformed, manhattan_mst(transformed)) == base);
		for (auto &p : transformed) p.x = -p.x;
		REQUIRE(tree_weight(transformed, manhattan_mst(transformed)) == base);
		for (auto &p : transformed) p += {12345, -777};
		REQUIRE(tree_weight(transformed, manhattan_mst(transformed)) == base);
	}
}

TEST_CASE("manhattan_mst: double coordinates parity spot check", "[geometry]") {
	using pd = point<double>;
	std::vector<pd> const ps{{0, 0}, {1.5, 0}, {1.5, 2.25}, {-3, 0.5}};
	auto const edges = manhattan_mst(ps);
	REQUIRE(edges.size() == 3);
	double total = 0;
	for (auto const &e : edges)
		total += std::abs(ps[e[0]].x - ps[e[1]].x) + std::abs(ps[e[0]].y - ps[e[1]].y);
	REQUIRE(total == 1.5 + 2.25 + 3.5);
}

TEST_CASE("manhattan_mst: 1e6 collinear points", "[geometry]") {
	constexpr std::int64_t n = 1'000'000;
	std::vector<lp> line;
	for (std::int64_t k = 0; k < n; ++k) line.push_back({k * 3 - n, 7});
	std::mt19937_64 rng(5);
	std::ranges::shuffle(line, rng);
	auto const edges = manhattan_mst(line);
	REQUIRE(edges.size() + 1 == line.size());
	REQUIRE(tree_weight(line, edges) == 3 * (n - 1));
}

TEST_CASE("manhattan_mst: narrow and floating coordinate types", "[geometry]") {
	using p8 = point<atl::int16>;
	std::vector<p8> const corners{{-128, -128}, {127, 127}, {-128, 127}, {127, -128}, {0, 0}, {-128, -128}};
	std::vector<lp> wide;
	for (auto const q : corners) wide.push_back({q.x, q.y});
	int128 total = 0;
	for (auto const &e : manhattan_mst(corners)) total += l1(wide[e[0]], wide[e[1]]);
	REQUIRE(total == prim_weight(wide));
	int128 contender = 0;
	for (auto const &e : atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(corners)) contender += l1(wide[e[0]], wide[e[1]]);
	REQUIRE(contender == total);
	std::vector<point<float>> const fs{{0.5f, 0.25f}, {1.5f, 0.25f}, {0.5f, 2.0f}};
	REQUIRE(manhattan_mst(fs).size() == 2);
}

TEST_CASE("manhattan_mst: an empty input is asserted", "[geometry]") {
	CHECK(atl_test::aborts([] { (void)manhattan_mst(std::vector<lp>{}); }));
}

TEST_CASE("manhattan_mst: invalid floating keys assert before sorting", "[geometry]") {
	std::vector<point<double>> const points{{0, 0}, {1, 0}, {4, 0}, {1e308, 1e308}, {1e308, 1e308}};
	CHECK(atl_test::aborts([&] { (void)atl::geometry::manhattan_mst(points); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(points); }));
	std::vector<point<double>> const nan{{0, 0}, {std::numeric_limits<double>::quiet_NaN(), 0}};
	CHECK(atl_test::aborts([&] { (void)atl::geometry::manhattan_mst(nan); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(nan); }));
}

namespace geometry_point_set_accepted_values {
	/**
	 * @brief A user sign policy: exact signs through the library policy, its
	 *        scalar `sign` a template over any ordered value, as the concept
	 *        requires.
	 */
	struct counting_policy {
		int offset = 0;

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int orientation(atl::geometry::point<T> const o, atl::geometry::point<T> const a,
												atl::geometry::point<T> const b) const {
			return atl::geometry::exact{}.orientation(o, a, b) + offset;
		}

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int cross_sign(atl::geometry::point<T> const a, atl::geometry::point<T> const b,
											   atl::geometry::point<T> const c, atl::geometry::point<T> const d) const {
			return atl::geometry::exact{}.cross_sign(a, b, c, d) + offset;
		}

		template<typename V>
		[[nodiscard]] constexpr int sign(V const v) const {
			return atl::geometry::exact{}.sign(v) + offset;
		}

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr bool equal(atl::geometry::point<T> const a, atl::geometry::point<T> const b) const {
			return a == b;
		}
	};

	/** @brief A policy deriving from the library one, overriding nothing. */
	struct derived_policy : atl::geometry::exact {};
}

namespace {
	/** @brief Every polygon function and the point-set algorithms on coordinate type `T`. */
	template<atl::geometry::coordinate T, atl::geometry::sign_policy<T> Policy>
	void exercise_polygons_and_point_sets(Policy const policy) {
		std::vector<point<T>> const square{{T(0), T(0)}, {T(4), T(0)}, {T(4), T(4)}, {T(0), T(4)}};
		REQUIRE(atl::geometry::polygon_area2(square) == atl::geometry::difference_product_t<T>(32));
		REQUIRE(atl::geometry::polygon_area(square) == 16.0);
		REQUIRE(atl::geometry::contains(square, point<T>{T(1), T(1)}, policy) == atl::geometry::containment::inside);
		REQUIRE(atl::geometry::winding_of(square, policy) == atl::geometry::winding::counterclockwise);
		REQUIRE(atl::geometry::convexity_of(square, policy).is_convex);
		REQUIRE(atl::geometry::normalized_polygon(square, policy) == square);
		REQUIRE(atl::geometry::polygon_centroid(square) == point<double>{2.0, 2.0});
		if constexpr (std::signed_integral<T>) REQUIRE(atl::geometry::lattice_count_of(square).interior == 9);
		std::vector<point<T>> const cloud{{T(0), T(0)}, {T(5), T(1)}, {T(1), T(1)}, {T(9), T(9)}};
		auto const pair = atl::geometry::closest_pair_indices(cloud);
		REQUIRE(pair[0] + pair[1] == 2);
		REQUIRE(atl::geometry::closest_pair(cloud)[0] != atl::geometry::closest_pair(cloud)[1]);
		REQUIRE(atl::experimental::closest_pair_indices_by_divide_and_conquer(cloud)[0] + pair[1] >= 0);
		REQUIRE(atl::experimental::closest_pair_indices_by_randomized_grid(cloud).size() == 2);
		REQUIRE(atl::geometry::manhattan_mst(cloud).size() == 3);
		REQUIRE(atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(cloud).size() == 3);
		REQUIRE(atl::geometry::max_points_in_circle(cloud, 1.0).count == 2);
	}
}

TEST_CASE("polygons and point sets: every coordinate type and policy form", "[geometry][accepted]") {
	exercise_polygons_and_point_sets<short>(atl::geometry::exact{});
	exercise_polygons_and_point_sets<atl::int16>(atl::geometry::fast{});
	exercise_polygons_and_point_sets<atl::int32>(atl::geometry::approx{0.5});
	exercise_polygons_and_point_sets<atl::int64>(geometry_point_set_accepted_values::counting_policy{});
	exercise_polygons_and_point_sets<long>(atl::geometry::exact{});
	exercise_polygons_and_point_sets<long long>(geometry_point_set_accepted_values::derived_policy{});
	exercise_polygons_and_point_sets<float>(atl::geometry::exact{});
	exercise_polygons_and_point_sets<double>(atl::geometry::fast{});
}
