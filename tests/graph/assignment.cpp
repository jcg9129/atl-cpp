#include <atl/experimental/graph/assignment_variants.hpp>
#include <atl/graph/assignment.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Matching validity + cost + the dual certificate (feasible everywhere,
	 * tight on matched, and on a rectangular matrix the larger side's
	 * potentials <= 0 with 0 on its unmatched vertices — an optimality proof
	 * for every shape by LP duality). */
	void certify(std::vector<std::vector<int64>> const &cost,
				 g::assignment_result<int64> const &r) {
		std::size_t const rows = cost.size(), cols = rows ? cost[0].size() : 0;
		std::vector<atl::uint8> used(cols, 0);
		int64 total = 0;
		std::size_t matched = 0;
		for (uint32 i = 0; i < rows; ++i) {
			uint32 const j = r.row_to_col[i];
			if (j == g::none) continue;
			REQUIRE(j < cols);
			REQUIRE(!used[j]);
			REQUIRE(r.col_to_row[j] == i);
			used[j] = 1;
			total += cost[i][j];
			++matched;
		}
		REQUIRE(matched == std::min(rows, cols));
		REQUIRE(total == r.cost);
		for (uint32 i = 0; i < rows; ++i)
			for (uint32 j = 0; j < cols; ++j) {
				REQUIRE(r.row_potential[i] + r.col_potential[j] <= cost[i][j]);
				if (r.row_to_col[i] == j)
					REQUIRE(r.row_potential[i] + r.col_potential[j] == cost[i][j]);
			}
		if (rows < cols)
			for (uint32 j = 0; j < cols; ++j) {
				REQUIRE(r.col_potential[j] <= 0);
				if (r.col_to_row[j] == g::none) REQUIRE(r.col_potential[j] == 0);
			}
		if (rows > cols)
			for (uint32 i = 0; i < rows; ++i) {
				REQUIRE(r.row_potential[i] <= 0);
				if (r.row_to_col[i] == g::none) REQUIRE(r.row_potential[i] == 0);
			}
	}

	/** @brief The auction's contract: a valid matching whose re-summed cost is the
	 * reported one (its duals are eps-approximate, so no certificate). */
	void check_matching(std::vector<std::vector<int64>> const &cost, g::assignment_result<int64> const &r) {
		std::size_t const n = cost.size();
		REQUIRE(r.row_to_col.size() == n);
		REQUIRE(r.col_to_row.size() == n);
		int64 total = 0;
		for (uint32 i = 0; i < n; ++i) {
			uint32 const j = r.row_to_col[i];
			REQUIRE(j < n);
			REQUIRE(r.col_to_row[j] == i);
			total += cost[i][j];
		}
		REQUIRE(total == r.cost);
	}

	void cross_check(std::vector<std::vector<int64>> const &cost, int64 want) {
		std::size_t const rows = cost.size(), cols = rows ? cost[0].size() : 0;
		if (rows <= cols) {
			auto const h = ex::assignment_by_shortest_augmenting_paths(cost);
			REQUIRE(h.cost == want);
			certify(cost, h);
		}
		if (rows == cols) {
			auto const a = ex::assignment_by_auction(cost);
			REQUIRE(a.cost == want);
			check_matching(cost, a);
		}
	}

	int64 brute(std::vector<std::vector<int64>> const &cost) {
		std::size_t const rows = cost.size(), cols = rows ? cost[0].size() : 0;
		std::size_t const k = std::min(rows, cols);
		std::vector<uint32> perm(std::max(rows, cols));
		std::iota(perm.begin(), perm.end(), 0);
		int64 best = std::numeric_limits<int64>::max();
		do {
			int64 sum = 0;
			for (std::size_t i = 0; i < k; ++i)
				sum += rows <= cols ? cost[i][perm[i]] : cost[perm[i]][i];
			best = std::min(best, sum);
		} while (std::ranges::next_permutation(perm).found);
		return best;
	}
}

static_assert(g::cost_matrix<std::vector<std::vector<int>>>);
static_assert(g::cost_matrix<std::vector<std::array<long, 4>>>);
static_assert(!g::cost_matrix<std::vector<int>>);
static_assert(!g::cost_matrix<std::vector<std::vector<std::string>>>);

namespace {
	template<typename Matrix, typename Sum = void>
	concept assignable = requires(Matrix const &m) { g::assignment<Sum>(m); };

	/** @brief A row type with random access but no size. */
	struct unsized_row {
		int const *data;

		int const *begin() const { return data; }

		std::unreachable_sentinel_t end() const { return {}; }
	};
}

static_assert(g::assignment_sum<int> && g::assignment_sum<int64>);
static_assert(!g::assignment_sum<uint32> && !g::assignment_sum<double> && !g::assignment_sum<char>
			  && !g::assignment_sum<bool>);
static_assert(!assignable<std::vector<std::vector<uint32>>>);
static_assert(assignable<std::vector<std::vector<uint32>>, int64>);
static_assert(!assignable<std::vector<std::vector<double>>>);
static_assert(!assignable<std::vector<std::string>>);
static_assert(!g::cost_matrix<std::vector<unsized_row>>);

TEST_CASE("assignment: random matrices certified against brute", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const rows = 1 + std::size_t(rng() % 7), cols = 1 + std::size_t(rng() % 7);
	int64 const range = rng() % 2 ? 10 : 1000000000;
	std::vector<std::vector<int64>> cost(rows, std::vector<int64>(cols));
	for (auto &row : cost)
		for (auto &x : row) x = int64(rng() % range) - range / 3;

	auto const r = g::assignment(cost);
	certify(cost, r);
	REQUIRE(r.cost == brute(cost));
	cross_check(cost, r.cost);
}

namespace {
	template<typename Matrix>
	concept auction_accepts = requires(Matrix const &m) { ex::assignment_by_auction(m); };
}

static_assert(auction_accepts<std::vector<std::vector<int>>>);
static_assert(!auction_accepts<std::vector<std::vector<double>>>);

TEST_CASE("assignment: auction sums an int32 matrix in a wider Sum", "[graph]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 40);
	// 1e9-range costs: (n+1) * cost overflows int32 at once, so only Sum = int64 is defined
	std::vector<std::vector<atl::int32>> cost(n, std::vector<atl::int32>(n));
	for (auto &row : cost)
		for (auto &x : row) x = atl::int32(int64(rng() % 2'000'000'001) - 1'000'000'000);
	REQUIRE(ex::assignment_by_auction<int64>(cost).cost == g::assignment<int64>(cost).cost);
}

TEST_CASE("assignment: independent-engine cross-check at mid size", "[graph]") {
	auto seed = GENERATE(take(3, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 30 + std::size_t(rng() % 20);
	std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
	for (auto &row : cost)
		for (auto &x : row) x = int64(rng() % 100000) - 30000;

	auto const r = g::assignment(cost);
	certify(cost, r);
	std::vector<int64> b(2 * n);
	std::vector<g::flow_arc<int64, int64>> arcs;
	for (uint32 i = 0; i < n; ++i) b[i] = 1;
	for (uint32 j = 0; j < n; ++j) b[n + j] = -1;
	for (uint32 i = 0; i < n; ++i)
		for (uint32 j = 0; j < n; ++j) arcs.push_back({i, uint32(n + j), 1, cost[i][j]});
	REQUIRE(r.cost == g::min_cost_feasible_flow(2 * n, arcs, b)->cost);
	cross_check(cost, r.cost);
}

TEST_CASE("assignment: rectangular cross-check against min-cost flow", "[graph]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const short_side = 30 + std::size_t(rng() % 21), large = 60 + std::size_t(rng() % 41);
	bool const is_wide = rng() % 2 != 0;
	std::size_t const rows = is_wide ? short_side : large, cols = is_wide ? large : short_side;
	std::vector<std::vector<int64>> cost(rows, std::vector<int64>(cols));
	for (auto &row : cost)
		for (auto &x : row) x = int64(rng() % 100000) - 30000;

	auto const r = g::assignment(cost);
	certify(cost, r);
	// rows -> cols, plus a hub vertex on the larger side: the smaller
	// side ships one unit per vertex, each larger-side vertex takes at most one
	std::size_t const n = rows + cols + 1;
	uint32 const hub = uint32(rows + cols);
	std::vector<int64> b(n, 0);
	std::vector<g::flow_arc<int64, int64>> arcs;
	for (uint32 i = 0; i < rows; ++i)
		for (uint32 j = 0; j < cols; ++j) arcs.push_back({i, uint32(rows + j), 1, cost[i][j]});
	if (is_wide) {
		for (uint32 i = 0; i < rows; ++i) b[i] = 1;
		for (uint32 j = 0; j < cols; ++j) arcs.push_back({uint32(rows + j), hub, 1, 0});
		b[hub] = -int64(rows);
	} else {
		// every column is matched: rows draw from a super source instead
		for (uint32 j = 0; j < cols; ++j) b[rows + j] = -1;
		for (uint32 i = 0; i < rows; ++i) arcs.push_back({hub, i, 1, 0});
		b[hub] = int64(cols);
	}
	REQUIRE(r.cost == g::min_cost_feasible_flow(n, arcs, b)->cost);
	cross_check(cost, r.cost);
}

TEST_CASE("assignment: canonical shapes", "[graph]") {
	std::vector<std::vector<int64>> diag{{0, 9, 9}, {9, 0, 9}, {9, 9, 0}};
	auto const d = g::assignment(diag);
	REQUIRE(d.cost == 0);
	REQUIRE(d.row_to_col == std::vector<uint32>{0, 1, 2});
	certify(diag, d);
	cross_check(diag, d.cost);

	std::vector<std::vector<int64>> wide{{5, 1, 8, 2}, {5, 1, 9, 4}};
	auto const w = g::assignment(wide);
	REQUIRE(w.cost == 1 + 2);
	certify(wide, w);
	cross_check(wide, w.cost);

	std::vector<std::vector<int64>> tall{{5, 5}, {1, 1}, {8, 9}, {2, 4}};
	auto const t = g::assignment(tall);
	REQUIRE(t.cost == 1 + 2);
	REQUIRE(t.row_to_col[2] == g::none);
	certify(tall, t);
	cross_check(tall, t.cost);

	std::vector<std::vector<int64>> flat(4, std::vector<int64>(4, 7));
	auto const f = g::assignment(flat);
	REQUIRE(f.cost == 28);
	certify(flat, f);
	cross_check(flat, f.cost);

	std::vector<std::vector<int64>> neg{{-5, -1}, {-2, -6}};
	auto const ng = g::assignment(neg);
	REQUIRE(ng.cost == -11);
	certify(neg, ng);
	cross_check(neg, ng.cost);

	REQUIRE(g::assignment(std::vector<std::vector<int64>>{}).cost == 0);

	std::vector<std::vector<int64>> one{{-42}};
	auto const single = g::assignment(one);
	REQUIRE(single.cost == -42);
	certify(one, single);
	cross_check(one, -42);

	// Monge (c = i*|4-j| is submodular): diagonal shortcut, certificate intact
	std::vector<std::vector<int64>> monge(5, std::vector<int64>(5));
	for (std::size_t i = 0; i < 5; ++i)
		for (std::size_t j = 0; j < 5; ++j) monge[i][j] = int64(i) * int64(j < 4 ? 4 - j : j - 4);
	auto const mg = g::assignment(monge);
	certify(monge, mg);
	cross_check(monge, mg.cost);
	REQUIRE(mg.cost == brute(monge));

	std::vector<std::vector<atl::int32>> narrow{{1000000000, -1000000000},
												{-1000000000, 999999999}};
	REQUIRE(g::assignment<int64>(narrow).cost == -2000000000);
}

TEST_CASE("assignment arithmetic at a conservative documented bound", "[graph]") {
	int64 const magnitude = std::numeric_limits<int64>::max() / 32;
	std::vector<std::vector<int64>> const costs{{magnitude, -magnitude, magnitude / 2},
												{-magnitude / 2, magnitude, 0},
												{0, magnitude / 2, -magnitude}};
	auto const r = g::assignment(costs);
	certify(costs, r);
	REQUIRE(r.cost == brute(costs));
	std::vector<std::vector<int64>> const empty_columns(3);
	auto const empty = g::assignment(empty_columns);
	REQUIRE(empty.cost == 0);
	REQUIRE(empty.row_to_col == std::vector<uint32>(3, g::none));
}
