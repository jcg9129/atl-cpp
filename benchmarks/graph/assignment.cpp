/**
 * @file
 * @brief Complete-operation latency for dense assignment: library Jonker-Volgenant,
 * SAP without preprocessing, exact epsilon auction and min-cost-flow reduction.
 * Separate loops isolate column-major/row-major reduction, push-back/row-insert
 * flattening and row-order/tiled transpose. Shapes cover uniform costs, ties,
 * product-modulo, banded, Monge, wide and tall matrices. Each sample includes
 * construction of the solver's working storage; contenders agree on objective.
 * Intel Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled,
 * 2026-10-15: -O2 -g -DNDEBUG -std=c++23, also with -march=native.
 * At 500 squared, uniform: library 2.8 ms versus SAP 7.1, auction 7.7,
 * flow reduction 33.7; ties favor SAP (0.74 versus 0.92 ms).
 * Full tables: docs/assignment.md. Minima of three per-run means, ten samples.
 */
#include <atl/experimental/graph/assignment_variants.hpp>
#include <atl/graph/assignment.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;

namespace {
	/** @brief Times one complete operation, including its local allocation. */
	template<typename Solver>
		requires std::invocable<Solver &>
	void measure(std::string const &name, Solver solve) {
		BENCHMARK(std::string(name)) { return std::invoke(solve); };
	}

	/** @brief Copies rows with one vector insert per row. */
	template<std::integral C>
	std::vector<C> flatten_by_row_insert(std::vector<std::vector<C>> const &m) {
		std::vector<C> f;
		for (auto const &row : m) f.insert(f.end(), row.begin(), row.end());
		return f;
	}

	/** @brief Collects column minima by row-major or column-major scans. */
	template<bool is_row_major, std::integral C>
	atl::uint64 column_reduction_by_scan_order(std::size_t n, std::vector<C> const &cost) {
		std::vector<int64> v(n);
		std::vector<uint32> row_to_col(n, g::none), col_to_row(n, g::none), arg(n, 0);
		std::vector<atl::uint8> once(n, 0);
		auto const at = [&](std::size_t i, std::size_t j) { return int64(cost[i * n + j]); };
		if constexpr (is_row_major) {
			for (std::size_t j = 0; j < n; ++j) v[j] = at(0, j);
			for (std::size_t i = 1; i < n; ++i)
				for (std::size_t j = 0; j < n; ++j)
					if (at(i, j) < v[j]) {
						v[j] = at(i, j);
						arg[j] = uint32(i);
					}
		}
		for (std::size_t j = n; j-- > 0;) {
			std::size_t best = 0;
			if constexpr (is_row_major) {
				best = arg[j];
			} else {
				for (std::size_t i = 1; i < n; ++i)
					if (at(i, j) < at(best, j)) best = i;
				v[j] = at(best, j);
			}
			if (row_to_col[best] == g::none) {
				row_to_col[best] = uint32(j);
				col_to_row[j] = uint32(best);
				once[best] = 1;
			} else {
				once[best] = 0;
			}
		}
		atl::uint64 h = 0;
		for (std::size_t j = 0; j < n; ++j) h ^= atl::uint64(v[j]) + col_to_row[j] + once[j];
		return h;
	}

	/** @brief Copies the matrix by row insert or per-element push_back. */
	template<bool is_row_insert, std::integral C>
	std::size_t flatten_by_row_copy(std::vector<std::vector<C>> const &m) {
		std::vector<C> flat;
		flat.reserve(m.size() * m[0].size());
		for (auto const &row : m)
			if constexpr (is_row_insert)
				flat.insert(flat.end(), std::ranges::begin(row), std::ranges::end(row));
			else
				for (auto const &x : row) flat.push_back(x);
		return std::size_t(flat[flat.size() / 2]) + flat.size();
	}

	/** @brief Scatters rows directly or in 32-by-32 tiles. */
	template<bool is_blocked, std::integral C>
	std::size_t transpose_by_tile_size(std::vector<std::vector<C>> const &m) {
		std::size_t const rows = m.size(), cols = m[0].size();
		std::vector<C> flat(rows * cols);
		if constexpr (is_blocked) {
			std::size_t constexpr block = 32;
			for (std::size_t i0 = 0; i0 < rows; i0 += std::min(block, rows - i0))
				for (std::size_t j0 = 0; j0 < cols; j0 += std::min(block, cols - j0))
					for (std::size_t i = i0; i < i0 + std::min(block, rows - i0); ++i)
						for (std::size_t j = j0; j < j0 + std::min(block, cols - j0); ++j) flat[j * rows + i] = m[i][j];
		} else {
			for (std::size_t i = 0; i < rows; ++i)
				for (std::size_t j = 0; j < cols; ++j) flat[j * rows + i] = m[i][j];
		}
		return std::size_t(flat[flat.size() / 2]) + flat.size();
	}

	template<std::integral C>
	void pass_rows(std::string const &name, std::vector<std::vector<C>> const &m) {
		auto const f = flatten_by_row_insert(m);
		std::size_t const n = m.size();
		REQUIRE(column_reduction_by_scan_order<false>(n, f) == column_reduction_by_scan_order<true>(n, f));
		REQUIRE(flatten_by_row_copy<false>(m) == flatten_by_row_copy<true>(m));
		measure(name + " | column reduction by column major", [&] { return column_reduction_by_scan_order<false>(n, f); });
		measure(name + " | column reduction by row major", [&] {
			return column_reduction_by_scan_order<true>(n, f);
		});
		measure(name + " | flatten by push back", [&] { return flatten_by_row_copy<false>(m); });
		measure(name + " | flatten by row insert", [&] { return flatten_by_row_copy<true>(m); });
	}
}

TEST_CASE("assignment contenders", "[assignment]") {
	std::mt19937 rng(1);

	/** @brief Encodes unit row supplies and column demands for the flow solver. */
	auto const assignment_by_min_cost_flow = [](std::vector<std::vector<int64>> const &cost) {
		std::size_t const n = cost.size();
		std::vector<int64> b(2 * n);
		for (std::size_t i = 0; i < n; ++i) b[i] = 1;
		for (std::size_t j = 0; j < n; ++j) b[n + j] = -1;
		std::vector<g::flow_arc<int64, int64>> arcs;
		arcs.reserve(n * n);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) arcs.push_back({i, uint32(n + j), 1, cost[i][j]});
		return g::min_cost_feasible_flow(2 * n, arcs, b)->cost;
	};

	auto race = [&](char const *name, std::vector<std::vector<int64>> cost, bool square = true) {
		auto const want = g::assignment(cost).cost;
		REQUIRE(want == ex::assignment_by_shortest_augmenting_paths(cost).cost);
		if (square) {
			REQUIRE(want == ex::assignment_by_auction(cost).cost);
			REQUIRE(want == assignment_by_min_cost_flow(cost));
		}

		measure(std::string(name) + " | library", [&] { return g::assignment(cost).cost; });
		if (square) pass_rows(name, cost);
		measure(std::string(name) + " | by shortest augmenting paths", [&] {
			return ex::assignment_by_shortest_augmenting_paths(cost).cost;
		});
		if (square) {
			measure(std::string(name) + " | by auction", [&] {
				return ex::assignment_by_auction(cost).cost;
			});
			measure(std::string(name) + " | by min cost flow reduction", [&] {
				return assignment_by_min_cost_flow(cost);
			});
		}
	};

	std::size_t const n = 500;
	{
		std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
		for (auto &row : cost)
			for (auto &x : row) x = int64(rng() % 1'000'000'000);
		race("uniform 1e9 500x500", std::move(cost));
	}
	{
		std::vector<std::vector<atl::int32>> cost(n, std::vector<atl::int32>(n));
		for (auto &row : cost)
			for (auto &x : row) x = atl::int32(rng() % 1'000'000'000);
		measure("uniform 1e9 int32 500x500 | library", [&] {
			return g::assignment<int64>(cost).cost;
		});
	}
	{
		std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < n; ++j)
				cost[i][j] = int64(i) * int64(n - j) + int64(rng() % 3) * 0;
		measure("monge 500x500 | library", [&] { return g::assignment(cost).cost; });
	}
	{
		std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
		for (auto &row : cost)
			for (auto &x : row) x = int64(rng() % 10);
		race("uniform 0..9 500x500", std::move(cost));
	}
	{
		std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < n; ++j) cost[i][j] = int64((i + 1) * (j + 1) % 4096);
		race("product mod 4096 500x500", std::move(cost));
	}
	{
		std::vector<std::vector<int64>> cost(n, std::vector<int64>(n));
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < n; ++j)
				cost[i][j] = int64(i < j ? j - i : i - j) * 1000 + int64(rng() % 100);
		race("banded distance 500x500", std::move(cost));
	}
	{
		std::vector<std::vector<int64>> cost(100, std::vector<int64>(5000));
		for (auto &row : cost)
			for (auto &x : row) x = int64(rng() % 1'000'000'000);
		race("rectangular 100x5000", std::move(cost), false);
	}
	{
		std::vector<std::vector<int64>> tall(5000, std::vector<int64>(100));
		for (auto &row : tall)
			for (auto &x : row) x = int64(rng() % 1'000'000'000);
		std::vector<std::vector<int64>> wide(100, std::vector<int64>(5000));
		for (std::size_t i = 0; i < 5000; ++i)
			for (std::size_t j = 0; j < 100; ++j) wide[j][i] = tall[i][j];
		REQUIRE(g::assignment(tall).cost == g::assignment(wide).cost);
		REQUIRE(transpose_by_tile_size<false>(tall) == transpose_by_tile_size<true>(tall));
		measure("tall 5000x100 | library", [&] { return g::assignment(tall).cost; });
		measure("tall 5000x100 as wide 100x5000 | library", [&] { return g::assignment(wide).cost; });
		measure("tall 5000x100 | transpose by row order", [&] { return transpose_by_tile_size<false>(tall); });
		measure("tall 5000x100 | transpose by tiles", [&] { return transpose_by_tile_size<true>(tall); });
		std::vector<std::vector<atl::int32>> tall32(2000, std::vector<atl::int32>(1000));
		for (auto &row : tall32)
			for (auto &x : row) x = atl::int32(rng() % 1'000'000'000);
		std::vector<std::vector<int64>> tall64(2000, std::vector<int64>(1000));
		for (std::size_t i = 0; i < 2000; ++i)
			for (std::size_t j = 0; j < 1000; ++j) tall64[i][j] = tall32[i][j];
		REQUIRE(transpose_by_tile_size<false>(tall32) == transpose_by_tile_size<true>(tall32));
		REQUIRE(transpose_by_tile_size<false>(tall64) == transpose_by_tile_size<true>(tall64));
		measure("tall 2000x1000 int32 | library", [&] { return g::assignment<int64>(tall32).cost; });
		measure("tall 2000x1000 int32 | transpose by row order", [&] { return transpose_by_tile_size<false>(tall32); });
		measure("tall 2000x1000 int32 | transpose by tiles", [&] { return transpose_by_tile_size<true>(tall32); });
		measure("tall 2000x1000 int64 | transpose by row order", [&] { return transpose_by_tile_size<false>(tall64); });
		measure("tall 2000x1000 int64 | transpose by tiles", [&] { return transpose_by_tile_size<true>(tall64); });
	}
	{
		for (std::size_t const size : {std::size_t(500), std::size_t(2000), std::size_t(4000)}) {
			std::vector<std::vector<int64>> cost(size, std::vector<int64>(size));
			for (auto &row : cost)
				for (auto &x : row) x = int64(rng() % 1'000'000'000);
			std::string const tag = " " + std::to_string(size) + "x" + std::to_string(size);
			pass_rows("uniform 1e9 int64" + tag, cost);
			if (size == 2000) measure("uniform 1e9 int64" + tag + " | library", [&] {
				return g::assignment(cost).cost;
			});
			std::vector<std::vector<atl::int32>> c32(size, std::vector<atl::int32>(size));
			for (std::size_t i = 0; i < size; ++i)
				for (std::size_t j = 0; j < size; ++j) c32[i][j] = atl::int32(cost[i][j]);
			pass_rows("uniform 1e9 int32" + tag, c32);
		}
		std::vector<std::vector<int64>> monge(3000, std::vector<int64>(3000));
		for (std::size_t i = 0; i < 3000; ++i)
			for (std::size_t j = 0; j < 3000; ++j) monge[i][j] = int64(i) * int64(3000 - j);
		measure("monge 3000x3000 | library", [&] { return g::assignment(monge).cost; });
		measure("monge 3000x3000 | flatten by push back", [&] { return flatten_by_row_copy<false>(monge); });
		measure("monge 3000x3000 | flatten by row insert", [&] { return flatten_by_row_copy<true>(monge); });
	}
}
