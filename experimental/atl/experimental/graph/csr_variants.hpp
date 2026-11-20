#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief CSR build passes that lost the race to the library `graph::csr`
 *        build (atl/graph/csr.hpp, which uses its offset table as the write
 *        cursor and shifts it once at the end).
 *
 * Shared contract: the raw arrays a `csr<Edge>` holds, built from an edge
 * list in stable input order per source; with `HasBothDirections` every
 * arc is stored both ways, as `csr::undirected` does. Cross-checked against
 * the library class in tests/graph/csr.cpp.
 */
namespace atl::experimental {
	/** @brief The two arrays of a CSR: `offset` (n + 1 row starts) and the flat `edges`. */
	template<graph::adjacency_edge Edge>
	struct csr_arrays {
		std::vector<uint32> offset; ///< Row starts, including the terminal edge count.
		std::vector<Edge> edges;	///< Stable adjacency elements grouped by source.
	};

	/**
	 * @brief Count and scatter with a cursor copy: count, prefix sum, then scatter
	 *        through a separate copy of the row starts.
	 *
	 * Lost to the in-place cursor, which saves the n-entry copy and its
	 * allocation: tied at n = 1e6 (arc<> directed 22.4 vs 22.6 ms, undirected
	 * 54.9 vs 56.4), behind at n = 1e7, m = 1e7 (directed 421 vs 399 ms,
	 * undirected 845 vs 823; native flags preserve this ordering) — x86-64 Xeon 8358,
	 * plain -O2, measured using benchmarks/graph/csr.cpp on 2026-10-15.
	 * @pre Endpoints are below `n`; `n + 1` fits `size_t`; vertex and stored-edge counts fit `uint32` (asserted).
	 * @note Not constexpr (runtime benchmark contender; constant evaluation has not been enabled).
	 * @complexity O(n + m) time and auxiliary storage
	 */
	template<graph::adjacency_edge Edge, bool HasBothDirections, graph::edge_list R>
		requires graph::detail::adjacent_buildable<Edge, std::ranges::range_value_t<R>>
			  && graph::detail::adjacent_buildable<Edge, std::remove_cvref_t<graph::detail::list_reference_t<R>>>
	[[nodiscard]] csr_arrays<Edge> csr_by_cursor_copy(std::size_t n, R const &arcs) {
		atl_assert(n <= graph::none && n < (std::numeric_limits<std::size_t>::max)());
		csr_arrays<Edge> out;
		out.offset.assign(n + 1, 0);
		[[maybe_unused]] std::size_t stored = 0;
		for (auto const &a : arcs) {
			atl_assert(graph::source(a) < n && graph::target(a) < n);
			atl_assert(stored <= (std::numeric_limits<uint32>::max)() - (HasBothDirections ? 2U : 1U));
			stored += HasBothDirections ? 2U : 1U;
			++out.offset[graph::source(a) + 1];
			if constexpr (HasBothDirections) ++out.offset[graph::target(a) + 1];
		}
		std::partial_sum(out.offset.begin(), out.offset.end(), out.offset.begin());
		out.edges.resize(out.offset.back());
		std::vector<uint32> cursor(out.offset.begin(), out.offset.end() - 1);
		for (auto const &a : arcs) {
			graph::vertex_id const s = graph::source(a), t = graph::target(a);
			out.edges[cursor[s]++] = graph::detail::adjacent_from<Edge>(a, t);
			if constexpr (HasBothDirections) out.edges[cursor[t]++] = graph::detail::adjacent_from<Edge>(a, s);
		}
		return out;
	}

	/**
	 * @brief The build on the library's bucket primitive
	 *        (`atl::detail::bucket_by_key`): bucket the stored-edge indices by
	 *        source, then gather the edges in bucket order. Needs a
	 *        random-access edge list; stored edge `k` is arc `k` (directed) or
	 *        direction `k % 2` of arc `k / 2` (both directions).
	 *
	 * Lost to the library direct scatter on every shape: the gather writes
	 * and rereads an m-entry index array the scatter never needs. n = 1e6,
	 * m = 2e6 arc<>: directed 35.8 vs 22.6 ms, undirected 84.7 vs 56.4;
	 * arc<int64>: 51.8 vs 43.4 and 143 vs 112; n = 1e7: 542 vs 399 and 1201
	 * vs 823 (x86-64 Xeon 8358, plain -O2, record
	 * benchmarks/graph/csr.cpp of 2026-10-15).
	 * @pre Endpoints are below `n`; `n + 1` fits `size_t`; vertex and stored-edge counts fit `uint32` (asserted).
	 * @note Not constexpr (runtime benchmark contender; constant evaluation has not been enabled).
	 * @complexity O(n + m) time and auxiliary storage
	 */
	template<graph::adjacency_edge Edge, bool HasBothDirections, graph::edge_list R>
		requires std::ranges::random_access_range<R const> && std::ranges::sized_range<R const>
			  && graph::detail::adjacent_buildable<Edge, std::ranges::range_value_t<R>>
			  && graph::detail::adjacent_buildable<Edge, std::remove_cvref_t<graph::detail::list_reference_t<R>>>
	[[nodiscard]] csr_arrays<Edge> csr_by_bucket_gather(std::size_t n, R const &arcs) {
		atl_assert(n <= graph::none && n < (std::numeric_limits<std::size_t>::max)());
		atl_assert(std::ranges::size(arcs) <= graph::none / (HasBothDirections ? 2U : 1U));
		atl_assert(std::ranges::all_of(arcs, [n]<graph::list_edge A>(A const &a) { return graph::source(a) < n && graph::target(a) < n; }));
		std::size_t const m = std::size_t(std::ranges::size(arcs)) * (HasBothDirections ? 2 : 1);
		atl_assert(n <= (std::numeric_limits<graph::vertex_id>::max)() && m <= (std::numeric_limits<uint32>::max)());
		auto const first = std::ranges::begin(arcs);
		csr_arrays<Edge> out;
		out.offset.resize(n + 1);
		std::vector<uint32> order(m);
		atl::detail::bucket_by_key(
			n,
			[&](std::size_t k) -> std::size_t {
				if constexpr (HasBothDirections) {
					auto const &a = first[std::ptrdiff_t(k / 2)];
					return k % 2 ? graph::target(a) : graph::source(a);
				} else {
					return graph::source(first[std::ptrdiff_t(k)]);
				}
			},
			out.offset, order);
		out.edges.reserve(m);
		for (uint32 const k : order) {
			if constexpr (HasBothDirections) {
				auto const &a = first[std::ptrdiff_t(k / 2)];
				out.edges.push_back(graph::detail::adjacent_from<Edge>(a, k % 2 ? graph::source(a) : graph::target(a)));
			} else {
				out.edges.push_back(graph::detail::adjacent_from<Edge>(first[std::ptrdiff_t(k)]));
			}
		}
		return out;
	}
}
