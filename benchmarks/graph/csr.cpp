/** @file
 * @brief CSR construction throughput: library row-offset cursors, by cursor copy,
 * by bucket gather, and by nested vectors. Directed and doubled arcs, weighted
 * and unweighted, at (n,m)=(1e6,2e6) and (1e7,1e7); construction alone is timed.
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG
 * -std=c++23, also -march=native, 2026-10-15: directed unweighted 22.6/22.8 ms
 * library, 22.4/23.7 cursor copy, 35.8/36.6 bucket gather at n=1e6.
 * Full tables: docs/graph.md.
 */
#include <atl/experimental/graph/csr_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string name, Run run) {
		BENCHMARK(std::move(name)) { return std::invoke(run); };
	}
}

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;

namespace {
	/**
	 * @brief The CSR build race on one random edge list: the library build
	 *        (offset table as its own cursor) against the separate cursor
	 *        copy and the bucket_by_key + gather pass, directed and
	 *        undirected; @p has_lists adds the `vector<vector>` build a
	 *        hand-written application would write instead.
	 */
	template<g::list_edge Arc>
		requires std::default_initializable<Arc> && std::movable<Arc>
			  && requires(Arc &a, uint32 v) { a.from = v; a.to = v; }
			  && (!g::weighted_list_edge<Arc> || requires(Arc &a) { a.weight = int64{}; })
	void race(std::string const &name, std::size_t n, std::size_t m, bool has_lists) {
		std::mt19937 rng(1);
		std::vector<Arc> arcs(m);
		for (auto &a : arcs) {
			a.from = uint32(rng() % n);
			a.to = uint32(rng() % n);
			if constexpr (g::weighted_list_edge<Arc>) a.weight = int64(rng());
		}
		using edge_t = g::detail::adjacent_of_t<Arc>;
		auto check = [&]<bool HasBothDirections>() {
			auto const library = [&] {
				if constexpr (HasBothDirections)
					return g::undirected_csr(n, arcs);
				else
					return g::csr(n, arcs);
			}();
			auto const copy = ex::csr_by_cursor_copy<edge_t, HasBothDirections>(n, arcs);
			auto const bucket = ex::csr_by_bucket_gather<edge_t, HasBothDirections>(n, arcs);
			REQUIRE(library.edge_count() == copy.edges.size());
			REQUIRE(bucket.offset == copy.offset);
			REQUIRE(bucket.edges.size() == copy.edges.size());
			std::vector<std::vector<edge_t>> lists(has_lists ? n : 0);
			if (has_lists) {
				for (auto const &a : arcs) {
					lists[a.from].push_back(g::detail::adjacent_from<edge_t>(a));
					if constexpr (HasBothDirections) lists[a.to].push_back(g::detail::adjacent_from<edge_t>(a, a.from));
				}
			}
			for (uint32 u = 0; u < n; ++u) {
				auto const row = library.out_edges(u);
				REQUIRE(row.size() == copy.offset[u + 1] - copy.offset[u]);
				if (has_lists) REQUIRE(lists[u].size() == row.size());
				for (std::size_t k = 0; k < row.size(); ++k) {
					auto const &copied = copy.edges[copy.offset[u] + k];
					auto const &gathered = bucket.edges[bucket.offset[u] + k];
					REQUIRE(g::target(row[k]) == g::target(copied));
					REQUIRE(g::target(row[k]) == g::target(gathered));
					if (has_lists) REQUIRE(g::target(row[k]) == g::target(lists[u][k]));
					if constexpr (g::weighted_adjacency_edge<edge_t>) {
						REQUIRE(g::weight(row[k]) == g::weight(copied));
						REQUIRE(g::weight(row[k]) == g::weight(gathered));
						if (has_lists) REQUIRE(g::weight(row[k]) == g::weight(lists[u][k]));
					}
				}
			}
		};
		check.template operator()<false>();
		check.template operator()<true>();

		measure(name + " | directed csr library", [&] { return g::csr(n, arcs).edge_count(); });
		measure(name + " | directed csr by cursor copy", [&] {
			return ex::csr_by_cursor_copy<edge_t, false>(n, arcs).offset[n / 2];
		});
		measure(name + " | directed csr by bucket gather", [&] {
			return ex::csr_by_bucket_gather<edge_t, false>(n, arcs).offset[n / 2];
		});
		measure(name + " | undirected csr library", [&] { return g::undirected_csr(n, arcs).edge_count(); });
		measure(name + " | undirected csr by cursor copy", [&] {
			return ex::csr_by_cursor_copy<edge_t, true>(n, arcs).offset[n / 2];
		});
		measure(name + " | undirected csr by bucket gather", [&] {
			return ex::csr_by_bucket_gather<edge_t, true>(n, arcs).offset[n / 2];
		});
		if (!has_lists) return;
		measure(name + " | directed csr by nested vectors", [&] {
			std::vector<std::vector<edge_t>> adj(n);
			for (auto const &a : arcs) adj[a.from].push_back(g::detail::adjacent_from<edge_t>(a));
			return adj[n / 2].size();
		});
		measure(name + " | undirected csr by nested vectors", [&] {
			std::vector<std::vector<edge_t>> adj(n);
			for (auto const &a : arcs) {
				adj[a.from].push_back(g::detail::adjacent_from<edge_t>(a));
				adj[a.to].push_back(g::detail::adjacent_from<edge_t>(a, a.from));
			}
			return adj[n / 2].size();
		});
	}
}

TEST_CASE("csr build", "[graph]") {
	race<g::arc<>>("n=1e6 m=2e6 arc<>", 1'000'000, 2'000'000, true);
	race<g::arc<int64>>("n=1e6 m=2e6 arc<int64>", 1'000'000, 2'000'000, true);
	race<g::arc<>>("n=1e7 m=1e7 arc<>", 10'000'000, 10'000'000, false);
}
