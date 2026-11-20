#include <atl/experimental/graph/offline_bipartiteness.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint8;
using vertex_pair = std::pair<atl::graph::vertex_id, atl::graph::vertex_id>;

namespace {
	/** @brief The oracle: 2-colour the current edge set by BFS after every toggle. */
	std::vector<uint8> brute_bipartite(std::size_t n, std::vector<vertex_pair> const &toggles) {
		std::set<vertex_pair> edges;
		std::vector<uint8> out;
		for (auto [u, v] : toggles) {
			vertex_pair const key{(std::min)(u, v), (std::max)(u, v)};
			if (!edges.erase(key)) edges.insert(key);
			std::vector<std::vector<uint32>> adj(n);
			for (auto [a, b] : edges) adj[a].push_back(b), adj[b].push_back(a);
			std::vector<int> colour(n, -1);
			bool is_bipartite = true;
			for (uint32 s = 0; s < n && is_bipartite; ++s) {
				if (colour[s] != -1) continue;
				colour[s] = 0;
				std::vector<uint32> queue{s};
				for (std::size_t h = 0; h < queue.size() && is_bipartite; ++h)
					for (uint32 w : adj[queue[h]]) {
						if (colour[w] == -1)
							colour[w] = colour[queue[h]] ^ 1, queue.push_back(w);
						else if (colour[w] == colour[queue[h]])
							is_bipartite = false;
					}
			}
			out.push_back(is_bipartite);
		}
		return out;
	}
}

TEST_CASE("offline bipartiteness example: random toggles against BFS 2-colouring", "[time_divide_and_conquer][example]") {
	std::mt19937_64 rng(813);
	for (int iter = 0; iter < 300; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 8);
		std::vector<vertex_pair> toggles(std::size_t(rng() % 60));
		bool const has_self_loops = iter % 5 == 0;
		if (n == 1 && !has_self_loops) toggles.clear(); // a lone vertex only has its loop
		for (auto &[u, v] : toggles) {
			do u = uint32(rng() % n), v = uint32(rng() % n);
			while (u == v && !has_self_loops);
		}
		CHECK(atl::experimental::bipartite_after_each(n, toggles) == brute_bipartite(n, toggles));
	}
}

TEST_CASE("offline bipartiteness example: fixed edge log", "[time_divide_and_conquer][example]") {
	std::vector<vertex_pair> const toggles{{1, 2}, {0, 2}, {0, 1}, {0, 1}, {0, 1}};
	CHECK(atl::experimental::bipartite_after_each(3, toggles) == std::vector<uint8>{1, 1, 0, 1, 0});
}

TEST_CASE("offline bipartiteness empty graph and reversed self-loop toggles", "[graph][bipartite]") {
	CHECK(atl::experimental::bipartite_after_each(0, {}).empty());
	std::vector<vertex_pair> const toggles{{0, 1}, {1, 0}, {0, 0}, {0, 0}};
	CHECK(atl::experimental::bipartite_after_each(2, toggles) == std::vector<uint8>{1, 1, 0, 1});
#ifdef ATL_ASSERT
	REQUIRE(atl_test::aborts([] {
		std::vector<vertex_pair> const invalid{{0, 0}, {0, 2}};
		static_cast<void>(atl::experimental::bipartite_after_each(2, invalid));
	}));
#endif
}
