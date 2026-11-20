#include <cstddef>
#include <utility>
/**
 * @file
 * @brief Full separator sweeps, parent-only construction and forest ranking throughput.
 * Contenders change stack/queue sweeping, label bucketing, size recomputation,
 * reused sizes or BFS views. Shapes are random, path, caterpillar, binary,
 * broom and forests of tiny components. Shared loops sum emitted depths;
 * structural tests certify decompositions that select different separators.
 * Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2 / -O2 -march=native, 2026-10-15: random n=2e5 ranking 47.8 / 47.2 ms,
 * by subtree sizes 67.8 / 66.6 ms. Full tables: docs/centroid-decomposition.md.
 */
#include <atl/experimental/graph/centroid_variants.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace {
	using tree_t = g::csr<g::vertex_id>;

	/** @brief Local copies of the ranking engine's parts, each with an alternative implementation behind a switch. */
	namespace contender {
		/** @brief BFS order of the forest: `order` itself as the queue (as library) or one `bfs_view` per vertex. */
		template<bool IsDirect>
		std::vector<uint32> forest_order_by_queue(tree_t const &t) {
			std::size_t const n = g::vertex_count(t);
			g::dense_marker marks(n);
			std::vector<uint32> order;
			if constexpr (IsDirect) {
				order.resize(n);
				std::size_t tail = 0;
				for (uint32 s = 0; s < n; ++s) {
					if (!marks.try_visit(s)) continue;
					std::size_t head = tail;
					order[tail++] = s;
					for (; head < tail; ++head)
						for (auto const w : g::out_edges(t, order[head]))
							if (marks.try_visit(w)) order[tail++] = w;
				}
			} else {
				order.reserve(n);
				for (uint32 v = 0; v < n; ++v)
					for (auto const &visit : g::bfs(t, v, marks)) order.push_back(visit.vertex);
			}
			return order;
		}

		template<bool IsDirect>
		std::vector<uint32> vertex_ranking_by_traversal(tree_t const &t) {
			std::size_t const n = g::vertex_count(t);
			auto const order = forest_order_by_queue<IsDirect>(t);
			std::vector<uint32> label(n), mask(n, 0);
			for (std::size_t i = n; i-- > 0;) {
				uint32 const v = order[i];
				uint32 used = 0, duplicated = 0;
				for (auto const w : g::out_edges(t, v)) {
					uint32 const m = mask[w];
					duplicated |= used & m;
					used |= m;
				}
				uint32 const floor_bit = uint32(1) << std::bit_width(duplicated);
				uint32 const possible = ~used & (~floor_bit + 1);
				uint32 const one_hot = possible & (~possible + 1);
				label[v] = one_hot;
				mask[v] = (used | one_hot) & (~one_hot + 1);
			}
			return label;
		}

		/** @brief The sweep, three-argument form: stack (library) or queue-in-scratch
		 * sweep; hand-rolled (library) or `bucket_by_key` label buckets. */
		template<bool IsQueue, bool IsBucketByKey, g::centroid_event Event>
		void centroid_visit_by_sweep(tree_t const &t, Event f) {
			std::size_t const n = g::vertex_count(t);
			if (n == 0) return;
			auto const label = g::detail::vertex_ranking(t);
			std::vector<uint32> by_label(n);
			if constexpr (IsBucketByKey) {
				std::array<uint32, 35> offsets{};
				atl::detail::bucket_by_key(
					34, [&](std::size_t v) { return std::size_t(33 - std::bit_width(label[v])); }, offsets,
					by_label);
			} else {
				std::array<uint32, 34> pos{};
				for (uint32 v = 0; v < n; ++v) ++pos[std::bit_width(label[v])];
				uint32 at = 0;
				for (uint32 r = 34; r-- > 0;) {
					uint32 const c = pos[r];
					pos[r] = at;
					at += c;
				}
				for (uint32 v = 0; v < n; ++v) by_label[pos[std::bit_width(label[v])]++] = v;
			}
			std::vector<g::centroid_vertex> scratch(n);
			std::vector<uint32> stack;
			std::vector<g::vertex_id> parent_center(n, g::none);
			for (uint32 const c : by_label) {
				std::size_t count = 0;
				scratch[count++] = {c, g::none, 0, g::none};
				uint32 branch = 0;
				for (auto const seed : g::out_edges(t, c)) {
					if (label[seed] >= label[c]) continue;
					if constexpr (IsQueue) {
						std::size_t head = count;
						scratch[count++] = {seed, c, 1, branch};
						for (; head < count; ++head) {
							auto const [v, p, depth, br] = scratch[head];
							parent_center[v] = c;
							for (auto const w : g::out_edges(t, v)) {
								if (w == p || label[w] >= label[c]) continue;
								scratch[count++] = {w, v, depth + 1, br};
							}
						}
					} else {
						stack.push_back(uint32(count));
						scratch[count++] = {seed, c, 1, branch};
						while (!stack.empty()) {
							auto const [v, p, depth, br] = scratch[stack.back()];
							stack.pop_back();
							parent_center[v] = c;
							for (auto const w : g::out_edges(t, v)) {
								if (w == p || label[w] >= label[c]) continue;
								stack.push_back(uint32(count));
								scratch[count++] = {w, v, depth + 1, br};
							}
						}
					}
					++branch;
				}
				f(c, parent_center[c], std::span<g::centroid_vertex const>(scratch.data(), count));
			}
		}
	}

	std::vector<std::pair<std::string, std::vector<uint32>>> shapes(std::size_t n) {
		std::mt19937 rng(1);
		std::vector<std::pair<std::string, std::vector<uint32>>> out;
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
		out.emplace_back("random", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = v - 1;
		out.emplace_back("path", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = v % 2 ? v - 1 : v - 2;
		out.emplace_back("caterpillar", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = (v - 1) / 2;
		out.emplace_back("binary", parent);
		// broom: a path over the first half, a star on its end over the rest
		for (uint32 v = 1; v < n; ++v) parent[v] = v < n / 2 ? v - 1 : uint32(n / 2 - 1);
		out.emplace_back("broom", parent);
		return out;
	}

	tree_t build(std::vector<uint32> const &parent, bool is_forest_of_pairs = false) {
		std::vector<g::arc<>> edges;
		for (uint32 v = 1; v < parent.size(); ++v)
			if (!is_forest_of_pairs || v % 2) edges.push_back({parent[v], v});
		return g::undirected_csr(parent.size(), edges);
	}

	using depth_event = std::function<void(uint32, uint32, std::span<g::centroid_vertex const>)>;

	template<typename Visit>
		requires std::invocable<Visit &, tree_t const &, depth_event>
	uint64 depth_sum(tree_t const &t, Visit visit) {
		uint64 acc = 0;
		visit(t, [&](uint32, uint32, std::span<g::centroid_vertex const> c) {
			for (auto const &r : c) acc += r.depth;
		});
		return acc;
	}
}

TEST_CASE("centroid decomposition contenders", "[centroid]") {
	for (auto const &[name, parent] : shapes(200'000)) {
		auto const tree = build(parent);
		auto const library = []<g::centroid_event Event>(tree_t const &t, Event f) { g::centroid_visit(t, f); };
		auto const queue = []<g::centroid_event Event>(tree_t const &t, Event f) { contender::centroid_visit_by_sweep<true, false>(t, f); };
		auto const bucket = []<g::centroid_event Event>(tree_t const &t, Event f) { contender::centroid_visit_by_sweep<false, true>(t, f); };
		auto const classic = []<g::centroid_event Event>(tree_t const &t, Event f) { ex::centroid_visit_by_subtree_sizes(t, f); };
		auto const reused = []<g::centroid_event Event>(tree_t const &t, Event f) { ex::centroid_visit_by_reused_sizes(t, f); };
		uint64 const want = depth_sum(tree, library);
		REQUIRE(depth_sum(tree, queue) == want);
		REQUIRE(depth_sum(tree, bucket) == want);
		REQUIRE(depth_sum(tree, classic) > 0);
		REQUIRE(depth_sum(tree, reused) == depth_sum(tree, classic));

		BENCHMARK(name + " 2e5 | centroid_visit library") { return depth_sum(tree, library); };
		BENCHMARK(name + " 2e5 | centroid_visit by queue sweep") { return depth_sum(tree, queue); };
		BENCHMARK(name + " 2e5 | centroid_visit by bucket labels") { return depth_sum(tree, bucket); };
		BENCHMARK(name + " 2e5 | centroid_visit by subtree sizes") { return depth_sum(tree, classic); };
		BENCHMARK(name + " 2e5 | centroid_visit by reused sizes") { return depth_sum(tree, reused); };
	}
}

TEST_CASE("centroid_decomposition contenders", "[centroid]") {
	for (std::size_t const n : {std::size_t(200'000), std::size_t(1'000'000)})
		for (auto const &[name, parent] : shapes(n)) {
			if (n > 200'000 && name != "random" && name != "path") continue;
			auto const tree = build(parent);
			REQUIRE(g::centroid_decomposition(tree) == ex::centroid_decomposition_by_component_sweep(tree));
			std::string const size = n > 200'000 ? " 1e6" : " 2e5";
			BENCHMARK(name + size + " | separator tree, direct (library)") { return g::centroid_decomposition(tree); };
			BENCHMARK(name + size + " | centroid_decomposition by component sweep") {
				return ex::centroid_decomposition_by_component_sweep(tree);
			};
		}
}

TEST_CASE("vertex ranking BFS order", "[centroid]") {
	std::size_t const n = 1'000'000;
	std::mt19937 rng(2);
	std::vector<uint32> parent(n, 0);
	for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
	std::vector<uint32> pairs(n, 0);
	for (uint32 v = 1; v < n; ++v) pairs[v] = v - 1;
	std::pair<char const *, tree_t> const cases[] = {
		{"random tree 1e6", build(parent)},
		{"1e6 singletons", g::undirected_csr(n, std::vector<g::arc<>>{})},
		{"5e5 pairs", build(pairs, true)},
	};
	for (auto const &[name, tree] : cases) {
		REQUIRE(contender::vertex_ranking_by_traversal<false>(tree) == g::detail::vertex_ranking(tree));
		BENCHMARK(std::string(name) + " | vertex_ranking library") { return g::detail::vertex_ranking(tree); };
		BENCHMARK(std::string(name) + " | vertex_ranking by view per vertex") { return contender::vertex_ranking_by_traversal<false>(tree); };
	}
}
