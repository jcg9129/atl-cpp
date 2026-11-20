#include <atl/experimental/ds/persistent_heap_variants.hpp>
#include <atl/experimental/graph/k_shortest_walks.hpp>
#include <atl/experimental/graph/k_shortest_walks_variants.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;
using arc = atl::graph::arc<int64>;
using algorithm = ex::persistent_heap_algorithm;

namespace {
	/**
	 * @brief The oracle: best-first search over (length, vertex) that lets
	 *        each vertex be popped at most `k` times; the pops of `target`
	 *        are the `k` shortest walks (the j-th pop of a vertex is its j-th
	 *        shortest walk from `source`, and every walk's prefix is a walk).
	 */
	[[nodiscard]] std::vector<int64> brute_force(std::size_t n, std::vector<arc> const &edges, uint32 source, uint32 target,
												 std::size_t k) {
		std::vector<std::vector<std::pair<uint32, int64>>> out(n);
		for (arc const &e : edges) out[e.from].emplace_back(e.to, e.weight);
		std::vector<std::size_t> pops(n, 0);
		std::vector<int64> lengths;
		using entry = std::pair<int64, uint32>;
		std::priority_queue<entry, std::vector<entry>, std::greater<>> queue;
		queue.emplace(0, source);
		while (!queue.empty() && lengths.size() < k) {
			auto const [d, v] = queue.top();
			queue.pop();
			if (pops[v]++ >= k) continue;
			if (v == target) lengths.push_back(d);
			for (auto const [to, w] : out[v]) queue.emplace(d + w, to);
		}
		return lengths;
	}

	/** @brief Every contender's answer must equal `expected`. */
	void check_all(std::size_t n, std::vector<arc> const &edges, uint32 source, uint32 target, std::size_t k,
				   std::vector<int64> const &expected) {
		using atl::experimental::cheaper_sidetrack;
		using atl::experimental::sidetrack;
		REQUIRE(atl::experimental::k_shortest_walks(n, edges, source, target, k) == expected);
		auto const tree = atl::experimental::shortest_paths_to(n, edges, target);
		auto const lists = atl::experimental::sidetracks_of(n, edges, tree);
		auto const each_algorithm = [&]<algorithm Algorithm>() {
			using heap = ex::persistent_heap_by_meld_policy<sidetrack, cheaper_sidetrack, Algorithm>;
			using forest = ex::persistent_heap_forest_by_meld_policy<sidetrack, cheaper_sidetrack, Algorithm>;
			REQUIRE(ex::k_shortest_walks_by_class_walk<heap>(n, tree, lists, source, target, k) == expected);
			REQUIRE(ex::k_shortest_walks_by_class_pop<heap>(n, tree, lists, source, target, k) == expected);
			REQUIRE(ex::k_shortest_walks_by_forest_walk<forest>(n, tree, lists, source, target, k) == expected);
			REQUIRE(ex::k_shortest_walks_by_forest_pop<forest>(n, tree, lists, source, target, k) == expected);
		};
		each_algorithm.template operator()<algorithm::rank_leftist>();
		each_algorithm.template operator()<algorithm::weight_biased>();
		each_algorithm.template operator()<algorithm::randomized>();
		REQUIRE(ex::k_shortest_walks_by_forest_pop<atl::persistent_heap_forest<sidetrack, cheaper_sidetrack>>(
					n, tree, lists, source, target, k)
				== expected);
		REQUIRE(ex::k_shortest_walks_by_class_walk<atl::persistent_heap<sidetrack, cheaper_sidetrack>>(n, tree, lists, source,
																									   target, k)
				== expected);
	}

	/** @brief `m` uniformly random arcs on `n` vertices, weights in `[0, max_weight]` (loops and parallel arcs included). */
	[[nodiscard]] std::vector<arc> random_graph(std::mt19937_64 &rng, std::size_t n, std::size_t m, int64 max_weight) {
		std::vector<arc> edges(m);
		for (arc &e : edges)
			e = {uint32(rng() % n), uint32(rng() % n), int64(rng() % uint64(max_weight + 1))};
		return edges;
	}
}

TEST_CASE("k_shortest_walks: a hand-checked graph", "[k_shortest_walks]") {
	// 0 -> 1 -> 3 (1 + 1), 0 -> 2 -> 3 (2 + 2), 0 -> 3 (5), and the loop 3 -> 1 (1)
	std::vector<arc> const edges{{0, 1, 1}, {1, 3, 1}, {0, 2, 2}, {2, 3, 2}, {0, 3, 5}, {3, 1, 1}};
	// 2; 4 (0-2-3); 4 (0-1-3-1-3); 5 (0-3); 6 (0-2-3-1-3), 6 (0-1-3-1-3-1-3); 7 (0-3-1-3) ...
	std::vector<int64> const expected{2, 4, 4, 5, 6, 6, 7};
	REQUIRE(brute_force(4, edges, 0, 3, 7) == expected);
	check_all(4, edges, 0, 3, 7, expected);
}

TEST_CASE("k_shortest_walks: edge cases", "[k_shortest_walks]") {
	std::vector<arc> const edges{{0, 1, 3}, {1, 1, 0}, {2, 0, 1}};
	check_all(3, edges, 0, 1, 0, {});			// k = 0
	check_all(3, edges, 1, 0, 5, {});			// unreachable target
	check_all(3, edges, 1, 1, 4, {0, 0, 0, 0}); // source == target, a zero self-loop
	check_all(3, edges, 0, 1, 3, {3, 3, 3});
	check_all(3, edges, 0, 2, 3, {});
	check_all(1, std::vector<arc>{}, 0, 0, 3, {0}); // only the empty walk
	std::vector<arc> const parallel{{0, 1, 2}, {0, 1, 2}, {0, 1, 1}};
	check_all(2, parallel, 0, 1, 5, {1, 2, 2});
}

TEST_CASE("k_shortest_walks: small random graphs against the brute force", "[k_shortest_walks]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 300; ++round) {
		std::size_t const n = std::size_t(1 + rng() % 7);
		std::size_t const m = std::size_t(rng() % 16);
		int64 const max_weight = round % 3 == 0 ? 1 : 9; // many equal lengths, zero weights
		auto const edges = random_graph(rng, n, m, max_weight);
		uint32 const s = uint32(rng() % n), t = uint32(rng() % n);
		std::size_t const k = std::size_t(1 + rng() % 40);
		check_all(n, edges, s, t, k, brute_force(n, edges, s, t, k));
	}
}

TEST_CASE("k_shortest_walks: contenders agree on larger graphs", "[k_shortest_walks]") {
	auto const seed = GENERATE(take(2, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 4; ++round) {
		std::size_t const n = 2000, m = 8000, k = 5000;
		auto const edges = random_graph(rng, n, m, 1000);
		auto const expected = atl::experimental::k_shortest_walks(n, edges, 0, 1, k);
		check_all(n, edges, 0, 1, k, expected);
	}
}

TEST_CASE("k shortest walks supports its int64 boundary and stable zero-weight trees", "[k_shortest_walks]") {
	int64 constexpr bound = (std::numeric_limits<int64>::max)() - 1;
	check_all(2, {{0, 1, bound}}, 0, 1, 1, {bound});
	check_all(2, {{0, 1, bound}, {0, 1, bound}}, 0, 1, 3, {bound, bound});
	check_all(3, {{0, 1, 0}, {1, 0, 0}, {1, 2, 1}}, 0, 2, 4, {1, 1, 1, 1});
	ex::sidetrack_lists const empty{{0, 0}, {}};
	REQUIRE(empty.of(0).empty());
}

TEST_CASE("k shortest walks asserts arithmetic and graph bounds", "[k_shortest_walks]") {
	int64 constexpr bound = (std::numeric_limits<int64>::max)() - 1;
	REQUIRE(atl_test::aborts([&] { (void)ex::shortest_paths_to(3, std::vector<arc>{{0, 1, bound}, {1, 2, 2}}, 2); }));
	REQUIRE(atl_test::aborts([&] { (void)ex::shortest_paths_to(2, std::vector<arc>{{0, 1, -1}}, 1); }));
	REQUIRE(atl_test::aborts([&] { (void)ex::shortest_paths_to(2, std::vector<arc>{{0, 2, 1}}, 1); }));
	REQUIRE(atl_test::aborts([&] { (void)ex::k_shortest_walks(1, std::vector<arc>{{0, 0, bound}}, 0, 0, 3); }));
}

namespace {
	struct borrowed_sidetrack {
		int64 extra;
		uint32 head;

		explicit borrowed_sidetrack(ex::sidetrack const &value) : extra(value.extra), head(value.head) {}

		borrowed_sidetrack(borrowed_sidetrack const &) = delete;
		borrowed_sidetrack &operator=(borrowed_sidetrack const &) = delete;
	};

	static_assert(!std::copy_constructible<borrowed_sidetrack>);

	// The heap versions are copyable; top() lends a noncopyable value whose
	// lifetime ends on the next top() call or version pop.
	struct borrowed_top_heap {
		using base = atl::persistent_heap<ex::sidetrack, ex::cheaper_sidetrack>;
		using node_view = base::node_view;
		base values{};
		mutable std::shared_ptr<borrowed_sidetrack> borrowed;

		bool empty() const { return values.empty(); }

		borrowed_sidetrack const &top() const {
			borrowed = std::make_shared<borrowed_sidetrack>(values.top());
			return *borrowed;
		}

		node_view root() const { return values.root(); }

		borrowed_top_heap pop() const {
			borrowed.reset();
			return {values.pop(), {}};
		}

		template<std::ranges::input_range R>
			requires requires(base const &heap, R &&range) { heap.push_range(std::forward<R>(range)); }
		borrowed_top_heap push_range(R &&range) const { return {values.push_range(std::forward<R>(range)), {}}; }
	};

	struct borrowed_top_forest {
		using base = atl::persistent_heap_forest<ex::sidetrack, ex::cheaper_sidetrack>;
		static constexpr uint32 empty_heap = base::empty_heap;
		base values{};
		mutable std::shared_ptr<borrowed_sidetrack> borrowed;

		bool empty(uint32 heap) const { return values.empty(heap); }

		borrowed_sidetrack const &top(uint32 heap) const {
			borrowed = std::make_shared<borrowed_sidetrack>(values.top(heap));
			return *borrowed;
		}

		uint32 left(uint32 heap) const { return values.left(heap); }

		uint32 right(uint32 heap) const { return values.right(heap); }

		uint32 pop(uint32 heap) {
			borrowed.reset();
			return values.pop(heap);
		}

		template<std::ranges::input_range R>
			requires requires(base &forest, uint32 heap, R &&range) { forest.push_range(heap, std::forward<R>(range)); }
		uint32 push_range(uint32 heap, R &&range) { return values.push_range(heap, std::forward<R>(range)); }
	};

	static_assert(ex::eppstein_class_heap<borrowed_top_heap>);
	static_assert(ex::eppstein_forest_heap<borrowed_top_forest>);
}

TEST_CASE("k_shortest_walks pop contenders accept noncopyable borrowed tops", "[k_shortest_walks][accepted_calls]") {
	std::vector<arc> const edges{{0, 1, 1}, {0, 1, 2}, {1, 1, 3}};
	auto const tree = ex::shortest_paths_to(2, edges, 1);
	auto const lists = ex::sidetracks_of(2, edges, tree);
	auto const expected = brute_force(2, edges, 0, 1, 8);
	REQUIRE(ex::k_shortest_walks_by_class_pop<borrowed_top_heap>(2, tree, lists, 0, 1, 8) == expected);
	REQUIRE(ex::k_shortest_walks_by_forest_pop<borrowed_top_forest>(2, tree, lists, 0, 1, 8) == expected);
}
