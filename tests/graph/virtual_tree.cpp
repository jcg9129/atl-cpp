#include <atl/experimental/graph/virtual_tree_variants.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace {
	/**
	 * @brief The O(n)-per-query oracle: keep a vertex if it is a key or at
	 *        least two of its child subtrees hold keys; its parent is the first
	 *        kept vertex up the parent chain. Vertices ordered by the lca's
	 *        entry time, the one input it takes from the structure under test.
	 */
	g::virtual_tree brute(std::vector<uint32> const &parent, g::lca const &tree, std::vector<uint32> const &keys) {
		std::size_t const n = parent.size();
		std::vector<uint32> depth(n, 0);
		for (uint32 v = 0; v < n; ++v)
			for (uint32 u = parent[v]; u != g::none; u = parent[u]) ++depth[v];
		std::vector<uint32> by_depth(n);
		std::iota(by_depth.begin(), by_depth.end(), 0u);
		std::ranges::sort(by_depth, std::greater<>{}, [&](uint32 v) { return depth[v]; });
		std::vector<char> is_key(n, 0), has_key(n, 0), is_kept(n, 0);
		std::vector<uint32> holding_children(n, 0);
		for (uint32 const v : keys) is_key[v] = has_key[v] = 1;
		for (uint32 const v : by_depth) {
			is_kept[v] = is_key[v] || holding_children[v] >= 2;
			if (has_key[v] && parent[v] != g::none) {
				has_key[parent[v]] = 1;
				++holding_children[parent[v]];
			}
		}
		g::virtual_tree out;
		for (uint32 v = 0; v < n; ++v)
			if (is_kept[v]) out.vertices.push_back(v);
		std::ranges::sort(out.vertices, {}, [&](uint32 v) { return tree.entry_time(v); });
		std::vector<uint32> local(n, g::none);
		for (uint32 i = 0; i < out.vertices.size(); ++i) local[out.vertices[i]] = i;
		for (uint32 const v : out.vertices) {
			uint32 u = parent[v];
			while (u != g::none && !is_kept[u]) u = parent[u];
			out.parent.push_back(u == g::none ? g::none : local[u]);
		}
		return out;
	}

	/** @brief A random tree of one shape on @p n vertices as a parent array, labels shuffled. */
	std::vector<uint32> random_tree(std::mt19937_64 &rng, std::size_t n, int shape) {
		std::vector<uint32> parent(n, g::none);
		for (uint32 v = 1; v < n; ++v) switch (shape) {
			case 0:
				parent[v] = v - 1;
				break; // path
			case 1:
				parent[v] = 0;
				break; // star
			case 2:
				parent[v] = v % 2 ? v - 1 : v - 2;
				break; // caterpillar: even spine, odd legs
			case 3:
				parent[v] = (v - 1) / 2;
				break; // binary
			default:
				parent[v] = uint32(rng() % v);
				break; // random recursive
			}
		std::vector<uint32> label(n);
		std::iota(label.begin(), label.end(), 0u);
		std::ranges::shuffle(label, rng);
		std::vector<uint32> relabelled(n, g::none);
		for (uint32 v = 0; v < n; ++v) relabelled[label[v]] = parent[v] == g::none ? g::none : label[parent[v]];
		return relabelled;
	}

	/** @brief One to four random trees of random shapes, concatenated into a forest when @p is_forest. */
	std::vector<uint32> random_forest(std::mt19937_64 &rng, std::size_t n, bool is_forest) {
		std::size_t const trees = is_forest ? 1 + rng() % 4 : 1;
		std::vector<uint32> parent;
		for (std::size_t t = 0; t < trees; ++t) {
			std::size_t const size = t + 1 == trees ? n - parent.size() : (n - parent.size()) / 2;
			if (size == 0) continue;
			auto const part = random_tree(rng, size, int(rng() % 5));
			uint32 const base = uint32(parent.size());
			for (uint32 const p : part) parent.push_back(p == g::none ? g::none : base + p);
		}
		return parent;
	}

	/** @brief The counting-dispatch, counting and comparison contenders, each result to @p check. */
	template<typename Check>
		requires std::invocable<Check const &, g::virtual_tree const &>
	void sorted_by_each(g::lca const &tree, std::vector<uint32> const &keys, atl::graph::detail::virtual_tree_scratch &s,
						Check const &check) {
		g::virtual_tree out;
		for (auto const method : {ex::virtual_tree_sort::counting_dispatch, ex::virtual_tree_sort::counting, ex::virtual_tree_sort::comparison}) {
			ex::virtual_tree_by_sort(tree, keys, method, s, out);
			check(out);
		}
		REQUIRE(std::ranges::all_of(s.bits, [](uint64 word) { return word == 0; }));
	}

	/** @brief Every raced contender, refilling one workspace per key policy. */
	struct contenders {
		g::lca const &tree;
		ex::lca_by_entry_sparse_table sparse;
		ex::virtual_tree_workspace<uint64> packed_work;
		ex::virtual_tree_workspace<uint32> narrow_work;
		atl::graph::detail::virtual_tree_scratch bitmap_work;

		explicit contenders(g::lca const &t) : tree(t), sparse(t) {}

		/** @brief Runs every contender on @p keys and hands each result to @p check. */
		template<typename Check>
			requires std::invocable<Check const &, g::virtual_tree const &>
		void each(std::vector<uint32> const &keys, Check const &check) {
			using enum ex::virtual_tree_algorithm;
			g::virtual_tree out;
			for (bool const is_counting : {false, true}) {
				ex::virtual_tree_by_policy<stack_sweep>(ex::packed_entry_keys{tree}, keys, packed_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<stack_sweep>(ex::projected_keys{tree}, keys, narrow_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<stack_sweep>(ex::entry_keys{tree}, keys, narrow_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<stack_sweep>(ex::sparse_entry_keys{sparse}, keys, narrow_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_depth(tree, keys, packed_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_raw_array(tree, keys, packed_work, out);
				check(out);
				ex::virtual_tree_by_policy<two_sort>(ex::packed_entry_keys{tree}, keys, packed_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<two_sort>(ex::projected_keys{tree}, keys, narrow_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<two_sort>(ex::entry_keys{tree}, keys, narrow_work, out, is_counting);
				check(out);
				ex::virtual_tree_by_policy<two_sort>(ex::sparse_entry_keys{sparse}, keys, narrow_work, out, is_counting);
				check(out);
			}
			sorted_by_each(tree, keys, bitmap_work, check);
		}
	};

	/** @brief Both arrays equal: same vertices in the same order, same local parents. */
	void require_same(g::virtual_tree const &got, g::virtual_tree const &want) {
		REQUIRE(got.vertices == want.vertices);
		REQUIRE(got.parent == want.parent);
	}
}

TEST_CASE("virtual_tree: random trees and forests against the brute", "[graph]") {
	auto seed = GENERATE(take(40, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % (rng() % 3 ? 30 : 300);
	auto const parent = random_forest(rng, n, rng() % 2);
	g::lca const tree(parent);
	contenders race(tree);
	g::virtual_tree reused;
	for (int q = 0; q < 40; ++q) {
		std::size_t const k = rng() % 4 == 0 ? rng() % (2 * n + 1) : rng() % 6;
		std::vector<uint32> keys(k);
		for (uint32 &v : keys) v = uint32(rng() % n);
		if (k > 0 && rng() % 3 == 0) keys[rng() % k] = tree.root(keys[0]); // with the root
		if (k > 1 && rng() % 3 == 0) keys[rng() % k] = keys[0];			   // a duplicate
		auto const want = brute(parent, tree, keys);
		require_same(g::virtual_tree_of(tree, keys), want);
		g::virtual_tree_of(tree, keys, reused);
		require_same(reused, want);
		race.each(keys, [&](g::virtual_tree const &got) { require_same(got, want); });
	}
}

TEST_CASE("virtual_tree: every dispatch path on a large tree", "[graph]") {
	// n = 2e4: up to 32 keys sort in stack buffers; 33 to n/64 use the heap
	// with entry ranks; from n/128 the bitmap; from n/64 depth ranks
	std::mt19937_64 rng(GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)()))));
	std::size_t const n = 20'000;
	auto const parent = random_forest(rng, n, rng() % 2);
	g::lca const tree(parent);
	g::virtual_tree reused;
	atl::graph::detail::virtual_tree_scratch bitmap_work;
	for (std::size_t const k : {0, 1, 2, 32, 33, 100, 156, 157, 312, 313, 624, 625, 5000}) {
		std::vector<uint32> keys(k);
		for (uint32 &v : keys) v = uint32(rng() % n);
		auto const want = brute(parent, tree, keys);
		require_same(g::virtual_tree_of(tree, keys), want);
		g::virtual_tree_of(tree, keys, reused);
		require_same(reused, want);
		sorted_by_each(tree, keys, bitmap_work, [&](g::virtual_tree const &got) { require_same(got, want); });
	}
	// an unsized input range takes the heap path
	std::vector<uint32> keys(10);
	for (uint32 &v : keys) v = uint32(rng() % n);
	auto unsized = keys | std::views::filter([](uint32) { return true; });
	require_same(g::virtual_tree_of(tree, unsized), brute(parent, tree, keys));
}

TEST_CASE("virtual_tree: edge cases", "[graph]") {
	// 0 - 1 - 3, 0 - 2; 4 alone
	std::vector<uint32> const parent{g::none, 0, 0, 1, g::none};
	g::lca const tree(parent);
	SECTION("no keys") {
		auto const vt = g::virtual_tree_of(tree, std::vector<uint32>{});
		CHECK(vt.vertices.empty());
		CHECK(vt.parent.empty());
	}
	SECTION("one key, repeated") {
		auto const vt = g::virtual_tree_of(tree, std::vector<uint32>{3, 3, 3});
		CHECK(vt.vertices == std::vector<uint32>{3});
		CHECK(vt.parent == std::vector<uint32>{g::none});
	}
	SECTION("the LCA joins two leaves") {
		auto const vt = g::virtual_tree_of(tree, std::vector<uint32>{2, 3});
		CHECK(vt.vertices == std::vector<uint32>{0, 3, 2}); // heavy child 1 first in the preorder
		CHECK(vt.parent == std::vector<uint32>{g::none, 0, 0});
	}
	SECTION("two trees give two roots") {
		auto const vt = g::virtual_tree_of(tree, std::vector<uint32>{4, 3, 1});
		CHECK(vt.vertices == std::vector<uint32>{1, 3, 4});
		CHECK(vt.parent == std::vector<uint32>{g::none, 0, g::none});
	}
	SECTION("refill shrinks a larger retained result") {
		g::virtual_tree vt;
		g::virtual_tree_of(tree, std::vector<uint32>{2, 3, 4}, vt);
		g::virtual_tree_of(tree, std::vector<uint32>{2}, vt);
		CHECK(vt.vertices == std::vector<uint32>{2});
		CHECK(vt.parent == std::vector<uint32>{g::none});
	}
	SECTION("a key out of range aborts") {
		CHECK(atl_test::aborts([&] { (void)g::virtual_tree_of(tree, std::vector<uint32>{1, 5}); }));
	}
	SECTION("the contenders reject a key out of range and take rvalue ranges") {
		ex::virtual_tree_workspace<uint64> work;
		g::virtual_tree out;
		CHECK(atl_test::aborts([&] {
			ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::stack_sweep>(ex::packed_entry_keys{tree}, std::vector<uint32>{1, 5},
																				work, out);
		}));
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::stack_sweep>(ex::packed_entry_keys{tree}, std::vector<uint32>{2, 3}, work,
																			out);
		CHECK(out.vertices == std::vector<uint32>{0, 3, 2});
	}
	SECTION("clearing the scratch leaves later queries correct") {
		g::virtual_tree_clear();
		auto const vt = g::virtual_tree_of(tree, std::vector<uint32>{2, 3});
		CHECK(vt.vertices == std::vector<uint32>{0, 3, 2});
		g::virtual_tree_clear();
		g::virtual_tree_clear();
		CHECK(g::virtual_tree_of(tree, std::vector<uint32>{4, 3, 1}).parent == std::vector<uint32>{g::none, 0, g::none});
	}
}

TEST_CASE("virtual_tree: bitmap word boundaries and reused scratch", "[graph]") {
	g::detail::virtual_tree_scratch scratch;
	g::virtual_tree out;
	for (std::size_t n : {129, 64, 65, 63, 1, 130}) {
		std::vector<uint32> parent(n, g::none);
		for (std::size_t i = 1; i < n; ++i) parent[i] = uint32((i - 1) / 2);
		g::lca const tree(parent);
		std::vector<uint32> keys;
		for (std::size_t i = n; i > 0; --i) {
			keys.push_back(uint32(i - 1));
			keys.push_back(uint32(i - 1));
		}
		g::detail::virtual_tree_build(tree, keys, scratch, out);
		require_same(out, brute(parent, tree, keys));
		REQUIRE(std::ranges::all_of(scratch.bits, [](uint64 word) { return word == 0; }));
	}
	std::vector<uint32> parent(65, g::none);
	g::lca const tree(parent);
	scratch.bits.assign(2, 0);
	scratch.bits[0] = 1;
	std::vector<uint64> keys;
	CHECK(atl_test::aborts([&] { g::detail::virtual_tree_bitmap_sort(tree, keys, scratch.bits); }));
}
