#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/utils/hash.hpp>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;

namespace {
	struct edge {
		uint32 symbol = 0, left = uint32(-1), right = uint32(-1);
		uint32 target = uint32(-1);
	};

	static_assert(atl::detail::sibling_node<edge>);

	// The pool never assigns nodes, so a const owner field (non-assignable node) is accepted.
	struct labelled_edge {
		uint32 symbol = 0, left = uint32(-1), right = uint32(-1);
		uint32 const label = 7;
	};

	static_assert(!std::is_copy_assignable_v<labelled_edge> && atl::detail::sibling_node<labelled_edge>);
}

TEST_CASE("hashed_child_pool: per-owner transition maps to arbitrary targets", "[ds]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);

	constexpr uint32 nil = atl::detail::hashed_child_pool<edge>::nil;
	atl::detail::hashed_child_pool<edge> pool;

	constexpr int States = 40, U = 100'000;
	std::vector<uint32> root(States, nil);
	std::map<std::pair<int, uint32>, uint32> oracle;

	auto set = [&](int s, uint32 sym, uint32 tgt) {
		auto const [e, r, was_inserted] = pool.emplace(root[s], sym);
		REQUIRE(was_inserted == !oracle.contains({s, sym}));
		root[s] = r;
		pool[e].target = tgt;
		oracle[{s, sym}] = tgt;
	};
	auto get = [&](int s, uint32 sym) -> uint32 {
		uint32 e = pool.find(root[s], sym);
		return e == nil ? nil : pool[e].target;
	};

	for (int op = 0; op < 6000; ++op) {
		int s = int(rng() % States);
		uint32 sym = uint32(rng() % U);
		uint32 tgt = uint32(rng() % States);
		set(s, sym, tgt);

		int q = int(rng() % States);
		uint32 qs = uint32(rng() % U);
		auto it = oracle.find({q, qs});
		REQUIRE(get(q, qs) == (it == oracle.end() ? nil : it->second));
	}

	for (int s = 0; s < States; ++s) {
		std::map<uint32, uint32> seen;
		pool.for_each(root[s], [&](uint32 e) { seen[pool[e].symbol] = pool[e].target; });
		std::map<uint32, uint32> want;
		for (auto const &[key, tgt] : oracle)
			if (key.first == s) want[key.second] = tgt;
		REQUIRE(seen == want);
	}
}

TEST_CASE("hashed_child_pool: sorted-symbol inserts stay balanced (no chain)", "[ds]") {
	// the raw-symbol failure case: ascending symbols. Correctness across a large
	// run is the proxy — a degenerate chain would also blow the stack/time.
	atl::detail::hashed_child_pool<edge> pool;
	uint32 root = atl::detail::hashed_child_pool<edge>::nil;
	constexpr int N = 200'000;
	for (int i = 0; i < N; ++i) {
		auto const [e, r, was_inserted] = pool.emplace(root, uint32(i));
		REQUIRE(was_inserted);
		root = r;
		pool[e].target = uint32(i);
	}
	for (int i = 0; i < N; ++i) REQUIRE(pool[pool.find(root, uint32(i))].target == uint32(i));
	REQUIRE(pool.find(root, uint32(N)) == atl::detail::hashed_child_pool<edge>::nil);
}

TEST_CASE("hashed_child_pool: detached nodes, traversal and clear", "[ds]") {
	atl::detail::hashed_child_pool<edge> pool;
	pool.reserve(1);
	auto const detached = pool.add();
	REQUIRE(pool[detached].target == pool.nil);
	auto const first = pool.emplace(pool.nil, 12);
	auto const duplicate = pool.emplace(first.root, 12);
	REQUIRE_FALSE(duplicate.was_inserted);
	REQUIRE(duplicate.node == first.node);
	REQUIRE(duplicate.root == first.root);
	REQUIRE(pool.size() == 2);
	auto const captured_seed = pool.seed();
	pool.clear();
	REQUIRE(pool.size() == 0);
	REQUIRE(pool.seed() == captured_seed);
	int visits = 0;
	pool.for_each(pool.nil, [&](uint32) { ++visits; });
	REQUIRE(visits == 0);
}
