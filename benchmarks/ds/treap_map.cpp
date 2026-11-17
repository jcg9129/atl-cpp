/**
 * @file
 * @brief Keyed-tree throughput on 400,000 random insert/erase/rank steps and
 * 10,000-key unions into copies of a million-draw set. std::map substitutes
 * membership for rank; those query timings are not equivalent. Element insertion
 * races the library union. Setup and copies are included in the timed loops.
 * Not measured yet: no CPU/compiler/flags/date record is available.
 */
#include <atl/ds/treap_map.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <map>
#include <random>
#include <utility>
#include <vector>

using ll = long long;

TEST_CASE("treap_map benchmarks", "[treap_map]") {
	constexpr int m = 400'000;
	std::mt19937 rng(1);
	std::vector<int> key(m);
	std::vector<ll> val(m);
	std::vector<char> kind(m);
	for (int i = 0; i < m; ++i) {
		key[i] = int(rng() % 1'000'000);
		val[i] = ll(rng() % 1000);
		kind[i] = char(rng() % 3);
	}

	auto plus_ = [](ll a, ll b) { return a + b; };

	auto const compute_checksum = [&]<typename Make, typename Assign, typename Erase, typename Query>(Make make, Assign assign, Erase erase, Query query)
		requires std::invocable<Make &> && std::invocable<Assign &, std::invoke_result_t<Make &> &, int, ll>
			  && std::is_invocable_r_v<std::size_t, Erase &, std::invoke_result_t<Make &> &, int>
			  && std::is_invocable_r_v<std::size_t, Query &, std::invoke_result_t<Make &> &, int>
	{
		auto tree = std::invoke(make);
		atl::uint64 acc = 0;
		for (int i = 0; i < m; ++i)
			if (kind[i] == 0)
				std::invoke(assign, tree, key[i], val[i]);
			else if (kind[i] == 1)
				acc += std::invoke(erase, tree, key[i]);
			else
				acc += std::invoke(query, tree, key[i]);
		return acc;
	};
	using tree = atl::treap_map<int, ll, std::ranges::less, decltype(plus_)>;
	auto const make_library = [&] { return tree(0LL, plus_); };
	auto const assign_library = [](tree &t, int k, ll v) { t.insert_or_assign(k, v); };
	auto const erase_library = [](tree &t, int k) { return std::size_t(t.erase(k)); };
	auto const rank_library = [](tree &t, int k) { return t.rank(k); };
	/** @brief Adapt std::map: membership is the timed stand-in for its unsupported rank. */
	auto const query_by_std_map = [](std::map<int, ll> &t, int k) { return t.count(k); };
	auto const make_std_map = [] { return std::map<int, ll>{}; };
	auto const assign_std_map = [](std::map<int, ll> &t, int k, ll v) { t[k] = v; };
	auto const erase_std_map = [](std::map<int, ll> &t, int k) { return t.erase(k); };
	// Membership permits an equivalent untimed check without adding linear rank walks to the measured loop.
	REQUIRE(compute_checksum(make_library, assign_library, erase_library, [](tree &t, int k) { return std::size_t(t.contains(k)); })
			== compute_checksum(make_std_map, assign_std_map, erase_std_map, query_by_std_map));
	BENCHMARK("library -- 4e5 insert/erase/rank") { return compute_checksum(make_library, assign_library, erase_library, rank_library); };
	BENCHMARK("by std map -- 4e5 insert/erase/membership (rank unsupported)") { return compute_checksum(make_std_map, assign_std_map, erase_std_map, query_by_std_map); };
	// union vs element-wise: both start from a deep copy of the same prebuilt
	// 1e6-key set (the copy cost is identical, so the difference is the merge).
	atl::treap_set<int> big;
	std::vector<int> small_keys(10'000);
	for (int i = 0; i < 1'000'000; ++i) big.insert(int(rng() % 10'000'000));
	for (auto &k : small_keys) k = int(rng() % 10'000'000);

	auto const merge_checksum = [&]<typename Merge>(Merge merge)
		requires std::is_invocable_r_v<std::size_t, Merge &, atl::treap_set<int> &>
	{
		auto a = big;
		return std::invoke(merge, a);
	};
	auto const merge_library = [&](atl::treap_set<int> &a) {
		auto b = a.sibling();
		for (int k : small_keys) b.insert(k);
		a.merge_union(std::move(b));
		return a.size();
	};
	/** @brief Merge by inserting each incoming key separately. */
	auto const merge_union_by_element_insertion = [&](atl::treap_set<int> &a) {
		for (int k : small_keys) a.insert(k);
		return a.size();
	};
	REQUIRE(merge_checksum(merge_library) == merge_checksum(merge_union_by_element_insertion));
	BENCHMARK("library -- 1e4-key union into a copy of 1e6") { return merge_checksum(merge_library); };
	BENCHMARK("by element insertion -- 1e4-key union into a copy of 1e6") { return merge_checksum(merge_union_by_element_insertion); };
}
