#include <atl/ds/coordinate_compression.hpp>
#include <atl/experimental/ds/coordinate_compression_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

namespace vws = std::views;

TEST_CASE("coordinate_compression: ranks, bounds, decompression", "[coordinate_compression]") {
	std::vector<int> const v{30, 10, 20, 10, 30, 30};
	atl::coordinate_compression cc(v);
	static_assert(std::is_same_v<decltype(cc)::value_type, int>);

	REQUIRE(cc.size() == 3);
	REQUIRE(cc[0] == 10);
	REQUIRE(cc[1] == 20);
	REQUIRE(cc[2] == 30);

	REQUIRE(cc.rank(10) == 0);
	REQUIRE(cc.rank(20) == 1);
	REQUIRE(cc.rank(30) == 2);
	// absent values: lower-bound semantics
	REQUIRE(cc.rank(5) == 0);
	REQUIRE(cc.rank(15) == 1);
	REQUIRE(cc.rank(99) == 3);
	// inclusive right endpoints: upper-bound semantics
	REQUIRE(cc.rank_upper(10) == 1);
	REQUIRE(cc.rank_upper(15) == 1);
	REQUIRE(cc.rank_upper(30) == 3);

	REQUIRE(cc.contains(20));
	REQUIRE_FALSE(cc.contains(15));

	for (std::size_t i = 0; i < cc.size(); ++i) REQUIRE(cc.rank(cc[i]) == i);
	REQUIRE(std::vector<int>(cc.begin(), cc.end()) == std::vector<int>{10, 20, 30});
}

TEST_CASE("coordinate_compression: projection and view pipelines", "[coordinate_compression]") {
	struct event {
		long long pos;
		int kind;
	};

	std::vector<event> const evs{{100, 0}, {5, 1}, {100, 0}, {42, 1}, {7, 0}};

	atl::coordinate_compression by_pos(evs, &event::pos);
	static_assert(std::is_same_v<decltype(by_pos)::value_type, long long>);
	REQUIRE(by_pos.size() == 4);
	REQUIRE(by_pos.rank(42) == 2);

	atl::coordinate_compression kind0(evs | vws::filter([](event const &e) { return e.kind == 0; }), &event::pos);
	REQUIRE(kind0.size() == 2);
	REQUIRE(kind0[0] == 7);
	REQUIRE(kind0[1] == 100);
}

TEST_CASE("coordinate_compression: custom compare", "[coordinate_compression]") {
	std::vector<int> const v{3, 1, 2};
	atl::coordinate_compression cc(v, std::identity{}, std::ranges::greater{});
	REQUIRE(cc[0] == 3);
	REQUIRE(cc[2] == 1);
	REQUIRE(cc.rank(3) == 0);
	REQUIRE(cc.rank(2) == 1);
	REQUIRE(cc.rank_upper(2) == 2);
	REQUIRE(cc.contains(2));
	REQUIRE_FALSE(cc.contains(5));
}

TEST_CASE("coordinate_compression: empty and single", "[coordinate_compression]") {
	atl::coordinate_compression<int> empty;
	REQUIRE(empty.empty());
	REQUIRE(empty.rank(5) == 0);
	REQUIRE(empty.rank_upper(5) == 0);
	REQUIRE_FALSE(empty.contains(5));

	atl::coordinate_compression one(std::vector<int>{7});
	REQUIRE(one.size() == 1);
	REQUIRE(one.rank(7) == 0);
	REQUIRE(one.rank_upper(7) == 1);
}

TEST_CASE("coordinate_compression: rvalue range moves its elements in", "[coordinate_compression]") {
	using atl_test::move_probe;
	auto const by_v = [](move_probe const &a, move_probe const &b) { return a.v < b.v; };

	std::vector<move_probe> vals;
	for (int i = 0; i < 8; ++i) vals.emplace_back(move_probe::sentinel);
	move_probe::reset();
	atl::coordinate_compression copied(vals, std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 8);

	move_probe::reset();
	atl::coordinate_compression moved(std::move(vals), std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved, never copied
	REQUIRE(moved.size() == 1);
}

TEST_CASE("coordinate_compression: rvalue borrowed ranges and views leave the source intact",
		  "[coordinate_compression]") {
	using atl_test::move_probe;
	auto const by_v = [](move_probe const &a, move_probe const &b) { return a.v < b.v; };
	auto const intact = [](std::vector<move_probe> const &v) {
		return std::ranges::all_of(v, [](move_probe const &p) { return p.v == move_probe::sentinel; });
	};
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));

	move_probe::reset();
	atl::coordinate_compression from_span(std::span<move_probe>(vals), std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(intact(vals));

	move_probe::reset();
	atl::coordinate_compression from_filter(vals | std::views::filter([](move_probe const &) { return true; }),
											std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(intact(vals));
	REQUIRE(from_filter.size() == 1);

	// A view yielding T&& still refers to the caller's storage: copied, never moved from
	// (a moved-from probe holds -1, so intact() detects it).
	move_probe::reset();
	atl::coordinate_compression from_as_rvalue(vals | std::views::as_rvalue, std::identity{}, by_v);
	REQUIRE(intact(vals));
	move_probe::reset();
	atl::coordinate_compression from_moving_transform(
		vals | std::views::transform([](move_probe &p) -> move_probe && { return std::move(p); }), std::identity{}, by_v);
	REQUIRE(intact(vals));
}

namespace {
	// The projection is checked against the element as actually forwarded: an
	// rvalue-only projection works for an owning rvalue range, and is rejected
	struct takes_rvalue {
		int operator()(std::string &&s) const { return int(s.size()); }
	};
}

static_assert(std::constructible_from<atl::coordinate_compression<int>, std::vector<std::string>, takes_rvalue>);
static_assert(!std::constructible_from<atl::coordinate_compression<int>, std::vector<std::string> &, takes_rvalue>);

TEST_CASE("coordinate_compression: bulk ranks match rank() on every shape", "[coordinate_compression]") {
	std::mt19937 rng(7);
	for (int iter = 0; iter < 400; ++iter) {
		int const n = int(rng() % 30), q = int(rng() % 60), range = 1 + int(rng() % 40);
		std::vector<int> base(n), queries(q);
		for (int &x : base) x = int(rng() % range) - range / 2;
		for (int &x : queries) x = int(rng() % (range + 6)) - range / 2 - 3;
		atl::coordinate_compression cc(base);
		for (bool sorted : {false, true}) {
			if (sorted) std::ranges::sort(queries);
			auto const got = cc.ranks(queries);
			REQUIRE(got.size() == queries.size());
			for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(got[i] == cc.rank(queries[i]));
		}
	}
}

TEST_CASE("coordinate_compression: bulk ranks of a non-sized forward range", "[coordinate_compression]") {
	std::mt19937 rng(11);
	auto const always = [](int) { return true; };
	for (int iter = 0; iter < 300; ++iter) {
		int const n = int(rng() % 30), q = int(rng() % 60), range = 1 + int(rng() % 40);
		std::vector<int> base(n), queries(q);
		for (int &x : base) x = int(rng() % range) - range / 2;
		for (int &x : queries) x = int(rng() % (range + 6)) - range / 2 - 3;
		atl::coordinate_compression cc(base);
		for (bool sorted : {false, true}) {
			if (sorted) std::ranges::sort(queries);
			auto const got = cc.ranks(queries | vws::filter(always));
			REQUIRE(got.size() == queries.size());
			for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(got[i] == cc.rank(queries[i]));
		}
	}
	atl::coordinate_compression const cc(std::vector<int>{10, 20, 30});
	std::vector<int> const one{25}, none;
	REQUIRE(cc.ranks(one | vws::filter(always)) == std::vector<atl::uint32>{2});
	REQUIRE(cc.ranks(none | vws::filter(always)).empty());
}

TEST_CASE("coordinate_compression: bulk ranks forms", "[coordinate_compression]") {
	atl::coordinate_compression const cc(std::vector<int>{10, 20, 30});
	// sorted with many queries (merge-walk), including values below/between/above and duplicates
	REQUIRE(cc.ranks(std::vector<int>{0, 10, 10, 15, 20, 31, 99}) == std::vector<atl::uint32>{0, 0, 0, 1, 1, 3, 3});
	// non-sized, non-forward-sorted pipeline and a descending universe
	REQUIRE(cc.ranks(std::vector<int>{5, 6, 7, 8} | vws::filter([](int) { return true; }))
			== std::vector<atl::uint32>{0, 0, 0, 0});
	atl::coordinate_compression const desc(std::vector<int>{1, 2, 3}, std::identity{}, std::ranges::greater{});
	REQUIRE(desc.ranks(std::vector<int>{3, 2, 1, 0}) == std::vector<atl::uint32>{0, 1, 2, 3});
	REQUIRE(atl::coordinate_compression<int>{}.ranks(std::vector<int>{1, 2}) == std::vector<atl::uint32>{0, 0});
	REQUIRE(cc.ranks(std::vector<int>{}).empty());
}

TEST_CASE("compress: universe and ranks match the two-step form", "[coordinate_compression]") {
	std::mt19937 rng(11);
	for (int iter = 0; iter < 400; ++iter) {
		int const n = int(rng() % 40), range = 1 + int(rng() % 30);
		std::vector<long long> v(n);
		for (auto &x : v) x = (long long)(rng() % range) * 1'000'000'007LL - range;
		auto const c = atl::compress(v);
		atl::coordinate_compression const ref(v);
		REQUIRE(std::vector<long long>(c.universe.begin(), c.universe.end()) == std::vector<long long>(ref.begin(), ref.end()));
		REQUIRE(c.ranks.size() == v.size());
		for (std::size_t i = 0; i < v.size(); ++i) REQUIRE(c.ranks[i] == ref.rank(v[i]));
	}
}

TEST_CASE("compress: projection, compare, move, empty", "[coordinate_compression]") {
	struct event {
		long long pos;
		int kind;
	};

	std::vector<event> const evs{{100, 0}, {5, 1}, {100, 0}, {42, 1}};
	auto const by_pos = atl::compress(evs, &event::pos);
	static_assert(std::is_same_v<decltype(by_pos.universe)::value_type, long long>);
	REQUIRE(by_pos.ranks == std::vector<atl::uint32>{2, 0, 2, 1});

	auto const desc = atl::compress(std::vector<int>{3, 1, 2, 3}, std::identity{}, std::ranges::greater{});
	REQUIRE(desc.ranks == std::vector<atl::uint32>{0, 2, 1, 0});
	REQUIRE(desc.universe[0] == 3);

	REQUIRE(atl::compress(std::vector<int>{}).ranks.empty());

	using atl_test::move_probe;
	auto const by_v = [](move_probe const &a, move_probe const &b) { return a.v < b.v; };
	std::vector<move_probe> vals;
	for (int i = 0; i < 8; ++i) vals.emplace_back(move_probe::sentinel);
	move_probe::reset();
	auto const copied = atl::compress(vals, std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 8);
	REQUIRE(copied.universe.size() == 1);
	move_probe::reset();
	auto const moved = atl::compress(std::move(vals), std::identity{}, by_v);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.ranks == std::vector<atl::uint32>(8, 0));
}

TEST_CASE("coordinate_compression: sorted_unique adoption", "[coordinate_compression]") {
	atl::coordinate_compression<int> const cc(atl::sorted_unique, std::vector<int>{1, 4, 9});
	REQUIRE(cc.size() == 3);
	REQUIRE(cc.rank(4) == 1);
	REQUIRE(cc.rank(5) == 2);
	atl::coordinate_compression<int> const none(atl::sorted_unique, std::vector<int>{});
	REQUIRE(none.empty());
	CHECK(atl_test::aborts([] { atl::coordinate_compression<int> bad(atl::sorted_unique, std::vector<int>{1, 1}); }));
	CHECK(atl_test::aborts([] { atl::coordinate_compression<int> bad(atl::sorted_unique, std::vector<int>{2, 1}); }));
}

TEST_CASE("compress_by_index_sort: matches coordinate_compression + rank()", "[coordinate_compression]") {
	// Oracle: the two-step form, coordinate_compression(values) then rank(value) per input element.
	std::mt19937 rng(23);
	for (int iter = 0; iter < 400; ++iter) {
		int const n = int(rng() % 40), range = 1 + int(rng() % 30);
		std::vector<long long> v(n);
		for (auto &x : v) x = (long long)(rng() % range) * 1'000'000'007LL - range;
		auto const c = atl::experimental::compress_by_index_sort(v);
		atl::coordinate_compression const ref(v);
		REQUIRE(std::vector<long long>(c.universe.begin(), c.universe.end()) == std::vector<long long>(ref.begin(), ref.end()));
		REQUIRE(c.ranks.size() == v.size());
		for (std::size_t i = 0; i < v.size(); ++i) REQUIRE(c.ranks[i] == ref.rank(v[i]));
	}
	REQUIRE(atl::experimental::compress_by_index_sort(std::vector<int>{}).ranks.empty());
}

TEST_CASE("compress_by_radix_sort: matches compress", "[coordinate_compression]") {
	// Oracle: the library compress (itself checked against the two-step form above), on signed and unsigned keys
	// spanning zero and the extremes, at every digit width.
	std::mt19937_64 rng(29);
	auto const check = []<int DigitBits, typename Value>
		requires(DigitBits == 8 || DigitBits == 16) && atl::integral<Value>
	(std::vector<Value> const &v) {
		auto const got = atl::experimental::compress_by_radix_sort<DigitBits>(v);
		auto const want = atl::experimental::compress_by_pair_sort(v);
		REQUIRE(atl::compress(v).ranks == want.ranks);
		REQUIRE(std::vector<Value>(got.universe.begin(), got.universe.end())
				== std::vector<Value>(want.universe.begin(), want.universe.end()));
		REQUIRE(got.ranks == want.ranks);
	};
	for (int iter = 0; iter < 400; ++iter) {
		int const n = int(rng() % 60), range = 1 + int(rng() % 40);
		std::vector<long long> s(n);
		std::vector<unsigned> u(n);
		std::vector<short> h(n);
		for (int i = 0; i < n; ++i) {
			long long const r = (long long)(rng() % range);
			s[i] = r == 0 ? std::numeric_limits<long long>::min()
				 : r == 1 ? std::numeric_limits<long long>::max()
						  : (r - range / 2) * 1'000'000'007LL;
			u[i] = r == 1 ? ~0u : unsigned(r * 2'654'435'761u);
			h[i] = short(r * 997 - 20'000);
		}
		check.operator()<8>(s);
		check.operator()<16>(s);
		check.operator()<8>(u);
		check.operator()<16>(h);
	}

	struct event {
		int pos;
	};

	auto const by_pos = atl::experimental::compress_by_radix_sort(std::vector<event>{{-3}, {7}, {-3}}, &event::pos);
	REQUIRE(by_pos.ranks == std::vector<atl::uint32>{0, 1, 0});
	REQUIRE(atl::experimental::compress_by_radix_sort(std::vector<int>{}).ranks.empty());
	static_assert(atl::experimental::compress_by_radix_sort(std::vector<int>{5, -1, 5, 3}).ranks
				  == std::vector<atl::uint32>{2, 0, 2, 1});
}

TEST_CASE("ranks_by_search_or_gallop: every dispatch matches rank()", "[coordinate_compression]") {
	// Oracle: rank() per query, including the former dispatch and galloping losers.
	using atl::experimental::ranks_policy;
	std::mt19937 rng(31);
	ranks_policy const policies[] = {{}, {64, true}, {0, true}};
	for (int iter = 0; iter < 400; ++iter) {
		int const m = int(rng() % 200), q = int(rng() % 80), range = 1 + int(rng() % 500);
		std::vector<int> base(m), queries(q);
		for (auto &x : base) x = int(rng() % range) - range / 2;
		for (auto &x : queries) x = int(rng() % (range + 20)) - range / 2 - 10;
		if (iter % 2 == 0) std::ranges::sort(queries);
		atl::coordinate_compression const cc(base);
		std::vector<int> const universe(cc.begin(), cc.end());
		std::vector<atl::uint32> want;
		for (int const x : queries) want.push_back(atl::uint32(cc.rank(x)));
		for (auto const &policy : policies)
			REQUIRE(atl::experimental::ranks_by_search_or_gallop(universe, std::ranges::less{}, queries, policy) == want);
	}
	// the thresholds are exact and do not overflow on huge divisors
	static_assert(ranks_policy::reaches(1, std::size_t(-1), 5) && !ranks_policy::reaches(0, std::size_t(-1), 5));
	static_assert(ranks_policy::reaches(2, 3, 6) && !ranks_policy::reaches(2, 3, 7) && ranks_policy::reaches(0, 0, 0));
	REQUIRE(atl::experimental::ranks_by_search_or_gallop(std::vector<int>{1, 2, 3}, std::ranges::less{}, std::vector<int>{3, 1},
														 {64, false})
			== std::vector<atl::uint32>{2, 0});
	std::vector<int> const desc{30, 20, 10};
	REQUIRE(atl::experimental::ranks_by_search_or_gallop(desc, std::ranges::greater{}, std::vector<int>{40, 25, 25, 5}, {0, true})
			== std::vector<atl::uint32>{0, 1, 1, 3});
	REQUIRE(atl::experimental::ranks_by_search_or_gallop(desc, std::ranges::greater{}, std::vector<int>{5, 25, 40, 25}, {64, false})
			== std::vector<atl::uint32>{3, 1, 0, 1});
	static_assert(atl::experimental::ranks_by_search_or_gallop(std::vector<int>{1, 4, 9}, std::ranges::less{}, std::vector<int>{0, 4, 5, 10},
															   {0, true})
				  == std::vector<atl::uint32>{0, 1, 2, 3});
	static_assert(atl::experimental::ranks_by_search_or_gallop(std::vector<int>{1, 4, 9}, std::ranges::less{}, std::vector<int>{10, 0, 5},
															   {64, false})
				  == std::vector<atl::uint32>{3, 0, 2});
}

TEST_CASE("ranks: fused walk keeps the prefix at every descent", "[coordinate_compression]") {
	auto const check = []<std::copyable Value, std::default_initializable Compare>
		requires std::strict_weak_order<Compare const &, Value const &, Value const &>
	(atl::coordinate_compression<Value, Compare> const &cc,
	 std::vector<Value> const &queries) {
		std::vector<atl::uint32> want;
		for (auto const &x : queries) {
			auto const rank = cc.rank(x);
			atl_assert(rank <= std::numeric_limits<atl::uint32>::max());
			want.push_back(atl::uint32(rank));
		}
		REQUIRE(cc.ranks(queries) == want);
		std::list<Value> const linked(queries.begin(), queries.end());
		REQUIRE(cc.ranks(linked) == want);
		auto unsized = queries | vws::filter([](Value const &) { return true; });
		static_assert(std::ranges::forward_range<decltype(unsized)>);
		static_assert(!std::ranges::sized_range<decltype(unsized)>);
		REQUIRE(cc.ranks(unsized) == want);
		std::vector<Value> const universe(cc.begin(), cc.end());
		REQUIRE(atl::experimental::ranks_by_sortedness_prescan(universe, Compare{}, queries) == want);
	};
	atl::coordinate_compression const cc(std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80});
	for (std::size_t const position : {0u, 1u, 3u, 6u}) {
		std::vector<int> queries{5, 15, 25, 35, 45, 55, 65, 75};
		queries[position + 1] = -1;
		check(cc, queries);
	}
	for (auto const &queries : {std::vector<int>{5, 20, 20, 55, 90}, std::vector<int>{30, 30, 30, 30},
								std::vector<int>{90, 70, 50, 30, 10, 0}, std::vector<int>{40, 40, 30, 30},
								std::vector<int>{7}, std::vector<int>{}})
		check(cc, queries);
	atl::coordinate_compression<int> const empty;
	check(empty, std::vector<int>{});
	check(empty, std::vector<int>{5, 1, 5});
	atl::coordinate_compression const desc(std::vector<int>{10, 20, 30, 40}, std::identity{}, std::ranges::greater{});
	for (auto const &queries : {std::vector<int>{50, 35, 35, 5}, std::vector<int>{50, 5, 35, 35},
								std::vector<int>{50, 35, 5, 6}, std::vector<int>{5, 50}})
		check(desc, queries);
	atl::coordinate_compression const words(std::vector<std::string>{"b", "d", "f", "h"});
	check(words, std::vector<std::string>{"a", "e", "c", "z", "d"});
	check(words, std::vector<std::string>{"z", "a", "e"});
	std::vector<int> base;
	for (int i = 0; i < 1025; ++i) base.push_back(2 * i);
	atl::coordinate_compression const threshold(base);
	for (int const q : {16, 17}) {
		std::vector<int> queries;
		for (int i = 0; i < q; ++i) queries.push_back(100 - i);
		check(threshold, queries);
		std::ranges::sort(queries);
		check(threshold, queries);
	}
	std::mt19937 rng(37);
	for (int iter = 0; iter < 500; ++iter) {
		int const m = int(rng() % 200), q = int(rng() % 80), range = 1 + int(rng() % 500);
		std::vector<int> values(m), queries(q);
		for (auto &x : values) x = int(rng() % range) - range / 2;
		for (auto &x : queries) x = int(rng() % (range + 20)) - range / 2 - 10;
		if (iter % 3 == 0) std::ranges::sort(queries);
		if (iter % 3 == 1) std::ranges::sort(queries.begin(), queries.begin() + q / 2);
		check(atl::coordinate_compression(values), queries);
	}
	static_assert(atl::coordinate_compression(std::array{1, 4, 9}).ranks(std::array{0, 4, 10, 5})
				  == std::vector<atl::uint32>{0, 1, 3, 2});
}

TEST_CASE("compress: fixed radix11 edge widths and comparison fallbacks", "[coordinate_compression]") {
	auto check = []<std::copyable Value>
		requires std::totally_ordered<Value>
	(std::vector<Value> const &v) {
		auto const got = atl::compress(v);
		auto const want = atl::experimental::compress_by_pair_sort(v);
		REQUIRE(got.ranks == want.ranks);
		REQUIRE(std::ranges::equal(got.universe, want.universe));
	};
	check(std::vector<signed char>{-128, 127, 0, -1, -128});
	check(std::vector<unsigned char>{255, 0, 128, 1, 255});
	check(std::vector<short>{-32768, 32767, 0, -1});
	check(std::vector<unsigned long long>{0, ~0ULL, 1ULL << 63, 2047, 2048});
	check(std::vector<int>{});
	check(std::vector<int>{7});
	check(std::vector<int>(65, -3));
	check(std::vector<bool>{true, false, true});
	check(std::vector<double>{2.5, -1.5, 2.5});
	REQUIRE(atl::compress(std::vector<int>{3, -1, 3}, std::identity{}, std::less<int>{}).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::compress(std::vector<int>{3, -1, 3}, std::identity{}, std::less<>{}).ranks == std::vector<atl::uint32>{1, 0, 1});
}

TEST_CASE("ranks: sort-merge boundaries and unsized queries", "[coordinate_compression]") {
	for (int const m : {0, 1, 63, 64, 65, 1024, 1025}) {
		std::vector<int> base;
		for (int i = 0; i < m; ++i) base.push_back(2 * i);
		atl::coordinate_compression const cc(base);
		int const min_merge = (m + 63) / 64;
		for (int const q : {0, 1, min_merge - 1, min_merge, min_merge + 1, m, 2 * m}) {
			std::vector<int> queries;
			for (int i = 0; i < q; ++i) queries.push_back(i % 3 == 0 ? 2 * m + 1 : -i);
			std::vector<atl::uint32> want;
			for (int x : queries) want.push_back(atl::uint32(cc.rank(x)));
			auto const original = queries;
			REQUIRE(cc.ranks(queries) == want);
			auto unsized = queries | std::views::filter([](int) { return true; });
			REQUIRE(cc.ranks(unsized) == want);
			REQUIRE(queries == original);
			std::ranges::sort(queries);
			want.clear();
			for (int x : queries) want.push_back(atl::uint32(cc.rank(x)));
			REQUIRE(cc.ranks(queries) == want);
		}
	}
	atl::coordinate_compression cc(std::vector<int>{30, 20, 10}, std::identity{}, std::ranges::greater{});
	REQUIRE(cc.ranks(std::vector<int>{5, 25, 40, 25}) == std::vector<atl::uint32>{3, 1, 0, 1});
}

TEST_CASE("ranks and compress: the bulk paths on large inputs", "[coordinate_compression]") {
	int const m = 1 << 16;
	std::vector<int> base;
	for (int i = 0; i < m; ++i) base.push_back(2 * i);
	atl::coordinate_compression const cc(base);
	for (int const q : {m / 64 - 1, m / 64, m / 64 + 1, m - 1, m, m + 1, 2 * m}) {
		std::vector<int> queries;
		for (int i = 0; i < q; ++i) queries.push_back(int((atl::int64(i) * 40503) % (2 * m + 3)) - 1);
		std::vector<atl::uint32> want;
		for (int x : queries) want.push_back(atl::uint32(cc.rank(x)));
		REQUIRE(cc.ranks(queries) == want);
		auto unsized = queries | std::views::filter([](int) { return true; });
		REQUIRE(cc.ranks(unsized) == want);
	}
	auto const check = []<std::copyable Value>
		requires std::totally_ordered<Value>
	(std::vector<Value> edges) {
		for (std::size_t const n : {std::size_t(2047), std::size_t(2048), std::size_t(2049), std::size_t(1) << 16}) {
			std::vector<Value> values;
			for (std::size_t i = 0; i < n; ++i) values.push_back(edges[i % edges.size()]);
			auto const got = atl::compress(values);
			std::vector<Value> sorted = values;
			std::ranges::sort(sorted);
			auto const [last, end] = std::ranges::unique(sorted);
			sorted.erase(last, end);
			REQUIRE(std::vector<Value>(got.universe.begin(), got.universe.end()) == sorted);
			for (std::size_t i = 0; i < values.size(); ++i) REQUIRE(got.universe[got.ranks[i]] == values[i]);
		}
	};
	check(std::vector<unsigned long long>{0, ~0ULL, 1ULL << 63, 2047, 2048});
	check(std::vector<long long>{std::numeric_limits<long long>::min(), -1, 0, 1, std::numeric_limits<long long>::max()});
	check(std::vector<signed char>{-128, -1, 0, 1, 127});
	check(std::vector<short>{-32768, -1, 0, 1, 32767});
}

TEST_CASE("ranks: preserve non-copyable and single-pass query APIs", "[coordinate_compression]") {
	std::vector<std::unique_ptr<int>> values;
	for (int x : {1, 4, 9}) values.push_back(std::make_unique<int>(x));
	auto const by_value = [](std::unique_ptr<int> const &a, std::unique_ptr<int> const &b) { return *a < *b; };
	atl::coordinate_compression cc(std::move(values), std::identity{}, by_value);
	std::vector<std::unique_ptr<int>> queries;
	for (int x : {10, 0, 5, 5}) queries.push_back(std::make_unique<int>(x));
	REQUIRE(cc.ranks(queries) == std::vector<atl::uint32>{3, 0, 2, 2});
	REQUIRE(*queries[0] == 10);
	atl::coordinate_compression const ints(std::array{1, 4, 9});
	std::istringstream input("10 0 5 5");
	REQUIRE(ints.ranks(std::ranges::istream_view<int>(input)) == std::vector<atl::uint32>{3, 0, 2, 2});
}

TEST_CASE("compress: 128-bit keys preserve high bits and signed extremes", "[coordinate_compression]") {
	auto const check = []<atl::integral Integer>
		requires(std::numeric_limits<Integer>::digits >= 127)
	() {
		std::vector<Integer> const edges{(std::numeric_limits<Integer>::min)(), (std::numeric_limits<Integer>::max)(),
										 Integer(0), Integer(1), Integer(Integer(1) << 100), Integer(Integer(1) << 64)};
		for (std::size_t n : {std::size_t(17), std::size_t(2048), std::size_t(2049)}) {
			std::vector<Integer> values;
			for (std::size_t i = 0; i < n; ++i) values.push_back(edges[i % edges.size()]);
			auto const oracle = atl::experimental::compress_by_pair_sort(values);
			auto const actual = atl::compress(values);
			REQUIRE(actual.ranks == oracle.ranks);
			REQUIRE(std::ranges::equal(actual.universe, oracle.universe));
			REQUIRE(atl::experimental::compress_by_radix_sort<8>(values).ranks == oracle.ranks);
			REQUIRE(atl::experimental::compress_by_radix_sort<16>(values).ranks == oracle.ranks);
		}
	};
	check.operator()<atl::int128>();
	check.operator()<atl::uint128>();
	check.operator()<atl::detail::software_int128>();
	check.operator()<atl::detail::software_uint128>();
}

TEST_CASE("coordinate_compression: prvalue view elements retain their value category", "[coordinate_compression]") {
	auto const make = [] { return std::views::iota(0, 4) | std::views::transform([](int i) { return std::make_unique<int>(i / 2); }); };
	auto const dereference = [](std::unique_ptr<int> &&p) { return *p; };
	atl::coordinate_compression const cc(make(), dereference);
	REQUIRE(cc.size() == 2);
	REQUIRE(atl::compress(make(), dereference).ranks == std::vector<atl::uint32>{0, 0, 1, 1});
	REQUIRE(atl::experimental::compress_by_pair_sort(make(), dereference).ranks == std::vector<atl::uint32>{0, 0, 1, 1});
	REQUIRE(atl::experimental::compress_by_index_sort(make(), dereference).ranks == std::vector<atl::uint32>{0, 0, 1, 1});
	REQUIRE(atl::experimental::compress_by_radix_sort(make(), dereference).ranks == std::vector<atl::uint32>{0, 0, 1, 1});
}
