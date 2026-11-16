#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>
#include "../expect_abort.hpp"

using atl::counting_sort;
using atl::uint32;
using atl::uint64;

namespace {
	struct item {
		int key;
		int tag; // original index — catches stability violations
		bool operator==(item const &) const = default;
	};
}

static_assert(atl::counting_sortable<std::vector<int> &>);
static_assert(!atl::counting_sortable<std::vector<int> const &>); // cannot permute in place
static_assert(!atl::counting_sortable<std::vector<double> &>);
static_assert(!atl::counting_sortable<std::vector<atl::int128> &>); // offsets are uint64

TEST_CASE("counting_sort: stable, matches std::stable_sort by projected key", "[utils]") {
	auto seed = GENERATE(take(40, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = static_cast<std::size_t>(rng() % 300);
	int const base = int(rng() % 400) - 200; // signed keys, incl. negative
	int const span = 1 + int(rng() % 120);
	std::vector<item> v(n);
	for (std::size_t i = 0; i < n; ++i) v[i] = {base + int(rng() % uint64(span)), int(i)};

	std::vector<item> want = v;
	std::ranges::stable_sort(want, {}, &item::key);

	SECTION("argsort idiom: sorting an index range gives the stable permutation") {
		std::vector<uint32> p(n);
		std::iota(p.begin(), p.end(), 0u); // Xcode 16.4 lacks ranges::iota.
		counting_sort(p, [&](uint32 i) { return v[i].key; });
		for (std::size_t k = 0; k < n; ++k) REQUIRE(v[p[k]] == want[k]);
		// the permutation is a bijection of [0, n)
		std::vector<char> hit(n, 0);
		for (uint32 idx : p) {
			REQUIRE(idx < n);
			REQUIRE(!std::exchange(hit[idx], char(1)));
		}
	}

	SECTION("in-place counting_sort equals std::stable_sort element-for-element") {
		counting_sort(v, &item::key);
		REQUIRE(v == want);
	}
}

TEST_CASE("counting_sort: identity projection, edge cases, cross-type reuse", "[utils]") {
	std::vector<int> a{5, 3, 3, 9, 1, 5, -2, 3, 1'000'000, -1'000'000};
	auto want = a;
	std::ranges::sort(want);
	counting_sort(a); // identity projection on a range of ints
	REQUIRE(a == want);

	std::vector<int> empty;
	counting_sort(empty);
	REQUIRE(empty.empty());

	std::vector<int> one{7};
	counting_sort(one);
	REQUIRE(one == std::vector<int>{7});

	std::vector<int> equal(50, 4);
	counting_sort(equal);
	REQUIRE(equal == std::vector<int>(50, 4));

	// the shared uint32 scratch is reused across element types
	std::vector<std::string> words{"cc", "aa", "bb", "aa"};
	counting_sort(words, [](std::string const &s) { return int(s[0]); });
	REQUIRE(words == std::vector<std::string>{"aa", "aa", "bb", "cc"});

	atl::counting_sort_clear(); // releasing the scratch must leave later calls correct
	counting_sort(a);
	REQUIRE(a == want);
}

TEST_CASE("counting_sort: extreme key magnitudes with a small span", "[utils]") {
	using atl::int64;
	// int64 keys straddling the type's extremes — the span (max-min) is tiny, but
	// a naive `max - min` in int64 would overflow (UB). Must sort correctly.
	int64 const imin = std::numeric_limits<int64>::min();
	int64 const imax = std::numeric_limits<int64>::max();
	// each cluster has a tiny span; a naive int64 max-min across the extremes overflows
	std::vector<int64> lows{imin + 2, imin, imin + 1, imin + 2, imin};
	auto wl = lows;
	std::ranges::stable_sort(wl);
	counting_sort(lows);
	REQUIRE(lows == wl);

	std::vector<int64> highs{imax, imax - 3, imax - 1, imax, imax - 2};
	auto wh = highs;
	std::ranges::stable_sort(wh);
	counting_sort(highs);
	REQUIRE(highs == wh);

	// uint64 keys with the high bit set — casting to int64 for min/max would
	// misorder them. Values above 2^63 must still sort above small ones.
	using atl::uint64;
	uint64 const big = uint64(1) << 63;
	std::vector<uint64> hi{big + 5, big, big + 5, big + 2, big + 1};
	auto whi = hi;
	std::ranges::stable_sort(whi);
	counting_sort(hi);
	REQUIRE(hi == whi);
}

TEST_CASE("bucket_by_key: offsets and stable order against a direct bucketing", "[utils]") {
	auto seed = GENERATE(take(30, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = static_cast<std::size_t>(rng() % 400), buckets = 1 + static_cast<std::size_t>(rng() % 50);
	std::vector<uint32> keys(n);
	for (auto &k : keys) k = uint32(rng() % buckets);
	std::vector<uint32> offsets(buckets + 1), order(n);
	atl::detail::bucket_by_key(buckets, [&](std::size_t i) { return keys[i]; }, std::span<uint32>(offsets), std::span<uint32>(order));

	std::vector<std::vector<uint32>> want(buckets);
	for (uint32 i = 0; i < n; ++i) want[keys[i]].push_back(i);
	REQUIRE(offsets[0] == 0);
	REQUIRE(offsets[buckets] == n);
	for (std::size_t b = 0; b < buckets; ++b)
		REQUIRE(std::vector<uint32>(order.begin() + offsets[b], order.begin() + offsets[b + 1]) == want[b]);
}

TEST_CASE("bucket_by_key: empty input, empty buckets, and asserted preconditions", "[utils]") {
	std::vector<uint32> offsets(4, 99), order;
	auto const none = [](std::size_t) { return std::size_t(0); };
	atl::detail::bucket_by_key(3, none, std::span<uint32>(offsets), std::span<uint32>(order));
	REQUIRE(offsets == std::vector<uint32>{0, 0, 0, 0});

	std::vector<std::size_t> const keys{2, 0, 2};
	order.resize(3);
	auto const key = [&](std::size_t i) { return keys[i]; };
	atl::detail::bucket_by_key(3, key, std::span<uint32>(offsets), std::span<uint32>(order));
	REQUIRE(offsets == std::vector<uint32>{0, 1, 1, 3});
	REQUIRE(order == std::vector<uint32>{1, 0, 2});

	CHECK(atl_test::aborts([&] { atl::detail::bucket_by_key(2, key, std::span<uint32>(offsets).first(3), std::span<uint32>(order)); }));
	CHECK(atl_test::aborts([&] { atl::detail::bucket_by_key(3, key, std::span<uint32>(offsets).first(3), std::span<uint32>(order)); }));
}
