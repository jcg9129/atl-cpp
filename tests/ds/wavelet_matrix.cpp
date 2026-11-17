#include <atl/ds/wavelet_matrix.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;

TEST_CASE("wavelet_matrix: all queries vs brute force", "[wavelet_matrix]") {
	auto seed = GENERATE(take(6, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 80;
	// mix wide values with a small clustered alphabet so duplicates and full
	// 30-bit values both occur
	std::vector<uint32> a(n);
	for (auto &v : a) v = rng() % 2 ? uint32(rng() % 5) : uint32(rng() % 1'000'000'000);

	atl::wavelet_matrix wm(a);
	REQUIRE(wm.size() == n);

	for (std::size_t i = 0; i < n; ++i) REQUIRE(wm.at(i) == a[i]);

	for (int step = 0; step < 300; ++step) {
		std::size_t const l = rng() % n, r = l + 1 + rng() % (n - l);
		std::vector<uint32> slice(a.begin() + long(l), a.begin() + long(r));
		std::ranges::sort(slice);

		std::size_t const k = rng() % (r - l);
		REQUIRE(wm.kth(l, r, k) == slice[k]);

		uint32 const x = rng() % 2 ? uint32(rng() % 8) : uint32(rng() % 2'000'000'000);
		std::size_t const want = std::ranges::lower_bound(slice, x) - slice.begin();
		REQUIRE(wm.count_less(l, r, x) == want);

		uint32 lo = uint32(rng() % 1'000'000'000), hi = uint32(rng() % 1'000'000'000);
		if (lo > hi) std::swap(lo, hi);
		std::size_t const cnt = std::ranges::lower_bound(slice, hi) - std::ranges::lower_bound(slice, lo);
		REQUIRE(wm.count(l, r, lo, hi) == cnt);
	}
}

TEST_CASE("wavelet_matrix: edges", "[wavelet_matrix]") {
	atl::wavelet_matrix<uint32> empty(std::vector<uint32>{});
	REQUIRE(empty.empty());
	REQUIRE(empty.count_less(0, 0, 5) == 0);

	atl::wavelet_matrix zeros(std::vector<uint32>(4, 0)); // width collapses to 1 level
	REQUIRE(zeros.kth(0, 4, 3) == 0);
	REQUIRE(zeros.count_less(0, 4, 0) == 0);
	REQUIRE(zeros.count_less(0, 4, 1) == 4);
	REQUIRE(zeros.count_less(1, 3, 3'000'000'000u) == 2); // upper wider than the matrix

	std::vector<uint32> big{std::numeric_limits<uint32>::max(), 0, std::numeric_limits<uint32>::max()};
	atl::wavelet_matrix wb(big);
	REQUIRE(wb.at(0) == std::numeric_limits<uint32>::max());
	REQUIRE(wb.kth(0, 3, 1) == std::numeric_limits<uint32>::max());
	REQUIRE(wb.count(0, 3, 0, std::numeric_limits<uint32>::max()) == 1);
}

TEST_CASE("wavelet_matrix: a length past uint32 trips the entry assert", "[wavelet_matrix]") {
	if constexpr (sizeof(std::size_t) > sizeof(uint32)) {
		auto const too_long = std::views::iota(uint64(0), (uint64(1) << 32) + 1);
		REQUIRE(atl_test::aborts([&] { atl::wavelet_matrix<uint64> wm(too_long); }));
		REQUIRE(atl_test::aborts([&] { atl::detail::bit_vector bv(std::size_t(too_long.size())); }));
	}
}

TEST_CASE("wavelet_matrix: the closed upper range [lo, max] via count_less", "[wavelet_matrix]") {
	constexpr uint32 max = std::numeric_limits<uint32>::max();
	std::vector<uint32> const a{max, 0, max, 7, max - 1};
	atl::wavelet_matrix const wm(a);
	REQUIRE(wm.count(0, 5, 7, max) == 2); // half-open: max never counted
	REQUIRE(5 - wm.count_less(0, 5, 7) == 4);
	REQUIRE(5 - wm.count_less(0, 5, max) == 2); // exactly the max values
}

TEST_CASE("wavelet_matrix: a value that does not fit T trips the entry assert", "[wavelet_matrix]") {
	REQUIRE(atl_test::aborts([] { atl::wavelet_matrix<uint32> wm(std::vector<int>{1, -1}); }));
	REQUIRE(atl_test::aborts([] { atl::wavelet_matrix<uint32> wm(std::vector<uint64>{uint64(1) << 32}); }));
	atl::wavelet_matrix<atl::uint8> fits(std::vector<int>{0, 255, 7});
	REQUIRE(fits.at(1) == 255);
}

namespace {
	/** @brief Brute-force check of every query on @p a, over @p queries random ranges. */
	template<atl::unsigned_integral T>
	void check_against_brute(std::vector<T> const &a, std::mt19937_64 &rng, int queries) {
		std::size_t const n = a.size();
		atl::wavelet_matrix<T> const wm(a);
		for (std::size_t i = 0; i < n; ++i) REQUIRE(wm.at(i) == a[i]);
		for (int step = 0; step < queries; ++step) {
			std::size_t const l = rng() % n, r = l + 1 + rng() % (n - l);
			std::vector<T> slice(a.begin() + long(l), a.begin() + long(r));
			std::ranges::sort(slice);
			std::size_t const k = rng() % (r - l);
			REQUIRE(wm.kth(l, r, k) == slice[k]);
			T const x = rng() % 4 ? a[rng() % n] + T(rng() % 2) : T(rng());
			REQUIRE(wm.count_less(l, r, x) == std::size_t(std::ranges::lower_bound(slice, x) - slice.begin()));
		}
	}
}

// lengths spanning many rank words and the n % 64 == 0 boundary, which rests on
// the padding word, and every element width, including uint64 levels at and above 32.
TEST_CASE("wavelet_matrix: many-word lengths and every element width", "[wavelet_matrix]") {
	auto seed = GENERATE(take(3, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : {std::size_t(63), std::size_t(64), std::size_t(65), std::size_t(128),
								std::size_t(1000 + rng() % 4000)}) {
		std::vector<uint32> a32(n);
		for (auto &v : a32) v = uint32(rng() % 3 ? rng() % 1000 : rng());
		check_against_brute(a32, rng, 200);
		std::vector<std::uint8_t> a8(n);
		for (auto &v : a8) v = std::uint8_t(rng());
		check_against_brute(a8, rng, 200);
		std::vector<std::uint16_t> a16(n);
		for (auto &v : a16) v = std::uint16_t(rng() % 2 ? rng() % 16 : rng());
		check_against_brute(a16, rng, 200);
		std::vector<uint64> a64(n);
		for (auto &v : a64) v = rng() % 2 ? (uint64(1) << 32) + rng() % 1000 : rng();
		check_against_brute(a64, rng, 200);
	}
}

TEST_CASE("wavelet_matrix: bit_vector ranks vs brute force", "[wavelet_matrix]") {
	auto seed = GENERATE(take(3, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : {std::size_t(0), std::size_t(1), std::size_t(64), std::size_t(127), std::size_t(5000)}) {
		std::vector<bool> bits(n);
		atl::detail::bit_vector bv(n);
		for (std::size_t i = 0; i < n; ++i)
			if (rng() % 3 == 0) bits[i] = true, bv.set(i);
		bv.finish();
		REQUIRE(bv.size() == n);
		uint32 ones = 0;
		for (std::size_t i = 0; i <= n; ++i) {
			REQUIRE(bv.rank1(i) == ones);
			REQUIRE(bv.rank0(i) == uint32(i) - ones);
			if (i < n) {
				REQUIRE(bv.at(i) == bits[i]);
				ones += bits[i];
			}
		}
		REQUIRE(bv.zeros() == uint32(n) - ones);
	}
}

TEST_CASE("wavelet_matrix: query entry points retain bounds checks", "[wavelet_matrix]") {
	atl::detail::bit_vector unfinished(2);
	REQUIRE(atl_test::aborts([&] { (void)unfinished.rank1(0); }));
	REQUIRE(atl_test::aborts([&] { (void)unfinished.rank0(0); }));
	atl::detail::bit_vector bits(2);
	bits.finish();
	REQUIRE(atl_test::aborts([&] { (void)bits.rank1(3); }));
	REQUIRE(atl_test::aborts([&] { (void)bits.rank0(3); }));
	REQUIRE(atl_test::aborts([&] { (void)bits.at(2); }));
	atl::wavelet_matrix<unsigned> const matrix(std::vector{1u, 2u});
	REQUIRE(atl_test::aborts([&] { (void)matrix.at(2); }));
	REQUIRE(atl_test::aborts([&] { (void)matrix.kth(0, 2, 2); }));
	REQUIRE(atl_test::aborts([&] { (void)matrix.count_less(0, 3, 1); }));
}
