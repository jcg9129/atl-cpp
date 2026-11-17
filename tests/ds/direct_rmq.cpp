#include <atl/ds/direct_rmq.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>
#include <random>
#include <span>
#include <string_view>
#include <vector>

using atl::uint64;

std::vector<int> generateRandomVector(std::mt19937 &rng, atl::uint32 size, int min, int max) {
	std::vector<int> randomValues(size);

	std::uniform_int_distribution<> dist(min, max);

	for (auto &value : randomValues) {
		value = dist(rng);
	}

	return randomValues;
}

std::vector<std::pair<std::size_t, std::size_t>> generateIntervals(std::mt19937 &rng, std::size_t size) {
	std::vector<std::pair<std::size_t, std::size_t>> randomValues(size);

	std::uniform_int_distribution<> dist(0, size - 1);

	for (auto &value : randomValues) {
		value = {dist(rng), dist(rng)};
		if (value.first > value.second)
			std::swap(value.first, value.second);
		value.second++;
	}

	return randomValues;
}

TEST_CASE("Test small RMQs", "[direct_rmq]") {
	auto seed = GENERATE(take(8, random(uint64(0), std::numeric_limits<atl::uint64>::max())));

	int const MAX_VALUE = 1024;
	int const SIZE = 128;

	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	auto values = generateRandomVector(rng, SIZE, -MAX_VALUE, MAX_VALUE);
	auto intervals = generateIntervals(rng, SIZE);
	auto rmq = atl::direct_rmq(values);

	for (auto [b, e] : intervals) {
		auto found = values[rmq.position(b, e)];

		auto expected = MAX_VALUE + 1;
		for (auto i = b; i < e; ++i)
			expected = std::min(expected, values[i]);

		REQUIRE(found == expected);
		REQUIRE(rmq.fold(b, e) == expected);
	}
}

TEST_CASE("Test large RMQs", "[direct_rmq]") {
	auto seed = GENERATE(take(1, random(uint64(0), std::numeric_limits<atl::uint64>::max())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	int const MAX_SIZE = 10000000;
	int const MAX_VALUE = 100000000;
	std::uniform_int_distribution<> dist(MAX_SIZE / 2, MAX_SIZE);
	auto size = dist(rng);

	auto values = generateRandomVector(rng, size, -MAX_VALUE, MAX_VALUE);
	auto intervals = generateIntervals(rng, size);
	auto rmq = atl::direct_rmq(values);

	for (auto [b, e] : intervals) {
		auto pos = rmq.position(b, e);
		REQUIRE(pos >= b);
		REQUIRE(pos < e);
	}
}

TEST_CASE("direct_rmq: equal minima report the leftmost, on every path", "[direct_rmq]") {
	// regression: the in-block stack popped EQUAL elements, so a query inside
	// one block returned the rightmost minimum while the block table and the
	// cross-block combine returned the leftmost ({1, 1} answered 1)
	std::vector<int> const pair{1, 1}, run(128, 7);
	REQUIRE(atl::direct_rmq(pair).position(0, 2) == 0);
	REQUIRE(atl::direct_rmq(run).position(0, 128) == 0);

	auto seed = GENERATE(take(6, random(uint64(0), ~uint64(0))));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	// few distinct values, sizes around and across the 64-wide blocks
	for (std::size_t const n : {std::size_t(1), std::size_t(2), std::size_t(63), std::size_t(64), std::size_t(65),
								std::size_t(130), std::size_t(300)}) {
		std::vector<int> const values = generateRandomVector(rng, atl::uint32(n), 0, 3);
		atl::direct_rmq const smallest(values);
		atl::direct_rmq<int, std::greater<>> const largest(values);
		for (std::size_t l = 0; l < n; ++l)
			for (std::size_t r = l + 1; r <= n; ++r) {
				std::size_t low = l, high = l;
				for (std::size_t i = l + 1; i < r; ++i) {
					if (values[i] < values[low]) low = i;
					if (values[i] > values[high]) high = i;
				}
				REQUIRE(smallest.position(l, r) == low);
				REQUIRE(largest.position(l, r) == high);
			}
	}
}

// Construction admits any borrowed contiguous range of T; a temporary owning
// range (which would leave the view dangling) and a mismatched element type
// are constraint failures, not hard errors.
static_assert(std::constructible_from<atl::direct_rmq<int>, std::vector<int> &>);
static_assert(std::constructible_from<atl::direct_rmq<int>, std::span<int const>>);
static_assert(std::constructible_from<atl::direct_rmq<char>, std::string_view>);
static_assert(!std::constructible_from<atl::direct_rmq<int>, std::vector<int>>);
static_assert(!std::constructible_from<atl::direct_rmq<int>, std::vector<long> &>);
static_assert(!std::convertible_to<std::vector<int> &, atl::direct_rmq<int>>); // explicit

TEST_CASE("direct_rmq: default-constructed, then assigned; borrowed rvalue views", "[direct_rmq]") {
	atl::direct_rmq<int> rmq;
	REQUIRE(rmq.empty());
	REQUIRE(rmq.size() == 0);
	std::vector<int> const values{5, 3, 8, 3, 1, 9};
	rmq = atl::direct_rmq<int>(values);
	REQUIRE(rmq.size() == values.size());
	REQUIRE(!rmq.empty());
	REQUIRE(rmq.position(0, 4) == 1);
	REQUIRE(rmq.fold(0, 6) == 1);

	atl::direct_rmq const spanned{std::span(values).subspan(2)};
	REQUIRE(spanned.size() == 4);
	REQUIRE(spanned.position(0, 4) == 2);
	atl::direct_rmq<int, std::greater<>> const largest(values, std::greater<>{});
	REQUIRE(largest.fold(0, 6) == 9);
}
