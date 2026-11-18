#include <atl/nt/mod_int.hpp>
#include <atl/nt/multiplicative.hpp>
#include <catch2/catch.hpp>

#include <vector>

using atl::uint64;

namespace {
	// constructible from 1 and multipliable, but not default-constructible: the
	// result table could not be sized
	struct no_default {
		uint64 v;

		constexpr explicit no_default(uint64 x) : v(x) {}

		friend constexpr no_default operator*(no_default a, no_default b) { return no_default(a.v * b.v); }
	};

	struct totient_rule {
		uint64 at(uint64 p, uint64, uint64 v, std::vector<uint64> const &) const { return v / p * (p - 1); }
	};

	using by_value = uint64 (*)(uint64, uint64, uint64, std::vector<uint64> const &);
}

static_assert((no_default(2) * no_default(3)).v == 6);
static_assert(atl::detail::multiplicative_value<uint64>);
static_assert(!atl::detail::multiplicative_value<no_default>);
static_assert(atl::partial_multiplicative_function<by_value, uint64>);
static_assert(!atl::partial_multiplicative_function<int, uint64>);
// a member pointer is invocable (with the object first) but not with this signature
static_assert(!atl::partial_multiplicative_function<decltype(&totient_rule::at), uint64>);

TEST_CASE("Test Sigma 0", "[sigma]") {
	auto sigma = atl::divisor_power_sum_range<uint64>(128, 0);
	auto num_divisors = atl::divisor_count_range(128);

	for (auto i = 0; i < 128; ++i) {
		REQUIRE(sigma[i] == num_divisors[i]);
	}
}

TEST_CASE("Test Sigma 1", "[sigma]") {
	auto sigma = atl::divisor_power_sum_range<uint64>(128, 1);
	auto num_divisors = atl::divisor_sum_range(128);

	for (auto i = 0; i < 128; ++i) {
		REQUIRE(sigma[i] == num_divisors[i]);
	}
}

TEST_CASE("Test Sigma 6", "[sigma]") {
	auto sigma = atl::divisor_power_sum_range<uint64>(31, 6);
	// Sequence from OEIS: https://oeis.org/A013954
	std::vector<uint64> target = {0, 1, 65, 730, 4161, 15626, 47450, 117650, 266305, 532171, 1015690, 1771562,
								  3037530, 4826810, 7647250, 11406980, 17043521, 24137570, 34591115, 47045882,
								  65019786, 85884500, 115151530, 148035890, 194402650, 244156251, 313742650,
								  387952660, 489541650, 594823322, 741453700};

	for (auto i = 0; i < 30; ++i) {
		REQUIRE(sigma[i] == target[i]);
	}
}

TEST_CASE("Test Sigma 24", "[sigma]") {
	using Z = atl::mod_int64<1'000'000'000'000'000'000>;
	auto sigma = atl::divisor_power_sum_range<Z>(14, 24);
	// Sequence from OEIS: https://oeis.org/A013972
	// 1, 16777217, 282429536482, 281474993487873, 59604644775390626, 4738381620767930594, 191581231380566414402,
	// 4722366764344638701569, 79766443077154939399843, 1000000059604644792167842, 9849732675807611094711842,
	// 79496851942053939878082786, 542800770374370512771595362
	std::vector<uint64> target = {0, 1,
								  16777217,
								  282429536482,
								  281474993487873,
								  59604644775390626,
								  738381620767930594,
								  581231380566414402,
								  366764344638701569,
								  443077154939399843,
								  59604644792167842,
								  675807611094711842,
								  942053939878082786,
								  374370512771595362};

	for (auto i = 0; i < 14; ++i) {
		REQUIRE(sigma[i] == Z(target[i]));
	}
}
