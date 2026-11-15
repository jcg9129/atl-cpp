#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using atl::uint64;

static_assert(std::invocable<atl::identity_hash const &, uint64>);
static_assert(std::invocable<atl::identity_hash const &, char>);
static_assert(!std::invocable<atl::identity_hash const &, double>); // integers only
static_assert(!std::invocable<atl::identity_hash const &, int *>);

TEST_CASE("hash: pinned seed is deterministic, distinct seeds differ", "[hash]") {
	atl::set_hash_seed(12345);
	uint64 const a = atl::hash<uint64>{}(999);
	atl::set_hash_seed(12345);
	REQUIRE(atl::hash<uint64>{}(999) == a);
	atl::set_hash_seed(67890);
	REQUIRE(atl::hash<uint64>{}(999) != a);
}

TEST_CASE("hash: avalanches structured keys", "[hash]") {
	// Keys sharing low bits (i << 20) must not cluster once hashed.
	atl::hash<uint64> h;
	constexpr std::size_t n = 200000, cap = 1 << 19; // ~0.4 load
	std::vector<int> bucket(cap, 0);
	std::size_t worst = 0;
	for (uint64 i = 0; i < n; ++i) {
		int const c = ++bucket[h(i << 20) & (cap - 1)];
		worst = std::max(worst, std::size_t(c));
	}
	REQUIRE(worst < 20); // near-uniform; a clustering hash would put ~n here
}

TEST_CASE("hash: supports the common key types", "[hash]") {
	int x = 0;
	REQUIRE(atl::hash<int>{}(-3) != 0);
	REQUIRE(atl::hash<double>{}(2.5) != 0);
	REQUIRE(atl::hash<std::string>{}(std::string("abc")) != 0);
	REQUIRE(atl::hash<std::string_view>{}(std::string_view("abc")) != 0);
	REQUIRE(atl::hash<int *>{}(&x) != 0);
	using pair_t = std::pair<int, int>;
	REQUIRE(atl::hash<pair_t>{}({1, 2}) != atl::hash<pair_t>{}({2, 1}));
	using tup_t = std::tuple<int, std::string, double>;
	REQUIRE(atl::hash<tup_t>{}({1, "k", 2.0}) != 0);

	// identity_hash returns the key unchanged
	REQUIRE(atl::identity_hash{}(uint64(42)) == 42);
}

TEST_CASE("hash: transparent string hash agrees across argument spellings", "[hash]") {
	char const *const literal = "transparent";
	std::string const owned(literal);
	std::size_t const want = atl::hash<std::string>{}(owned);
	REQUIRE(atl::hash<std::string>{}(std::string_view(owned)) == want);
	REQUIRE(atl::hash<std::string>{}(literal) == want);
	REQUIRE(atl::hash<std::string_view>{}(literal) == want);
	// hash<char const *> is the pointer hash: a different address, same contents
	std::string const copy(owned);
	REQUIRE(atl::hash<char const *>{}(owned.c_str()) != atl::hash<char const *>{}(copy.c_str()));
}

TEST_CASE("hash: enum keys", "[hash]") {
	enum class color : int { red,
							 green,
							 blue };

	enum plain : unsigned char { a,
								 b,
								 c };

	REQUIRE(atl::hash<color>{}(color::green) == atl::hash<color>{}(color::green));
	REQUIRE(atl::hash<color>{}(color::green) != atl::hash<color>{}(color::blue));
	REQUIRE(atl::hash<plain>{}(b) == atl::hash<plain>{}(b));
	// enum hashes the same as its underlying value
	REQUIRE(atl::hash<color>{}(color::blue) == atl::hash<int>{}(2));
}

TEST_CASE("hash: vector and array keys", "[hash]") {
	using atl::hash;
	// integer element buffers (bytewise path): equal content agrees, different differs
	REQUIRE(hash<std::vector<int>>{}({1, 2, 3}) == hash<std::vector<int>>{}({1, 2, 3}));
	REQUIRE(hash<std::vector<int>>{}({1, 2, 3}) != hash<std::vector<int>>{}({1, 2, 4}));
	REQUIRE(hash<std::array<int, 3>>{}({1, 2, 3}) == hash<std::array<int, 3>>{}({1, 2, 3}));

	// vector<double> takes the per-element fold, so -0.0 and +0.0 (which compare
	// equal) must hash equal — bytewise hashing would wrongly differ here.
	REQUIRE(hash<std::vector<double>>{}({-0.0, 1.5}) == hash<std::vector<double>>{}({0.0, 1.5}));

	// non-integer element types fold per element
	REQUIRE(hash<std::vector<std::string>>{}({"a", "bb"}) == hash<std::vector<std::string>>{}({"a", "bb"}));
	REQUIRE(hash<std::vector<bool>>{}({true, false, true}) == hash<std::vector<bool>>{}({true, false, true}));
}

TEST_CASE("hash: no crafted pair collides under every seed", "[hash]") {
	// Mixing each 8-byte block on its own and combining with xor-and-multiply
	// is linear in the top bit: flipping bit 63 of two consecutive MIXED blocks
	// cancels, so this pair of 16-byte keys would collide whatever the seed.
	// old_mix is that rejected block mix.
	constexpr uint64 old_multiplier = 0x9ddfea08eb382d69ULL;
	uint64 inverse = old_multiplier; // Newton iteration for the inverse mod 2^64
	for (int i = 0; i < 6; ++i) inverse *= 2 - old_multiplier * inverse;
	auto const old_mix = [&](uint64 k) {
		k *= old_multiplier;
		k ^= k >> 47;
		return k * old_multiplier;
	};
	auto const old_unmix = [&](uint64 k) {
		k *= inverse;
		k ^= k >> 47;
		return k * inverse;
	};
	std::array<uint64, 2> const first{0x1111111111111111ULL, 0x2222222222222222ULL};
	std::array<uint64, 2> const second{old_unmix(old_mix(first[0]) ^ (uint64(1) << 63)),
									   old_unmix(old_mix(first[1]) ^ (uint64(1) << 63))};
	REQUIRE(first != second);
	for (uint64 const seed : {uint64(1), uint64(2), uint64(0xdeadbeef), uint64(0x123456789abcdef)})
		REQUIRE(atl::detail::hash_bytes(first.data(), 16, seed) != atl::detail::hash_bytes(second.data(), 16, seed));

	// the same pair through the public string hash
	std::string const a(reinterpret_cast<char const *>(first.data()), 16),
		b(reinterpret_cast<char const *>(second.data()), 16);
	for (uint64 const seed : {uint64(7), uint64(99), uint64(123456789)}) {
		atl::set_hash_seed(seed);
		REQUIRE(atl::hash<std::string>{}(a) != atl::hash<std::string>{}(b));
	}

#if ATL_HAS_INT128
	// Hashing a 128-bit key as f(low + seed) ^ g(high) would give keys whose
	// g(high) agree in their low bits one home slot under every seed. These
	// high halves all have g(high) == j << 32.
	auto const unmix_splitmix64 = [](uint64 z) {
		z ^= z >> 31, z ^= z >> 62;
		z *= 0x319642b2d24d8ec3ULL;
		z ^= z >> 27, z ^= z >> 54;
		z *= 0x96de1b173f119089ULL;
		z ^= z >> 30, z ^= z >> 60;
		return z - 0x9e3779b97f4a7c15ULL;
	};
	REQUIRE(atl::detail::splitmix64(unmix_splitmix64(uint64(5) << 32)) == uint64(5) << 32);
	for (uint64 const seed : {uint64(7), uint64(99), uint64(123456789)}) {
		atl::set_hash_seed(seed);
		std::vector<uint64> slots;
		for (uint64 j = 0; j < 2000; ++j) {
			atl::uint128 const key = (atl::uint128(unmix_splitmix64(j << 32)) << 64) | 12345u;
			slots.push_back(atl::hash<atl::uint128>{}(key) & 0xfffff); // a 2^20-slot table
		}
		std::ranges::sort(slots);
		auto const distinct = std::size_t(std::ranges::unique(slots).begin() - slots.begin());
		REQUIRE(distinct > 1900); // 2000 random draws from 2^20 slots barely repeat
	}
#endif
}

TEST_CASE("hash: a packed pair of small integers is injective", "[hash]") {
	std::vector<std::size_t> seen;
	for (int a = -20; a <= 20; ++a)
		for (int b = -20; b <= 20; ++b) seen.push_back(atl::hash<std::pair<int, int>>{}({a, b}));
	std::ranges::sort(seen);
	REQUIRE(std::ranges::adjacent_find(seen) == seen.end());
	using small = std::pair<atl::int8, atl::uint16>;
	REQUIRE(atl::hash<small>{}({-1, 0}) != atl::hash<small>{}({0, 0xffff}));
	REQUIRE(atl::hash<std::pair<bool, bool>>{}({true, false}) != atl::hash<std::pair<bool, bool>>{}({false, true}));
}

namespace {
	template<typename T>
	concept has_float_value_hash = requires(T value) { atl::detail::hash_float_value(value, uint64(0)); };
	template<typename T>
	concept has_float_layout = requires {
		atl::detail::has_x87_value_bytes<T>;
		atl::detail::has_binary128_value_bytes<T>;
		atl::detail::uses_float_value_fallback<T>;
	};
	static_assert(has_float_layout<long double>);
	static_assert(!has_float_layout<int>);
	static_assert(has_float_value_hash<long double>);
	static_assert(!has_float_value_hash<int>);
	static_assert(!atl::detail::has_x87_value_bytes<double>);
	static_assert(!atl::detail::has_binary128_value_bytes<double>);
#ifdef ATL_FORCE_FALLBACKS
	static_assert(atl::detail::uses_float_value_fallback<long double> == (sizeof(long double) > 8));
#endif
}

namespace hash_test {
	struct key {
		int value;
	};

	struct unhashable {};

	struct missing_const {};

	struct throwing {};

	struct bad_result {};

	struct throwing_conversion {};

	struct converted {
		operator atl::uint64() const noexcept { return 42; }
	};

	struct throws_conversion {
		operator atl::uint64() const { return 42; }
	};
}

namespace atl {
	template<>
	struct hash<hash_test::missing_const> {
		uint64 operator()(hash_test::missing_const const &) const noexcept { return 1; }
	};

	template<>
	struct hash<hash_test::key> {
		hash_test::converted operator()(hash_test::key const &) const noexcept { return {}; }
	};

	template<>
	struct hash<hash_test::key const> {
		uint64 operator()(hash_test::key const &k) const noexcept { return uint64(k.value); }
	};

	template<>
	struct hash<hash_test::throwing> {
		uint64 operator()(hash_test::throwing const &) const { return 0; }
	};

	template<>
	struct hash<hash_test::bad_result> {
		void operator()(hash_test::bad_result const &) const noexcept {}
	};

	template<>
	struct hash<hash_test::throwing_conversion> {
		hash_test::throws_conversion operator()(hash_test::throwing_conversion const &) const noexcept { return {}; }
	};

	// The buffer and packing paths must remain available without element hashing.
	template<>
	struct hash<unsigned short const> {};
}

namespace {
	template<typename T>
	concept has_hash = requires(T const &x) { { atl::hash<T>{}(x) } noexcept -> std::convertible_to<uint64>; };
	using hash_test::key;
	using hash_test::unhashable;
	static_assert(has_hash<std::pair<key const, int>>);
	static_assert(has_hash<std::tuple<int &>>);
	static_assert(has_hash<std::tuple<hash_test::missing_const const>>);
	static_assert(!has_hash<std::pair<hash_test::missing_const const, int>>);
	static_assert(!has_hash<std::array<hash_test::missing_const const, 2>>);
	static_assert(has_hash<std::pair<bool const, int>>);
	static_assert(has_hash<std::tuple<int &, key const, std::vector<key>>>);
	static_assert(has_hash<std::vector<key>>);
	static_assert(has_hash<std::array<key, 2>>);
	static_assert(has_hash<std::array<key const, 2>>);
	static_assert(has_hash<std::pair<std::tuple<key>, std::array<key, 2>>>);
	static_assert(!has_hash<std::pair<unhashable, int>>);
	static_assert(!has_hash<std::tuple<unhashable>>);
	static_assert(!has_hash<std::vector<unhashable>>);
	static_assert(!has_hash<std::array<unhashable, 2>>);
	static_assert(!has_hash<std::vector<std::tuple<unhashable>>>);
	static_assert(!has_hash<std::tuple<hash_test::throwing>>);
	static_assert(!has_hash<std::pair<hash_test::throwing, int>>);
	static_assert(!has_hash<std::vector<hash_test::throwing>>);
	static_assert(!has_hash<std::array<hash_test::throwing, 2>>);
	static_assert(!has_hash<std::tuple<hash_test::bad_result>>);
	static_assert(!has_hash<std::tuple<hash_test::throwing_conversion>>);
	static_assert(!atl::detail::nothrow_hashable<unsigned short const>);
	static_assert(atl::detail::nothrow_hashable<key>);
	static_assert(has_hash<std::array<unsigned short const, 2>>);
	static_assert(has_hash<std::pair<unsigned short const, unsigned short const>>);
	static_assert(atl::detail::packable_hash_pair<int const, int>);
	static_assert(!atl::detail::packable_hash_pair<bool const, int>);
	static_assert(atl::detail::byte_hashable<unsigned>);
	static_assert(!atl::detail::byte_hashable<bool> && !atl::detail::byte_hashable<double>);
	static_assert(!atl::detail::byte_hashable<unsigned volatile>);

}

TEST_CASE("hash: custom composite calls", "[hash]") {
	using packed = std::pair<unsigned short const, unsigned short const>;
	using buffer = std::array<unsigned short const, 2>;
	REQUIRE(atl::hash<packed>{}({1, 2}) == atl::hash<packed>{}({1, 2}));
	REQUIRE(atl::hash<buffer>{}({1, 2}) == atl::hash<buffer>{}({1, 2}));
	int value = 17;
	REQUIRE(atl::hash<std::tuple<int &>>{}(std::tie(value)) == atl::hash<std::tuple<int>>{}({value}));
	REQUIRE(atl::hash<std::pair<bool const, int>>{}({true, 1}) == atl::hash<std::pair<bool, int>>{}({true, 1}));
	REQUIRE(atl::hash<std::pair<key const, int>>{}({{7}, 1}) == std::size_t(atl::detail::hash_combine(7, atl::hash<int>{}(1))));
	REQUIRE(atl::hash<std::vector<key>>{}({{1}, {2}}) == atl::hash<std::vector<key>>{}({{1}, {2}}));
	REQUIRE(atl::hash<std::array<key, 2>>{}({{{1}, {2}}}) == atl::hash<std::array<key, 2>>{}({{{1}, {2}}}));
}

TEST_CASE("hash: wide floats use value bytes and canonical zeros", "[hash]") {
	using limits = std::numeric_limits<long double>;
	atl::hash<long double> const h;
	for (long double value : {0.0L, -0.0L, 1.25L, -3.5L, limits::min(), limits::denorm_min(), limits::max(), limits::infinity(), -limits::infinity()}) {
		long double const rebuilt = std::isfinite(value) ? std::scalbn(std::scalbn(value, -1), 1) : std::copysign(limits::infinity(), value);
		if (value == rebuilt) {
			REQUIRE(h(value) == h(rebuilt));
			REQUIRE(atl::detail::hash_float_value(value, 7) == atl::detail::hash_float_value(rebuilt, 7));
		}
	}
	REQUIRE(h(-0.0L) == h(0.0L));
	REQUIRE(h(limits::denorm_min()) == h(std::nextafter(0.0L, 1.0L)));
	REQUIRE(h(limits::quiet_NaN()) == h(-limits::quiet_NaN()));
	REQUIRE(atl::detail::hash_float_value(-0.0L, 7) == atl::detail::hash_float_value(0.0L, 7));
	REQUIRE(atl::detail::hash_float_value(limits::denorm_min(), 7) == atl::detail::hash_float_value(std::nextafter(0.0L, 1.0L), 7));
	REQUIRE(atl::detail::hash_float_value(limits::infinity(), 7) != atl::detail::hash_float_value(-limits::infinity(), 7));
	REQUIRE(atl::detail::hash_float_value(limits::quiet_NaN(), 7) == atl::detail::hash_float_value(-limits::quiet_NaN(), 7));
	REQUIRE(atl::hash<std::vector<long double>>{}({-0.0L, 1.25L}) == atl::hash<std::vector<long double>>{}({0.0L, 1.25L}));
	if constexpr (atl::detail::has_x87_value_bytes<long double>) {
		long double original = 1.25L, altered = original;
		auto *bytes = reinterpret_cast<unsigned char *>(&altered);
		for (std::size_t i = 10; i < sizeof(altered); ++i) bytes[i] = 0xa5;
		REQUIRE(original == altered);
		REQUIRE(h(original) == h(altered));
		REQUIRE(atl::hash<std::pair<long double, int>>{}({original, 1}) == atl::hash<std::pair<long double, int>>{}({altered, 1}));
	}
}
