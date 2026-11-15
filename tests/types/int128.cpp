#include <atl/types/int128_format.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <format>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include "../expect_abort.hpp"

namespace {
	using unsigned_integer = atl::detail::software_uint128;
	using signed_integer = atl::detail::software_int128;
	using namespace atl::detail;

	/** @brief Check whether an arithmetic expression is accepted without implicit floating or enum operands. */
	template<typename Left, typename Right>
	concept has_addition = requires(Left left, Right right) { left + right; };
	/** @brief Check whether three-way comparison participates. */
	template<typename Left, typename Right>
	concept has_three_way = requires(Left left, Right right) { left <=> right; };

	/** @brief An enum that must not enter software arithmetic implicitly. */
	enum unscoped { enum_value = 1 };
	/** @brief A scoped enum that must not enter software arithmetic implicitly. */
	enum class scoped { value = 1 };
	static_assert(!has_addition<unsigned_integer, double> && !has_addition<double, signed_integer>);
	static_assert(!has_addition<signed_integer, unscoped> && !has_addition<unscoped, unsigned_integer>);
	static_assert(!has_addition<signed_integer, scoped>);
	static_assert(!has_three_way<unsigned_integer, int> && !has_three_way<int, unsigned_integer>);
	static_assert(!has_three_way<signed_integer, unsigned_integer> && !has_three_way<unsigned_integer, signed_integer>);
	static_assert(!has_three_way<signed_integer, bool> && !has_three_way<bool, unsigned_integer>);
	static_assert(has_three_way<signed_integer, unsigned long long> && has_three_way<unsigned_integer, unsigned long long>);
	static_assert(std::same_as<decltype(1 << unsigned_integer{3}), int>);
	static_assert(std::same_as<decltype(true >> signed_integer{0}), int>);
	static_assert(std::same_as<decltype(signed_integer{1} << atl::uint64{3}), signed_integer>);
	static_assert(std::same_as<decltype(false ? signed_integer{} : unsigned_integer{}), unsigned_integer>);
	static_assert(std::is_convertible_v<signed_integer, unsigned_integer> && !std::is_convertible_v<unsigned_integer, signed_integer>);
	static_assert(!std::is_convertible_v<double, unsigned_integer> && !std::is_convertible_v<signed_integer, int>);
	static_assert(std::is_trivially_copyable_v<unsigned_integer> && std::is_trivially_copyable_v<signed_integer>);
	static_assert(std::is_trivially_default_constructible_v<unsigned_integer> && std::is_trivially_default_constructible_v<signed_integer>);
	static_assert(std::is_standard_layout_v<unsigned_integer> && std::is_standard_layout_v<signed_integer>);
	static_assert(sizeof(unsigned_integer) == 16 && alignof(unsigned_integer) == alignof(std::uint64_t));

	/** @brief Prove that the concrete classes are valid structural template arguments. */
	template<unsigned_integer Value>
	struct structural_value {};

	using structural_proof = structural_value<unsigned_integer{1}>;

	static_assert(atl::traits::is_signed_integral_v<signed_integer const volatile>);
	static_assert(atl::traits::is_unsigned_integral_v<unsigned_integer const volatile>);
	static_assert(std::same_as<atl::traits::make_unsigned_t<signed_integer const volatile>, unsigned_integer const volatile>);
	static_assert(std::same_as<atl::traits::make_signed_t<unsigned_integer const volatile>, signed_integer const volatile>);
	static_assert(std::same_as<std::common_type_t<unsigned_integer const, signed_integer volatile>, unsigned_integer>);
	static_assert(!std::is_integral_v<signed_integer> && !std::is_integral_v<unsigned_integer>);

#if defined(ATL_FORCE_FALLBACKS) || defined(ATL_NO_INT128)
	static_assert(std::same_as<atl::int128, signed_integer> && std::same_as<atl::uint128, unsigned_integer>);
#endif
#if defined(ATL_FORCE_FALLBACKS) || defined(ATL_PORTABLE_WIDE_ARITHMETIC)
	static_assert(uses_portable_wide_arithmetic);
#endif

	constexpr bool constexpr_arithmetic() {
		constexpr unsigned_integer a{UINT64_MAX, 7}, b{3, 1};
		static_assert(a + 1 == unsigned_integer{0, 8});
		static_assert((a + b) - b == a && (a ^ b) == ((a | b) & ~(a & b)));
		static_assert((a * 3) / 3 == a && (a / b) * b + a % b == a);
		static_assert((a << 64) == unsigned_integer{0, UINT64_MAX});
		static_assert((a >> 64) == 7 && (a << 0) == a && (a >> 127) == 0);
		static_assert(signed_integer{-17} / 5 == -3 && signed_integer{-17} % 5 == -2);
		static_assert((signed_integer{-17} >> 64) == -1);
		static_assert(double(unsigned_integer{1} << 100) == 0x1p100);
		static_assert(signed_integer(-1.75) == -1);
		static_assert(multiply_wide(UINT64_MAX, UINT64_MAX).high == UINT64_MAX - 1);
		static_assert(multiply_high_wide({UINT64_MAX, UINT64_MAX}, {UINT64_MAX, UINT64_MAX}).low == UINT64_MAX - 1);
		static_assert(add_with_carry(UINT64_MAX, 0, 1).carry == 1);
		static_assert(subtract_with_borrow(0, UINT64_MAX, 1).borrow == 1);
		static_assert(divide_wide_bounded(UINT64_MAX, 0, 1).quotient == UINT64_MAX);
		auto u = a;
		u += b;
		u -= b;
		u *= 3;
		u /= 3;
		u %= a;
		u |= 3;
		u ^= 1;
		u &= 2;
		u <<= 65;
		u >>= 65;
		if (u++ != 2 || --u != 2 || u-- != 2 || ++u != 2) return false;
		auto v = signed_integer{-7};
		v += 2;
		v -= 3;
		v *= 2;
		v /= 3;
		v %= 4;
		return v == -1 && !unsigned_integer{} && +v == v && -v == 1;
	}

	static_assert(constexpr_arithmetic());
	static_assert(atl::detail::floor_midpoint(std::numeric_limits<signed_integer>::min(), std::numeric_limits<signed_integer>::max()) == -1);
	static_assert(atl::detail::floor_midpoint(unsigned_integer{}, std::numeric_limits<unsigned_integer>::max()) == (unsigned_integer{1} << 127) - 1);

	// Small exact values give the same oracle on targets without a native wide integer.
	template<builtin_integer Integer>
	void check_portable_builtin() {
		Integer const small = Integer(1);
		unsigned_integer const u{42};
		signed_integer const s{-42};
		static_assert(std::same_as<decltype(u + small), unsigned_integer>);
		static_assert(std::same_as<decltype(s + small), signed_integer>);
		REQUIRE(u + small == 43);
		REQUIRE(small + s == -41);
		REQUIRE(u - small == 41);
		REQUIRE(small - s == 43);
		REQUIRE(u * small == 42);
		REQUIRE(small * s == -42);
		REQUIRE(u / small == 42);
		REQUIRE(small / s == 0);
		REQUIRE(s % small == 0);
		REQUIRE(small % u == 1);
		REQUIRE((u & small) == 0);
		REQUIRE((small | u) == 43);
		REQUIRE((s ^ small) == -41);
		REQUIRE(u > small);
		REQUIRE(small > s);
		REQUIRE(u != small);
		REQUIRE((u << small) == 84);
		REQUIRE((s >> small) == -21);
		REQUIRE((small << unsigned_integer{2}) == 4);
		REQUIRE((small >> signed_integer{1}) == 0);
		REQUIRE(Integer(unsigned_integer(small)) == small);
		REQUIRE(Integer(signed_integer(small)) == small);
	}

	/** @brief Edge patterns covering both limb boundaries and signed extrema. */
	std::array<unsigned_integer, 12> const edges = {{{0}, {1}, {UINT64_MAX}, {0, 1}, {1, 1}, {std::uint64_t(1) << 63}, {UINT64_MAX, UINT64_MAX >> 1}, {0, std::uint64_t(1) << 63}, {UINT64_MAX, UINT64_MAX}, {0, UINT64_MAX}, {1, UINT64_MAX}, {UINT64_MAX, 1}}};

#if defined(__SIZEOF_INT128__)
	/** @brief Native unsigned oracle for a software bit pattern. */
	__uint128_t native(unsigned_integer value) { return (__uint128_t(value.high) << 64) | value.low; }

	/** @brief Native signed oracle for a software bit pattern. */
	__int128_t native(signed_integer value) { return __int128_t(native(unsigned_integer(value))); }

	/** @brief Identity for built-in operands. */
	template<builtin_integer Integer>
	Integer native(Integer value) { return value; }

	/** @brief Match a software bit pattern against a native result without signed overflow. */
	template<software_integer Integer, builtin_integer Native>
	void matches(Integer actual, Native expected) {
		REQUIRE(native(unsigned_integer(actual)) == __uint128_t(expected));
	}

	// The native reference intentionally compares mixed signedness: that is the behaviour under test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

	/** @brief Test every mixed arithmetic and comparison operator and its result type. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	void check_pair(Left left, Right right) {
		using Common = decltype(native(left) + native(right));
		using Expected = std::conditional_t<std::same_as<Common, __uint128_t>, unsigned_integer, signed_integer>;
		static_assert(std::same_as<decltype(left + right), Expected>);
		static_assert(std::same_as<decltype(left - right), Expected>);
		static_assert(std::same_as<decltype(left * right), Expected>);
		static_assert(std::same_as<decltype(left / right), Expected>);
		static_assert(std::same_as<decltype(left % right), Expected>);
		static_assert(std::same_as<decltype(left & right), Expected>);
		static_assert(std::same_as<decltype(left | right), Expected>);
		static_assert(std::same_as<decltype(left ^ right), Expected>);
		static_assert(std::same_as<std::common_type_t<Left, Right>, Expected>);
		static_assert(has_three_way<Left, Right> == requires { native(left) <=> native(right); });
		auto const a = native(left);
		auto const b = native(right);
		matches(left + right, __uint128_t(a) + __uint128_t(b));
		matches(left - right, __uint128_t(a) - __uint128_t(b));
		matches(left * right, __uint128_t(a) * __uint128_t(b));
		matches(left & right, Common(a) & Common(b));
		matches(left | right, Common(a) | Common(b));
		matches(left ^ right, Common(a) ^ Common(b));
		REQUIRE((left == right) == (a == b));
		REQUIRE((left != right) == (a != b));
		REQUIRE((left < right) == (a < b));
		REQUIRE((left <= right) == (a <= b));
		REQUIRE((left > right) == (a > b));
		REQUIRE((left >= right) == (a >= b));
		if constexpr (has_three_way<Left, Right>) REQUIRE((left <=> right) == (a <=> b));
#define CHECK_MIXED_COMPOUND(operation)                   \
	{                                                     \
		auto value = left;                                \
		auto const expected = Left(left operation right); \
		REQUIRE(&(value operation## = right) == &value);  \
		REQUIRE(value == expected);                       \
	}
		CHECK_MIXED_COMPOUND(+);
		CHECK_MIXED_COMPOUND(-);
		CHECK_MIXED_COMPOUND(*);
		CHECK_MIXED_COMPOUND(&);
		CHECK_MIXED_COMPOUND(|);
		CHECK_MIXED_COMPOUND(^);

		if (b != 0 && !(std::same_as<Common, __int128_t> && __uint128_t(a) == (__uint128_t(1) << 127) && __uint128_t(b) == ~__uint128_t(0))) {
			matches(left / right, Common(a) / Common(b));
			matches(left % right, Common(a) % Common(b));
			CHECK_MIXED_COMPOUND(/);
			CHECK_MIXED_COMPOUND(%);
		}
#undef CHECK_MIXED_COMPOUND
	}

#pragma GCC diagnostic pop

	/** @brief Check promoted left result types and both shift assignments for each supported operand pair. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	void check_shift_pair(Left left, Right right) {
		using Expected = decltype(+native(left));
		using Result = std::conditional_t<software_integer<Left>, Left, Expected>;
		static_assert(std::same_as<decltype(left << right), Result>);
		static_assert(std::same_as<decltype(left >> right), Result>);
		auto const count = unsigned(native(right));
		auto const expected_left = Expected(__uint128_t(native(left)) << count);
		auto const expected_right = Expected(native(left) >> count);
		if constexpr (software_integer<Left>) {
			matches(left << right, expected_left);
			matches(left >> right, expected_right);
		} else {
			REQUIRE((left << right) == expected_left);
			REQUIRE((left >> right) == expected_right);
		}
		auto changed = left;
		REQUIRE(&(changed <<= right) == &changed);
		REQUIRE(changed == Left(left << right));
		changed = left;
		REQUIRE(&(changed >>= right) == &changed);
		REQUIRE(changed == Left(left >> right));
	}

	/** @brief Check mixed operands and conversions for one built-in spelling. */
	template<builtin_integer Integer>
	void check_builtin() {
		for (auto const bits : edges) {
			check_shift_pair(bits, Integer(3));
			check_shift_pair(signed_integer(bits), Integer(3));
			for (Integer const value : {Integer(0), Integer(1), Integer(-1), std::numeric_limits<Integer>::min(), std::numeric_limits<Integer>::max()}) {
				check_pair(bits, value);
				check_pair(value, bits);
				check_pair(signed_integer(bits), value);
				check_pair(value, signed_integer(bits));
				check_shift_pair(value, unsigned_integer{3});
				check_shift_pair(value, signed_integer{3});
				matches(unsigned_integer(value), __uint128_t(value));
				matches(signed_integer(value), __uint128_t(value));
			}
			if constexpr (std::same_as<Integer, bool>) {
				REQUIRE(bool(bits) == bool(native(bits)));
				REQUIRE(bool(signed_integer(bits)) == bool(native(signed_integer(bits))));
			} else {
				REQUIRE(Integer(bits) == Integer(native(bits)));
				REQUIRE(Integer(signed_integer(bits)) == Integer(native(signed_integer(bits))));
			}
		}
	}

	/** @brief Check floating conversions at random patterns and rounding boundaries. */
	template<std::floating_point Floating>
	void check_floating() {
		std::mt19937_64 random(1781);
		for (int iteration = 0; iteration < 2000; ++iteration) {
			unsigned_integer const value{random(), random()};
			REQUIRE(Floating(value) == Floating(native(value)));
			REQUIRE(Floating(signed_integer(value)) == Floating(native(signed_integer(value))));
			Floating const small = Floating(std::int64_t(random())) / Floating(128);
			matches(signed_integer(small), __int128_t(small));
			matches(unsigned_integer(std::fabs(small)), __uint128_t(std::fabs(small)));
		}
		for (int bit = 1; bit < 128; ++bit) {
			auto const power = unsigned_integer{1} << bit;
			for (auto const value : {power - 1, power, power + 1}) REQUIRE(Floating(value) == Floating(native(value)));
		}
		constexpr int precision = std::numeric_limits<Floating>::digits;
		if constexpr (precision < 128) {
			auto const leading = unsigned_integer{1} << 127;
			auto const half = unsigned_integer{1} << (127 - precision);
			for (auto const value : {leading + half - 1, leading + half, leading + half + 1, leading + 3 * half}) {
				REQUIRE(Floating(value) == Floating(native(value)));
				REQUIRE(Floating(signed_integer(-value)) == Floating(native(signed_integer(-value))));
			}
		}

		REQUIRE(unsigned_integer(Floating(-0.5)) == 0);
		REQUIRE(signed_integer(Floating(-1.75)) == -1);
	}
#endif
}

TEST_CASE("wide arithmetic primitives", "[int128]") {
	std::mt19937_64 random(9102);
	for (int iteration = 0; iteration < 20000; ++iteration) {
		auto const left = random(), right = random(), carry = random() & 1;
		auto const sum = add_with_carry(left, right, carry);
		auto const difference = subtract_with_borrow(left, right, carry);
		// Independent of the native type: the word results wrap, and the carry / borrow flag the wrap.
		REQUIRE(sum.value == left + right + carry);
		REQUIRE(sum.carry == std::uint64_t(left + right < left || left + right + carry < left + right));
		REQUIRE(difference.value == left - right - carry);
		REQUIRE(difference.borrow == std::uint64_t(left < right || left - right < carry));
		auto const product = multiply_wide(left, right);
		REQUIRE(multiply_high(left, right) == product.high);
		auto const divisor = random() | 1;
		auto const bounded_high = random() % divisor;
		auto const bounded = divide_wide_bounded(left, bounded_high, divisor);
		REQUIRE(bounded.remainder < divisor);
		auto const division = divide_wide(left, right, divisor);
		REQUIRE(division.remainder < divisor);
		auto const reconstructed = add_words(multiply_words(division.quotient, {divisor, 0}), {division.remainder, 0});
		REQUIRE(reconstructed.low == left);
		REQUIRE(reconstructed.high == right);
#if defined(__SIZEOF_INT128__)
		REQUIRE(bounded.quotient == ((__uint128_t(bounded_high) << 64) | left) / divisor);
		REQUIRE(bounded.remainder == ((__uint128_t(bounded_high) << 64) | left) % divisor);
		auto const native_sum = __uint128_t(left) + right + carry;
		REQUIRE(sum.value == std::uint64_t(native_sum));
		REQUIRE(sum.carry == std::uint64_t(native_sum >> 64));
		auto const native_difference = __uint128_t(left) - right - carry;
		REQUIRE(difference.value == std::uint64_t(native_difference));
		REQUIRE(difference.borrow == (std::uint64_t(native_difference >> 64) & 1));
		REQUIRE(native(unsigned_integer(product.low, product.high)) == __uint128_t(left) * right);
		REQUIRE(native(unsigned_integer(division.quotient.low, division.quotient.high)) == native(unsigned_integer(left, right)) / divisor);
		REQUIRE(division.remainder == native(unsigned_integer(left, right)) % divisor);
#endif
		unsigned_integer const a{random(), random()}, b{random(), random()};
		auto const high = multiply_high_wide({a.low, a.high}, {b.low, b.high});
		// At most eight 16x16 products per column fit in 35 bits, on every target.
		std::array<std::uint64_t, 16> digits{};
		std::uint64_t column = 0;
		for (int index = 0; index < 16; ++index) {
			for (int first = 0; first < 8; ++first)
				if (index - first >= 0 && index - first < 8)
					column += std::uint64_t((a >> (16 * first)) & 65535) * std::uint64_t((b >> (16 * (index - first))) & 65535);
			digits[index] = column & 65535;
			column >>= 16;
		}
		REQUIRE(high.low == (digits[8] | (digits[9] << 16) | (digits[10] << 32) | (digits[11] << 48)));
		REQUIRE(high.high == (digits[12] | (digits[13] << 16) | (digits[14] << 32) | (digits[15] << 48)));
	}
	for (std::uint64_t divisor : {std::uint64_t(1), std::uint64_t(2), std::uint64_t(UINT32_MAX), std::uint64_t(1) << 32, std::uint64_t(1) << 63, UINT64_MAX}) {
		auto const result = divide_wide_bounded(UINT64_MAX, divisor - 1, divisor);
		REQUIRE(result.quotient == UINT64_MAX);
		REQUIRE(result.remainder == divisor - 1);
	}
	for (unsigned shift = 0; shift < 64; ++shift) {
		std::uint64_t const leading_bit = std::uint64_t(1) << (63 - shift);
		for (auto const divisor : {leading_bit, leading_bit | (leading_bit - 1)}) {
			for (auto const low : {std::uint64_t(0), std::uint64_t(1), UINT64_MAX}) {
				for (auto const high : {std::uint64_t(0), divisor / 2, divisor - 1}) {
					auto const result = divide_wide_bounded(low, high, divisor);
#if defined(__SIZEOF_INT128__)
					auto const dividend = (__uint128_t(high) << 64) | low;
					REQUIRE(result.quotient == dividend / divisor);
					REQUIRE(result.remainder == dividend % divisor);
#else
					auto const reconstructed = add_words(multiply_wide(result.quotient, divisor), {result.remainder, 0});
					REQUIRE(reconstructed.low == low);
					REQUIRE(reconstructed.high == high);
#endif
					REQUIRE(result.remainder < divisor);
				}
			}
		}
	}

	auto const high = multiply_high_wide({UINT64_MAX, UINT64_MAX}, {UINT64_MAX, UINT64_MAX});
	REQUIRE(high.low == UINT64_MAX - 1);
	REQUIRE(high.high == UINT64_MAX);
}

TEST_CASE("software integers arithmetic and mixed operands", "[int128]") {
	check_portable_builtin<bool>();
	check_portable_builtin<char>();
	check_portable_builtin<signed char>();
	check_portable_builtin<unsigned char>();
	check_portable_builtin<wchar_t>();
	check_portable_builtin<char8_t>();
	check_portable_builtin<char16_t>();
	check_portable_builtin<char32_t>();
	check_portable_builtin<short>();
	check_portable_builtin<unsigned short>();
	check_portable_builtin<int>();
	check_portable_builtin<unsigned int>();
	check_portable_builtin<long>();
	check_portable_builtin<unsigned long>();
	check_portable_builtin<long long>();
	check_portable_builtin<unsigned long long>();
#if defined(__SIZEOF_INT128__)
	check_builtin<bool>();
	check_builtin<char>();
	check_builtin<signed char>();
	check_builtin<unsigned char>();
	check_builtin<wchar_t>();
	check_builtin<char8_t>();
	check_builtin<char16_t>();
	check_builtin<char32_t>();
	check_builtin<short>();
	check_builtin<unsigned short>();
	check_builtin<int>();
	check_builtin<unsigned int>();
	check_builtin<long>();
	check_builtin<unsigned long>();
	check_builtin<long long>();
	check_builtin<unsigned long long>();
	check_builtin<__int128_t>();
	check_builtin<__uint128_t>();
	std::mt19937_64 random(7712);
	for (int iteration = 0; iteration < 4000; ++iteration) {
		unsigned_integer const a{random(), random()}, b{random(), random()};
		check_pair(a, b);
		check_pair(signed_integer(a), signed_integer(b));
		check_pair(a, signed_integer(b));
		check_pair(signed_integer(a), b);
	}
	// Divisors of every width: full-width random pairs almost always give quotients 0 or 1.
	for (int iteration = 0; iteration < 200000; ++iteration) {
		auto const dividend = (__uint128_t(random()) << 64 | random()) >> (random() % 128);
		auto divisor = (__uint128_t(random()) << 64 | random()) >> (random() % 128);
		if (divisor == 0) divisor = 1;
		unsigned_integer const a{std::uint64_t(dividend), std::uint64_t(dividend >> 64)};
		unsigned_integer const b{std::uint64_t(divisor), std::uint64_t(divisor >> 64)};
		matches(a / b, dividend / divisor);
		matches(a % b, dividend % divisor);
	}
	for (auto const a : edges)
		for (auto const b : edges) {
			check_pair(a, b);
			check_pair(signed_integer(a), signed_integer(b));
			check_pair(a, signed_integer(b));
			check_pair(signed_integer(a), b);
		}
#endif
	for (auto const bits : edges) {
		REQUIRE(unsigned_integer(-signed_integer(bits)) == -bits);
		REQUIRE(unsigned_integer(~signed_integer(bits)) == ~bits);
		REQUIRE(+bits == bits);
		REQUIRE(+signed_integer(bits) == signed_integer(bits));
		REQUIRE(!bits == !bool(bits));
		REQUIRE(!signed_integer(bits) == !bool(bits));
		REQUIRE((bits && true) == bool(bits));
		REQUIRE((false || bits) == bool(bits));
		auto changed = bits;
		REQUIRE(changed++ == bits);
		REQUIRE(changed == bits + 1);
		REQUIRE(--changed == bits);
		REQUIRE(changed-- == bits);
		REQUIRE(++changed == bits);
		auto signed_changed = signed_integer(bits);
		REQUIRE(signed_changed++ == signed_integer(bits));
		REQUIRE(--signed_changed == signed_integer(bits));
		REQUIRE(signed_changed-- == signed_integer(bits));
		REQUIRE(++signed_changed == signed_integer(bits));
		for (int shift : {0, 1, 31, 32, 63, 64, 65, 126, 127}) {
			REQUIRE((bits << unsigned_integer(shift)) == (bits << shift));
			REQUIRE((bits >> signed_integer(shift)) == (bits >> shift));
#if defined(__SIZEOF_INT128__)
			matches(bits << shift, native(bits) << shift);
			matches(bits >> shift, native(bits) >> shift);
			matches(signed_integer(bits) << shift, native(bits) << shift);
			matches(signed_integer(bits) >> shift, native(signed_integer(bits)) >> shift);
#endif
		}
	}
	REQUIRE((1 << unsigned_integer{3}) == 8);
	REQUIRE((-8 >> signed_integer{2}) == -2);
	REQUIRE((true << signed_integer{2}) == 4);
	int builtin_value = -42;
	REQUIRE(&(builtin_value += unsigned_integer{2}) == &builtin_value);
	REQUIRE(builtin_value == -40);
	REQUIRE(&(builtin_value >>= signed_integer{2}) == &builtin_value);
	REQUIRE(builtin_value == -10);
	builtin_value = 1;
	REQUIRE(&(builtin_value <<= unsigned_integer{3}) == &builtin_value);
	REQUIRE(builtin_value == 8);
	unsigned_integer unsigned_value{123};
	signed_integer signed_value{-123};
	REQUIRE(&(unsigned_value += -4) == &unsigned_value);
	REQUIRE(unsigned_value == 119);
	REQUIRE(&(signed_value += unsigned_integer{4}) == &signed_value);
	REQUIRE(signed_value == -119);
#define CHECK_COMPOUND(operation, operand)                             \
	{                                                                  \
		auto u = unsigned_value;                                       \
		auto s = signed_value;                                         \
		auto const expected_u = unsigned_integer(u operation operand); \
		auto const expected_s = signed_integer(s operation operand);   \
		REQUIRE(&(u operation## = operand) == &u);                     \
		REQUIRE(u == expected_u);                                      \
		REQUIRE(&(s operation## = operand) == &s);                     \
		REQUIRE(s == expected_s);                                      \
	}
	CHECK_COMPOUND(-, 4);
	CHECK_COMPOUND(*, 4);
	CHECK_COMPOUND(/, 4);
	CHECK_COMPOUND(%, 4);
	CHECK_COMPOUND(&, 4);
	CHECK_COMPOUND(|, 4);
	CHECK_COMPOUND(^, 4);
	CHECK_COMPOUND(<<, 4);
	CHECK_COMPOUND(>>, 4);
#undef CHECK_COMPOUND
	for (auto const dividend : edges)
		for (auto const divisor : edges)
			if (divisor) {
				auto const quotient = dividend / divisor, remainder = dividend % divisor;
				REQUIRE(quotient * divisor + remainder == dividend);
				REQUIRE(remainder < divisor);
			}
}

TEST_CASE("software division reconstruction without a native oracle", "[int128]") {
	std::mt19937_64 random(1108);
	for (int iteration = 0; iteration < 4000; ++iteration) {
		unsigned_integer const a = unsigned_integer{random(), random()} >> (random() % 128);
		unsigned_integer const b = (unsigned_integer{random(), random()} >> (random() % 128)) | 1;
		REQUIRE((a / b) * b + a % b == a);
		REQUIRE(a % b < b);
		signed_integer const sa(a), sb(b);
		if (sa == std::numeric_limits<signed_integer>::min() && sb == -1) continue;
		signed_integer const q = sa / sb, r = sa % sb;
		REQUIRE(q * sb + r == sa);
		REQUIRE((r == 0 || (r < 0) == (sa < 0)));
		auto const magnitude = [](signed_integer value) {
			auto const bits = unsigned_integer(value);
			return value < 0 ? -bits : bits;
		};
		REQUIRE(magnitude(r) < magnitude(sb));
	}
	unsigned_integer const dividend = unsigned_integer{1} << 127;
	unsigned_integer const divisor = (unsigned_integer{1} << 64) + 1;
	REQUIRE(dividend / divisor == (unsigned_integer{1} << 63) - 1);
	REQUIRE(dividend % divisor == (unsigned_integer{1} << 63) + 1);
}

TEST_CASE("software integers floating conversion", "[int128]") {
#if defined(__SIZEOF_INT128__)
	check_floating<float>();
	check_floating<double>();
	check_floating<long double>();
#endif
	REQUIRE(unsigned_integer(std::numeric_limits<float>::max()) == unsigned_integer{0, UINT64_MAX - ((std::uint64_t(1) << 40) - 1)});
	REQUIRE(signed_integer(-0.5) == 0);
	REQUIRE(signed_integer(-0x1p127) == std::numeric_limits<signed_integer>::min());
	REQUIRE(unsigned_integer(std::nextafter(0x1p128, 0.0)) == unsigned_integer(0, UINT64_MAX - 2047));
	REQUIRE(float((unsigned_integer{1} << 24) + 1) == 0x1p24f);
	REQUIRE(float((unsigned_integer{1} << 24) + 3) == 0x1.000004p24f);
	REQUIRE(double((unsigned_integer{1} << 53) + 1) == 0x1p53);
	REQUIRE(double((unsigned_integer{1} << 53) + 3) == 0x1.0000000000002p53);
	REQUIRE(float(unsigned_integer{UINT64_MAX, UINT64_MAX}) == std::numeric_limits<float>::infinity());
}

TEST_CASE("software integers double-double components", "[int128]") {
	constexpr auto sparse = words_double_double<double>({1, std::uint64_t(1) << 63});
	static_assert(sparse.high == 0x1p127 && sparse.low == 1);
	constexpr auto maximum = words_double_double<double>({UINT64_MAX, UINT64_MAX});
	static_assert(maximum.high == 0x1p128 && maximum.low == -1);

	auto const reconstruct = [](unsigned_integer value) {
		auto const components = words_double_double<double>({value.low, value.high});
		REQUIRE(components.high == double(value));
		// Add in exact integer arithmetic: a double sum would discard the low component again.
		auto result = components.high == 0x1p128 ? unsigned_integer{} : unsigned_integer(components.high);
		if (components.low < 0)
			result -= unsigned_integer(-components.low);
		else
			result += unsigned_integer(components.low);
		REQUIRE(result == value);
	};
	reconstruct(unsigned_integer{});
	reconstruct(unsigned_integer{UINT64_MAX, UINT64_MAX});
	for (int bit = 1; bit < 128; ++bit) {
		auto const power = unsigned_integer{1} << bit;
		for (auto const value : {power - 1, power, power + 1}) reconstruct(value);
	}
	std::mt19937_64 random(6021);
	for (int iteration = 0; iteration < 2000; ++iteration) {
		unsigned_integer const significand{random(), random() & ((std::uint64_t(1) << 42) - 1)};
		reconstruct(significand << (iteration % 23));
		auto const power = unsigned_integer{1} << 127;
		auto const residual = unsigned_integer(random() & ((std::uint64_t(1) << 53) - 1));
		reconstruct(power + residual);
		reconstruct(power - residual);
	}
	for (auto const offset : {1, 3}) {
		auto const value = (unsigned_integer{1} << 127) + offset * (unsigned_integer{1} << 74);
		auto const components = words_double_double<double>(operand_words(value));
		REQUIRE(components.high == (offset == 1 ? 0x1p127 : 0x1.0000000000002p127));
		REQUIRE(components.low == (offset == 1 ? 0x1p74 : -0x1p74));
		reconstruct(value);
	}
	for (auto const offset : {1, 3}) {
		auto const residual = (unsigned_integer{1} << 53) + offset;
		auto const positive = words_double_double<double>(operand_words((unsigned_integer{1} << 127) + residual));
		auto const negative = words_double_double<double>(operand_words((unsigned_integer{1} << 127) - residual));
		REQUIRE(positive.high == 0x1p127);
		REQUIRE(positive.low == (offset == 1 ? 0x1p53 : 0x1.0000000000002p53));
		REQUIRE(negative.high == 0x1p127);
		REQUIRE(negative.low == -positive.low);
	}
	if constexpr (std::numeric_limits<long double>::digits == 106) {
		REQUIRE(static_cast<long double>(unsigned_integer{1, std::uint64_t(1) << 63}) - 0x1p127L == 1);
		REQUIRE(static_cast<long double>(signed_integer{1, std::uint64_t(1) << 63}) + 0x1p127L == 1);
	}
}

#if defined(__cpp_lib_format)
TEST_CASE("software integers dynamic width and Unicode fill", "[int128]") {
	REQUIRE(std::format("{:{}}", unsigned_integer{42}, 0) == "42");
	REQUIRE(std::format("{:{}}", signed_integer{-42}, 0u) == "-42");
	REQUIRE(std::format("{0:>{1}}", unsigned_integer{42}, 0ull) == "42");
	REQUIRE(std::format(L"{:{}}", unsigned_integer{42}, 0) == L"42");
	unsigned_integer value{42};
	auto const reject_width = [&]<typename Width>(Width width) {
		REQUIRE_THROWS_AS(std::vformat("{:{}}", std::make_format_args(value, width)), std::format_error);
	};
	reject_width(-1);
	reject_width(-1ll);
	reject_width(static_cast<unsigned long long>(std::numeric_limits<int>::max()) + 1);
	reject_width(0.0);
	reject_width(false);
	reject_width('0');
	reject_width(std::string("0"));

	REQUIRE(std::format("{:é>3}", unsigned_integer{1}) == "éé1");
	REQUIRE(std::format("{:\u20ac<4}", signed_integer{-1}) == "-1\u20ac\u20ac");
	REQUIRE(std::format("{:\U0001f642^4}", unsigned_integer{1}) == "\U0001f6421\U0001f642\U0001f642");
	REQUIRE(std::format("{:\U0010ffff>3}", unsigned_integer{1}) == "\U0010ffff\U0010ffff1");
	REQUIRE(std::format("{:é>{}}", unsigned_integer{1}, 0) == "1");
	REQUIRE(std::format("{:é>+#7x}", signed_integer{42}) == "éé+0x2a");
	REQUIRE(std::format(L"{:é>3}", unsigned_integer{1}) == L"éé1");
	REQUIRE(std::format(L"{:\U0001f642<3}", unsigned_integer{1}) == L"1\U0001f642\U0001f642");

	for (std::string const specification : {"{:{>3}", "{:}>3}", "{:ab>3}", "{:\x80>3}", "{:\xc0\xaf>3}", "{:\xe0\x80\x80>3}", "{:\xed\xa0\x80>3}", "{:\xf0\x80\x80\x80>3}", "{:\xf4\x90\x80\x80>3}", "{:\xf5\x80\x80\x80>3}", "{:\xc2>3}", "{:\xf0\x9f"}) {
		REQUIRE_THROWS_AS(std::vformat(specification, std::make_format_args(value)), std::format_error);
	}
	for (std::wstring const specification : {L"{:{>3}", L"{:}>3}", L"{:\xd800>3}", L"{:\xdc00>3}", L"{:\x178}"}) {
		REQUIRE_THROWS_AS(std::vformat(specification, std::make_wformat_args(value)), std::format_error);
	}
	if constexpr (sizeof(wchar_t) > 2) {
		std::uint32_t invalid_scalar = 0x110000;
		std::wstring specification = L"{:";
		specification += static_cast<wchar_t>(invalid_scalar);
		specification += L">3}";
		REQUIRE_THROWS_AS(std::vformat(specification, std::make_wformat_args(value)), std::format_error);
	}
}
#endif

TEST_CASE("software integers limits hash and format", "[int128]") {
	using SignedLimits = std::numeric_limits<signed_integer>;
	using UnsignedLimits = std::numeric_limits<unsigned_integer>;
	REQUIRE(SignedLimits::min() == signed_integer(0, std::uint64_t(1) << 63));
	REQUIRE(SignedLimits::lowest() == SignedLimits::min());
	REQUIRE(SignedLimits::max() == signed_integer(UINT64_MAX, UINT64_MAX >> 1));
	REQUIRE(UnsignedLimits::min() == 0);
	REQUIRE(UnsignedLimits::max() == unsigned_integer(UINT64_MAX, UINT64_MAX));
	REQUIRE(SignedLimits::digits == 127);
	REQUIRE(UnsignedLimits::digits == 128);
	REQUIRE(SignedLimits::digits10 == 38);
	REQUIRE(UnsignedLimits::digits10 == 38);
	REQUIRE(SignedLimits::is_modulo);
	REQUIRE(UnsignedLimits::is_modulo);
	REQUIRE(std::hash<unsigned_integer>{}(unsigned_integer{1}) != std::hash<unsigned_integer>{}(unsigned_integer{0, 1}));
	REQUIRE(std::hash<signed_integer>{}(signed_integer{-1}) == std::hash<unsigned_integer>{}(unsigned_integer{-1}));
#if defined(__cpp_lib_format)
	REQUIRE(std::format("{}", SignedLimits::min()) == "-170141183460469231731687303715884105728");
	REQUIRE(std::format("{}", UnsignedLimits::max()) == "340282366920938463463374607431768211455");
	REQUIRE(std::format("{:+#012x}", signed_integer{-42}) == "-0x00000002a");
	REQUIRE(std::format("{:#B}", unsigned_integer{5}) == "0B101");
	REQUIRE(std::format("{:#o}", unsigned_integer{0}) == "0");
	REQUIRE(std::format("{:#o}", unsigned_integer{9}) == "011");
	REQUIRE(std::format("{:*^9}", signed_integer{-42}) == "***-42***");
	REQUIRE(std::format("{: <+8X}", signed_integer{42}) == "+2A     ");
	REQUIRE(std::format("{:0>8}", signed_integer{-42}) == "00000-42");
	REQUIRE(std::format("{:>{}}", unsigned_integer{42}, 5) == "   42");
	REQUIRE(std::format("{0:>{1}}", unsigned_integer{42}, 5) == "   42");
	REQUIRE(std::format(L"{:#x}", unsigned_integer{42}) == L"0x2a");
	unsigned_integer format_value{42};
	REQUIRE_THROWS_AS(std::vformat("{:f}", std::make_format_args(format_value)), std::format_error);
#endif
}

TEST_CASE("software integer preconditions", "[int128]") {
	REQUIRE(atl_test::aborts([] { (void)add_with_carry(0, 0, 2); }));
	REQUIRE(atl_test::aborts([] { (void)subtract_with_borrow(0, 0, 2); }));
	REQUIRE(atl_test::aborts([] { (void)divide_wide_bounded(0, 0, 0); }));
	REQUIRE(atl_test::aborts([] { (void)divide_wide_bounded(0, 5, 5); }));
	REQUIRE(atl_test::aborts([] { (void)divide_wide(0, 0, 0); }));
	REQUIRE(atl_test::aborts([] { (void)divide_normalized_digit(1, 0, 1); }));
	REQUIRE(atl_test::aborts([] { (void)divide_normalized_digit(0, 0, 1); }));
	REQUIRE(atl_test::aborts([] { (void)divide_normalized_digit(0, std::uint64_t(1) << 32, std::uint64_t(1) << 63); }));
	REQUIRE(atl_test::aborts([] { (void)shift_left_words({0, 0}, 128); }));
	REQUIRE(atl_test::aborts([] { (void)shift_right_words({0, 0}, 128); }));
	REQUIRE(atl_test::aborts([] { (void)(unsigned_integer{1} / 0); }));
	REQUIRE(atl_test::aborts([] { (void)(unsigned_integer{1} % 0); }));
	REQUIRE(atl_test::aborts([] { (void)(signed_integer{1} / 0); }));
	REQUIRE(atl_test::aborts([] { (void)(signed_integer{1} % 0); }));
	REQUIRE(atl_test::aborts([] { (void)(std::numeric_limits<signed_integer>::min() / -1); }));
	REQUIRE(atl_test::aborts([] { (void)(std::numeric_limits<signed_integer>::min() % -1); }));
	REQUIRE(atl_test::aborts([] { (void)(unsigned_integer{1} << -1); }));
	REQUIRE(atl_test::aborts([] { (void)(signed_integer{1} >> 128); }));
	REQUIRE(atl_test::aborts([] { (void)(1 << unsigned_integer{0, 1}); }));
	REQUIRE(atl_test::aborts([] { (void)(1 << unsigned_integer{32}); }));
	REQUIRE(atl_test::aborts([] { (void)(1 >> signed_integer{-1}); }));
	REQUIRE(atl_test::aborts([] { (void)unsigned_integer(std::numeric_limits<double>::quiet_NaN()); }));
	REQUIRE(atl_test::aborts([] { (void)signed_integer(std::numeric_limits<double>::infinity()); }));
	REQUIRE(atl_test::aborts([] { (void)unsigned_integer(-1.0); }));
	REQUIRE(atl_test::aborts([] { (void)unsigned_integer(0x1p128); }));
	REQUIRE(atl_test::aborts([] { (void)signed_integer(0x1p127); }));
	REQUIRE(atl_test::aborts([] { (void)signed_integer(0x1p127); }));
	REQUIRE(atl_test::aborts([] { (void)signed_integer(-0x1.0000000000001p127); }));
}
