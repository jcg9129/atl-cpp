#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
#include <atl/bits/xor_basis.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <string>

using atl::uint32;
using atl::uint64;

static_assert(atl::lowest_one(atl::uint8(0)) == 0 && atl::lowest_one(atl::uint8(0xff)) == 1
			  && atl::lowest_one(atl::uint8(0x60)) == 0x20);
static_assert(atl::lowest_one(uint64(0)) == 0 && atl::lowest_one(~uint64(0)) == 1
			  && atl::lowest_one(uint64(1) << 63) == uint64(1) << 63);
static_assert(atl::lowest_one(atl::uint128(0)) == 0 && atl::lowest_one(~atl::uint128(0)) == 1
			  && atl::lowest_one(atl::uint128(1) << 127) == atl::uint128(1) << 127);

static_assert(atl::next_bits_combination(atl::uint32(0b0110)) == 0b1001);
static_assert(atl::next_bits_combination(atl::uint8(0b11000000)) == 0);
static_assert(atl::next_bits_combination(atl::uint32(0)) == 0);
static_assert(atl::next_bits_combination(atl::uint128(0b0111) << 100)
			  == ((atl::uint128(1) << 103) | 0b11));

static_assert([] {
	atl::xor_basis<atl::uint32> basis(std::array<atl::uint32, 3>{6, 5, 3}); // 3 = 6 ^ 5
	atl::xor_basis<atl::uint32> other;
	other.insert(8);
	basis.merge(other);
	return basis.rank() == 3 && basis.max_xor() == 14 && basis.min_nonzero() == 3 && basis.kth_smallest(5) == 11
		&& basis.count_below(14) == 7 && basis.contains(11) && !basis.contains(16) && basis.min_xor(17) == 17
		&& basis.representable_count() == 8;
}());
static_assert([] {
	atl::stamped_xor_basis<atl::uint64> basis;
	std::array<atl::uint64, 4> const a{3, 1, 2, 1};
	for (atl::uint32 i = 0; i < 4; ++i) basis.insert(a[i], i);
	return basis.rank(0) == 2 && basis.rank(2) == 2 && basis.rank(3) == 1 && basis.max_xor(3) == 1
		&& basis.contains(3, 1) && !basis.contains(2, 3) && basis.basis_from(1).max_xor() == 3;
}());

TEST_CASE("bits APIs are constant-evaluable", "[bits][constexpr]") {
	STATIC_REQUIRE(atl::next_bits_combination(uint32(6)) == 9);
}

namespace {
	template<typename Word>
	concept has_basis = requires { typename atl::xor_basis<Word>; };
	template<typename Word, typename Stamp>
	concept has_stamped_basis = requires { typename atl::stamped_xor_basis<Word, Stamp>; };
	static_assert(has_basis<atl::uint128> && !has_basis<int> && !has_basis<bool>);
	static_assert(has_stamped_basis<unsigned, std::string> && !has_stamped_basis<int, unsigned>);
	static_assert(!has_stamped_basis<unsigned, void>);
	static_assert(atl::word_range<std::array<unsigned, 2> &, atl::uint128>);
	static_assert(!atl::word_range<std::array<std::string, 2> &, unsigned>);

	template<atl::unsigned_integral Word>
	constexpr bool wide_word_contracts() {
		constexpr auto width = std::size_t(std::numeric_limits<Word>::digits);
		constexpr Word top = Word(Word(1) << (width - 1));
		constexpr Word maximum = (std::numeric_limits<Word>::max)();
		for (std::size_t bit = 0; bit < width; ++bit) {
			Word const single = Word(Word(1) << bit);
			if (atl::lowest_one(single) != single) return false;
			if (atl::next_bits_combination(single) != (bit + 1 == width ? Word(0) : Word(single << 1))) return false;
		}
		if (atl::next_bits_combination(maximum) != 0 || atl::next_bits_combination(Word(0)) != 0) return false;
		atl::xor_basis<Word> full;
		for (std::size_t bit = 0; bit < width; ++bit)
			if (!full.insert(Word(Word(1) << bit))) return false;
		if (full.rank() != width || full.pivot_mask() != maximum || full.vector(width - 1) != top
			|| full.kth_smallest(maximum) != maximum || full.count_below(maximum) != maximum
			|| full.max_xor() != maximum || full.reduce(maximum) != 0) return false;
		full.merge(full);
		if (full.rank() != width) return false;
		full.clear();
		if (!full.empty() || full.representable_count() != 1) return false;
		atl::stamped_xor_basis<Word, int> stamped;
		if (!stamped.insert(top, 7) || !stamped.insert(Word(top | 1), 9)) return false;
		if (stamped.rank(9) != 1 || stamped.max_xor(9) != Word(top | 1) || stamped.stamp(width - 1) != 9) return false;
		if (stamped.insert(1, 11) || stamped.rank() != 2 || stamped.rank(10) != 1) return false;
		auto const suffix = stamped.basis_from(10);
		if (suffix.max_xor() != 1 || stamped.contains(top, 10) || !stamped.contains(1, 10)) return false;
		stamped.clear();
		return stamped.rank() == 0 && stamped.rank(0) == 0 && stamped.max_xor(0) == 0;
	}

	static_assert(wide_word_contracts<atl::uint8>());
	static_assert(wide_word_contracts<atl::uint16>());
	static_assert(wide_word_contracts<atl::uint32>());
	static_assert(wide_word_contracts<atl::uint64>());
	static_assert(wide_word_contracts<atl::uint128>());
	static_assert(wide_word_contracts<atl::detail::software_uint128>());
}
