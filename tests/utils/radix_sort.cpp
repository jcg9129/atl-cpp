#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
	template<typename Key>
	concept has_record = requires { typename atl::detail::keyed_position<Key>; };
	template<int Bits, typename Key>
	concept can_sort = requires(std::vector<atl::detail::keyed_position<Key>> &v) { atl::detail::radix_sort<Bits>(v); };
	static_assert(has_record<unsigned> && can_sort<7, unsigned>);
	static_assert(!has_record<bool> && !has_record<char> && !has_record<char16_t> && !has_record<int>);
	static_assert(!has_record<unsigned const> && !has_record<unsigned volatile>);
	static_assert(!can_sort<7, bool> && !can_sort<7, char>);
	static_assert(!can_sort<7, unsigned const> && !can_sort<7, unsigned volatile>);
	static_assert(!can_sort<0, unsigned> && !can_sort<17, unsigned>);
	static_assert(can_sort<16, unsigned char>);

	template<atl::unsigned_integral Key>
		requires std::same_as<Key, std::remove_cv_t<Key>>
	void check_sort() {
		using record = atl::detail::keyed_position<Key>;
		constexpr Key top = Key(1) << (std::numeric_limits<Key>::digits - 1);
		std::vector<record> values{{top, 0}, {1, 1}, {0, 2}, {top, 3}, {1, 4}, {std::numeric_limits<Key>::max(), 5}};
#if ATL_HAS_INT128
		if constexpr (sizeof(Key) > 8) {
			values.push_back({(Key(1) << 65) | 1, 6});
			values.push_back({(Key(1) << 64) | 1, 7});
		}
#endif
		auto expected = values;
		std::ranges::stable_sort(expected, {}, &record::key);
		atl::detail::radix_sort<7>(values);
		REQUIRE(values.size() == expected.size());
		for (std::size_t i = 0; i < values.size(); ++i) {
			REQUIRE(values[i].key == expected[i].key);
			REQUIRE(values[i].position == expected[i].position);
		}
		values.clear();
		atl::detail::radix_sort<7>(values);
		REQUIRE(values.empty());
		values.push_back({top, 42});
		atl::detail::radix_sort<7>(values);
		REQUIRE(values[0].key == top);
		REQUIRE(values[0].position == 42);
	}
}

TEST_CASE("radix sort: full-width keys and stable positions", "[utils]") {
	check_sort<atl::uint8>();
	check_sort<atl::uint64>();
#if ATL_HAS_INT128
	static_assert(has_record<atl::uint128> && can_sort<7, atl::uint128>);
	check_sort<atl::uint128>();
#endif
	std::vector<atl::detail::keyed_position<unsigned char>> bytes{{255, 0}, {0, 1}};
	atl::detail::radix_sort<16>(bytes);
	REQUIRE(bytes[0].key == 0);
}
