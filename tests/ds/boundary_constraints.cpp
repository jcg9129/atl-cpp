#include <atl/ds/coordinate_compression.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/ds/flat_hash_set.hpp>
#include <atl/ds/hierarchical_bitset.hpp>
#include <atl/ds/persistent_array.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/experimental/ds/coordinate_compression_variants.hpp>
#include <atl/experimental/ds/weighted_dsu_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/int128.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace approved_hash_test {
	struct throwing_mix {
		struct result {
			operator std::size_t() const noexcept { return 0; }

			friend std::size_t operator&(result, std::size_t) { return 0; }
		};

		result operator()(int) const noexcept { return {}; }
	};

	struct throwing_return {
		struct index {
			explicit operator std::size_t() const noexcept { return 0; }

			operator bool() const { return false; }
		};

		struct result {
			operator std::size_t() const noexcept { return 0; }

			friend index operator&(result, std::size_t) noexcept { return {}; }
		};

		result operator()(int) const noexcept { return {}; }
	};
}

namespace atl::detail {
	template<>
	inline constexpr bool hash_low_bits_usable<approved_hash_test::throwing_mix> = true;
	template<>
	inline constexpr bool hash_low_bits_usable<approved_hash_test::throwing_return> = true;
}

namespace {
	template<typename Map, typename Key, typename Fallback>
	concept can_at_or = requires(Map const &map, Key const &key, Fallback &&fallback) {
		map.at_or(key, std::forward<Fallback>(fallback));
	};

	static_assert(can_at_or<atl::flat_hash_map<int, int>, int, short>);
	static_assert(can_at_or<atl::flat_hash_map<std::string, int>, std::string_view, int>);
	static_assert(!can_at_or<atl::flat_hash_map<int, std::unique_ptr<int>>, int, std::unique_ptr<int>>);
	static_assert(!can_at_or<atl::flat_hash_map<std::string, std::unique_ptr<int>>, std::string_view, std::unique_ptr<int>>);
	static_assert(!can_at_or<atl::flat_hash_map<int, int>, int, std::string>);

	struct explicit_value {
		int value = 0;
		constexpr explicit_value() = default;

		constexpr explicit explicit_value(int x) : value(x) {}

		constexpr explicit explicit_value(explicit_value const &) = default;
		constexpr explicit_value(explicit_value &&) = default;
		constexpr explicit_value &operator=(explicit_value const &) = default;
		constexpr explicit_value &operator=(explicit_value &&) = default;
		constexpr auto operator<=>(explicit_value const &) const = default;
	};

	static_assert(!can_at_or<atl::flat_hash_map<int, explicit_value>, int, explicit_value>);

	struct move_compare {
		constexpr move_compare() = default;
		constexpr explicit move_compare(move_compare &&) = default;
		move_compare(move_compare const &) = delete;

		constexpr bool operator()(int a, int b) const { return a < b; }
	};

	struct immovable_compare {
		immovable_compare() = default;
		immovable_compare(immovable_compare &&) = delete;

		bool operator()(int a, int b) const { return a < b; }
	};

	template<typename Range, typename Compare = std::ranges::less>
	concept can_compress = requires(Range &&range, Compare &&comp) {
		atl::compress(std::forward<Range>(range), std::identity{}, Compare(std::forward<Compare>(comp)));
	};

	template<typename Range, typename Compare = std::ranges::less>
	concept can_deduce_compression = requires(Range &&range, Compare &&comp) {
		atl::coordinate_compression(std::forward<Range>(range), std::identity{}, Compare(std::forward<Compare>(comp)));
	};

	template<typename Range>
	concept can_pair_compress = requires(Range &&range) { atl::experimental::compress_by_pair_sort(std::forward<Range>(range)); };

	template<typename Range>
	concept can_index_compress = requires(Range &&range) { atl::experimental::compress_by_index_sort(std::forward<Range>(range)); };

	template<typename Range>
	concept can_radix_compress = requires(Range &&range) { atl::experimental::compress_by_radix_sort(std::forward<Range>(range)); };
	static_assert(can_radix_compress<std::vector<int>>);
	static_assert(!can_radix_compress<std::vector<std::unique_ptr<int>>>);

	static_assert(std::movable<explicit_value> && !std::convertible_to<explicit_value const &, explicit_value>);
	static_assert(can_compress<std::vector<explicit_value> &> && can_deduce_compression<std::vector<explicit_value> &>);
	static_assert(can_pair_compress<std::vector<explicit_value> &> && can_index_compress<std::vector<explicit_value> &>);
	static_assert(can_compress<std::vector<std::unique_ptr<int>>> && can_deduce_compression<std::vector<std::unique_ptr<int>>>);
	static_assert(!can_compress<std::vector<std::unique_ptr<int>> &> && !can_deduce_compression<std::vector<std::unique_ptr<int>> &>);
	static_assert(!can_pair_compress<std::vector<std::unique_ptr<int>> &> && !can_index_compress<std::vector<std::unique_ptr<int>> &>);
	static_assert(can_compress<std::vector<int>, move_compare> && can_deduce_compression<std::vector<int>, move_compare>);
	static_assert(!can_compress<std::vector<int>, immovable_compare> && !can_deduce_compression<std::vector<int>, immovable_compare>);
	static_assert(!std::constructible_from<atl::coordinate_compression<int, immovable_compare>, atl::sorted_unique_t, std::vector<int>, immovable_compare>);

	struct rvalue_position {
		operator std::size_t() && { return 1; }

		operator std::size_t() & = delete;
	};

	struct explicit_lvalue_position {
		int value;

		operator std::size_t() && { return std::size_t(value); }

		explicit operator std::size_t() & { return std::size_t(value); }
	};

	static_assert(std::convertible_to<rvalue_position, std::size_t>);
	static_assert(!std::constructible_from<atl::hierarchical_bitset, std::size_t, std::vector<rvalue_position>>);
	static_assert(std::constructible_from<atl::hierarchical_bitset, std::size_t, std::vector<explicit_lvalue_position>>);

	// Independent value/reference types expose the difference between the range's shape and its loop binding.
	template<typename Value, typename Reference, typename Category = std::forward_iterator_tag>
		requires std::is_object_v<Value> && std::is_object_v<std::remove_reference_t<Reference>>
			  && std::convertible_to<std::remove_reference_t<Reference> &, Reference>
	struct reference_range {
		struct iterator {
			using value_type = Value;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = Category;
			std::remove_reference_t<Reference> *pointer = nullptr;

			Reference operator*() const { return *pointer; }

			iterator &operator++() {
				++pointer;
				return *this;
			}

			iterator operator++(int) {
				auto snapshot = *this;
				++*this;
				return snapshot;
			}

			bool operator==(iterator const &) const = default;
		};

		std::remove_reference_t<Reference> *data;
		std::size_t count;

		iterator begin() const { return {data}; }

		iterator end() const { return {data + count}; }

		std::size_t size() const { return count; }
	};

	struct position_value {
		std::size_t value;

		operator std::size_t() const { return value; }
	};

	struct lvalue_position_proxy : position_value {
		explicit operator std::size_t() & { return value; }

		explicit operator std::size_t() && = delete;
	};

	struct rvalue_position_proxy : position_value {
		explicit operator std::size_t() & = delete;

		explicit operator std::size_t() && { return value; }
	};

	using lvalue_positions = reference_range<position_value, lvalue_position_proxy>;
	using rvalue_positions = reference_range<position_value, rvalue_position_proxy>;
	static_assert(std::ranges::input_range<lvalue_positions> && std::ranges::input_range<rvalue_positions>);
	static_assert(!std::constructible_from<std::size_t, std::ranges::range_reference_t<lvalue_positions>>);
	static_assert(std::constructible_from<atl::hierarchical_bitset, std::size_t, lvalue_positions>);
	static_assert(std::constructible_from<std::size_t, std::ranges::range_reference_t<rvalue_positions>>);
	static_assert(!std::constructible_from<atl::hierarchical_bitset, std::size_t, rvalue_positions>);

	using output_range = reference_range<int, int &, std::output_iterator_tag>;
	static_assert(std::ranges::sized_range<output_range> && !std::ranges::input_range<output_range>);
	static_assert(!std::constructible_from<atl::persistent_array<int>, output_range>);
	static_assert(!std::constructible_from<atl::semi_persistent_array<int>, output_range>);
	static_assert(std::constructible_from<atl::persistent_array<int>, std::vector<int>>);
	static_assert(std::constructible_from<atl::semi_persistent_array<int>, std::vector<int>>);

	template<typename Range>
	concept can_deduce_array = requires(Range &&range) { atl::persistent_array(std::forward<Range>(range)); };

	template<typename Range>
	concept can_deduce_semi_array = requires(Range &&range) { atl::semi_persistent_array(std::forward<Range>(range)); };

	static_assert(can_deduce_array<std::vector<int>> && can_deduce_semi_array<std::vector<int>>);
	static_assert(!can_deduce_array<output_range> && !can_deduce_semi_array<output_range>);

	struct explicit_op {
		constexpr explicit explicit_op() = default;
		constexpr explicit explicit_op(explicit_op &&) = default;
		explicit_op(explicit_op const &) = delete;

		constexpr int operator()(int a, int b) const { return a + b; }
	};

	struct explicit_inverse {
		constexpr explicit explicit_inverse() = default;
		constexpr explicit explicit_inverse(explicit_inverse &&) = default;
		explicit_inverse(explicit_inverse const &) = delete;

		constexpr int operator()(int a) const { return -a; }
	};

	struct no_default_potential {
		int value;

		explicit no_default_potential(int x) : value(x) {}

		bool operator==(no_default_potential const &) const = default;

		no_default_potential operator+(no_default_potential const &other) const { return no_default_potential(value + other.value); }

		no_default_potential operator-() const { return no_default_potential(-value); }
	};

	static_assert(!std::constructible_from<atl::weighted_dsu<no_default_potential>, std::size_t>);
	static_assert(std::constructible_from<atl::weighted_dsu<no_default_potential>, std::size_t, no_default_potential>);
	static_assert(!std::constructible_from<atl::rollback_weighted_dsu<no_default_potential>, std::size_t>);
	static_assert(std::constructible_from<atl::rollback_weighted_dsu<no_default_potential>, std::size_t, no_default_potential>);

	struct no_default_op {
		explicit no_default_op(int) {}

		int operator()(int a, int b) const { return a + b; }
	};

	struct no_default_inverse {
		explicit no_default_inverse(int) {}

		int operator()(int a) const { return -a; }
	};

	struct immovable_op {
		immovable_op() = default;
		immovable_op(immovable_op &&) = delete;

		int operator()(int a, int b) const { return a + b; }
	};

	struct immovable_inverse {
		immovable_inverse() = default;
		immovable_inverse(immovable_inverse &&) = delete;

		int operator()(int a) const { return -a; }
	};

	template<typename Op, typename Inverse>
		requires atl::binary_operation<Op, int> && atl::inverse_operation<Inverse, int>
	using weighted_variant = atl::experimental::weighted_dsu_by_compression_and_layout<atl::experimental::weighted_dsu_compression::recursive,
																					   atl::experimental::weighted_dsu_layout::split, int, Op, Inverse>;

	template<template<typename, typename> typename Dsu>
		requires requires {
			typename Dsu<explicit_op, explicit_inverse>;
			typename Dsu<immovable_op, explicit_inverse>;
			typename Dsu<explicit_op, immovable_inverse>;
			typename Dsu<no_default_op, no_default_inverse>;
			typename Dsu<no_default_op, explicit_inverse>;
			typename Dsu<explicit_op, no_default_inverse>;
		}
	constexpr bool operation_constraints() {
		static_assert(std::constructible_from<Dsu<explicit_op, explicit_inverse>, std::size_t>);
		static_assert(std::constructible_from<Dsu<explicit_op, explicit_inverse>, std::size_t, int>);
		static_assert(requires { Dsu<explicit_op, explicit_inverse>(3, 0, explicit_op{}); });
		static_assert(requires { Dsu<explicit_op, explicit_inverse>(3, 0, explicit_op{}, explicit_inverse{}); });
		static_assert(!requires { Dsu<immovable_op, explicit_inverse>(3, 0, immovable_op{}, explicit_inverse{}); });
		static_assert(!requires { Dsu<explicit_op, immovable_inverse>(3, 0, explicit_op{}, immovable_inverse{}); });
		static_assert(!std::constructible_from<Dsu<no_default_op, no_default_inverse>, std::size_t>);
		static_assert(!std::constructible_from<Dsu<no_default_op, no_default_inverse>, std::size_t, int>);
		static_assert(!std::constructible_from<Dsu<explicit_op, no_default_inverse>, std::size_t, int, explicit_op>);
		static_assert(std::constructible_from<Dsu<no_default_op, explicit_inverse>, std::size_t, int, no_default_op>);
		static_assert(std::constructible_from<Dsu<no_default_op, no_default_inverse>, std::size_t, int, no_default_op, no_default_inverse>);
		return true;
	}

	template<typename Op, typename Inverse>
		requires atl::binary_operation<Op, int> && atl::inverse_operation<Inverse, int>
	using weighted = atl::weighted_dsu<int, Op, Inverse>;

	template<typename Op, typename Inverse>
		requires atl::binary_operation<Op, int> && atl::inverse_operation<Inverse, int>
	using rollback_weighted = atl::rollback_weighted_dsu<int, Op, Inverse>;

	template<typename Op, typename Inverse>
		requires atl::binary_operation<Op, int> && atl::inverse_operation<Inverse, int>
	using rollback_variant = atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<int, Op, Inverse>;

	static_assert(operation_constraints<weighted>());
	static_assert(operation_constraints<rollback_weighted>());
	static_assert(operation_constraints<weighted_variant>());
	static_assert(operation_constraints<rollback_variant>());

	template<bool CanChain>
	struct proxy_backing {
		std::vector<int> values;
		proxy_backing() = default;

		proxy_backing(std::size_t n, int value) : values(n, value) {}

		int operator[](std::size_t i) const { return values[i]; }

		std::size_t size() const { return values.size(); }

		struct update {
			std::vector<int> values;

			operator proxy_backing() && {
				proxy_backing result;
				result.values = std::move(values);
				return result;
			}

			update set(std::size_t i, int &&value) &&
				requires CanChain
			{
				values[i] = value;
				return std::move(*this);
			}
		};

		update set(std::size_t i, int &&value) && {
			values[i] = value;
			return {std::move(values)};
		}
	};

	static_assert(atl::detail::persistent_signed_array<proxy_backing<true>>);
	static_assert(!atl::detail::persistent_signed_array<proxy_backing<false>>);
	static_assert(std::convertible_to<decltype(std::declval<proxy_backing<false>>().set(0, 0)), proxy_backing<false>>);

	struct bad_lifetime : atl::item_lifetime {
		std::string begin;
	};

	using bad_lifetimes = reference_range<atl::item_lifetime, bad_lifetime &>;
	static_assert(std::ranges::forward_range<bad_lifetimes const>);
	static_assert(!atl::lifetime_range<bad_lifetimes>);
	static_assert(atl::lifetime_range<reference_range<atl::item_lifetime, atl::item_lifetime &>>);

	struct contextual_bool {
		bool value;

		explicit operator bool() const { return value; }
	};

	struct endpoint {
		std::size_t value;

		operator std::size_t() const { return value; }
	};

	struct cv_lifetime {
		endpoint begin{}, end{};
		cv_lifetime() = default;
		cv_lifetime(cv_lifetime const &) = default;

		cv_lifetime(cv_lifetime const volatile &other) : begin{other.begin.value}, end{other.end.value} {}

		cv_lifetime(cv_lifetime const volatile &&other) : begin{other.begin.value}, end{other.end.value} {}
	};

	using volatile_lifetimes = reference_range<cv_lifetime, cv_lifetime volatile &>;
	static_assert(std::ranges::forward_range<volatile_lifetimes const>);
	static_assert(atl::detail::member_lifetime<cv_lifetime>);
	static_assert(!atl::lifetime_range<volatile_lifetimes>);
	static_assert(atl::lifetime_range<reference_range<cv_lifetime, cv_lifetime &>>);

	struct kind_expression {
		atl::event_kind value;

		contextual_bool operator==(atl::event_kind other) const { return {value == other}; }
	};

	struct comparison_event : atl::item_event<int> {
		kind_expression kind;
	};

	struct bad_kind {
		operator atl::event_kind() const { return atl::event_kind::add; }

		bool operator==(atl::event_kind) const = delete;
	};

	struct bad_event : atl::item_event<int> {
		bad_kind kind;
	};

	struct bad_key_event : atl::item_event<int> {
		std::string key;
	};

	using comparison_events = reference_range<atl::item_event<int>, comparison_event &>;
	static_assert(std::ranges::input_range<comparison_events const>);
	static_assert(atl::item_event_range<comparison_events>);
	static_assert(!atl::item_event_range<reference_range<atl::item_event<int>, bad_event &>>);
	static_assert(!atl::item_event_range<reference_range<atl::item_event<int>, bad_key_event &>>);

	struct throwing_hash {
		std::size_t operator()(int x) const { return std::size_t(x); }
	};

	struct user_hash {
		std::size_t operator()(int x) const noexcept { return std::size_t(x); }
	};

	struct throwing_conversion_hash {
		struct result {
			operator std::size_t() const { return 0; }
		};

		result operator()(int) const noexcept { return {}; }
	};

	struct mixed_hash {
		using is_transparent = void;

		std::size_t operator()(std::string const &s) const noexcept { return atl::hash<std::string>{}(s); }

		std::size_t operator()(std::string_view s) const { return atl::hash<std::string_view>{}(s); }
	};

	template<typename Hash>
	concept can_hash_map = requires { typename atl::flat_hash_map<int, int, Hash>; };

	template<typename Hash>
	concept can_hash_set = requires { typename atl::flat_hash_set<int, Hash>; };

	template<typename Table, typename Key>
	concept can_find = requires(Table &table, Key const &key) { table.find(key); table.contains(key); table.erase(key); };

	static_assert(can_hash_map<user_hash> && can_hash_set<user_hash>);
	static_assert(!can_hash_map<throwing_hash> && !can_hash_set<throwing_hash>);
	static_assert(!can_hash_map<approved_hash_test::throwing_mix> && !can_hash_set<approved_hash_test::throwing_mix>);
	static_assert(!can_hash_map<approved_hash_test::throwing_return> && !can_hash_set<approved_hash_test::throwing_return>);
	static_assert(!can_hash_map<throwing_conversion_hash> && !can_hash_set<throwing_conversion_hash>);
	static_assert(!atl::detail::nothrow_hasher_for<throwing_conversion_hash, int>);
	static_assert(can_find<atl::flat_hash_set<std::string>, std::string_view>);
	static_assert(!can_find<atl::flat_hash_set<std::string, mixed_hash>, std::string_view>);
	static_assert(!can_find<atl::flat_hash_map<std::string, int, mixed_hash>, std::string_view>);
	static_assert(!can_at_or<atl::flat_hash_map<std::string, int, mixed_hash>, std::string_view, int>);

	enum class enum_key { zero };
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<int>, int>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<atl::int128>, atl::int128>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<atl::uint128>, atl::uint128>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<atl::detail::software_int128>, atl::detail::software_int128>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<atl::detail::software_uint128>, atl::detail::software_uint128>);
	static_assert(atl::detail::nothrow_hasher_for<atl::identity_hash, atl::uint128>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<enum_key>, enum_key>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<double>, double>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<long double>, long double>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::string>, std::string_view>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::string_view>, std::string_view>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<int *>, int *>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::pair<int, int>>, std::pair<int, int>>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::tuple<int, double>>, std::tuple<int, double>>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::vector<int>>, std::vector<int>>);
	static_assert(atl::detail::nothrow_hasher_for<atl::hash<std::array<int, 2>>, std::array<int, 2>>);
}

TEST_CASE("approved interfaces: explicit coordinate construction and move-only comparators", "[coordinate_compression]") {
	std::vector<explicit_value> input;
	input.emplace_back(3);
	input.emplace_back(1);
	input.emplace_back(3);
	atl::coordinate_compression cc(input);
	REQUIRE(cc.size() == 2);
	REQUIRE(cc.rank(explicit_value(3)) == 1);
	atl::coordinate_compression<explicit_value> explicit_coordinates(std::array{3, 1, 3});
	REQUIRE(explicit_coordinates.rank(explicit_value(3)) == 1);
	REQUIRE(atl::compress(input).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::experimental::compress_by_pair_sort(input).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::experimental::compress_by_index_sort(input).ranks == std::vector<atl::uint32>{1, 0, 1});

	atl::coordinate_compression<int, move_compare> moved(std::array{3, 1, 3}, std::identity{}, move_compare{});
	REQUIRE(moved.rank(2) == 1);
	REQUIRE(moved.rank_upper(3) == 2);
	REQUIRE(moved.contains(1));
	REQUIRE(!moved.contains(2));
	REQUIRE(moved.ranks(std::array{1, 3, 0, 2}) == std::vector<atl::uint32>{0, 1, 0, 1});
	std::vector<int> larger_universe;
	for (int i = 0; i < 256; ++i) larger_universe.push_back(i);
	atl::coordinate_compression<int, move_compare> search(atl::sorted_unique, std::move(larger_universe), move_compare{});
	REQUIRE(search.ranks(std::array{42}) == std::vector<atl::uint32>{42});
	REQUIRE(atl::experimental::ranks_by_search_or_gallop(std::vector{1, 3, 5, 7}, move_compare{}, std::array{5}, {.merge_divisor = 1, .is_galloping = true}) == std::vector<atl::uint32>{2});
	REQUIRE(atl::compress(std::array{3, 1, 3}, std::identity{}, move_compare{}).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::experimental::compress_by_pair_sort(std::array{3, 1, 3}, std::identity{}, move_compare{}).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::experimental::compress_by_index_sort(std::array{3, 1, 3}, std::identity{}, move_compare{}).ranks == std::vector<atl::uint32>{1, 0, 1});
	REQUIRE(atl::experimental::ranks_by_search_or_gallop(std::vector{1, 3}, move_compare{}, std::array{3, 1}) == std::vector<atl::uint32>{1, 0});
	REQUIRE(atl::experimental::ranks_by_sortedness_prescan(std::vector{1, 3}, move_compare{}, std::array{1, 3}) == std::vector<atl::uint32>{0, 1});
}

TEST_CASE("approved interfaces: named position lvalues and chainable backing proxies", "[hierarchical_bitset][persistent_dsu]") {
	atl::hierarchical_bitset bits(8, std::vector<explicit_lvalue_position>{{1}, {5}, {1}});
	REQUIRE(bits.size() == 2);
	REQUIRE(bits.contains(5));
	lvalue_position_proxy positions[2]{{{2}}, {{6}}};
	atl::hierarchical_bitset proxy_bits(8, lvalue_positions{positions, 2});
	REQUIRE(proxy_bits.contains(2));
	REQUIRE(proxy_bits.contains(6));
	atl::persistent_dsu<proxy_backing<true>> dsu(3);
	auto united = dsu.unite(0, 1);
	REQUIRE(united.same(0, 1));
	REQUIRE(united.size(0) == 2);
	REQUIRE(!dsu.same(0, 1));
}

TEST_CASE("approved interfaces: event references need comparisons rather than enum conversions", "[time_divide_and_conquer]") {
	comparison_event records[2];
	records[0].key = 4;
	records[0].kind.value = atl::event_kind::add;
	records[1].key = 4;
	records[1].kind.value = atl::event_kind::remove;
	auto result = atl::lifetimes_of(comparison_events{records, 2});
	REQUIRE(result == std::vector<atl::item_lifetime>{{0, 1}});
}

TEST_CASE("approved interfaces: weighted operations allow explicit move and default construction", "[weighted_dsu]") {
	atl::weighted_dsu<int, explicit_op, explicit_inverse> defaults(3);
	atl::weighted_dsu<int, explicit_op, explicit_inverse> identity(3, 0);
	atl::weighted_dsu<int, explicit_op, explicit_inverse> composition(3, 0, explicit_op{});
	atl::weighted_dsu<int, explicit_op, explicit_inverse> all(3, 0, explicit_op{}, explicit_inverse{});
	for (auto *dsu : {&defaults, &identity, &composition, &all}) {
		REQUIRE(dsu->unite(0, 1, 7) == atl::unite_outcome::merged);
		REQUIRE(dsu->difference(0, 1) == 7);
	}
	atl::rollback_weighted_dsu<int, explicit_op, explicit_inverse> rollback_defaults(3);
	atl::rollback_weighted_dsu<int, explicit_op, explicit_inverse> rollback_identity(3, 0);
	atl::rollback_weighted_dsu<int, explicit_op, explicit_inverse> rollback_composition(3, 0, explicit_op{});
	atl::rollback_weighted_dsu<int, explicit_op, explicit_inverse> rollback_all(3, 0, explicit_op{}, explicit_inverse{});
	for (auto *dsu : {&rollback_defaults, &rollback_identity, &rollback_composition, &rollback_all}) {
		REQUIRE(dsu->unite(0, 1, 7) == atl::unite_outcome::merged);
		dsu->undo();
		REQUIRE(!dsu->same(0, 1));
	}
}

TEST_CASE("approved interfaces: experimental weighted constructors store explicit operations", "[weighted_dsu]") {
	weighted_variant<explicit_op, explicit_inverse> defaults(3);
	weighted_variant<explicit_op, explicit_inverse> identity(3, 0);
	weighted_variant<explicit_op, explicit_inverse> composition(3, 0, explicit_op{});
	weighted_variant<explicit_op, explicit_inverse> all(3, 0, explicit_op{}, explicit_inverse{});
	for (auto *dsu : {&defaults, &identity, &composition, &all}) {
		REQUIRE(dsu->unite(0, 1, 7) == atl::unite_outcome::merged);
		REQUIRE(dsu->difference(0, 1) == 7);
	}
	rollback_variant<explicit_op, explicit_inverse> rollback_defaults(3);
	rollback_variant<explicit_op, explicit_inverse> rollback_identity(3, 0);
	rollback_variant<explicit_op, explicit_inverse> rollback_composition(3, 0, explicit_op{});
	rollback_variant<explicit_op, explicit_inverse> rollback_all(3, 0, explicit_op{}, explicit_inverse{});
	for (auto *dsu : {&rollback_defaults, &rollback_identity, &rollback_composition, &rollback_all}) {
		REQUIRE(dsu->unite(0, 1, 7) == atl::unite_outcome::merged);
		dsu->undo();
		REQUIRE(!dsu->same(0, 1));
	}
}

#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/ds/difference_fenwick.hpp>
#include <atl/ds/direct_rmq.hpp>
#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/sparse_table.hpp>
#include <atl/ds/wavelet_matrix.hpp>
#include <forward_list>

namespace ds2_boundaries {
	struct not_callable {};

	struct bad_result {
		void operator()(int, int) const {}
	};

	template<typename Op>
	concept fenwick_operation = requires { typename atl::binary_indexed_tree<int, 1, Op>; };
	template<typename Op>
	concept sparse_operation = requires { typename atl::sparse_table<int, Op>; typename atl::disjoint_sparse_table<int, Op>; };
	template<typename Op>
	concept difference_operation = requires { typename atl::difference_fenwick<int, Op>; };
	template<typename Function>
	concept envelope = requires { typename atl::li_chao_tree<int, std::less<>, Function>; typename atl::dynamic_li_chao_tree<int, std::less<>, Function>; };
	template<typename Tree, typename Predicate>
	concept can_search = requires(Tree const &tree, Predicate pred) { tree.max_right(pred); };

	struct rvalue_number {
		constexpr operator unsigned() && { return 1; }
	};

	using rvalue_numbers = std::array<rvalue_number, 2>;
	static_assert(!std::constructible_from<atl::binary_indexed_tree<unsigned>, std::array<std::size_t, 1>, rvalue_numbers &>);
	static_assert(!std::constructible_from<atl::wavelet_matrix<unsigned>, rvalue_numbers &>);
	static_assert(!std::constructible_from<atl::binary_indexed_tree<int>, std::array<std::size_t, 1>, std::vector<std::string>>);

	struct nonconst_size_range {
		int values[2]{};

		constexpr int *begin() { return values; }

		constexpr int *end() { return values + 2; }

		constexpr std::size_t size() { return 2; }
	};

	static_assert(std::ranges::sized_range<nonconst_size_range>);
	static_assert(!std::constructible_from<atl::binary_indexed_tree<int>, nonconst_size_range &>);
	using mutable_rows = decltype(std::views::iota(0, 2) | std::views::transform([offset = 0](int x) mutable { return std::array{x + offset, x}; }));
	static_assert(!std::constructible_from<atl::binary_indexed_tree<int, 2>, mutable_rows &>);

	static_assert(!fenwick_operation<not_callable> && !fenwick_operation<bad_result>);
	static_assert(!sparse_operation<not_callable> && !sparse_operation<bad_result>);
	static_assert(!difference_operation<not_callable> && !difference_operation<bad_result>);
	static_assert(!envelope<not_callable>);
	static_assert(!can_search<atl::binary_indexed_tree<int>, not_callable>);
	static_assert(!std::constructible_from<atl::binary_indexed_tree<int, 2>, std::vector<std::forward_list<int>>>);
	static_assert(!std::constructible_from<atl::binary_indexed_tree<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::sparse_table<int, std::plus<>>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::disjoint_sparse_table<int, std::plus<>>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::wavelet_matrix<unsigned>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::direct_rmq<int>, std::vector<int>>);
	static_assert(!std::constructible_from<atl::direct_rmq<int>, std::vector<short> &>);
	static_assert(!std::constructible_from<atl::direct_rmq<int>, std::span<int volatile>>);
	static_assert(!atl::line_value<std::string>);
}

#include <atl/ds/dual_segment_tree.hpp>
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <atl/ds/dynamic_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/ds/segment_tree_beats.hpp>

namespace segment_family_boundaries {
	using dense = atl::segment_tree<int, std::plus<>>;
	using lazy = atl::lazy_segment_tree<int, int, std::plus<>, std::plus<>, std::plus<>>;
	using dual = atl::dual_segment_tree<int, int, std::plus<>, std::plus<>>;
	using commuting = atl::dual_segment_tree<int, int, std::plus<>, std::plus<>, atl::commutative_tags_t>;
	using engine = atl::segment_tree_beats<atl::beats::extremes<int>, atl::beats::clamp_add<int>>;
	using beats = atl::beats_segment_tree<int>;

	struct invalid_fill {
		int operator()(std::size_t) const;
		void *operator()(std::size_t, std::size_t) const;
	};

	template<typename Fill>
	concept sparse_lazy_fill = requires { typename atl::dynamic_lazy_segment_tree<int, int, std::plus<>, std::plus<>, std::plus<>, Fill>; };
	static_assert(!sparse_lazy_fill<invalid_fill>);
	static_assert(!std::constructible_from<dense, std::vector<std::string>, int>);
	static_assert(!std::constructible_from<lazy, std::vector<std::string>, int, int>);
	static_assert(!std::constructible_from<dual, std::vector<std::string>, int>);
	static_assert(!std::constructible_from<commuting, atl::commutative_tags_t, std::vector<std::string>, int>);
	static_assert(!std::constructible_from<engine, std::vector<std::string>>);
	static_assert(!std::constructible_from<engine, std::vector<std::string>, atl::beats::extremes<int>, atl::beats::clamp_add<int>>);
	static_assert(!std::constructible_from<beats, std::vector<std::string>>);

	struct bad_update {
		void operator()(int) const;
	};

	template<typename Tree, typename Update>
	concept can_update = requires(Tree &tree, Update update) { tree.update(0, update); };
	static_assert(!can_update<dense, bad_update>);
	static_assert(!can_update<atl::dynamic_segment_tree<int, std::plus<>>, bad_update>);
}

#include <atl/experimental/ds/segment_tree_beats_variants.hpp>

namespace segment_family_boundaries {
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_recursive_compact_nodes<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_explicit_stack<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_fixed_clamp_add<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_three_adds<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_optional_mapping<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_unconditional_push<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_no_collapse<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_tagless_layout<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::experimental::beats_segment_tree_by_tagless_layout<int, int, atl::experimental::beats_layout::soa>, std::vector<std::string>>);
}

namespace segment_family_boundaries {
	struct not_callable {};

	template<typename Tree, typename Predicate>
	concept can_search = requires(Tree &tree, Predicate predicate) { tree.max_right(0, predicate); };
	template<typename Tree, typename Rule>
	concept can_derive = requires(Tree &tree, Rule rule) { tree.apply_derived(0, 1, rule); };
	static_assert(!can_search<dense, not_callable>);
	static_assert(!can_search<lazy, not_callable>);
	static_assert(!can_search<engine, not_callable>);
	static_assert(!can_search<beats, not_callable>);
	static_assert(!can_derive<engine, not_callable>);
}

#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_map.hpp>
#include <atl/ds/splay_sequence.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/ds/treap_sequence.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>

namespace treap_splay_boundaries {
	struct not_callable {};

	struct bad_update {
		void operator()(int) const;
	};

	struct non_const_range {
		int *begin();
		int *end();
		std::size_t size();
	};

	template<typename Range>
	concept can_build = requires(atl::splay_links &links, Range const &range) { atl::splay_engine::build_balanced(links, range); };
	static_assert(!can_build<non_const_range>);
	static_assert(!can_build<std::vector<std::string>>);
	template<typename Tree, typename Update>
	concept can_update = requires(Tree &tree, Update update) { tree.update(0, update); };
	static_assert(!can_update<atl::treap_map<int, int>, bad_update>);
	static_assert(!can_update<atl::splay_map<int, int>, bad_update>);
	template<typename Tree, typename Predicate>
	concept can_search = requires(Tree &tree, Predicate predicate) { tree.max_right(0, predicate); };
	static_assert(!can_search<atl::treap_sequence<int, std::plus<>>, not_callable>);
	static_assert(!can_search<atl::splay_sequence<int, std::plus<>>, not_callable>);
	static_assert(!std::constructible_from<atl::treap_sequence<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::splay_sequence<int>, std::vector<std::string>>);
	static_assert(!std::constructible_from<atl::treap_sequence<int, std::plus<>>, std::vector<std::string>, int, std::plus<>>);
	static_assert(!std::constructible_from<atl::splay_sequence<int, std::plus<>>, std::vector<std::string>, int, std::plus<>>);
	template<typename Engine, typename Store>
	concept can_splay = requires(Store &store) { Engine::splay(store, 1); };
	using parity_engine = atl::experimental::splay_engine_by_strategy<atl::experimental::splay_shape::bottom_up, atl::experimental::splay_path_push::parity_walk>;
	using buffer_engine = atl::experimental::splay_engine_by_strategy<atl::experimental::splay_shape::bottom_up, atl::experimental::splay_path_push::store_buffer>;

	struct push_store : atl::splay_links {
		void push(atl::uint32);
	};

	static_assert(!can_splay<parity_engine, atl::splay_links>);
	static_assert(!can_splay<buffer_engine, push_store>);
	static_assert(!atl::splay_probe<not_callable>);
	static_assert(!atl::detail::treap_store<atl::splay_links>);
}

namespace treap_splay_boundaries {
	struct supplied_order {
		supplied_order() = delete;
		bool operator()(int, int) const;
	};

	static_assert(!std::default_initializable<atl::treap_map<int, int, supplied_order>>);
	static_assert(!std::default_initializable<atl::splay_map<int, int, supplied_order>>);
}

namespace treap_splay_boundaries {
	struct parent_store_signature {
		atl::uint32 left(atl::uint32) const;
		atl::uint32 right(atl::uint32) const;
		atl::uint32 size(atl::uint32) const;
		atl::uint32 priority(atl::uint32) const;
		atl::uint32 rebind(atl::uint32, atl::uint32, atl::uint32);
		atl::uint32 push(atl::uint32);
		atl::uint32 parent(atl::uint32) const;
		void set_parent(atl::uint32, atl::uint32);
	};

	static_assert(atl::detail::treap_parent_store<parent_store_signature>);
	template<typename Store>
	concept can_insert_by_handle = requires(Store &store) { atl::detail::treap_base::insert_before_node(store, 1, 2); };
	static_assert(!can_insert_by_handle<parent_store_signature>);
	template<typename Draw>
	concept can_merge_weighted = requires(parent_store_signature &store, Draw draw) { atl::detail::treap_base::merge_weighted(store, 1, 2, draw); };
	static_assert(!can_merge_weighted<not_callable>);

	struct fold_store : atl::sized_splay_links {
		int const &value(atl::uint32) const;
		int const &aggregate(atl::uint32) const;
	};

	template<typename Fold>
	concept can_fold_prefix = requires(fold_store &store, Fold fold) { atl::splay_engine::max_right(store, 0, 0, fold, [](int) { return true; }); };
	static_assert(!can_fold_prefix<not_callable>);
}

#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/ds/top_tree.hpp>
#include <atl/experimental/ds/euler_tour_tree_variants.hpp>

namespace dynamic_forest_boundary_constraints {
	template<typename Value, typename Op>
	concept path_algebra = requires { typename atl::link_cut_tree<Value, Op>; };
	template<typename Value, typename Op>
	concept tour_algebra = requires { typename atl::euler_tour_tree<Value, Op>; };
	template<typename Value, typename Op, typename Inverse>
	concept subtree_algebra = requires { typename atl::subtree_link_cut_tree<Value, Op, Inverse>; };

	static_assert(!path_algebra<int, std::identity>);
	static_assert(!tour_algebra<int, std::identity>);
	static_assert(!subtree_algebra<int, std::plus<>, std::plus<>>);
	static_assert(path_algebra<int, std::plus<>>);
	static_assert(tour_algebra<int, std::plus<>>);

	using path = atl::link_cut_tree<int, std::plus<>>;
	using subtree = atl::subtree_link_cut_tree<int, std::plus<>, std::negate<>>;
	using tour = atl::euler_tour_tree<int, std::plus<>>;
	using clusters = atl::sum_top_tree<int>;
	using splay_tour = atl::experimental::euler_tour_tree_by_splay<int, std::plus<>>;

	static_assert(!std::constructible_from<path, std::vector<std::string> &, int, std::plus<>>);
	static_assert(!std::constructible_from<subtree, std::vector<std::string> &, int, std::plus<>, std::negate<>>);
	static_assert(!std::constructible_from<tour, std::vector<std::string> &, int, std::plus<>>);
	static_assert(!std::constructible_from<splay_tour, std::vector<std::string> &, int, std::plus<>>);
	static_assert(!std::constructible_from<clusters, std::vector<std::string> &>);
}

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/ds/persistent_heap.hpp>
#include <atl/ds/persistent_lazy_segment_tree.hpp>
#include <atl/ds/persistent_segment_tree.hpp>
#include <atl/ds/persistent_treap_sequence.hpp>

namespace publication_persistent_constraints {
	struct invalid_callable {};

	template<typename Compare>
	concept heap_order = requires { typename atl::persistent_heap<int, Compare>; typename atl::persistent_heap_forest<int, Compare>; };
	template<typename Compare>
	concept meldable_order = requires { typename atl::meldable_heap_forest<int, Compare>; };
	template<typename Op>
	concept segment_fold = requires { typename atl::persistent_segment_tree<int, Op>; };
	template<typename Op>
	concept sequence_fold = requires { typename atl::persistent_treap_sequence<int, Op>; };
	static_assert(!heap_order<invalid_callable>);
	static_assert(!meldable_order<invalid_callable>);
	static_assert(!segment_fold<invalid_callable>);
	static_assert(!sequence_fold<invalid_callable>);
	template<typename Update>
	concept point_update = requires(atl::persistent_segment_tree<int, std::plus<>> const &tree, Update update) { tree.update(0, update); };
	static_assert(!point_update<invalid_callable>);
	template<typename Choose>
	concept descent = requires(atl::persistent_segment_tree<int, std::plus<>> const &tree, Choose choose) { tree.descend(tree, tree, choose); };
	static_assert(!descent<invalid_callable>);
	static_assert(heap_order<std::reference_wrapper<std::less<>>>);
}

namespace publication_forwarding_constraints {
	struct invalid_element {};

	using range = std::vector<invalid_element>;
	static_assert(!std::constructible_from<atl::persistent_segment_tree<int, std::plus<>>, range &, int, std::plus<>>);
	static_assert(!std::constructible_from<atl::persistent_treap_sequence<int>, range &>);
}

#include <atl/experimental/ds/link_cut_tree_variants.hpp>
#include <atl/experimental/ds/top_tree_variants.hpp>

namespace owner_interface_constraints {
	using path_variant = atl::experimental::link_cut_tree_by_configuration<atl::splay_engine, atl::splay_layout_interleaved, true, false>;
	using top_variant = atl::experimental::top_tree_by_interleaved_layout<int, int, int, std::plus<>, std::plus<>, std::plus<>, std::identity, atl::commutative_fold_t>;

	struct non_const_parents {
		atl::uint32 *begin();
		atl::uint32 *end();
	};

	struct input_parents {
		struct iterator {
			using value_type = atl::uint32;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::input_iterator_tag;
			atl::uint32 const *position{};
			atl::uint32 operator*() const;
			iterator &operator++();
			void operator++(int);
			bool operator==(iterator const &) const = default;
		};

		iterator begin() const;
		iterator end() const;
	};

	static_assert(std::ranges::forward_range<non_const_parents>);
	static_assert(!std::ranges::range<non_const_parents const>);
	static_assert(std::ranges::input_range<input_parents const>);
	static_assert(atl::graph::vertex_range<input_parents const>);
	static_assert(!std::ranges::forward_range<input_parents const>);

	template<typename Tree, typename Parents>
	concept can_assign = requires(Tree &tree, Parents const &parents) { tree.assign_parents(parents); };
	template<typename Tree, typename Parents>
	concept can_assign_weights = requires(Tree &tree, Parents const &parents, std::vector<int> &weights) { tree.assign_parents(parents, weights); };
	template<typename Parents>
	constexpr bool rejects_parents = !can_assign<atl::link_cut_tree<>, Parents> && !can_assign<path_variant, Parents> && !can_assign<top_variant, Parents> && !can_assign<atl::subtree_link_cut_tree<int, std::plus<>, std::negate<>>, Parents> && !can_assign<atl::euler_tour_tree<>, Parents> && !can_assign<atl::sum_top_tree<int>, Parents> && !can_assign<atl::experimental::euler_tour_tree_by_splay<>, Parents> && !can_assign_weights<atl::edge_link_cut_tree<int, std::plus<>>, Parents> && !can_assign_weights<atl::diameter_top_tree<int>, Parents>;
	static_assert(rejects_parents<non_const_parents>);
	static_assert(rejects_parents<input_parents>);
	static_assert(can_assign<atl::link_cut_tree<>, std::vector<atl::uint32>>);
	static_assert(can_assign_weights<atl::edge_link_cut_tree<int, std::plus<>>, std::vector<atl::uint32>>);

	struct supplied_value {
		supplied_value() = delete;
		explicit supplied_value(int);
	};

	struct combine {
		supplied_value operator()(supplied_value const &, supplied_value const &) const;
	};

	struct invert {
		supplied_value operator()(supplied_value const &) const;
	};

	struct map {
		int operator()(supplied_value const &, int) const;
	};

	template<typename Value>
	concept path_value = requires { typename atl::link_cut_tree<Value, combine>; };
	template<typename Value>
	concept tour_value = requires { typename atl::euler_tour_tree<Value, combine>; };
	template<typename Value>
	concept splay_tour_value = requires { typename atl::experimental::euler_tour_tree_by_splay<Value, combine>; };
	template<typename Value>
	concept variant_path_value = requires { typename atl::experimental::link_cut_tree_by_configuration<atl::splay_engine, atl::splay_layout_interleaved, true, false, Value, combine>; };
	template<typename Value>
	concept subtree_value = requires { typename atl::subtree_link_cut_tree<Value, combine, invert>; };
	template<typename Tag>
	concept path_tag = requires { typename atl::link_cut_tree<int, std::plus<>, Tag, map, combine>; };
	template<typename Tag>
	concept tour_tag = requires { typename atl::euler_tour_tree<int, std::plus<>, Tag, map, combine>; };
	static_assert(std::copyable<supplied_value>);
	static_assert(!std::default_initializable<supplied_value>);
	static_assert(atl::detail::link_cut_algebra<supplied_value, combine, void, void, void, void, void>);
	static_assert(atl::detail::euler_tour_algebra<supplied_value, combine, void, void, void>);
	static_assert(atl::detail::link_cut_algebra<int, std::plus<>, supplied_value, map, combine, void, void>);
	static_assert(atl::detail::euler_tour_algebra<int, std::plus<>, supplied_value, map, combine>);
	static_assert(!path_value<supplied_value>);
	static_assert(!tour_value<supplied_value>);
	static_assert(!splay_tour_value<supplied_value>);
	static_assert(!variant_path_value<supplied_value>);
	static_assert(!subtree_value<supplied_value>);
	static_assert(!path_tag<supplied_value>);
	static_assert(!tour_tag<supplied_value>);

	using captured_compare = decltype([offset = 0](int a, int b) { return a + offset < b; });
	static_assert(!std::default_initializable<atl::persistent_heap<int, captured_compare>>);
	static_assert(std::constructible_from<atl::persistent_heap<int, captured_compare>, captured_compare>);

	struct assign_sum {
		assign_sum() = default;
		assign_sum(assign_sum const &) = delete;
		assign_sum(assign_sum &&) = default;
		assign_sum &operator=(assign_sum const &) = default;

		int operator()(int a, int b) const { return a + b; }
	};

	struct move_sum {
		move_sum() = default;
		move_sum(move_sum const &) = delete;
		move_sum(move_sum &&) = default;

		int operator()(int a, int b) const { return a + b; }
	};

	struct copy_sum {
		copy_sum() = default;
		copy_sum(copy_sum const &) = default;
		copy_sum(copy_sum &&) = delete;

		int operator()(int a, int b) const { return a + b; }
	};

	template<typename Sequence>
	concept can_compact = requires(Sequence const &sequence) { sequence.compacted(); };
	static_assert(std::is_copy_assignable_v<assign_sum> && !std::copy_constructible<assign_sum>);
	static_assert(std::is_copy_constructible_v<copy_sum> && !std::copy_constructible<copy_sum>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, move_sum>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, assign_sum>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, assign_sum, int, assign_sum, assign_sum>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, copy_sum>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, copy_sum, int, copy_sum, copy_sum>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, std::plus<>, int, move_sum, std::plus<>>>);
	static_assert(!can_compact<atl::persistent_treap_sequence<int, std::plus<>, int, std::plus<>, move_sum>>);
	static_assert(can_compact<atl::persistent_treap_sequence<int>>);
	static_assert(can_compact<atl::persistent_treap_sequence<int, std::plus<>>>);
}
