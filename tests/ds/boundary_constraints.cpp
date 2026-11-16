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
				auto previous = *this;
				++*this;
				return previous;
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

		friend no_default_potential operator+(no_default_potential const &a, no_default_potential const &b) { return no_default_potential(a.value + b.value); }

		friend no_default_potential operator-(no_default_potential const &a) { return no_default_potential(-a.value); }
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
