#include <atl/ds/coordinate_compression.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/experimental/ds/coordinate_compression_variants.hpp>
#include <atl/experimental/ds/weighted_dsu_variants.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
	struct nonconst_less {
		constexpr bool operator()(int a, int b) { return a < b; }
	};

	constexpr bool check_int_universe(auto const &cc) {
		return cc.size() == 2 && cc[0] == 1 && cc[1] == 2
			&& cc.rank(2) == 1 && cc.rank_upper(2) == 2 && cc.contains(1) && !cc.contains(3)
			&& cc.ranks(std::array{1, 2}) == std::vector<atl::uint32>{0, 1}
			&& cc.ranks(std::array{2, 1, 2}) == std::vector<atl::uint32>{1, 0, 1};
	}

	// A const reference_wrapper must keep invoking the referenced non-const comparator.
	static_assert([] {
		nonconst_less less;
		std::vector<int> v{2, 1};
		atl::coordinate_compression cc(v, std::identity{}, std::ref(less));
		if (!check_int_universe(cc)) return false;
		auto check = [](auto const &c) { return check_int_universe(c.universe) && c.ranks == std::vector<atl::uint32>{1, 0}; };
		if (!check(atl::compress(v, std::identity{}, std::ref(less)))
			|| !check(atl::experimental::compress_by_pair_sort(v, std::identity{}, std::ref(less)))
			|| !check(atl::experimental::compress_by_index_sort(v, std::identity{}, std::ref(less)))) return false;

		// Exercise binary search below the merge threshold as well as the merge/sort paths above.
		std::vector<int> large;
		for (int i = 0; i < 129; ++i) large.push_back(i);
		atl::coordinate_compression<int, std::reference_wrapper<nonconst_less>> search(atl::sorted_unique, std::move(large), std::ref(less));
		if (search.ranks(std::array{64}) != std::vector<atl::uint32>{64}) return false;

		std::vector<int> universe{1, 2};
		return atl::experimental::ranks_by_search_or_gallop(universe, std::ref(less), std::array{2, 1}) == std::vector<atl::uint32>{1, 0}
			&& atl::experimental::ranks_by_search_or_gallop(universe, std::ref(less), std::array{2}, {.merge_divisor = 1, .is_galloping = true}) == std::vector<atl::uint32>{1}
			&& atl::experimental::ranks_by_sortedness_prescan(universe, std::ref(less), std::array{1, 2}) == std::vector<atl::uint32>{0, 1}
			&& atl::experimental::ranks_by_sortedness_prescan(universe, std::ref(less), std::array{2, 1}) == std::vector<atl::uint32>{1, 0};
	}());

	enum E : int {
		zero,
		one
	};

	struct enum_iterator {
		using value_type = E;
		using difference_type = std::ptrdiff_t;
		using iterator_concept = std::input_iterator_tag;
		int *p;

		constexpr int &operator*() const { return *p; }

		constexpr enum_iterator &operator++() {
			++p;
			return *this;
		}

		constexpr void operator++(int) { ++p; }

		constexpr bool operator==(enum_iterator const &) const = default;
	};

	using enum_range = std::ranges::subrange<enum_iterator>;
	static_assert(std::ranges::input_range<enum_range> && !std::ranges::forward_range<enum_range>);
	static_assert(!std::constructible_from<E, int &> && !std::constructible_from<E, int const &>);

	// Identity preserves the iterator's enum value_type even though dereference yields int&.
	static_assert([] {
		int a[]{0, 1, 0};
		auto r = std::ranges::subrange(enum_iterator{a}, enum_iterator{a + 3});
		auto check = [](auto const &c) {
			static_assert(std::same_as<typename std::remove_cvref_t<decltype(c.universe)>::value_type, E>);
			return c.universe.size() == 2 && c.universe[0] == zero && c.universe[1] == one
				&& c.ranks == std::vector<atl::uint32>{0, 1, 0};
		};
		atl::coordinate_compression cc(r);
		static_assert(std::same_as<decltype(cc)::value_type, E>);
		atl::coordinate_compression<E> explicit_type(r);
		return cc.size() == 2 && explicit_type.rank(one) == 1
			&& check(atl::compress(r)) && check(atl::compress(std::move(r)))
			&& check(atl::experimental::compress_by_pair_sort(r))
			&& check(atl::experimental::compress_by_index_sort(r));
	}());

	struct throwing_destructor_less {
		constexpr ~throwing_destructor_less() noexcept(false) {}

		constexpr bool operator()(int a, int b) const { return a < b; }
	};

	static_assert(!std::destructible<throwing_destructor_less>);
	static_assert(std::is_constructible_v<throwing_destructor_less, throwing_destructor_less &&>);

	// Construction and algorithms need no noexcept guarantee from the comparator destructor.
	static_assert([] {
		atl::coordinate_compression<int, throwing_destructor_less> adopted(atl::sorted_unique, std::vector<int>{1, 2}, throwing_destructor_less{});
		atl::coordinate_compression sorted(std::array{2, 1}, std::identity{}, throwing_destructor_less{});
		return check_int_universe(adopted) && check_int_universe(sorted)
			&& check_int_universe(atl::compress(std::array{2, 1}, std::identity{}, throwing_destructor_less{}).universe)
			&& check_int_universe(atl::experimental::compress_by_pair_sort(std::array{2, 1}, std::identity{}, throwing_destructor_less{}).universe)
			&& check_int_universe(atl::experimental::compress_by_index_sort(std::array{2, 1}, std::identity{}, throwing_destructor_less{}).universe);
	}());

	struct counted_potential {
		int *transfers = nullptr;
		constexpr counted_potential() = default;

		constexpr explicit counted_potential(int &count) : transfers(&count) {}

		constexpr counted_potential(counted_potential const &other) : transfers(other.transfers) {
			if (transfers) ++*transfers;
		}

		constexpr counted_potential(counted_potential &&other) : transfers(other.transfers) {
			if (transfers) ++*transfers;
		}

		constexpr counted_potential &operator=(counted_potential const &) = default;
		constexpr counted_potential &operator=(counted_potential &&) = default;
		constexpr bool operator==(counted_potential const &) const = default;

		constexpr counted_potential operator+(counted_potential const &) const { return {}; }

		constexpr counted_potential operator-() const { return {}; }
	};

	struct counted_add {
		int *copies = nullptr;
		constexpr counted_add() = default;

		constexpr explicit counted_add(int &count) : copies(&count) {}

		// No move constructor: transferring an rvalue copies it, as in the review example.
		constexpr counted_add(counted_add const &other) : copies(other.copies) {
			if (copies) ++*copies;
		}

		template<typename T>
		constexpr T operator()(T const &a, T const &b) const { return a + b; }
	};

	struct counted_inverse {
		int *copies = nullptr;
		constexpr counted_inverse() = default;

		constexpr explicit counted_inverse(int &count) : copies(&count) {}

		constexpr counted_inverse(counted_inverse const &other) : copies(other.copies) {
			if (copies) ++*copies;
		}

		template<typename T>
		constexpr T operator()(T const &a) const { return -a; }
	};

	template<typename Potential, typename Op, typename Inverse>
	using interleaved_variant = atl::experimental::weighted_dsu_by_compression_and_layout<atl::experimental::weighted_dsu_compression::recursive, atl::experimental::weighted_dsu_layout::interleaved, Potential, Op, Inverse>;

	template<typename Potential, typename Op, typename Inverse>
	using split_variant = atl::experimental::weighted_dsu_by_compression_and_layout<atl::experimental::weighted_dsu_compression::recursive, atl::experimental::weighted_dsu_layout::split, Potential, Op, Inverse>;

	template<template<typename, typename, typename> typename Dsu>
	constexpr bool check_constructor_transfers() {
		int copies = 0;
		Dsu<int, counted_add, std::negate<>> example(0, 0, counted_add(copies));
		if (copies != 1 || example.elements() != 0) return false;

		using dsu_type = Dsu<counted_potential, counted_add, counted_inverse>;
		int identity_transfers = 0, op_copies = 0, inverse_copies = 0;
		dsu_type all(0, counted_potential(identity_transfers), counted_add(op_copies), counted_inverse(inverse_copies));
		if (op_copies != 1 || inverse_copies != 1) return false;
		int const expected_identity_transfers = identity_transfers;

		identity_transfers = op_copies = 0;
		dsu_type composition(0, counted_potential(identity_transfers), counted_add(op_copies));
		if (op_copies != 1 || identity_transfers != expected_identity_transfers) return false;
		identity_transfers = 0;
		dsu_type identity(0, counted_potential(identity_transfers));
		return identity_transfers == expected_identity_transfers;
	}

	// Compile-time counters verify the costs without building or running a test executable.
	static_assert(check_constructor_transfers<atl::weighted_dsu>());
	static_assert(check_constructor_transfers<atl::rollback_weighted_dsu>());
	static_assert(check_constructor_transfers<interleaved_variant>());
	static_assert(check_constructor_transfers<split_variant>());
	static_assert(check_constructor_transfers<atl::experimental::rollback_weighted_dsu_by_interleaved_nodes>());

	struct Add {
		inline static int copies = 0;
		Add() = default;

		Add(Add const &) { ++copies; }

		int operator()(int a, int b) const { return a + b; }
	};
}

TEST_CASE("weighted constructors: copy-only operations transfer once", "[weighted_dsu]") {
	auto check = []<typename Dsu> {
		Add::copies = 0;
		Dsu d(0, 0, Add{});
		REQUIRE(Add::copies == 1);
		REQUIRE(d.elements() == 0);
	};
	check.template operator()<atl::weighted_dsu<int, Add>>();
	check.template operator()<atl::rollback_weighted_dsu<int, Add>>();
	check.template operator()<interleaved_variant<int, Add, std::negate<>>>();
	check.template operator()<split_variant<int, Add, std::negate<>>>();
	check.template operator()<atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<int, Add>>();
}
