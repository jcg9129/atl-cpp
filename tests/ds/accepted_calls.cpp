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

#include <optional>

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

#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/ds/difference_fenwick.hpp>
#include <atl/ds/direct_rmq.hpp>
#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/sparse_table.hpp>
#include <atl/ds/wavelet_matrix.hpp>
#include <span>

namespace ds2_accepted {
	struct nonconst_sum {
		constexpr int operator()(int x, int y) { return x + y; }
	};

	struct nonconst_min {
		constexpr int operator()(int x, int y) { return x < y ? x : y; }
	};

	struct nonconst_inverse {
		constexpr int operator()(int x) { return -x; }
	};

	struct nonconst_compare {
		constexpr bool operator()(long long x, long long y) { return x < y; }
	};

	static_assert([] {
		nonconst_sum sum;
		std::array const values{1, 2, 3};
		using Op = std::reference_wrapper<nonconst_sum>;
		atl::binary_indexed_tree<int, 1, Op> sized(3, 0, std::ref(sum));
		atl::binary_indexed_tree<int, 1, Op> nested(values, 0, std::ref(sum));
		atl::binary_indexed_tree<int, 1, Op> flat(std::array<std::size_t, 1>{3}, std::span(values), 0, std::ref(sum));
		sized.update(6, 1);
		int const limit = 3;
		auto pred = [limit](int x) mutable { return x <= limit; };
		auto factory = atl::make_binary_indexed_tree(0, 3);
		auto from = atl::make_binary_indexed_tree_from(values);
		auto from_flat = atl::make_binary_indexed_tree_from(values, 3);
		auto product = atl::make_binary_indexed_tree_from<std::multiplies<>>(1, values);
		return sized.prefix_fold(2) == 6 && nested.max_right(std::ref(pred)) == 2 && flat.prefix_fold(3) == 6
			&& factory.prefix_fold(3) == 0 && from.prefix_fold(3) == 6 && from_flat.prefix_fold(3) == 6 && product.prefix_fold(3) == 6;
	}());
	static_assert([] {
		int const extra = 0;
		auto min = [extra](int x, int y) { return (x < y ? x : y) + extra; };
		nonconst_min mutable_min;
		std::array values{3, 1, 2};
		atl::sparse_table table(std::span{values}, std::ref(mutable_min));
		atl::disjoint_sparse_table disjoint(std::move(values), min);
		nonconst_compare less;
		atl::direct_rmq rmq(values, std::ref(less));
		return table.fold(0, 3) == 1 && disjoint.fold(0, 3) == 1 && rmq.position(0, 3) == 1;
	}());
	static_assert([] {
		nonconst_sum sum;
		nonconst_inverse inverse;
		using Tree = atl::difference_fenwick<int, std::reference_wrapper<nonconst_sum>, std::reference_wrapper<nonconst_inverse>>;
		Tree filled(3, 2, 0, std::ref(sum), std::ref(inverse));
		Tree ranged(std::array{1, 2, 3}, 0, std::ref(sum), std::ref(inverse));
		filled.apply(0, 2, 3);
		ranged.apply(1, 4);
		return filled.at(1) == 5 && ranged.at(1) == 6;
	}());
	static_assert([] {
		nonconst_compare compare;
		atl::li_chao_tree<long long, std::reference_wrapper<nonconst_compare>> array(0, 10, std::ref(compare));
		atl::li_chao_tree<long long, std::reference_wrapper<nonconst_compare>> points(std::vector<long long>{1, 7}, std::ref(compare));
		atl::dynamic_li_chao_tree<long long, std::reference_wrapper<nonconst_compare>> dynamic(0, 10, std::ref(compare));
		array.insert(2, 1);
		points.insert(2, 1);
		dynamic.insert(2, 1);
		return array.query(7) == 15 && points.query(7) == 15 && dynamic.query(7) == 15;
	}());

	enum digit : unsigned { one = 1,
							three = 3 };

	struct qualified_number {
		unsigned value;

		constexpr operator unsigned() && { return value; }

		explicit constexpr operator unsigned() const & { return value; }
	};

	static_assert([] {
		std::array values{qualified_number{3}, qualified_number{1}, qualified_number{2}};
		atl::binary_indexed_tree<unsigned> flat(std::array<std::size_t, 1>{3}, values);
		atl::binary_indexed_tree<unsigned> nested(values);
		atl::wavelet_matrix<unsigned> matrix(values);
		return flat.prefix_fold(3) == 6 && nested.prefix_fold(3) == 6
			&& matrix.at(0) == 3 && matrix.kth(0, 3, 1) == 2 && values[0].value == 3;
	}());

	static_assert([] {
		auto values = std::views::iota(0, 3) | std::views::transform([offset = 1](int x) mutable { return x + offset; });
		static_assert(std::ranges::sized_range<decltype(values)> && !std::ranges::range<decltype(values) const>);
		atl::binary_indexed_tree<int> leaf(values);
		std::array rows{values, values};
		atl::binary_indexed_tree<int, 2> nested(rows);
		return leaf.prefix_fold(3) == 6 && nested.prefix_fold(2, 3) == 12;
	}());

	struct number {
		unsigned value;

		constexpr operator unsigned() const { return value; }
	};

	static_assert([] {
		std::array const digits{digit::three, digit::one};
		atl::wavelet_matrix<unsigned> enum_values(digits);
		if (enum_values.kth(0, 2, 0) != 1) return false;
		std::array const values{number{3}, number{1}, number{2}};
		atl::wavelet_matrix<unsigned> matrix(values);
		auto view = values | std::views::transform([](number x) { return x.value; });
		atl::wavelet_matrix<unsigned> from_view(std::move(view));
		return matrix.kth(0, 3, 1) == 2 && from_view.at(0) == 3;
	}());
	// A prvalue-producing view owns each yielded temporary, even though it does not own the range.
	static_assert([] {
		auto values = std::views::iota(0, 3) | std::views::transform([](int x) { return x + 1; });
		atl::binary_indexed_tree<int> tree(std::move(values));
		return tree.prefix_fold(3) == 6;
	}());
}

static_assert([] {
	using Coordinate = long long;
	using Value = atl::detail::software_int128;
	atl::li_chao_tree<Coordinate, std::less<>, atl::linear_function<Value>> tree(0, 10);
	tree.insert(atl::linear_function<Value>{Value(1) << 90, 3});
	return tree.query(7) == (Value(7) << 90) + 3;
}());

#include <atl/ds/dual_segment_tree.hpp>
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <atl/ds/dynamic_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/ds/segment_tree_beats.hpp>
#include "move_probe.hpp"

TEST_CASE("segment family: explicit range conversions and reference-wrapped callables", "[segment_family][accepted_calls]") {
	enum class number { one = 1,
						two = 2,
						three = 3 };
	std::array const input{number::one, number::two, number::three};
	auto add = [offset = 0](int a, int b) { return a + b + offset; };
	auto mapping = [](int, int value) { return value; };
	using op = std::reference_wrapper<decltype(add)>;
	using map = std::reference_wrapper<decltype(mapping)>;
	atl::segment_tree<int, op> dense(input, 0, std::ref(add));
	atl::lazy_segment_tree<int, int, op, map, op> lazy(input, 0, 0, std::ref(add), std::ref(mapping), std::ref(add));
	atl::dual_segment_tree<int, int, map, op> dual(input, 0, std::ref(mapping), std::ref(add));
	atl::dual_segment_tree<int, int, map, op, atl::commutative_tags_t> commuting(atl::commutative_tags, input, 0, std::ref(mapping), std::ref(add));
	atl::beats_segment_tree<int> beats(input);
	atl::segment_tree_beats<atl::beats::extremes<int>, atl::beats::clamp_add<int>> engine(input);
	auto summaries = input | std::views::transform([](number x) { return atl::beats::extremes<int>::of(int(x)); });
	atl::segment_tree_beats generic(summaries, atl::beats::extremes<int>::identity(), atl::beats::clamp_add<int>::identity(), atl::beats::merge{}, atl::beats::mapping{}, atl::beats::compose{});
	REQUIRE(dense.fold() == 6);
	REQUIRE(lazy.fold() == 6);
	REQUIRE(dual.at(2) == 3);
	REQUIRE(commuting.at(0) == 1);
	REQUIRE(beats.sum(0, 3) == 6);
	REQUIRE(engine.sum(0, 3) == 6);
	REQUIRE(generic.sum(0, 3) == 6);
	auto summary_predicate = [limit = 9](atl::beats::extremes<int> const &value) { return value.sum <= limit; };
	REQUIRE(beats.max_right(0, std::ref(summary_predicate)) == 3);
	REQUIRE(beats.min_left(3, std::cref(summary_predicate)) == 0);
	REQUIRE(engine.max_right(0, std::ref(summary_predicate)) == 3);
	REQUIRE(engine.min_left(3, std::cref(summary_predicate)) == 0);
	auto rule = [value = 2](atl::beats::extremes<int> const &) -> std::optional<atl::beats::clamp_add<int>> { return atl::beats::clamp_add<int>{0, value, value}; };
	engine.apply_derived(0, 3, std::ref(rule));
	REQUIRE(engine.sum(0, 3) == 6);

	atl::dynamic_segment_tree<int, op> sparse(3, 0, std::ref(add));
	auto update = [increment = 1](int x) mutable { return x + increment++; };
	dense.update(0, std::ref(update));
	sparse.update(2, std::ref(update));
	REQUIRE(dense.at(0) == 2);
	REQUIRE(sparse.at(2) == 2);
	auto predicate = [limit = 9](int x) { return x <= limit; };
	REQUIRE(dense.max_right(0, std::ref(predicate)) == 3);
	REQUIRE(dense.min_left(3, std::cref(predicate)) == 0);
	REQUIRE(lazy.max_right(0, std::ref(predicate)) == 3);
	REQUIRE(lazy.min_left(3, std::cref(predicate)) == 0);
	REQUIRE(sparse.max_right(0, std::ref(predicate)) == 3);
	REQUIRE(sparse.min_left(3, std::cref(predicate)) == 0);
	int visited = 0;
	auto visit = [&visited](int) { ++visited; };
	dense.for_each_segment(0, 3, std::ref(visit));
	REQUIRE(visited > 0);
	auto fill = [factor = 2](std::size_t width) { return int(width) * factor; };
	atl::dynamic_lazy_segment_tree sparse_lazy(3, 0, 0, std::ref(add), std::ref(mapping), std::ref(add), std::cref(fill));
	REQUIRE(sparse_lazy.fold() == 6);
	REQUIRE(sparse_lazy.max_right(0, std::ref(predicate)) == 3);
}

TEST_CASE("segment family: views copy lvalue elements while owning ranges and prvalues move", "[segment_family][accepted_calls]") {
	using probe = atl_test::move_probe;
	auto op = [](probe const &a, probe const &b) { return probe(a.v + b.v); };
	auto mapping = [](int, probe const &a) { return probe(a.v); };
	auto compose = std::plus<>{};
	std::vector<probe> input{probe(probe::sentinel)};
	auto view = input | std::views::transform([](probe &x) -> probe && { return std::move(x); });
	probe::reset();
	atl::segment_tree dense(std::move(view), probe(0), op);
	REQUIRE(input[0].v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	REQUIRE(dense.at(0).v == probe::sentinel);
	probe::reset();
	atl::lazy_segment_tree lazy(input | std::views::all, probe(0), 0, op, mapping, compose);
	REQUIRE(input[0].v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	probe::reset();
	atl::dual_segment_tree dual(input | std::views::all, 0, mapping, compose);
	REQUIRE(input[0].v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	probe::reset();
	atl::segment_tree owned(std::move(input), probe(0), op);
	REQUIRE(probe::sentinel_copies == 0);
	REQUIRE(owned.at(0).v == probe::sentinel);
	auto prvalues = std::views::iota(0, 1) | std::views::transform([](int) { return probe(probe::sentinel); });
	probe::reset();
	atl::dual_segment_tree temporary(prvalues, 0, mapping, compose);
	REQUIRE(probe::sentinel_copies == 0);
	REQUIRE(temporary.at(0).v == probe::sentinel);
}

#include <atl/experimental/ds/segment_tree_beats_variants.hpp>

TEMPLATE_TEST_CASE("segment family: experimental ranges accept explicit elements", "[segment_family][accepted_calls]",
				   atl::experimental::beats_segment_tree_by_recursive_compact_nodes<int>,
				   atl::experimental::beats_segment_tree_by_explicit_stack<int>,
				   atl::experimental::beats_segment_tree_by_fixed_clamp_add<int>,
				   atl::experimental::beats_segment_tree_by_three_adds<int>,
				   atl::experimental::beats_segment_tree_by_optional_mapping<int>,
				   atl::experimental::beats_segment_tree_by_unconditional_push<int>,
				   atl::experimental::beats_segment_tree_by_no_collapse<int>,
				   (atl::experimental::beats_segment_tree_by_tagless_layout<int, int, atl::experimental::beats_layout::aos_aligned>),
				   (atl::experimental::beats_segment_tree_by_tagless_layout<int, int, atl::experimental::beats_layout::soa>)) {
	enum class value { one = 1,
					   two = 2,
					   three = 3 };
	std::array const input{value::one, value::two, value::three};
	TestType library(input);
	REQUIRE(library.sum(0, 3) == 6);
	library.add(0, 3, 1);
	REQUIRE(library.sum(0, 3) == 9);
}

namespace segment_member_calls {
	struct value {
		int sum = 0;

		constexpr value merge(value const &other) const { return {sum + other.sum}; }

		constexpr value increment() const { return {sum + 1}; }

		constexpr bool within_limit() const { return sum <= 9; }

		constexpr int inspect() const { return sum; }
	};
}

TEST_CASE("segment family: member pointers follow the callable concepts", "[segment_family][accepted_calls]") {
	using segment_member_calls::value;
	atl::segment_tree library(std::array{value{1}, value{2}, value{3}}, value{}, &value::merge);
	library.update(0, &value::increment);
	REQUIRE(library.fold().sum == 7);
	REQUIRE(library.max_right(0, &value::within_limit) == 3);
	REQUIRE(library.min_left(3, &value::within_limit) == 0);
	library.for_each_segment(0, 3, &value::inspect);
	atl::dynamic_segment_tree sparse(3, value{}, &value::merge);
	sparse.update(2, &value::increment);
	REQUIRE(sparse.fold().sum == 1);
	REQUIRE(sparse.max_right(0, &value::within_limit) == 3);
	REQUIRE(sparse.min_left(3, &value::within_limit) == 0);
}
