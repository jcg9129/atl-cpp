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
		int limit = 3;
		auto pred = [limit](int x) mutable { return x <= limit; };
		auto factory = atl::make_binary_indexed_tree(0, 3);
		auto from = atl::make_binary_indexed_tree_from(values);
		auto from_flat = atl::make_binary_indexed_tree_from(values, 3);
		auto product = atl::make_binary_indexed_tree_from<std::multiplies<>>(1, values);
		return sized.prefix_fold(2) == 6 && nested.max_right(std::ref(pred)) == 2 && flat.prefix_fold(3) == 6
			&& factory.prefix_fold(3) == 0 && from.prefix_fold(3) == 6 && from_flat.prefix_fold(3) == 6 && product.prefix_fold(3) == 6;
	}());
	static_assert([] {
		int extra = 0;
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
	// GCC can diagnose the possible vtable read in std::invoke on smaller objects, even for nonvirtual members.
	struct alignas(void *) value {
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

#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_map.hpp>
#include <atl/ds/splay_sequence.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/ds/treap_sequence.hpp>
#include "move_probe.hpp"

namespace treap_splay_accepted {
	enum class element : int { one = 1,
							   two = 2 };

	struct converted_element {
		explicit operator element() const { return element::one; }
	};

	struct mutable_sum {
		int operator()(int a, int b) { return a + b; }
	};

	struct mutable_order {
		bool operator()(int a, int b) { return a < b; }
	};

	using copy_view = decltype(std::declval<std::vector<atl_test::move_probe> &>() | std::views::transform(std::identity{}));

	template<typename Sequence>
		requires std::constructible_from<Sequence, copy_view> && requires(Sequence &sequence) {
			sequence.at(0);
			sequence.for_each([](auto const &) {});
		}
	void check_view_copy() {
		std::vector<atl_test::move_probe> source{atl_test::move_probe::sentinel};
		auto view = source | std::views::transform(std::identity{});
		atl_test::move_probe::reset();
		// The view owns no elements, including when the constructor receives it as an rvalue.
		Sequence values(std::move(view));
		REQUIRE(source.front().v == atl_test::move_probe::sentinel);
		REQUIRE(values.at(0).v == atl_test::move_probe::sentinel);
		REQUIRE(atl_test::move_probe::sentinel_copies == 1);
	}
}

TEST_CASE("treap and splay: accepted callable and range forms", "[accepted_calls]") {
	using namespace treap_splay_accepted;
	mutable_sum sum;
	mutable_order order;
	auto captured_sum = [offset = 0](int a, int b) { return a + b + offset; };
	auto captured_order = [offset = 0](int a, int b) { return a + offset < b + offset; };
	std::array entries{std::pair{1, 3}, std::pair{2, 5}};
	atl::treap_map<int, int, decltype(captured_order), decltype(captured_sum)> treap(atl::sorted_unique, entries, 0, captured_sum, captured_order);
	atl::splay_map<int, int, decltype(captured_order), decltype(captured_sum)> splay(atl::sorted_unique, entries, 0, captured_sum, captured_order);
	REQUIRE(treap.fold() == 8);
	REQUIRE(splay.fold() == 8);
	atl::treap_map<int, int, std::reference_wrapper<mutable_order>, std::reference_wrapper<mutable_sum>> wrapped_treap(0, std::ref(sum), std::ref(order));
	atl::splay_map<int, int, std::reference_wrapper<mutable_order>, std::reference_wrapper<mutable_sum>> wrapped_splay(0, std::ref(sum), std::ref(order));
	wrapped_treap.insert(1, 5);
	wrapped_splay.insert(1, 5);
	REQUIRE(wrapped_treap.fold() == wrapped_splay.fold());
	auto update_value = [increment = 2](int value) mutable { return value + increment; };
	REQUIRE(treap.update(1, std::ref(update_value)) == 5);
	REQUIRE(splay.update(1, update_value) == 5);
	auto const const_update = [](int value) { return value + 1; };
	REQUIRE(treap.update(1, const_update) == 6);
	REQUIRE(splay.update(1, const_update) == 6);
	std::array initial{1, 2, 3};
	atl::treap_sequence<int, decltype(captured_sum)> sequence(initial, 0, captured_sum);
	atl::splay_sequence<int, decltype(captured_sum)> adjusted(initial, 0, captured_sum);
	auto predicate = [limit = 3](int value) mutable { return value <= limit; };
	REQUIRE(sequence.max_right(0, std::ref(predicate)) == 2);
	REQUIRE(adjusted.max_right(0, predicate) == 2);
	int total = 0;
	auto visitor = [&total](int value) mutable { total += value; };
	sequence.for_each(std::ref(visitor));
	adjusted.for_each(visitor);
	REQUIRE(total == 12);
	std::array<converted_element, 2> explicit_values{};
	atl::treap_sequence<element> enums(explicit_values);
	atl::splay_sequence<element> splay_enums(explicit_values);
	REQUIRE(enums.at(0) == element::one);
	REQUIRE(splay_enums.at(1) == element::one);
	check_view_copy<atl::treap_sequence<atl_test::move_probe>>();
	check_view_copy<atl::splay_sequence<atl_test::move_probe>>();

	atl::sized_splay_links links;
	std::array ids{links.add_node(), links.add_node()};
	std::array<element, 2> enum_ids{element::one, element::two};
	auto root = atl::splay_engine::build_balanced(links, enum_ids);
	REQUIRE(root == ids[1]);
	auto probe = [](atl::uint32) mutable { return atl::splay_step::left; };
	REQUIRE(atl::splay_engine::descend(links, root, std::ref(probe)) == ids[0]);
	atl::splay_engine::for_each(links, ids[0], [](atl::uint32) {});
}

namespace treap_splay_accepted {
	struct fixed_order {
		fixed_order() = default;
		fixed_order(fixed_order const &) = delete;

		bool operator()(int a, int b) const { return a < b; }
	};

	struct const_sum {
		int operator()(int const &a, int const &b) const { return a + b; }

		void operator()(int &, int &) const = delete;
		void operator()(int const &, int const &) = delete;
	};

	struct member_value {
		int number = 0;

		member_value combine(member_value const &other) const { return {number + other.number}; }

		member_value increment() const { return {number + 1}; }
	};
}

TEST_CASE("treap and splay: invocation matches const-operand contracts", "[accepted_calls]") {
	using namespace treap_splay_accepted;
	atl::treap_map<int, int, fixed_order> treap_order;
	atl::splay_map<int, int, fixed_order> splay_order;
	treap_order.insert(1, 2);
	splay_order.insert(1, 2);
	REQUIRE(treap_order.at(1) == splay_order.at(1));
	atl::treap_map<int, int, std::ranges::less, const_sum> treap(0, const_sum{});
	atl::splay_map<int, int, std::ranges::less, const_sum> splay(0, const_sum{});
	treap.insert(1, 2);
	splay.insert(1, 2);
	treap.insert(2, 3);
	splay.insert(2, 3);
	REQUIRE(treap.fold() == 5);
	REQUIRE(splay.fold(0, 3) == 5);
	atl::treap_sequence<int, const_sum> sequence(std::array{1, 2, 3}, 0, const_sum{});
	atl::splay_sequence<int, const_sum> adjusted(std::array{1, 2, 3}, 0, const_sum{});
	REQUIRE(sequence.fold(0, 3) == 6);
	REQUIRE(adjusted.fold(0, 3) == 6);
	auto predicate = [](int value) { return value <= 3; };
	REQUIRE(sequence.max_right(0, predicate) == 2);
	REQUIRE(adjusted.max_right(0, predicate) == 2);
	auto const member_sum = &member_value::combine;
	atl::treap_map<int, member_value, std::ranges::less, decltype(member_sum)> member_treap(member_value{}, member_sum);
	atl::splay_map<int, member_value, std::ranges::less, decltype(member_sum)> member_splay(member_value{}, member_sum);
	REQUIRE(member_treap.update(3, &member_value::increment).number == 1);
	REQUIRE(member_splay.update(3, &member_value::increment).number == 1);
	REQUIRE(member_treap.fold().number == 1);
	REQUIRE(member_splay.fold().number == 1);
}

TEST_CASE("treap and splay: captured tag algebras construct directly", "[accepted_calls]") {
	struct sum_width {
		int sum = 0, width = 0;
	};

	auto fold_values = [zero = 0](sum_width const &a, sum_width const &b) { return sum_width{a.sum + b.sum + zero, a.width + b.width}; };
	auto map_values = [scale = 1](int tag, sum_width const &value) { return sum_width{value.sum + scale * tag * value.width, value.width}; };
	auto compose_tags = [zero = 0](int a, int b) { return a + b + zero; };
	using treap = atl::treap_sequence<sum_width, decltype(fold_values), int, decltype(map_values), decltype(compose_tags)>;
	using splay = atl::splay_sequence<sum_width, decltype(fold_values), int, decltype(map_values), decltype(compose_tags)>;
	treap sequence(std::array{sum_width{2, 1}, sum_width{3, 1}}, sum_width{}, 0, fold_values, map_values, compose_tags);
	splay adjusted(std::array{sum_width{2, 1}, sum_width{3, 1}}, sum_width{}, 0, fold_values, map_values, compose_tags);
	sequence.apply(0, 2, 4);
	adjusted.apply(0, 2, 4);
	REQUIRE(sequence.fold().sum == 13);
	REQUIRE(adjusted.fold().sum == 13);
	REQUIRE(sequence.at(0).sum == 6);
	REQUIRE(adjusted.at(0).sum == 6);
}

#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/ds/top_tree.hpp>
#include "move_probe.hpp"

namespace dynamic_forest_accepted_calls {
	struct mutable_add {
		constexpr long long operator()(long long a, long long b) { return a + b; }
	};

	struct mutable_negate {
		constexpr long long operator()(long long a) { return -a; }
	};

	static_assert([] {
		mutable_add combine;
		mutable_negate invert;
		std::vector<long long> values{1, 2, 3};
		auto view = values | std::views::all;
		atl::link_cut_tree path(std::move(view), 0LL, std::ref(combine));
		atl::subtree_link_cut_tree subtree(values, 0LL, std::ref(combine), std::ref(invert));
		atl::euler_tour_tree tour(values | std::views::all, 0LL, std::ref(combine));
		atl::sum_top_tree<long long> clusters(values | std::views::all);
		std::array<atl::uint32, 3> const parents{atl::graph::none, 0, 0};
		path.assign_parents(parents);
		subtree.assign_parents(parents);
		tour.assign_parents(parents);
		clusters.assign_parents(parents);
		long long increment = 4;
		auto update = [increment](long long value) mutable { return value + increment; };
		path.update(1, std::ref(update));
		subtree.update(1, std::ref(update));
		tour.update(1, std::ref(update));
		clusters.update(1, std::ref(update));
		return path.fold(1, 2) == 10 && subtree.subtree_fold(0) == 10 && tour.component_fold(0) == 10
			&& clusters.component_sum(0) == 10 && values == std::vector<long long>{1, 2, 3};
	}());

	struct probe_add {
		atl_test::move_probe operator()(atl_test::move_probe const &a, atl_test::move_probe const &b) const { return {a.v + b.v}; }
	};

	template<typename Tree>
		requires std::constructible_from<Tree, std::ranges::ref_view<std::vector<atl_test::move_probe>>, atl_test::move_probe, probe_add>
	void check_borrowed_values(int expected_copies) {
		std::vector<atl_test::move_probe> values{atl_test::move_probe::sentinel};
		atl_test::move_probe::reset();
		Tree tree(values | std::views::all, atl_test::move_probe{}, probe_add{});
		REQUIRE(values[0].v == atl_test::move_probe::sentinel);
		REQUIRE(atl_test::move_probe::sentinel_copies == expected_copies);
		REQUIRE(tree.at(0).v == atl_test::move_probe::sentinel);
	}
}

TEST_CASE("dynamic forest range constructors preserve borrowed values", "[accepted_calls]") {
	using namespace dynamic_forest_accepted_calls;
	check_borrowed_values<atl::link_cut_tree<atl_test::move_probe, probe_add>>(3);
	check_borrowed_values<atl::euler_tour_tree<atl_test::move_probe, probe_add>>(2);
}

namespace dynamic_forest_accepted_calls {
	static_assert([] {
		auto combine = [identity = 0LL](long long a, long long b) { return a + b + identity; };
		auto invert = [identity = 0LL](long long a) { return identity - a; };
		std::vector<long long> values{1, 2};
		atl::link_cut_tree a(values, 0LL, combine);
		atl::link_cut_tree b(2, 0LL, combine);
		atl::subtree_link_cut_tree c(values, 0LL, combine, invert);
		atl::subtree_link_cut_tree d(2, 0LL, combine, invert);
		atl::euler_tour_tree e(values, 0LL, combine);
		atl::euler_tour_tree f(2, 0LL, combine);
		a.link(0, 1);
		b.link(0, 1);
		c.link(0, 1);
		d.link(0, 1);
		e.link(0, 1);
		f.link(0, 1);
		return a.fold(0, 1) == 3 && b.fold(0, 1) == 0 && c.subtree_fold(1) == 3 && d.subtree_fold(1) == 0
			&& e.component_fold(0) == 3 && f.component_fold(0) == 0;
	}());

	struct sum_width {
		long long sum = 0, width = 0;
	};

	static_assert([] {
		auto combine = [zero = 0LL](sum_width a, sum_width b) { return sum_width{a.sum + b.sum + zero, a.width + b.width}; };
		auto mapping = [zero = 0LL](long long tag, sum_width value) { return sum_width{value.sum + tag * value.width + zero, value.width}; };
		auto compose = [zero = 0LL](long long a, long long b) { return a + b + zero; };
		std::vector<sum_width> values{{1, 1}, {2, 1}};
		atl::link_cut_tree path(values, sum_width{}, 0LL, combine, mapping, compose);
		atl::link_cut_tree empty_path(2, sum_width{}, 0LL, combine, mapping, compose);
		atl::euler_tour_tree tour(values, sum_width{}, 0LL, combine, mapping, compose);
		atl::euler_tour_tree empty_tour(2, sum_width{}, 0LL, combine, mapping, compose);
		path.link(0, 1);
		empty_path.link(0, 1);
		tour.link(0, 1);
		empty_tour.link(0, 1);
		path.apply(0, 1, 2);
		tour.component_apply(0, 2);
		return path.fold(0, 1).sum == 7 && tour.component_fold(0).sum == 7
			&& empty_path.fold(0, 1).sum == 0 && empty_tour.component_fold(0).sum == 0;
	}());
}

#include <atl/experimental/ds/euler_tour_tree_variants.hpp>

namespace dynamic_forest_accepted_calls {
	static_assert([] {
		auto combine = [zero = 0LL](sum_width a, sum_width b) { return sum_width{a.sum + b.sum + zero, a.width + b.width}; };
		auto mapping = [zero = 0LL](long long tag, sum_width value) { return sum_width{value.sum + tag * value.width + zero, value.width}; };
		auto compose = [zero = 0LL](long long a, long long b) { return a + b + zero; };
		std::vector<sum_width> values{{1, 1}, {2, 1}};
		atl::experimental::euler_tour_tree_by_splay a(values, sum_width{}, 0LL, combine, mapping, compose);
		atl::experimental::euler_tour_tree_by_splay b(2, sum_width{}, 0LL, combine, mapping, compose);
		a.link(0, 1);
		b.link(0, 1);
		a.component_apply(0, 2);
		auto sum = [zero = 0LL](long long x, long long y) { return x + y + zero; };
		atl::experimental::euler_tour_tree_by_splay c(std::vector<long long>{1, 2}, 0LL, sum);
		atl::experimental::euler_tour_tree_by_splay d(2, 0LL, sum);
		c.link(0, 1);
		d.link(0, 1);
		return a.component_fold(0).sum == 7 && b.component_fold(0).sum == 0 && c.component_fold(0) == 3 && d.component_fold(0) == 0;
	}());
}

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/ds/persistent_heap.hpp>
#include <atl/ds/persistent_lazy_segment_tree.hpp>
#include <atl/ds/persistent_list.hpp>
#include <atl/ds/persistent_segment_tree.hpp>
#include <atl/ds/persistent_treap_sequence.hpp>
#include <atl/ds/persistent_vector.hpp>
#include <span>
#include "move_probe.hpp"

TEST_CASE("publication persistent callables accept captures and reference wrappers", "[accepted_calls]") {
	int offset = 0; // not a constant expression, so the lambdas must capture it
	auto plus = [offset](int a, int b) { return a + b + offset; };
	auto mapping = [offset](int f, int x) { return x + f + offset; };
	auto less = [offset](int a, int b) mutable { return a + offset < b; };
	atl::persistent_heap<int, decltype(less)> heap(less);
	heap = heap.push(2);
	REQUIRE(heap.top() == 2);
	atl::persistent_heap_forest<int, decltype(std::ref(less))> forest(std::ref(less));
	REQUIRE(forest.top(forest.push(forest.empty_heap, 3)) == 3);
	atl::meldable_heap_forest<int, decltype(std::ref(less))> meldable(1, std::ref(less));
	meldable.emplace(0, 4);
	REQUIRE(meldable.pop(0) == 4);
	std::array values{1, 2, 3};
	atl::persistent_segment_tree tree(values, 0, std::ref(plus));
	auto update = [delta = 1](int x) mutable { return x + delta; };
	REQUIRE(tree.update(0, std::ref(update)).at(0) == 2);
	auto choose = [go_left = true](int, int) mutable { return go_left; };
	REQUIRE(decltype(tree)::descend(tree, tree, std::ref(choose)) == 0);
	atl::persistent_lazy_segment_tree lazy(values, 0, 0, std::ref(plus), std::ref(mapping), std::ref(plus));
	REQUIRE(lazy.update(0, update).at(0) == 2);
	atl::persistent_treap_sequence sequence(values, 0, plus);
	REQUIRE(sequence.compacted().fold() == 6);
	atl::persistent_treap_sequence tagged(values, 0, 0, plus, mapping, plus);
	REQUIRE(tagged.compacted().at(2) == 3);
}

TEST_CASE("persistent treap range forwarding preserves views and moves owned values", "[accepted_calls][persistent_treap_sequence]") {
	using probe = atl_test::move_probe;
	std::vector<probe> values{probe(probe::sentinel), probe(2)};
	probe::reset();
	atl::persistent_treap_sequence copied{std::span(values)};
	REQUIRE(values[0].v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	REQUIRE(copied.at(0).v == probe::sentinel);
	probe::reset();
	auto view = values | std::views::filter([](probe const &) { return true; });
	// A sized view over caller storage remains a copy source even as an rvalue.
	auto sized_view = std::views::all(values);
	atl::persistent_treap_sequence copied_view(std::move(sized_view));
	REQUIRE(values[0].v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	REQUIRE(copied_view.at(0).v == probe::sentinel);
	probe::reset();
	atl::persistent_treap_sequence moved(std::move(values));
	REQUIRE(probe::sentinel_copies == 0);
	REQUIRE(moved.at(0).v == probe::sentinel);
	(void)view;
}

TEST_CASE("persistent sized constructors accept enum values and yielded prvalues", "[accepted_calls]") {
	enum value { first = 1,
				 second = 2 };

	std::array values{first, second};
	auto prvalues = values | std::views::transform([delta = 0](value x) { return int(x) + delta; });
	atl::persistent_list<int> list(values);
	atl::persistent_vector<int> vector{std::span(values)};
	atl::semi_persistent_vector<int> partial(prvalues);
	atl::persistent_segment_tree<int, std::plus<>> tree(prvalues, 0);
	auto plus = std::plus<>{};
	auto identity_map = [](int, int x) { return x; };
	atl::persistent_lazy_segment_tree lazy(prvalues, 0, 0, plus, identity_map, plus);
	atl::persistent_treap_sequence<int> sequence(prvalues);
	REQUIRE(list[1] == 2);
	REQUIRE(vector[0] == 1);
	REQUIRE(partial[1] == 2);
	REQUIRE(tree.fold() == 3);
	REQUIRE(lazy.fold() == 3);
	REQUIRE(sequence.at(1) == 2);
}

namespace publication_review_calls {
	struct rvalue_proxy {
		int value;

		operator int() && { return value; }
	};

	struct copy_sum {
		int bias = 0;

		copy_sum() = default;

		copy_sum(copy_sum const &) = default;

		copy_sum(copy_sum &&) = delete;

		copy_sum &operator=(copy_sum const &) = default;

		constexpr int operator()(int a, int b) const { return a + b + bias; }
	};

	struct copy_identity {
		copy_identity() = default;

		copy_identity(copy_identity const &) = default;

		copy_identity(copy_identity &&) = delete;

		int operator()(int value) const { return value; }
	};

	struct assign_sum {
		assign_sum() = default;

		assign_sum(assign_sum const &) = delete;

		assign_sum(assign_sum &&) = default;

		assign_sum &operator=(assign_sum const &) = default;

		int operator()(int a, int b) const { return a + b; }
	};

	struct copy_accumulator {
		int value = 0;

		constexpr copy_accumulator() = default;

		constexpr explicit copy_accumulator(int x) : value(x) {}

		constexpr copy_accumulator(copy_accumulator const &) = default;

		copy_accumulator(copy_accumulator &&) = delete;

		constexpr copy_accumulator &operator=(copy_accumulator const &) = default;
	};

	struct accumulator_store : atl::sized_splay_links {
		std::array<copy_accumulator, 4> values{};

		constexpr copy_accumulator const &value(atl::uint32 x) const { return values[x]; }

		constexpr copy_accumulator const &aggregate(atl::uint32 x) const { return values[x]; }
	};

	static_assert([] {
		accumulator_store store;
		auto root = store.add_node();
		store.values[root] = copy_accumulator(3);
		copy_accumulator initial;
		auto combine = [](copy_accumulator const &a, copy_accumulator const &b) { return copy_accumulator(a.value + b.value); };
		auto result = atl::splay_engine::max_right(store, root, initial, combine, [](copy_accumulator const &a) { return a.value <= 3; });
		return result.length == 1;
	}());

	struct moving_probe_max {
		static inline int calls = 0;

		atl_test::move_probe operator()(atl_test::move_probe a, atl_test::move_probe const &b) const {
			++calls;
			if (a.v >= b.v) return a;
			return b;
		}
	};

	struct longer_probe {
		static inline int calls = 0;

		atl_test::move_probe const &operator()(atl_test::move_probe const &a, atl_test::move_probe const &b) const {
			++calls;
			return a.v >= b.v ? a : b;
		}
	};
}

TEST_CASE("sequence builders accept move-only prvalues and rvalue-only proxies", "[accepted_calls]") {
	using namespace publication_review_calls;
	auto pointers = std::views::iota(1, 4) | std::views::transform([](int x) { return std::make_unique<int>(x); });
	atl::treap_sequence<std::shared_ptr<int>> treap(pointers);
	atl::splay_sequence<std::shared_ptr<int>> splay(pointers);
	atl::persistent_treap_sequence<std::shared_ptr<int>> persistent(pointers);
	atl::persistent_vector<std::shared_ptr<int>> vector(pointers);
	atl::persistent_list<std::shared_ptr<int>> list(pointers);
	REQUIRE(*treap.at(1) == 2);
	REQUIRE(*splay.at(1) == 2);
	REQUIRE(*persistent.at(1) == 2);
	REQUIRE(*vector[1] == 2);
	REQUIRE(*list[1] == 2);
	auto proxies = std::views::iota(1, 4) | std::views::transform([](int x) { return rvalue_proxy{x}; });
	atl::treap_sequence<int> proxy_treap(proxies);
	atl::splay_sequence<int> proxy_splay(proxies);
	atl::persistent_treap_sequence<int> proxy_persistent(proxies);
	atl::persistent_vector<int> proxy_vector(proxies);
	atl::persistent_list<int> proxy_list(proxies);
	atl::link_cut_tree<int, std::plus<>> path(proxies, 0);
	atl::euler_tour_tree<int, std::plus<>> tour(proxies, 0);
	atl::sum_top_tree<int> top(proxies);
	REQUIRE(proxy_treap.at(1) == 2);
	REQUIRE(proxy_splay.at(1) == 2);
	REQUIRE(proxy_persistent.at(1) == 2);
	REQUIRE(proxy_vector[1] == 2);
	REQUIRE(proxy_list[1] == 2);
	REQUIRE(path.at(1) == 2);
	REQUIRE(tour.at(1) == 2);
	REQUIRE(top.at(1) == 2);
}

TEST_CASE("algebra pools accept copy construction with deleted moves", "[accepted_calls]") {
	using namespace publication_review_calls;
	copy_sum combine;
	std::array const values{1, 2, 3};
	atl::treap_sequence<int, copy_sum> treap(values, 0, combine);
	atl::splay_sequence<int, copy_sum> splay(values, 0, combine);
	atl::treap_map<int, int, std::ranges::less, copy_sum> map(0, combine);
	atl::splay_map<int, int, std::ranges::less, copy_sum> adjusted(0, combine);
	map.insert(1, 6);
	adjusted.insert(1, 6);
	atl::persistent_treap_sequence<int, copy_sum> persistent(values, 0, combine);
	atl::persistent_treap_sequence<int, copy_sum, int, copy_sum, copy_sum> tagged(values, 0, 0, combine, combine, combine);
	atl::persistent_segment_tree<int, copy_sum> segment(values, 0, combine);
	atl::persistent_lazy_segment_tree<int, int, copy_sum, copy_sum, copy_sum> lazy(values, 0, 0, combine, combine, combine);
	atl::link_cut_tree<int, copy_sum> path(values, 0, combine);
	atl::euler_tour_tree<int, copy_sum> tour(values, 0, combine);
	atl::experimental::euler_tour_tree_by_splay<int, copy_sum> splay_tour(values, 0, combine);
	atl::top_tree<int, int, int, copy_sum, copy_sum, copy_sum, copy_identity, atl::commutative_fold_t> top(values, 0, 0, combine, combine, combine, copy_identity{});
	top.link(0, 1);
	REQUIRE(top.fold(0, 1) == 3);
	REQUIRE(treap.fold() == 6);
	REQUIRE(splay.fold() == 6);
	REQUIRE(map.fold() == 6);
	REQUIRE(adjusted.fold() == 6);
	REQUIRE(persistent.fold() == 6);
	REQUIRE(tagged.fold() == 6);
	REQUIRE(segment.fold() == 6);
	REQUIRE(lazy.fold() == 6);
	REQUIRE(path.at(1) == 2);
	REQUIRE(tour.at(1) == 2);
	REQUIRE(splay_tour.at(1) == 2);
	atl::persistent_treap_sequence<int, assign_sum> assignable(values, 0, assign_sum{});
	REQUIRE(assignable.fold() == 6);
	atl::persistent_treap_sequence<int, assign_sum, int, assign_sum, assign_sum> assignable_tags(values, 0, 0, assign_sum{}, assign_sum{}, assign_sum{});
	REQUIRE(assignable_tags.fold() == 6);
}

TEST_CASE("treap algebra reference results avoid intermediate payload copies", "[accepted_calls][move_probe]") {
	using probe = atl_test::move_probe;
	using publication_review_calls::longer_probe;
	atl::treap_sequence<probe, longer_probe> sequence(std::vector<probe>{}, probe{}, longer_probe{});
	atl::treap_map<int, probe, std::ranges::less, longer_probe> map(probe{}, longer_probe{});
	map.reserve(32);
	for (int i = 0; i < 16; ++i) {
		probe::reset();
		sequence.insert(std::size_t(i), probe(probe::sentinel));
		// The node's initial aggregate and all pulls assign directly from references.
		REQUIRE(probe::sentinel_copy_constructions == 0);
		probe::reset();
		map.insert(i, probe(probe::sentinel));
		REQUIRE(probe::sentinel_copy_constructions == 0);
	}
	probe::reset();
	REQUIRE(sequence.fold().v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	probe::reset();
	REQUIRE(map.fold().v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	probe::reset();
	longer_probe::calls = 0;
	REQUIRE(map.fold(0, 16).v == probe::sentinel);
	// Each helper returns one payload copy for its pair of operations; nested
	// reference results are used directly, as in the algebra expression.
	REQUIRE(longer_probe::calls > 0);
	REQUIRE(probe::sentinel_copy_constructions == longer_probe::calls / 2);
}

TEST_CASE("treap nested operations move their intermediate payloads", "[accepted_calls][move_probe]") {
	using probe = atl_test::move_probe;
	using publication_review_calls::moving_probe_max;
	atl::treap_sequence<probe, moving_probe_max> sequence(std::vector<probe>{}, probe{}, moving_probe_max{});
	atl::treap_map<int, probe, std::ranges::less, moving_probe_max> map(probe{}, moving_probe_max{});
	for (int i = 0; i < 16; ++i) {
		probe::reset();
		moving_probe_max::calls = 0;
		sequence.insert(std::size_t(i), probe(probe::sentinel));
		// Each pull copies its lvalue input once; the outer op moves the intermediate.
		// The new sequence node also copy-assigns its initial aggregate once.
		REQUIRE(probe::sentinel_copies == moving_probe_max::calls / 2 + 1);
		probe::reset();
		moving_probe_max::calls = 0;
		map.insert(i, probe(probe::sentinel));
		REQUIRE(probe::sentinel_copies == moving_probe_max::calls / 2);
	}
	probe::reset();
	REQUIRE(sequence.fold().v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
	probe::reset();
	REQUIRE(map.fold().v == probe::sentinel);
	REQUIRE(probe::sentinel_copies == 1);
}

TEST_CASE("sequence constructors accept const operation types", "[accepted_calls]") {
	using op = std::plus<int> const;
	std::array const values{1, 2, 3};
	atl::treap_sequence<int, op> treap_count(3, 0, std::plus<int>{});
	atl::treap_sequence<int, op> treap_range(values, 0, std::plus<int>{});
	atl::splay_sequence<int, op> splay_count(3, 0, std::plus<int>{});
	atl::splay_sequence<int, op> splay_range(values, 0, std::plus<int>{});
	atl::treap_sequence<int, op, int, op, op> treap_tag_count(3, 0, 0, std::plus<int>{}, std::plus<int>{}, std::plus<int>{});
	atl::treap_sequence<int, op, int, op, op> treap_tag_range(values, 0, 0, std::plus<int>{}, std::plus<int>{}, std::plus<int>{});
	atl::splay_sequence<int, op, int, op, op> splay_tag_count(3, 0, 0, std::plus<int>{}, std::plus<int>{}, std::plus<int>{});
	atl::splay_sequence<int, op, int, op, op> splay_tag_range(values, 0, 0, std::plus<int>{}, std::plus<int>{}, std::plus<int>{});
	atl::persistent_treap_sequence<int, op> persistent(values, 0, std::plus<int>{});
	atl::persistent_treap_sequence<int, op, int, op, op> persistent_tags(values, 0, 0, std::plus<int>{}, std::plus<int>{}, std::plus<int>{});
	REQUIRE(treap_count.fold() == 0);
	REQUIRE(treap_range.fold() == 6);
	REQUIRE(splay_count.fold() == 0);
	REQUIRE(splay_range.fold() == 6);
	REQUIRE(treap_tag_count.fold() == 0);
	REQUIRE(treap_tag_range.fold() == 6);
	REQUIRE(splay_tag_count.fold() == 0);
	REQUIRE(splay_tag_range.fold() == 6);
	REQUIRE(persistent.fold() == 6);
	REQUIRE(persistent_tags.fold() == 6);
}
