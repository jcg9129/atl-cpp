#include <atl/ds/coordinate_compression.hpp>
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/hash_table.hpp>
#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/ds/detail/key_value_entry.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/detail/radix.hpp>
#include <atl/ds/dsu.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/ds/flat_hash_set.hpp>
#include <atl/ds/hierarchical_bitset.hpp>
#include <atl/ds/persistent_array.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/static_vector.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
	struct ordering {
		constexpr ordering() = default;
		constexpr ordering(ordering &&) = default;
		ordering(ordering const &) = delete;

		constexpr bool operator()(int a, int b) const { return a < b; }
	};

	constexpr bool compression_check() {
		auto result = atl::compress(std::array{3, 1, 3}, std::identity{}, ordering{});
		return result.universe.rank(3) == 1 && result.universe.rank_upper(3) == 2
			&& result.universe.contains(1) && !result.universe.contains(2)
			&& result.universe.ranks(std::array{3, 0, 1}) == std::vector<atl::uint32>{1, 0, 0};
	}

	constexpr bool constructors_check() {
		atl::weighted_dsu<int> defaults(3);
		atl::weighted_dsu<int> identity(3, 0);
		atl::weighted_dsu<int> composition(3, 0, std::plus<>{});
		atl::rollback_weighted_dsu<int> rollback_defaults(3);
		atl::rollback_weighted_dsu<int> rollback_identity(3, 0);
		atl::rollback_weighted_dsu<int> rollback_composition(3, 0, std::plus<>{});
		return defaults.elements() == 3 && identity.elements() == 3 && composition.elements() == 3
			&& rollback_defaults.elements() == 3 && rollback_identity.elements() == 3 && rollback_composition.elements() == 3;
	}

	static_assert(compression_check());
	static_assert(constructors_check());
}

TEST_CASE("approved interfaces remain constexpr", "[constexpr]") {
	STATIC_REQUIRE(compression_check());
	STATIC_REQUIRE(constructors_check());
}

// Allocations must be released within each constant evaluation.
static_assert([] {
	atl::dsu d(4);
	d.unite(0, 1);
	d.unite(2, 3);
	return d.count() == 2 && d.same(0, 1) && !d.same(1, 2) && d.size(0) == 2;
}());
static_assert([] {
	atl::rollback_dsu d(3);
	d.unite(0, 1);
	auto const t = d.time();
	d.unite(1, 2);
	d.rollback(t);
	return d.same(0, 1) && !d.same(0, 2);
}());
static_assert([] {
	atl::rollback_dsu d(4);
	d.unite(0, 1);
	d.unite(1, 2);
	bool const is_merged = d.count() == 2 && d.size(2) == 3 && d.find(2) == d.find(0);
	d.undo();
	return is_merged && d.count() == 3 && d.size(0) == 2 && d.find(2) == 2 && d.time() == 1;
}());

namespace {
	template<atl::signed_integral Potential>
	constexpr bool weighted_operations_check() {
		atl::weighted_dsu<Potential> d(4);
		bool const has_merged = d.unite(0, 1, 5) == atl::unite_outcome::merged && d.unite(1, 2, 3) == atl::unite_outcome::merged;
		bool const is_checked = d.unite(0, 2, 8) == atl::unite_outcome::consistent && d.unite(2, 0, 8) == atl::unite_outcome::contradicts;
		return has_merged && is_checked && d.difference(2, 0) == -8 && !d.difference(0, 3) && d.size(1) == 3 && d.count() == 2;
	}

	template<atl::signed_integral Potential>
	constexpr bool rollback_weighted_operations_check() {
		atl::rollback_weighted_dsu<Potential> d(3);
		d.unite(0, 1, 2);
		auto const t = d.time();
		d.unite(1, 2, 4);
		bool const is_joined = d.difference(0, 2) == 6 && d.find(2) == d.find(0);
		d.rollback(t);
		return is_joined && !d.same(0, 2) && d.difference(1, 0) == -2 && d.unite(2, 1, 1) == atl::unite_outcome::merged;
	}
}

static_assert(weighted_operations_check<atl::int32>() && weighted_operations_check<atl::int64>());
static_assert(rollback_weighted_operations_check<atl::int32>() && rollback_weighted_operations_check<atl::int64>());
static_assert(weighted_operations_check<atl::int128>() && rollback_weighted_operations_check<atl::int128>());

static_assert([] {
	std::array<std::pair<int, int>, 3> const lifetimes{{{0, 4}, {1, 3}, {2, 2}}};
	std::array<std::pair<int, int>, 3> const edge{{{0, 1}, {1, 2}, {2, 3}}};
	auto const counts = [&]<atl::restore_mode<atl::rollback_dsu> Mode>(Mode mode) {
		atl::rollback_dsu d(4);
		std::array<std::size_t, 4> out{};
		atl::time_divide_and_conquer(d, 4, lifetimes, [&](atl::rollback_dsu &s, atl::uint32 i) { s.unite(edge[i].first, edge[i].second); }, [&](atl::rollback_dsu &s, atl::uint32 t) { out[t] = s.count(); }, [](atl::uint32 t) { return t != 2; }, mode);
		return out;
	};
	std::array<std::size_t, 4> const expected{3, 2, 0, 3};
	return counts(atl::default_restore<atl::rollback_dsu>()) == expected && counts(atl::by_copy) == expected;
}());

// Sparse ranks preserve half-open endpoints and original callback indices.
static_assert([] {
	std::array<atl::item_lifetime, 6> const lifetimes{{{0, 33}, {6, 7}, {7, 8}, {8, 31}, {31, 32}, {32, 33}}};
	auto const check = [&]<atl::restore_mode<atl::uint32> Mode>(Mode mode, bool has_queries) {
		atl::uint32 state = 0;
		std::array<atl::uint32, 2> visited{}, masks{};
		atl::uint32 visits = 0, predicate_calls = 0, inserts = 0;
		atl::time_divide_and_conquer(state, 33, lifetimes, [&](atl::uint32 &s, atl::uint32 i) { ++inserts; s |= atl::uint32(1) << i; }, [&](atl::uint32 &s, atl::uint32 t) { visited[visits] = t; masks[visits++] = s; }, [&](atl::uint32 t) { ++predicate_calls; return has_queries && (t == 7 || t == 31); }, mode);
		return state == 0 && predicate_calls == 33
			&& (has_queries ? visits == 2 && visited == std::array<atl::uint32, 2>{7, 31} && masks == std::array<atl::uint32, 2>{5, 17} : visits == 0 && inserts == 0);
	};
	auto const rollback = atl::by_rollback{[](atl::uint32 s) { return s; }, [](atl::uint32 &s, atl::uint32 token) { s = token; }};
	return check(rollback, true) && check(atl::by_copy, true) && check(rollback, false) && check(atl::by_copy, false);
}());

static_assert([] {
	atl::static_vector<int, 4> v{3, 1};
	v.push_back(4);
	int const x = 1;
	v.push_back(x);
	int sum = 0;
	for (int const e : v) sum += e;
	atl::static_vector<int, 4> const &c = v;
	return v.size() == 4 && !v.empty() && c[2] == 4 && sum == 9;
}());
static_assert([] {
	struct pair {
		int a = 0, b = 0;
		constexpr pair() = default;
		constexpr pair(int a_, int b_) : a(a_), b(b_) {}
	};
	atl::static_vector<pair, 2> v;
	v.emplace_back(2, 5);
	return v.size() == 1 && v[0].a == 2 && v[0].b == 5;
}());
static_assert([] {
	std::array const src{5, 6, 7};
	atl::static_vector<int, 4> v(src);
	v.pop_back();
	v.resize(4);
	v.back() = 9;
	atl::static_vector<int, 4> w{5, 6, 0, 9};
	bool const is_equal = v == w && (v <=> w) == 0;
	w.resize(1, 3);
	bool const is_ordered = w < v && *v.data() == 5 && v.front() == 5;
	v.clear();
	return is_equal && is_ordered && v.empty() && decltype(v)::capacity() == 4;
}());
static_assert([] {
	std::array const a{30, 10, 20, 10, 30};
	atl::coordinate_compression const cc(a);
	return cc.size() == 3 && cc.rank(20) == 1 && cc.rank(15) == 1 && cc.rank_upper(20) == 2 && cc.contains(10) && !cc.contains(25) && cc[2] == 30;
}());
static_assert([] {
	std::array const a{-3, 2, -1};
	atl::coordinate_compression const cc(a, [](int x) { return x < 0 ? -x : x; }, std::ranges::greater{});
	return cc.size() == 3 && cc[0] == 3 && cc.rank(2) == 1 && cc.rank(0) == 3;
}());
static_assert([] {
	std::array const a{30, 10, 20, 10, 30};
	atl::coordinate_compression const cc(a);
	auto const merged = cc.ranks(std::array{5, 10, 15, 30, 40});
	auto const searched = cc.ranks(std::array{40, 10, 25, 5});
	auto const both = atl::compress(a);
	return merged == std::vector<atl::uint32>{0, 0, 1, 2, 3} && searched == std::vector<atl::uint32>{3, 0, 2, 0}
		&& both.universe.size() == 3 && both.ranks == std::vector<atl::uint32>{2, 0, 1, 0, 2};
}());
static_assert([] {
	auto const c = atl::compress(std::array<signed char, 5>{-128, 127, 0, -1, -128});
	atl::coordinate_compression const cc(std::array{0, 2, 4, 6, 8});
	return c.ranks == std::vector<atl::uint32>{0, 3, 2, 1, 0} && cc.ranks(std::array{9, -1, 4, 4}) == std::vector<atl::uint32>{5, 0, 2, 2};
}());

static_assert(atl::detail::radix_index<16>::bits == 4 && atl::detail::radix_index<16>::mask == 15);
static_assert(atl::detail::radix_index<16>::height_for(0) == 0 && atl::detail::radix_index<16>::height_for(1) == 0);
static_assert(atl::detail::radix_index<16>::height_for(16) == 0 && atl::detail::radix_index<16>::height_for(17) == 1);
static_assert(atl::detail::radix_index<16>::height_for(256) == 1 && atl::detail::radix_index<16>::height_for(257) == 2);
static_assert(atl::detail::radix_index<2>::height_for(2) == 0 && atl::detail::radix_index<2>::height_for(3) == 1);
static_assert(atl::detail::radix_index<32>::height_for(1024) == 1 && atl::detail::radix_index<32>::height_for(1025) == 2);
static_assert(atl::detail::radix_index<16>::digit(0x1234, 0) == 4 && atl::detail::radix_index<16>::digit(0x1234, 1) == 3);
static_assert(atl::detail::radix_index<16>::digit(0x1234, 3) == 1 && atl::detail::radix_index<16>::digit(0x1234, 4) == 0);
static_assert(!atl::detail::valid_radix<1> && !atl::detail::valid_radix<12> && atl::detail::valid_radix<1u << 16> && !atl::detail::valid_radix<1u << 17>);
static_assert(atl::detail::radix_index<16>::height_for(atl::uint128(1) << 100) == 24);
static_assert(atl::detail::radix_index<16>::digit(atl::uint128(0x1234) << 64, 17) == 3);

static_assert([] {
	atl::flat_hash_set<atl::uint32, atl::identity_hash> s;
	for (atl::uint32 x = 0; x < 20; ++x) s.insert(x * 7);
	bool const is_duplicate_rejected = !s.insert(14);
	s.erase(21);
	bool const is_reinserted = s.insert(21);
	std::size_t visited = 0;
	for (atl::uint32 const x : s) visited += x % 7 == 0;
	return is_duplicate_rejected && is_reinserted && s.size() == 20 && visited == 20 && s.contains(133) && !s.contains(140) && s.find(140) == s.end();
}());
static_assert([] {
	atl::flat_hash_map<atl::uint32, int, atl::identity_hash> m;
	for (atl::uint32 x = 0; x < 20; ++x) m[x % 5] += int(x);
	m.insert_or_assign(7, 1);
	bool const is_existing_kept = !m.insert(7, 2);
	return m.size() == 6 && m.at(0) == 30 && m.at(4) == 46 && m.at(7) == 1 && m.at_or(8, -1) == -1 && is_existing_kept;
}());
static_assert([] {
	struct wide_hash {
		using is_transparent = void;
		constexpr std::size_t operator()(atl::uint64 x) const noexcept { return std::size_t(x * 0x9E3779B97F4A7C15ull >> 20); }
	};
	atl::flat_hash_map<atl::uint32, int, wide_hash> m;
	atl::flat_hash_set<atl::uint32, wide_hash> s;
	for (atl::uint32 x = 0; x < 20; ++x) {
		m[x] = int(x) * 2;
		s.insert(x);
	}
	atl::uint64 const wide = 7;
	bool const is_erased = m.erase(atl::uint64(3)) && s.erase(atl::uint64(3));
	return m.at(wide) == 14 && m.contains(wide) && m.count(wide) == 1 && m.find(wide)->second == 14 && m.at_or(atl::uint64(3), -1) == -1
		&& is_erased && !s.contains(atl::uint64(3)) && s.contains(wide) && s.find(wide) != s.end() && !m.contains(atl::uint64(99));
}());
static_assert([] {
	std::array<std::pair<atl::uint32, int>, 3> const entries{{{1, 10}, {2, 20}, {1, 30}}};
	atl::flat_hash_map<atl::uint32, int, atl::identity_hash> m(entries);
	bool const is_first_kept = m.at(1) == 10;
	bool const is_emplaced = m.try_emplace(3, 4) && !m.try_emplace(3, 5) && m.emplace(4u, 6) && !m.emplace(4u, 7);
	std::array<atl::uint32, 4> const keys{5, 6, 5, 7};
	atl::flat_hash_set<atl::uint32, atl::identity_hash> s(keys);
	s.insert(keys);
	return is_first_kept && is_emplaced && m.size() == 4 && m.at(3) == 4 && m.at(4) == 6 && s.size() == 3;
}());
static_assert([] {
	atl::hierarchical_bitset t(4097);
	t.insert(3);
	t.insert(700);
	t.insert(4096);
	t.erase(700);
	return t.size() == 2 && t.next(4) == 4096 && t.prev(4095) == 3 && t.min() == 3 && t.max() == 4096 && !t.contains(700)
		&& t.next(0) == 3 && t.next(4097) == atl::hierarchical_bitset::npos;
}());
static_assert([] {
	atl::hierarchical_bitset const t(200, std::array<std::size_t, 4>{64, 5, 199, 5});
	return t.size() == 3 && t.next(6) == 64 && t.prev(198) == 64 && t.max() == 199;
}());

static_assert([] {
	atl::detail::persistent_arena<int> nodes(-1);
	nodes.reserve_extra(2);
	auto const a = nodes.make(7), b = nodes.make(9);
	nodes[a] += nodes[b];
	auto const &read = std::as_const(nodes);
	atl::detail::persistent_arena<int> empty;
	return read[0] == -1 && read[a] == 16 && read.size() == 3 && empty.make(4) == 0 && empty[0] == 4;
}());
static_assert(atl::detail::key_value_entry<std::pair<int, int>, int, long>);
static_assert(atl::detail::key_value_entry<std::array<int, 2>, int, int>);
static_assert(!atl::detail::key_value_entry<int, int, int>);
static_assert(!atl::detail::heterogeneous_key_for<atl::uint64, atl::uint32, std::less<>>);
static_assert(atl::detail::heterogeneous_key_for<std::string_view, std::string, std::less<>>);
static_assert(atl::detail::hashed_heterogeneous_key_for<std::string_view, std::string, atl::hash<std::string>, std::equal_to<>>);
static_assert(std::same_as<atl::detail::forwarded_element_t<std::vector<int>>, int &&>);
static_assert(std::same_as<atl::detail::forwarded_element_t<std::vector<int> &>, int const &>);
static_assert(std::same_as<atl::detail::forwarded_element_t<std::span<int>>, int const &>);
static_assert([] { int x = 7; return &atl::detail::forward_element<std::span<int>>(x) == &x; }());

namespace {
	// enable_view alone is insufficient: an opted-in, immovable range is not a view.
	struct immovable_range : std::ranges::view_base {
		immovable_range() = default;
		immovable_range(immovable_range &&) = delete;

		int *begin() const { return nullptr; }

		int *end() const { return nullptr; }
	};

	static_assert(std::ranges::enable_view<immovable_range> && !std::ranges::view<immovable_range>);
	static_assert(atl::detail::moves_range_elements<immovable_range>);
	static_assert(!atl::detail::moves_range_elements<std::ranges::owning_view<std::vector<int>>>);

	// Copy/move checks must not recurse through the range constructor's view check.
	static_assert(std::copyable<atl::coordinate_compression<int>>);
	static_assert(std::copyable<atl::persistent_array<int>>);
	static_assert(std::copyable<atl::semi_persistent_array<int>>);
	static_assert(std::same_as<atl::detail::forwarded_element_t<atl::persistent_array<int>>, int const &&>);

	// An own-type range constructor with an explicit projection remains a valid call.
	static_assert([] {
		atl::coordinate_compression<int> source(std::array{3, 1, 3});
		atl::coordinate_compression<int> projected(std::move(source), [](int const &&x) { return -x; });
		return projected.size() == 2 && projected[0] == -3 && projected[1] == -1;
	}());

	struct child_node {
		atl::uint32 symbol{}, left{}, right{};
	};
}

static_assert(atl::detail::sibling_node<child_node>);
static_assert(atl::detail::hashed_child_pool<child_node>::nil == (std::numeric_limits<atl::uint32>::max)());

static_assert(atl::detail::persistent_signed_array<atl::persistent_array<atl::int32>>);
static_assert(atl::detail::persistent_signed_array<atl::semi_persistent_array<atl::int128>>);
static_assert(!atl::detail::persistent_signed_array<atl::persistent_array<atl::uint128>>);
static_assert([] {
	using table = atl::detail::hash_table<atl::uint32, atl::uint32, atl::detail::set_key_of, atl::identity_hash, std::equal_to<>>;
	table t(4);
	auto const [slot, was_inserted] = t.find_or_prepare(atl::uint32(7));
	bool const is_stored = was_inserted && t.slot(slot) == 7 && t.find(7u) == slot;
	bool const is_erased = t.erase(7u) && !t.contains(7u);
	t.find_or_prepare(atl::uint32(9));
	t.clear();
	return is_stored && is_erased && t.empty() && t.begin() == t.end();
}());

// Arithmetic lookups convert to Key; a distinct nonconvertible view proves transparent probing.
static_assert([] {
	struct lookup {
		atl::uint64 value;
	};
	struct key {
		atl::uint32 value{};
		constexpr bool operator==(key const &) const = default;
		constexpr bool operator==(lookup other) const { return value == other.value; }
	};
	struct transparent_hash {
		using is_transparent = void;
		constexpr std::size_t operator()(key k) const noexcept { return k.value; }
		constexpr std::size_t operator()(lookup k) const noexcept { return std::size_t(k.value); }
	};
	static_assert(atl::detail::hashed_heterogeneous_key_for<lookup, key, transparent_hash, std::equal_to<>>);
	atl::flat_hash_map<key, int, transparent_hash> map;
	atl::flat_hash_set<key, transparent_hash> set;
	map.insert(key{7}, 14);
	set.insert(key{7});
	bool const is_found = map.at(lookup{7}) == 14 && map.contains(lookup{7}) && set.find(lookup{7}) != set.end();
	return is_found && map.erase(lookup{7}) && set.erase(lookup{7}) && !map.contains(lookup{7});
}());

#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/ds/detail/bit_vector.hpp>
#include <atl/ds/difference_fenwick.hpp>
#include <atl/ds/direct_rmq.hpp>
#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
#include <atl/ds/sparse_table.hpp>
#include <atl/ds/wavelet_matrix.hpp>
#include <atl/experimental/ds/dynamic_li_chao_tree_variants.hpp>
#include <atl/experimental/ds/li_chao_tree_variants.hpp>
#include <atl/utils/floor_midpoint.hpp>

namespace ds2_checks {
	static_assert([] {
		atl::binary_indexed_tree<int> bit(5);
		bit.update(3, 0);
		bit.update(5, 2);
		return bit.prefix_fold(3) == 8 && bit.size() == 5;
	}());
	static_assert([] {
		atl::binary_indexed_tree<int, 2> bit({3, 3});
		bit.update(5, 1, 1);
		bit.update(2, 2, 0);
		return bit.prefix_fold(3, 3) + bit.prefix_fold(2, 2) == 12;
	}());
	static_assert([] {
		auto bit = atl::make_binary_indexed_tree(0LL, 3, 3);
		bit.update(5, 1, 1);
		return bit.prefix_fold(3, 3) == 5;
	}());
	static_assert([] {
		std::array<int, 5> vals{3, 1, 4, 1, 5};
		auto bit = atl::make_binary_indexed_tree_from(vals);
		return bit.prefix_fold(5) == 14 && bit.size() == 5
			&& bit.max_right([](int s) { return s <= 8; }) == 3
			&& bit.max_right([](int s) { return s < 100; }) == 5;
	}());
	static_assert([] {
		std::array<std::array<int, 2>, 2> grid{{{1, 2}, {3, 4}}};
		auto bit = atl::make_binary_indexed_tree_from(grid);
		atl::binary_indexed_tree<int, 2> sized({3, 4});
		return bit.prefix_fold(2, 2) == 10 && sized.sizes()[0] == 3 && sized.sizes()[1] == 4;
	}());
	static_assert([] {
		std::array<int, 5> a{4, 2, 3, 1, 5};
		atl::direct_rmq rmq(a);
		atl::direct_rmq<int> assigned;
		bool const was_empty = assigned.empty() && assigned.size() == 0;
		assigned = atl::direct_rmq<int>(std::span<int const>(a));
		return rmq.position(0, 5) == 3 && rmq.fold(0, 5) == 1
			&& was_empty && assigned.size() == 5 && assigned.position(1, 3) == 1;
	}());
	static_assert([] {
		std::array const a{4, 2, 7, 1, 9};
		atl::sparse_table const st(a, [](int x, int y) { return x < y ? x : y; });
		atl::disjoint_sparse_table const dst(a, [](int x, int y) { return x + y; });
		atl::sparse_table<int, std::plus<>> empty;
		atl::disjoint_sparse_table<int, std::plus<>> empty_disjoint;
		return st.fold(0, 5) == 1 && st.fold(1, 3) == 2 && dst.fold(0, 5) == 23 && dst.fold(2, 4) == 8
			&& st.size() == 5 && !st.empty() && dst.size() == 5 && !dst.empty()
			&& empty.empty() && empty.size() == 0 && empty_disjoint.empty() && empty_disjoint.size() == 0;
	}());
	static_assert([] {
		std::array const a{1, 2, 3, 4};
		atl::difference_fenwick<int> fen(a, 0);
		fen.apply(0, 3, 5);
		fen.apply(2, 1);
		fen.apply(0, 4, 1);
		atl::difference_fenwick<int> filled(std::size_t(3), 2, 0);
		filled.apply(1, 3, 4);
		return fen.at(0) == 7 && fen.at(1) == 8 && fen.at(2) == 10 && fen.at(3) == 5
			&& filled.at(0) == 2 && filled.at(1) == 6 && filled.at(2) == 6;
	}());
	static_assert([] {
		using tree = atl::difference_fenwick<int, std::bit_xor<>, std::identity>;
		tree empty(std::size_t(0), 0, 0);
		empty.apply(0, 0, 1);
		tree bits(std::array{1, 2, 3}, 0);
		bits.apply(0, 3, 4);
		bits.apply(2, 4);
		return empty.empty() && empty.size() == 0 && bits.size() == 3 && !bits.empty()
			&& bits.at(0) == 5 && bits.at(1) == 6 && bits.at(2) == 3;
	}());
	static_assert([] {
		std::array<atl::uint32, 6> const a{5, 1, 4, 1, 7, 0};
		atl::wavelet_matrix const wm(a);
		atl::wavelet_matrix<atl::uint32> empty;
		return wm.size() == 6 && !wm.empty() && wm.at(4) == 7 && wm.kth(1, 5, 2) == 4
			&& wm.count_less(1, 5, 4) == 2 && wm.count(0, 6, 1, 5) == 3 && empty.empty() && empty.size() == 0;
	}());
	static_assert([] {
		atl::detail::bit_vector bv(130);
		bv.set(0);
		bv.set_word(1, 1);
		bv.set(129);
		bv.finish();
		return bv.size() == 130 && bv.at(64) && !bv.at(65) && bv.rank1(65) == 2 && bv.rank0(130) == 127 && bv.zeros() == 127;
	}());

	template<typename Tree>
		requires requires(Tree &tree) { tree.insert(1, 0, 0, 10); tree.insert(0, 4); { tree.query(7) == 3 } -> std::convertible_to<bool>; }
	constexpr bool envelope(Tree tree) {
		tree.insert(1, 0, 0, 10);
		tree.insert(-1, 10, 3, 8);
		if (*tree.query(7) != 3 || *tree.query(9) != 9) return false;
		tree.insert(0, 4);
		return *tree.query(9) == 4 && *tree.query(7) == 3 && *tree.query(1) == 1;
	}

	static_assert(envelope(atl::li_chao_tree<long long>(0, 10)));
	static_assert(envelope(atl::li_chao_tree<long long>(std::vector<long long>{1, 7, 9})));
	static_assert(envelope(atl::experimental::li_chao_tree_by_inline_flags<long long>(0, 10)));
	static_assert(envelope(atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<long long>(0, 10)));
	static_assert(envelope(atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<long long>(std::vector<long long>{1, 7, 9})));
	static_assert(envelope(atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<long long, std::less<>, atl::linear_function<long long>, false, atl::experimental::coordinate_search::branchless>(std::vector<long long>{1, 7, 9})));
	static_assert(envelope(atl::dynamic_li_chao_tree<long long>(-1000, 1000)));
	static_assert(envelope(atl::dynamic_li_chao_tree<atl::int128>(-1000, 1000)));
	static_assert(envelope(atl::experimental::dynamic_li_chao_tree_by_materialized_paths<long long>(-1000, 1000)));
	static_assert(envelope(atl::experimental::dynamic_li_chao_tree_by_two_bound_clips<long long>(-1000, 1000)));
	static_assert(envelope(atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<long long, std::less<>, atl::linear_function<long long>, true, true>(-1000, 1000)));
	static_assert(envelope(atl::experimental::dynamic_li_chao_tree_by_optional_dominated_exit<long long>(-1000, 1000)));

	struct parabola {
		long long shift = 0, lift = 0;

		constexpr long long operator()(long long x) const { return (x - shift) * (x - shift) + lift; }
	};

	template<typename Tree>
		requires requires(Tree &tree) { tree.insert(parabola{}); tree.insert(parabola{}, 3, 10); { tree.query(4) == 4 } -> std::convertible_to<bool>; }
	constexpr bool parabolas(Tree tree) {
		tree.insert(parabola{2, 0});
		tree.insert(parabola{6, 1}, 3, 10);
		return *tree.query(4) == 4 && *tree.query(5) == 2 && *tree.query(0) == 4;
	}

	static_assert(parabolas(atl::li_chao_tree<long long, std::less<>, parabola>(0, 10)));
	static_assert(parabolas(atl::dynamic_li_chao_tree<long long, std::less<>, parabola>(-100, 100)));
	static_assert(atl::detail::floor_midpoint(-7LL, 4LL) == -2);
	static_assert(atl::detail::floor_midpoint((std::numeric_limits<long long>::min)(), (std::numeric_limits<long long>::max)()) == -1);
	static_assert([] {
		atl::dynamic_li_chao_tree<atl::int64> tree(0, 1'000'000);
		tree.reserve(8);
		tree.insert(0, 1);
		tree.insert(0, 1, 1, 999'999);
		if (tree.node_count() != 1) return false;
		tree.insert(0, 0, 5, 6);
		return !tree.empty() && tree.query(4) == 1 && tree.query(5) == 0 && tree.query(6) == 1;
	}());

	// Exercise both concrete software classes even on native-int128 targets.
	template<atl::integral Value>
	constexpr bool overflow_boundaries() {
		constexpr Value low = (std::numeric_limits<Value>::lowest)();
		constexpr Value high = (std::numeric_limits<Value>::max)();
		if (!atl::detail::fits_on(atl::linear_function<Value>{1, 0}, low, high)) return false;
		if (atl::detail::fits_on(atl::linear_function<Value>{1, 1}, high, high)) return false;
		if (atl::detail::fits_on(atl::linear_function<Value>{2, 0}, high, high)) return false;
		if constexpr (atl::signed_integral<Value>) {
			if (atl::detail::fits_on(atl::linear_function<Value>{-1, 0}, low, low)) return false;
			if (atl::detail::fits_on(atl::linear_function<Value>{1, -1}, low, low)) return false;
			if (!atl::detail::fits_on(atl::linear_function<Value>{-1, -1}, high, high)) return false;
		}
		return true;
	}

	static_assert(overflow_boundaries<int>() && overflow_boundaries<unsigned>());
	static_assert(overflow_boundaries<atl::int64>() && overflow_boundaries<atl::uint64>());
	static_assert(overflow_boundaries<atl::int128>() && overflow_boundaries<atl::uint128>());
	static_assert(overflow_boundaries<atl::detail::software_int128>() && overflow_boundaries<atl::detail::software_uint128>());
	static_assert([] {
		using Value = atl::detail::software_uint128;
		std::array const a{Value(1) << 100, Value(0), (Value(1) << 100) + 1};
		atl::wavelet_matrix const wm(a);
		return wm.at(2) == a[2] && wm.kth(0, 3, 1) == a[0] && wm.count_less(0, 3, a[2]) == 2;
	}());
	static_assert(atl::linear_function<int>{2, 3}(4) == 11 && atl::linear_function<int>{2, 3} == atl::linear_function<int>{2, 3});
}

static_assert([] {
	atl::detail::bit_vector empty;
	empty.finish();
	return empty.size() == 0 && empty.rank1(0) == 0 && empty.rank0(0) == 0 && empty.zeros() == 0;
}());
static_assert(atl::detail::li_chao_width(-7, 4) == 11);
static_assert(atl::detail::li_chao_width(0u, 2147483647u) == 2147483647u);
static_assert([] {
	using Wide = atl::detail::software_int128;
	atl::li_chao_tree<Wide> points(std::vector<Wide>{Wide(1) << 70, (Wide(1) << 70) + 1});
	points.insert(2, 3);
	return points.query(Wide(1) << 70) == (Wide(1) << 71) + 3;
}());
static_assert([] {
	atl::li_chao_tree<int> array;
	atl::dynamic_li_chao_tree<int> dynamic;
	array.insert(1, 2);
	dynamic.insert(1, 2);
	return array.empty() && array.size() == 0 && dynamic.empty() && dynamic.node_count() == 0;
}());

#include <atl/ds/dual_segment_tree.hpp>
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <atl/ds/dynamic_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/ds/segment_tree_beats.hpp>

namespace segment_family_checks {
	static_assert([] {
		std::array const a{4, 2, 7, 1, 9};
		atl::segment_tree st(a, 0, [](int x, int y) { return x + y; });
		st.set(3, 6);
		int segments = 0;
		st.for_each_segment(1, 5, [&](int) { ++segments; });
		return st.fold(1, 4) == 15 && st.fold() == 28 && st.at(3) == 6
			&& st.max_right(0, [](int x) { return x <= 13; }) == 3
			&& st.min_left(5, [](int x) { return x <= 15; }) == 3 && segments == 3;
	}());
	static_assert([] {
		atl::segment_tree st(std::size_t(3), 0, [](int x, int y) { return x > y ? x : y; });
		st.set(1, 5);
		return st.fold(0, 3) == 5 && st.fold(2, 3) == 0;
	}());
	static_assert([] {
		struct node {
			long long sum, width;
		};
		std::array<node, 5> const a{{{4, 1}, {2, 1}, {7, 1}, {1, 1}, {9, 1}}};
		atl::lazy_segment_tree st(a, node{0, 0}, 0LL, [](node x, node y) { return node{x.sum + y.sum, x.width + y.width}; }, [](long long f, node x) { return node{x.sum + f * x.width, x.width}; }, std::plus<>{});
		st.apply(1, 4, 10);
		st.apply(4, 1);
		return st.fold(0, 5).sum == 54 && st.fold(2, 4).sum == 28 && st.at(1).sum == 12 && st.fold().sum == 54
			&& st.max_right(0, [](node x) { return x.sum <= 33; }) == 3
			&& st.min_left(5, [](node x) { return x.sum <= 21; }) == 3;
	}());
	static_assert([] {
		atl::dual_segment_tree st(std::array{1, 2, 3, 4}, 0, std::plus<>{}, std::plus<>{});
		st.apply(0, 3, 5);
		st.apply(2, 1);
		st.set(0, 0);
		return st.at(0) == 0 && st.at(1) == 7 && st.at(2) == 9 && st.at(3) == 4 && st.size() == 4;
	}());
	static_assert([] {
		atl::dual_segment_tree st(atl::commutative_tags, std::array{1, 2, 3, 4}, 0, std::plus<>{}, std::plus<>{});
		st.apply(0, 3, 5);
		st.apply(2, 1);
		st.set(0, 0);
		st.apply(0, 4, 1);
		return st.at(0) == 1 && st.at(1) == 8 && st.at(2) == 10 && st.at(3) == 5;
	}());
	static_assert([] {
		atl::segment_tree<int, std::plus<>> st;
		bool const was_empty = st.empty() && st.fold() == 0;
		st = atl::segment_tree<int, std::plus<>>(std::array{4, 2, 7}, 0);
		int calls = 0;
		st.update(1, [&calls](int x) { ++calls; return x * 10; });
		return was_empty && st.fold() == 31 && st.at(1) == 20 && calls == 1;
	}());
	static_assert([] {
		using tree = atl::lazy_segment_tree<int, int, std::plus<>, std::plus<>, std::plus<>>;
		tree st;
		bool const was_empty = st.empty();
		st = tree(std::size_t(3), 0, 0);
		st.apply(0, 2, 5);
		return was_empty && st.fold(0, 1) == 5 && st.at(2) == 0;
	}());
	static_assert([] {
		using tree = atl::dual_segment_tree<int, int, std::plus<>, std::plus<>>;
		tree st;
		bool const was_empty = st.empty() && st.values().empty();
		st = tree(std::array{1, 2, 3, 4}, 0);
		st.apply(1, 4, 3);
		st.apply(0, 2, 1);
		return was_empty && st.values() == std::vector{2, 6, 6, 7};
	}());
	static_assert([] {
		atl::beats_segment_tree<long long> st(std::array<long long, 5>{4, 2, 7, 1, 9});
		st.chmin(0, 5, 5);
		st.chmax(1, 4, 3);
		st.add(0, 2, 10);
		st.assign(4, 5, 0);
		st.set(2, 6);
		return st.sum(0, 5) == 36 && st.min(0, 5) == 0 && st.max(1, 4) == 13 && st.element(1) == 13
			&& st.max_right(0, [](auto const &v) { return v.sum <= 27; }) == 2
			&& st.min_left(5, [](auto const &v) { return v.max <= 6; }) == 2
			&& st.max_right(1, [](auto const &v) { return v.max_count <= 1 && v.second_min >= 6; }) == 4;
	}());
	static_assert([] {
		atl::beats_segment_tree<int> st;
		bool const was_empty = st.empty() && st.sum(0, 0) == 0;
		st = atl::beats_segment_tree<int>(std::size_t(3));
		st.add(0, 3, 2);
		st.chmin(1, 2, -1);
		return was_empty && st.sum(0, 3) == 3 && st.min(0, 3) == -1 && st.size() == 3;
	}());
	static_assert([] {
		int const top = (std::numeric_limits<int>::max)() - 1;
		atl::beats_segment_tree<int, long long> st(std::array{top, 1, top, 2, 3});
		st.chmin(0, 5, 2);
		st.assign(0, 5, -top);
		st.add(0, 5, top);
		st.add(0, 5, top);
		return st.element(3) == top && st.sum(1, 4) == 3LL * top;
	}());
	static_assert([] {
		using tree = atl::segment_tree_beats<atl::beats::extremes<long long>, atl::beats::clamp_add<long long>>;
		tree st(std::array<long long, 5>{4, 2, 7, 1, 9});
		st.chmin(0, 5, 5);
		st.chmax(1, 4, 3);
		st.add(0, 2, 10);
		st.assign(4, 5, 0);
		st.set(2, 6);
		auto const f = st.fold(1, 4);
		tree none;
		return st.sum(0, 5) == 36 && st.min(0, 5) == 0 && st.max(1, 4) == 13 && st.element(1) == 13
			&& f.max_count == 1 && f.second_min == 6 && st.fold().second_max == 13
			&& st.max_right(0, [](auto const &v) { return v.sum <= 27; }) == 2
			&& st.min_left(5, [](auto const &v) { return v.max <= 6; }) == 2 && none.empty();
	}());
	static_assert([] {
		atl::segment_tree_beats<atl::beats::mod_sum<int>, atl::beats::assign<int>> st(std::array{9, 4, 15, 7});
		st.modulo(0, 4, 5);
		st.assign(1, 3, 8);
		st.modulo(0, 4, 3);
		return st.sum(0, 4) == 7 && st.max(0, 4) == 2 && st.min(0, 4) == 1;
	}());
	static_assert([] {
		atl::segment_tree_beats<atl::beats::sqrt_add_sum<long long>, atl::beats::add<long long>> st(std::array<long long, 4>{16, 17, 100, 3});
		st.floor_sqrt(0, 4);
		st.add(0, 2, 5);
		st.divide(0, 4, 2);
		return st.sum(0, 4) == 13 && st.max(0, 4) == 5 && st.element(3) == 0;
	}());
	static_assert([] {
		atl::segment_tree_beats<atl::beats::bits_max<unsigned>, atl::beats::bit_mask<unsigned>> st(std::array{12u, 5u, 9u});
		st.bit_and(0, 3, 6u);
		st.bit_or(1, 3, 3u);
		return st.max(0, 3) == 7 && st.element(0) == 4 && st.max(2, 3) == 3;
	}());
	static_assert([] {
		struct node {
			int max, second, count, sum;
		};
		std::array<node, 4> const a{{{5, -1000, 1, 5}, {3, -1000, 1, 3}, {8, -1000, 1, 8}, {1, -1000, 1, 1}}};
		atl::segment_tree_beats st(a, node{-1000, -1000, 0, 0}, 1000, [](node x, node y) {
				if (x.max == y.max) return node{x.max, (std::max)(x.second, y.second), x.count + y.count, x.sum + y.sum};
				node const &hi = x.max > y.max ? x : y, &lo = x.max > y.max ? y : x;
				return node{hi.max, (std::max)(hi.second, lo.max), hi.count, x.sum + y.sum}; }, [](int const &c, node &v) {
				if (v.max <= c) return true;
				if (c <= v.second) return false;
				v.sum -= (v.max - c) * v.count; v.max = c; return true; }, [](int x, int y) { return (std::min)(x, y); });
		st.apply(0, 4, 4);
		return st.fold().sum == 12 && st.fold(1, 3).sum == 7;
	}());
	static_assert([] {
		using wide = atl::int128;
		wide const top = wide(1) << 100;
		atl::beats_segment_tree<wide> st(std::array{top, top - 1});
		st.assign(0, 2, -top);
		st.add(0, 2, top);
		st.add(0, 2, top);
		return st.sum(0, 2) == top * 2 && atl::detail::beats_floor_sqrt(wide(1) << 100) == wide(1) << 50;
	}());
	static_assert(atl::update_function<decltype([](int x) { return x + 1; }), int>);
	static_assert(std::same_as<atl::dynamic_segment_tree<int, std::plus<>>::value_type, int>);
	static_assert(atl::fill_function<decltype([](std::size_t lo, std::size_t hi) { return int(hi - lo); }), int>);
}

namespace dynamic_segment_checks {
	static_assert([] {
		using tree = atl::dynamic_segment_tree<int, std::plus<>>;
		tree empty;
		if (!empty.empty() || empty.size() != 0 || empty.node_count() != 1 || empty.fold() != 0 || empty.fold(0, 0) != 0) return false;
		if (empty.max_right(0, [](int x) { return x == 0; }) != 0 || empty.min_left(0, [](int x) { return x == 0; }) != 0) return false;
		empty.reserve(1);
		tree st(9, 0);
		if (st.empty() || st.size() != 9 || st.node_count() != 1 || st.at(4) != 0) return false;
		st.set(6, 7);
		if (st.node_count() != 2 || st.at(6) != 7 || st.at(5) != 0) return false;
		st.set(1, 3);
		int calls = 0;
		st.update(1, [&calls](int x) { ++calls; return x * 2; });
		st.update(3, [&calls](int x) { ++calls; return x + 4; });
		st.set(6, 9);
		auto const count = st.node_count();
		st.reserve(50);
		auto const &view = st;
		return calls == 2 && count > 2 && view.node_count() == count && view.fold() == 19
			&& view.fold(1, 6) == 10 && view.fold(4, 4) == 0 && view.at(3) == 4 && view.at(0) == 0
			&& view.max_right(0, [](int x) { return x <= 10; }) == 6
			&& view.max_right(7, [](int x) { return x == 0; }) == 9
			&& view.min_left(9, [](int x) { return x <= 13; }) == 2
			&& view.min_left(1, [](int x) { return x == 0; }) == 0
			&& view.node_count() == count;
	}());

	// Explicit construction avoids MSVC constexpr aggregate-temporary initialization failures.
	struct sum_width {
		int sum = 0, width = 0;

		constexpr sum_width(int total = 0, int count = 0) : sum(total), width(count) {}
	};

	struct affine {
		int scale = 1, bias = 0;

		constexpr affine(int factor = 1, int offset = 0) : scale(factor), bias(offset) {}

		constexpr bool operator==(affine const &) const = default;
	};

	static_assert([] {
		auto op = [](sum_width x, sum_width y) { return sum_width{x.sum + y.sum, x.width + y.width}; };
		auto mapping = [](affine f, sum_width x) { return sum_width{f.scale * x.sum + f.bias * x.width, x.width}; };
		auto compose = [](affine f, affine g) { return affine{f.scale * g.scale, f.scale * g.bias + f.bias}; };
		auto fill = [](std::size_t width) { return sum_width{int(width), int(width)}; };
		atl::dynamic_lazy_segment_tree st(7, sum_width{}, affine{}, op, mapping, compose, fill);
		decltype(st) empty{};
		empty.reserve(1);
		empty.reserve_nodes(20);
		empty.apply(0, 0, affine{2, 1});
		if (!empty.empty() || empty.size() != 0 || empty.node_count() != 1 || empty.fold().sum != 0 || empty.fold(0, 0).sum != 0) return false;
		if (empty.max_right(0, [](sum_width x) { return x.sum == 0; }) != 0) return false;
		if (st.empty() || st.size() != 7 || st.node_count() != 2 || st.fold().sum != 7) return false;
		st.apply(0, 7, affine{2, 1});
		if (st.fold(1, 6).sum != 15 || st.at(2).sum != 3 || st.max_right(0, [](sum_width x) { return x.sum <= 10; }) != 3 || st.node_count() != 2) return false;
		st.apply(1, 6, affine{1, 2});
		st.apply(2, affine{2, 0});
		st.set(4, sum_width{1, 1});
		auto const count = st.node_count();
		st.reserve(5);
		st.reserve_nodes(256);
		auto const &view = st;
		return count > 2 && view.size() == 7 && view.fold().sum == 32 && view.fold().width == 7
			&& view.fold(1, 5).sum == 21 && view.fold(3, 3).sum == 0 && view.at(2).sum == 10
			&& view.max_right(0, [](sum_width x) { return x.sum <= 18; }) == 3
			&& view.max_right(3, [](sum_width x) { return x.sum <= 11; }) == 6
			&& view.max_right(7, [](sum_width x) { return x.sum == 0; }) == 7
			&& view.node_count() == count;
	}());

	static_assert([] {
		int const identity = (std::numeric_limits<int>::max)();
		// MSVC constexpr evaluation loses std::optional's engagement flag during tag copies.
		struct tag {
			int value = 0;
			bool has_value = false;

			constexpr tag() = default;

			constexpr explicit tag(int assigned) : value(assigned), has_value(true) {}

			constexpr bool operator==(tag const &) const = default;
		};
		auto min = [](int x, int y) { return (std::min)(x, y); };
		auto mapping = [](tag f, int x) { return f.has_value ? f.value : x; };
		auto compose = [](tag f, tag g) { return f.has_value ? f : g; };
		auto fill = [](std::size_t lo, std::size_t hi) { return lo == hi ? (std::numeric_limits<int>::max)() : int(lo) + 10; };
		atl::lazy_segment_tree dense(std::array{10, 11, 12, 13, 14}, identity, tag{}, min, mapping, compose);
		atl::dynamic_lazy_segment_tree sparse(5, identity, tag{}, min, mapping, compose, fill);
		// Assignment maps the element identity to a value; only nonempty intervals receive tags.
		if (mapping(tag{7}, identity) != 7) return false;
		sparse.apply(0, 5, tag{7});
		dense.apply(0, 5, tag{7});
		sparse.apply(1, 4, tag{3});
		dense.apply(1, 4, tag{3});
		sparse.apply(2, tag{9});
		dense.apply(2, tag{9});
		sparse.set(3, 8);
		dense.set(3, 8);
		return sparse.fold() == 3 && dense.fold() == 3 && sparse.fold(2, 4) == 8 && dense.fold(2, 4) == 8
			&& sparse.at(4) == 7 && dense.at(4) == 7 && sparse.fold(3, 3) == identity && dense.fold(3, 3) == identity
			&& sparse.max_right(0, [](int x) { return x >= 7; }) == 1 && dense.max_right(0, [](int x) { return x >= 7; }) == 1;
	}());

	template<atl::signed_integral Element, atl::signed_integral Sum>
	constexpr bool wrapping_pending_add() {
		Element const top = (std::numeric_limits<Element>::max)() / 4;
		atl::beats_segment_tree<Element, Sum> st(std::array<Element, 2>{Element(-top), top});
		st.assign(0, 2, Element(-top));
		for (int i = 0; i < 6; ++i) {
			st.add(0, 2, top);
			st.chmin(0, 2, Element(-top));
		}
		if (st.element(1) != -top || st.sum(0, 2) != Sum(-top) * 2) return false;
		st.assign(0, 2, top);
		for (int i = 0; i < 6; ++i) {
			st.add(0, 2, Element(-top));
			st.chmax(0, 2, top);
		}
		return st.element(0) == top && st.sum(0, 2) == Sum(top) * 2;
	}
}

static_assert(dynamic_segment_checks::wrapping_pending_add<signed char, long long>());
static_assert(dynamic_segment_checks::wrapping_pending_add<long long, atl::int128>());
static_assert(dynamic_segment_checks::wrapping_pending_add<atl::detail::software_int128, atl::detail::software_int128>());

#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/ds/splay_engine.hpp>

namespace treap_proof {
	struct array_store {
		struct node {
			atl::uint32 left = 0, right = 0, size = 0, priority = 0;
		};

		std::array<node, 9> nodes{};
		atl::uint64 draw = 0;
		std::array<atl::uint32, 8> traversal{};

		constexpr atl::uint32 left(atl::uint32 x) const { return nodes[x].left; }

		constexpr atl::uint32 right(atl::uint32 x) const { return nodes[x].right; }

		constexpr atl::uint32 size(atl::uint32 x) const { return nodes[x].size; }

		constexpr atl::uint32 priority(atl::uint32 x) const { return nodes[x].priority; }

		constexpr atl::uint32 push(atl::uint32 x) { return x; }

		constexpr atl::uint32 rebind(atl::uint32 x, atl::uint32 l, atl::uint32 r) {
			nodes[x].left = l;
			nodes[x].right = r;
			nodes[x].size = 1 + nodes[l].size + nodes[r].size;
			return x;
		}

		// Return the cursor by value: recursive constexpr accumulators must not alias a caller's local.
		constexpr std::size_t in_order(atl::uint32 x, std::size_t cursor) {
			if (x == 0) return cursor;
			cursor = in_order(nodes[x].left, cursor);
			traversal[cursor++] = x;
			return in_order(nodes[x].right, cursor);
		}
	};

	constexpr bool is_in_order(array_store &store, atl::uint32 root, atl::uint32 first, atl::uint32 count) {
		if (store.in_order(root, 0) != count || store.size(root) != count) return false;
		for (atl::uint32 i = 0; i < count; ++i)
			if (store.traversal[i] != first + i) return false;
		return true;
	}

	static_assert([] {
		using base = atl::detail::treap_base;
		// On the heap: MSVC rejects a caller's local passed by reference down a recursion (docs/conventions.md).
		auto const owned = std::make_unique<array_store>();
		array_store &store = *owned;
		constexpr std::array<atl::uint32, 8> priorities{5, 1, 7, 3, 8, 2, 6, 4};
		atl::uint32 root = 0;
		for (atl::uint32 id = 1; id <= 8; ++id) {
			store.nodes[id] = {0, 0, 1, priorities[id - 1]};
			root = base::merge(store, root, id);
		}
		bool const is_merged = is_in_order(store, root, 1, 8);
		auto const [a, b] = base::split_at(store, root, 3);
		bool const is_split = is_in_order(store, a, 1, 3) && is_in_order(store, b, 4, 5);
		auto const random_draw = [&store] { return store.draw += 0x9e3779b97f4a7c15ULL; };
		auto const weighted = base::merge_weighted(store, a, b, std::ref(random_draw));
		bool const is_weighted = is_in_order(store, weighted, 1, 8);
		auto const [c, d] = base::split_at(store, weighted, 5);
		return is_merged && is_split && is_weighted && is_in_order(store, base::merge_spines(store, c, d), 1, 8);
	}());

	static_assert([] {
		using base = atl::detail::treap_base;
		// Every cut on both spine directions exercises the recursive result's two fields.
		for (bool is_left_spine : std::array{false, true}) {
			for (atl::uint32 k = 0; k <= 8; ++k) {
				auto const owned = std::make_unique<array_store>();
				array_store &store = *owned;
				atl::uint32 root = 0;
				for (atl::uint32 id = 1; id <= 8; ++id) {
					store.nodes[id] = {0, 0, 1, is_left_spine ? id : 9 - id};
					root = base::merge(store, root, id);
				}
				auto const parts = base::split_at(store, root, k);
				if (!is_in_order(store, parts.first, 1, k) || !is_in_order(store, parts.second, k + 1, 8 - k)) return false;
				root = base::merge(store, parts.first, parts.second);
				auto const keyed = base::split(store, root, [k](array_store &, atl::uint32 id) { return id <= k; });
				if (!is_in_order(store, keyed.first, 1, k) || !is_in_order(store, keyed.second, k + 1, 8 - k)) return false;
			}
		}
		return true;
	}());
}

namespace splay_engine_checks {
	struct sum_rope : atl::sized_splay_links {
		std::vector<long long> values{0}, sums{0};
		std::vector<bool> is_marked{false};

		constexpr atl::uint32 make(long long value) {
			values.push_back(value);
			sums.push_back(value);
			is_marked.push_back(false);
			return add_node();
		}

		constexpr long long const &value(atl::uint32 x) const { return values[x]; }

		constexpr long long const &aggregate(atl::uint32 x) const { return sums[x]; }

		constexpr void pull(atl::uint32 x) {
			pull_size(x);
			sums[x] = sums[child(x, 0)] + values[x] + sums[child(x, 1)];
		}

		constexpr void push(atl::uint32 x) {
			if (!is_marked[x]) return;
			auto const left = child(x, 0), right = child(x, 1);
			set_child(x, 0, right);
			set_child(x, 1, left);
			if (left != 0) is_marked[left] = !is_marked[left];
			if (right != 0) is_marked[right] = !is_marked[right];
			is_marked[x] = false;
		}
	};

	constexpr bool reversible_rope() {
		using engine = atl::splay_engine;
		sum_rope store;
		store.reserve(8);
		atl::uint32 root = 0;
		for (long long value = 1; value <= 6; ++value) root = engine::insert_at(store, root, atl::uint32(value - 1), store.make(value));
		auto const [a, bc] = engine::split_at(store, root, 1);
		auto const [b, c] = engine::split_at(store, bc, 4);
		store.is_marked[b] = true;
		root = engine::join(store, engine::join(store, a, b), c);
		engine::splay(store, 3);
		root = engine::erase(store, 3);
		std::vector<long long> order;
		engine::for_each(store, root, [&](atl::uint32 x) { order.push_back(store.values[x]); });
		auto const [top, length] = engine::max_right(store, engine::splay_at(store, root, 0), 0LL, std::plus<>{}, [](long long value) { return value <= 10; });
		return order == std::vector<long long>{1, 5, 4, 2, 6} && store.sums[top] == 18 && length == 3;
	}

	static_assert(reversible_rope());
	static_assert([] {
		atl::splay_links links;
		links.reserve(3);
		auto const a = links.add_node(), b = links.add_node(), c = links.add_node();
		std::array ids{a, b, c};
		auto root = atl::splay_engine::build_balanced(links, ids);
		atl::splay_engine::rotate(links, a);
		if (links.parent(a) != 0) return false;
		atl::splay_engine::splay_pushed(links, c);
		auto const [prefix, suffix] = atl::splay_engine::split_before(links, b);
		if (prefix != a || suffix != b) return false;
		root = atl::splay_engine::join(links, prefix, suffix);
		auto const [part, rest] = atl::splay_engine::split_after(links, b);
		return links.node_count() == 3 && part == b && rest == c && root != 0;
	}());
}

#include <atl/experimental/ds/splay_engine_variants.hpp>

namespace splay_engine_checks {
	static_assert([] {
		atl::experimental::reversible_fold_rope<long long, std::plus<>> store(0, std::plus<>{});
		auto const a = store.make(2), b = store.make(3);
		auto const root = atl::splay_engine::join(store, a, b);
		store.flip(root);
		store.push(root);
		store.pull(root);
		auto &scratch = store.splay_scratch();
		scratch.push_back(a);
		bool const is_sum = store.aggregate(root) == 5 && store.value(a) == 2;
		store.clear_reversed(root);
		return is_sum && !store.is_reversed(root) && scratch.front() == a;
	}());
}

namespace treap_proof {
	struct parent_store : array_store {
		std::array<atl::uint32, 9> parents{};

		constexpr atl::uint32 parent(atl::uint32 x) const { return parents[x]; }

		constexpr void set_parent(atl::uint32 x, atl::uint32 p) { parents[x] = p; }

		constexpr void pull(atl::uint32 x) { nodes[x].size = 1 + size(left(x)) + size(right(x)); }

		constexpr atl::uint32 rebind(atl::uint32 x, atl::uint32 l, atl::uint32 r) {
			array_store::rebind(x, l, r);
			parents[l] = parents[r] = x;
			return x;
		}
	};

	static_assert([] {
		using base = atl::detail::treap_base;
		parent_store store;
		for (atl::uint32 id = 1; id <= 4; ++id) store.nodes[id] = {0, 0, 1, 10 - id};
		auto root = base::merge(store, base::merge(store, 1, 2), 3);
		if (base::root_of(store, 3) != root) return false;
		auto const [a, b] = base::split_at_node(store, 2, atl::detail::treap_split_place::right);
		if (a != 1 || store.parent(a) != 0 || store.parent(b) != 0) return false;
		root = base::merge(store, a, b);
		base::insert_before_node(store, 3, 4);
		auto const [prefix, suffix] = base::split_at_node(store, 4, atl::detail::treap_split_place::excluded);
		return store.size(prefix) == 2 && store.size(suffix) == 1 && store.size(4) == 1
			&& base::root_of(store, 2) == prefix && root == 1;
	}());
}

namespace splay_engine_checks {
	using constexpr_rope = atl::experimental::reversible_fold_rope<int, std::plus<>>;

	template<typename Engine>
		requires atl::detail::splay_sequence_engine<Engine, constexpr_rope>
			  && requires(constexpr_rope &store) { Engine::push_path(store, atl::uint32{}); }
	constexpr bool strategy_case() {
		constexpr_rope store(0, std::plus<>{});
		auto const a = store.make(1), b = store.make(2), c = store.make(3);
		auto root = Engine::build_balanced(store, std::array{a, b, c});
		store.flip(root);
		Engine::push_path(store, c);
		Engine::splay(store, c);
		root = Engine::descend(store, c, [](atl::uint32) { return atl::splay_step::right; });
		int total = 0;
		Engine::for_each(store, root, [&total, &store](atl::uint32 x) { total += store.value(x); });
		if constexpr (requires { Engine::read_at(store, root, 0); }) {
			auto const [top, first] = Engine::read_at(store, root, 0);
			return total == 6 && store.value(first) == 3 && store.aggregate(top) == 6;
		} else {
			return total == 6 && store.aggregate(root) == 6;
		}
	}

	template<atl::experimental::splay_shape Shape>
	constexpr bool push_strategy_cases() {
		using atl::experimental::splay_engine_by_strategy;
		using atl::experimental::splay_path_push;
		return strategy_case<splay_engine_by_strategy<Shape, splay_path_push::ancestor_stack>>()
			&& strategy_case<splay_engine_by_strategy<Shape, splay_path_push::store_buffer>>()
			&& strategy_case<splay_engine_by_strategy<Shape, splay_path_push::pointer_reversal>>()
			&& strategy_case<splay_engine_by_strategy<Shape, splay_path_push::parity_walk>>();
	}

	static_assert(push_strategy_cases<atl::experimental::splay_shape::top_down>());
	static_assert(push_strategy_cases<atl::experimental::splay_shape::bottom_up>());
	static_assert(push_strategy_cases<atl::experimental::splay_shape::bottom_up_repush>());
	static_assert(push_strategy_cases<atl::experimental::splay_shape::semi>());
	static_assert(atl::detail::splay_selectable_layout<atl::splay_layout_interleaved>);
	static_assert(atl::detail::splay_selectable_layout<atl::splay_layout_split>);
	static_assert(!atl::detail::splay_selectable_layout<atl::detail::splay_layout_columnar>);
}

#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/ds/top_tree.hpp>
#include <atl/experimental/ds/euler_tour_tree_variants.hpp>

namespace link_cut_tree_checks {
	using atl::uint32;

	/** @brief Affine composition modulo 97 detects reversed path aggregates. */
	struct affine {
		long long a = 1, b = 0;
		constexpr bool operator==(affine const &) const = default;
	};

	struct affine_then {
		constexpr affine operator()(affine const &f, affine const &g) const { return {g.a * f.a % 97, (g.a * f.b + g.b) % 97}; }
	};

	constexpr bool forest_queries() {
		constexpr uint32 none = atl::graph::none;
		atl::link_cut_tree t(6);
		t.link(0, 1);
		t.link(1, 2);
		t.link(2, 3);
		t.link(5, 3);
		bool const is_rooted = t.root(0) == 3 && t.parent(0) == 1 && t.depth(0) == 3 && t.lca(0, 5) == 3
							&& t.distance(0, 5) == 4 && t.ancestor(0, 2) == 2 && t.path_vertex(0, 5, 3) == 3
							&& t.lca(0, 4) == none && !t.connected(0, 4);
		t.make_root(0);
		bool const is_rerooted = t.root(5) == 0 && t.depth(5) == 4 && t.parent(1) == 0;
		t.cut(2, 3);
		t.cut_from_parent(2);
		atl::link_cut_tree u(4);
		u.assign_parents(std::vector<uint32>{none, 0, 1, 1});
		return is_rooted && is_rerooted && t.root(5) == 3 && t.root(2) == 2 && !t.connected(1, 2) && u.lca(2, 3) == 1
			&& u.depth(3) == 2;
	}

	static_assert(forest_queries());

	constexpr bool path_folds() {
		std::vector<affine> const values{{2, 1}, {3, 4}, {5, 6}, {7, 8}};
		atl::link_cut_tree t(values, affine{}, affine_then{});
		t.link(0, 1);
		t.link(2, 1);
		t.link(3, 2);
		affine_then const combine;
		affine const forward = combine(combine(combine(values[0], values[1]), values[2]), values[3]);
		affine const backward = combine(combine(combine(values[3], values[2]), values[1]), values[0]);
		bool const is_ordered = t.fold(0, 3) == forward && t.fold(3, 0) == backward;
		t.make_root(1);
		t.set(2, affine{1, 1});
		return is_ordered && t.fold(0, 2) == combine(combine(values[0], values[1]), affine{1, 1}) && t.at(2) == affine{1, 1};
	}

	static_assert(path_folds());

	struct sized_sum {
		long long sum = 0, count = 0;
	};

	struct sized_add {
		constexpr sized_sum operator()(sized_sum x, sized_sum y) const { return {x.sum + y.sum, x.count + y.count}; }
	};

	struct add_to_each {
		constexpr sized_sum operator()(long long c, sized_sum x) const { return {x.sum + c * x.count, x.count}; }
	};

	struct larger {
		constexpr long long operator()(long long a, long long b) const { return a < b ? b : a; }
	};

	constexpr bool lazy_paths_and_edges() {
		atl::link_cut_tree t(std::vector<sized_sum>(4, {1, 1}), sized_sum{}, 0LL, sized_add{}, add_to_each{}, std::plus<>{});
		t.link(0, 1);
		t.link(1, 2);
		t.link(3, 1);
		t.apply(0, 2, 10);
		t.apply(3, 0, 1);
		bool const is_lazy = t.fold(2, 3).sum == 3 + 10 * 2 + 1 * 2 && t.at(0).sum == 12 && t.at(3).sum == 2;
		atl::link_cut_tree e(atl::edge_values, atl::commutative_fold, 4, 0LL, larger{});
		uint32 const e01 = e.link(0, 1, 5);
		e.link(1, 2, 9);
		e.link(3, 1, 2);
		bool const is_weighted = e.fold(0, 2) == 9 && e.fold(0, 3) == 5 && e.depth(0) == 2 && e.parent(3) == 1;
		e.cut(1, 2);
		e.cut(e01);
		bool const is_tried = e.try_fold(1, 3) == 2 && !e.try_fold(0, 3).has_value() && t.try_fold(0, 0).has_value();
		return is_lazy && is_weighted && is_tried && e.edge_count() == 1 && e.edge(3, 1) != atl::graph::none && !e.connected(0, 3);
	}

	static_assert(lazy_paths_and_edges());
}

namespace subtree_link_cut_tree_checks {
	constexpr bool subtree_queries() {
		atl::subtree_link_cut_tree t(std::vector<long long>{1, 2, 4, 8}, 0LL, std::plus<>{}, std::negate<>{});
		t.link(0, 1);
		t.link(2, 1);
		t.link(3, 1);
		bool const is_rooted = t.subtree_fold(1) == 15 && t.subtree_fold(2) == 4 && t.subtree_size(1) == 4
							&& t.subtree_fold(1, 2) == 11 && t.subtree_size(1, 3) == 3 && t.component_size(0) == 4;
		t.make_root(0);
		t.update(3, [](long long a) { return a + 16; });
		bool const is_rerooted = t.subtree_fold(1) == 2 + 4 + 24 && t.fold(0, 3) == 1 + 2 + 24 && t.subtree_fold(2, 1) == 4;
		t.cut(1, 2);
		atl::subtree_link_cut_tree u(std::vector<long long>{1, 1, 1}, 0LL, std::plus<>{}, std::negate<>{});
		u.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 0});
		return is_rooted && is_rerooted && t.component_size(0) == 3 && t.subtree_fold(2) == 4 && u.subtree_fold(0) == 3
			&& u.subtree_size(2, 0) == 1;
	}

	static_assert(subtree_queries());
}

namespace euler_tour_tree_checks {
	struct sized_sum {
		long long sum = 0, count = 0;
		constexpr bool operator==(sized_sum const &) const = default;
	};

	struct sized_add {
		constexpr sized_sum operator()(sized_sum const &x, sized_sum const &y) const { return {x.sum + y.sum, x.count + y.count}; }
	};

	struct add_to_each {
		constexpr sized_sum operator()(long long c, sized_sum const &x) const { return {x.sum + c * x.count, x.count}; }
	};

	constexpr bool tours() {
		atl::euler_tour_tree t(std::vector<sized_sum>{{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, sized_sum{}, 0LL,
							   sized_add{}, add_to_each{}, std::plus<>{});
		t.link(0, 1);
		t.link(2, 1);
		t.link(3, 0);
		bool ok = t.connected(2, 3) && !t.connected(4, 0) && t.component_size(1) == 4;
		t.subtree_apply(1, 0, 10);
		ok = ok && t.subtree_fold(0, 1) == sized_sum{5, 2} && t.subtree_fold(1, 0) == sized_sum{25, 2};
		t.cut(1, 0);
		t.link(4, 2);
		t.component_apply(4, 1);
		ok = ok && t.component_fold(2) == sized_sum{13 + 14 + 6, 3} && t.at(0) == sized_sum{1, 1};
		t.cut(2, 4);
		t.link(3, 2);
		return ok && t.component_size(0) == 4 && t.subtree_fold(2, 3) == sized_sum{14 + 13, 2};
	}

	static_assert(tours());

	constexpr bool structure_only() {
		atl::euler_tour_tree t(3);
		t.link(0, 1);
		bool const was_connected = t.connected(1, 0);
		t.cut(0, 1);
		return was_connected && !t.connected(0, 1) && t.component_size(2) == 1;
	}

	static_assert(structure_only());
}

namespace top_tree_checks {
	constexpr bool clusters() {
		atl::extremes_top_tree<long long> t(std::vector<long long>{1, 2, 4, 8});
		t.link(0, 1);
		t.link(2, 1);
		t.link(3, 1);
		t.subtree_add(1, 10);
		t.path_assign(0, 2, 5);
		bool const is_lazy = t.subtree_sum(1) == 5 + 5 + 5 + 18 && t.subtree_max(1, 2) == 18 && t.path_min(3, 0) == 5
						  && t.subtree_size(1) == 4 && t.root(0) == 1;
		t.make_root(3);
		t.cut(1, 2);
		bool const is_cut = t.component_sum(0) == 5 + 5 + 18 && t.component_size(2) == 1 && t.parent(0) == 1 && t.depth(0) == 2;
		atl::diameter_top_tree<long long> d(4);
		atl::uint32 const e01 = d.link(0, 1, 5);
		d.link(1, 2, 3);
		d.link(1, 3, 4);
		bool const is_diameter = d.diameter(2) == 9 && d.eccentricity(2) == 8 && d.farthest(3).vertex == 0 && d.path_length(2, 3) == 7;
		d.set(e01, 1);
		atl::marked_distance_top_tree<long long> m(3);
		m.link(0, 1, 2);
		m.link(1, 2, 3);
		m.set_marked(0, true);
		m.set_marked(2, true);
		auto compress = [](long long a, long long b) { return a + b; };
		auto add_vertex = [](long long light, long long v) { return light + v; };
		auto add_edge = [](long long child) { return child; };
		atl::top_tree g(3, 1LL, 0LL, 0LL, compress, compress, add_vertex, add_edge);
		g.link(0, 1);
		g.link(2, 1);
		return is_lazy && is_cut && is_diameter && d.diameter(0) == 7 && m.marked_distance_sum(1) == 5 && m.marked_count(0) == 2
			&& g.component_fold(0) == 3 && g.subtree_fold(0, 1) == 1;
	}

	static_assert(clusters());

	// Bulk construction reaches balanced rake builds through caller-local views.
	static_assert([] {
		atl::sum_top_tree<atl::int128> t(std::vector<atl::int128>{1, 2, 3, 4, 5});
		t.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 0, 0, 3});
		t.path_add(1, 4, 2);
		return t.component_sum(0) == 23 && t.subtree_sum(3) == 13;
	}());
}

namespace euler_tour_tree_u1_checks {
	using euler_tour_tree_checks::add_to_each;
	using euler_tour_tree_checks::sized_add;
	using euler_tour_tree_checks::sized_sum;

	template<typename Tree>
		requires std::constructible_from<Tree, std::vector<sized_sum>, sized_sum, long long, sized_add, add_to_each, std::plus<>>
			  && requires(Tree &t) {
					 t.link(0, 1);
					 t.cut(0, 1);
					 t.subtree_apply(0, 1, 1LL);
					 t.component_apply(0, 1LL);
					 { t.connected(0, 1) } -> std::same_as<bool>;
					 { t.component_size(0) } -> std::convertible_to<atl::uint32>;
					 { t.subtree_fold(0, 1) } -> std::same_as<sized_sum>;
					 { t.component_fold(0) } -> std::same_as<sized_sum>;
					 { t.at(0) } -> std::same_as<sized_sum>;
				 }
	constexpr bool tours() {
		Tree t(std::vector<sized_sum>{{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, sized_sum{}, 0LL, sized_add{}, add_to_each{}, std::plus<>{});
		t.link(0, 1);
		t.link(2, 1);
		t.link(3, 0);
		bool ok = t.connected(2, 3) && !t.connected(4, 0) && t.component_size(1) == 4;
		t.subtree_apply(1, 0, 10);
		ok = ok && t.subtree_fold(0, 1) == sized_sum{5, 2} && t.subtree_fold(1, 0) == sized_sum{25, 2};
		t.cut(1, 0);
		t.link(4, 2);
		t.component_apply(4, 1);
		ok = ok && t.component_fold(2) == sized_sum{13 + 14 + 6, 3} && t.at(0) == sized_sum{1, 1};
		t.cut(2, 4);
		t.link(3, 2);
		return ok && t.component_size(0) == 4 && t.subtree_fold(2, 3) == sized_sum{14 + 13, 2};
	}

	static_assert(tours<atl::euler_tour_tree<sized_sum, sized_add, long long, add_to_each, std::plus<>>>());
	static_assert(tours<atl::experimental::euler_tour_tree_by_splay<sized_sum, sized_add, long long, add_to_each, std::plus<>>>());

	template<typename Tree>
		requires std::constructible_from<Tree, std::size_t> && requires(Tree &t, std::vector<atl::uint32> const &parents) {
			t.assign_parents(parents);
			t.cut(0, 1);
			{ t.connected(0, 1) } -> std::same_as<bool>;
			{ t.component_size(0) } -> std::convertible_to<atl::uint32>;
		}
	constexpr bool structure_only() {
		Tree t(4);
		t.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 0, 2});
		bool const was_connected = t.connected(3, 1) && t.component_size(0) == 4;
		t.cut(0, 2);
		return was_connected && !t.connected(0, 3) && t.component_size(2) == 2;
	}

	static_assert(structure_only<atl::euler_tour_tree<>>());
	static_assert(structure_only<atl::experimental::euler_tour_tree_by_splay<>>());
}

namespace dynamic_forest_int128_checks {
	static_assert([] {
		std::vector<atl::int128> values{1, 2, 4};
		atl::link_cut_tree path(values, atl::int128{}, std::plus<>{});
		atl::subtree_link_cut_tree subtree(values, atl::int128{}, std::plus<>{}, std::negate<>{});
		atl::euler_tour_tree tour(values, atl::int128{}, std::plus<>{});
		std::array<atl::uint32, 3> parents{atl::graph::none, 0, 0};
		path.assign_parents(parents);
		subtree.assign_parents(parents);
		tour.assign_parents(parents);
		path.update(0, [](atl::int128 value) { return value + 8; });
		return path.fold(1, 2) == 15 && subtree.subtree_fold(0) == 7 && tour.component_fold(0) == 7;
	}());
}

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/ds/persistent_heap.hpp>
#include <atl/experimental/ds/meldable_heap_variants.hpp>

namespace publication_heap_checks {
	template<typename Forest>
		requires std::constructible_from<Forest, std::size_t> && requires(Forest &forest) {
			forest.push(0, 1LL);
			forest.add_to_all(0, 1LL);
			forest.meld(0, 1);
			{ forest.pop(0) } -> std::convertible_to<long long>;
			{ forest.top(0) } -> std::convertible_to<long long>;
			{ forest.empty(0) } -> std::convertible_to<bool>;
			{ forest.size(0) } -> std::convertible_to<std::size_t>;
		}
	constexpr bool heap_forest() {
		Forest forest(3);
		forest.push(0, 5);
		forest.push(0, 1);
		forest.push(1, 4);
		forest.add_to_all(0, 10);
		forest.meld(1, 0);
		forest.push(2, 7);
		forest.meld(2, 1);
		if (forest.size(2) != 4 || !forest.empty(0) || forest.top(2) != 15) return false;
		long long const first = forest.pop(2);
		forest.add_to_all(2, -1);
		return first == 15 && forest.pop(2) == 10 && forest.pop(2) == 6 && forest.pop(2) == 3 && forest.empty(2);
	}

	static_assert(heap_forest<atl::meldable_heap_forest<long long, std::less<>, long long>>());
	static_assert(heap_forest<atl::experimental::meldable_heap_forest_by_rank_leftist<long long, std::less<>, long long>>());
	static_assert(heap_forest<atl::experimental::meldable_heap_forest_by_skew<long long, std::less<>, long long>>());
	static_assert(heap_forest<atl::experimental::meldable_heap_forest_by_pairing<long long, std::less<>, long long>>());
	static_assert(heap_forest<atl::experimental::meldable_heap_forest_by_split_pairing<long long, std::less<>, long long>>());
	static_assert(heap_forest<atl::experimental::meldable_heap_forest_by_binary<long long, std::less<>, long long>>());

	static_assert([] {
		atl::addressable_meldable_heap_forest<long long> forest(2);
		auto const x = forest.push(0, 5);
		auto const y = forest.push(1, 3);
		forest.push(1, 4);
		forest.meld(1, 0);
		forest.promote(1, y, 9);
		return forest.top(1) == 9 && forest.value(y) == 9 && forest.erase(1, x) == 5 && forest.pop(1) == 9
			&& forest.pop(1) == 4 && forest.empty(1);
	}());

	struct point {
		long long key, label;
	};

	static_assert([] {
		auto const later = [](point const &a, point const &b) { return a.key > b.key; };
		auto const move = [](long long shift, point p) { p.key += shift; return p; };
		auto const combine = [](long long newer, long long older) { return newer + older; };
		atl::meldable_heap_forest<point, decltype(later), long long, decltype(move), decltype(combine)> forest(1, later, 0, move, combine);
		forest.reserve(4);
		std::size_t const second = forest.add_heap();
		forest.emplace(0, 5, 1);
		forest.emplace(atl::uint32(second), 2, 2);
		forest.add_to_all(0, -4);
		forest.meld(atl::uint32(second), 0);
		point const first = forest.pop(atl::uint32(second));
		return forest.heap_count() == 2 && first.key == 1 && first.label == 1 && forest.top(atl::uint32(second)).label == 2;
	}());

	static_assert([] {
		atl::meldable_heap_forest<int, std::less<>, int> forest(2);
		forest.release(0);
		forest.push(0, 2);
		forest.push(0, 4);
		forest.add_to_all(0, 10);
		forest.push(1, 7);
		forest.meld(0, 1);
		forest.release(0);
		forest.push(0, 3);
		return forest.empty(1) && forest.size(0) == 1 && forest.pop(0) == 3 && forest.empty(0);
	}());
}

namespace persistent_heap_forest_checks {
	static_assert([] {
		atl::persistent_heap_forest<int> forest;
		auto const e = forest.empty_heap;
		auto const a = forest.push(forest.push(forest.push(e, 3), 9), 1);
		auto const b = forest.push(forest.push(e, 7), 5);
		auto const m = forest.meld(a, b);
		auto const c = forest.pop(m);
		auto const d = forest.push_range(c, std::vector<int>{8, 2});
		if (forest.top(a) != 9 || forest.top(b) != 7 || forest.top(m) != 9 || forest.top(c) != 7 || forest.top(d) != 8) return false;
		auto h = d;
		for (int expected : {8, 7, 5, 3, 2, 1}) {
			if (forest.empty(h) || forest.top(h) != expected) return false;
			h = forest.pop(h);
		}
		return forest.empty(h) && forest.top(a) == 9;
	}());

	static_assert([] {
		for (int const n : {0, 1, 2, 3, 4, 7, 8, 9, 16, 17}) {
			atl::persistent_heap_forest<int, std::greater<>> forest;
			std::vector<int> values;
			for (int i = n; i > 0; --i) values.push_back(i);
			auto const base = forest.push(forest.empty_heap, -1);
			auto unsized = values | std::views::filter([](int) { return true; });
			auto h = forest.push_range(base, unsized);
			if (forest.top(h) != -1) return false;
			h = forest.pop(h);
			for (int i = 1; i <= n; ++i) {
				if (forest.empty(h) || forest.top(h) != i) return false;
				h = forest.pop(h);
			}
			if (!forest.empty(h) || forest.top(base) != -1) return false;
		}
		return true;
	}());

	static_assert([] {
		atl::persistent_heap_forest<atl::int128> forest;
		auto const a = forest.push_range(forest.empty_heap, std::array<atl::int128, 3>{1, atl::int128(1) << 90, 3});
		return forest.top(a) == (atl::int128(1) << 90) && forest.top(forest.pop(a)) == 3;
	}());
}

namespace algebra_forwarding_checks {
	struct reference_max {
		constexpr int const &operator()(int const &a, int const &b) const { return a >= b ? a : b; }

		void operator()(int &, int &) const = delete;
	};

	struct value_category {
		constexpr int operator()(int const &a, int const &b) const { return a + b; }

		constexpr int operator()(int &&a, int &&b) const { return a + b + 10; }
	};

	struct copy_only {
		constexpr copy_only() = default;

		constexpr copy_only(copy_only const &) = default;

		copy_only(copy_only &&) = delete;
	};

	static_assert([] {
		int left = 3, right = 2;
		copy_only callable;
		copy_only copied(atl::detail::move_or_copy(callable));
		(void)copied;
		return &atl::detail::invoke_algebra<int>(reference_max{}, left, right) == &left
			&& atl::detail::invoke_algebra<int>(value_category{}, 1, 2) == 13
			&& atl::detail::invoke_algebra<int>(value_category{}, left, right) == 5;
	}());
}
