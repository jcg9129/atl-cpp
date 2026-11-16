#include <atl/ds/coordinate_compression.hpp>
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
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
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
