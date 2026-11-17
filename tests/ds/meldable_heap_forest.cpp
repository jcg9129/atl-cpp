// The certifying oracle for the whole family (library forest and every
// experimental contender) is the unsorted vector per heap in
// check_against_oracle / check_handles, not any one implementation.
#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/experimental/ds/meldable_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint8;

namespace {
	/** @brief A slope-trick breakpoint: ordered by position only, a lazy add moves the position. */
	struct breakpoint {
		int64 position, weight;

		breakpoint &operator+=(int64 shift) {
			position += shift;
			return *this;
		}

		friend bool operator==(breakpoint const &, breakpoint const &) = default;
	};

	/** @brief Orders breakpoints by position; equal positions are equivalent whatever their weights. */
	struct by_position {
		bool operator()(breakpoint const &a, breakpoint const &b) const { return a.position < b.position; }
	};

	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using library = atl::meldable_heap_forest<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using leftist = atl::experimental::meldable_heap_forest_by_rank_leftist<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using recursive_leftist = atl::experimental::meldable_heap_forest_by_rank_leftist<T, Compare, Shift, Mapping, Compose, uint8, true>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using wide_rank_leftist = atl::experimental::meldable_heap_forest_by_rank_leftist<T, Compare, Shift, Mapping, Compose, uint32>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using skew = atl::experimental::meldable_heap_forest_by_skew<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using branch_light_skew = atl::experimental::meldable_heap_forest_by_skew<T, Compare, Shift, Mapping, Compose, true>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using pairing = atl::experimental::meldable_heap_forest_by_pairing<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using branch_light_pairing = atl::experimental::meldable_heap_forest_by_pairing<T, Compare, Shift, Mapping, Compose, true>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using non_recycling_pairing = atl::experimental::meldable_heap_forest_by_pairing<T, Compare, Shift, Mapping, Compose, false, false>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using split_pairing = atl::experimental::meldable_heap_forest_by_split_pairing<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using binary = atl::experimental::meldable_heap_forest_by_binary<T, Compare, Shift, Mapping, Compose>;
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	using bulk_binary = atl::experimental::meldable_heap_forest_by_binary<T, Compare, Shift, Mapping, Compose, true>;

	/**
	 * @brief A keyed item with no operators at all, shifted through lambdas:
	 *        the lazy add moves `key` by a @ref move_by.
	 */
	struct item {
		int64 key, label;
	};

	/** @brief A shift with no operators: move an item's key by `delta`. */
	struct move_by {
		int64 delta;
	};

	/** @brief Orders items by key. */
	struct by_key {
		bool operator()(item const &a, item const &b) const { return a.key < b.key; }
	};

	using add = atl::shift_by_addition;
	using compose = atl::compose_by_addition;
	inline constexpr auto move_item = [](move_by const &shift, item value) {
		value.key += shift.delta;
		return value;
	};
	inline constexpr auto compose_moves = [](move_by const &newer, move_by const &older) {
		return move_by{newer.delta + older.delta};
	};
	using move_item_t = std::remove_const_t<decltype(move_item)>;
	using compose_moves_t = std::remove_const_t<decltype(compose_moves)>;

	/** @brief Whether two elements are the same object state (the oracle's lookup). */
	bool is_identical(int64 a, int64 b) { return a == b; }

	bool is_identical(breakpoint const &a, breakpoint const &b) { return a == b; }

	bool is_identical(item const &a, item const &b) { return a.key == b.key && a.label == b.label; }

	/** @brief A random element: short_values positions so that ties are common. */
	template<typename T>
		requires std::same_as<T, int64> || std::same_as<T, breakpoint> || std::same_as<T, item>
	T random_element(std::mt19937_64 &rng, int64 range) {
		int64 const position = int64(rng() % std::uint64_t(range)) - range / 2;
		if constexpr (std::is_same_v<T, breakpoint>)
			return breakpoint{position, int64(rng() % 1000)};
		else if constexpr (std::is_same_v<T, item>)
			return item{position, int64(rng() % 1000)};
		else
			return position;
	}

	/**
	 * @brief Random push / pop / meld / add-to-all traffic over a few heaps
	 *        against an unsorted vector per heap: after every operation each
	 *        touched heap's size, and its top up to equivalence, must match;
	 *        a popped element must be one the oracle holds. With a lazy add the
	 *        forest is built by the explicit-identity constructor and the
	 *        oracle applies the same `Mapping`.
	 */
	template<template<typename, typename, typename, typename, typename> typename Forest, typename T, typename Compare,
			 typename Shift, typename Mapping = add, typename Compose = compose>
		requires atl::meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	void check_against_oracle(std::uint64_t seed, std::size_t heap_count, std::size_t steps, int64 range) {
		std::mt19937_64 rng(seed);
		using forest_type = Forest<T, Compare, Shift, Mapping, Compose>;
		forest_type forest = [&] {
			if constexpr (std::is_void_v<Shift>)
				return forest_type(heap_count);
			else
				return forest_type(heap_count, Compare{}, Shift{0}, Mapping{}, Compose{});
		}();
		std::vector<std::vector<T>> oracle(heap_count);
		Compare const comp{};
		auto const equivalent = [&](T const &a, T const &b) { return !comp(a, b) && !comp(b, a); };
		auto const check_heap = [&](std::size_t h) {
			REQUIRE(forest.size(uint32(h)) == oracle[h].size());
			REQUIRE(forest.empty(uint32(h)) == oracle[h].empty());
			if (!oracle[h].empty()) REQUIRE(equivalent(forest.top(uint32(h)), std::ranges::max(oracle[h], comp)));
		};
		for (std::size_t step = 0; step < steps; ++step) {
			std::size_t const h = rng() % forest.heap_count();
			unsigned const kind = unsigned(rng() % 10);
			if (kind < 4) {
				T value = random_element<T>(rng, range);
				oracle[h].push_back(value);
				if (kind == 0)
					forest.emplace(uint32(h), std::move(value));
				else
					forest.push(uint32(h), std::move(value));
			} else if (kind < 7) {
				if (oracle[h].empty()) continue;
				T const popped = forest.pop(uint32(h));
				REQUIRE(equivalent(popped, std::ranges::max(oracle[h], comp)));
				auto const it = std::ranges::find_if(oracle[h], [&](T const &x) { return is_identical(x, popped); });
				REQUIRE(it != oracle[h].end());
				oracle[h].erase(it);
			} else if (kind < 9) {
				std::size_t const from = rng() % forest.heap_count();
				forest.meld(uint32(h), uint32(from));
				if (from != h) {
					oracle[h].insert(oracle[h].end(), oracle[from].begin(), oracle[from].end());
					oracle[from].clear();
					check_heap(from);
				}
			} else if constexpr (!std::is_void_v<Shift>) {
				Shift const shift{int64(rng() % std::uint64_t(range)) - range / 2};
				forest.add_to_all(uint32(h), shift);
				for (T &value : oracle[h]) value = Mapping{}(shift, value);
			} else if (rng() % 8 == 0) {
				oracle.emplace_back();
				REQUIRE(forest.add_heap() == oracle.size() - 1);
			}
			check_heap(h);
		}
		for (std::size_t h = 0; h < forest.heap_count(); ++h) {
			std::ranges::sort(oracle[h], comp);
			while (!oracle[h].empty()) {
				REQUIRE(equivalent(forest.pop(uint32(h)), oracle[h].back()));
				oracle[h].pop_back();
			}
			REQUIRE(forest.empty(uint32(h)));
		}
	}

	/** @brief Every element type / order / lazy-add combination, many seeds, a few heap counts. */
	template<template<typename, typename, typename, typename, typename> typename Forest, bool HasLambdaShift = true>
	void check_all() {
		for (std::uint64_t seed = 1; seed <= 30; ++seed) {
			std::size_t const heap_count = 1 + seed % 5;
			check_against_oracle<Forest, int64, std::less<>, void>(seed, heap_count, 1500, 50);
			check_against_oracle<Forest, int64, std::greater<>, void>(seed, heap_count, 1500, 1'000'000);
			check_against_oracle<Forest, int64, std::less<>, int64>(seed, heap_count, 1500, 40);
			check_against_oracle<Forest, int64, std::greater<>, int64>(seed, heap_count, 1500, 1'000);
			check_against_oracle<Forest, breakpoint, by_position, int64>(seed, heap_count, 1500, 30);
			if constexpr (HasLambdaShift)
				check_against_oracle<Forest, item, by_key, move_by, move_item_t, compose_moves_t>(seed, heap_count, 1500,
																								  30);
		}
	}

	/**
	 * @brief Pushes `2, -1, 4, -2, ..., 2n, -n`, then one element below them
	 *        all: in a skew heap the last push walks a right spine of `n`
	 *        nodes (each pair adds one), deep enough to overflow the stack of
	 *        a recursive meld at n = 10^6. Then everything pops in order.
	 */
	template<template<typename, typename, typename, typename, typename> typename Forest>
	void check_long_spine(int64 n) {
		Forest<int64, std::less<>, void, add, compose> forest(1);
		for (int64 i = 1; i <= n; ++i) {
			forest.push(0, 2 * i);
			forest.push(0, -i);
		}
		forest.push(0, -n - 1);
		REQUIRE(forest.size(0) == std::size_t(2 * n + 1));
		for (int64 i = n; i >= 1; --i) REQUIRE(forest.pop(0) == 2 * i);
		for (int64 i = 1; i <= n + 1; ++i) REQUIRE(forest.pop(0) == -i);
		REQUIRE(forest.empty(0));
	}

	/** @brief Empty and single-element cases, self-meld, melding empty heaps, add-to-all on an empty heap. */
	template<template<typename, typename, typename, typename, typename> typename Forest>
	void check_edges() {
		Forest<int64, std::less<>, int64, add, compose> forest;
		REQUIRE(forest.heap_count() == 0);
		REQUIRE(forest.add_heap() == 0);
		REQUIRE(forest.add_heap() == 1);
		REQUIRE(forest.empty(0));
		forest.meld(0, 1);
		REQUIRE(forest.empty(0));
		forest.add_to_all(0, 5); // nothing to move; later pushes are unaffected
		forest.push(0, 7);
		REQUIRE(forest.top(0) == 7);
		forest.meld(0, 0);
		REQUIRE(forest.size(0) == 1);
		forest.meld(1, 0);
		REQUIRE(forest.empty(0));
		REQUIRE(forest.size(1) == 1);
		forest.add_to_all(1, -10);
		REQUIRE(forest.top(1) == -3);
		REQUIRE(forest.pop(1) == -3);
		REQUIRE(forest.empty(1));
		forest.push(1, 1); // reuses the freed node
		REQUIRE(forest.top(1) == 1);
	}

	/** @brief The asserted preconditions: an empty heap's top / pop, an out-of-range id. */
	template<template<typename, typename, typename, typename, typename> typename Forest>
	void check_deaths() {
		CHECK(atl_test::aborts([] {
			Forest<int64, std::less<>, void, add, compose> forest(2);
			(void)forest.top(1);
		}));
		CHECK(atl_test::aborts([] {
			Forest<int64, std::less<>, void, add, compose> forest(2);
			forest.pop(0);
		}));
		CHECK(atl_test::aborts([] {
			Forest<int64, std::less<>, void, add, compose> forest(2);
			forest.push(2, 1);
		}));
		CHECK(atl_test::aborts([] {
			Forest<int64, std::less<>, void, add, compose> forest(2);
			forest.meld(0, 2);
		}));
		using forest_type = Forest<int64, std::less<>, void, add, compose>;
		if constexpr (sizeof(std::size_t) > sizeof(uint32) && !std::same_as<forest_type, atl::meldable_heap_forest<int64>>) {
			std::size_t const invalid = std::size_t((std::numeric_limits<uint32>::max)()) + std::size_t(1);
			CHECK(atl_test::aborts([] {
				forest_type forest(1);
				forest.push(invalid, 7);
			}));
			CHECK(atl_test::aborts([] {
				forest_type forest(1);
				forest.meld(0, invalid);
			}));
		}
	}

	template<template<typename, typename, typename, typename, typename> typename Forest, bool HasLambdaShift = true>
	void check_everything() {
		check_all<Forest, HasLambdaShift>();
		check_long_spine<Forest>(1000);
		check_edges<Forest>();
		check_deaths<Forest>();
	}
}

TEST_CASE("meldable_heap_forest matches the oracle", "[meldable_heap_forest]") { check_everything<library>(); }

TEST_CASE("meldable_heap_forest: slope-trick breakpoints shifted through lambdas", "[meldable_heap_forest]") {
	// min-heap of (position, weight) breakpoints, the lazy add moving positions only
	auto const move = [](int64 shift, item p) {
		p.key += shift;
		return p;
	};
	auto const combine = [](int64 newer, int64 older) { return newer + older; };
	atl::meldable_heap_forest heaps_of_breakpoints = [&] {
		auto const later = [](item const &a, item const &b) { return a.key > b.key; };
		return atl::meldable_heap_forest<item, decltype(later), int64, decltype(move), decltype(combine)>(
			3, later, 0, move, combine);
	}();
	heaps_of_breakpoints.push(1, item{5, 2});
	heaps_of_breakpoints.emplace(1, 3, 1);
	heaps_of_breakpoints.push(2, item{4, 7});
	heaps_of_breakpoints.add_to_all(1, 10); // {15, 13}
	heaps_of_breakpoints.meld(2, 1);		// {4, 13, 15}
	heaps_of_breakpoints.add_to_all(2, -1); // {3, 12, 14}
	heaps_of_breakpoints.push(2, item{0, 9});
	REQUIRE(heaps_of_breakpoints.size(2) == 4);
	std::vector<std::pair<int64, int64>> popped;
	while (!heaps_of_breakpoints.empty(2)) {
		item const p = heaps_of_breakpoints.pop(2);
		popped.emplace_back(p.key, p.label);
	}
	REQUIRE(popped == std::vector<std::pair<int64, int64>>{{0, 9}, {3, 7}, {12, 1}, {14, 2}});
}

namespace {
	/** @brief Orders items by key, larger key worse: a min-heap. */
	struct by_key_descending {
		bool operator()(item const &a, item const &b) const { return a.key > b.key; }
	};

	/**
	 * @brief Random push / pop / meld / promote / erase traffic with handles
	 *        on the addressable forest against a (handle, item) list
	 *        per heap; labels are unique, so a pop identifies its handle.
	 */
	template<std::strict_weak_order<item const &, item const &> Compare>
	void check_handles(std::uint64_t seed, std::size_t heap_count, std::size_t steps, int64 range) {
		std::mt19937_64 rng(seed);
		using forest_type = atl::addressable_meldable_heap_forest<item, Compare>;
		using handle = typename forest_type::handle;
		forest_type forest(heap_count);
		std::vector<std::vector<std::pair<handle, item>>> oracle(heap_count);
		Compare const comp{};
		int64 next_label = 0;
		auto const best = [&](std::size_t h) {
			return std::ranges::max(oracle[h], comp, [](auto const &entry) -> item const & { return entry.second; });
		};
		auto const equivalent = [&](item const &a, item const &b) { return !comp(a, b) && !comp(b, a); };
		auto const check_heap = [&](std::size_t h) {
			REQUIRE(forest.size(uint32(h)) == oracle[h].size());
			if (!oracle[h].empty()) REQUIRE(equivalent(forest.top(uint32(h)), best(h).second));
			for (auto const &[x, element] : oracle[h]) REQUIRE(is_identical(forest.value(x), element));
		};
		auto const remove_label = [&](std::size_t h, int64 label) {
			auto const it = std::ranges::find_if(oracle[h], [&](auto const &entry) { return entry.second.label == label; });
			REQUIRE(it != oracle[h].end());
			oracle[h].erase(it);
		};
		for (std::size_t step = 0; step < steps; ++step) {
			std::size_t const h = rng() % heap_count;
			unsigned const kind = unsigned(rng() % 12);
			if (kind < 4) {
				item const element{int64(rng() % std::uint64_t(range)), next_label++};
				handle const x = kind == 0 ? forest.emplace(uint32(h), element.key, element.label) : forest.push(uint32(h), element);
				oracle[h].emplace_back(x, element);
			} else if (kind < 6) {
				if (oracle[h].empty()) continue;
				item const popped = forest.pop(uint32(h));
				REQUIRE(equivalent(popped, best(h).second));
				remove_label(h, popped.label);
			} else if (kind < 8) {
				std::size_t const from = rng() % heap_count;
				forest.meld(uint32(h), uint32(from));
				if (from != h) {
					oracle[h].insert(oracle[h].end(), oracle[from].begin(), oracle[from].end());
					oracle[from].clear();
				}
			} else if (kind < 10) {
				if (oracle[h].empty()) continue;
				auto &[x, element] = oracle[h][rng() % oracle[h].size()];
				int64 const step_size = int64(rng() % std::uint64_t(range / 2 + 1));
				item promoted{element.key + step_size, element.label};
				if (comp(promoted, element)) promoted.key = element.key - step_size;
				forest.promote(uint32(h), x, promoted);
				element = promoted;
			} else {
				if (oracle[h].empty()) continue;
				std::size_t const i = rng() % oracle[h].size();
				REQUIRE(is_identical(forest.erase(uint32(h), oracle[h][i].first), oracle[h][i].second));
				oracle[h].erase(oracle[h].begin() + std::ptrdiff_t(i));
			}
			check_heap(h);
		}
		for (std::size_t h = 0; h < heap_count; ++h) {
			while (!oracle[h].empty()) {
				item const popped = forest.pop(uint32(h));
				REQUIRE(equivalent(popped, best(h).second));
				remove_label(h, popped.label);
			}
			REQUIRE(forest.empty(uint32(h)));
		}
	}
}

TEST_CASE("meldable_heap_forest: promote and erase by handle match the oracle", "[meldable_heap_forest]") {
	for (std::uint64_t seed = 1; seed <= 40; ++seed) {
		check_handles<by_key>(seed, 1 + seed % 4, 2000, 40);
		check_handles<by_key_descending>(seed, 1 + seed % 4, 2000, 1'000'000);
	}
}

TEST_CASE("meldable_heap_forest: handle preconditions abort", "[meldable_heap_forest]") {
	using forest_type = atl::addressable_meldable_heap_forest<int64>;
	CHECK(atl_test::aborts([] {
		forest_type forest(1);
		auto const x = forest.push(0, 5);
		forest.promote(0, x, 4); // worse under std::less<>: not toward the top
	}));
	CHECK(atl_test::aborts([] {
		forest_type forest(1);
		auto const x = forest.push(0, 5);
		forest.push(0, 7);
		forest.pop(0);
		forest.pop(0);
		forest.erase(0, x); // already popped
	}));
	CHECK(atl_test::aborts([] {
		forest_type forest(1);
		auto const x = forest.push(0, 5);
		forest.erase(0, x);
		(void)forest.value(x);
	}));
	CHECK(atl_test::aborts([] {
		forest_type forest(1);
		forest.push(0, 5);
		forest.promote(0, 0, 9); // 0 is never a handle
	}));
	forest_type forest(2);
	auto const x = forest.push(0, 5);
	auto const y = forest.push(1, 3);
	forest.meld(1, 0); // handles survive the meld
	forest.promote(1, y, 8);
	REQUIRE(forest.top(1) == 8);
	REQUIRE(forest.erase(1, x) == 5);
	REQUIRE(forest.value(y) == 8);
	REQUIRE(forest.size(1) == 1);
}

TEST_CASE("leftist heap forest matches the oracle", "[meldable_heap_forest]") {
	check_everything<leftist>();
	check_everything<recursive_leftist>();
	check_everything<wide_rank_leftist>();
}

TEST_CASE("skew heap forest matches the oracle", "[meldable_heap_forest]") {
	check_everything<skew>();
	check_everything<branch_light_skew>();
}

TEST_CASE("skew heap meld survives a right spine of 10^6 nodes", "[meldable_heap_forest]") {
	check_long_spine<skew>(1'000'000);
	check_long_spine<branch_light_skew>(1'000'000);
}

TEST_CASE("pairing heap forest matches the oracle", "[meldable_heap_forest]") {
	check_everything<pairing>();
	check_everything<branch_light_pairing>();
	check_everything<non_recycling_pairing>();
	check_everything<split_pairing>();
}

TEST_CASE("binary heap forest matches the oracle", "[meldable_heap_forest]") {
	// the baseline's per-heap offset needs an inverse shift (`-shift`), which
	// move_by lacks, so it skips the operator-free item
	check_everything<binary, false>();
	check_everything<bulk_binary, false>();
}

TEST_CASE("pairing heap forest: release empties a heap and recycles its nodes", "[meldable_heap_forest]") {
	using heap_forest = atl::meldable_heap_forest<int64, std::less<>, int64>;
	auto const seed = GENERATE(take(12, random(atl::uint64(0), (std::numeric_limits<atl::uint64>::max)())));
	std::mt19937_64 rng(seed);
	heap_forest::heap_id const heaps = heap_forest::heap_id(1 + seed % 6);
	heap_forest forest(heaps);
	std::vector<std::multiset<int64>> oracle(heaps);
	for (int step = 0; step < 1500; ++step) {
		heap_forest::heap_id const a = heap_forest::heap_id(rng() % heaps), b = heap_forest::heap_id(rng() % heaps);
		switch (rng() % 7) {
		case 0:
		case 1:
		case 2: {
			int64 const v = int64(rng() % 100);
			forest.push(a, v);
			oracle[a].insert(v);
			break;
		}
		case 3:
			if (!oracle[a].empty()) {
				REQUIRE(forest.pop(a) == *oracle[a].rbegin());
				oracle[a].erase(std::prev(oracle[a].end()));
			}
			break;
		case 4:
			forest.meld(a, b);
			if (a != b) oracle[a].merge(oracle[b]);
			break;
		case 5: {
			int64 const d = int64(rng() % 7) - 3;
			forest.add_to_all(a, d);
			std::multiset<int64> moved;
			for (int64 v : oracle[a]) moved.insert(v + d);
			oracle[a] = std::move(moved);
			break;
		}
		default:
			forest.release(a);
			oracle[a].clear();
		}
		REQUIRE(forest.size(a) == oracle[a].size());
		if (!oracle[a].empty()) REQUIRE(forest.top(a) == *oracle[a].rbegin());
	}
	for (std::size_t h = 0; h < heaps; ++h)
		while (!oracle[h].empty()) {
			REQUIRE(forest.pop(uint32(h)) == *oracle[h].rbegin());
			oracle[h].erase(std::prev(oracle[h].end()));
		}
}

TEST_CASE("meldable_heap_forest: release handles empty, star, chain and recycled nodes", "[meldable_heap_forest]") {
	for (int const n : {0, 1, 2, 3, 31, 32, 33, 10000}) {
		for (bool const is_increasing : {false, true}) {
			atl::meldable_heap_forest<int, std::less<>, int> forest(2);
			forest.push(1, 99);
			for (int i = 0; i < n; ++i) forest.push(0, is_increasing ? i : -i);
			forest.add_to_all(0, 7);
			forest.release(0);
			forest.release(0);
			REQUIRE(forest.empty(0));
			REQUIRE(forest.size(0) == 0);
			REQUIRE(forest.top(1) == 99);
			for (int i = 0; i < n; ++i) forest.push(0, i);
			for (int i = n; i > 0; --i) REQUIRE(forest.pop(0) == i - 1);
			REQUIRE(forest.empty(0));
		}
	}
	atl::addressable_meldable_heap_forest<int> forest(2);
	auto const a = forest.push(0, 1);
	auto const b = forest.push(1, 9);
	forest.release(0);
	CHECK(atl_test::aborts([&] { (void)forest.value(a); }));
	REQUIRE(forest.value(b) == 9);
	auto const reused = forest.push(0, 3);
	forest.promote(0, reused, 5);
	REQUIRE(forest.erase(0, reused) == 5);
	CHECK(atl_test::aborts([&] { forest.release(2); }));
}

TEST_CASE("meldable self-meld validates its heap id", "[meldable_heap_forest]") {
	REQUIRE(atl_test::aborts([] {
		atl::meldable_heap_forest<int> forest(1);
		forest.meld(1, 1);
	}));
}
