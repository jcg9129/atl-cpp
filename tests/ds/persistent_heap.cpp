// The certifying oracle for the whole family (library class and every
// experimental contender) is the std::multiset copied per version in
// check_class_form / check_forest_form, not any one implementation.
#include <atl/ds/persistent_heap.hpp>
#include <atl/experimental/ds/persistent_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;
using algorithm = ex::persistent_heap_algorithm;

namespace {
	/** @brief The oracle's top: the comp-largest element of a multiset ordered by `Compare`. */
	template<std::strict_weak_order<int64, int64> Compare>
	int64 oracle_top(std::multiset<int64, Compare> const &s) {
		return *s.rbegin();
	}

	/** @brief What a node walk saw: element count and null-path length. */
	struct walk_result {
		std::size_t count = 0;
		int npl = 0;
	};

	/**
	 * @brief Walk a heap tree through `left` / `right` / `value` callables:
	 *        checks heap order at every node and, by `Algorithm`, the leftist
	 *        invariant (by null-path length or by size); collects the values.
	 */
	template<algorithm Algorithm, std::strict_weak_order<int64, int64> Compare, std::copyable Node,
			 std::regular_invocable<Node> Left, std::regular_invocable<Node> Right, std::regular_invocable<Node> Value,
			 std::predicate<Node> IsEmpty>
	walk_result walk(Node x, Left const &left, Right const &right, Value const &value, IsEmpty const &is_empty,
					 std::multiset<int64, Compare> &seen) {
		if (is_empty(x)) return {};
		seen.insert(value(x));
		Compare comp;
		Node const l = left(x), r = right(x);
		if (!is_empty(l)) REQUIRE_FALSE(comp(value(x), value(l)));
		if (!is_empty(r)) REQUIRE_FALSE(comp(value(x), value(r)));
		walk_result const a = walk<Algorithm, Compare>(l, left, right, value, is_empty, seen);
		walk_result const b = walk<Algorithm, Compare>(r, left, right, value, is_empty, seen);
		if constexpr (Algorithm == algorithm::rank_leftist) REQUIRE(a.npl >= b.npl);
		if constexpr (Algorithm == algorithm::weight_biased) REQUIRE(a.count >= b.count);
		return {a.count + b.count + 1, (std::min)(a.npl, b.npl) + 1};
	}

	/**
	 * @brief A random version DAG on a class-form heap: each step picks one
	 *        or two existing versions and pushes, pops, melds, push_ranges or
	 *        (with a lazy add) shifts, keeping every version; checks top and
	 *        size of the new version against a multiset copied per version,
	 *        and walks every version's tree at the end.
	 */
	template<std::copyable Heap, algorithm Algorithm, std::strict_weak_order<int64, int64> Compare, bool HasShift>
	void check_class_form(uint64 seed, int steps, int value_range) {
		std::mt19937_64 rng(seed);
		using oracle = std::multiset<int64, Compare>;
		std::vector<Heap> versions;
		std::vector<oracle> reference;
		versions.emplace_back();
		reference.emplace_back();
		auto random_value = [&] { return int64(rng() % uint64(value_range)) - value_range / 2; };
		for (int step = 0; step < steps; ++step) {
			std::size_t const p = rng() % versions.size();
			oracle next = reference[p];
			int const op = int(rng() % (HasShift ? 6 : 5));
			Heap made;
			if (op == 0 || op == 1) {
				int64 const v = random_value();
				made = op == 0 ? versions[p].push(v) : versions[p].emplace(v);
				next.insert(v);
			} else if (op == 2) {
				if (next.empty()) continue;
				made = versions[p].pop();
				next.erase(std::prev(next.end()));
			} else if (op == 3) {
				std::size_t const q = rng() % versions.size();
				made = versions[p].meld(versions[q]);
				next.insert(reference[q].begin(), reference[q].end());
			} else if (op == 4) {
				std::vector<int64> values(rng() % 6);
				for (auto &v : values) v = random_value();
				made = versions[p].push_range(values);
				next.insert(values.begin(), values.end());
			} else {
				if constexpr (HasShift) {
					int64 const shift = random_value();
					made = versions[p].add_to_all(shift);
					oracle shifted;
					for (int64 v : next) shifted.insert(v + shift);
					next = std::move(shifted);
				}
			}
			REQUIRE(made.size() == next.size());
			REQUIRE(made.empty() == next.empty());
			if (!next.empty()) REQUIRE(made.top() == oracle_top(next));
			versions.push_back(std::move(made));
			reference.push_back(std::move(next));
		}
		for (std::size_t k = 0; k < versions.size(); ++k) {
			REQUIRE(versions[k].size() == reference[k].size());
			if (!reference[k].empty()) REQUIRE(versions[k].top() == oracle_top(reference[k]));
			if constexpr (!HasShift) {
				oracle seen;
				using view = typename Heap::node_view;
				walk_result const r = walk<Algorithm, Compare>(
					versions[k].root(), [](view x) { return x.left(); }, [](view x) { return x.right(); },
					[](view x) { return x.value(); }, [](view x) { return x.empty(); }, seen);
				REQUIRE(r.count == reference[k].size());
				REQUIRE(seen == reference[k]);
			}
		}
		// drain one version through the linear (moving) idiom
		std::size_t const k = versions.size() - 1;
		Heap h = versions[k];
		oracle ref = reference[k];
		while (!ref.empty()) {
			REQUIRE(h.top() == oracle_top(ref));
			h = std::move(h).pop();
			ref.erase(std::prev(ref.end()));
		}
		REQUIRE(h.empty());
	}

	/** @brief The same version DAG on a forest-form heap (versions are ids). */
	template<std::default_initializable Forest, algorithm Algorithm, std::strict_weak_order<int64, int64> Compare>
	void check_forest_form(uint64 seed, int steps, int value_range) {
		std::mt19937_64 rng(seed);
		using oracle = std::multiset<int64, Compare>;
		Forest forest;
		std::vector<uint32> versions{Forest::empty_heap};
		std::vector<oracle> reference(1);
		auto random_value = [&] { return int64(rng() % uint64(value_range)) - value_range / 2; };
		for (int step = 0; step < steps; ++step) {
			std::size_t const p = rng() % versions.size();
			oracle next = reference[p];
			int const op = int(rng() % 4);
			uint32 made = 0;
			if (op == 0) {
				int64 const v = random_value();
				made = forest.push(versions[p], v);
				next.insert(v);
			} else if (op == 1) {
				if (next.empty()) continue;
				made = forest.pop(versions[p]);
				next.erase(std::prev(next.end()));
			} else if (op == 2) {
				std::size_t const q = rng() % versions.size();
				made = forest.meld(versions[p], versions[q]);
				next.insert(reference[q].begin(), reference[q].end());
			} else {
				std::vector<int64> values(rng() % 6);
				for (auto &v : values) v = random_value();
				made = forest.push_range(versions[p], values);
				next.insert(values.begin(), values.end());
			}
			REQUIRE(forest.empty(made) == next.empty());
			if (!next.empty()) REQUIRE(forest.top(made) == oracle_top(next));
			if constexpr (Algorithm == algorithm::weight_biased) REQUIRE(forest.size(made) == next.size());
			versions.push_back(made);
			reference.push_back(std::move(next));
		}
		for (std::size_t k = 0; k < versions.size(); ++k) {
			oracle seen;
			walk_result const r = walk<Algorithm, Compare>(
				versions[k], [&](uint32 x) { return forest.left(x); }, [&](uint32 x) { return forest.right(x); },
				[&](uint32 x) { return forest.top(x); }, [&](uint32 x) { return forest.empty(x); }, seen);
			REQUIRE(r.count == reference[k].size());
			REQUIRE(seen == reference[k]);
		}
	}

	/** @brief Every form of one experimental contender: class (two value ranges), class with a lazy add, forest. */
	template<algorithm Algorithm, std::strict_weak_order<int64, int64> Compare>
	void check_variant(uint64 seed) {
		check_class_form<ex::persistent_heap_by_meld_policy<int64, Compare, Algorithm>, Algorithm, Compare, false>(seed, 1500,
																												   40);
		check_class_form<ex::persistent_heap_by_meld_policy<int64, Compare, Algorithm>, Algorithm, Compare, false>(seed, 800,
																												   1 << 30);
		check_class_form<ex::persistent_heap_by_meld_policy<int64, Compare, Algorithm, int64>, Algorithm, Compare, true>(
			seed, 1500, 40);
		check_forest_form<ex::persistent_heap_forest_by_meld_policy<int64, Compare, Algorithm>, Algorithm, Compare>(seed, 1500,
																													40);
	}

	/** @brief The temporary-vector vector heapify against the library in-place heapify, on sized and unsized ranges. */
	template<algorithm Algorithm>
	void check_heapify_ranges(uint64 seed) {
		using place_heap = ex::persistent_heap_by_meld_policy<int64, std::less<>, Algorithm, void, atl::shift_by_addition,
															  atl::compose_by_addition, false, true>;
		std::mt19937_64 rng(seed);
		for (std::size_t const n : {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 100, 1000}) {
			std::vector<int64> values(n);
			for (auto &v : values) v = int64(rng() % 50) - 25;
			atl::persistent_heap<int64> library(values);
			place_heap built(values);
			auto odd = values | std::views::filter([](int64 v) { return v % 2 != 0; });
			atl::persistent_heap<int64> unsized(odd);
			std::vector<int64> odd_sorted(odd.begin(), odd.end());
			std::ranges::sort(odd_sorted, std::greater<>{});
			REQUIRE(built.size() == n);
			REQUIRE(unsized.size() == odd_sorted.size());
			auto unsized_grown = atl::persistent_heap<int64>(std::vector<int64>{}).push_range(odd);
			for (int64 const v : odd_sorted) {
				REQUIRE(unsized.top() == v);
				REQUIRE(unsized_grown.top() == v);
				unsized = std::move(unsized).pop();
				unsized_grown = std::move(unsized_grown).pop();
			}
			REQUIRE(unsized.empty());
			REQUIRE(unsized_grown.empty());
			auto drained = std::move(built);
			auto expected = std::move(library);
			while (!expected.empty()) {
				REQUIRE(drained.top() == expected.top());
				drained = std::move(drained).pop();
				expected = std::move(expected).pop();
			}
			REQUIRE(drained.empty());
			std::vector<int64> all = values;
			all.insert(all.end(), {5, -5});
			std::ranges::sort(all, std::greater<>{});
			auto grown = atl::persistent_heap<int64>(std::vector<int64>{5, -5}).push_range(values);
			REQUIRE(grown.size() == all.size());
			for (int64 const v : all) {
				REQUIRE(grown.top() == v);
				grown = std::move(grown).pop();
			}
			REQUIRE(grown.empty());
		}
	}
}

TEST_CASE("persistent_heap variants: random version DAG against a multiset per version", "[persistent_heap]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	check_variant<algorithm::rank_leftist, std::less<>>(seed);
	check_variant<algorithm::rank_leftist, std::greater<>>(seed);
	check_variant<algorithm::weight_biased, std::less<>>(seed);
	check_variant<algorithm::weight_biased, std::greater<>>(seed);
	check_variant<algorithm::randomized, std::less<>>(seed);
	check_variant<algorithm::randomized, std::greater<>>(seed);
}

TEST_CASE("persistent_heap variants: temporary-vector vector heapify matches library in-place heapify", "[persistent_heap]") {
	auto const seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	check_heapify_ranges<algorithm::rank_leftist>(seed);
	check_heapify_ranges<algorithm::weight_biased>(seed);
	check_heapify_ranges<algorithm::randomized>(seed);
}

TEST_CASE("persistent_heap: random version DAG against a multiset per version", "[persistent_heap]") {
	auto const seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	check_class_form<atl::persistent_heap<int64>, algorithm::randomized, std::less<>, false>(seed, 2000, 40);
	check_class_form<atl::persistent_heap<int64, std::greater<>>, algorithm::randomized, std::greater<>, false>(
		seed, 2000, 1 << 30);
	check_forest_form<atl::persistent_heap_forest<int64>, algorithm::randomized, std::less<>>(seed, 2000, 40);
	check_forest_form<atl::persistent_heap_forest<int64, std::greater<>>, algorithm::randomized, std::greater<>>(
		seed, 2000, 1 << 30);
}

TEST_CASE("persistent_heap_forest: ids, push_range from the empty heap, subtrees as versions", "[persistent_heap]") {
	atl::persistent_heap_forest<int, std::greater<>> forest; // min-heaps
	uint32 const e = decltype(forest)::empty_heap;
	REQUIRE(forest.empty(e));
	uint32 const a = forest.push_range(e, std::vector<int>{6, 2, 8, 4});
	REQUIRE(forest.top(a) == 2);
	uint32 const b = forest.emplace(a, 1);
	REQUIRE(forest.top(b) == 1);
	REQUIRE(forest.top(a) == 2); // old id intact
	uint32 const c = forest.meld(b, b);
	std::multiset<int> drained;
	for (uint32 h = c; !forest.empty(h); h = forest.pop(h)) drained.insert(forest.top(h));
	REQUIRE(drained == std::multiset<int>{1, 1, 2, 2, 4, 4, 6, 6, 8, 8});
	for (uint32 child : {forest.left(a), forest.right(a)})
		if (!forest.empty(child)) REQUIRE(forest.top(child) >= forest.top(a));
	REQUIRE(forest.pop(forest.push(e, 3)) == e);
	std::size_t const before = forest.arena_size();
	forest.reserve(1000);
	REQUIRE(forest.arena_size() == before);
	CHECK(atl_test::aborts([&] { (void)forest.top(e); }));
	CHECK(atl_test::aborts([&] { (void)forest.pop(e); }));
	CHECK(atl_test::aborts([&] { (void)forest.left(e); }));
	CHECK(atl_test::aborts([&] { (void)forest.meld(a, uint32(forest.arena_size())); }));
}

TEST_CASE("persistent_heap: construction, CTAD, push_range and old versions", "[persistent_heap]") {
	std::vector<int> const data{5, 1, 9, 3, 9, 7};
	atl::persistent_heap heap(data); // CTAD: persistent_heap<int>
	static_assert(std::is_same_v<decltype(heap), atl::persistent_heap<int>>);
	REQUIRE(heap.size() == 6);
	REQUIRE(heap.top() == 9);

	atl::persistent_heap min_heap(data, std::greater<>{});
	static_assert(std::is_same_v<decltype(min_heap), atl::persistent_heap<int, std::greater<>>>);
	REQUIRE(min_heap.top() == 1);

	auto const popped = heap.pop().pop();
	REQUIRE(popped.top() == 7);
	REQUIRE(heap.top() == 9); // the old version is intact
	REQUIRE(heap.size() == 6);

	auto const more = popped.push_range(std::vector<int>{2, 8, 11});
	REQUIRE(more.size() == 7);
	REQUIRE(more.top() == 11);
	auto const melded = more.meld(heap);
	REQUIRE(melded.size() == 13);
	auto const self = heap.meld(heap);
	REQUIRE(self.size() == 12);
	std::multiset<int> drained;
	for (auto h = self; !h.empty(); h = std::move(h).pop()) drained.insert(h.top());
	REQUIRE(drained == std::multiset<int>{1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 9, 9});

	atl::persistent_heap<int> const empty;
	REQUIRE(empty.empty());
	REQUIRE(empty.root().empty());
	REQUIRE(empty.push_range(std::span<int const>{}).empty());
	auto const one = empty.emplace(4);
	REQUIRE(one.top() == 4);
	REQUIRE(one.root().left().empty());
	REQUIRE(one.root().right().empty());
	REQUIRE(one.pop().empty());

	atl::persistent_heap<std::string> words(std::vector<std::string>{"pear", "apple"});
	REQUIRE(words.arena_size() == 3); // the null node and the two built nodes
	words.reserve(100);
	REQUIRE(words.arena_size() == 3);
	REQUIRE(words.emplace(3, 'z').top() == "zzz");
	REQUIRE(words.arena_size() == 4); // the new top is the fresh leaf itself: nothing copied
}

TEST_CASE("persistent_heap: push moves the value into place", "[persistent_heap]") {
	using atl_test::move_probe;

	struct by_value {
		bool operator()(move_probe const &a, move_probe const &b) const { return a.v < b.v; }
	};

	atl::persistent_heap<move_probe, by_value> heap(std::vector<move_probe>{1, 2, 3, 4, 5}, by_value{});
	move_probe::reset();
	auto const a = heap.push(move_probe(move_probe::sentinel));
	REQUIRE(a.top().v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
	move_probe::reset();
	auto const b = std::move(heap).emplace(move_probe::sentinel);
	REQUIRE(b.top().v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
}

TEST_CASE("persistent_heap: asserted preconditions", "[persistent_heap]") {
	atl::persistent_heap<int> const empty;
	CHECK(atl_test::aborts([&] { (void)empty.top(); }));
	CHECK(atl_test::aborts([&] { (void)empty.pop(); }));
	CHECK(atl_test::aborts([&] { (void)empty.root().value(); }));
	CHECK(atl_test::aborts([&] { (void)empty.root().left(); }));
	atl::persistent_heap<int> const other_pool;
	CHECK(atl_test::aborts([&] { (void)empty.meld(other_pool); }));
	CHECK(atl_test::aborts([&] { (void)empty.push(1).meld(other_pool.push(2)); }));
}

TEST_CASE("persistent_heap: single-pass input ranges preserve versions", "[persistent_heap]") {
	for (int const n : {0, 1, 2, 3, 7, 8, 9, 64, 65}) {
		std::string text;
		for (int i = 0; i < n; ++i) text += std::to_string(i) + " ";
		std::istringstream input(text), other(text), third(text);
		atl::persistent_heap<int> base(std::vector<int>{-1});
		auto grown = base.push_range(std::ranges::istream_view<int>(input));
		atl::persistent_heap<int> built{std::ranges::istream_view<int>(other)};
		atl::persistent_heap_forest<int> forest;
		auto const old = forest.push(forest.empty_heap, -1);
		auto h = forest.push_range(old, std::ranges::istream_view<int>(third));
		for (int i = n; i > 0; --i) {
			REQUIRE(grown.top() == i - 1);
			REQUIRE(built.top() == i - 1);
			REQUIRE(forest.top(h) == i - 1);
			grown = std::move(grown).pop();
			built = std::move(built).pop();
			h = forest.pop(h);
		}
		REQUIRE(built.empty());
		REQUIRE(grown.top() == -1);
		REQUIRE(forest.top(h) == -1);
		REQUIRE(base.top() == -1);
		REQUIRE(forest.top(old) == -1);
	}
}

TEST_CASE("persistent_heap: heapify forwards owning ranges and preserves views", "[persistent_heap]") {
	using atl_test::move_probe;

	struct by_value {
		bool operator()(move_probe const &a, move_probe const &b) const { return a.v < b.v; }
	};

	std::vector<move_probe> source{1, move_probe::sentinel, 3};
	move_probe::reset();
	atl::persistent_heap<move_probe, by_value> copied(source);
	REQUIRE(source[1].v == move_probe::sentinel);
	REQUIRE(copied.top().v == move_probe::sentinel);
	atl::persistent_heap<move_probe, by_value> viewed{std::span<move_probe>(source)};
	REQUIRE(source[1].v == move_probe::sentinel);
	REQUIRE(viewed.top().v == move_probe::sentinel);
	move_probe::reset();
	atl::persistent_heap<move_probe, by_value> moved(std::move(source));
	REQUIRE(moved.top().v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
	std::vector<move_probe> more{2, move_probe::sentinel, 4};
	atl::persistent_heap_forest<move_probe, by_value> forest;
	move_probe::reset();
	auto const root = forest.push_range(forest.empty_heap, std::move(more));
	REQUIRE(forest.top(root).v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
}
