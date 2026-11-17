#include <atl/ds/segment_tree_beats.hpp>
#include <atl/experimental/ds/segment_tree_beats_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	using wide = atl::int128;

	template<typename Tree, typename Element, typename Sum>
	concept preset_tree = atl::signed_integral<Element> && atl::signed_integral<Sum>
					   && std::constructible_from<Tree, std::vector<Element> const &>
					   && requires(Tree &tree, Element x) {
							  tree.chmin(0, 1, x);
							  tree.chmax(0, 1, x);
							  tree.add(0, 1, x);
							  tree.assign(0, 1, x);
							  tree.set(0, x);
							  { tree.sum(0, 1) } -> std::convertible_to<Sum>;
							  { tree.min(0, 1) } -> std::convertible_to<Element>;
							  { tree.max(0, 1) } -> std::convertible_to<Element>;
							  { tree.element(0) } -> std::convertible_to<Element>;
						  };

	/** @brief The generic engine on the clamp-add algebra: the class's reference, and what generic members need. */
	template<atl::signed_integral Element, atl::signed_integral Sum = Element>
	using clamp_add_engine = atl::segment_tree_beats<atl::beats::extremes<Element, Sum>, atl::beats::clamp_add<Element>>;

	/** @brief The sizes every oracle run covers: 0, 1, powers of two and their neighbours, then random ones. */
	std::vector<std::size_t> oracle_sizes(std::mt19937_64 &rng) {
		std::vector<std::size_t> sizes{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};
		for (int i = 0; i < 6; ++i) sizes.push_back(std::size_t(rng() % 71));
		return sizes;
	}

	/** @brief A uniform integer in `[lo, hi]`. */
	template<atl::integral T>
	T uniform(std::mt19937_64 &rng, T lo, T hi) {
		return T(lo + T(rng() % uint64(hi - lo + 1)));
	}

	/** @brief A random half-open range of `[0, n)`. */
	std::pair<std::size_t, std::size_t> random_range(std::mt19937_64 &rng, std::size_t n) {
		std::size_t l = rng() % (n + 1), r = rng() % (n + 1);
		if (l > r) std::swap(l, r);
		return {l, r};
	}

	/**
	 * @brief Runs @p steps random chmin / chmax / add / assign / set / sum / min / max /
	 *        element operations on @p tree and on a brute-force array, comparing every
	 *        query. Values start in `[lo, hi]`; clamp arguments reach 2 past either end,
	 *        adds lie in `[-step, step]`. Works for every contender exposing the preset
	 *        members, so the experimental variants share it: the brute-force array is the
	 *        family's certifying oracle (O(n) per operation, nothing to get wrong).
	 */
	template<typename Tree, atl::signed_integral Element = int64, atl::signed_integral Sum = wide>
		requires preset_tree<Tree, Element, Sum>
	void preset_oracle(std::mt19937_64 &rng, std::size_t n, std::type_identity_t<Element> lo, std::type_identity_t<Element> hi,
					   std::type_identity_t<Element> step, int steps) {
		std::vector<Element> a(n);
		for (auto &x : a) x = uniform(rng, lo, hi);
		Tree tree(a);
		for (int s = 0; s < steps; ++s) {
			if (n == 0) {
				REQUIRE(tree.sum(0, 0) == 0);
				return;
			}
			std::size_t l = rng() % (n + 1), r = rng() % (n + 1);
			if (l > r) std::swap(l, r);
			int const kind = int(rng() % 9);
			if (kind <= 3 || kind == 4) {
				Element const x = kind == 2 ? uniform(rng, Element(-step), step) : uniform(rng, Element(lo - 2), Element(hi + 2));
				for (std::size_t i = l; i < r; ++i) switch (kind) {
					case 0:
						a[i] = (std::min)(a[i], x);
						break;
					case 1:
						a[i] = (std::max)(a[i], x);
						break;
					case 2:
						a[i] = Element(a[i] + x);
						break;
					default:
						a[i] = x;
						break;
					}
				switch (kind) {
				case 0:
					tree.chmin(l, r, x);
					break;
				case 1:
					tree.chmax(l, r, x);
					break;
				case 2:
					tree.add(l, r, x);
					break;
				default:
					tree.assign(l, r, x);
					break;
				}
			} else if (kind == 5) {
				std::size_t const i = rng() % n;
				a[i] = uniform(rng, lo, hi);
				tree.set(i, a[i]);
			} else if (kind == 6) {
				std::size_t const i = rng() % n;
				REQUIRE(tree.element(i) == a[i]);
			} else {
				Sum want = 0;
				for (std::size_t i = l; i < r; ++i) want += Sum(a[i]);
				REQUIRE(Sum(tree.sum(l, r)) == want);
				if (l < r) {
					REQUIRE(tree.min(l, r) == *std::ranges::min_element(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r)));
					REQUIRE(tree.max(l, r) == *std::ranges::max_element(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r)));
				}
				if constexpr (requires { tree.fold(l, r).max_count; }) {
					auto const f = tree.fold(l, r);
					REQUIRE(f.width == r - l);
					if (l < r) {
						std::vector<Element> part(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r));
						std::ranges::sort(part);
						Element const mx = part.back(), mn = part.front();
						REQUIRE(f.max == mx);
						REQUIRE(f.min == mn);
						REQUIRE(f.max_count == uint32(std::ranges::count(part, mx)));
						REQUIRE(f.min_count == uint32(std::ranges::count(part, mn)));
						auto const below = std::ranges::find_if(part.rbegin(), part.rend(), [&](Element v) { return v < mx; });
						Element const want_second = below == part.rend() ? (std::numeric_limits<Element>::lowest)() : *below;
						REQUIRE(f.second_max == want_second);
						auto const above = std::ranges::find_if(part, [&](Element v) { return v > mn; });
						REQUIRE(f.second_min == (above == part.end() ? (std::numeric_limits<Element>::max)() : *above));
					}
				}
			}
		}
	}

	/** @brief `preset_oracle` over every size of @ref oracle_sizes. */
	template<typename Tree, atl::signed_integral Element = int64, atl::signed_integral Sum = wide>
		requires preset_tree<Tree, Element, Sum>
	void preset_oracle_all_sizes(uint64 seed, std::type_identity_t<Element> lo, std::type_identity_t<Element> hi, std::type_identity_t<Element> step,
								 int steps) {
		std::mt19937_64 rng(seed);
		for (std::size_t const n : oracle_sizes(rng)) preset_oracle<Tree, Element, Sum>(rng, n, lo, hi, step, steps);
	}
}

// The generic members are the engine's alone; the class is not the engine under another name.
template<typename Tree>
concept has_a_generic_member = requires(Tree &t, atl::beats::clamp_add<int64> const &f, atl::beats::extremes<int64> const &s) {
	requires requires { t.apply(0, f); } || requires { t.apply(0, 0, f); }
				 || requires { t.apply_derived(0, 0, [](auto const &) { return std::optional<atl::beats::clamp_add<int64>>{}; }); } || requires { t.set(0, s); };
};
static_assert(has_a_generic_member<clamp_add_engine<int64>>);
static_assert(!has_a_generic_member<atl::beats_segment_tree<int64>>);
static_assert(!std::same_as<atl::beats_segment_tree<int64>, clamp_add_engine<int64>>);
static_assert(std::same_as<atl::beats_segment_tree<int64>::value_type, atl::beats::extremes<int64>>);

TEMPLATE_TEST_CASE("segment_tree_beats: preset vs brute force on narrow values (ties)", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	auto const seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	preset_oracle_all_sizes<TestType>(seed, -5, 5, 3, 400);
}

TEMPLATE_TEST_CASE("segment_tree_beats: preset vs brute force on wide values", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	preset_oracle_all_sizes<TestType>(seed, -1'000'000'000'000, 1'000'000'000'000, 1'000'000'000'000, 400);
}

TEMPLATE_TEST_CASE("segment_tree_beats: int32 elements with int64 sums", "[segment_tree_beats]", (atl::beats_segment_tree<atl::int32, int64>),
				   (clamp_add_engine<atl::int32, int64>)) {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	preset_oracle_all_sizes<TestType, atl::int32, int64>(seed, -1'000'000, 1'000'000, 1000, 400);
}

TEMPLATE_TEST_CASE("segment_tree_beats: int128 sums of values near 2^61", "[segment_tree_beats]", (atl::beats_segment_tree<int64, atl::int128>),
				   (clamp_add_engine<int64, atl::int128>)) {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	int64 const big = int64(1) << 61;
	preset_oracle_all_sizes<TestType>(seed, big - 6, big, int64(2), 300);
	preset_oracle_all_sizes<TestType>(seed, -big, -big + 6, int64(2), 300);
}

TEMPLATE_TEST_CASE("segment_tree_beats: int128 elements", "[segment_tree_beats]", atl::beats_segment_tree<atl::int128>,
				   clamp_add_engine<atl::int128>) {
	auto const seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	atl::int128 const big = atl::int128(1) << 100;
	preset_oracle_all_sizes<TestType, atl::int128, atl::int128>(seed, big - 4, big + 4, 3, 300);
}

TEMPLATE_TEST_CASE("segment_tree_beats: every contender vs brute force", "[segment_tree_beats][variants]",
				   atl::experimental::beats_segment_tree_by_recursive_compact_nodes<int64>, atl::experimental::beats_segment_tree_by_explicit_stack<int64>,
				   atl::experimental::beats_segment_tree_by_fixed_clamp_add<int64>, atl::beats_segment_tree<int64>, clamp_add_engine<int64>,
				   (atl::experimental::beats_segment_tree_by_tagless_layout<int64, int64, atl::experimental::beats_layout::aos_aligned>),
				   (atl::experimental::beats_segment_tree_by_tagless_layout<int64, int64, atl::experimental::beats_layout::soa>),
				   atl::experimental::beats_segment_tree_by_three_adds<int64>, atl::experimental::beats_segment_tree_by_optional_mapping<int64>,
				   atl::experimental::beats_segment_tree_by_unconditional_push<int64>, atl::experimental::beats_segment_tree_by_no_collapse<int64>) {
	auto const seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	preset_oracle_all_sizes<TestType, int64, int64>(seed, -5, 5, 3, 400);
	preset_oracle_all_sizes<TestType, int64, int64>(seed, -1'000'000'000'000, 1'000'000'000'000, 1'000'000'000'000, 200);
}

TEST_CASE("segment_tree_beats: contenders agree with the library tree on long runs", "[segment_tree_beats][variants]") {
	std::mt19937_64 rng(GENERATE(take(2, random(uint64(0), (std::numeric_limits<uint64>::max)()))));
	std::size_t const n = 3000;
	std::vector<int64> a(n);
	for (auto &x : a) x = uniform<int64>(rng, -1000, 1000);
	atl::beats_segment_tree<int64> library(a);
	clamp_add_engine<int64> v1(a);
	atl::experimental::beats_segment_tree_by_recursive_compact_nodes<int64> v2(a);
	atl::experimental::beats_segment_tree_by_fixed_clamp_add<int64> v4(a);
	atl::experimental::beats_segment_tree_by_three_adds<int64> v6(a);
	for (int s = 0; s < 20000; ++s) {
		auto [l, r] = std::pair{rng() % (n + 1), rng() % (n + 1)};
		if (l > r) std::swap(l, r);
		int64 const x = uniform<int64>(rng, -1000, 1000);
		auto const each = [&](auto &&f) {
			f(library);
			f(v1);
			f(v2);
			f(v4);
			f(v6);
		};
		switch (rng() % 5) {
		case 0:
			each([&](auto &t) { t.chmin(l, r, x); });
			break;
		case 1:
			each([&](auto &t) { t.chmax(l, r, x); });
			break;
		case 2:
			each([&](auto &t) { t.add(l, r, x / 10); });
			break;
		case 3:
			each([&](auto &t) { t.assign(l, r, x); });
			break;
		default:
			if (l < r) {
				int64 const want = library.sum(l, r);
				REQUIRE(v1.sum(l, r) == want);
				REQUIRE(v2.sum(l, r) == want);
				REQUIRE(v4.sum(l, r) == want);
				REQUIRE(v6.sum(l, r) == want);
			}
		}
	}
}

TEMPLATE_TEST_CASE("segment_tree_beats: extreme values next to the sentinels", "[segment_tree_beats]", (atl::beats_segment_tree<int64, wide>),
				   (clamp_add_engine<int64, wide>)) {
	int64 const top = (std::numeric_limits<int64>::max)() - 1, bottom = (std::numeric_limits<int64>::lowest)() + 1;
	std::vector<int64> const a{top, bottom, 0, top, bottom};
	TestType tree(a);
	REQUIRE(tree.max(0, 5) == top);
	REQUIRE(tree.min(0, 5) == bottom);
	REQUIRE(tree.sum(0, 5) == wide(top) * 2 + wide(bottom) * 2);
	tree.chmin(0, 5, top - 1);
	tree.chmax(0, 5, bottom + 1);
	REQUIRE(tree.max(0, 5) == top - 1);
	REQUIRE(tree.min(0, 5) == bottom + 1);
	tree.add(2, 3, top - 1);
	REQUIRE(tree.element(2) == top - 1);
	tree.assign(0, 5, bottom);
	REQUIRE(tree.sum(0, 5) == wide(bottom) * 5);
	tree.add(0, 5, 1);
	REQUIRE(tree.element(4) == bottom + 1);
	REQUIRE(tree.max_right(0, [](auto const &v) { return v.max_count == v.width; }) == 5);
}

TEMPLATE_TEST_CASE("segment_tree_beats: whole-range chmin cascade and collapse", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	std::size_t const n = 1000;
	std::vector<int64> a(n);
	for (std::size_t i = 0; i < n; ++i) a[i] = int64(i);
	TestType tree(a);
	for (int64 x = int64(n); x-- > 0;) {
		tree.chmin(0, n, x);
		REQUIRE(tree.max(0, n) == x);
		REQUIRE(tree.sum(0, n) == x * (x + 1) / 2 + x * int64(n - 1 - std::size_t(x)));
	}
	REQUIRE(tree.max_right(0, [](auto const &v) { return v.max_count == v.width; }) == n);
	tree.chmax(0, n, 7);
	REQUIRE(tree.sum(0, n) == 7 * int64(n));
}

TEMPLATE_TEST_CASE("segment_tree_beats: empty and default and single-element trees", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	TestType none;
	REQUIRE(none.size() == 0);
	REQUIRE(none.sum(0, 0) == 0);
	none.chmin(0, 0, 3);
	REQUIRE(none.max_right(0, [](auto const &) { return true; }) == 0);
	REQUIRE(none.min_left(0, [](auto const &) { return true; }) == 0);
	TestType zeros(std::size_t(4));
	REQUIRE(zeros.sum(0, 4) == 0);
	REQUIRE(zeros.max(0, 4) == 0);
	zeros.add(1, 3, 5);
	REQUIRE(zeros.sum(0, 4) == 10);
	TestType one(std::vector<int64>{9});
	one.chmin(0, 1, 4);
	one.add(0, 1, -10);
	one.chmax(0, 1, -2);
	REQUIRE(one.element(0) == -2);
	REQUIRE(one.max_right(0, [](auto const &v) { return v.width == 0 || v.max == -2; }) == 1);
	REQUIRE(one.size() == 1);
}

TEST_CASE("segment_tree_beats: the engine's own empty and fold", "[segment_tree_beats]") {
	clamp_add_engine<int64> none;
	REQUIRE(none.empty());
	REQUIRE(none.fold() == atl::beats::extremes<int64>::identity());
	clamp_add_engine<int64> one(std::vector<int64>{9});
	REQUIRE(!one.empty());
	REQUIRE(one.fold().width == 1);
}

namespace {
	/** @brief The extremes summary of `[l, r)` computed from a sorted copy: nothing shared with the tree's merge. */
	template<atl::signed_integral Element, atl::signed_integral Sum>
	atl::beats::extremes<Element, Sum> brute_summary(std::vector<Element> const &a, std::size_t l, std::size_t r) {
		using summary = atl::beats::extremes<Element, Sum>;
		if (l == r) return summary::identity();
		std::vector<Element> part(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r));
		std::ranges::sort(part);
		summary s;
		s.max = part.back(), s.min = part.front();
		s.max_count = uint32(std::ranges::count(part, s.max)), s.min_count = uint32(std::ranges::count(part, s.min));
		for (Element const v : part) {
			s.sum = Sum(s.sum + Sum(v));
			if (v < s.max) s.second_max = v;
		}
		auto const above = std::ranges::find_if(part, [&](Element v) { return v > s.min; });
		if (above != part.end()) s.second_min = *above;
		s.width = uint32(r - l);
		return s;
	}

	/** @brief Checks `max_right` and `min_left` from @p at with @p pred against a scan of @ref brute_summary. */
	template<typename Tree, atl::signed_integral Element, atl::signed_integral Sum, std::predicate<atl::beats::extremes<Element, Sum> const &> Pred>
		requires preset_tree<Tree, Element, Sum>
	void check_search(Tree &tree, std::vector<Element> const &a, std::size_t at, Pred const &pred) {
		std::size_t end = at, begin = at;
		while (end < a.size() && pred(brute_summary<Element, Sum>(a, at, end + 1))) ++end;
		while (begin > 0 && pred(brute_summary<Element, Sum>(a, begin - 1, at))) --begin;
		REQUIRE(tree.max_right(at, pred) == end);
		REQUIRE(tree.min_left(at, pred) == begin);
	}

	/**
	 * @brief Random chmin / chmax / add / assign mixes with a search after each: predicates
	 *        over every monotone field of the summary, the two extreme predicates, and the
	 *        exact summary handed to the predicate.
	 */
	template<typename Tree, atl::signed_integral Element, atl::signed_integral Sum>
		requires preset_tree<Tree, Element, Sum>
	void search_oracle(std::mt19937_64 &rng, std::size_t n, Element lo, Element hi, Element step, int steps) {
		std::vector<Element> a(n);
		for (auto &x : a) x = uniform(rng, lo, hi);
		Tree tree(a);
		for (int s = 0; s < steps; ++s) {
			auto const [l, r] = random_range(rng, n);
			Element const x = uniform(rng, Element(lo - 2), Element(hi + 2));
			switch (rng() % 5) {
			case 0:
				tree.chmin(l, r, x);
				for (std::size_t i = l; i < r; ++i) a[i] = (std::min)(a[i], x);
				break;
			case 1:
				tree.chmax(l, r, x);
				for (std::size_t i = l; i < r; ++i) a[i] = (std::max)(a[i], x);
				break;
			case 2: {
				Element const d = uniform(rng, Element(-step), step);
				tree.add(l, r, d);
				for (std::size_t i = l; i < r; ++i) a[i] = Element(a[i] + d);
				break;
			}
			case 3:
				tree.assign(l, r, x);
				std::ranges::fill(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r), x);
				break;
			default:
				break;
			}
			std::size_t const at = rng() % (n + 1);
			uint32 const w = uint32(rng() % (n + 2));
			Element const t = uniform(rng, Element(lo - 2), Element(hi + 2));
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.max <= t; });
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.min >= t; });
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.second_max <= t; });
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.second_min >= t; });
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.width <= w; });
			check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.width == 0 || Sum(v.max) - Sum(v.min) <= Sum(step); });
			check_search<Tree, Element, Sum>(tree, a, at, [](auto const &) { return true; });
			check_search<Tree, Element, Sum>(tree, a, at, [](auto const &v) { return v.width == 0; });
			if (std::ranges::all_of(a, [](Element v) { return v >= 0; }) && t >= 0)
				check_search<Tree, Element, Sum>(tree, a, at, [&](auto const &v) { return v.sum <= Sum(t); });
			// the predicate sees the exact summary of every prefix it is asked about
			if (std::size_t const width = r - l; width > 0) {
				atl::beats::extremes<Element, Sum> seen;
				REQUIRE(tree.max_right(l, [&](auto const &v) {
					if (v.width == width) seen = v;
					return v.width <= width;
				}) == r);
				REQUIRE(bool(seen == brute_summary<Element, Sum>(a, l, r)));
				seen = {};
				REQUIRE(tree.min_left(r, [&](auto const &v) {
					if (v.width == width) seen = v;
					return v.width <= width;
				}) == l);
				REQUIRE(bool(seen == brute_summary<Element, Sum>(a, l, r)));
			}
		}
	}
}

TEMPLATE_TEST_CASE("segment_tree_beats: max_right and min_left vs brute force, ops mixed in", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	auto const seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : oracle_sizes(rng)) {
		search_oracle<TestType, int64, int64>(rng, n, -5, 5, 3, 120);
		search_oracle<TestType, int64, int64>(rng, n, 0, 9, 2, 120);
		search_oracle<TestType, int64, int64>(rng, n, -1'000'000'000'000, 1'000'000'000'000, 1'000'000'000, 60);
	}
}

TEST_CASE("segment_tree_beats: max_right and min_left with a wider sum", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : oracle_sizes(rng)) {
		search_oracle<atl::beats_segment_tree<atl::int32, int64>, atl::int32, int64>(rng, n, -1'000'000, 1'000'000, 1000, 80);
		search_oracle<atl::beats_segment_tree<int64, wide>, int64, wide>(rng, n, int64(1) << 61, (int64(1) << 61) + 8, 2, 80);
	}
}

namespace {
	/**
	 * @brief Drives the class and the engine with one random stream and compares everything
	 *        both answer: sum, min, max, element, every field of a range's summary (the class
	 *        hands it to a predicate, the engine folds), and both searches. Arguments come
	 *        from a pool holding the two ends of `[lo, hi]`, their neighbours, values around
	 *        zero and random picks, so ties are common and the ends are hit. @p add_budget
	 *        caps the total `|x|` of the adds; 0 means none.
	 */
	template<atl::signed_integral Element, atl::signed_integral Sum>
	void engine_agreement(std::mt19937_64 &rng, std::size_t n, Element lo, Element hi, Element add_budget, int steps) {
		std::vector<Element> pool{lo, Element(lo + 1), Element(lo + 2), Element(hi - 2), Element(hi - 1), hi, Element(-1), Element(0), Element(1)};
		for (int i = 0; i < 6; ++i) pool.push_back(std::clamp(Element(rng()), lo, hi));
		auto const pick = [&] { return pool[rng() % pool.size()]; };
		std::vector<Element> initial(n);
		for (auto &x : initial) x = pick();
		atl::beats_segment_tree<Element, Sum> tree(initial);
		clamp_add_engine<Element, Sum> engine(initial);
		REQUIRE(tree.size() == engine.size());
		for (int s = 0; s < steps; ++s) {
			if (n == 0) return;
			auto const [l, r] = random_range(rng, n);
			Element const x = pick();
			switch (rng() % 8) {
			case 0:
				tree.chmin(l, r, x), engine.chmin(l, r, x);
				break;
			case 1:
				tree.chmax(l, r, x), engine.chmax(l, r, x);
				break;
			case 2: {
				Element const d = std::clamp(Element(rng()), Element(-add_budget), add_budget);
				add_budget = Element(add_budget - (d < 0 ? Element(-d) : d));
				tree.add(l, r, d), engine.add(l, r, d);
				break;
			}
			case 3:
				tree.assign(l, r, x), engine.assign(l, r, x);
				break;
			case 4: {
				std::size_t const i = rng() % n;
				tree.set(i, x), engine.set(i, x);
				break;
			}
			default:
				REQUIRE(tree.sum(l, r) == engine.sum(l, r));
				if (l < r) {
					REQUIRE(tree.min(l, r) == engine.min(l, r));
					REQUIRE(tree.max(l, r) == engine.max(l, r));
					atl::beats::extremes<Element, Sum> seen;
					std::size_t const width = r - l;
					REQUIRE(tree.max_right(l, [&](auto const &v) {
						if (v.width == width) seen = v;
						return v.width <= width;
					}) == r);
					REQUIRE(bool(seen == engine.fold(l, r)));
				}
				std::size_t const i = rng() % n;
				REQUIRE(tree.element(i) == engine.element(i));
				auto const below = [t = x](auto const &v) { return v.max <= t; };
				auto const above = [t = x](auto const &v) { return v.min >= t; };
				REQUIRE(tree.max_right(l, below) == engine.max_right(l, below));
				REQUIRE(tree.min_left(r, above) == engine.min_left(r, above));
			}
		}
		for (std::size_t i = 0; i < n; ++i) REQUIRE(tree.element(i) == engine.element(i));
		REQUIRE(tree.sum(0, n) == engine.fold().sum);
	}

	/** @brief @ref engine_agreement over every oracle size: first with the type's whole value range and no adds, then a quarter of it with adds. */
	template<atl::signed_integral Element, atl::signed_integral Sum>
	void engine_agreement_all_sizes(uint64 seed, int steps) {
		std::mt19937_64 rng(seed);
		Element const top = (std::numeric_limits<Element>::max)();
		for (std::size_t const n : oracle_sizes(rng)) {
			engine_agreement<Element, Sum>(rng, n, Element((std::numeric_limits<Element>::lowest)() + 1), Element(top - 1), Element(0), steps);
			engine_agreement<Element, Sum>(rng, n, Element(-(top / 4)), Element(top / 4), Element(top / 4), steps);
		}
	}
}

TEST_CASE("segment_tree_beats: the class agrees with the engine, extreme values, Sum wider than Element", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	engine_agreement_all_sizes<atl::int8, atl::int32>(seed, 300);
	engine_agreement_all_sizes<atl::int16, int64>(seed, 300);
	engine_agreement_all_sizes<atl::int32, int64>(seed, 300);
	engine_agreement_all_sizes<int64, atl::int128>(seed, 300);
}

TEST_CASE("segment_tree_beats: the class agrees with the engine, Sum equal to Element", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : oracle_sizes(rng)) {
		engine_agreement<int64, int64>(rng, n, -1'000'000'000'000, 1'000'000'000'000, 1'000'000'000'000, 300);
		engine_agreement<atl::int32, atl::int32>(rng, n, -1'000'000, 1'000'000, 1'000'000, 300);
	}
	atl::int128 const big = atl::int128(1) << 100;
	for (std::size_t const n : oracle_sizes(rng)) engine_agreement<atl::int128, atl::int128>(rng, n, -big, big, big, 200);
}

// The add of a single-valued node is void: only the two trees' single-value shortcuts keep the
// stale values below it, and the sum of its adds, from overflowing here.
TEMPLATE_TEST_CASE("segment_tree_beats: adds on an assigned range, up to the sentinels", "[segment_tree_beats]", (atl::beats_segment_tree<int64, wide>),
				   (clamp_add_engine<int64, wide>)) {
	int64 const top = (std::numeric_limits<int64>::max)() - 1, bottom = (std::numeric_limits<int64>::lowest)() + 1;
	for (std::size_t const n : {std::size_t(2), std::size_t(5), std::size_t(8), std::size_t(33)}) {
		std::vector<int64> a(n);
		for (std::size_t i = 0; i < n; ++i) a[i] = i % 2 ? top : bottom;
		TestType tree(a);
		REQUIRE(tree.sum(0, 1) == bottom);
		tree.assign(0, n, bottom);
		tree.add(0, n, top);  // -1, past every stale top
		tree.add(0, n, top);  // the adds now sum past int64
		tree.add(0, n, -top); // -1
		tree.add(0, n, -top + 3);
		REQUIRE(tree.element(n - 1) == bottom + 3);
		REQUIRE(tree.max_right(0, [&](auto const &v) { return v.width == 0 || v.max == bottom + 3; }) == n);
		tree.assign(0, n, top);
		tree.add(0, n, bottom);
		tree.add(0, n, bottom + 2); // the adds now sum below int64
		REQUIRE(tree.min(n / 2, n) == top + bottom + bottom + 2);
		REQUIRE(tree.sum(0, n) == wide(top + bottom + bottom + 2) * wide(n));
		REQUIRE(tree.element(0) == top + bottom + bottom + 2);
	}
}

namespace {
	/**
	 * @brief The class's tighter bound: every add goes as far as `|a[i]| <= A` lets it, and
	 *        clamps pull the values back, so the adds keep recharging while the overflow
	 *        asserts watch `3A` against `Element` and `4nA` against `Sum`.
	 */
	template<atl::signed_integral Element, atl::signed_integral Sum>
	void bounded_stream(std::mt19937_64 &rng, std::size_t n, Element bound, int steps) {
		std::vector<Element> a(n);
		for (auto &x : a) x = rng() % 2 ? bound : Element(-bound);
		atl::beats_segment_tree<Element, Sum> tree(a);
		for (int s = 0; s < steps && n > 0; ++s) {
			auto const [l, r] = random_range(rng, n);
			Element const x = rng() % 3 == 0 ? uniform(rng, Element(-bound), bound) : (rng() % 2 ? bound : Element(-bound));
			auto const first = a.begin() + std::ptrdiff_t(l), last = a.begin() + std::ptrdiff_t(r);
			switch (rng() % 6) {
			case 0:
				tree.chmin(l, r, x);
				std::ranges::for_each(first, last, [&](Element &v) { v = (std::min)(v, x); });
				break;
			case 1:
				tree.chmax(l, r, x);
				std::ranges::for_each(first, last, [&](Element &v) { v = (std::max)(v, x); });
				break;
			case 2:
			case 3:
				if (l < r) {
					Element const d = rng() % 2 ? Element(bound - *std::ranges::max_element(first, last)) : Element(-bound - *std::ranges::min_element(first, last));
					tree.add(l, r, d);
					std::ranges::for_each(first, last, [&](Element &v) { v = Element(v + d); });
				}
				break;
			case 4:
				tree.assign(l, r, x);
				std::ranges::fill(first, last, x);
				break;
			default:
				Sum want = 0;
				for (auto it = first; it != last; ++it) want = Sum(want + Sum(*it));
				REQUIRE(tree.sum(l, r) == want);
				REQUIRE(tree.element(rng() % n) >= Element(-bound));
				if (l < r) {
					REQUIRE(tree.min(l, r) == *std::ranges::min_element(first, last));
					REQUIRE(tree.max(l, r) == *std::ranges::max_element(first, last));
				}
			}
		}
		for (std::size_t i = 0; i < n; ++i) REQUIRE(tree.element(i) == a[i]);
	}
}

TEST_CASE("segment_tree_beats: a stream bounded at all times fits 3A and 4nA", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : oracle_sizes(rng)) { // n <= 70
		bounded_stream<atl::int32, atl::int32>(rng, n, 7'000'000, 600);
		bounded_stream<atl::int16, atl::int16>(rng, n, 116, 600);
		bounded_stream<atl::int8, atl::int32>(rng, n, 42, 600);
		REQUIRE(4 * 70 * 7'000'000 <= (std::numeric_limits<atl::int32>::max)());
		REQUIRE(4 * 70 * 116 <= (std::numeric_limits<atl::int16>::max)());
		REQUIRE(3 * 42 < (std::numeric_limits<atl::int8>::max)());
	}
}

TEST_CASE("segment_tree_beats: adds on a clamped-flat range, up to the sentinels", "[segment_tree_beats]") {
	int64 const top = (std::numeric_limits<int64>::max)() - 1, bottom = (std::numeric_limits<int64>::lowest)() + 1;
	std::vector<int64> const high{top, top - 1, top - 2, top - 3}, low{bottom, bottom + 1, bottom + 2, bottom + 3};
	atl::beats_segment_tree<int64, wide> below(high), above(low);
	below.chmin(0, 4, -5);
	below.add(0, 4, top - 1);
	REQUIRE(below.element(2) == top - 6);
	REQUIRE(below.sum(1, 3) == wide(top - 6) * 2);
	above.chmax(0, 4, 5);
	above.add(0, 4, bottom + 1);
	REQUIRE(above.element(2) == bottom + 6);
	REQUIRE(above.sum(1, 3) == wide(bottom + 6) * 2);
}

// Each chmin fits Sum at the root, but the leaf's whole way down does not: its difference must never be computed.
TEMPLATE_TEST_CASE("segment_tree_beats: a stale single value far from the parent's extreme, narrow Sum", "[segment_tree_beats]",
				   (atl::beats_segment_tree<atl::int32>), (clamp_add_engine<atl::int32>)) {
	atl::int32 const far = 2'100'000'000, near = 1'070'000'000;
	TestType falling(std::array{far, atl::int32(-near)});
	falling.chmin(0, 2, 500'000'000);
	falling.chmin(0, 2, -1'000'000'000);
	REQUIRE(falling.element(0) == -1'000'000'000);
	REQUIRE(falling.element(1) == -near);
	REQUIRE(falling.sum(0, 2) == -1'000'000'000 - near);
	TestType rising(std::array{atl::int32(-far), near});
	rising.chmax(0, 2, -500'000'000);
	rising.chmax(0, 2, 1'000'000'000);
	REQUIRE(rising.element(0) == 1'000'000'000);
	REQUIRE(rising.element(1) == near);
	REQUIRE(rising.sum(0, 2) == 1'000'000'000 + near);
}

TEST_CASE("segment_tree_beats: engine fold and point apply and max_right / min_left", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	using clamp = atl::beats::clamp_add<int64>;
	for (std::size_t const n : oracle_sizes(rng)) {
		std::vector<int64> a(n);
		for (auto &x : a) x = uniform<int64>(rng, 0, 9);
		clamp_add_engine<int64> tree(a);
		for (int s = 0; s < 300 && n > 0; ++s) {
			std::size_t l = rng() % (n + 1), r = rng() % (n + 1);
			if (l > r) std::swap(l, r);
			switch (rng() % 4) {
			case 0: { // point apply of a general clamp-add tag
				std::size_t const i = rng() % n;
				int64 const lower = uniform<int64>(rng, 0, 9), upper = uniform<int64>(rng, lower, 9);
				clamp const f{uniform<int64>(rng, -2, 2), lower, upper};
				tree.apply(i, f);
				a[i] = (std::min)((std::max)(a[i] + f.add, lower), upper);
				break;
			}
			case 1: { // range apply of a general clamp-add tag
				int64 const lower = uniform<int64>(rng, 0, 9), upper = uniform<int64>(rng, lower, 9);
				clamp const f{uniform<int64>(rng, -2, 2), lower, upper};
				tree.apply(l, r, f);
				for (std::size_t i = l; i < r; ++i) a[i] = (std::min)((std::max)(a[i] + f.add, lower), upper);
				break;
			}
			case 2: { // largest r' with every value of [l, r') at most t
				int64 const t = uniform<int64>(rng, 0, 9);
				std::size_t want = l;
				while (want < n && a[want] <= t) ++want;
				REQUIRE(tree.max_right(l, [&](auto const &v) { return v.max <= t; }) == want);
				break;
			}
			default: { // smallest l' with the sum of [l', r) at most t (values are nonnegative)
				int64 const t = uniform<int64>(rng, 0, 30);
				std::size_t want = r;
				int64 total = 0;
				while (want > 0 && total + a[want - 1] <= t) total += a[--want];
				REQUIRE(tree.min_left(r, [&](auto const &v) { return v.sum <= t; }) == want);
			}
			}
			std::size_t const i = rng() % n;
			REQUIRE(tree.at(i).max == a[i]);
		}
		if (n > 0) REQUIRE(tree.fold().sum == std::accumulate(a.begin(), a.end(), int64(0)));
	}
}

namespace {
	/** @brief Compares sum, min, max over `[l, r)` and one element with the brute-force array. */
	template<typename Tree, atl::integral Element, atl::integral Sum>
		requires requires(Tree &tree) { tree.max(0, 1); tree.element(0); }
	void check_queries(Tree &tree, std::vector<Element> const &a, std::size_t l, std::size_t r, std::mt19937_64 &rng) {
		if constexpr (requires { tree.sum(l, r); }) {
			Sum want = 0;
			for (std::size_t i = l; i < r; ++i) want = Sum(want + Sum(a[i]));
			REQUIRE(tree.sum(l, r) == want);
		}
		if (l < r) {
			auto const first = a.begin() + std::ptrdiff_t(l), last = a.begin() + std::ptrdiff_t(r);
			if constexpr (requires { tree.min(l, r); }) REQUIRE(tree.min(l, r) == *std::ranges::min_element(first, last));
			REQUIRE(tree.max(l, r) == *std::ranges::max_element(first, last));
		}
		if (!a.empty()) {
			std::size_t const i = rng() % a.size();
			REQUIRE(tree.element(i) == a[i]);
		}
	}
}

TEST_CASE("segment_tree_beats: modulo and assign and sum vs brute force", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	using tree_type = atl::segment_tree_beats<atl::beats::mod_sum<int64>, atl::beats::assign<int64>>;
	for (std::size_t const n : oracle_sizes(rng)) {
		int64 const top = rng() % 2 ? 60 : 1'000'000'000'000;
		std::vector<int64> a(n);
		for (auto &x : a) x = uniform<int64>(rng, 0, top);
		tree_type tree(a);
		for (int s = 0; s < 300; ++s) {
			auto const [l, r] = random_range(rng, n);
			switch (rng() % 4) {
			case 0: {
				int64 const m = uniform<int64>(rng, 1, rng() % 2 ? 70 : top + 5);
				tree.modulo(l, r, m);
				for (std::size_t i = l; i < r; ++i) a[i] %= m;
				break;
			}
			case 1: {
				int64 const x = uniform<int64>(rng, 0, top);
				tree.assign(l, r, x);
				std::ranges::fill(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r), x);
				break;
			}
			case 2:
				if (n > 0) {
					std::size_t const i = rng() % n;
					a[i] = uniform<int64>(rng, 0, top);
					tree.set(i, a[i]);
				}
				break;
			default:
				check_queries<tree_type, int64, int64>(tree, a, l, r, rng);
			}
		}
	}
}

TEST_CASE("segment_tree_beats: unsigned modulo with wider sums", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	using tree_type = atl::segment_tree_beats<atl::beats::mod_sum<uint32, uint64>, atl::beats::assign<uint32>>;
	for (std::size_t const n : oracle_sizes(rng)) {
		std::vector<uint32> a(n);
		for (auto &x : a) x = uint32(rng());
		tree_type tree(a);
		for (int s = 0; s < 200; ++s) {
			auto const [l, r] = random_range(rng, n);
			if (rng() % 3 == 0) {
				uint32 const m = uint32(1 + rng() % (rng() % 2 ? 100 : 4'000'000'000));
				tree.modulo(l, r, m);
				for (std::size_t i = l; i < r; ++i) a[i] %= m;
			} else if (rng() % 2) {
				uint32 const x = uint32(rng());
				tree.assign(l, r, x);
				std::ranges::fill(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r), x);
			} else {
				check_queries<tree_type, uint32, uint64>(tree, a, l, r, rng);
			}
		}
	}
}

TEST_CASE("segment_tree_beats: floor sqrt and divide and add vs brute force", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	using tree_type = atl::segment_tree_beats<atl::beats::sqrt_add_sum<int64>, atl::beats::add<int64>>;
	auto const exact_sqrt = [](int64 v) {
		int64 r = 0;
		while ((r + 1) * (r + 1) <= v) ++r;
		return r;
	};
	auto const floor_div = [](int64 v, int64 d) { return v >= 0 ? v / d : -((-v + d - 1) / d); };
	for (std::size_t const n : oracle_sizes(rng)) {
		int64 const top = rng() % 2 ? 50 : 1'000'000'000;
		std::vector<int64> a(n);
		for (auto &x : a) x = uniform<int64>(rng, 0, top);
		tree_type tree(a);
		for (int s = 0; s < 300; ++s) {
			auto const [l, r] = random_range(rng, n);
			switch (rng() % 5) {
			case 0: // floor sqrt, only where every value is nonnegative
				if (std::ranges::all_of(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r), [](int64 v) { return v >= 0; })) {
					tree.floor_sqrt(l, r);
					for (std::size_t i = l; i < r; ++i) a[i] = exact_sqrt(a[i]);
				}
				break;
			case 1: {
				int64 const d = uniform<int64>(rng, 1, 6);
				tree.divide(l, r, d);
				for (std::size_t i = l; i < r; ++i) a[i] = floor_div(a[i], d);
				break;
			}
			case 2: {
				int64 const x = uniform<int64>(rng, -5, top / 2);
				tree.add(l, r, x);
				for (std::size_t i = l; i < r; ++i) a[i] += x;
				break;
			}
			case 3:
				if (n > 0) {
					std::size_t const i = rng() % n;
					a[i] = uniform<int64>(rng, 0, top);
					tree.set(i, a[i]);
				}
				break;
			default:
				check_queries<tree_type, int64, int64>(tree, a, l, r, rng);
			}
		}
	}
}

TEST_CASE("segment_tree_beats: floor sqrt of large values", "[segment_tree_beats]") {
	int64 const big = (std::numeric_limits<int64>::max)() - 1;
	std::vector<int64> const a{big, big - 1, 3037000499LL * 3037000499LL, 3037000499LL * 3037000499LL - 1, 4, 3, 0, 1};
	atl::segment_tree_beats<atl::beats::sqrt_add_sum<int64, wide>, atl::beats::add<int64>> tree(a);
	tree.floor_sqrt(0, 8);
	std::vector<int64> const want{3037000499LL, 3037000499LL, 3037000499LL, 3037000498LL, 2, 1, 0, 1};
	for (std::size_t i = 0; i < 8; ++i) REQUIRE(tree.element(i) == want[i]);
	static_assert(atl::detail::beats_floor_sqrt(int64(9223372036854775807)) == 3037000499LL);
	static_assert(atl::detail::beats_floor_sqrt(int64(15)) == 3 && atl::detail::beats_floor_sqrt(int64(16)) == 4);
	atl::int128 const huge = (std::numeric_limits<atl::int128>::max)();
	atl::int128 const root = atl::detail::beats_floor_sqrt(huge);
	REQUIRE(root <= huge / root);
	REQUIRE(root + 1 > huge / (root + 1));
	static_assert(atl::detail::beats_floor_sqrt(atl::int128(1) << 100) == atl::int128(1) << 50);
}

TEST_CASE("segment_tree_beats: bitwise and / or with max vs brute force", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto const run = [&]<atl::unsigned_integral Element>(Element) {
		using tree_type = atl::segment_tree_beats<atl::beats::bits_max<Element>, atl::beats::bit_mask<Element>>;
		for (std::size_t const n : oracle_sizes(rng)) {
			std::vector<Element> a(n);
			for (auto &x : a) x = Element(rng() & rng());
			tree_type tree(a);
			for (int s = 0; s < 300; ++s) {
				auto const [l, r] = random_range(rng, n);
				Element const x = Element(rng() % 2 ? rng() | rng() : rng() & rng() & rng());
				switch (rng() % 4) {
				case 0:
					tree.bit_and(l, r, x);
					for (std::size_t i = l; i < r; ++i) a[i] &= x;
					break;
				case 1: {
					Element const y = Element(x & Element(rng() & rng()));
					tree.bit_or(l, r, y);
					for (std::size_t i = l; i < r; ++i) a[i] |= y;
					break;
				}
				case 2:
					if (n > 0) {
						std::size_t const i = rng() % n;
						a[i] = Element(rng());
						tree.set(i, a[i]);
					}
					break;
				default:
					if (l < r) REQUIRE(tree.max(l, r) == *std::ranges::max_element(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r)));
					if (n > 0) {
						std::size_t const i = rng() % n;
						REQUIRE(tree.element(i) == a[i]);
					}
				}
			}
			for (std::size_t i = 0; i < n; ++i) REQUIRE(tree.element(i) == a[i]);
		}
	};
	run(uint32{});
	run(uint8{});
	run(uint64{});
}

namespace {
	/** @brief A hand-written chmin + sum + max algebra, as a user would pass it. */
	struct chmin_node {
		int64 max = (std::numeric_limits<int64>::lowest)(), second = (std::numeric_limits<int64>::lowest)(), count = 0, sum = 0;
	};

	auto const chmin_op = [](chmin_node const &a, chmin_node const &b) {
		if (a.max == b.max) return chmin_node{a.max, (std::max)(a.second, b.second), a.count + b.count, a.sum + b.sum};
		chmin_node const &hi = a.max > b.max ? a : b, &lo = a.max > b.max ? b : a;
		return chmin_node{hi.max, (std::max)(hi.second, lo.max), hi.count, a.sum + b.sum};
	};

	auto const chmin_mapping = [](int64 const &x, chmin_node &s) {
		if (x >= s.max) return true;
		if (x <= s.second) return false;
		s.sum -= (s.max - x) * s.count;
		s.max = x;
		return true;
	};

	auto const chmin_compose = [](int64 f, int64 g) { return (std::min)(f, g); };

	/** @brief The library mapping, counting its calls into @p calls. */
	struct counting_mapping {
		std::size_t *calls = nullptr;

		template<typename Tag, typename Value>
			requires atl::fallible_action<atl::beats::mapping, Tag, Value>
		constexpr bool operator()(Tag const &f, Value &s) const {
			++*calls;
			return atl::beats::mapping{}(f, s);
		}
	};
}

// Deduction from the element identity rejects narrowing the range's values.
template<typename Value, typename Identity>
concept beats_deducible = requires(std::vector<Value> &v, Identity id) {
	atl::segment_tree_beats(v, id, (std::numeric_limits<int64>::max)(), chmin_op, chmin_mapping, chmin_compose);
};
static_assert(beats_deducible<chmin_node, chmin_node>);
static_assert(!beats_deducible<int, chmin_node>);

TEST_CASE("segment_tree_beats: hand-written chmin algebra through CTAD", "[segment_tree_beats]") {
	auto const seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t const n : oracle_sizes(rng)) {
		std::vector<int64> a(n);
		std::vector<chmin_node> init;
		for (auto &x : a) {
			x = uniform<int64>(rng, 0, 20);
			init.push_back({x, (std::numeric_limits<int64>::lowest)(), 1, x});
		}
		atl::segment_tree_beats tree(init, chmin_node{}, (std::numeric_limits<int64>::max)(), chmin_op, chmin_mapping, chmin_compose);
		for (int s = 0; s < 300; ++s) {
			auto const [l, r] = random_range(rng, n);
			if (rng() % 2) {
				int64 const x = uniform<int64>(rng, 0, 20);
				tree.apply(l, r, x);
				for (std::size_t i = l; i < r; ++i) a[i] = (std::min)(a[i], x);
			} else {
				chmin_node const f = tree.fold(l, r);
				REQUIRE(f.sum == std::accumulate(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r), int64(0)));
				if (l < r) REQUIRE(f.max == *std::ranges::max_element(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r)));
			}
		}
	}
}

TEST_CASE("segment_tree_beats: beat visits stay within the amortized bound", "[segment_tree_beats]") {
	std::size_t const n = 1 << 12, q = 1 << 12;
	double const lg = 12;
	std::mt19937_64 rng(7);
	std::vector<atl::beats::extremes<int64>> a(n);
	for (auto &x : a) x = atl::beats::extremes<int64>::of(uniform<int64>(rng, 0, 1'000'000'000));
	SECTION("chmin + sum, no add: O((n + q) log n)") {
		std::size_t calls = 0;
		atl::segment_tree_beats<atl::beats::extremes<int64>, atl::beats::clamp_add<int64>, atl::beats::merge, counting_mapping> tree(
			a, atl::beats::extremes<int64>::identity(), atl::beats::clamp_add<int64>::identity(), {}, counting_mapping{&calls}, {});
		for (std::size_t s = 0; s < q; ++s) {
			auto const [l, r] = random_range(rng, n);
			tree.chmin(l, r, tree.max(0, n) - int64(rng() % 1000));
		}
		REQUIRE(double(calls) <= 4 * double(n + q) * lg);
	}
	SECTION("add-recharge adversary: O((n + q) log^2 n)") {
		std::size_t calls = 0;
		atl::segment_tree_beats<atl::beats::extremes<int64>, atl::beats::clamp_add<int64>, atl::beats::merge, counting_mapping> tree(
			std::vector(n, atl::beats::extremes<int64>::of(0)), atl::beats::extremes<int64>::identity(), atl::beats::clamp_add<int64>::identity(), {},
			counting_mapping{&calls}, {});
		for (std::size_t s = 0; s < q; ++s) {
			std::size_t const m = rng() % n;
			tree.add(0, m, 1);
			tree.add(m, n, 2);
			tree.chmin(0, n, tree.max(0, n) - 1);
		}
		REQUIRE(double(calls) <= 4 * double(n + q) * lg * lg);
	}
}

TEMPLATE_TEST_CASE("segment_tree_beats: preconditions of the named members abort", "[segment_tree_beats]", atl::beats_segment_tree<int64>,
				   clamp_add_engine<int64>) {
	std::vector<int64> const a{1, 2, 3, 4};
	using tree_type = TestType;
	REQUIRE(atl_test::aborts([&] { tree_type(a).set(4, 0); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).element(4); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).chmin(3, 2, 0); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).chmax(0, 5, 0); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).sum(2, 5); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).min(1, 1); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).max(2, 2); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).chmin(0, 4, (std::numeric_limits<int64>::max)()); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).chmax(0, 4, (std::numeric_limits<int64>::lowest)()); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).assign(0, 4, (std::numeric_limits<int64>::lowest)()); }));
	REQUIRE(atl_test::aborts([] { tree_type(std::vector<int64>{(std::numeric_limits<int64>::lowest)()}); }));
	REQUIRE(atl_test::aborts([] { tree_type(std::vector<int64>{(std::numeric_limits<int64>::max)() - 1}).add(0, 1, 1); }));
	REQUIRE(atl_test::aborts([&] { tree_type(a).set(0, (std::numeric_limits<int64>::max)()); }));
	REQUIRE(atl_test::aborts([] { tree_type((std::numeric_limits<std::size_t>::max)()); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).max_right(5, [](auto const &) { return true; }); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).min_left(5, [](auto const &) { return true; }); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).max_right(0, [](auto const &) { return false; }); }));
	REQUIRE(atl_test::aborts([&] { (void)tree_type(a).min_left(4, [](auto const &) { return false; }); }));
	// a sum overflowing Sum, with the same elements on either side of the build
	using narrow = std::conditional_t<std::same_as<tree_type, atl::beats_segment_tree<int64>>, atl::beats_segment_tree<atl::int32>, clamp_add_engine<atl::int32>>;
	REQUIRE(atl_test::aborts([] { narrow(std::vector<atl::int32>(4, 1'000'000'000)); }));
	REQUIRE(atl_test::aborts([] {
		narrow t(std::vector<atl::int32>(4, 500'000'000));
		t.add(0, 4, 500'000'000);
	}));
}

TEST_CASE("segment_tree_beats: preconditions of the engine alone abort", "[segment_tree_beats]") {
	std::vector<int64> const a{1, 2, 3, 4};
	using tree_type = clamp_add_engine<int64>;
	REQUIRE(atl_test::aborts([&] { tree_type(a).apply(0, 4, atl::beats::clamp_add<int64>{0, 5, 3}); }));
	REQUIRE(atl_test::aborts([&] {
		auto bad = atl::beats::extremes<int64>::of(3);
		bad.width = 2;
		tree_type(a).set(0, bad);
	}));
	REQUIRE(atl_test::aborts([] {
		std::vector<atl::beats::extremes<int64>> const leaves{atl::beats::extremes<int64>::identity()};
		atl::segment_tree_beats tree(leaves, atl::beats::extremes<int64>::identity(), atl::beats::clamp_add<int64>::identity(),
									 atl::beats::merge{}, atl::beats::mapping{}, atl::beats::compose{});
	}));
	using mod_tree = atl::segment_tree_beats<atl::beats::mod_sum<int64>, atl::beats::assign<int64>>;
	REQUIRE(atl_test::aborts([&] { mod_tree(a).modulo(0, 4, 0); }));
	REQUIRE(atl_test::aborts([] { mod_tree(std::vector<int64>{5, -1}).modulo(0, 2, 3); }));
	using sqrt_tree = atl::segment_tree_beats<atl::beats::sqrt_add_sum<int64>, atl::beats::add<int64>>;
	REQUIRE(atl_test::aborts([] { sqrt_tree(std::vector<int64>{5, -1}).floor_sqrt(0, 2); }));
	REQUIRE(atl_test::aborts([&] { sqrt_tree(a).divide(0, 4, 0); }));
	// a mapping that fails on a single element breaks the engine's first law
	auto const never = [](int64 const &, chmin_node &) { return false; };
	REQUIRE(atl_test::aborts([&] {
		atl::segment_tree_beats tree(std::size_t(4), chmin_node{}, int64(0), chmin_op, never, chmin_compose);
		tree.apply(0, 4, int64(1));
	}));
	// a rule that declines a single element
	REQUIRE(atl_test::aborts([&] { sqrt_tree(a).apply_derived(0, 4, [](auto const &) { return std::optional<atl::beats::add<int64>>{}; }); }));
	// a mapping that absorbs on a pair but not on its elements breaks the push law
	auto const only_pairs = [](int64 const &x, chmin_node &s) {
		if (s.count == 1) return false;
		s.max = (std::min)(s.max, x);
		return true;
	};
	REQUIRE(atl_test::aborts([&] {
		std::vector<chmin_node> const init(4, chmin_node{5, (std::numeric_limits<int64>::lowest)(), 1, 5});
		atl::segment_tree_beats tree(init, chmin_node{}, (std::numeric_limits<int64>::max)(), chmin_op, only_pairs, chmin_compose);
		tree.apply(0, 2, int64(1));
		(void)tree.fold(0, 1);
	}));
}

TEST_CASE("beats_segment_tree: at, fold and empty agree with the engine", "[segment_tree_beats]") {
	std::mt19937_64 rng(20261101);
	auto const same = [](auto const &a, auto const &b) {
		return a.max == b.max && a.second_max == b.second_max && a.min == b.min && a.second_min == b.second_min && a.sum == b.sum
			&& a.max_count == b.max_count && a.min_count == b.min_count && a.width == b.width;
	};
	for (std::size_t const n : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(8), std::size_t(33)}) {
		std::vector<int64> a(n);
		for (auto &x : a) x = int64(rng() % 41) - 20;
		atl::beats_segment_tree<int64> tree(a);
		clamp_add_engine<int64> engine(a);
		for (int step = 0; step < 400; ++step) {
			std::size_t l = rng() % (n + 1), r = rng() % (n + 1);
			if (l > r) std::swap(l, r);
			int64 const x = int64(rng() % 41) - 20;
			switch (rng() % 4) {
			case 0:
				tree.chmin(l, r, x), engine.chmin(l, r, x);
				break;
			case 1:
				tree.chmax(l, r, x), engine.chmax(l, r, x);
				break;
			case 2:
				tree.add(l, r, x % 3), engine.add(l, r, x % 3);
				break;
			default:
				tree.assign(l, r, x), engine.assign(l, r, x);
				break;
			}
			std::size_t const i = rng() % n;
			REQUIRE(same(tree.at(i), engine.at(i)));
			std::size_t p = rng() % (n + 1), q = rng() % (n + 1);
			if (p > q) std::swap(p, q);
			REQUIRE(same(tree.fold(p, q), engine.fold(p, q)));
			REQUIRE(same(tree.fold(), engine.fold()));
		}
		REQUIRE(!tree.empty());
	}
	REQUIRE(atl::beats_segment_tree<int64>().empty());
}
