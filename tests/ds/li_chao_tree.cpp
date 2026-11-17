#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
#include <atl/experimental/ds/dynamic_li_chao_tree_variants.hpp>
#include <atl/experimental/ds/li_chao_tree_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::int64;

namespace {
	template<typename Tree>
	concept envelope_tree = requires(Tree &tree, typename Tree::function_type f, typename Tree::coordinate_type x) {
		tree.insert(f);
		tree.insert(f, x, x);
		{ tree.query(x) } -> std::same_as<std::optional<typename Tree::value_type>>;
	};

	/**
	 * @brief Every inserted segment, scanned per query; full lines are
	 *        `[min, max)`. The oracle of the li chao tree families, library and
	 *        experimental (tests/ds/li_chao_tree.cpp throughout).
	 */
	template<std::strict_weak_order<int64, int64> Compare = std::less<>>
	struct brute_envelope {
		struct piece {
			int64 slope, intercept, l, r;
		};

		std::vector<piece> pieces;
		Compare comp{};

		void insert(int64 a, int64 b, int64 l = std::numeric_limits<int64>::min(),
					int64 r = std::numeric_limits<int64>::max()) {
			pieces.push_back({a, b, l, r});
		}

		[[nodiscard]] std::optional<int64> query(int64 x) const {
			std::optional<int64> best;
			for (auto const &p : pieces)
				if (p.l <= x && x < p.r) {
					int64 const v = p.slope * x + p.intercept;
					if (!best || comp(v, *best)) best = v;
				}
			return best;
		}
	};

	/**
	 * @brief Random line_add_get_min / segment_add_get_min traffic on the
	 *        domain points `points` against the brute force: inserts of lines
	 *        and segments (ends anywhere, in or out of the domain) interleaved
	 *        with queries at domain points.
	 */
	template<typename Tree, std::strict_weak_order<int64, int64> Compare>
		requires requires(Tree &tree, int64 x) {
			tree.insert(x, x);
			tree.insert(x, x, x, x);
			{ tree.query(x) } -> std::same_as<std::optional<int64>>;
		}
	void stress(Tree &tree, std::vector<int64> const &points, std::mt19937_64 &rng, int operations, bool has_segments,
				Compare comp) {
		brute_envelope<Compare> brute{{}, comp};
		int64 const lowest = points.front() - 3, span = points.back() - points.front() + 7;
		for (int op = 0; op < operations; ++op) {
			int const kind = int(rng() % 3);
			if (kind == 0) {
				int64 const x = points[rng() % points.size()];
				REQUIRE(tree.query(x) == brute.query(x));
				continue;
			}
			int64 const a = int64(rng() % 21) - 10, b = int64(rng() % 201) - 100;
			if (kind == 1 || !has_segments) {
				tree.insert(a, b);
				brute.insert(a, b);
			} else {
				int64 l = lowest + int64(rng() % std::uint64_t(span)), r = lowest + int64(rng() % std::uint64_t(span));
				if (l > r) std::swap(l, r);
				tree.insert(a, b, l, r);
				brute.insert(a, b, l, r);
			}
		}
		for (int64 const x : points) REQUIRE(tree.query(x) == brute.query(x));
	}
}

TEST_CASE("li_chao_tree over an integer range matches the brute force", "[li_chao_tree]") {
	std::mt19937_64 rng(51);
	for (int iter = 0; iter < 400; ++iter) {
		int64 const lo = int64(rng() % 41) - 20, n = 1 + int64(rng() % 40);
		std::vector<int64> points(static_cast<std::size_t>(n));
		for (int64 i = 0; i < n; ++i) points[std::size_t(i)] = lo + i;
		if (iter % 2 == 0) {
			atl::li_chao_tree<int64> tree(lo, lo + n);
			stress(tree, points, rng, 150, iter % 4 == 0, std::less<>{});
		} else {
			atl::li_chao_tree<int64, std::greater<>> tree(lo, lo + n);
			stress(tree, points, rng, 150, iter % 4 == 1, std::greater<>{});
			atl::experimental::li_chao_tree_by_inline_flags<int64, std::greater<>> flagged(lo, lo + n);
			stress(flagged, points, rng, 150, iter % 4 == 1, std::greater<>{});
		}
	}
}

TEST_CASE("li_chao_tree over given coordinates matches the brute force", "[li_chao_tree]") {
	std::mt19937_64 rng(52);
	for (int iter = 0; iter < 400; ++iter) {
		std::vector<int64> points;
		for (std::size_t i = 0, count = 1 + rng() % 30; i < count; ++i) points.push_back(int64(rng() % 201) - 100);
		std::ranges::sort(points);
		points.erase(std::ranges::unique(points).begin(), points.end());
		atl::li_chao_tree<int64> tree(points);
		CHECK(tree.size() == points.size());
		stress(tree, points, rng, 150, iter % 2 == 0, std::less<>{});
		atl::experimental::li_chao_tree_by_inline_flags<int64> flagged(points);
		stress(flagged, points, rng, 150, iter % 2 == 0, std::less<>{});
	}
}

namespace {
	/**
	 * @brief `check.operator()<Tree>()` for every query-order and clip-overwrite contender of
	 *        @ref atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite , each knob alone,
	 *        both, and neither (the tree with both features disabled).
	 */
	template<atl::integral Coordinate, typename Compare = std::less<>, typename Check>
		requires requires(Check const &check) { check.template operator()<atl::dynamic_li_chao_tree<Coordinate, Compare>>(); }
	void for_each_tuned_tree(Check const &check) {
		using function = atl::linear_function<Coordinate>;
		auto const tuned = [&]<bool IsQueryWholeFirst, bool OverwritesLostClips> {
			check.template operator()<atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<
				Coordinate, Compare, function, IsQueryWholeFirst, OverwritesLostClips>>();
		};
		tuned.template operator()<false, false>();
		tuned.template operator()<true, false>();

		tuned.template operator()<false, true>();
		tuned.template operator()<true, true>();
	}

	/** @brief `check.operator()<Tree>(Compare{})` for the library dynamic tree and each experimental contender. */
	template<typename Compare, typename Check>
		requires requires(Check const &check) {
			check.template operator()<atl::dynamic_li_chao_tree<int64, Compare>>(Compare{});
		}
	void for_each_dynamic_tree(Check const &check) {
		namespace experimental = atl::experimental;
		check.template operator()<atl::dynamic_li_chao_tree<int64, Compare>>(Compare{});
		check.template operator()<experimental::dynamic_li_chao_tree_by_materialized_paths<int64, Compare>>(Compare{});
		check.template operator()<experimental::dynamic_li_chao_tree_by_two_bound_clips<int64, Compare>>(Compare{});
		for_each_tuned_tree<int64, Compare>([&]<envelope_tree Tree> { check.template operator()<Tree>(Compare{}); });
	}
}

TEST_CASE("dynamic_li_chao_tree matches the brute force", "[li_chao_tree]") {
	std::mt19937_64 rng(53);
	for (int iter = 0; iter < 400; ++iter) {
		// a small window of query points inside a domain of width 2^59 (slopes
		// up to 10 keep every line within int64 there)
		std::vector<int64> points;
		for (std::size_t i = 0, count = 1 + rng() % 30; i < count; ++i) points.push_back(int64(rng() % 201) - 100);
		std::ranges::sort(points);
		points.erase(std::ranges::unique(points).begin(), points.end());
		int64 const far = int64(1) << 58;
		bool const has_segments = iter % 3 != 0, is_wide = iter % 4 < 2;
		int64 const lo = is_wide ? -far : points.front(), hi = is_wide ? far : points.back() + 1;
		auto const check = [&]<envelope_tree Tree, std::strict_weak_order<int64, int64> Compare>(Compare comp) {
			Tree tree(lo, hi, comp);
			stress(tree, points, rng, 150, has_segments, comp);
		};
		if (iter % 2 == 0)
			for_each_dynamic_tree<std::less<>>(check);
		else
			for_each_dynamic_tree<std::greater<>>(check);
	}
}

namespace {
	/**
	 * @brief Segments with ends spread over a wide domain `[lo, hi)` (deep
	 *        boundary paths, later insertions crossing earlier ones' ends),
	 *        with queries at and beside every segment end, against the brute force.
	 */
	template<typename Tree, std::strict_weak_order<int64, int64> Compare>
		requires std::constructible_from<Tree, int64, int64, Compare> && requires(Tree &tree, int64 x) {
			tree.insert(x, x);
			tree.insert(x, x, x, x);
			{ tree.query(x) } -> std::same_as<std::optional<int64>>;
		}
	void stress_wide(int64 lo, int64 hi, std::mt19937_64 &rng, Compare comp) {
		Tree tree(lo, hi, comp);
		brute_envelope<Compare> brute{{}, comp};
		std::vector<int64> probes{lo, hi - 1};
		auto const point = [&] { return lo + int64(rng() % std::uint64_t(hi - lo)); };
		for (int op = 0; op < 300; ++op) {
			int64 const a = int64(rng() % 21) - 10, b = int64(rng() % 2001) - 1000;
			if (op % 7 == 0) {
				tree.insert(a, b);
				brute.insert(a, b);
				continue;
			}
			int64 l = point(), r = point();
			if (l > r) std::swap(l, r);
			if (op % 5 == 0) r = l + int64(rng() % 4); // short: one or two points, or empty
			if (op % 11 == 0) r = hi;
			tree.insert(a, b, l, r);
			brute.insert(a, b, l, r);
			for (int64 const x : {l - 1, l, r - 1, r})
				if (lo <= x && x < hi) probes.push_back(x);
			if (op % 3 == 0) {
				int64 const x = probes[rng() % probes.size()];
				REQUIRE(tree.query(x) == brute.query(x));
			}
		}
		for (int64 const x : probes) REQUIRE(tree.query(x) == brute.query(x));
	}
}

TEST_CASE("dynamic li chao trees: segments over a wide domain match the brute force", "[li_chao_tree]") {
	std::mt19937_64 rng(57);
	for (int iter = 0; iter < 60; ++iter) {
		// widths from tiny to 2e9 (slopes up to 10 keep every value within int64)
		int64 const width = iter % 3 == 0 ? 1 + int64(rng() % 64) : iter % 3 == 1 ? 2'000'000'001
																				  : 1 + int64(rng() % 100'000);
		int64 const lo = int64(rng() % 2001) - 1000 - width / 2, hi = lo + width;
		auto const check = [&]<envelope_tree Tree, std::strict_weak_order<int64, int64> Compare>(Compare comp) { stress_wide<Tree>(lo, hi, rng, comp); };
		for_each_dynamic_tree<std::less<>>(check);
		for_each_dynamic_tree<std::greater<>>(check);
	}
}

namespace {
	/** @brief `(x - shift)^2 + lift`: two members differ by a linear function of `x`, so they cross at most once. */
	struct parabola {
		int64 shift = 0, lift = 0;

		[[nodiscard]] constexpr int64 operator()(int64 x) const { return (x - shift) * (x - shift) + lift; }
	};

	/** @brief `scale * sqrt(x) + lift` on `x >= 0`: differences are monotone in `x`, so members cross at most once. */
	struct root_curve {
		double scale = 0, lift = 0;

		[[nodiscard]] double operator()(double x) const { return scale * std::sqrt(x) + lift; }
	};

	/**
	 * @brief Random whole-domain and segment insertions of a non-linear
	 *        family interleaved with queries, against the brute-force best
	 *        over the covering members.
	 */
	template<typename Tree, typename Function, typename Coordinate, typename Draw, typename Compare>
		requires std::is_invocable_r_v<Function, Draw &> && atl::envelope_function<Function, Coordinate, Compare>
			  && std::totally_ordered<Coordinate> && requires(Tree &tree, Function const &f, Coordinate x) {
					 tree.insert(f);
					 tree.insert(f, x, x);
					 { tree.query(x) } -> std::same_as<std::optional<atl::envelope_value_t<Function, Coordinate>>>;
				 }
	void stress_family(Tree &tree, std::vector<Coordinate> const &points, std::mt19937_64 &rng, Draw draw,
					   Compare comp) {
		struct piece {
			Function f;
			Coordinate l, r;
			bool is_whole;
		};

		std::vector<piece> pieces;
		auto const brute = [&](Coordinate x) {
			std::optional<decltype(Function{}(x))> best;
			for (auto const &p : pieces)
				if (p.is_whole || (!(x < p.l) && x < p.r))
					if (auto const v = p.f(x); !best || comp(v, *best)) best = v;
			return best;
		};
		for (int op = 0; op < 200; ++op) {
			if (op % 3 == 0) {
				Coordinate const x = points[rng() % points.size()];
				REQUIRE(tree.query(x) == brute(x));
				continue;
			}
			Function const f = draw();
			if (op % 3 == 1) {
				tree.insert(f);
				pieces.push_back({f, {}, {}, true});
			} else {
				Coordinate l = points[rng() % points.size()], r = points[rng() % points.size()];
				if (r < l) std::swap(l, r);
				tree.insert(f, l, r);
				pieces.push_back({f, l, r, false});
			}
		}
		for (Coordinate const x : points) REQUIRE(tree.query(x) == brute(x));
	}
}

TEST_CASE("li chao trees over a non-linear family (shifted parabolas)", "[li_chao_tree]") {
	std::mt19937_64 rng(55);
	auto const draw = [&rng] { return parabola{int64(rng() % 41) - 20, int64(rng() % 401) - 200}; };
	for (int iter = 0; iter < 300; ++iter) {
		int64 const lo = int64(rng() % 41) - 20, n = 1 + int64(rng() % 40);
		std::vector<int64> range(static_cast<std::size_t>(n));
		for (int64 i = 0; i < n; ++i) range[std::size_t(i)] = lo + i;
		std::vector<int64> sparse;
		for (std::size_t i = 0, count = 1 + rng() % 30; i < count; ++i) sparse.push_back(int64(rng() % 201) - 100);
		std::ranges::sort(sparse);
		sparse.erase(std::ranges::unique(sparse).begin(), sparse.end());
		if (iter % 2 == 0) {
			atl::li_chao_tree<int64, std::less<>, parabola> over_range(lo, lo + n);
			stress_family<decltype(over_range), parabola>(over_range, range, rng, draw, std::less<>{});
			atl::li_chao_tree<int64, std::less<>, parabola> over_points(sparse);
			stress_family<decltype(over_points), parabola>(over_points, sparse, rng, draw, std::less<>{});
			atl::dynamic_li_chao_tree<int64, std::less<>, parabola> dynamic(-1000, 1000);
			stress_family<decltype(dynamic), parabola>(dynamic, sparse, rng, draw, std::less<>{});
			atl::experimental::li_chao_tree_by_inline_flags<int64, std::less<>, parabola> flagged(sparse);
			stress_family<decltype(flagged), parabola>(flagged, sparse, rng, draw, std::less<>{});
		} else {
			// the maximum: crossing at most once does not depend on the direction
			atl::li_chao_tree<int64, std::greater<>, parabola> over_range(lo, lo + n);
			stress_family<decltype(over_range), parabola>(over_range, range, rng, draw, std::greater<>{});
			atl::dynamic_li_chao_tree<int64, std::greater<>, parabola> dynamic(lo, lo + n);
			stress_family<decltype(dynamic), parabola>(dynamic, range, rng, draw, std::greater<>{});
		}
	}
}

TEST_CASE("li_chao_tree over scale * sqrt(x) + lift on double coordinates", "[li_chao_tree]") {
	std::mt19937_64 rng(56);
	auto const draw = [&rng] { return root_curve{double(int64(rng() % 21) - 10), double(int64(rng() % 41) - 20)}; };
	for (int iter = 0; iter < 200; ++iter) {
		std::vector<double> points;
		for (std::size_t i = 0, count = 1 + rng() % 30; i < count; ++i) points.push_back(double(rng() % 400) / 4);
		std::ranges::sort(points);
		points.erase(std::ranges::unique(points).begin(), points.end());
		atl::li_chao_tree<double, std::less<>, root_curve> tree(points);
		stress_family<decltype(tree), root_curve>(tree, points, rng, draw, std::less<>{});
	}
}

namespace {
	/**
	 * @brief Suffix-heavy segment traffic on `tree` over the domain points
	 *        `points` (a non-power-of-two count, mostly): ends exactly at the
	 *        domain's end `hi`, past it, at the last point (which excludes
	 *        it), one past it, or anywhere, mixed with whole-domain inserts and
	 *        queries; checked against the brute-force best over the covering
	 *        members.
	 */
	template<typename Function, typename Tree, typename Draw, std::strict_weak_order<int64, int64> Compare>
		requires atl::envelope_function<Function, int64, Compare> && std::is_invocable_r_v<Function, Draw &>
			  && requires(Tree &tree, Function const &f, int64 x) {
					 tree.insert(f);
					 tree.insert(f, x, x);
					 { tree.query(x) } -> std::same_as<std::optional<int64>>;
				 }
	void stress_suffixes(Tree &tree, std::vector<int64> const &points, int64 hi, std::mt19937_64 &rng, Draw draw,
						 Compare comp) {
		struct piece {
			Function f;
			int64 l, r;
		};

		std::vector<piece> pieces;
		auto const brute = [&](int64 x) {
			std::optional<int64> best;
			for (auto const &p : pieces)
				if (p.l <= x && x < p.r)
					if (int64 const v = p.f(x); !best || comp(v, *best)) best = v;
			return best;
		};
		int64 const far = std::numeric_limits<int64>::max();
		for (int op = 0; op < 120; ++op) {
			if (op % 3 == 0) {
				int64 const x = points[rng() % points.size()];
				REQUIRE(tree.query(x) == brute(x));
				continue;
			}
			Function const f = draw();
			if (op % 10 == 1) {
				tree.insert(f);
				pieces.push_back({f, std::numeric_limits<int64>::min(), far});
				continue;
			}
			int64 const l = op % 5 == 2 ? points.front() - int64(rng() % 3) : points[rng() % points.size()];
			int64 const ends[] = {hi, hi + 1 + int64(rng() % 5), far, points.back(), points.back() + 1,
								  points[rng() % points.size()]};
			int64 const r = ends[rng() % std::size(ends)];
			tree.insert(f, l, r);
			pieces.push_back({f, l, r});
		}
		for (int64 const x : points) REQUIRE(tree.query(x) == brute(x));
	}
}

TEST_CASE("li_chao_tree: segments reaching the last point (padding leaves)", "[li_chao_tree]") {
	std::mt19937_64 rng(58);
	auto const draw_line = [&rng] { return atl::linear_function<int64>{int64(rng() % 21) - 10, int64(rng() % 201) - 100}; };
	auto const draw_parabola = [&rng] { return parabola{int64(rng() % 41) - 20, int64(rng() % 401) - 200}; };
	for (int iter = 0; iter < 300; ++iter) {
		// sizes 1 to 70: powers of two and their neighbours included
		int64 const n = 1 + int64(iter % 70), lo = int64(rng() % 41) - 20;
		std::vector<int64> range(static_cast<std::size_t>(n));
		for (int64 i = 0; i < n; ++i) range[std::size_t(i)] = lo + i;
		std::vector<int64> sparse;
		for (int64 i = 0; i < n; ++i) sparse.push_back(lo + 3 * i + int64(rng() % 3));
		// the domain's end for a coordinate list: just past the last point
		int64 const sparse_hi = sparse.back() + 1;
		if (iter % 2 == 0) {
			atl::li_chao_tree<int64> over_range(lo, lo + n);
			stress_suffixes<atl::linear_function<int64>>(over_range, range, lo + n, rng, draw_line, std::less<>{});
			atl::li_chao_tree<int64> over_points(sparse);
			stress_suffixes<atl::linear_function<int64>>(over_points, sparse, sparse_hi, rng, draw_line, std::less<>{});
			atl::li_chao_tree<int64, std::less<>, parabola> parabolas(lo, lo + n);
			stress_suffixes<parabola>(parabolas, range, lo + n, rng, draw_parabola, std::less<>{});
		} else {
			atl::li_chao_tree<int64, std::greater<>> over_range(lo, lo + n);
			stress_suffixes<atl::linear_function<int64>>(over_range, range, lo + n, rng, draw_line, std::greater<>{});
			atl::li_chao_tree<int64, std::greater<>> over_points(sparse);
			stress_suffixes<atl::linear_function<int64>>(over_points, sparse, sparse_hi, rng, draw_line, std::greater<>{});
			atl::li_chao_tree<int64, std::greater<>, parabola> parabolas(sparse);
			stress_suffixes<parabola>(parabolas, sparse, sparse_hi, rng, draw_parabola, std::greater<>{});
		}
	}
}

TEST_CASE("li chao trees with slopes and coordinates up to 1e9", "[li_chao_tree]") {
	// line_add_get_min / segment_add_get_min: |a|, |x| <= 1e9, |b| <= 1e18,
	// so every value fits int64; checked against an int128 brute force
	std::mt19937_64 rng(54);
	int64 const bound = 1'000'000'000;
	std::vector<int64> points;
	for (int i = 0; i < 300; ++i) points.push_back(int64(rng() % std::uint64_t(2 * bound + 1)) - bound);
	points.push_back(-bound), points.push_back(bound);
	std::ranges::sort(points);
	points.erase(std::ranges::unique(points).begin(), points.end());
	atl::li_chao_tree<int64> compressed(points);
	atl::dynamic_li_chao_tree<int64> dynamic(-bound, bound + 1);

	struct piece {
		int64 a, b, l, r;
	};

	std::vector<piece> pieces;
	for (int op = 0; op < 3000; ++op) {
		if (op % 3 == 2) {
			int64 const x = points[rng() % points.size()];
			std::optional<int128> best;
			for (auto const &p : pieces)
				if (p.l <= x && x < p.r) best = std::min(best.value_or(int128(p.a) * x + p.b), int128(p.a) * x + p.b);
			auto const got = compressed.query(x), got_dynamic = dynamic.query(x);
			REQUIRE(got.has_value() == best.has_value());
			REQUIRE(got == got_dynamic);
			if (best) REQUIRE(int128(*got) == *best);
			continue;
		}
		int64 const a = int64(rng() % std::uint64_t(2 * bound + 1)) - bound;
		int64 const b = int64(rng() % 2'000'000'000'000'000'001ULL) - 1'000'000'000'000'000'000LL;
		int64 l = int64(rng() % std::uint64_t(2 * bound + 1)) - bound, r = int64(rng() % std::uint64_t(2 * bound + 1)) - bound;
		if (l > r) std::swap(l, r);
		if (op % 2 == 0) l = -bound, r = bound + 1;
		compressed.insert(a, b, l, r);
		dynamic.insert(a, b, l, r);
		pieces.push_back({a, b, l, r});
	}
}

namespace {
	/** @brief `check.operator()<Tree>()` for every (endpoint carrying, coordinate search) combination of the tuned contender over `Function`. */
	template<typename Compare, typename Function, typename Check>
	void for_each_tuned_tree(Check const &check) {
		namespace experimental = atl::experimental;
		using atl::experimental::coordinate_search;
		auto const one = [&]<bool Carries, coordinate_search Search> {
			check.template operator()<experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<int64, Compare, Function, Carries, Search>>();
		};
		one.template operator()<false, coordinate_search::standard>();
		one.template operator()<false, coordinate_search::branchless>();
		one.template operator()<false, coordinate_search::eytzinger>();
		one.template operator()<true, coordinate_search::standard>();
		one.template operator()<true, coordinate_search::branchless>();
		one.template operator()<true, coordinate_search::eytzinger>();
	}
}

TEST_CASE("li_chao_tree_by_endpoint_carry_and_coordinate_search matches the brute force in every configuration", "[li_chao_tree]") {
	std::mt19937_64 rng(59);
	auto const draw_parabola = [&rng] { return parabola{int64(rng() % 41) - 20, int64(rng() % 401) - 200}; };
	for (int iter = 0; iter < 300; ++iter) {
		// sizes 1 to 70 (powers of two and their neighbours), dense and sparse domains
		int64 const n = 1 + int64(iter % 70), lo = int64(rng() % 41) - 20;
		std::vector<int64> range(static_cast<std::size_t>(n));
		for (int64 i = 0; i < n; ++i) range[std::size_t(i)] = lo + i;
		std::vector<int64> sparse;
		for (int64 i = 0; i < n; ++i) sparse.push_back(lo + 3 * i + int64(rng() % 3));
		int64 const sparse_hi = sparse.back() + 1;
		auto const run = [&]<std::strict_weak_order<int64, int64> Compare>(Compare const &compare) {
			for_each_tuned_tree<Compare, atl::linear_function<int64>>([&]<envelope_tree Tree>() {
				Tree over_range(lo, lo + n);
				stress(over_range, range, rng, 100, iter % 2 == 0, compare);
				Tree over_points(sparse);
				stress(over_points, sparse, rng, 100, iter % 2 == 1, compare);
				Tree suffixes(sparse);
				stress_suffixes<atl::linear_function<int64>>(
					suffixes, sparse, sparse_hi, rng,
					[&rng] { return atl::linear_function<int64>{int64(rng() % 21) - 10, int64(rng() % 201) - 100}; }, compare);
			});
			for_each_tuned_tree<Compare, parabola>([&]<envelope_tree Tree>() {
				Tree over_points(sparse);
				stress_family<Tree, parabola>(over_points, sparse, rng, draw_parabola, compare);
			});
		};
		if (iter % 2 == 0)
			run(std::less<>{});
		else
			run(std::greater<>{});
	}
	// the empty and one-point coordinate lists
	for_each_tuned_tree<std::less<>, atl::linear_function<int64>>([]<envelope_tree Tree>() {
		Tree empty{std::vector<int64>{}};
		empty.insert(1, 2);
		CHECK(empty.size() == 0);
		Tree one{std::vector<int64>{7}};
		one.insert(2, 1);
		one.insert(-1, 30);
		CHECK(one.query(7) == 15);
	});
}

TEST_CASE("li chao trees: edge cases", "[li_chao_tree]") {
	atl::li_chao_tree<int64> empty_range(5, 5);
	CHECK(empty_range.empty());
	empty_range.insert(1, 1);
	empty_range.insert(1, 1, 0, 10);
	atl::li_chao_tree<int64> placeholder;
	CHECK(placeholder.empty());
	atl::dynamic_li_chao_tree<int64> empty_dynamic(5, 5);
	CHECK(empty_dynamic.empty());
	empty_dynamic.insert(1, 1);

	atl::li_chao_tree<int64> one(7, 8);
	CHECK(!one.query(7));
	one.insert(2, 1, 8, 20); // misses the only point
	CHECK(!one.query(7));
	one.insert(2, 1);
	CHECK(one.query(7) == 15);

	// segment ends far outside the range (x - lo would overflow)
	atl::li_chao_tree<int64> narrow(-5, 5);
	narrow.insert(1, 0, std::numeric_limits<int64>::min(), std::numeric_limits<int64>::max());
	CHECK(narrow.query(4) == 4);
	CHECK(narrow.query(-5) == -5);
	atl::dynamic_li_chao_tree<int64> narrow_dynamic(-5, 5);
	narrow_dynamic.insert(1, 0, std::numeric_limits<int64>::min(), std::numeric_limits<int64>::max());
	CHECK(narrow_dynamic.query(4) == 4);

	// a full line creates one node at most (slopes up to 3 fit int64 over +-2^60)
	atl::dynamic_li_chao_tree<int64> dynamic(std::numeric_limits<int64>::min() / 8, std::numeric_limits<int64>::max() / 8);
	for (int64 i = 0; i < 100; ++i) dynamic.insert(i % 7 - 3, i * i % 13);
	CHECK(dynamic.node_count() <= 100);
	CHECK(dynamic.query(0) == 0);

	// coordinates in double, a segment whose ends fall between them
	atl::li_chao_tree<double> real(std::vector<double>{-1.5, 0.0, 0.5, 2.0});
	real.insert(1.0, 0.0, -1.0, 1.0); // covers 0.0 and 0.5
	CHECK(!real.query(-1.5));
	CHECK(real.query(0.5) == 0.5);
	CHECK(!real.query(2.0));
	real.insert(-1.0, 1.0);
	CHECK(real.query(0.5) == 0.5);
	CHECK(real.query(2.0) == -1.0);
}

namespace {
	/**
	 * @brief Random segments and lines on `Tree` (`dynamic_li_chao_tree<Coordinate>`
	 *        by default) over `[lo, hi)`, queried at and beside the segment ends against
	 *        an int128 brute force (for the clipped leaves' bound stored in
	 *        the child fields, `int32`, or in a field of its own, `int128`).
	 */
	template<atl::integral Coordinate, typename Tree = atl::dynamic_li_chao_tree<Coordinate>>
		requires std::constructible_from<Tree, Coordinate, Coordinate> && requires(Tree &tree, Coordinate x) {
			tree.insert(x, x, x, x);
			{ tree.query(x) } -> std::same_as<std::optional<Coordinate>>;
		}
	void stress_coordinate_type(Coordinate lo, Coordinate hi, std::mt19937_64 &rng) {
		Tree tree(lo, hi);

		struct piece {
			Coordinate a, b, l, r;
		};

		std::vector<piece> pieces;
		std::vector<Coordinate> probes{lo, Coordinate(hi - 1)};
		auto const point = [&] {
			atl::uint128 const width = atl::uint128(int128(hi) - int128(lo));
			return Coordinate(int128(lo) + int128((atl::uint128(rng()) << 64 | rng()) % width));
		};
		auto const brute = [&](Coordinate x) {
			std::optional<int128> best;
			for (auto const &p : pieces)
				if (p.l <= x && x < p.r) best = std::min(best.value_or(int128(p.a) * x + p.b), int128(p.a) * x + p.b);
			return best;
		};
		for (int op = 0; op < 400; ++op) {
			Coordinate const a = Coordinate(int64(rng() % 5) - 2), b = Coordinate(int64(rng() % 2001) - 1000);
			Coordinate l = point(), r = point();
			if (r < l) std::swap(l, r);
			if (op % 9 == 0) l = lo, r = hi;
			tree.insert(a, b, l, r);
			pieces.push_back({a, b, l, r});
			for (Coordinate const x : {Coordinate(l - 1), l, Coordinate(r - 1), r})
				if (lo <= x && x < hi) probes.push_back(x);
			if (op % 4 == 0) {
				Coordinate const x = probes[rng() % probes.size()];
				auto const got = tree.query(x);
				auto const want = brute(x);
				REQUIRE(got.has_value() == want.has_value());
				if (got) REQUIRE(int128(*got) == *want);
			}
		}
		for (Coordinate const x : probes) {
			auto const got = tree.query(x);
			auto const want = brute(x);
			REQUIRE(got.has_value() == want.has_value());
			if (got) REQUIRE(int128(*got) == *want);
		}
	}
}

TEST_CASE("dynamic_li_chao_tree: clipped leaves", "[li_chao_tree]") {
	// one segment strictly inside an empty 2e9 domain: the root (containing
	// both ends) and one clipped leaf per side, not two 31-node paths
	atl::dynamic_li_chao_tree<int64> tree(-1'000'000'000, 1'000'000'001);
	tree.insert(1, 0, -123'456'789, 987'654'321);
	CHECK(tree.node_count() == 3);
	CHECK(tree.query(-123'456'789) == -123'456'789);
	CHECK(tree.query(987'654'320) == 987'654'320);
	CHECK(!tree.query(987'654'321));
	CHECK(!tree.query(-123'456'790));
	// a line through the clipped leaves pushes them down one level
	tree.insert(0, 5);
	CHECK(tree.query(0) == 0);
	CHECK(tree.query(987'654'321) == 5);
	CHECK(tree.query(-1'000'000'000) == 5);

	std::mt19937_64 rng(59);
	for (int iter = 0; iter < 20; ++iter) {
		stress_coordinate_type<atl::int32>(-1'000'000'000, 1'000'000'001, rng);
		stress_coordinate_type<int128>(-(int128(1) << 90), int128(1) << 90, rng);
		stress_coordinate_type<int128>(int128(-50), int128(50), rng);
	}
}

namespace {
	/**
	 * @brief Random segments and lines (slopes `0..max_slope`, intercepts
	 *        `0..1000`) on `Tree` (`dynamic_li_chao_tree<Coordinate>` by default) over `[lo, hi)`
	 *        for an unsigned `Coordinate`, queried at and beside the segment
	 *        ends against a uint128 brute force. `max_slope` is 0 where
	 *        `slope * x` would not fit the coordinate type.
	 */
	template<atl::integral Coordinate, typename Tree = atl::dynamic_li_chao_tree<Coordinate>>
		requires std::constructible_from<Tree, Coordinate, Coordinate> && requires(Tree &tree, Coordinate x) {
			tree.insert(x, x, x, x);
			{ tree.query(x) } -> std::same_as<std::optional<Coordinate>>;
		}
	void stress_unsigned(Coordinate lo, Coordinate hi, Coordinate max_slope, std::mt19937_64 &rng) {
		using wide = atl::uint128;
		Tree tree(lo, hi);

		struct piece {
			Coordinate a, b, l, r;
		};

		std::vector<piece> pieces;
		std::vector<Coordinate> probes{lo, Coordinate(hi - 1)};
		auto const point = [&] { return Coordinate(wide(lo) + (wide(rng()) << 64 | rng()) % (wide(hi) - lo)); };
		auto const brute = [&](Coordinate x) {
			std::optional<wide> best;
			for (auto const &p : pieces)
				if (p.l <= x && x < p.r) best = std::min(best.value_or(wide(p.a) * x + p.b), wide(p.a) * x + p.b);
			return best;
		};
		auto const check = [&](Coordinate x) {
			auto const got = tree.query(x);
			auto const want = brute(x);
			REQUIRE(got.has_value() == want.has_value());
			if (got) REQUIRE(wide(*got) == *want);
		};
		for (int op = 0; op < 300; ++op) {
			Coordinate const a = max_slope == 0 ? Coordinate(0) : Coordinate(rng() % (wide(max_slope) + 1));
			Coordinate const b = Coordinate(rng() % 1001);
			Coordinate l = point(), r = point();
			if (r < l) std::swap(l, r);
			if (op % 9 == 0) l = lo, r = hi;
			tree.insert(a, b, l, r);
			pieces.push_back({a, b, l, r});
			for (Coordinate const x : {Coordinate(l - 1), l, Coordinate(r - 1), r})
				if (lo <= x && x < hi) probes.push_back(x);
			if (op % 4 == 0) check(probes[rng() % probes.size()]);
		}
		for (Coordinate const x : probes) check(x);
	}
}

TEST_CASE("dynamic_li_chao_tree: unsigned coordinates", "[li_chao_tree]") {
	using atl::uint32;
	using atl::uint64;
	std::mt19937_64 rng(61);
	for (int iter = 0; iter < 20; ++iter) {
		// slopes 0..2 over a small domain, constants over the top of the type
		stress_unsigned<uint32>(0, 1000, 2, rng);
		stress_unsigned<uint32>(std::numeric_limits<uint32>::max() - 1'000'000, std::numeric_limits<uint32>::max(), 0, rng);
		stress_unsigned<uint32>(0, std::numeric_limits<uint32>::max(), 0, rng);
		stress_unsigned<uint64>(0, 1000, 2, rng);
		stress_unsigned<uint64>(std::numeric_limits<uint64>::max() - (uint64(1) << 40), std::numeric_limits<uint64>::max(), 0, rng);
		stress_unsigned<uint64>(0, std::numeric_limits<uint64>::max(), 0, rng);
		stress_unsigned<uint64>(uint64(1) << 63, (uint64(1) << 63) + 100, 0, rng);
	}

	// a domain ending at the type's maximum (hi itself is excluded)
	atl::dynamic_li_chao_tree<uint64> top(std::numeric_limits<uint64>::max() - 10, std::numeric_limits<uint64>::max());
	top.insert(0, 7);
	top.insert(0, 3, std::numeric_limits<uint64>::max() - 4, std::numeric_limits<uint64>::max());
	CHECK(top.query(std::numeric_limits<uint64>::max() - 10) == 7);
	CHECK(top.query(std::numeric_limits<uint64>::max() - 4) == 3);
	CHECK(top.query(std::numeric_limits<uint64>::max() - 1) == 3);
}

TEST_CASE("li chao trees reject what they cannot hold", "[li_chao_tree]") {
	CHECK(atl_test::aborts([] { atl::li_chao_tree<int64> tree(5, 4); }));
	CHECK(atl_test::aborts([] { atl::li_chao_tree<int64> tree(std::vector<int64>{1, 3, 3}); }));
	CHECK(atl_test::aborts([] { atl::dynamic_li_chao_tree<int64> tree(5, 4); }));
	CHECK(atl_test::aborts([] {
		atl::li_chao_tree<int64> tree(0, 10);
		(void)tree.query(10);
	}));
	CHECK(atl_test::aborts([] {
		atl::li_chao_tree<int64> tree(std::vector<int64>{1, 3, 5});
		(void)tree.query(4);
	}));
	CHECK(atl_test::aborts([] {
		atl::dynamic_li_chao_tree<int64> tree(0, 10);
		(void)tree.query(-1);
	}));
	// slope 2^40 times x = 2^30 overflows int64
	CHECK(atl_test::aborts([] {
		atl::li_chao_tree<int64> tree(std::vector<int64>{0, int64(1) << 30});
		tree.insert(int64(1) << 40, 0);
	}));
	CHECK(atl_test::aborts([] {
		atl::dynamic_li_chao_tree<int64> tree(0, int64(1) << 40);
		tree.insert(int64(1) << 30, 0, 0, int64(1) << 40);
	}));
}

TEST_CASE("dynamic_li_chao_tree: whole-first queries and clip overwrites", "[li_chao_tree]") {
	using atl::uint32;
	using atl::uint64;
	// the int64 brute-force cases above run every contender through
	// for_each_dynamic_tree; these add the bound packed in 32 bits, in a field
	// of its own (int128) and unsigned coordinates
	std::mt19937_64 rng(67);
	for (int iter = 0; iter < 8; ++iter) {
		for_each_tuned_tree<atl::int32>([&]<envelope_tree Tree> {
			stress_coordinate_type<atl::int32, Tree>(-1'000'000'000, 1'000'000'001, rng);
		});
		for_each_tuned_tree<int128>([&]<envelope_tree Tree> {
			stress_coordinate_type<int128, Tree>(-(int128(1) << 90), int128(1) << 90, rng);
			stress_coordinate_type<int128, Tree>(int128(-50), int128(50), rng);
		});
		for_each_tuned_tree<uint32>([&]<envelope_tree Tree> {
			stress_unsigned<uint32, Tree>(0, 1000, 2, rng);
			stress_unsigned<uint32, Tree>(0, std::numeric_limits<uint32>::max(), 0, rng);
		});
		for_each_tuned_tree<uint64>([&]<envelope_tree Tree> {
			stress_unsigned<uint64, Tree>(0, 1000, 2, rng);
			stress_unsigned<uint64, Tree>(0, std::numeric_limits<uint64>::max(), 0, rng);
			stress_unsigned<uint64, Tree>(uint64(1) << 63, (uint64(1) << 63) + 100, 0, rng);
		});
	}

	using function = atl::linear_function<int64>;
	int64 const lo = -1'000'000'000, hi = 1'000'000'001, l = -123'456'789, r = 987'654'321;
	// The optional-exit copy isolates the pruning condition from layout differences.
	for (int iter = 0; iter < 20; ++iter) {
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<int64> copy(lo, hi);
		for (int op = 0; op < 200; ++op) {
			int64 const a = int64(rng() % 21) - 10, b = int64(rng() % 2001) - 1000;
			int64 x = lo + int64(rng() % std::uint64_t(hi - lo)), y = lo + int64(rng() % std::uint64_t(hi - lo));
			if (y < x) std::swap(x, y);
			if (op % 5 == 0) {
				library.insert(a, b);
				copy.insert(a, b);
			} else {
				library.insert(a, b, x, y);
				copy.insert(a, b, x, y);
			}
			REQUIRE(library.node_count() <= copy.node_count());
			REQUIRE(library.query(x) == copy.query(x));
		}
	}
	// a segment beaten by the root's line makes no node
	{
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<int64> without_exit(lo, hi);
		library.insert(0, 0);
		without_exit.insert(0, 0);
		library.insert(0, 5, l, r);
		without_exit.insert(0, 5, l, r);
		CHECK(library.node_count() == 1);
		CHECK(without_exit.node_count() == 3);
		CHECK(without_exit.query(l) == 0);
	}
	// a segment over a clipped leaf's whole clip range, better on all
	// of it, empties the leaf instead of pushing it down
	{
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<int64, std::less<>, function, false, true> overwriting(lo, hi);
		library.insert(1, 0, l, r);
		overwriting.insert(1, 0, l, r);
		library.insert(0, -2'000'000'000, l, r);
		overwriting.insert(0, -2'000'000'000, l, r);
		CHECK(overwriting.node_count() < library.node_count());
		for (int64 const x : {l - 1, l, int64(0), r - 1, r}) CHECK(overwriting.query(x) == library.query(x));
	}
	// Clip overwrite during whole-line descent: y = x, sent left from the root, beats the
	// clipped leaf's segment on its whole clip range and overwrites it in place
	{
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<int64, std::less<>, function, false, true> overwriting(lo, hi);
		for (auto const &[a, b, from, to] : {std::array{int64(0), int64(5), l, r}, std::array{int64(0), int64(0), lo, hi},
											 std::array{int64(1), int64(0), lo, hi}}) {
			library.insert(a, b, from, to);
			overwriting.insert(a, b, from, to);
		}
		CHECK(overwriting.node_count() < library.node_count());
		for (int64 const x : {lo, l - 1, l, int64(-1), int64(0), int64(1), r - 1, r, hi - 1})
			CHECK(overwriting.query(x) == library.query(x));
	}
}

TEST_CASE("dynamic_li_chao_tree: library dominated exit and improving segments", "[li_chao_tree]") {
	std::mt19937_64 rng(731);
	for (auto const &[lo, hi] : {std::pair<int64, int64>{0, 1'000'000}, {-1'000'000'000, 1'000'000'001}}) {
		stress_coordinate_type<int64>(lo, hi, rng);
		atl::dynamic_li_chao_tree<int64> tree(lo, hi);
		tree.insert(0, 0);
		for (auto const &[l, r] : {std::pair{lo, lo}, {lo, lo + 1}, {hi - 1, hi}, {lo + 1, hi - 1}}) {
			tree.insert(0, 0, l, r);
			CHECK(tree.node_count() == 1);
		}
		brute_envelope<> brute;
		brute.insert(0, 0);
		for (int64 i = 1; i <= 100; ++i) {
			int64 const l = lo + i, r = hi - i;
			tree.insert(0, -i, l, r);
			brute.insert(0, -i, l, r);
			for (int64 x : {lo, l - 1, l, r - 1, r, hi - 1}) CHECK(tree.query(x) == brute.query(x));
		}
		tree.insert(0, -200);
		tree.insert(0, -150, lo + 1, hi - 1);
		CHECK(tree.query(lo + 1) == -200);
	}
	for (int64 hi : std::array<int64, 6>{0, 1, 2, 999'999, 1'000'001, 1'000'000'000'000LL}) {
		atl::dynamic_li_chao_tree<int64> tree(0, hi);
		tree.insert(0, 0);
		tree.insert(0, 5, 0, hi);
		if (hi != 0) CHECK(tree.query(hi - 1) == 0);
	}
}

TEST_CASE("Li Chao overflow checks include products and software wide values", "[li_chao_tree]") {
	using Value = atl::detail::software_int128;
	CHECK(atl_test::aborts([] {
		atl::li_chao_tree<Value> tree(std::vector<Value>{(std::numeric_limits<Value>::lowest)()});
		tree.insert(-1, -1);
	}));
	CHECK(atl_test::aborts([] {
		atl::dynamic_li_chao_tree<Value> tree(0, 2);
		tree.insert(1, (std::numeric_limits<Value>::max)());
	}));
	CHECK(atl_test::aborts([] {
		atl::dynamic_li_chao_tree<atl::uint64> tree(0, 3);
		tree.insert((std::numeric_limits<atl::uint64>::max)(), 0);
	}));
	atl::dynamic_li_chao_tree<Value> tree((std::numeric_limits<Value>::lowest)(), (std::numeric_limits<Value>::max)());
	tree.insert(1, 0);
	tree.insert(0, 5, -2, 3);
	CHECK(tree.query((std::numeric_limits<Value>::lowest)()) == (std::numeric_limits<Value>::lowest)());
	CHECK(tree.query(0) == 0);
}
