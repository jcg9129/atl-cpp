#pragma once

/**
 * @file
 * @brief @ref atl::experimental::balance_leaf_paths, a worked example of
 *        @ref atl::convex_piecewise_linear_forest (experimental, not part of
 *        the library): change the edge costs of a rooted tree at the least
 *        weighted total change so that every root-to-leaf path has the same
 *        total, optionally with a lower bound on the new costs, and
 *        reconstruct the new costs.
 */

#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace atl::experimental {
	using atl::convex_piecewise_linear_forest;
	using atl::int128;
	using atl::int64;
	using atl::minimizer_interval;
	using atl::slope_clip_interval;
	using atl::uint32;

	/** @brief The result of @ref atl::experimental::balance_leaf_paths. */
	struct leaf_path_balance {
		/** @brief The least total change: the sum of `weight[v] * |new_cost[v] - cost[v]|`. */
		int64 total_cost = 0;
		/** @brief The common total of every root-to-leaf path under `new_cost` (0 for a single vertex). */
		int64 path_sum = 0;
		/** @brief The new cost of the edge from `parent[v]` to `v`, per vertex `v` (`new_cost[0] = 0`). */
		std::vector<int64> new_cost;
	};

	namespace detail {
		/** @brief The checked tree of a leaf-balancing run, shared with the experimental contenders. */
		struct leaf_path_plan {
			/** @brief Vertex `v`'s children are `children[child_start[v] .. child_start[v + 1])`. */
			std::vector<uint32> child_start;
			/** @brief Every vertex in breadth-first order from the root (parents before children). */
			std::vector<uint32> order;
			/** @brief The sum and the largest of the edge weights. */
			int64 weight_sum = 0, max_weight = 0;

			/** @brief Whether `v` has no children. */
			[[nodiscard]] constexpr bool is_leaf(uint32 v) const { return child_start[v] == child_start[v + 1]; }
		};

		/**
		 * @brief Check the inputs of @ref atl::experimental::balance_leaf_paths
		 *        (every asserted precondition but the pool's record limit) and
		 *        order the vertices breadth-first (no recursion).
		 * @complexity O(n)
		 */
		[[nodiscard]] constexpr leaf_path_plan plan_leaf_paths(std::span<uint32 const> parent, std::span<int64 const> cost,
															   std::span<int64 const> weight, std::optional<int64> min_cost) {
			std::size_t const n = parent.size();
			atl_assert(n >= 1 && cost.size() == n && weight.size() == n);
			atl_assert(n < std::numeric_limits<uint32>::max());
			leaf_path_plan plan;
			// children by counting sort on the parent, then a BFS order from the root
			plan.child_start.assign(n + 1, 0);
			std::vector<uint32> children(n == 0 ? 0 : n - 1);
			int64 const weight_cap =
				min_cost ? std::numeric_limits<int64>::max() / 4 / int64(n) : std::numeric_limits<int64>::max() / 2;
			for (std::size_t v = 1; v < n; ++v) {
				atl_assert(parent[v] < n);
				atl_assert(weight[v] >= 0);
				atl_assert(!min_cost || cost[v] >= *min_cost);
				atl_assert(weight[v] <= weight_cap - plan.weight_sum);
				plan.weight_sum += weight[v];
				plan.max_weight = std::max(plan.max_weight, weight[v]);
				++plan.child_start[parent[v] + 1];
			}
			for (std::size_t v = 0; v < n; ++v) plan.child_start[v + 1] += plan.child_start[v];
			{
				std::vector<uint32> fill(plan.child_start.begin(), plan.child_start.end() - 1);
				for (std::size_t v = 1; v < n; ++v) children[fill[parent[v]]++] = uint32(v);
			}
			plan.order.reserve(n);
			plan.order.push_back(0);
			for (std::size_t head = 0; head < plan.order.size(); ++head)
				for (uint32 i = plan.child_start[plan.order[head]]; i < plan.child_start[plan.order[head] + 1]; ++i)
					plan.order.push_back(children[i]);
			atl_assert(plan.order.size() == n); // a vertex on a cycle is never reached from the root
			return plan;
		}

		/** @brief Whether an `int128` fits `int64`. */
		[[nodiscard]] constexpr bool fits_int64(int128 const &x) {
			return !(x < int128(std::numeric_limits<int64>::min())) && !(int128(std::numeric_limits<int64>::max()) < x);
		}

		/**
		 * @brief The top-down half of @ref atl::experimental::balance_leaf_paths,
		 *        shared with the experimental contenders: from the root's path
		 *        total, each child takes `clamp(x_parent - cost[v])` into its
		 *        clip range (capped by the bound), and the result is assembled.
		 *
		 * The subtraction, the clamp and the change `|new_cost - cost|` are
		 * formed in `int128`: an input such as `cost[v] = INT64_MIN` with
		 * weight 0 makes `x_parent - cost[v]` leave `int64` although the path
		 * totals, the new costs and the total cost all fit. Those three are
		 * narrowed back, each asserted to fit.
		 */
		[[nodiscard]] constexpr leaf_path_balance settle_leaf_paths(std::span<uint32 const> parent, std::span<int64 const> cost,
																	std::span<int64 const> weight, std::optional<int64> min_cost,
																	leaf_path_plan const &plan,
																	std::vector<slope_clip_interval<int64>> const &range,
																	int64 root_total) {
			std::size_t const n = parent.size();
			bool const is_zero_bound = min_cost && *min_cost == 0;
			leaf_path_balance result;
			result.new_cost.assign(n, 0);
			result.path_sum = root_total;
			std::vector<int64> path_value(n, 0);
			path_value[0] = root_total;
			int128 total = 0;
			for (std::size_t i = 1; i < n; ++i) {
				uint32 const v = plan.order[i];
				int128 const above = path_value[parent[v]];
				int128 below = above - cost[v];
				if (range[v].lower && below < int128(*range[v].lower)) below = *range[v].lower;
				if (range[v].upper && int128(*range[v].upper) < below) below = *range[v].upper;
				if (min_cost && int128(above - *min_cost) < below) below = above - *min_cost;
				if (is_zero_bound && below < int128(0)) below = 0;
				atl_assert(fits_int64(below)); // a path total (documented to fit)
				path_value[v] = int64(below);
				int128 const new_cost = above - below;
				atl_assert(fits_int64(new_cost));
				result.new_cost[v] = int64(new_cost);
				int128 const change = new_cost - cost[v];
				total += int128(weight[v]) * (change < int128(0) ? -change : change);
			}
			atl_assert(fits_int64(total));
			result.total_cost = int64(total);
			return result;
		}
	}

	/**
	 * @brief Choose new edge costs `X_v` minimizing the sum of `weight[v] *
	 *        |X_v - cost[v]|` such that every root-to-leaf path has the same
	 *        total, optionally with every `X_v >= min_cost`; returns the
	 *        minimum, the common path total and one optimal assignment.
	 *
	 * Ties: the root's path total is the smallest minimizer of its function
	 * (when the minimizers are unbounded below, which needs zero weights, the
	 * function's leftmost breakpoint, or 0 without one), and each child's
	 * total is the clamp of `x_parent - cost[v]` into its clip range, so an
	 * edge keeps its old cost whenever that is optimal given its parent's
	 * total (then capped by the bound, with `min_cost`).
	 *
	 * The tree is rooted at vertex 0 and given by `parent[v]` for `v >= 1`
	 * (`parent[0]` is ignored); `cost[v]` and `weight[v]` belong to the edge
	 * `(parent[v], v)`. Bottom-up, `f_v(x)` is the least cost of making every
	 * path from `v` down to a leaf total `x`: a leaf is the indicator of `x =
	 * 0`, an inner vertex the sum of its children's functions, and passing
	 * the edge above `v` is the infimal convolution with `weight[v] * |t -
	 * cost[v]|` (clip the slopes to `+-weight[v]`, translate by `cost[v]`,
	 * on a @ref atl::convex_piecewise_linear_forest), recording the clip range
	 * `[lo_v, hi_v]`. Top-down, the root takes a minimizer of its function
	 * and each child `x_v = clamp(x_parent - cost[v], lo_v, hi_v)`, so `X_v =
	 * x_parent - x_v`. With `min_cost`, the edge kernel is restricted to `t >=
	 * min_cost` (a clip to slopes `<= weight[v]`, a segment of slope
	 * `-weight[v]` and length `cost[v] - min_cost`, a translation by
	 * `min_cost`), and `x_v` is further capped at `x_parent - min_cost`. A
	 * leaf's domain wall (its own edge `>= min_cost`) is a steep left slope:
	 * `2 * (sum of weights) + 1` in general, only `max weight - weight[v]`
	 * at `min_cost == 0` (no wall record for unit weights; `x_v >= 0` settles
	 * the tie).
	 *
	 * @param parent `parent[v] < n` for `v >= 1`, forming a tree rooted at 0.
	 * @param cost The edge costs, `cost[0]` ignored.
	 * @param weight The edge weights, `weight[v] >= 0`, `weight[0]` ignored.
	 * @param min_cost A lower bound on every new cost (`0` enforces nonnegative costs), or none.
	 * @pre the three spans have one size `n >= 1` and describe a tree
	 *      (asserted: a cycle or an unreachable vertex is caught); weights are
	 *      nonnegative and, with `min_cost`, `cost[v] >= min_cost` (asserted);
	 *      the sum of weights is at most `max(int64) / 2`, and with `min_cost`
	 *      `4 * n * (sum of weights)` fits int64 (asserted: the domain walls'
	 *      slopes); every path total, every translated breakpoint and the
	 *      total cost fit int64 (documented; the reconstruction forms its
	 *      differences and changes in `int128`, so nothing else has to, and
	 *      asserts that the three results fit). The forest's pool holds fewer
	 *      than `2^32 - 1` breakpoint records over the run (asserted by the
	 *      pool): at most two per leaf plus, with `min_cost`, one per moved
	 *      piece, `2n + M` in all (`M` below). A bounded path can exceed it:
	 *      `n = 1e5`, unit costs, `min_cost = 0`, `weight[1] = 4n` and
	 *      `weight[v] = n - v` below moves about `5e9` pieces.
	 * @complexity Without `min_cost`, O(n log n) time (amortized pairing-heap
	 *      pops) and O(n) memory. With `min_cost`, each edge also moves `m_v`
	 *      breakpoints, at most `2 * weight[v]` of them and at most the
	 *      breakpoints alive, and each move is a new record: O((n + M) log n)
	 *      time and O(n + M) memory for `M` the sum of the `m_v`. That is
	 *      O(n log n) for bounded weights; with weights up to 1e6 a path can
	 *      reach `M` of order `n^2` (the path above). No recursion: the orders
	 *      are built by a breadth-first scan, so a path of 10^6 vertices needs
	 *      no stack depth.
	 */
	[[nodiscard]] constexpr leaf_path_balance balance_leaf_paths(std::span<uint32 const> parent, std::span<int64 const> cost,
																 std::span<int64 const> weight,
																 std::optional<int64> min_cost = std::nullopt) {
		std::size_t const n = parent.size();
		detail::leaf_path_plan const plan = detail::plan_leaf_paths(parent, cost, weight, min_cost);
		std::vector<uint32> const &order = plan.order;

		// bottom-up: f_v after its edge, merged into the parent; function v = vertex v
		convex_piecewise_linear_forest<int64, int64> forest(n);
		// n records suffice for a free-cost run (one per leaf); a bounded run grows past it on some
		// shapes (about 1.5n on the unit-weight caterpillar). Larger hints measured slower on repeated
		// calls, which return and fault the larger blocks again
		forest.reserve(n);
		std::vector<slope_clip_interval<int64>> range(n);
		// with a lower bound, a leaf's domain wall (its edge >= min_cost) is a left slope
		// no clip may cross. At min_cost == 0 every wall sits at 0 with no breakpoint left
		// of it, so a slope of max_weight suffices (a tie is settled by the x >= 0 cap
		// below; unit weights need no wall record at all); otherwise the walls differ by
		// depth, sibling slopes add up to weight_sum on top, and the wall must beat that
		bool const is_zero_bound = min_cost && *min_cost == 0;
		int64 const wall = is_zero_bound ? plan.max_weight : 2 * plan.weight_sum + 1;
		for (std::size_t i = n; i-- > 1;) {
			uint32 const v = order[i];
			int64 const c = cost[v], d = weight[v];
			auto const f = forest[forest.id(v)];
			if (plan.is_leaf(v)) {
				range[v] = {0, 0};
				if (min_cost) f.add_falling_hinge(*min_cost, wall - d);
				f.add_absolute(c, d);
			} else if (min_cost) {
				range[v].upper = f.clip_slopes(f.leftmost_slope(), d).upper;
				range[v].lower = f.insert_segment(-d, c - *min_cost);
				f.translate(*min_cost);
			} else {
				range[v] = f.clip_slopes(-d, d);
				f.translate(c);
			}
			forest.merge(forest.id(parent[v]), f.id());
		}

		// top-down: the root at its smallest minimizer, each child at its optimal
		// predecessor. The root's function is dropped, so only its lighter end is eaten:
		// clip_slopes(0, 0) would empty both heaps, and a bounded root is heavy on the
		// left, where the min-heap holds every stale entry of the run
		int64 root_total = 0;
		if (n > 1) {
			auto const root = forest[forest.id(0)];
			int64 const left = -root.leftmost_slope(), right = root.rightmost_slope();
			if (left > 0 && left <= right) {
				root_total = *root.take_suffix_minimum().lower;
			} else {
				// right of the minimum eaten (or nothing to eat): the smallest minimizer is
				// the rightmost breakpoint left, or the leftmost when f is nondecreasing
				if (left > 0 && right > 0) (void)root.take_prefix_minimum();
				minimizer_interval<int64> const minimizers = root.minimizers();
				root_total = minimizers.lower ? *minimizers.lower : minimizers.upper ? *minimizers.upper
																					 : 0;
			}
		}
		return detail::settle_leaf_paths(parent, cost, weight, min_cost, plan, range, root_total);
	}
}
