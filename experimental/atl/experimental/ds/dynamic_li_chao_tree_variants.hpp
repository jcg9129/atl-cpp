#pragma once

/**
 * @file
 * @brief Race losers for @ref atl::dynamic_li_chao_tree's segment insertion,
 *        and the query-order and clip-overwrite variants (@ref atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite;
 *        2026-11-01 verdict below).
 *
 * The library tree stops a segment's boundary path at the first node that
 * does not exist yet and leaves the segment there as a clipped leaf (a
 * childless node holding it on `[bound, last]` or `[first, bound]`, the bound
 * packed into the unused child fields for a coordinate of 64 bits or less),
 * pushed one level down when a later insertion passes through.
 *
 * - @ref atl::experimental::dynamic_li_chao_tree_by_materialized_paths, materialized paths, which create every node on both boundary paths down to single
 *   integers, about two per level of the tree for each segment.
 * - @ref atl::experimental::dynamic_li_chao_tree_by_two_bound_clips, a clipped leaf stores both ends of its
 *   clip, so a missing node containing both ends of a segment becomes a leaf
 *   too; a 48-byte node instead of 32 for `int64` lines.
 *
 * Race verdict (benchmarks/ds/li_chao_tree.cpp, records of
 * 2026-10-23, Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, ms, plain -O2 / -O2 -march=native; 2e5
 * inserts + 2e5 mixed operations, int64 lines, every contender with the
 * endpoint pruning):
 *
 * | shape | by one bound clips, without dominated exit | materialized | clipped, both ends |
 * |-------------------------------|-----------|--------------|--------------------|
 * | segments, `\|x\| <= 1e9` | 622 / 647 | 1194 / 1214 | 731 / 755 |
 * | suffix segments, `\|x\| <= 1e9` | 272 / 284 | 601 / 607 | 309 / 324 |
 * | segments, `[0, 1e6)` | 428 / 454 | 435 / 450 | 487 / 515 |
 * | suffix segments, `[0, 1e6)` | 220 / 235 | 221 / 223 | 256 / 268 |
 * | lines, `\|x\| <= 1e9` | 7.4 / 7.5 | 6.6 / 7.0 | 7.3 / 8.2 |
 *
 * Materialized paths lost 1.9-2.2x on the wide domain and tie on the dense
 * one; clipped leaves with both ends lost to one end and a side on every
 * segment shape (wider nodes). Also raced on the same record and removed:
 * the bound in a field of its own (677 / 704 on wide segments: packing buys
 * 8%), descents from a segment's canonical nodes interleaved in lockstep
 * (662 / 684) and the same with a prefetch of each next node (664 / 694):
 * once the endpoint pruning ends most descents within a few levels, there
 * is no miss latency left for them to overlap. The lines row moves between
 * 6.7 and 7.9 ms across the half-clipped variants, whose line path is the
 * same code.
 */

#include <atl/ds/linear_function.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief @ref atl::dynamic_li_chao_tree with a clipped leaf's clip stored
	 *        as both ends, `[clip_first, clip_last]`, instead of one end and
	 *        a side: a missing node that contains both ends of a segment
	 *        becomes a clipped leaf too, at the cost of a wider node; same
	 *        interface and semantics.
	 *
	 * @tparam Coordinate A signed or unsigned integer type for the domain.
	 * @tparam Compare `comp(a, b)` reads "a is better than b" on the values.
	 * @tparam Function The family's member type (@ref atl::envelope_function),
	 *         `linear_function<Coordinate>` by default.
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
		requires integral<Coordinate> && envelope_function<Function, Coordinate, Compare>
	class dynamic_li_chao_tree_by_two_bound_clips {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr dynamic_li_chao_tree_by_two_bound_clips() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions and no nodes yet.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr dynamic_li_chao_tree_by_two_bound_clips(Coordinate lo, Coordinate hi, Compare comp = {})
			: comp_(std::move(comp)), first_(lo), last_(hi), is_empty_(!(lo < hi)) {
			atl_assert(!(hi < lo));
			if (!is_empty_) --last_;
		}

		/** @brief Whether the domain has no points. */
		[[nodiscard]] constexpr bool empty() const { return is_empty_; }

		/** @brief The number of materialized nodes. */
		[[nodiscard]] constexpr std::size_t node_count() const { return nodes_.empty() ? 0 : nodes_.size() - 1; }

		/** @brief Reserve room for @p nodes materialized nodes. @pre `nodes < SIZE_MAX` (asserted). */
		constexpr void reserve(std::size_t nodes) {
			atl_assert(nodes < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(nodes + 1);
		}

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Function f) {
			if (is_empty_) return;
			atl_assert(atl::detail::evaluates_on(f, first_, last_));
			if (root_ == 0) root_ = make_node();
			descend(root_, first_, last_, std::move(f));
		}

		/**
		 * @brief Add `f` on the integers `x` with `l <= x < r` of the domain
		 *        only (a segment); an empty range adds nothing.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Function const &f, Coordinate l, Coordinate r) {
			if (is_empty_ || !(l < r)) return;
			--r;
			if (l < first_) l = first_;
			if (last_ < r) r = last_;
			if (r < l) return;
			atl_assert(atl::detail::evaluates_on(f, l, r));
			root_ = insert_segment(root_, first_, last_, l, r, f);
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its value at every domain point fits (asserted at the two ends)
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate l, Coordinate r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, std::move(l), std::move(r));
		}

		/**
		 * @brief The best value at `x` over the lines and segments covering
		 *        it, or none when nothing covers it.
		 * @pre `lo <= x < hi` (asserted)
		 * @complexity O(log W)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			atl_assert(!is_empty_ && !(x < first_) && !(last_ < x));
			std::optional<value_type> best;
			Coordinate first = first_, last = last_;
			for (uint32 node = root_; node != 0;) {
				node_type const &here = nodes_[node];
				if (here.kind == held_kind::whole
					|| (here.kind == held_kind::clipped && !(x < here.clip_first) && !(here.clip_last < x))) {
					value_type value = here.held(x);
					if (!best || comp_(value, *best)) best = std::move(value);
				}
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				if (x <= middle) {
					node = here.left;
					last = middle;
				} else {
					node = here.right;
					first = middle + 1;
				}
			}
			return best;
		}

	private:
		/** @brief What a node holds: nothing, a function on its whole interval, or a clipped one (then no children). */
		enum class held_kind : uint8 { none,
									   whole,
									   clipped };

		/** @brief A materialized node: a function (see `kind`), its clip when clipped, and child ids (0 = none). */
		struct node_type {
			Function held{};
			Coordinate clip_first{}, clip_last{};
			uint32 left = 0, right = 0;
			held_kind kind = held_kind::none;
		};

		/** @brief Append an empty node; fewer than `2^32 - 1` nodes asserted (id 0 is the null sentinel). */
		constexpr uint32 make_node() {
			atl_assert(nodes_.size() < std::size_t(std::numeric_limits<uint32>::max()));
			nodes_.emplace_back();
			return uint32(nodes_.size() - 1);
		}

		/** @brief A new childless node over `[first, last]` holding `f` on `[l, r]`, a sub-interval. */
		constexpr uint32 make_leaf(Coordinate const &first, Coordinate const &last, Coordinate const &l, Coordinate const &r,
								   Function f) {
			uint32 const id = make_node();
			node_type &leaf = nodes_[id];
			leaf.held = std::move(f);
			if (l == first && r == last) {
				leaf.kind = held_kind::whole;
			} else {
				leaf.kind = held_kind::clipped;
				leaf.clip_first = l;
				leaf.clip_last = r;
			}
			return id;
		}

		/** @brief Push the clipped function of the leaf `node` over `[first, last]` into new children. */
		constexpr void push_clip(uint32 node, Coordinate const &first, Coordinate const &last) {
			node_type &here = nodes_[node];
			here.kind = held_kind::none;
			Function f = std::move(here.held);
			Coordinate const l = here.clip_first, r = here.clip_last;
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			if (l <= middle) {
				uint32 const child = make_leaf(first, middle, l, (std::min)(r, middle), f);
				nodes_[node].left = child;
			}
			if (middle < r) {
				uint32 const child = make_leaf(middle + 1, last, (std::max)(l, Coordinate(middle + 1)), r, std::move(f));
				nodes_[node].right = child;
			}
		}

		/** @brief Settle `added` into the subtree of the existing `node` over `[first, last]`. */
		constexpr void descend(uint32 node, Coordinate first, Coordinate last, Function added) {
			while (true) {
				if (nodes_[node].kind == held_kind::clipped) push_clip(node, first, last);
				node_type &here = nodes_[node];
				if (here.kind == held_kind::none) {
					here.held = std::move(added);
					here.kind = held_kind::whole;
					return;
				}
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				bool const wins_middle = comp_(added(middle), here.held(middle));
				bool const wins_first = comp_(added(first), here.held(first));
				// winning (or losing) at both ends: on all of the range, as in atl::dynamic_li_chao_tree
				if (wins_first == wins_middle && comp_(added(last), here.held(last)) == wins_middle) {
					if (wins_middle) here.held = std::move(added);
					return;
				}
				if (wins_middle) std::swap(here.held, added);
				if (first == last) return;
				bool const goes_left = wins_first != wins_middle;
				uint32 child = goes_left ? here.left : here.right;
				if (goes_left)
					last = middle;
				else
					first = middle + 1;
				if (child == 0) {
					child = make_node();
					(goes_left ? nodes_[node].left : nodes_[node].right) = child;
				}
				node = child;
			}
		}

		/** @brief Insert `added` on `[l, r]` into the subtree of `node` (0 = none yet) over `[first, last]`. */
		constexpr uint32 insert_segment(uint32 node, Coordinate first, Coordinate last, Coordinate const &l, Coordinate const &r,
										Function const &added) {
			if (node == 0) return make_leaf(first, last, (std::max)(l, first), (std::min)(r, last), added);
			if (!(first < l) && !(r < last)) {
				descend(node, first, last, added);
				return node;
			}
			if (nodes_[node].kind == held_kind::clipped) push_clip(node, first, last);
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			if (l <= middle) {
				uint32 const child = insert_segment(nodes_[node].left, first, middle, l, r, added);
				nodes_[node].left = child;
			}
			if (middle < r) {
				uint32 const child = insert_segment(nodes_[node].right, middle + 1, last, l, r, added);
				nodes_[node].right = child;
			}
			return node;
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<node_type> nodes_ = std::vector<node_type>(1);
		Coordinate first_{}, last_{};
		uint32 root_ = 0;
		bool is_empty_ = true;
	};

	/**
	 * @brief @ref atl::dynamic_li_chao_tree with a
	 *        segment materializes every node on both of its boundary paths,
	 *        down to single integers; same interface and semantics (and the
	 *        same endpoint pruning in its descents).
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
		requires integral<Coordinate> && envelope_function<Function, Coordinate, Compare>
	class dynamic_li_chao_tree_by_materialized_paths {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr dynamic_li_chao_tree_by_materialized_paths() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions and no nodes yet.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr dynamic_li_chao_tree_by_materialized_paths(Coordinate lo, Coordinate hi, Compare comp = {})
			: comp_(std::move(comp)), first_(lo), last_(hi), is_empty_(!(lo < hi)) {
			atl_assert(!(hi < lo));
			if (!is_empty_) --last_;
		}

		/** @brief Whether the domain has no points. */
		[[nodiscard]] constexpr bool empty() const { return is_empty_; }

		/** @brief The number of materialized nodes. */
		[[nodiscard]] constexpr std::size_t node_count() const { return nodes_.empty() ? 0 : nodes_.size() - 1; }

		/** @brief Reserve room for @p nodes materialized nodes. @pre `nodes < SIZE_MAX` (asserted). */
		constexpr void reserve(std::size_t nodes) {
			atl_assert(nodes < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(nodes + 1);
		}

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log W), at most one new node
		 */
		constexpr void insert(Function f) {
			if (is_empty_) return;
			atl_assert(atl::detail::evaluates_on(f, first_, last_));
			if (root_ == 0) root_ = make_node();
			descend(root_, first_, last_, std::move(f));
		}

		/**
		 * @brief Add `f` on the integers `x` with `l <= x < r` of the domain
		 *        only (a segment); an empty range adds nothing.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Function const &f, Coordinate l, Coordinate r) {
			if (is_empty_ || !(l < r)) return;
			--r;
			if (l < first_) l = first_;
			if (last_ < r) r = last_;
			if (r < l) return;
			atl_assert(atl::detail::evaluates_on(f, l, r));
			root_ = insert_segment(root_, first_, last_, l, r, f);
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its value at every domain point fits (asserted at the two ends)
		 * @complexity O(log W), at most one new node
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate l, Coordinate r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, std::move(l), std::move(r));
		}

		/**
		 * @brief The best value at `x` over the lines and segments covering
		 *        it, or none when nothing covers it.
		 * @pre `lo <= x < hi` (asserted)
		 * @complexity O(log W)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			atl_assert(!is_empty_ && !(x < first_) && !(last_ < x));
			std::optional<value_type> best;
			Coordinate first = first_, last = last_;
			for (uint32 node = root_; node != 0;) {
				node_type const &here = nodes_[node];
				if (here.is_used) {
					value_type value = here.held(x);
					if (!best || comp_(value, *best)) best = std::move(value);
				}
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				if (x <= middle) {
					node = here.left;
					last = middle;
				} else {
					node = here.right;
					first = middle + 1;
				}
			}
			return best;
		}

	private:
		/** @brief A materialized node: a function unless `is_used` is false, and child ids (0 = none). */
		struct node_type {
			Function held{};
			uint32 left = 0, right = 0;
			bool is_used = false;
		};

		/**
		 * @brief Append an empty node and return its id: the one place node
		 *        ids are narrowed, so fewer than `2^32 - 1` nodes is asserted
		 *        here (id 0 is the null sentinel).
		 */
		constexpr uint32 make_node() {
			atl_assert(nodes_.size() < std::size_t(std::numeric_limits<uint32>::max()));
			nodes_.emplace_back();
			return uint32(nodes_.size() - 1);
		}

		/**
		 * @brief Settle `added` into the subtree of the existing `node` over
		 *        `[first, last]`, creating at most one node: keep the better
		 *        function at each midpoint, send the other into the half where it
		 *        can still win.
		 */
		constexpr void descend(uint32 node, Coordinate first, Coordinate last, Function added) {
			while (true) {
				node_type &here = nodes_[node];
				if (!here.is_used) {
					here.held = std::move(added);
					here.is_used = true;
					return;
				}
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				bool const wins_middle = comp_(added(middle), here.held(middle));
				bool const wins_first = comp_(added(first), here.held(first));
				// crossing at most once, a function that wins (or loses) at both
				// ends of the node's range wins (or loses) on all of it: the
				// loser is dropped instead of walked down the tree
				if (wins_first == wins_middle && comp_(added(last), here.held(last)) == wins_middle) {
					if (wins_middle) here.held = std::move(added);
					return;
				}
				if (wins_middle) std::swap(here.held, added);
				if (first == last) return;
				bool const goes_left = wins_first != wins_middle;
				uint32 child = goes_left ? here.left : here.right;
				if (goes_left)
					last = middle;
				else
					first = middle + 1;
				if (child == 0) {
					child = make_node();
					(goes_left ? nodes_[node].left : nodes_[node].right) = child;
				}
				node = child;
			}
		}

		/**
		 * @brief Insert `added` on `[l, r]` into the subtree of `node` (0 = not
		 *        yet created) over `[first, last]`, which intersects `[l, r]`;
		 *        returns the subtree's node id.
		 */
		constexpr uint32 insert_segment(uint32 node, Coordinate first, Coordinate last, Coordinate const &l, Coordinate const &r,
										Function const &added) {
			if (node == 0) node = make_node();
			if (!(first < l) && !(r < last)) {
				descend(node, first, last, added);
				return node;
			}
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			if (l <= middle) {
				uint32 const child = insert_segment(nodes_[node].left, first, middle, l, r, added);
				nodes_[node].left = child;
			}
			if (middle < r) {
				uint32 const child = insert_segment(nodes_[node].right, middle + 1, last, l, r, added);
				nodes_[node].right = child;
			}
			return node;
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<node_type> nodes_ = std::vector<node_type>(1);
		Coordinate first_{}, last_{};
		uint32 root_ = 0;
		bool is_empty_ = true;
	};

	/**
	 * @brief Segment insertion without dominated pruning and the whole-first /
	 *        clip-overwrite losers, kept against the library tree as oracle.
	 *
	 * Record ds-li_chao_tree, 2026-11-01, Xeon Platinum 8280, plain -O2 /
	 * -march=native, ms, int64 random lines. Without dominated-segment pruning, random segments
	 * on |x|<=1e9 take 546 / 574 against 28.4 / 28.8 with the exit;
	 * on [0,1e6), 362 / 392 against 27.8 / 28.1. Suffixes take 230 / 251
	 * against 20.2 / 21.1, and 190 / 209 against 18.5 / 19.2 respectively.
	 * Improving segments exercise the exit when it never fires: on Xeon Platinum 8168,
	 * GCC 15.2, RelWithDebInfo, assertions off, 2026-11-03, the library takes
	 * 122 / 131 ms and the optional-exit copy without it 111 / 124 ms (-O2 / -O2 -march=native).
	 * Whole-first queries lose on lines: wide 7.3 / 7.1 against 7.0 / 6.7,
	 * dense 4.2 / 4.0 against 3.7 / 3.7 (+4% to +14%). Clip overwrite
	 * loses random segments on [0,1e6): 381 / 396 against 362 / 392 (+5% / +1%).
	 * Both knobs false preserve the behavior with both features disabled, node for node.
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>,
			 bool IsQueryWholeFirst = false, bool OverwritesLostClips = false>
		requires integral<Coordinate> && envelope_function<Function, Coordinate, Compare>
	class dynamic_li_chao_tree_by_query_order_and_clip_overwrite {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr dynamic_li_chao_tree_by_query_order_and_clip_overwrite() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions and no nodes yet.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr dynamic_li_chao_tree_by_query_order_and_clip_overwrite(Coordinate lo, Coordinate hi, Compare comp = {})
			: comp_(std::move(comp)), first_(lo), last_(hi), is_empty_(!(lo < hi)) {
			atl_assert(!(hi < lo));
			if (!is_empty_) --last_;
		}

		/** @brief Whether the domain has no points. */
		constexpr bool empty() const { return is_empty_; }

		/** @brief The number of materialized nodes. */
		constexpr std::size_t node_count() const { return nodes_.empty() ? 0 : nodes_.size() - 1; }

		/** @brief Reserve room for @p nodes materialized nodes. @pre `nodes < SIZE_MAX` (asserted). */
		constexpr void reserve(std::size_t nodes) {
			atl_assert(nodes < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(nodes + 1);
		}

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Function f) {
			if (is_empty_) return;
			atl_assert(atl::detail::evaluates_on(f, first_, last_));
			if (root_ == 0) root_ = make_node();
			descend(root_, first_, last_, std::move(f));
		}

		/**
		 * @brief Add `f` on the integers `x` with `l <= x < r` of the domain
		 *        only (a segment); an empty range adds nothing.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Function const &f, Coordinate l, Coordinate r) {
			if (is_empty_ || !(l < r)) return;
			--r;
			if (l < first_) l = first_;
			if (last_ < r) r = last_;
			if (r < l) return;
			atl_assert(atl::detail::evaluates_on(f, l, r));
			root_ = insert_segment(root_, first_, last_, l, r, f);
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its value at every domain point fits (asserted at the two ends)
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate l, Coordinate r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, std::move(l), std::move(r));
		}

		/**
		 * @brief The best value at `x` over the lines and segments covering
		 *        it, or none when nothing covers it.
		 * @pre `lo <= x < hi` (asserted)
		 * @complexity O(log W)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			atl_assert(!is_empty_ && !(x < first_) && !(last_ < x));
			std::optional<value_type> best;
			Coordinate first = first_, last = last_;
			auto const step = [&](node_type const &here) {
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				if (x <= middle) {
					last = middle;
					return here.left;
				}
				first = middle + 1;
				return here.right;
			};
			auto const take_clipped = [&](node_type const &leaf) {
				if (leaf.kind == held_kind::from_bound ? !(x < bound_of(leaf)) : !(bound_of(leaf) < x)) {
					value_type value = leaf.held(x);
					if (!best || comp_(value, *best)) best = std::move(value);
				}
			};
			uint32 node = root_;
			if constexpr (IsQueryWholeFirst) {
				// a clipped leaf has no children, so it can only end the path
				for (; node != 0; node = step(nodes_[node])) {
					node_type const &here = nodes_[node];
					if (here.kind > held_kind::whole) break;
					if (here.kind == held_kind::whole) {
						value_type value = here.held(x);
						if (!best || comp_(value, *best)) best = std::move(value);
					}
				}
				if (node != 0) take_clipped(nodes_[node]);
			} else {
				while (node != 0) {
					node_type const &here = nodes_[node];
					if (here.kind == held_kind::whole) {
						value_type value = here.held(x);
						if (!best || comp_(value, *best)) best = std::move(value);
					} else if (here.kind != held_kind::none) {
						// a clipped leaf, the last node on the path
						take_clipped(here);
						break;
					}
					node = step(here);
				}
			}
			return best;
		}

	private:
		/**
		 * @brief What a node holds: nothing, a function on its whole interval,
		 *        or a clipped one on `[bound, last]` / `[first, bound]` (then
		 *        the node has no children).
		 */
		enum class held_kind : uint8 { none,
									   whole,
									   from_bound,
									   to_bound };

		/** @brief A clipped leaf's bound lives in its unused child fields when it fits there. */
		static constexpr bool is_bound_packed = sizeof(Coordinate) <= sizeof(uint64);

		/** @brief Stands in for the bound field when the bound is packed. */
		struct packed_bound {};

		/** @brief A materialized node: a function (see `kind`) and child ids (0 = none). */
		struct node_type {
			Function held{};
			[[no_unique_address]] std::conditional_t<is_bound_packed, packed_bound, Coordinate> bound{};
			uint32 left = 0, right = 0;
			held_kind kind = held_kind::none;
		};

		/** @brief The bound of the clipped leaf `leaf`. */
		[[nodiscard]] static constexpr Coordinate bound_of(node_type const &leaf) {
			if constexpr (is_bound_packed)
				return Coordinate(uint64(leaf.left) | uint64(leaf.right) << 32);
			else
				return leaf.bound;
		}

		/** @brief Make the childless `leaf` clipped as `kind` at `bound`. */
		static constexpr void clip(node_type &leaf, held_kind kind, Coordinate const &bound) {
			leaf.kind = kind;
			if constexpr (is_bound_packed) {
				// a two's complement round trip through uint64 (modular since C++20)
				uint64 const bits = uint64(bound);
				leaf.left = uint32(bits);
				leaf.right = uint32(bits >> 32);
			} else {
				leaf.bound = bound;
			}
		}

		/**
		 * @brief Append an empty node and return its id: the one place node
		 *        ids are narrowed, so fewer than `2^32 - 1` nodes is asserted
		 *        here (id 0 is the null sentinel).
		 */
		constexpr uint32 make_node() {
			atl_assert(nodes_.size() < std::size_t(std::numeric_limits<uint32>::max()));
			nodes_.emplace_back();
			return uint32(nodes_.size() - 1);
		}

		/** @brief A new childless node holding `f` as `kind` (at `bound` unless whole). */
		constexpr uint32 make_leaf(held_kind kind, Coordinate const &bound, Function f) {
			uint32 const id = make_node();
			node_type &leaf = nodes_[id];
			leaf.held = std::move(f);
			if (kind == held_kind::whole)
				leaf.kind = kind;
			else
				clip(leaf, kind, bound);
			return id;
		}

		/**
		 * @brief Whether `challenger` is better than `incumbent` at none of
		 *        `from`, `to` and their floor midpoint, so (crossing at most
		 *        once) nowhere on `[from, to]`.
		 */
		[[nodiscard]] constexpr bool wins_nowhere(Function const &challenger, Function const &incumbent, Coordinate const &from,
												  Coordinate const &to) const {
			Coordinate const middle = atl::detail::floor_midpoint(from, to);
			return !comp_(challenger(from), incumbent(from)) && !comp_(challenger(middle), incumbent(middle)) && !comp_(challenger(to), incumbent(to));
		}

		/**
		 * @brief Whether the function of the clipped leaf `leaf` over
		 *        `[first, last]` wins nowhere on its clip range against `added`.
		 */
		[[nodiscard]] constexpr bool loses_clip(node_type const &leaf, Coordinate const &first, Coordinate const &last,
												Function const &added) const {
			Coordinate const bound = bound_of(leaf);
			return leaf.kind == held_kind::from_bound ? wins_nowhere(leaf.held, added, bound, last)
													  : wins_nowhere(leaf.held, added, first, bound);
		}

		/** @brief Empty the clipped leaf `leaf`, clearing the bound bits packed in its child fields. */
		static constexpr void unclip(node_type &leaf) {
			leaf.kind = held_kind::none;
			leaf.left = leaf.right = 0;
		}

		/**
		 * @brief Push the function of the clipped leaf `node` over
		 *        `[first, last]` into new children (whole where it covers
		 *        one, clipped where it does not), leaving `node` empty.
		 */
		constexpr void push_clip(uint32 node, Coordinate const &first, Coordinate const &last) {
			node_type &here = nodes_[node];
			held_kind const kind = here.kind;
			Coordinate const bound = bound_of(here);
			here.kind = held_kind::none;
			Function f = std::move(here.held);
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			uint32 left = 0, right = 0;
			if (kind == held_kind::from_bound) {
				if (bound <= middle) {
					left = make_leaf(bound == first ? held_kind::whole : held_kind::from_bound, bound, f);
					right = make_leaf(held_kind::whole, bound, std::move(f));
				} else {
					right = make_leaf(bound == middle + 1 ? held_kind::whole : held_kind::from_bound, bound, std::move(f));
				}
			} else if (middle < bound) {
				left = make_leaf(held_kind::whole, bound, f);
				right = make_leaf(bound == last ? held_kind::whole : held_kind::to_bound, bound, std::move(f));
			} else {
				left = make_leaf(bound == middle ? held_kind::whole : held_kind::to_bound, bound, std::move(f));
			}
			nodes_[node].left = left;
			nodes_[node].right = right;
		}

		/**
		 * @brief Settle `added` into the subtree of the existing `node` over
		 *        `[first, last]`, creating at most two nodes: keep the better
		 *        function at each midpoint, send the other into the half where it
		 *        can still win.
		 */
		constexpr void descend(uint32 node, Coordinate first, Coordinate last, Function added) {
			while (true) {
				if (nodes_[node].kind != held_kind::whole) {
					if (nodes_[node].kind != held_kind::none) {
						if (OverwritesLostClips && loses_clip(nodes_[node], first, last, added))
							unclip(nodes_[node]);
						else
							push_clip(node, first, last);
					}
					node_type &here = nodes_[node];
					here.held = std::move(added);
					here.kind = held_kind::whole;
					return;
				}
				node_type &here = nodes_[node];
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				bool const wins_middle = comp_(added(middle), here.held(middle));
				bool const wins_first = comp_(added(first), here.held(first));
				// crossing at most once, a function that wins (or loses) at both
				// ends of the node's range wins (or loses) on all of it: the
				// loser is dropped instead of walked down the tree
				if (wins_first == wins_middle && comp_(added(last), here.held(last)) == wins_middle) {
					if (wins_middle) here.held = std::move(added);
					return;
				}
				if (wins_middle) std::swap(here.held, added);
				bool const goes_left = wins_first != wins_middle;
				uint32 child = goes_left ? here.left : here.right;
				if (goes_left)
					last = middle;
				else
					first = middle + 1;
				if (child == 0) {
					child = make_node();
					(goes_left ? nodes_[node].left : nodes_[node].right) = child;
				}
				node = child;
			}
		}

		/**
		 * @brief Insert `added` on `[l, r]` into the subtree of `node` (0 = not
		 *        yet created) over `[first, last]`, which intersects `[l, r]`;
		 *        returns the subtree's node id. A missing node becomes a leaf
		 *        holding `added` (clipped unless covered), except one that
		 *        contains both `l` and `r` strictly inside.
		 */
		constexpr uint32 insert_segment(uint32 node, Coordinate first, Coordinate last, Coordinate const &l, Coordinate const &r,
										Function const &added) {
			bool const covers_first = !(first < l), covers_last = !(r < last);
			if (node == 0) {
				if (covers_first && covers_last) return make_leaf(held_kind::whole, l, added);
				if (covers_first) return make_leaf(held_kind::to_bound, r, added);
				if (covers_last) return make_leaf(held_kind::from_bound, l, added);
				node = make_node();
			} else if (covers_first && covers_last) {
				descend(node, first, last, added);
				return node;
			} else if (nodes_[node].kind > held_kind::whole) {
				// the clip range lies inside the segment exactly when the
				// segment covers the leaf's end on the clip's side and its bound
				bool const covers_clip = nodes_[node].kind == held_kind::from_bound
										   ? covers_last && !(bound_of(nodes_[node]) < l)
										   : covers_first && !(r < bound_of(nodes_[node]));
				if (OverwritesLostClips && covers_clip && loses_clip(nodes_[node], first, last, added))
					unclip(nodes_[node]);
				else
					push_clip(node, first, last);
			}
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			if (l <= middle) {
				uint32 const child = insert_segment(nodes_[node].left, first, middle, l, r, added);
				nodes_[node].left = child;
			}
			if (middle < r) {
				uint32 const child = insert_segment(nodes_[node].right, middle + 1, last, l, r, added);
				nodes_[node].right = child;
			}
			return node;
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<node_type> nodes_ = std::vector<node_type>(1);
		Coordinate first_{}, last_{};
		uint32 root_ = 0;
		bool is_empty_ = true;
	};

	/**
	 * @brief @ref atl::dynamic_li_chao_tree with the same layout whose
	 *        only switch is the dominated-segment exit, so a race against
	 *        the library tree isolates that one feature from the codegen of a
	 *        different class. `true` is the library behavior, node for node;
	 *        `false` descends into a segment's covered halves even when a whole
	 *        node dominates it. "Copy as library" against the library row shows
	 *        the copy's own noise.
	 *
	 * @tparam HasDominatedSegmentExit Whether a segment insertion stops at a
	 *         whole node that the segment beats nowhere on its covered part.
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>,
			 bool HasDominatedSegmentExit = true>
		requires integral<Coordinate> && envelope_function<Function, Coordinate, Compare>
	class dynamic_li_chao_tree_by_optional_dominated_exit {
	public:
		/** @brief The domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief The inserted function type. */
		using function_type = Function;
		/** @brief The function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr dynamic_li_chao_tree_by_optional_dominated_exit() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions and no nodes yet.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr dynamic_li_chao_tree_by_optional_dominated_exit(Coordinate lo, Coordinate hi, Compare comp = {})
			: comp_(std::move(comp)), first_(lo), last_(hi), is_empty_(!(lo < hi)) {
			atl_assert(!(hi < lo));
			if (!is_empty_) --last_;
		}

		/** @brief Whether the domain has no points. */
		constexpr bool empty() const { return is_empty_; }

		/** @brief The number of materialized nodes. */
		[[nodiscard]] constexpr std::size_t node_count() const { return nodes_.empty() ? 0 : nodes_.size() - 1; }

		/** @brief Reserve room for @p nodes materialized nodes. @pre `nodes < SIZE_MAX` (asserted). */
		constexpr void reserve(std::size_t nodes) {
			atl_assert(nodes < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(nodes + 1);
		}

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Function f) {
			if (is_empty_) return;
			atl_assert(atl::detail::evaluates_on(f, first_, last_));
			if (root_ == 0) root_ = make_node();
			descend(root_, first_, last_, std::move(f));
		}

		/**
		 * @brief Add `f` on the integers `x` with `l <= x < r` of the domain
		 *        only (a segment); an empty range adds nothing.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Function const &f, Coordinate l, Coordinate r) {
			if (is_empty_ || !(l < r)) return;
			--r;
			if (l < first_) l = first_;
			if (last_ < r) r = last_;
			if (r < l) return;
			atl_assert(atl::detail::evaluates_on(f, l, r));
			root_ = insert_segment(root_, first_, last_, l, r, f);
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its value at every domain point fits (asserted at the two ends)
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate l, Coordinate r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, std::move(l), std::move(r));
		}

		/**
		 * @brief The best value at `x` over the lines and segments covering
		 *        it, or none when nothing covers it.
		 * @pre `lo <= x < hi` (asserted)
		 * @complexity O(log W)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			atl_assert(!is_empty_ && !(x < first_) && !(last_ < x));
			std::optional<value_type> best;
			Coordinate first = first_, last = last_;
			for (uint32 node = root_; node != 0;) {
				node_type const &here = nodes_[node];
				if (here.kind == held_kind::whole) {
					value_type value = here.held(x);
					if (!best || comp_(value, *best)) best = std::move(value);
				} else if (here.kind != held_kind::none) {
					// a clipped leaf, the last node on the path
					if (here.kind == held_kind::from_bound ? !(x < bound_of(here)) : !(bound_of(here) < x)) {
						value_type value = here.held(x);
						if (!best || comp_(value, *best)) best = std::move(value);
					}
					break;
				}
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				if (x <= middle) {
					node = here.left;
					last = middle;
				} else {
					node = here.right;
					first = middle + 1;
				}
			}
			return best;
		}

	private:
		/**
		 * @brief What a node holds: nothing, a function on its whole interval,
		 *        or a clipped one on `[bound, last]` / `[first, bound]` (then
		 *        the node has no children).
		 */
		enum class held_kind : uint8 { none,
									   whole,
									   from_bound,
									   to_bound };

		/** @brief A clipped leaf's bound lives in its unused child fields when it fits there. */
		static constexpr bool is_bound_packed = sizeof(Coordinate) <= sizeof(uint64);

		/** @brief Stands in for the bound field when the bound is packed. */
		struct packed_bound {};

		/** @brief A materialized node: a function (see `kind`) and child ids (0 = none). */
		struct node_type {
			Function held{};
			[[no_unique_address]] std::conditional_t<is_bound_packed, packed_bound, Coordinate> bound{};
			uint32 left = 0, right = 0;
			held_kind kind = held_kind::none;
		};

		/** @brief The bound of the clipped leaf `leaf`. */
		[[nodiscard]] static constexpr Coordinate bound_of(node_type const &leaf) {
			if constexpr (is_bound_packed)
				return Coordinate(uint64(leaf.left) | uint64(leaf.right) << 32);
			else
				return leaf.bound;
		}

		/** @brief Make the childless `leaf` clipped as `kind` at `bound`. */
		static constexpr void clip(node_type &leaf, held_kind kind, Coordinate const &bound) {
			leaf.kind = kind;
			if constexpr (is_bound_packed) {
				// a two's complement round trip through uint64 (modular since C++20)
				uint64 const bits = uint64(bound);
				leaf.left = uint32(bits);
				leaf.right = uint32(bits >> 32);
			} else {
				leaf.bound = bound;
			}
		}

		/**
		 * @brief Append an empty node and return its id: the one place node
		 *        ids are narrowed, so fewer than `2^32 - 1` nodes is asserted
		 *        here (id 0 is the null sentinel).
		 */
		constexpr uint32 make_node() {
			atl_assert(nodes_.size() < std::size_t(std::numeric_limits<uint32>::max()));
			nodes_.emplace_back();
			return uint32(nodes_.size() - 1);
		}

		/** @brief A new childless node holding `f` as `kind` (at `bound` unless whole). */
		constexpr uint32 make_leaf(held_kind kind, Coordinate const &bound, Function f) {
			uint32 const id = make_node();
			node_type &leaf = nodes_[id];
			leaf.held = std::move(f);
			if (kind == held_kind::whole)
				leaf.kind = kind;
			else
				clip(leaf, kind, bound);
			return id;
		}

		/** @brief Whether a function loses throughout the covered interval; crossing at most once makes the endpoint tests sufficient. */
		[[nodiscard]] constexpr bool wins_nowhere(Function const &challenger, Function const &incumbent,
												  Coordinate const &from, Coordinate const &to) const {
			Coordinate const middle = atl::detail::floor_midpoint(from, to);
			return !comp_(challenger(from), incumbent(from)) && !comp_(challenger(middle), incumbent(middle))
				&& !comp_(challenger(to), incumbent(to));
		}

		/**
		 * @brief Push the function of the clipped leaf `node` over
		 *        `[first, last]` into new children (whole where it covers
		 *        one, clipped where it does not), leaving `node` empty.
		 */
		constexpr void push_clip(uint32 node, Coordinate const &first, Coordinate const &last) {
			node_type &here = nodes_[node];
			held_kind const kind = here.kind;
			Coordinate const bound = bound_of(here);
			here.kind = held_kind::none;
			Function f = std::move(here.held);
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			uint32 left = 0, right = 0;
			if (kind == held_kind::from_bound) {
				if (bound <= middle) {
					left = make_leaf(bound == first ? held_kind::whole : held_kind::from_bound, bound, f);
					right = make_leaf(held_kind::whole, bound, std::move(f));
				} else {
					right = make_leaf(bound == middle + 1 ? held_kind::whole : held_kind::from_bound, bound, std::move(f));
				}
			} else if (middle < bound) {
				left = make_leaf(held_kind::whole, bound, f);
				right = make_leaf(bound == last ? held_kind::whole : held_kind::to_bound, bound, std::move(f));
			} else {
				left = make_leaf(bound == middle ? held_kind::whole : held_kind::to_bound, bound, std::move(f));
			}
			nodes_[node].left = left;
			nodes_[node].right = right;
		}

		/**
		 * @brief Settle `added` into the subtree of the existing `node` over
		 *        `[first, last]`, creating at most two nodes: keep the better
		 *        function at each midpoint, send the other into the half where it
		 *        can still win.
		 */
		constexpr void descend(uint32 node, Coordinate first, Coordinate last, Function added) {
			while (true) {
				if (nodes_[node].kind != held_kind::whole) {
					if (nodes_[node].kind != held_kind::none) push_clip(node, first, last);
					node_type &here = nodes_[node];
					here.held = std::move(added);
					here.kind = held_kind::whole;
					return;
				}
				node_type &here = nodes_[node];
				Coordinate const middle = atl::detail::floor_midpoint(first, last);
				bool const wins_middle = comp_(added(middle), here.held(middle));
				bool const wins_first = comp_(added(first), here.held(first));
				// crossing at most once, a function that wins (or loses) at both
				// ends of the node's range wins (or loses) on all of it: the
				// loser is dropped instead of walked down the tree
				if (wins_first == wins_middle && comp_(added(last), here.held(last)) == wins_middle) {
					if (wins_middle) here.held = std::move(added);
					return;
				}
				if (wins_middle) std::swap(here.held, added);
				bool const goes_left = wins_first != wins_middle;
				uint32 child = goes_left ? here.left : here.right;
				if (goes_left)
					last = middle;
				else
					first = middle + 1;
				if (child == 0) {
					child = make_node();
					(goes_left ? nodes_[node].left : nodes_[node].right) = child;
				}
				node = child;
			}
		}

		/**
		 * @brief Insert `added` on `[l, r]` into the subtree of `node` (0 = not
		 *        yet created) over `[first, last]`, which intersects `[l, r]`;
		 *        returns the subtree's node id. A missing node becomes a leaf
		 *        holding `added` (clipped unless covered), except one that
		 *        contains both `l` and `r` strictly inside.
		 */
		constexpr uint32 insert_segment(uint32 node, Coordinate first, Coordinate last, Coordinate const &l, Coordinate const &r,
										Function const &added) {
			bool const covers_first = !(first < l), covers_last = !(r < last);
			if (node == 0) {
				if (covers_first && covers_last) return make_leaf(held_kind::whole, l, added);
				if (covers_first) return make_leaf(held_kind::to_bound, r, added);
				if (covers_last) return make_leaf(held_kind::from_bound, l, added);
				node = make_node();
			} else if (covers_first && covers_last) {
				descend(node, first, last, added);
				return node;
			} else if (nodes_[node].kind == held_kind::whole) {
				if (HasDominatedSegmentExit
					&& wins_nowhere(added, nodes_[node].held, covers_first ? first : l, covers_last ? last : r))
					return node;
			} else if (nodes_[node].kind > held_kind::whole) {
				push_clip(node, first, last);
			}
			Coordinate const middle = atl::detail::floor_midpoint(first, last);
			if (l <= middle) {
				uint32 const child = insert_segment(nodes_[node].left, first, middle, l, r, added);
				nodes_[node].left = child;
			}
			if (middle < r) {
				uint32 const child = insert_segment(nodes_[node].right, middle + 1, last, l, r, added);
				nodes_[node].right = child;
			}
			return node;
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<node_type> nodes_ = std::vector<node_type>(1);
		Coordinate first_{}, last_{};
		uint32 root_ = 0;
		bool is_empty_ = true;
	};
}
