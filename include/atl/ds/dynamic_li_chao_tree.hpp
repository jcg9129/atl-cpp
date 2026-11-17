#pragma once

/**
 * @file
 * @brief @ref atl::dynamic_li_chao_tree, the lower (or upper) envelope of lines
 *        (or of any family crossing at most once) and their segments over a
 *        huge integer domain `[lo, hi)`, nodes created
 *        only where functions settle -- the online counterpart of @ref atl::li_chao_tree's coordinate mode.
 */

#include <atl/ds/linear_function.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Lines `y = slope * x + intercept` -- or members of any function
	 *        family in which two members cross at most once -- and segments
	 *        over every integer of `[lo, hi)` -- 10^9, 10^18 wide -- keeping the best value at
	 *        each point, with no query points known up front (the Li Chao
	 *        tree over a node pool).
	 *
	 * The same envelope as @ref atl::li_chao_tree (descents and their endpoint
	 * pruning included), on the implicit tree that
	 * halves `[lo, hi)` at floor midpoints down to single integers, with only
	 * the nodes that hold a function (or lie above a segment's canonical
	 * nodes) materialized. A segment's boundary path stops at the first node
	 * that does not exist yet: the segment stays there as a clipped leaf, a
	 * childless node holding it on `[bound, last]` or `[first, bound]` of its
	 * interval only, pushed one level down when a later insertion passes
	 * through. A whole-domain insertion creates at most two nodes, a segment
	 * insertion O(log W) for a domain of width `W`. Node ids are `uint32`.
	 *
	 * A segment insertion stops at a whole node that dominates the segment on
	 * its covered part. The race measured int64 lines on random traffic only.
	 *
	 * Memory on the measured x86-64 ABI: a node is 32 bytes for `int64` lines and 64 bytes for `int128`
	 * lines; a segment insertion adds about `2 log W` nodes (a whole-domain
	 * one at most two).
	 *
	 * @tparam Coordinate A signed or unsigned integer type for the domain
	 *         (`int64`, `int128`).
	 * @tparam Compare `comp(a, b)` reads "a is better than b" on the values:
	 *         std::less<> for the minimum, std::greater<> for the maximum.
	 * @tparam Function The family's member type (@ref atl::envelope_function),
	 *         `linear_function<Coordinate>` by default, which also gets the
	 *         `insert(slope, intercept)` shorthands and an exact overflow
	 *         check.
	 * @pre any two inserted functions cross at most once over the domain
	 *      (not checkable; documented); every evaluation at a covered point
	 *      is valid, including intermediates (asserted at insertion for integer lines).
	 *      Comparisons must remain ordered; floating evaluations must be finite.
	 *      Midpoints use bit arithmetic without forming a domain width or endpoint sum.
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
		requires integral<Coordinate> && envelope_function<Function, Coordinate, Compare>
	class dynamic_li_chao_tree {
	public:
		/** @brief The domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief The inserted function type. */
		using function_type = Function;
		/** @brief The function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr dynamic_li_chao_tree() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions and no nodes yet.
		 * @param lo Inclusive domain bound.
		 * @param hi Exclusive domain bound.
		 * @param comp Strict weak order defining the better value.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr dynamic_li_chao_tree(Coordinate lo, Coordinate hi, Compare comp = {})
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
		 * @param f Function whose evaluations satisfy the class preconditions.
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Function f) {
			if (is_empty_) return;
			atl_assert(detail::evaluates_on(f, first_, last_));
			if (root_ == 0) root_ = make_node();
			descend(root_, first_, last_, std::move(f));
		}

		/**
		 * @brief Add `f` on the integers `x` with `l <= x < r` of the domain
		 *        only (a segment); an empty range adds nothing.
		 * @param f Function to add.
		 * @param l Inclusive segment bound.
		 * @param r Exclusive segment bound.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Function const &f, Coordinate l, Coordinate r) {
			if (is_empty_ || !(l < r)) return;
			--r;
			if (l < first_) l = first_;
			if (last_ < r) r = last_;
			if (r < l) return;
			atl_assert(detail::evaluates_on(f, l, r));
			root_ = insert_segment(root_, first_, last_, l, r, f);
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its product and sum at every domain point fit (asserted at the two ends)
		 * @complexity O(log W), at most two new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 W), O(log W) new nodes
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate l, Coordinate r)
			requires detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, std::move(l), std::move(r));
		}

		/**
		 * @brief The best value at `x` over the lines and segments covering
		 *        it, or none when nothing covers it.
		 * @param x Domain coordinate to evaluate.
		 * @return Best covering value, or nullopt if no function covers x.
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
				Coordinate const middle = detail::floor_midpoint(first, last);
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
			atl_assert(nodes_.size() < std::size_t((std::numeric_limits<uint32>::max)()));
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
			Coordinate const middle = detail::floor_midpoint(from, to);
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
			Coordinate const middle = detail::floor_midpoint(first, last);
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
				Coordinate const middle = detail::floor_midpoint(first, last);
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
				if (wins_nowhere(added, nodes_[node].held, covers_first ? first : l, covers_last ? last : r)) return node;
			} else if (nodes_[node].kind > held_kind::whole) {
				push_clip(node, first, last);
			}
			Coordinate const middle = detail::floor_midpoint(first, last);
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
