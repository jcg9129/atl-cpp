#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace atl::detail {
	/**
	 * @brief A node store a treap engine can drive.
	 *
	 * The engine is storage- and payload-blind: it navigates through `left`/
	 * `right`/`size`/`priority` and mutates through exactly two id-in/id-out
	 * primitives, which is what lets one engine serve both in-place and
	 * path-copying (persistent) stores at compile time:
	 *
	 *  - `rebind(t, l, r)`: rewire `t`'s children and recompute its augmentation
	 *    (size and any aggregate), returning the id to use afterwards — the same
	 *    id for an in-place store, a fresh copy for a persistent one.
	 *  - `push(t)`: settle `t`'s pending tag one level (a no-op id-return when
	 *    the store has no tags), under the same id-in/id-out contract.
	 *
	 * Id 0 is the null tree; `size(0)` must be 0. That `rebind` really recomputes
	 * the augmentation is a documented contract — a concept can only enforce the
	 * shape.
	 */
	template<typename S>
	concept treap_store = requires(S s, S const cs, uint32 t, uint32 l, uint32 r) {
		{ cs.left(t) } -> std::convertible_to<uint32>;
		{ cs.right(t) } -> std::convertible_to<uint32>;
		{ cs.size(t) } -> std::convertible_to<uint32>;
		{ cs.priority(t) } -> std::convertible_to<uint32>;
		{ s.rebind(t, l, r) } -> std::convertible_to<uint32>;
		{ s.push(t) } -> std::convertible_to<uint32>;
	};

	/**
	 * @brief A `treap_store` that also keeps a parent link per node, so a
	 *        node can be reached by id and walked upward (`treap_base::root_of`,
	 *        `treap_base::split_at_node`): what an Euler tour tree needs to
	 *        find a vertex's tree without a key to descend by.
	 *
	 * The extra contract (documented, not checkable): `rebind`, `rebind_left`
	 * and `rebind_right` write `t` as the parent of the children they install
	 * (writing the null node's parent is allowed and never read), and every
	 * tree root has parent 0. `treap_base::merge` keeps the root rule (its
	 * root is one of its two roots); `treap_base::split` does not (its
	 * results are former interior nodes), so a parent store splits with
	 * `treap_base::split_at_node` or clears the returned roots' parents.
	 */
	template<typename S>
	concept treap_parent_store = treap_store<S> && requires(S s, S const cs, uint32 t, uint32 p) {
		{ cs.parent(t) } -> std::convertible_to<uint32>;
		s.set_parent(t, p);
	};

	/** @brief Where `treap_base::split_at_node` puts the node it splits at. */
	enum class treap_split_place : uint8 {
		left,	  ///< the node ends the left part
		right,	  ///< the node starts the right part
		excluded, ///< the node is detached as a singleton, in neither part
	};

	/**
	 * @brief Structural treap algorithms over any `treap_store` — the merge /
	 *        split pair every operation composes from, written once for all
	 *        stores (mutable sequence, keyed map, persistent variants).
	 *
	 * Priorities form a max-heap: the higher-priority node is the root. All
	 *  expected-O(log n) bounds assume priorities drawn uniformly at random.
	 */
	struct treap_base {
		/**
		 * @brief Rewire only the left child, through the store's single-sided fast path
		 * when it offers one (in-place stores do; path-copying stores rebuild the
		 * whole node anyway, so they provide only the two-sided form).
		 */
		template<treap_store S>
		static constexpr uint32 rebind_left(S &s, uint32 t, uint32 l) {
			if constexpr (requires { { s.rebind_left(t, l) } -> std::convertible_to<uint32>; })
				return s.rebind_left(t, l);
			else
				return s.rebind(t, l, s.right(t));
		}

		/** @brief Mirror of `rebind_left`. */
		template<treap_store S>
		static constexpr uint32 rebind_right(S &s, uint32 t, uint32 r) {
			if constexpr (requires { { s.rebind_right(t, r) } -> std::convertible_to<uint32>; })
				return s.rebind_right(t, r);
			else
				return s.rebind(t, s.left(t), r);
		}

		/**
		 * @brief Join two trees, every element of @p a before every element of @p b.
		 * @complexity expected O(log n)
		 */
		template<treap_store S>
		static constexpr uint32 merge(S &s, uint32 a, uint32 b) {
			if (a == 0 || b == 0) return a | b;
			if (s.priority(a) < s.priority(b)) {
				b = s.push(b);
				return rebind_left(s, b, merge(s, a, s.left(b)));
			}
			a = s.push(a);
			return rebind_right(s, a, merge(s, s.right(a), b));
		}

		/**
		 * @brief Split into (left, right) parts: a node joins the left part
		 *        exactly when @p goes_left says so, and the predicate must be
		 *        monotone along the in-order sequence (a prefix goes left).
		 *
		 * The predicate is invoked on already-pushed nodes and is shared by
		 * reference through the recursion, so a stateful (mutable) predicate —
		 * counting positions, comparing keys, accumulating aggregates — sees
		 * every probed node exactly once, in root-to-leaf order.
		 * @complexity expected O(log n)
		 */
		template<treap_store S, typename GoesLeft>
			requires std::predicate<GoesLeft &, S &, uint32>
		static constexpr std::pair<uint32, uint32> split(S &s, uint32 t, GoesLeft &&goes_left) {
			if (t == 0) return {0, 0};
			t = s.push(t);
			if (std::invoke(goes_left, s, t)) {
				auto const parts = split(s, s.right(t), goes_left);
				uint32 const middle = parts.first, right = parts.second;
				return {rebind_right(s, t, middle), right};
			}
			auto const parts = split(s, s.left(t), goes_left);
			uint32 const left = parts.first, middle = parts.second;
			return {left, rebind_left(s, t, middle)};
		}

		/**
		 * @brief Map a uniform 64-bit draw onto `[0, total)`: multiply-shift reduction — the same negligible bias as `draw % total`
		 *        (at most ~total/2^64 relative, and `total < 2^34` here)
		 *        without a hardware division on every merge step. The widened
		 *        product also works with software 128-bit integers.
		 * @pre `0 < total < 2^34`.
		 */
		static constexpr uint64 bounded_draw(uint64 draw, uint64 total) {
			using wide = traits::widen_mul_type_t<uint64>;
			if constexpr (traits::has_widen_mul_v<uint64>)
				return uint64(wide(draw) * total >> 64);
			else
				return draw % total;
		}

		/**
		 * @brief Join like `merge`, but the root is chosen by a size-weighted
		 *        coin from @p rand instead of stored priorities — the
		 *        size-weighted randomized join.
		 *
		 * This is the correct merge for **persistent** treaps: path copying
		 * shares subtrees between versions, so a tree may be merged with (a copy
		 * of) itself, and stored priorities then compare nodes against
		 * themselves — the correlation voids the expected-depth bound. A fresh
		 * coin per call cannot be correlated by sharing. @p rand is any
		 * `() -> uint64` uniform source.
		 * @complexity expected O(log n)
		 */
		template<treap_store S, typename Rand>
			requires std::invocable<Rand &> && std::convertible_to<std::invoke_result_t<Rand &>, uint64>
		static constexpr uint32 merge_weighted(S &s, uint32 a, uint32 b, Rand &&rand) {
			if (a == 0 || b == 0) return a | b;
			uint64 const sa = s.size(a), total = sa + s.size(b);
			if (bounded_draw(std::invoke(rand), total) < sa) {
				a = s.push(a);
				return rebind_right(s, a, merge_weighted(s, s.right(a), b, rand));
			}
			b = s.push(b);
			return rebind_left(s, b, merge_weighted(s, a, s.left(b), rand));
		}

		/**
		 * @brief Join for trees whose sizes have **saturated** (overflowed the
		 *        stored width): pin the result's root from one input and splice
		 *        the other in whole, instead of interleaving by a coin.
		 *
		 * With saturated sizes the weighted coin degenerates to ~50/50, and an
		 * interleaving merge then costs the *sum* of the opposing spines —
		 * repeated giant merges (self-concat, cross-merging lineages) grow both
		 * spines without bound. This join copies only the **shorter** of a's
		 * right spine and b's left spine (found by a lockstep walk, no stored
		 * rank fields), burying it in the interior where end-splits never look:
		 * the result's left spine is a's plus at most one node, its right spine
		 * is b's plus at most one — each merge grows one end-spine by ≤ 1, and
		 * long spines are consumed, not compounded.
		 *
		 * The result is not priority-shaped; use only with stores that join by
		 * per-call rules (see `merge_weighted`), never with `merge`.
		 * @complexity O(min(right-spine(a), left-spine(b)))
		 */
		template<treap_store S>
		static constexpr uint32 merge_spines(S &s, uint32 a, uint32 b) {
			if (a == 0 || b == 0) return a | b;
			uint32 x = a, y = b;
			while (x != 0 && y != 0) {
				x = s.right(x);
				y = s.left(y);
			}
			if (x == 0) { // a's right spine is the cheaper to consume
				b = s.push(b);
				return rebind_left(s, b, splice_right(s, a, s.left(b)));
			}
			a = s.push(a);
			return rebind_right(s, a, splice_left(s, s.right(a), b));
		}

		/**
		 * @brief Consume @p a's entire right spine, hanging @p b past its last node.
		 * @complexity O(right-spine(a))
		 */
		template<treap_store S>
		static constexpr uint32 splice_right(S &s, uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			a = s.push(a);
			return rebind_right(s, a, splice_right(s, s.right(a), b));
		}

		/**
		 * @brief Mirror of `splice_right:` consume @p b's entire left spine.
		 * @complexity O(left-spine(b))
		 */
		template<treap_store S>
		static constexpr uint32 splice_left(S &s, uint32 a, uint32 b) {
			if (b == 0) return a;
			if (a == 0) return b;
			b = s.push(b);
			return rebind_left(s, b, splice_left(s, a, s.left(b)));
		}

		/**
		 * @brief The root of the tree holding @p x, by the parent links. No writes.
		 * @pre `x != 0` @complexity O(depth of x), expected O(log n)
		 */
		template<treap_parent_store S>
		static constexpr uint32 root_of(S const &s, uint32 x) {
			for (uint32 p = s.parent(x); p != 0; p = s.parent(p)) x = p;
			return x;
		}

		/**
		 * @brief Split the tree holding @p x at @p x itself, bottom-up: the
		 *        handle form of `split`, for parent stores (no key or index
		 *        to descend by). Both returned roots get parent 0; on
		 *        `treap_split_place::excluded` x is left a detached singleton.
		 *        A nonzero @p stop (an ancestor-or-self of x) splits only the
		 *        subtree rooted there, leaving its old parent's child slot stale
		 *        for the caller to rewire.
		 *
		 * Walking up from x, each ancestor joins the part on its own side of x
		 * and adopts the part built so far as its child on x's side: the parts
		 * keep the heap order (an ancestor outranks its subtree), so this is
		 * exactly the tree the top-down split would build, without the
		 * descent's comparisons and without recursion.
		 * @pre every node from x to its root is pushed (no pending tag above or
		 *      at x), so the rebinds pull current children
		 * @complexity O(depth of x), expected O(log n)
		 */
		template<treap_parent_store S>
		static constexpr std::pair<uint32, uint32> split_at_node(S &s, uint32 x, treap_split_place place,
																 uint32 stop = 0) {
			uint32 const up = s.parent(x);
			uint32 l = 0, r = 0;
			if (place == treap_split_place::left) {
				r = s.right(x);
				l = rebind_right(s, x, 0);
			} else if (place == treap_split_place::right) {
				l = s.left(x);
				r = rebind_left(s, x, 0);
			} else {
				l = s.left(x);
				r = s.right(x);
				s.rebind(x, 0, 0);
				s.set_parent(x, 0);
			}
			for (uint32 c = x, p = up; c != stop && p != 0;) {
				uint32 const next = s.parent(p);
				if (s.right(p) == c)
					l = rebind_right(s, p, l);
				else
					r = rebind_left(s, p, r);
				c = p;
				p = next;
			}
			s.set_parent(l, 0);
			s.set_parent(r, 0);
			return {l, r};
		}

		/**
		 * @brief Insert the whole tree @p m into the tree holding @p v, just
		 *        before v, without splitting and re-merging the big tree: only
		 *        the subtree that m's root outranks is split and merged.
		 *
		 * Top-down, m's root would settle at the first node on the way to the
		 * gap before v that it outranks. On v's root path that node is the
		 * highest ancestor y it outranks: y's subtree (holding v) is split at v
		 * and rebuilt as left, m, right in y's place. When v itself outranks m,
		 * the gap lies on the right spine of v's left subtree: m merges after
		 * the first spine node it outranks, in that node's place. Every node
		 * above the rebuilt subtree is pulled once.
		 * @pre @p m is a root (parent 0) of another tree; v's root path is
		 *      pushed
		 * @complexity O(depth of v) plus the merges of the rebuilt subtree,
		 *             expected O(log n)
		 */
		template<treap_parent_store S>
			requires requires(S &store, uint32 id) { store.pull(id); }
		static constexpr void insert_before_node(S &s, uint32 v, uint32 m) {
			if (m == 0) return;
			uint32 const pm = s.priority(m);
			uint32 above = 0, sub = 0;
			bool side = false;
			if (s.priority(v) >= pm) {
				above = v;
				side = false;
				uint32 y = s.left(v);
				while (y != 0 && s.priority(y) >= pm) {
					above = s.push(y);
					side = true;
					y = s.right(y);
				}
				sub = merge(s, y, m); // merge reads no parents: y's stale one is overwritten below
			} else {
				uint32 y = v;
				for (uint32 p = s.parent(y); p != 0 && s.priority(p) < pm; p = s.parent(p)) y = p;
				above = s.parent(y);
				side = above != 0 && s.right(above) == y;
				auto const [l, r] = split_at_node(s, v, treap_split_place::right, y);
				sub = merge(s, merge(s, l, m), r);
				if (above == 0) return; // y was the root: sub is the new one (parent 0)
			}
			if (side)
				rebind_right(s, above, sub);
			else
				rebind_left(s, above, sub);
			for (uint32 x = s.parent(above); x != 0; x = s.parent(x)) s.pull(x);
		}

		/**
		 * @brief Split into the first @p k elements and the rest.
		 * @complexity expected O(log n)
		 */
		template<treap_store S>
		static constexpr std::pair<uint32, uint32> split_at(S &s, uint32 t, uint32 k) {
			if (t == 0) return {0, 0};
			t = s.push(t);
			uint32 const left_size = s.size(s.left(t));
			if (left_size < k) {
				auto const parts = split_at(s, s.right(t), k - left_size - 1);
				uint32 const middle = parts.first, right = parts.second;
				return {rebind_right(s, t, middle), right};
			}
			auto const parts = split_at(s, s.left(t), k);
			uint32 const left = parts.first, middle = parts.second;
			return {left, rebind_left(s, t, middle)};
		}
	};
}
