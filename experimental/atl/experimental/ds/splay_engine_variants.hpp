#pragma once

/**
 * @file
 * @brief Splay contenders varying descent, ancestor propagation and node layout.
 *
 * Top-down descent restructures while walking. Bottom-up descent pushes while
 * walking and rotates on the return path. Repush adds a second ancestor push
 * pass; semi-splay changes indexed reads to rotate only halfway on zig-zigs.
 * Handle splaying uses an ancestor stack, store-owned scratch, pointer reversal,
 * or two upward parity passes for reversal-only payloads.
 *
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-24,
 * benchmarks/ds/splay_sequence.cpp, -O2, n=1e6, interleaved storage:
 * uniform insert/erase/fold takes 1545 ms top-down, 1922 ms bottom-up,
 * 2065 ms with repush, 1950 ms semi-splay. Handle splay plus prefix reversal:
 * ancestor stack 1942 ms, store buffer 1940 ms, pointer reversal 1969 ms,
 * parity walk 1890 ms. The library uses top-down descent and ancestor scratch.
 * Complete tables and both flag sets: docs/splay-sequence.md.
 */

#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_sequence.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::experimental {
	/** @brief Splay shape contenders (see the file comment). */
	enum class splay_shape : uint8 {
		bottom_up_repush, ///< Repeat ancestor propagation at the reached node.
		top_down,		  ///< Restructure during the descent.
		semi,			  ///< Use partial rotations for indexed reads.
		bottom_up		  ///< Rotate on an already-pushed return path.
	};

	/** @brief Choose how a handle splay settles its ancestor path. */
	enum class splay_path_push : uint8 {
		ancestor_stack,	  ///< Reuse a thread-local ancestor vector.
		store_buffer,	  ///< Use the store's ancestor vector.
		pointer_reversal, ///< Temporarily reverse parent links to avoid a vector.
		parity_walk		  ///< Use two upward reversal-parity passes.
	};

	/** @brief A store owning the scratch buffer for `store_buffer`. */
	template<typename Store>
	concept splay_scratch_store = splay_store<Store> && requires(Store &s) {
		{ s.splay_scratch() } -> std::same_as<std::vector<uint32> &>;
	};

	/**
	 * @brief A store whose only pending tag is reversal, exposing it for
	 *        `parity_walk`: `flip(x)` toggles the mark and orients x's own
	 *        aggregates, `clear_reversed(x)` drops the mark alone.
	 */
	template<typename Store>
	concept splay_reversal_store = splay_store<Store> && requires(Store &s, uint32 x) {
		{ s.is_reversed(x) } -> std::convertible_to<bool>;
		s.flip(x);
		s.clear_reversed(x);
	};

	/** @brief A splay engine with the shape and push strategy chosen at compile time. */
	template<splay_shape Shape, splay_path_push Push>
	struct splay_engine_by_strategy : atl::detail::splay_composites<splay_engine_by_strategy<Shape, Push>> {
		/** @brief Push the path from x's splay root down to x. */
		template<splay_store Store>
			requires(Push != splay_path_push::parity_walk || splay_reversal_store<Store>)
				 && (Push != splay_path_push::store_buffer || !atl::detail::splay_has_push<Store> || splay_scratch_store<Store>)
		static constexpr void push_path(Store &s, uint32 x) {
			if constexpr (Push == splay_path_push::parity_walk) {
				if constexpr (splay_reversal_store<Store>) push_path_parity(s, x);
			} else if constexpr (!atl::detail::splay_has_push<Store>) {
				return;
			} else if constexpr (Push == splay_path_push::pointer_reversal) {
				push_path_reversal(s, x);
			} else if constexpr (Push == splay_path_push::store_buffer) {
				if constexpr (splay_scratch_store<Store>) {
					if (atl::detail::splay_is_root(s, x))
						s.push(x);
					else
						atl::detail::splay_push_path_with(s, x, s.splay_scratch());
				}
			} else {
				atl::detail::splay_push_path(s, x);
			}
		}

		/** @brief Make @p x the root of its splay tree. @pre `x != 0` */
		template<splay_store Store>
			requires requires(Store &store, uint32 id) { push_path(store, id); }
		static constexpr void splay(Store &s, uint32 x) {
			atl_assert(x != 0);
			push_path(s, x);
			atl::detail::splay_pushed(s, x);
		}

		/** @brief As `atl::splay_engine::descend`, in the chosen shape. */
		template<splay_store Store, splay_probe Probe>
		static constexpr uint32 descend(Store &s, uint32 root, Probe &&probe) {
			if (root == 0) return 0;
			if constexpr (Shape == splay_shape::top_down) {
				return splay_engine::descend(s, root, probe);
			} else {
				uint32 const x = walk(s, root, probe);
				if constexpr (Shape == splay_shape::bottom_up_repush)
					splay(s, x);
				else
					atl::detail::splay_pushed(s, x);
				return x;
			}
		}

		/**
		 * @brief Read-only access by rank for the semi-splay shape: returns
		 *        (new root, node of rank k); the node rises about halfway.
		 */
		template<sized_splay_store Store>
			requires(Shape == splay_shape::semi)
		static constexpr std::pair<uint32, uint32> read_at(Store &s, uint32 root, uint32 k) {
			atl_assert(root != 0 && k < s.size(root));
			auto probe = [&s, k](uint32 x) mutable {
				uint32 const left_size = s.size(s.child(x, 0));
				if (k < left_size) return splay_step::left;
				if (k == left_size) return splay_step::stop;
				k -= left_size + 1;
				return splay_step::right;
			};
			uint32 const x = walk(s, root, probe);
			uint32 y = x; // the rising node, pulled when it is left behind
			while (!atl::detail::splay_is_root(s, y)) {
				uint32 const p = s.parent(y);
				if (atl::detail::splay_is_root(s, p)) {
					atl::detail::splay_rotate_up(s, y);
					break;
				}
				uint32 const g = s.parent(p);
				if ((s.child(g, 1) == p) == (s.child(p, 1) == y)) {
					atl::detail::splay_pull(s, y);
					atl::detail::splay_rotate_up(s, p);
					y = p;
				} else {
					atl::detail::splay_rotate_up(s, y);
					atl::detail::splay_rotate_up(s, y);
				}
			}
			atl::detail::splay_pull(s, y);
			return {y, x};
		}

		/** @brief Visit node ids in sequence order without recursion. */
		template<splay_store Store, typename Visit>
			requires std::invocable<Visit &, uint32>
		static constexpr void for_each(Store &s, uint32 root, Visit &&visit) {
			splay_engine::for_each(s, root, visit);
		}

	private:
		/** @brief Walk down, pushing, to the node the probe stops at. */
		template<splay_store Store, splay_probe Probe>
		static constexpr uint32 walk(Store &s, uint32 x, Probe &probe) {
			for (;;) {
				atl::detail::splay_push(s, x);
				splay_step const step = probe(x);
				if (step == splay_step::stop) return x;
				uint32 const next = s.child(x, step == splay_step::right);
				if (next == 0) return x;
				x = next;
			}
		}

		/**
		 * @brief Buffer-free push: parent links point down the path on the way up
		 *        and are restored on the way down.
		 */
		template<splay_store Store>
			requires atl::detail::splay_has_push<Store>
		static constexpr void push_path_reversal(Store &s, uint32 x) {
			uint32 below = 0, y = x, top_parent;
			for (;;) {
				bool const is_top = atl::detail::splay_is_root(s, y);
				uint32 const p = s.parent(y);
				s.set_parent(y, below);
				if (is_top) {
					top_parent = p;
					break;
				}
				below = y;
				y = p;
			}
			for (uint32 up = top_parent;;) {
				s.push(y);
				uint32 const down = s.parent(y);
				s.set_parent(y, up);
				if (y == x) break;
				up = y;
				y = down;
			}
		}

		/**
		 * @brief Two upward passes: the path's total reversal parity, then each node
		 *        settled from the parity of the marks above it.
		 */
		template<splay_reversal_store Store>
		static constexpr void push_path_parity(Store &s, uint32 x) {
			bool parity = false; // marks at the current node and above
			for (uint32 y = x;; y = s.parent(y)) {
				parity = parity != bool(s.is_reversed(y));
				if (atl::detail::splay_is_root(s, y)) break;
			}
			for (uint32 y = x, below = 0;;) {
				bool const is_above_odd = parity != bool(s.is_reversed(y));
				if (is_above_odd) s.flip(y); // what the ancestors' pushes would have done to y
				if (s.is_reversed(y)) {
					uint32 const l = s.child(y, 0), r = s.child(y, 1);
					s.set_child(y, 0, r);
					s.set_child(y, 1, l);
					// the child on the path already accounted for y's mark
					if (l != 0 && l != below) s.flip(l);
					if (r != 0 && r != below) s.flip(r);
					s.clear_reversed(y);
				}
				if (atl::detail::splay_is_root(s, y)) break;
				below = y;
				parity = is_above_odd;
				y = s.parent(y);
			}
		}
	};

	/**
	 * @brief `atl::splay_sequence` over any engine and node layout, the
	 *        full interface (so the oracle tests run every contender).
	 */
	template<typename Engine, atl::detail::splay_sequence_layout Layout, std::semiregular T, typename Op = void, typename F = void,
			 typename Mapping = void, typename Compose = void, bool IsReversible = false>
		requires atl::detail::splay_sequence_algebra<T, Op, F, Mapping, Compose>
			  && atl::detail::splay_sequence_engine<Engine,
													atl::detail::splay_sequence_pool<Layout, T, Op, F, Mapping, Compose, IsReversible>>
	class splay_sequence_by_strategy
		: public atl::detail::splay_sequence_impl<splay_sequence_by_strategy<Engine, Layout, T, Op, F, Mapping, Compose, IsReversible>,
												  Engine, Layout, T, Op, F, Mapping, Compose, IsReversible> {
		using base = atl::detail::splay_sequence_impl<splay_sequence_by_strategy, Engine, Layout, T, Op, F, Mapping, Compose,
													  IsReversible>;

	public:
		/** @brief Inherit the tier-specific constructors. */
		using base::base;
	};

	/**
	 * @brief A reversible fold rope over `atl::sized_splay_links:` the
	 *        engine's worked example with the fold kept in both orders (so a
	 *        non-commutative @p Op survives reversal), plus the hooks of
	 *        every push strategy. Fields are public for tests and benchmarks.
	 */
	template<std::semiregular Value, typename Op>
		requires binary_operation<Op, Value>
	struct reversible_fold_rope : sized_splay_links {
		/** @brief A node's value, its subtree fold in both orders, and the pending-reversal mark. */
		struct payload {
			Value value{};				///< Own node value.
			Value aggregate{};			///< Subtree fold in sequence order.
			Value reversed_aggregate{}; ///< Subtree fold in reverse order.
			bool is_marked = false;		///< reversal pending for the children
		};

		/** @brief Payloads indexed like links; sentinel zero holds the fold identity. */
		std::vector<payload> payloads;
		/** @brief Store-owned ancestor scratch for the buffer strategy. */
		std::vector<uint32> scratch;
		/** @brief Associative fold over node values. */
		[[no_unique_address]] Op op;

		/** @brief An empty rope: node 0 holds the identity in every fold. */
		constexpr reversible_fold_rope(Value identity, Op fold) : op(std::move(fold)) { payloads.push_back(payload{identity, identity, identity, false}); }

		/** @brief A fresh detached node holding @p v; returns its id. */
		constexpr uint32 make(Value v) {
			uint32 const id = add_node();
			payloads.push_back({v, v, v, false});
			return id;
		}

		/** @brief Engine hook: size and both folds from the children. */
		constexpr void pull(uint32 x) {
			pull_size(x);
			payload &n = payloads[x];
			payload const &l = payloads[child(x, 0)], &r = payloads[child(x, 1)];
			n.aggregate = std::invoke(std::as_const(op), std::invoke(std::as_const(op), l.aggregate, std::as_const(n.value)), r.aggregate);
			n.reversed_aggregate = std::invoke(std::as_const(op), std::invoke(std::as_const(op), r.reversed_aggregate, std::as_const(n.value)), l.reversed_aggregate);
		}

		/** @brief Parity-walk hook: whether x owes its children a reversal. */
		[[nodiscard]] constexpr bool is_reversed(uint32 x) const { return payloads[x].is_marked; }

		/** @brief Parity-walk hook: toggle x's mark and swap its two folds (no-op on node 0). */
		constexpr void flip(uint32 x) {
			if (x == 0) return;
			payload &n = payloads[x];
			n.is_marked = !n.is_marked;
			std::ranges::swap(n.aggregate, n.reversed_aggregate);
		}

		/** @brief Parity-walk hook: drop x's mark alone. */
		constexpr void clear_reversed(uint32 x) { payloads[x].is_marked = false; }

		/** @brief Engine hook: settle x's mark (swap its children, flip theirs). */
		constexpr void push(uint32 x) {
			if (!payloads[x].is_marked) return;
			uint32 const l = child(x, 0), r = child(x, 1);
			set_child(x, 0, r);
			set_child(x, 1, l);
			flip(l);
			flip(r);
			payloads[x].is_marked = false;
		}

		/** @brief x's subtree fold in sequence order (for `max_right`). */
		[[nodiscard]] constexpr Value const &aggregate(uint32 x) const { return payloads[x].aggregate; }

		/** @brief x's own value (for `max_right`). */
		[[nodiscard]] constexpr Value const &value(uint32 x) const { return payloads[x].value; }

		/** @brief Store-buffer hook: the scratch path buffer the store owns. */
		constexpr std::vector<uint32> &splay_scratch() { return scratch; }
	};
}
