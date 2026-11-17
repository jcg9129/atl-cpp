#pragma once

/**
 * @file
 * @brief @ref atl::meldable_heap_forest — many priority queues over one shared node
 *        pool, with push, pop, O(1) meld and a lazy add-to-all per heap.
 */

#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The default lazy-add mapping: `mapping(shift, value)` is a copy of
	 *        `value` with `+= shift` applied (any number, or any type with that
	 *        operator). Pass your own callable when `T` has no such operator.
	 */
	struct shift_by_addition {
		/** @brief Return value shifted by addition. @param shift Amount to add. @param value Element to shift. */
		template<typename Shift, typename T>
			requires std::copy_constructible<T> && requires(T &value, Shift const &shift) { value += shift; }
		[[nodiscard]] constexpr T operator()(Shift const &shift, T value) const {
			value += shift;
			return value;
		}
	};

	/**
	 * @brief The default lazy-add composition: `compose(newer, older)` is
	 *        `older` with `+= newer` applied.
	 */
	struct compose_by_addition {
		/** @brief Compose additive shifts, applying older first. @param newer Next shift. @param older Pending shift. */
		template<typename Shift>
			requires std::copy_constructible<Shift> && requires(Shift &pending, Shift const &shift) { pending += shift; }
		[[nodiscard]] constexpr Shift operator()(Shift const &newer, Shift older) const {
			older += newer;
			return older;
		}
	};

	/**
	 * @brief The parameters of a @ref atl::meldable_heap_forest — movable elements,
	 *        a strict weak order on them, and either no lazy add (`Shift =
	 *        void`, `Mapping` and `Compose` unused) or a copyable `Shift` acting
	 *        on elements through `Mapping` (an @ref atl::action, `mapping(shift,
	 *        value)` returning the moved element) and composing through
	 *        `Compose` (`compose(newer, older)`: `older` first, then `newer`),
	 *        the convention of @ref atl::lazy_segment_tree.
	 *
	 * Not checkable, so documented: one shift applied to every element of a
	 * heap must keep their order under `Compare` (true for adding a number,
	 * barring overflow, or for moving the position of a `(position, weight)`
	 * pair ordered by position), the identity shift must change nothing, and
	 * `mapping(compose(f, g), x) == mapping(f, mapping(g, x))`.
	 */
	template<typename T, typename Compare, typename Shift, typename Mapping, typename Compose>
	concept meldable_heap_parameters = std::movable<T> && std::strict_weak_order<Compare &, T const &, T const &>
									&& (std::is_void_v<Shift>
										|| (std::copyable<Shift> && action<Mapping, Shift, T> && binary_operation<Compose, Shift>));

	namespace detail {
		/** @brief The stored tag type when a forest has no lazy add: empty, so `[[no_unique_address]]` drops it. */
		struct no_heap_shift {};

		/**
		 * @brief The stored predecessor link of a forest without handles: empty,
		 *        and a different type from `no_heap_shift`, since two empty
		 *        `[[no_unique_address]]` members of one type cannot share an
		 *        address (the node would grow from 16 to 24 bytes for int64).
		 */
		struct no_heap_link {};

		/** @brief The per-node tag type: `Shift`, or the empty `no_heap_shift` for `Shift = void`. */
		template<typename Shift>
		using heap_tag_t = std::conditional_t<std::is_void_v<Shift>, no_heap_shift, Shift>;

		/**
		 * @brief Whether a pending shift is known to be the identity, so handing
		 *        it down can be skipped (never known without `==` on `Shift`).
		 */
		template<std::copyable Shift>
		constexpr bool is_identity_shift(Shift const &pending, Shift const &identity) {
			if constexpr (std::equality_comparable<Shift>)
				return pending == identity;
			else
				return false;
		}

		/** @brief One heap of a forest: its root node (0 = empty) and its element count. */
		struct heap_record {
			uint32 root = 0;
			uint32 size = 0;
		};

		/**
		 * @brief The heap ids of a node-pool forest: one `heap_record` per
		 *        heap, ids `0, 1, ...` in creation order, and the id-checked
		 *        queries every forest shares.
		 */
		class heap_directory {
		public:
			/** @brief A heap's id: `0, 1, ...` in creation order. */
			using heap_id = uint32;

		protected:
			std::vector<heap_record> heaps_;

			/** @brief No heaps. */
			constexpr heap_directory() = default;

			/** @pre `heap_count < 2^32` (asserted) */
			constexpr explicit heap_directory(std::size_t heap_count) {
				atl_assert(heap_count < (std::numeric_limits<uint32>::max)());
				heaps_.resize(heap_count);
			}

			/** @pre `heap < heap_count()` (asserted) */
			constexpr heap_record &record(heap_id heap) {
				atl_assert(heap < heaps_.size());
				return heaps_[heap];
			}

			/** @pre `heap < heap_count()` (asserted) */
			constexpr heap_record const &record(heap_id heap) const {
				atl_assert(heap < heaps_.size());
				return heaps_[heap];
			}

		public:
			/** @brief The number of heaps, empty ones included. */
			constexpr std::size_t heap_count() const noexcept { return heaps_.size(); }

			/**
			 * @brief A new empty heap; returns its id, `heap_count() - 1`.
			 * @complexity O(1) amortized
			 */
			constexpr heap_id add_heap() {
				atl_assert(heaps_.size() + 1 < (std::numeric_limits<uint32>::max)());
				heaps_.emplace_back();
				return heap_id(heaps_.size() - 1);
			}

			/** @brief The number of elements in heap `heap`. */
			constexpr std::size_t size(heap_id heap) const { return record(heap).size; }

			/** @brief Whether heap `heap` holds no element. */
			constexpr bool empty(heap_id heap) const { return record(heap).size == 0; }
		};
	}

	/**
	 * @brief Many priority queues ("heaps", ids `0, 1, ...`) over one shared
	 *        node pool: push, top, pop, size, an O(1) meld of one heap into
	 *        another and, with `Shift`, a lazy `add_to_all` per heap -- the
	 *        bottom-up "meld the children's heaps into the parent" pattern of
	 *        tree DPs, the slope trick and min-cost flow on trees.
	 *
	 * Each heap is a pairing heap: a heap-ordered tree whose children form a
	 * first-child / next-sibling list. Push and meld link two roots (the
	 * worse one becomes the first child of the better); pop pairs the old
	 * root's children left to right and folds the pairs right to left
	 * (two-pass pairing), in place through the sibling links. Nodes live in
	 * one `std::vector` with `uint32` links; popped nodes go on a free list
	 * that later pushes reuse, so the pool never exceeds the peak number of
	 * live elements.
	 *
	 * The order follows `std::priority_queue`: `top` is the comp-largest
	 * element, so `std::less<>` (the default) gives a max-heap and
	 * `std::greater<>` a min-heap.
	 *
	 * The lazy add (`Shift` not `void`) follows the @ref atl::lazy_segment_tree
	 * convention: `mapping(shift, value)` returns the shifted element,
	 * `compose(newer, older)` the combined pending shift, and an identity
	 * shift is given at construction (`Shift{}` with the short constructor);
	 * the defaults use `+=`. `add_to_all(heap, s)` shifts the elements in the
	 * heap at that moment; elements pushed later are given in the same
	 * (real) coordinates and are not shifted, and melding keeps every
	 * element's shifted value. A pending shift sits on a node for the node's
	 * descendants (the node's own value is already shifted), so `top` reads
	 * a stored value and a shift is handed down one level whenever a link or
	 * a pop touches the node (skipped when it compares equal to the identity).
	 *
	 * @tparam T The element type (movable).
	 * @tparam Compare A strict weak order on `T`; `top` is the comp-largest.
	 * @tparam Shift The lazy add's type, or `void` for none (no per-node tag
	 *         is stored then).
	 * @tparam Mapping `mapping(shift, value) -> T` (@ref atl::action); unused
	 *         without a lazy add.
	 * @tparam Compose `compose(newer, older) -> Shift`, `older` applied first
	 *         (@ref atl::binary_operation); unused without a lazy add.
	 * @tparam HasHandles `true` (only without a lazy add; see @ref
	 *         addressable_meldable_heap_forest) adds `promote`, `erase` and
	 *         `value` by the handle `push` returns, at the price of a
	 *         predecessor link per node, kept up to date on every link: up to 2.1x
	 *         the time on the race's meld workloads (docs).
	 * @pre (documented, see @ref atl::meldable_heap_parameters) a shift keeps the
	 *      order of the elements it moves.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition, bool HasHandles = false>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose> && (!HasHandles || std::is_void_v<Shift>)
	class meldable_heap_forest : public detail::heap_directory {
		/** @brief Lazy tag type. */
		using tag_type = detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;
		// promote / erase need each node's predecessor in its list; with a lazy
		// add they would also need the pending shifts of the whole ancestor
		// path (unbounded depth), so handles exclude a lazy add
		static constexpr bool has_handles = HasHandles;
		static constexpr uint32 freed = (std::numeric_limits<uint32>::max)(); // `prev` of a node on the free list

		struct node {
			T value;
			[[no_unique_address]] tag_type tag{};
			uint32 child = 0, sibling = 0;
			/** @brief The parent if this is a first child, else the preceding sibling; 0 for a root. */
			[[no_unique_address]] std::conditional_t<has_handles, uint32, detail::no_heap_link> prev{};
		};

		/** @brief The node without a lazy add or handles: what `node` must not outgrow then. */
		struct bare_node {
			T value;
			uint32 child, sibling;
		};

		// Empty tag/link members must not grow an int64 node from 16 to 24 bytes.
		static_assert(has_shift || has_handles || sizeof(node) == sizeof(bare_node));

		std::vector<node> nodes_; // node x (1-based; 0 means none) is nodes_[x - 1]
		uint32 free_ = 0;		  // the free list, linked through `child`
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] tag_type shift_identity_{};

		constexpr node &at(uint32 x) { return nodes_[x - 1]; }

		constexpr node const &at(uint32 x) const { return nodes_[x - 1]; }

		/** @brief Shift `x`'s own value and add the shift to what it owes its descendants. */
		constexpr void apply(uint32 x, tag_type const &shift) {
			node &n = at(x);
			n.value = mapping_(shift, std::move(n.value));
			n.tag = compose_(shift, n.tag);
		}

		/**
		 * @brief Hand `x`'s pending shift to its first child and next sibling:
		 *        in the binary view of the lists those are the roots of
		 *        everything below `x`, so this is O(1).
		 */
		constexpr void push_down(uint32 x) {
			if constexpr (has_shift) {
				node &n = at(x);
				if (detail::is_identity_shift(n.tag, shift_identity_)) return;
				if (n.child != 0) apply(n.child, n.tag);
				if (n.sibling != 0) apply(n.sibling, n.tag);
				n.tag = shift_identity_;
			}
		}

		/** @brief A fresh node holding `T(args...)`, from the free list when it has one. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 allocate(Args &&...args) {
			if (free_ != 0) {
				uint32 const x = free_;
				node &n = at(x);
				free_ = n.child;
				n.value = T(std::forward<Args>(args)...);
				n.tag = shift_identity_;
				n.child = n.sibling = 0;
				if constexpr (has_handles) n.prev = 0;
				return x;
			}
			atl_assert(nodes_.size() + 1 < (std::numeric_limits<uint32>::max)());
			nodes_.push_back(node{T(std::forward<Args>(args)...), shift_identity_});
			return uint32(nodes_.size());
		}

		/** @brief Link two roots with no pending shift and no sibling; returns the new root. */
		constexpr uint32 link(uint32 a, uint32 b) {
			// raced against a conditional select: a tie (docs/meldable-heap-forest.md, tuning)
			if (comp_(at(a).value, at(b).value)) std::swap(a, b);
			uint32 const first_child = at(a).child;
			at(b).sibling = first_child;
			at(a).child = b;
			if constexpr (has_handles) {
				at(b).prev = a;
				if (first_child != 0) at(first_child).prev = b;
			}
			return a;
		}

		/** @brief Link two roots, either possibly 0 (empty); returns the new root. */
		constexpr uint32 meld_roots(uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			push_down(a);
			push_down(b);
			return link(a, b);
		}

		/** @brief Two-pass pairing of the sibling list starting at `first`; returns the new root (0 for none). */
		constexpr uint32 pair_up(uint32 first) {
			if (first == 0) return 0;
			uint32 pairs = 0; // the pairs so far, last first, linked through `sibling`
			while (first != 0) {
				push_down(first);
				uint32 const second = at(first).sibling;
				if (second == 0) {
					at(first).sibling = pairs;
					pairs = first;
					break;
				}
				push_down(second);
				uint32 const rest = at(second).sibling;
				uint32 const winner = link(first, second);
				at(winner).sibling = pairs;
				pairs = winner;
				first = rest;
			}
			uint32 root = pairs;
			pairs = at(root).sibling;
			while (pairs != 0) {
				uint32 const rest = at(pairs).sibling;
				root = link(root, pairs);
				pairs = rest;
			}
			at(root).sibling = 0;
			if constexpr (has_handles) at(root).prev = 0;
			return root;
		}

		/** @brief Put node `x` on the free list. */
		constexpr void release_node(uint32 x) {
			at(x).child = free_;
			if constexpr (has_handles) at(x).prev = freed;
			free_ = x;
		}

		/** @brief Detach the non-root `x` (with its subtree) from its parent's child list. */
		constexpr void cut(uint32 x)
			requires has_handles
		{
			node &n = at(x);
			if (at(n.prev).child == x)
				at(n.prev).child = n.sibling;
			else
				at(n.prev).sibling = n.sibling;
			if (n.sibling != 0) at(n.sibling).prev = n.prev;
			n.sibling = n.prev = 0;
		}

		/** @brief Whether `x` names a live node (not freed, in range); O(1). */
		constexpr bool is_live(uint32 x) const
			requires has_handles
		{
			return x != 0 && x <= nodes_.size() && at(x).prev != freed;
		}

	public:
		/** @brief The element type. */
		using value_type = T;
		/** @brief The lazy add's type (`void`: none). */
		using shift_type = Shift;
		/** @brief The order; `top` is the comp-largest element. */
		using compare_type = Compare;
		/**
		 * @brief An element's node id, returned by push / emplace (used by
		 *        promote / erase / value when `HasHandles`): stable while
		 *        the element is in any heap of the forest (meld and other
		 *        elements' pops never move it), invalid once it is popped or
		 *        erased (the id may then be reused by a later push).
		 */
		using handle = uint32;

		/**
		 * @brief `heap_count` empty heaps, ids `[0, heap_count)`. With a lazy
		 *        add, its identity is `Shift{}` and `Mapping` / `Compose` are
		 *        value-initialized (the defaults, or stateless lambdas).
		 * @pre `heap_count < 2^32` (asserted)
		 * @complexity O(heap_count)
		 */
		constexpr explicit meldable_heap_forest(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose>))
			: heap_directory(heap_count), comp_(std::move(comp)) {}

		/**
		 * @brief `heap_count` empty heaps with the lazy add given by its
		 *        identity and callables.
		 * @pre `heap_count < 2^32` (asserted)
		 * @complexity O(heap_count)
		 */
		constexpr meldable_heap_forest(std::size_t heap_count, Compare comp, tag_type shift_identity,
									   Mapping mapping = {}, Compose compose = {})
			requires has_shift
			: heap_directory(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  shift_identity_(std::move(shift_identity)) {}

		/** @brief Room for `node_count` live elements without reallocating the pool. */
		constexpr void reserve(std::size_t node_count) { nodes_.reserve(node_count); }

		/**
		 * @brief The comp-largest element of heap `heap` (shifted by every add made while it was in a heap).
		 * @pre `heap < heap_count()` and the heap is not empty (asserted)
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr T const &top(heap_id heap) const {
			atl_assert(record(heap).size != 0);
			return at(record(heap).root).value;
		}

		/**
		 * @brief Insert `T(args...)` into heap `heap`.
		 * @pre `heap < heap_count()`; fewer than `2^32 - 1` live elements in the forest (asserted)
		 * @complexity O(1) amortized (the pool may grow)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr handle emplace(heap_id heap, Args &&...args) {
			detail::heap_record &h = record(heap);
			uint32 const x = allocate(std::forward<Args>(args)...);
			h.root = meld_roots(h.root, x);
			++h.size;
			return x;
		}

		/**
		 * @brief Insert `value` into heap `heap`.
		 * @pre as @ref emplace
		 * @complexity O(1) amortized
		 */
		constexpr handle push(heap_id heap, T value) { return emplace(heap, std::move(value)); }

		/**
		 * @brief Remove the top of heap `heap` and return it.
		 * @pre `heap < heap_count()` and the heap is not empty (asserted)
		 * @complexity O(log n) amortized
		 */
		constexpr T pop(heap_id heap) {
			detail::heap_record &h = record(heap);
			atl_assert(h.size != 0);
			uint32 const x = h.root;
			push_down(x);
			T top_value = std::move(at(x).value);
			h.root = pair_up(at(x).child);
			--h.size;
			release_node(x);
			return top_value;
		}

		/**
		 * @brief Empty heap `heap`, returning every node to the free list without
		 *        comparisons or pairing. Invalidates its handles, as repeated pop does.
		 * @pre `heap < heap_count()` (asserted)
		 * @complexity O(k) for k elements, O(1) auxiliary space
		 */
		constexpr void release(heap_id heap) {
			detail::heap_record &h = record(heap);
			uint32 x = h.root;
			h = {};
			while (x != 0) {
				node &n = at(x);
				uint32 next = n.sibling;
				if (n.child != 0) {
					uint32 tail = n.child;
					while (at(tail).sibling != 0) tail = at(tail).sibling;
					at(tail).sibling = next;
					next = n.child;
				}
				release_node(x);
				x = next;
			}
		}

		/**
		 * @brief The element of the live handle `x`.
		 * @pre `x` is live: pushed, not yet popped or erased (asserted)
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr T const &value(handle x) const
			requires has_handles
		{
			atl_assert(is_live(x));
			return at(x).value;
		}

		/**
		 * @brief Replace the element of handle `x`, in heap `heap`, by `value`,
		 *        which must not be worse: `comp(value, old)` is false. The
		 *        element moves toward the top (with `std::less<>`, a max-heap,
		 *        this is an increase-key; with `std::greater<>` a decrease-key).
		 *        Only with `HasHandles` (@ref atl::addressable_meldable_heap_forest).
		 * @pre `x` is live (asserted) and in heap `heap` (documented: checking
		 *      would walk to the root); `value` is not worse (asserted)
		 * @complexity O(1) here, plus the later pops' amortized O(log n)
		 *      (whether the pairing heap's decrease-key is O(1) amortized is
		 *      open; its known bounds are O(log log n) to O(log n))
		 */
		constexpr void promote(heap_id heap, handle x, T value)
			requires has_handles
		{
			detail::heap_record &h = record(heap);
			atl_assert(is_live(x));
			atl_assert(!comp_(value, at(x).value));
			at(x).value = std::move(value);
			if (x == h.root) return;
			cut(x);
			h.root = link(h.root, x);
		}

		/**
		 * @brief Remove the element of handle `x` from heap `heap` and return
		 *        it. Only with `HasHandles` (@ref atl::addressable_meldable_heap_forest).
		 * @pre `x` is live (asserted) and in heap `heap` (documented)
		 * @complexity O(log n) amortized
		 */
		constexpr T erase(heap_id heap, handle x)
			requires has_handles
		{
			detail::heap_record &h = record(heap);
			atl_assert(is_live(x));
			atl_assert(h.size != 0);
			if (x == h.root) return pop(heap);
			cut(x);
			T erased = std::move(at(x).value);
			h.root = meld_roots(h.root, pair_up(at(x).child));
			--h.size;
			release_node(x);
			return erased;
		}

		/**
		 * @brief Move every element of heap `from` into heap `into`, leaving
		 *        `from` empty (still a valid id); a no-op when `into == from`.
		 * @pre both ids `< heap_count()` (asserted)
		 * @complexity O(1)
		 */
		constexpr void meld(heap_id into, heap_id from) {
			if (into == from) {
				(void)record(into);
				return;
			}
			detail::heap_record &a = record(into);
			detail::heap_record &b = record(from);
			a.root = meld_roots(a.root, b.root);
			a.size += b.size;
			b = {};
		}

		/**
		 * @brief `value = mapping(shift, value)` for every element now in heap
		 *        `heap` (later pushes are not shifted); a no-op on an empty heap.
		 * @pre `heap < heap_count()` (asserted)
		 * @complexity O(1)
		 */
		constexpr void add_to_all(heap_id heap, tag_type const &shift)
			requires has_shift
		{
			uint32 const x = record(heap).root;
			if (x != 0) apply(x, shift);
		}
	};

	/**
	 * @brief @ref atl::meldable_heap_forest with handles (no lazy add): `push`
	 *        returns a handle, and `promote(heap, handle, value)` (move an
	 *        element toward the top), `erase(heap, handle)` and
	 *        `value(handle)` act on it.
	 */
	template<typename T, typename Compare = std::less<>>
	using addressable_meldable_heap_forest =
		meldable_heap_forest<T, Compare, void, shift_by_addition, compose_by_addition, true>;
}
