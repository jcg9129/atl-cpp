#pragma once
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Dynamic (implicit) lazy segment tree: range apply and range fold
	 *        over a huge `[0, n)` — nodes materialized only where operations
	 *        land.
	 *
	 * The lazy sibling of @ref atl::dynamic_segment_tree, in the
	 * @ref atl::lazy_segment_tree algebra vocabulary (element monoid, tag monoid,
	 * `compose(f, g)` = "`g` first", distributive `mapping`). The one addition
	 * the huge domain forces is **`fill`**: the fold of a stretch of *default*
	 * elements, as either `fill(width)` (uniform defaults; law: `fill(a + b)
	 * == op(fill(a), fill(b))`) or the position-aware `fill(lo, hi)`
	 * (non-uniform defaults, e.g. a periodic base array answered by an RMQ;
	 * law: `fill(lo, hi) == op(fill(lo, m), fill(m, hi))`). An unmaterialized
	 * subtree folds to its fill and materializes lazily with that aggregate
	 * when a tag must settle onto it — so range applies over vast untouched
	 * interior are exact without ever expanding it (carry widths inside `Value`
	 * when tags need them: `fill(w) = {0, w}` for sums under affine maps).
	 *
	 * Reads (`fold`, `at`) accumulate pending tags and never
	 * allocate; `apply` materializes O(log n) nodes per call. Give `Tag`
	 * `operator==` when possible — identity tags are then skipped when
	 * settling, saving both `mapping` calls and child materializations.
	 *
	 * @tparam Value   The element type.
	 * @tparam Tag     The tag type applied to ranges.
	 * @tparam Op      Element fold: any `op(Value, Value)` convertible to `Value`.
	 * @tparam Mapping Tag application: any `mapping(Tag, Value)` convertible to `Value`.
	 * @tparam Compose Tag composition: any `compose(Tag, Tag)` convertible to `Tag`.
	 * @tparam Fill    Default-fold: `fill(width)` or `fill(lo, hi)`, convertible to `Value`,
	 *                 enforced by @ref atl::fill_function.
	 */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose, fill_function<Value> Fill>
	class dynamic_lazy_segment_tree {
		/**
		 * @brief `aggregate` already includes this node's own tag; `tag` is pending for the
		 * subtree below — including children not yet materialized.
		 */
		struct node {
			Value aggregate;
			Tag tag;
			uint32 left = 0, right = 0;
		};

		[[no_unique_address]] Op op_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] Fill fill_;
		Value element_identity_;
		Tag tag_identity_;
		std::size_t n_ = 0;
		std::vector<node> nodes_;
		uint32 root_ = 0;

		/**
		 * @brief The untouched aggregate of `[lo, hi)`: the position-aware fill form
		 * when the callable provides it, the width-only form otherwise.
		 */
		constexpr Value filled(std::size_t lo, std::size_t hi) const {
			if constexpr (detail::position_fill<Fill>)
				return std::invoke(std::as_const(fill_), lo, hi);
			else
				return std::invoke(std::as_const(fill_), hi - lo);
		}

		constexpr uint32 make(Value aggregate) {
			atl_assert(nodes_.size() < 0xFFFFFFFFull);
			nodes_.push_back({std::move(aggregate), tag_identity_, 0, 0});
			return uint32(nodes_.size() - 1);
		}

		constexpr bool is_identity(Tag const &g) const {
			if constexpr (std::equality_comparable<Tag>)
				return g == tag_identity_;
			else
				return false;
		}

		/**
		 * @brief Settle @p x's pending tag one level. An identity tag is a no-op with
		 * **no materialization**; a live one must land on both children, so
		 * absent ones materialize with their untouched aggregate (`fill`).
		 */
		constexpr void push(uint32 x, std::size_t lo, std::size_t mid, std::size_t hi) {
			if (is_identity(nodes_[x].tag)) return;
			if (nodes_[x].left == 0) {
				uint32 const c = make(filled(lo, mid));
				nodes_[x].left = c;
			}
			if (nodes_[x].right == 0) {
				uint32 const c = make(filled(mid, hi));
				nodes_[x].right = c;
			}
			node &nd = nodes_[x];
			for (uint32 const c : {nd.left, nd.right}) {
				node &ch = nodes_[c];
				ch.aggregate = std::invoke(std::as_const(mapping_), nd.tag, ch.aggregate);
				ch.tag = std::invoke(std::as_const(compose_), nd.tag, ch.tag);
			}
			nd.tag = tag_identity_;
		}

		/** @brief The child of @p x on the descent side, materialized if absent. */
		constexpr uint32 touch(uint32 x, bool goes_left, std::size_t clo, std::size_t chi) {
			uint32 c = goes_left ? nodes_[x].left : nodes_[x].right;
			if (c == 0) {
				c = make(filled(clo, chi));
				(goes_left ? nodes_[x].left : nodes_[x].right) = c;
			}
			return c;
		}

		/** @brief Aggregate of @p x's child covering `[clo, chi)`, null-aware. */
		constexpr Value child_aggregate(uint32 c, std::size_t clo, std::size_t chi) const {
			return c == 0 ? filled(clo, chi) : nodes_[c].aggregate;
		}

		constexpr void apply_rec(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &f) {
			if (l <= lo && hi <= r) {
				node &nd = nodes_[x];
				nd.aggregate = std::invoke(std::as_const(mapping_), f, nd.aggregate);
				nd.tag = std::invoke(std::as_const(compose_), f, nd.tag);
				return;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			push(x, lo, mid, hi);
			if (l < mid) apply_rec(touch(x, true, lo, mid), lo, mid, l, r, f);
			if (r > mid) apply_rec(touch(x, false, mid, hi), mid, hi, l, r, f);
			node &nd = nodes_[x];
			nd.aggregate = std::invoke(std::as_const(op_), child_aggregate(nd.left, lo, mid), child_aggregate(nd.right, mid, hi));
		}

		/**
		 * @brief Fold of the overlap with `[l, r)` under @p acc, the composition of
		 * pending tags above @p x. Pure — untouched subtrees answer via fill.
		 */
		[[nodiscard]] constexpr Value fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &acc) const {
			if (r <= lo || hi <= l) return element_identity_;
			std::size_t const olo = lo < l ? l : lo, ohi = hi < r ? hi : r;
			if (x == 0) return std::invoke(std::as_const(mapping_), acc, filled(olo, ohi));
			node const &nd = nodes_[x];
			if (l <= lo && hi <= r) return std::invoke(std::as_const(mapping_), acc, nd.aggregate);
			Tag const acc2 = std::invoke(std::as_const(compose_), acc, nd.tag);
			std::size_t const mid = lo + (hi - lo) / 2;
			return std::invoke(std::as_const(op_), fold(nd.left, lo, mid, l, r, acc2), fold(nd.right, mid, hi, l, r, acc2));
		}

		/**
		 * @brief Aggregate of `[lo, hi)` under ancestor tags @p acc — the node's own
		 * aggregate (or its fill when unmaterialized), tag applied.
		 */
		constexpr Value aggregate_under(uint32 x, std::size_t lo, std::size_t hi, Tag const &acc) const {
			return std::invoke(std::as_const(mapping_), acc, x == 0 ? filled(lo, hi) : nodes_[x].aggregate);
		}

		/**
		 * @brief max_right recursion: within `[lo, hi)` (whose left edge is at or past
		 * `l`), extend the running prefix @p sm rightward while @p pred holds.
		 * Returns the first index that breaks it, or `hi` if the whole node
		 * fits (then @p sm folds it in). @p acc carries ancestor tags.
		 */
		template<std::predicate<Value const &> Pred>
		constexpr std::size_t max_right_descend(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, Tag const &acc, Value &sm,
												Pred &pred) const {
			if (l <= lo) {
				Value whole = std::invoke(std::as_const(op_), sm, aggregate_under(x, lo, hi, acc));
				if (std::invoke(pred, whole)) {
					sm = std::move(whole);
					return hi;
				}
				if (hi - lo == 1) return lo;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			// An if, not `x == 0 ? acc : compose(...)`: MSVC's constant evaluation loses the
			// members of a class-type conditional-operator result.
			Tag const acc2 = [&]() -> Tag {
				if (x == 0) return acc;
				return std::invoke(std::as_const(compose_), acc, nodes_[x].tag);
			}();
			uint32 const lc = x == 0 ? 0 : nodes_[x].left, rc = x == 0 ? 0 : nodes_[x].right;
			if (l < mid) {
				std::size_t const b = max_right_descend(lc, lo, mid, l, acc2, sm, pred);
				if (b != mid) return b;
			}
			return max_right_descend(rc, mid, hi, l, acc2, sm, pred);
		}

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = Value;
		/** @brief Range action type. */
		using tag_type = Tag;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/**
		 * @brief An empty tree over `[0, 0)` with value-initialized identities — a
		 * placeholder to assign a real tree to later. @complexity O(1)
		 */
		constexpr dynamic_lazy_segment_tree()
			requires std::default_initializable<Value> && std::default_initializable<Tag> && std::default_initializable<Op>
				  && std::default_initializable<Mapping> && std::default_initializable<Compose>
				  && std::default_initializable<Fill>
			: dynamic_lazy_segment_tree(0, Value{}, Tag{}, Op{}, Mapping{}, Compose{}, Fill{}) {}

		/**
		 * @brief `[0, n)` of default elements: untouched `[l, r)` folds to `fill(l, r)` when the position form is
		 * invocable, otherwise `fill(r - l)`. An element uses `[i, i + 1)`.
		 * @complexity O(1)
		 */
		constexpr dynamic_lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{},
											Compose compose = Compose{}, Fill fill = Fill{})
			: op_(std::move(op)), mapping_(std::move(mapping)), compose_(std::move(compose)),
			  fill_(std::move(fill)), element_identity_(std::move(identity)), tag_identity_(std::move(tag_identity)), n_(n) {
			nodes_.reserve(2);
			nodes_.push_back({element_identity_, tag_identity_, 0, 0}); // null sentinel
			if (n > 0) root_ = make(filled(0, n));
		}

		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }

		/** @brief Node-pool entries, including the null sentinel. @complexity O(1) */
		constexpr std::size_t node_count() const noexcept { return nodes_.size(); }

		/**
		 * @brief Apply tag @p f to every element of `[l, r)`.
		 * @pre `l <= r <= size()` @complexity O(log n) time and new nodes
		 */
		constexpr void apply(std::size_t l, std::size_t r, Tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			apply_rec(root_, 0, n_, l, r, f);
		}

		/** @brief Assign @p x to element @p i. @pre `i < size()` @complexity O(log n) */
		constexpr void set(std::size_t i, Value x) {
			atl_assert(i < n_);
			nodes_[settle_to_leaf(i)].aggregate = std::move(x);
			pull_path(root_, 0, n_, i);
		}

		/**
		 * @brief Apply tag @p f to the single element at @p i. @pre `i < size()`
		 * @complexity O(log n) time and new nodes
		 */
		constexpr void apply(std::size_t i, Tag const &f) {
			atl_assert(i < n_);
			uint32 const leaf = settle_to_leaf(i);
			nodes_[leaf].aggregate = std::invoke(std::as_const(mapping_), f, nodes_[leaf].aggregate);
			pull_path(root_, 0, n_, i);
		}

		/**
		 * @brief Fold of `[l, r)`, or `identity` if empty — const, allocation-free.
		 * @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= n_);
			if (l == r) return element_identity_;
			return fold(root_, 0, n_, l, r, tag_identity_);
		}

		/** @brief Fold of the whole range. @complexity O(1) */
		[[nodiscard]] constexpr Value const &fold() const { return nodes_[root_].aggregate; }

		/** @brief The value at @p i. @pre `i < size()` @complexity O(log n) */
		[[nodiscard]] constexpr Value at(std::size_t i) const {
			atl_assert(i < n_);
			// one root-to-leaf walk composing the path tags and one mapping at the end:
			// no op calls against the identity, unlike fold(i, i + 1)
			Tag acc = tag_identity_;
			uint32 x = root_;
			std::size_t lo = 0, hi = n_;
			while (x != 0 && hi - lo > 1) {
				node const &nd = nodes_[x];
				acc = std::invoke(std::as_const(compose_), acc, nd.tag);
				std::size_t const mid = lo + (hi - lo) / 2;
				if (i < mid)
					x = nd.left, hi = mid;
				else
					x = nd.right, lo = mid;
			}
			return std::invoke(std::as_const(mapping_), acc, x == 0 ? filled(i, i + 1) : nodes_[x].aggregate);
		}

		/**
		 * @brief Largest `r` in `[l, size()]` such that `pred(fold(l, r))` holds.
		 * Requires @p pred monotone (once false, stays false as `r` grows) and
		 * `pred(identity)` true. Const and allocation-free — reads accumulate
		 * pending tags and answer untouched subtrees via `fill`, materializing
		 * nothing. @pre `l <= size()` @complexity O(log n)
		 */
		template<std::predicate<Value const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) const {
			atl_assert(l <= n_ && std::invoke(pred, element_identity_));
			if (l == n_) return n_;
			Value sm = element_identity_;
			return max_right_descend(root_, 0, n_, l, tag_identity_, sm, pred);
		}

		/**
		 * @brief Reserve node capacity for @p ops further mutations (an apply
		 * materializes at most four nodes per level — a generous worst case;
		 * see `reserve_nodes` to plan from a measured count instead).
		 * @pre The capacity estimate fits `size_t`.
		 */
		constexpr void reserve(std::size_t ops) {
			std::size_t const per_operation = 4 * (std::bit_width(std::max<std::size_t>(n_, 2) - 1) + 1);
			atl_assert(ops <= ((std::numeric_limits<std::size_t>::max)() - nodes_.size()) / per_operation);
			nodes_.reserve(nodes_.size() + ops * per_operation);
		}

		/**
		 * @brief Reserve capacity for @p count nodes exactly — for memory-tight
		 * environments where the worst-case `reserve` estimate would
		 * overshoot and vector doubling would spike the peak (pair with
		 * `node_count` measurements on representative workloads).
		 */
		constexpr void reserve_nodes(std::size_t count) { nodes_.reserve(count); }

	private:
		/**
		 * @brief Settle every tag on the root-to-leaf path of @p i, materializing
		 * the path, so the leaf's aggregate is absolute; returns the leaf's id.
		 * Ids on the path stay valid (nodes are only appended).
		 */
		constexpr uint32 settle_to_leaf(std::size_t i) {
			uint32 at = root_;
			std::size_t lo = 0, hi = n_;
			while (hi - lo > 1) {
				std::size_t const mid = lo + (hi - lo) / 2;
				push(at, lo, mid, hi);
				bool const goes_left = i < mid;
				at = touch(at, goes_left, goes_left ? lo : mid, goes_left ? mid : hi);
				(goes_left ? hi : lo) = mid;
			}
			return at;
		}

		constexpr void pull_path(uint32 x, std::size_t lo, std::size_t hi, std::size_t i) {
			if (hi - lo == 1) return;
			std::size_t const mid = lo + (hi - lo) / 2;
			node &nd = nodes_[x];
			pull_path(i < mid ? nd.left : nd.right, i < mid ? lo : mid, i < mid ? mid : hi, i);
			nd.aggregate = std::invoke(std::as_const(op_), child_aggregate(nd.left, lo, mid), child_aggregate(nd.right, mid, hi));
		}
	};

	/** @brief Deduce stored element and callable types from the constructor arguments. */
	template<typename Value, typename Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose,
			 fill_function<Value> Fill>
	dynamic_lazy_segment_tree(std::size_t, Value, Tag, Op, Mapping, Compose, Fill)
		-> dynamic_lazy_segment_tree<Value, Tag, Op, Mapping, Compose, Fill>;
}
