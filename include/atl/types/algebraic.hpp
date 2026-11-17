#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace atl::detail {
	/** @brief Select the position form whenever it is callable, independently of its result type. */
	template<typename Fill>
	concept position_fill = std::invocable<Fill const &, std::size_t, std::size_t>;
}

namespace atl {
	/**
	 * @brief A binary operation `op(a, b)` on two `T`s whose result is convertible
	 *        to `R`.
	 *
	 * @tparam Op The operation type (e.g. std::plus<>, std::multiplies<>, a lambda).
	 * @tparam T  The operand type.
	 * @tparam R  The result type. Defaults to `T`, i.e. a *closed* operation
	 *            `T x T -> T` (the case reducing structures need).
	 *
	 * Checked on `Op const &`: the structures call `op` from const members, so a
	 * `mutable` lambda is rejected here rather than inside their bodies.
	 *
	 * @note The algebraic laws are preconditions on the caller, stated by each
	 *       structure: a monoid (`op` associative, with the supplied identity)
	 *       for segment_tree, a commutative monoid for binary_indexed_tree, an
	 *       idempotent commutative semigroup for sparse_table, and so on. A
	 *       concept checks only that the call is well-formed, not that the
	 *       operation obeys the law (as std::sort cannot verify that its
	 *       comparator is a strict weak ordering).
	 */
	template<typename Op, typename T, typename R = T>
	concept binary_operation =
		std::regular_invocable<Op const &, T const &, T const &> && std::convertible_to<std::invoke_result_t<Op const &, T const &, T const &>, R>;

	/**
	 * @brief A unary `inverse(a)` on a `T` whose result is convertible to `T`: the
	 *        inverse map of a group `(T, op, identity)`.
	 *
	 * @tparam Inverse The inverse type (e.g. std::negate<> for an additive group).
	 * @tparam T       The element type.
	 *
	 * @note Group structures such as weighted_dsu additionally require
	 *       `op(a, inverse(a)) == op(inverse(a), a) == identity` — a law that is a
	 *       precondition on the caller, not checkable by a concept.
	 */
	template<typename Inverse, typename T>
	concept inverse_operation =
		std::regular_invocable<Inverse const &, T const &> && std::convertible_to<std::invoke_result_t<Inverse const &, T const &>, T>;

	/**
	 * @brief An action `map(f, s)` of an `F` on an `S`: a callable taking a tag and
	 *        an element and yielding an element.
	 *
	 * @tparam Mapping The action type (e.g. a lambda `(F, S) -> S`).
	 * @tparam F       The tag (operator) type acting on the elements.
	 * @tparam S       The element type acted upon.
	 *
	 * @note Lazy structures additionally require the action to distribute over the
	 *       element operation (`map(f, op(a, b)) == op(map(f, a), map(f, b))`) and to
	 *       respect tag composition — algebraic laws that are preconditions on the
	 *       caller, not checkable by a concept.
	 */
	template<typename Mapping, typename F, typename S>
	concept action = std::regular_invocable<Mapping const &, F const &, S const &>
				  && std::convertible_to<std::invoke_result_t<Mapping const &, F const &, S const &>, S>;

	/**
	 * @brief A fallible in-place action `map(f, s)`: applies the tag @p f to the
	 *        element aggregate @p s and returns `true`, or returns `false` when
	 *        the aggregate alone cannot absorb the tag (the summary is then
	 *        unspecified, and the structure recomputes it from the children).
	 *
	 * The tag application of `segment_tree_beats`. Checked on
	 * `Mapping const &`, like @ref atl::action.
	 *
	 * @tparam Mapping The action type (e.g. a lambda `(F const &, S &) -> bool`).
	 * @tparam F       The tag type.
	 * @tparam S       The element aggregate type, modified in place.
	 *
	 * @note The laws (a single element never fails; a tag that succeeded on a
	 *       node succeeds on both of its children; distributivity over the
	 *       element operation where it succeeds) are preconditions on the
	 *       caller, not checkable by a concept.
	 */
	template<typename Mapping, typename F, typename S>
	concept fallible_action = std::invocable<Mapping const &, F const &, S &>
						   && std::same_as<std::invoke_result_t<Mapping const &, F const &, S &>, bool>;

	/**
	 * @brief A point update `f(x)`: invocable on a `T const &`, with a result
	 *        convertible to `T` — the `update(i, f)` callable of the point-update
	 *        trees, which replace an element by `f(element)` in one descent.
	 *
	 * @tparam Update The callable type, as deduced from a forwarding parameter
	 *                (invoked as an lvalue, so a stateful or `mutable` callable is
	 *                accepted: it runs exactly once per call).
	 * @tparam T      The element type.
	 */
	template<typename Update, typename T>
	concept update_function =
		std::invocable<Update &, T const &> && std::convertible_to<std::invoke_result_t<Update &, T const &>, T>;

	/**
	 * @brief The default-fold of an untouched span of positions: `fill(width)` or
	 *        the position-aware `fill(lo, hi)` for `[lo, hi)`, with a result
	 *        convertible to `S`.
	 *
	 * Checked on `Fill const &` (the structure calls it from const members), so a
	 * `mutable` lambda is rejected here rather than failing inside the body.
	 * The position-aware form wins whenever it is invocable; its result must
	 * convert to `S`, otherwise the callable is rejected. The width form is
	 * considered only when the position-aware form is not invocable.
	 *
	 * @tparam Fill The callable type.
	 * @tparam S    The element (aggregate) type.
	 */
	template<typename Fill, typename S>
	concept fill_function = (detail::position_fill<Fill>
							 && std::convertible_to<std::invoke_result_t<Fill const &, std::size_t, std::size_t>, S>)
						 || (!detail::position_fill<Fill> && std::regular_invocable<Fill const &, std::size_t>
							 && std::convertible_to<std::invoke_result_t<Fill const &, std::size_t>, S>);

	/**
	 * @brief Tag selecting the commutative-tag mode of `dual_segment_tree` and
	 *        `persistent_lazy_segment_tree`: as their trailing template argument,
	 *        or as the first constructor argument (@ref atl::commutative_tags), which
	 *        deduces it.
	 *
	 * The caller promises `compose(f, g) == compose(g, f)` for every pair of tags
	 * the structure will hold (range add, xor, chmin, chmax — not assignment or
	 * affine maps). The structures then never reorder tags: the dual tree skips
	 * its boundary pushes, the persistent tree keeps tags permanently on the nodes
	 * they were applied to. Unchecked, like every algebraic law.
	 */
	struct commutative_tags_t {
		explicit commutative_tags_t() = default;
	};

	/** @brief The @ref atl::commutative_tags_t tag value. */
	inline constexpr commutative_tags_t commutative_tags{};

	/**
	 * @brief A tag-order mode of the lazy-tag structures: `void` (tags may not
	 *        commute; the default) or @ref atl::commutative_tags_t.
	 */
	template<typename TagOrder>
	concept tag_order = std::is_void_v<TagOrder> || std::same_as<TagOrder, commutative_tags_t>;

	/**
	 * @brief Tag selecting the commutative-fold mode of `link_cut_tree`: as
	 *        its `FoldOrder` template argument, or as the first constructor
	 *        argument (@ref atl::commutative_fold), which deduces it.
	 *
	 * The caller promises `op(a, b) == op(b, a)` for the element fold (sum, min,
	 * max, xor; not composition of maps). A path reversal then leaves every fold
	 * unchanged, so the structure keeps one aggregate per node instead of one
	 * per direction: measured 10-20% faster on x86-64 (docs/link-cut-tree.md).
	 * Unchecked, like every algebraic law.
	 */
	struct commutative_fold_t {
		explicit commutative_fold_t() = default;
	};

	/** @brief The @ref atl::commutative_fold_t tag value. */
	inline constexpr commutative_fold_t commutative_fold{};

	/**
	 * @brief A fold-order mode: `void` (the fold may not commute; the default)
	 *        or @ref atl::commutative_fold_t.
	 */
	template<typename FoldOrder>
	concept fold_order = std::is_void_v<FoldOrder> || std::same_as<FoldOrder, commutative_fold_t>;

	/**
	 * @brief The vertex merge of a top tree, `add_vertex(light, value)`: a
	 *        vertex's own `Value` with the set of subtrees hanging at it (a
	 *        `Point`) as a one-vertex path cluster (a `Path`).
	 *
	 * @tparam AddVertex The callable type.
	 * @tparam Point     The hanging-set cluster type.
	 * @tparam Value     The vertex value type.
	 * @tparam Path      The path cluster type.
	 */
	template<typename AddVertex, typename Point, typename Value, typename Path>
	concept top_vertex_operation = std::regular_invocable<AddVertex const &, Point const &, Value const &>
								&& std::convertible_to<std::invoke_result_t<AddVertex const &, Point const &, Value const &>, Path>;

	/**
	 * @brief The edge merge of a top tree, `add_edge(child)`: a child's path
	 *        cluster (its top the child) seen from its parent as a hanging set.
	 *
	 * @tparam AddEdge The callable type.
	 * @tparam Path    The path cluster type.
	 * @tparam Point   The hanging-set cluster type.
	 */
	template<typename AddEdge, typename Path, typename Point>
	concept top_edge_operation =
		std::regular_invocable<AddEdge const &, Path const &> && std::convertible_to<std::invoke_result_t<AddEdge const &, Path const &>, Point>;

	/**
	 * @brief A two-tag action on a path cluster, `map_path(spine, light, p)`:
	 *        tag @p spine on the vertices of the cluster's spine, tag @p light
	 *        on every vertex hanging off it (the lazy tier of a top tree).
	 *
	 * @tparam MapPath The callable type.
	 * @tparam Tag     The tag type.
	 * @tparam Path    The path cluster type.
	 *
	 * @note The laws (distributivity over the top tree's merges) are
	 *       preconditions on the caller, as for @ref atl::action.
	 */
	template<typename MapPath, typename Tag, typename Path>
	concept path_action = std::regular_invocable<MapPath const &, Tag const &, Tag const &, Path const &>
					   && std::convertible_to<std::invoke_result_t<MapPath const &, Tag const &, Tag const &, Path const &>, Path>;
}
