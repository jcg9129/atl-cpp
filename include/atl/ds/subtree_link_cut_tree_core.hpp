#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/link_cut_tree_vertex.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl {
	namespace detail {
		/**
		 * @brief The whole @ref atl::subtree_link_cut_tree in any node layout: the
		 *        @ref atl::link_cut_tree machinery with the subtree-aggregate store
		 *        variant (the `Inverse` argument), re-exporting the structural
		 *        members and path folds, adding the subtree queries.
		 *
		 * Inherited privately: path updates and the edge tier do not exist
		 * here, and the constructors take the inverse.
		 */
		template<splay_layout Layout, std::copyable Value, binary_operation<Value> Op, inverse_operation<Value> Inverse>
			requires(std::is_void_v<Value> || std::default_initializable<Value>)
		class subtree_link_cut_tree_impl
			: link_cut_tree_impl<splay_engine, Layout, true, false, Value, Op, void, void, void, commutative_fold_t, void, Inverse> {
			using base = link_cut_tree_impl<splay_engine, Layout, true, false, Value, Op, void, void, void, commutative_fold_t, void,
											Inverse>;
			using base::access;
			using base::check_vertex;
			using base::node_of;
			using base::s_;
			using base::splay_rightmost;

			/**
			 * @brief After `access(x)`: whether node @p p is x's parent. Splays x
			 *        back to the top when the predecessor search moved it.
			 */
			constexpr bool has_parent_node(uint32 x, uint32 p) {
				uint32 const l = s_.child(x, 0);
				if (l == 0) return false;
				bool const is_parent = splay_rightmost(l) == p;
				splay_engine::splay(s_, x);
				return is_parent;
			}

			/** @brief Where the side of edge {v, p} holding v is read from. */
			struct edge_side {
				uint32 node;		///< v's node (the side is v's subtree), else p's (the side is all but p's subtree)
				bool is_complement; ///< p is v's child: the side is v's whole tree minus p's subtree, with p's node on top
			};

			/**
			 * @brief Locate the side of edge {v, p} holding v: after the call,
			 *        `side.node` tops its whole tree (accessed), so its own and its
			 *        total aggregates are current. @pre {v, p} is an edge (asserted)
			 */
			constexpr edge_side side_of_edge(uint32 v, uint32 p) {
				check_vertex(v);
				check_vertex(p);
				uint32 const xv = node_of(v), xp = node_of(p);
				access(xv);
				if (has_parent_node(xv, xp)) return {xv, false};
				access(xp);
				atl_assert(has_parent_node(xp, xv)); // {v, p} is an edge
				return {xp, true};
			}

		public:
			using base::none;
			using typename base::value_type;

			/**
			 * @brief An empty forest (no vertices).
			 * @complexity O(1)
			 */
			constexpr subtree_link_cut_tree_impl()
				requires std::default_initializable<Op> && std::default_initializable<Inverse> && std::default_initializable<Value>
				: base(0, Value{}, Op{}, Inverse{}) {}

			/**
			 * @brief @p n isolated vertices, each holding @p identity, folded with
			 *        @p op and inverted by @p inverse.
			 * @pre `n < 2^32 - 1` (asserted) @complexity O(n)
			 */
			constexpr subtree_link_cut_tree_impl(std::size_t n, Value identity, Op op = Op{}, Inverse inverse = Inverse{})
				: base(n, std::move(identity), std::move(op), std::move(inverse)) {}

			/**
			 * @brief One isolated vertex per element of @p values. Elements are
			 *        moved out only of an rvalue range that owns them.
			 * @complexity O(n)
			 */
			template<std::ranges::sized_range R>
				requires std::constructible_from<Value, forwarded_element_t<R>>
			constexpr subtree_link_cut_tree_impl(R &&values, std::type_identity_t<Value> identity, Op op = Op{},
												 Inverse inverse = Inverse{})
				: base(std::forward<R>(values), std::move(identity), std::move(op), std::move(inverse)) {}

			using base::ancestor;
			using base::assign_parents;
			using base::at;
			using base::connected;
			using base::cut;
			using base::cut_from_parent;
			using base::depth;
			using base::distance;
			using base::empty;
			using base::fold;
			using base::lca;
			using base::link;
			using base::make_root;
			using base::parent;
			using base::path_vertex;
			using base::root;
			using base::set;
			using base::size;
			using base::update;

			/**
			 * @brief The fold of @p v's subtree under the current root: v and
			 *        every vertex below it.
			 * @pre `v < size()` (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr value_type subtree_fold(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x); // x is now the deepest node of its preferred path: its subtree is all virtual
				return s_.own_subtree_aggregate(x);
			}

			/**
			 * @brief The fold of the side of edge {v, p} holding @p v: v's
			 *        subtree if p were the root. The root is unchanged.
			 * @pre {v, p} is an edge (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr value_type subtree_fold(uint32 v, uint32 p) {
				auto const [x, is_complement] = side_of_edge(v, p);
				if (!is_complement) return s_.own_subtree_aggregate(x);
				return s_.op(s_.total_aggregate(x), s_.inverse(s_.own_subtree_aggregate(x)));
			}

			/**
			 * @brief The number of vertices in @p v's subtree under the current root.
			 * @pre `v < size()` (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 subtree_size(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				return 1 + s_.virtual_size(x);
			}

			/**
			 * @brief The number of vertices on the side of edge {v, p} holding @p v.
			 *        The root is unchanged.
			 * @pre {v, p} is an edge (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 subtree_size(uint32 v, uint32 p) {
				auto const [x, is_complement] = side_of_edge(v, p);
				uint32 const own = 1 + s_.virtual_size(x);
				return is_complement ? s_.total_size(x) - own : own;
			}

			/**
			 * @brief The number of vertices in @p v's tree.
			 * @pre `v < size()` (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 component_size(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x); // x tops the root path's splay tree, whose total is the whole tree
				return s_.total_size(x);
			}
		};
	}

	/**
	 * @brief Link-cut tree with subtree aggregates: a forest under link, cut
	 *        and re-rooting, with the structural queries and path folds of
	 *        @ref atl::link_cut_tree plus folds and sizes of subtrees (under the
	 *        current root, or on one side of an edge) and component sizes, each
	 *        amortized O(log n). No path updates: subtree folds need the
	 *        inverse, which lazy tags would not maintain.
	 *
	 * Each node keeps, beside its path aggregate, the fold of its virtual
	 * children's whole subtrees (the trees hanging off it by path-parent
	 * pointers); every access moves one child between preferred and virtual,
	 * adding its total and subtracting the other's through @p Inverse.
	 *
	 * The root of each tree changes only by `make_root`, `link` (u's side joins
	 * below v's root) and `cut` (the detached side is rooted at its endpoint).
	 * Structural and aggregate queries splay; untagged `at` is const.
	 * Vertex ids are `uint32` in
	 * `[0, size())`; `none` answers "no vertex".
	 *
	 * @tparam Value   Default-initializable vertex value.
	 * @tparam Op      The fold: @pre (Value, Op, identity) is a COMMUTATIVE
	 *                 GROUP (associative, commutative, with @p Inverse).
	 * @tparam Inverse `inverse(a)`: `op(a, inverse(a)) == identity`.
	 */
	template<std::copyable Value, binary_operation<Value> Op, inverse_operation<Value> Inverse>
		requires(std::is_void_v<Value> || std::default_initializable<Value>)
	class subtree_link_cut_tree : public detail::subtree_link_cut_tree_impl<detail::splay_layout_interleaved, Value, Op, Inverse> {
		using base = detail::subtree_link_cut_tree_impl<detail::splay_layout_interleaved, Value, Op, Inverse>;

	public:
		using base::base;
	};

	/** @brief `Value` from the identity. */
	template<std::copyable Value, binary_operation<Value> Op, inverse_operation<Value> Inverse>
	subtree_link_cut_tree(std::size_t, Value, Op, Inverse) -> subtree_link_cut_tree<Value, Op, Inverse>;
	/** @brief `Value` from the identity; the elements must convert to it without narrowing. */
	template<std::ranges::sized_range R, std::copyable Value, binary_operation<Value> Op, inverse_operation<Value> Inverse>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	subtree_link_cut_tree(R &&, Value, Op, Inverse) -> subtree_link_cut_tree<Value, Op, Inverse>;
}
