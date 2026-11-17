#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/dynamic_forest.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief The template-argument contract of @ref atl::euler_tour_tree : tiers
		 *        switched on by non-`void` arguments (values with their fold, then
		 *        lazy tags), each tier's algebra well-formed.
		 */
		template<typename Value, typename Op, typename Tag, typename Mapping, typename Compose>
		concept euler_tour_algebra = (std::is_void_v<Value> == std::is_void_v<Op>)
								  && (std::is_void_v<Value> || (std::copyable<Value> && binary_operation<Op, Value>))
								  && (std::is_void_v<Tag> == std::is_void_v<Mapping>)
								  && (std::is_void_v<Tag> == std::is_void_v<Compose>)
								  && (std::is_void_v<Tag>
									  || (!std::is_void_v<Value> && std::copyable<Tag> && action<Mapping, Tag, Value>
										  && binary_operation<Compose, Tag>));

		/** @brief An absent node field or algebra slot (distinct per @p Id, so absent fields overlap). */
		template<int Id>
		struct euler_tour_no_field {};

		/**
		 * @brief The node pool of an @ref atl::euler_tour_tree : an in-place
		 *        `treap_parent_store`, one interleaved record per tour node
		 *        (child links, size, priority, and on the value tiers the value,
		 *        its subtree fold and the pending tag), the parent links in an
		 *        array of their own so root walks stream 4 bytes a node.
		 *
		 * The rebinds write the parent of every child they install (the null
		 * node's parent absorbs the writes for empty sides, branch-free) and
		 * pull size and aggregate. A tag at x is already reflected in x's own
		 * value and aggregate and owed to its children (the @ref atl::treap_sequence
		 * scheme), so every root aggregate is current. Node 0 is the null
		 * sentinel: size 0 and the identity aggregate. There is no reversal:
		 * tour order only ever rotates.
		 */
		template<typename Value, typename Op, typename Tag, typename Mapping, typename Compose>
			requires euler_tour_algebra<Value, Op, Tag, Mapping, Compose> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
		struct euler_tour_store {
			static constexpr bool has_value = !std::is_void_v<Value>;
			static constexpr bool has_tags = !std::is_void_v<Tag>;

			using value_field = std::conditional_t<has_value, Value, euler_tour_no_field<0>>;
			using tag_field = std::conditional_t<has_tags, Tag, euler_tour_no_field<1>>;

			template<typename Algebra, int Id>
			using slot = std::conditional_t<std::is_void_v<Algebra>, euler_tour_no_field<Id>, Algebra>;

			/** @brief One tour node (a vertex or an arc); its parent link lives in `parents`. */
			struct node {
				uint32 child[2]{};
				uint32 size = 0;
				uint32 priority = 0;
				[[no_unique_address]] value_field value{};
				[[no_unique_address]] value_field aggregate{};
				[[no_unique_address]] tag_field tag{};
			};

			std::vector<node> nodes;
			std::vector<uint32> parents;
			[[no_unique_address]] slot<Op, 2> op;
			[[no_unique_address]] slot<Mapping, 3> mapping;
			[[no_unique_address]] slot<Compose, 4> compose;
			[[no_unique_address]] tag_field tag_identity{};

			constexpr euler_tour_store() = default;

			constexpr euler_tour_store(std::vector<node> values, std::vector<uint32> links, slot<Op, 2> fold, slot<Mapping, 3> map, slot<Compose, 4> composition, tag_field tag)
				: nodes(std::move(values)), parents(std::move(links)), op(detail::move_or_copy(fold)),
				  mapping(detail::move_or_copy(map)), compose(detail::move_or_copy(composition)), tag_identity(std::move(tag)) {}

			/** @brief t's left child (0 for none). */
			[[nodiscard]] constexpr uint32 left(uint32 t) const { return nodes[t].child[0]; }

			/** @brief t's right child (0 for none). */
			[[nodiscard]] constexpr uint32 right(uint32 t) const { return nodes[t].child[1]; }

			/** @brief The node count of t's subtree (0 for the sentinel). */
			[[nodiscard]] constexpr uint32 size(uint32 t) const { return nodes[t].size; }

			/** @brief t's heap priority. */
			[[nodiscard]] constexpr uint32 priority(uint32 t) const { return nodes[t].priority; }

			/** @brief t's parent (0 for a root). */
			[[nodiscard]] constexpr uint32 parent(uint32 t) const { return parents[t]; }

			/** @brief Set t's parent link (t = 0 writes the sentinel's, never read). */
			constexpr void set_parent(uint32 t, uint32 p) { parents[t] = p; }

			/**
			 * @brief Reset every node: @p count detached nodes (size 1) after the
			 *        sentinel (size 0), values and aggregates @p identity, tags
			 *        @p tag_identity_value, priorities drawn from @p seed.
			 */
			constexpr void assign(std::size_t count, value_field const &identity, tag_field const &tag_identity_value,
								  uint64 seed) {
				tag_identity = tag_identity_value;
				node fresh{};
				fresh.size = 1;
				if constexpr (has_value) fresh.value = fresh.aggregate = identity;
				if constexpr (has_tags) fresh.tag = tag_identity;
				nodes.assign(count + 1, fresh);
				nodes[0].size = 0;
				parents.assign(count + 1, 0);
				for (std::size_t x = 1; x <= count; ++x)
					nodes[x].priority = uint32(splitmix64(seed + x * 0x9e3779b97f4a7c15ULL) >> 32);
			}

			/** @brief Size and aggregate from the children, branchless through null sides. */
			constexpr void pull(uint32 t) {
				node &n = nodes[t];
				node const &l = nodes[n.child[0]], &r = nodes[n.child[1]];
				n.size = l.size + 1 + r.size;
				if constexpr (has_value) n.aggregate = op(op(l.aggregate, n.value), r.aggregate);
			}

			/** @brief Install both children of t (and t as their parent), pull t; in place, so t is returned. */
			constexpr uint32 rebind(uint32 t, uint32 l, uint32 r) {
				nodes[t].child[0] = l;
				nodes[t].child[1] = r;
				set_parent(l, t);
				set_parent(r, t);
				pull(t);
				return t;
			}

			/** @brief The left-only form of `rebind`. */
			constexpr uint32 rebind_left(uint32 t, uint32 l) {
				nodes[t].child[0] = l;
				set_parent(l, t);
				pull(t);
				return t;
			}

			/** @brief The right-only form of `rebind`. */
			constexpr uint32 rebind_right(uint32 t, uint32 r) {
				nodes[t].child[1] = r;
				set_parent(r, t);
				pull(t);
				return t;
			}

			/**
			 * @brief Fold tag @p f into node @p t: value and aggregate now, the tag
			 *        composed for the subtree below. @pre `t != 0`
			 */
			constexpr void apply_tag(uint32 t, tag_field const &f)
				requires has_tags
			{
				node &n = nodes[t];
				n.value = mapping(f, n.value);
				n.aggregate = mapping(f, n.aggregate);
				n.tag = compose(f, n.tag);
			}

			/** @brief Settle t's pending tag one level down (a no-op below the lazy tier). */
			constexpr uint32 push(uint32 t) {
				if constexpr (has_tags) {
					Tag &pending = nodes[t].tag;
					// an identity tag has nothing to settle: skip the child mappings
					if constexpr (std::equality_comparable<Tag>)
						if (pending == tag_identity) return t;
					if (uint32 const l = left(t); l != 0) apply_tag(l, pending);
					if (uint32 const r = right(t); r != 0) apply_tag(r, pending);
					pending = tag_identity;
				}
				return t;
			}
		};
	}

	/**
	 * @brief Euler tour tree: an unrooted forest under link and cut, with
	 *        connectivity, component sizes, and folds and lazy updates over a
	 *        commutative monoid on whole components and on the subtree hanging
	 *        off either side of an edge, each expected O(log n).
	 *
	 * Each tree is kept as its Euler tour in one treap with parent links: one
	 * node per vertex and one per arc (two per edge), the tour read
	 * cyclically, priorities drawn from a per-process seed. No path queries
	 * (that is @ref atl::link_cut_tree); in exchange the subtree of v seen from its
	 * neighbour p is one contiguous tour segment, so subtree updates cost what
	 * folds cost.
	 *
	 * What the parent links buy:
	 *  - queries that only read (`connected`, `component_size`,
	 *    `component_fold`, and `subtree_fold` / `at` below the lazy tier) walk
	 *    up to the root without writing a node;
	 *  - splitting at a node is bottom-up (`detail::treap_base::split_at_node`),
	 *    after settling the pending tags on its root path (lazy tier only);
	 *  - the subtree segment between the arcs p -> v and v -> p is folded or
	 *    tagged in place from the two root paths and their lowest common
	 *    ancestor: no split, no merge;
	 *  - `link` rotates u's tour (split at u, rejoin) and inserts it with the
	 *    two new arcs just before v (`detail::treap_base::insert_before_node`):
	 *    only the subtree the block's root outranks is split and merged.
	 * Joins are `detail::treap_base::merge`, recursive to the treap depth
	 * (expected O(log n), so a long path is no deeper than a random tree).
	 * This store uses fewer writes than the splay-engine contender (
	 * `atl::experimental::euler_tour_tree_by_splay`) on churn, links and vertex
	 * add + subtree sum in the ds-euler_tour_tree race, subtree add and
	 * assign_parents within noise (2026-10-25, Xeon Platinum 8168;
	 * docs/euler-tour-tree.md).
	 *
	 * Tiers are switched on by non-`void` template arguments, in the order of
	 * @ref atl::link_cut_tree : structure only; `Value` + `Op` (vertex values,
	 * component and subtree folds); + `Tag`, `Mapping`, `Compose` (lazy
	 * component and subtree updates). Queries are non-`const` (they reuse
	 * member scratch and, on the lazy tier, settle tags). Vertex ids are
	 * `uint32` in `[0, size())`.
	 *
	 * Arc nodes hold the identity and fold into every aggregate, so:
	 *  - `op` must commute (the tour's order is not the tree's);
	 *  - `mapping(f, identity) == identity` for every tag (else arcs pick up
	 *    values); a tag whose effect depends on how many vertices it covers
	 *    (add to a sum, assign under a sum) carries that count inside `Value`,
	 *    the width convention: `{sum, count}` with identity `{0, 0}`, vertices
	 *    `{a, 1}`, `mapping(add c, {s, k}) = {s + c k, k}`.
	 *
	 * @tparam Value   Default-initializable when non-void. Vertex value; `void` for structure only.
	 * @tparam Op      The fold, associative and commutative; `op(identity, a) == a`.
	 * @tparam Tag     Default-initializable when non-void. Update tag (needs `Value`).
	 * @tparam Mapping `mapping(f, a)`, distributing over `Op`, fixing the identity.
	 * @tparam Compose `compose(f, g)` = "g first, then f".
	 */
	template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void>
		requires detail::euler_tour_algebra<Value, Op, Tag, Mapping, Compose> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
	class euler_tour_tree {
		using store_type = detail::euler_tour_store<Value, Op, Tag, Mapping, Compose>;
		using base = detail::treap_base;
		using place = detail::treap_split_place;

		static constexpr bool has_value = store_type::has_value;
		static constexpr bool has_tags = store_type::has_tags;

	public:
		/** @brief The value type (a placeholder on the structure-only tier). */
		using value_type = typename store_type::value_field;
		/** @brief The tag type (a placeholder below the lazy tier). */
		using tag_type = typename store_type::tag_field;

	private:
		using op_slot = typename store_type::template slot<Op, 2>;
		using mapping_slot = typename store_type::template slot<Mapping, 3>;
		using compose_slot = typename store_type::template slot<Compose, 4>;

		static constexpr uint32 none = graph::none;

		store_type s_;
		uint32 n_ = 0;
		/** @brief (min, max) endpoint pair -> edge id e; edge e owns arc nodes `arc_base(e)` (min -> max) and the next one */
		flat_hash_map<uint64, uint32, detail::forest_edge_hash> edges_;
		std::vector<uint32> free_ids_;
		/** @brief Root-path scratch (node first, root last), kept to reuse its capacity. */
		std::vector<uint32> path_, other_path_;

		[[nodiscard]] static constexpr uint32 node_of(uint32 v) { return v + 1; }

		[[nodiscard]] constexpr uint32 arc_base(uint32 e) const { return n_ + 1 + 2 * e; }

		[[nodiscard]] static constexpr uint64 edge_key(uint32 u, uint32 v) { return detail::forest_edge_key(u, v); }

		/** @brief The arc node u -> v of edge @p e (whose endpoints are u and v). */
		[[nodiscard]] constexpr uint32 arc(uint32 e, uint32 u, uint32 v) const { return arc_base(e) + (u < v ? 0 : 1); }

		/** @brief The id of edge u - v, asserted to exist. */
		[[nodiscard]] constexpr uint32 edge_id(uint32 u, uint32 v) const {
			uint32 const e = edges_.at_or(edge_key(u, v), none);
			atl_assert(e != none); // u - v is an edge
			return e;
		}

		[[nodiscard]] constexpr uint32 root_of(uint32 x) const { return base::root_of(s_, x); }

		/** @brief Fill @p path with x and its ancestors, the root last. */
		constexpr void collect(uint32 x, std::vector<uint32> &path) const {
			path.clear();
			for (; x != 0; x = s_.parent(x)) path.push_back(x);
		}

		/** @brief Push the tags of `path[0, count)` top-down (an ancestor before its descendants). */
		constexpr void settle(std::vector<uint32> const &path, std::size_t count) {
			for (std::size_t k = count; k-- > 0;) s_.push(path[k]);
		}

		/** @brief Settle every tag on x's root path, x included (lazy tier only). */
		constexpr void settle_above(uint32 x) {
			if constexpr (has_tags) {
				collect(x, path_);
				settle(path_, path_.size());
			}
		}

		/** @brief Pull x and every ancestor, bottom-up (after a value change at x). */
		constexpr void pull_up(uint32 x) {
			for (; x != 0; x = s_.parent(x)) s_.pull(x);
		}

		/** @brief Split x's tree at x with x placed per @p where; both parts are roots. */
		constexpr std::pair<uint32, uint32> split_at(uint32 x, place where) {
			settle_above(x);
			return base::split_at_node(s_, x, where);
		}

		/** @brief Merge the roots @p a and @p b (a's tour first); the result is a root. */
		constexpr uint32 join(uint32 a, uint32 b) { return base::merge(s_, a, b); }

		/**
		 * @brief The tour segment strictly between arc @p in and arc @p out, read
		 *        cyclically from @p in: v's side of the edge whose arcs they are.
		 *        Collects both root paths, settles their tags (lazy tier), then
		 *        hands @p side_visit each whole subtree of the segment (possibly 0)
		 *        and @p node_visit each single node; returns how many nodes of
		 *        each path lie strictly below their lowest common ancestor (for
		 *        the caller's bottom-up pulls).
		 *
		 * With in before out, the segment is in's later side up to the lca and
		 * out's earlier side up to it, plus the lca when it is neither arc. With
		 * out before in, it wraps: everything before out and everything after
		 * in, each path walked to the root (an ancestor above the lca is on the
		 * same side of both, so exactly one walk takes it).
		 */
		template<typename SideVisit, typename NodeVisit>
			requires std::invocable<SideVisit &, uint32> && std::invocable<NodeVisit &, uint32>
		constexpr std::pair<std::size_t, std::size_t> visit_segment(uint32 in, uint32 out, SideVisit &&side_visit,
																	NodeVisit &&node_visit) {
			collect(in, path_);
			collect(out, other_path_);
			std::vector<uint32> const &pa = path_, &pb = other_path_;
			std::size_t i = pa.size() - 1, j = pb.size() - 1;
			while (i > 0 && j > 0 && pa[i - 1] == pb[j - 1]) {
				--i;
				--j;
			}
			uint32 const lca = pa[i];
			bool is_in_first; // in and out are distinct, so one lies strictly below the lca
			if (i == 0)
				is_in_first = s_.right(in) == pb[j - 1];
			else if (j == 0)
				is_in_first = s_.left(out) == pa[i - 1];
			else
				is_in_first = s_.left(lca) == pa[i - 1];
			if constexpr (has_tags) {
				settle(pa, pa.size());
				settle(pb, j);
			}
			// the later side of in (dir 1) or the earlier side of out (dir 0), walked through path[0, stop)
			auto const walk = [&](std::vector<uint32> const &path, std::size_t stop, bool dir) {
				side_visit(s_.nodes[path[0]].child[dir]);
				for (std::size_t k = 1; k < stop; ++k) {
					uint32 const z = path[k];
					if (s_.nodes[z].child[!dir] == path[k - 1]) {
						node_visit(z);
						side_visit(s_.nodes[z].child[dir]);
					}
				}
			};
			if (is_in_first) {
				if (i > 0) walk(pa, i, true);
				if (j > 0) walk(pb, j, false);
				if (i > 0 && j > 0) node_visit(lca);
			} else {
				walk(pb, pb.size(), false);
				walk(pa, pa.size(), true);
			}
			return {i, j};
		}

		/** @brief @p n isolated vertices and 2 (n - 1) free arc nodes, all identities. */
		constexpr void init(std::size_t n, value_type const &identity, tag_type const &tag_identity) {
			// ids are 32-bit with `none` reserved: n vertex nodes and 2 (n - 1) arc nodes
			atl_assert(n <= std::size_t(graph::none) / 3);
			n_ = uint32(n);
			std::size_t const edge_slots = n > 0 ? n - 1 : 0;
			uint64 seed = 0x2545f4914f6cdd1dULL; // constant evaluation has no seed to read and no adversary
			if !consteval {
				seed ^= detail::hash_seed();
			}
			s_.assign(n + 2 * edge_slots, identity, tag_identity, seed);
			edges_.clear();
			edges_.reserve(edge_slots);
			free_ids_.resize(edge_slots);
			// popped from the back: the lowest id first
			for (std::size_t i = 0; i < edge_slots; ++i) free_ids_[i] = uint32(edge_slots - 1 - i);
		}

		/** @brief One isolated vertex per element of @p values. */
		template<std::ranges::sized_range R>
			requires std::constructible_from<Value, detail::forwarded_element_t<R>>
		constexpr void init_values(R &&values, value_type const &identity, tag_type const &tag_identity) {
			init(std::ranges::size(values), identity, tag_identity);
			uint32 x = 1;
			for (auto &&v : values) {
				auto &n = s_.nodes[x++];
				n.value = static_cast<Value>(detail::forward_element<R>(v));
				n.aggregate = n.value;
			}
		}

		/**
		 * @brief Build one treap over the detached, tag-free nodes @p ids in
		 *        order, in O(size): the Cartesian tree of their priorities by the
		 *        rightmost-spine stack; each node is pulled once its subtree is
		 *        final (when it leaves the spine).
		 */
		constexpr void build(std::vector<uint32> const &ids, std::vector<uint32> &spine) {
			spine.clear();
			for (uint32 const x : ids) {
				uint32 last = 0;
				while (!spine.empty() && s_.priority(spine.back()) < s_.priority(x)) {
					last = spine.back();
					spine.pop_back();
					s_.pull(last);
				}
				s_.nodes[x].child[0] = last;
				s_.nodes[x].child[1] = 0;
				s_.set_parent(last, x);
				if (spine.empty()) {
					s_.set_parent(x, 0);
				} else {
					s_.nodes[spine.back()].child[1] = x;
					s_.set_parent(x, spine.back());
				}
				spine.push_back(x);
			}
			for (std::size_t k = spine.size(); k-- > 0;) s_.pull(spine[k]);
		}

		constexpr void check_vertex(uint32 v) const { atl_assert(v < n_); }

	public:
		/**
		 * @brief An empty forest (no vertices).
		 * @complexity O(1)
		 */
		constexpr euler_tour_tree()
			requires std::default_initializable<op_slot> && std::default_initializable<mapping_slot>
				  && std::default_initializable<compose_slot> && std::default_initializable<value_type>
				  && std::default_initializable<tag_type>
		{
			init(0, value_type{}, tag_type{});
		}

		/**
		 * @brief @p n isolated vertices (structure-only tier).
		 * @pre `n <= (2^32 - 1) / 3` (asserted) @complexity O(n)
		 */
		constexpr explicit euler_tour_tree(std::size_t n)
			requires(!has_value)
		{
			init(n, value_type{}, tag_type{});
		}

		/**
		 * @brief @p n isolated vertices, each holding @p identity, folded with
		 *        the commutative @p op.
		 * @pre `n <= (2^32 - 1) / 3` (asserted) @complexity O(n)
		 */
		constexpr euler_tour_tree(std::size_t n, value_type identity, op_slot op = op_slot{})
			requires(has_value && !has_tags)
			: s_{{}, {}, detail::move_or_copy(op), {}, {}, {}} {
			init(n, identity, tag_type{});
		}

		/**
		 * @brief One isolated vertex per element of @p values. Elements are
		 *        moved out only of an rvalue range that owns them.
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires(has_value && !has_tags && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
		constexpr euler_tour_tree(R &&values, std::type_identity_t<value_type> identity, op_slot op = op_slot{})
			: s_{{}, {}, detail::move_or_copy(op), {}, {}, {}} {
			init_values(std::forward<R>(values), identity, tag_type{});
		}

		/**
		 * @brief @p n isolated vertices holding @p identity, with lazy tags
		 *        (`compose(f, g)` = "g first, then f", plus
		 *        `mapping(f, identity) == identity`).
		 * @complexity O(n)
		 */
		constexpr euler_tour_tree(std::size_t n, value_type identity, tag_type tag_identity, op_slot op = op_slot{},
								  mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
			requires has_tags
			: s_{{}, {}, detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), {}} {
			init(n, identity, tag_identity);
		}

		/** @brief One isolated vertex per element of @p values, with lazy tags. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires(has_tags && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
		constexpr euler_tour_tree(R &&values, std::type_identity_t<value_type> identity,
								  std::type_identity_t<tag_type> tag_identity, op_slot op = op_slot{},
								  mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
			: s_{{}, {}, detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), {}} {
			init_values(std::forward<R>(values), identity, tag_identity);
		}

		/** @brief The number of vertices. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether there are no vertices. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }

		/**
		 * @brief Link every vertex v with `parents[v] != none` (and `!= v`) to
		 *        `parents[v]` in O(n): each tree's tour is written by one DFS and
		 *        built as one treap by the Cartesian-tree stack.
		 * @tparam Parents A forward vertex range iterable through const.
		 * @pre no edges yet; `parents` has size() entries and forms a forest
		 *      (all asserted) @complexity O(n) plus n hash insertions
		 */
		template<typename Parents>
			requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const>
		constexpr void assign_parents(Parents const &parents) {
			atl_assert(edges_.empty());
			std::vector<uint32> p;
			p.reserve(n_);
			for (auto const v : parents) {
				atl_assert(p.size() < n_);
				uint32 const parent = uint32(v);
				atl_assert(parent == none || std::cmp_equal(parent, v)); // no truncation into a valid id
				p.push_back(parent);
			}
			atl_assert(detail::is_parent_forest(p, n_));
			// children in CSR form (counting sort by parent)
			std::vector<uint32> start(std::size_t(n_) + 1, 0), children(n_), up_edge(n_, none);
			auto const is_root = [&p](uint32 v) { return p[v] == none || p[v] == v; };
			for (uint32 v = 0; v < n_; ++v)
				if (!is_root(v)) ++start[p[v] + 1];
			for (uint32 v = 0; v < n_; ++v) start[v + 1] += start[v];
			{
				std::vector<uint32> fill(start.begin(), start.end() - 1);
				for (uint32 v = 0; v < n_; ++v)
					if (!is_root(v)) children[fill[p[v]]++] = v;
			}
			// an isolated vertex may hold a tag owed to children it never had: drop it before it gains some
			if constexpr (has_tags)
				for (uint32 x = 1; x <= n_; ++x) s_.nodes[x].tag = s_.tag_identity;
			uint32 next_id = 0;
			std::vector<uint32> tour, stack, spine, cursor(start.begin(), start.end() - 1);
			tour.reserve(3 * std::size_t(n_));
			for (uint32 r = 0; r < n_; ++r) {
				if (!is_root(r) || start[r] == start[r + 1]) continue; // isolated vertices stay single nodes
				tour.clear();
				tour.push_back(node_of(r));
				stack.push_back(r);
				while (!stack.empty()) {
					uint32 const v = stack.back();
					if (cursor[v] < start[v + 1]) {
						uint32 const c = children[cursor[v]++];
						uint32 const e = next_id++;
						edges_.insert(edge_key(v, c), e);
						up_edge[c] = e;
						tour.push_back(arc(e, v, c));
						tour.push_back(node_of(c));
						stack.push_back(c);
					} else {
						stack.pop_back();
						if (!stack.empty()) tour.push_back(arc(up_edge[v], v, stack.back()));
					}
				}
				build(tour, spine);
			}
			// the unused ids, popped from the back lowest first (link / cut may have shuffled the old list)
			std::size_t const edge_slots = n_ > 0 ? n_ - 1 : 0;
			free_ids_.resize(edge_slots - next_id);
			for (std::size_t i = 0; i < free_ids_.size(); ++i) free_ids_[i] = uint32(edge_slots - 1 - i);
		}

		/**
		 * @brief Add the edge u - v.
		 * @pre `u, v < size()`, u and v in different trees (asserted)
		 * @complexity expected O(log n) plus one hash insertion (expected O(1))
		 */
		constexpr void link(uint32 u, uint32 v) {
			check_vertex(u);
			check_vertex(v);
			atl_assert(!connected(u, v));
			uint32 const e = free_ids_.back();
			free_ids_.pop_back();
			edges_.insert(edge_key(u, v), e);
			// u's tour rotated to start at u, spliced into v's just before v's node
			// (inside one of v's corners): A, v->u, [u, D], C, u->v, [v, B]
			auto const [c, d] = split_at(node_of(u), place::right);
			uint32 const middle = join(join(arc(e, v, u), join(d, c)), arc(e, u, v));
			settle_above(node_of(v));
			base::insert_before_node(s_, node_of(v), middle);
		}

		/**
		 * @brief Remove the edge u - v; its arc nodes return to the free list.
		 * @pre `u, v < size()`, u - v is an edge (asserted)
		 * @complexity expected O(log n) plus one hash erasure (expected O(1))
		 */
		constexpr void cut(uint32 u, uint32 v) {
			check_vertex(u);
			check_vertex(v);
			uint32 const e = edge_id(u, v);
			edges_.erase(edge_key(u, v));
			free_ids_.push_back(e);
			uint32 const x = arc(e, u, v), y = arc(e, v, u);
			// tour = L, x, R with y in R (R1, y, R2: R1 is one side, R2 + L the other)
			// or in L (L1, y, L2: L2 is one side, L1 + R the other)
			auto const [l, r] = split_at(x, place::excluded);
			uint32 y_root;
			if constexpr (has_tags) {
				collect(y, path_);
				y_root = path_.back();
				settle(path_, path_.size());
			} else {
				y_root = root_of(y);
			}
			auto const [before_y, after_y] = base::split_at_node(s_, y, place::excluded);
			if (y_root == r) // r != 0 here: y_root is a real node
				join(l, after_y);
			else
				join(before_y, r);
		}

		/**
		 * @brief Whether u and v are in the same tree.
		 * @pre `u, v < size()` (asserted) @complexity expected O(log n), no writes
		 */
		[[nodiscard]] constexpr bool connected(uint32 u, uint32 v) {
			check_vertex(u);
			check_vertex(v);
			return u == v || root_of(node_of(u)) == root_of(node_of(v));
		}

		/**
		 * @brief The number of vertices in v's tree.
		 * @pre `v < size()` (asserted) @complexity expected O(log n), no writes
		 */
		[[nodiscard]] constexpr std::size_t component_size(uint32 v) {
			check_vertex(v);
			return (s_.size(root_of(node_of(v))) + 2) / 3; // k vertices and k - 1 edges: 3k - 2 tour nodes
		}

		/**
		 * @brief The fold of every vertex value in v's tree.
		 * @pre `v < size()` (asserted) @complexity expected O(log n), no writes
		 */
		[[nodiscard]] constexpr value_type component_fold(uint32 v)
			requires has_value
		{
			check_vertex(v);
			return s_.nodes[root_of(node_of(v))].aggregate;
		}

		/**
		 * @brief Apply tag @p f to every vertex value in v's tree.
		 * @pre `v < size()` (asserted) @complexity expected O(log n)
		 */
		constexpr void component_apply(uint32 v, tag_type const &f)
			requires has_tags
		{
			check_vertex(v);
			s_.apply_tag(root_of(node_of(v)), f);
		}

		/**
		 * @brief The fold of the vertex values on v's side of the edge
		 *        v - @p parent: the subtree of v when the tree is rooted at
		 *        @p parent's side. No re-rooting, no restructuring.
		 * @pre `v, parent < size()`, v - parent is an edge (asserted)
		 * @complexity expected O(log n) plus a hash lookup
		 */
		[[nodiscard]] constexpr value_type subtree_fold(uint32 v, uint32 parent)
			requires has_value
		{
			check_vertex(v);
			check_vertex(parent);
			uint32 const e = edge_id(v, parent);
			value_type acc = s_.nodes[0].aggregate; // the identity
			visit_segment(
				arc(e, parent, v), arc(e, v, parent), [&](uint32 t) { acc = s_.op(acc, s_.nodes[t].aggregate); },
				[&](uint32 z) { acc = s_.op(acc, s_.nodes[z].value); });
			return acc;
		}

		/**
		 * @brief Apply tag @p f to every vertex value on v's side of the edge
		 *        v - @p parent. No re-rooting, no restructuring.
		 * @pre `v, parent < size()`, v - parent is an edge (asserted)
		 * @complexity expected O(log n) plus a hash lookup
		 */
		constexpr void subtree_apply(uint32 v, uint32 parent, tag_type const &f)
			requires has_tags
		{
			check_vertex(v);
			check_vertex(parent);
			uint32 const e = edge_id(v, parent);
			auto const [i, j] = visit_segment(
				arc(e, parent, v), arc(e, v, parent),
				[&](uint32 t) {
					if (t != 0) s_.apply_tag(t, f);
				},
				[&](uint32 z) { s_.nodes[z].value = s_.mapping(f, s_.nodes[z].value); });
			// both branches below the lca, then the shared path above it
			for (std::size_t k = 0; k < i; ++k) s_.pull(path_[k]);
			for (std::size_t k = 0; k < j; ++k) s_.pull(other_path_[k]);
			for (std::size_t k = i; k < path_.size(); ++k) s_.pull(path_[k]);
		}

		/**
		 * @brief The value of vertex @p v. O(1) without tags; with tags, by value
		 *        after settling the tags above it (expected O(log n)).
		 * @pre `v < size()` (asserted)
		 */
		[[nodiscard]] constexpr value_type const &at(uint32 v)
			requires(has_value && !has_tags)
		{
			check_vertex(v);
			return s_.nodes[node_of(v)].value;
		}

		/** @copydoc at */
		[[nodiscard]] constexpr value_type at(uint32 v)
			requires has_tags
		{
			check_vertex(v);
			settle_above(node_of(v));
			return s_.nodes[node_of(v)].value;
		}

		/**
		 * @brief Replace the value of vertex @p v.
		 * @pre `v < size()` (asserted) @complexity expected O(log n)
		 */
		template<typename NewValue>
			requires has_value && std::assignable_from<Value &, NewValue>
		constexpr void set(uint32 v, NewValue &&a) {
			check_vertex(v);
			uint32 const x = node_of(v);
			settle_above(x);
			s_.nodes[x].value = std::forward<NewValue>(a);
			pull_up(x);
		}

		/**
		 * @brief Replace the value `a` of vertex @p v by `f(a)`.
		 * @pre `v < size()` (asserted) @complexity expected O(log n)
		 */
		template<update_function<value_type> Update>
			requires has_value
		constexpr void update(uint32 v, Update &&f) {
			check_vertex(v);
			uint32 const x = node_of(v);
			settle_above(x);
			s_.nodes[x].value = std::invoke(f, std::as_const(s_.nodes[x].value));
			pull_up(x);
		}
	};

	/** @brief Structure only: the empty forest or @p n isolated vertices. */
	euler_tour_tree() -> euler_tour_tree<>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	euler_tour_tree(std::size_t) -> euler_tour_tree<>;
	/** @brief Vertex values: `Value` from the identity. */
	template<std::copyable Value, binary_operation<Value> Op>
	euler_tour_tree(std::size_t, Value, Op) -> euler_tour_tree<Value, Op>;
	/** @brief `Value` from the identity; the elements must convert to it without narrowing. */
	template<std::ranges::sized_range R, std::copyable Value, binary_operation<Value> Op>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	euler_tour_tree(R &&, Value, Op) -> euler_tour_tree<Value, Op>;
	/** @brief Lazy updates: `Value` and `Tag` from the identities. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	euler_tour_tree(std::size_t, Value, Tag, Op, Mapping, Compose) -> euler_tour_tree<Value, Op, Tag, Mapping, Compose>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op,
			 action<Tag, Value> Mapping, binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	euler_tour_tree(R &&, Value, Tag, Op, Mapping, Compose) -> euler_tour_tree<Value, Op, Tag, Mapping, Compose>;
}
