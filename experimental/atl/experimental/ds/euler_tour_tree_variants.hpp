#pragma once
#include <atl/ds/detail/algebra.hpp>

/**
 * @file
 * @brief Euler-tour forests with tours stored in splay trees rather than
 *        the parent-linked treaps of @ref atl::euler_tour_tree.
 *
 * benchmarks/ds/euler_tour_tree.cpp, Xeon Platinum 8168, GCC 15.2,
 * RelWithDebInfo, assertions off, 2026-10-25, N=Q=2e5,
 * -O2 / -O2 -march=native: churn takes 208 / 208 ms by splay versus
 * library 180 / 173 ms; repeated links 115 / 116 versus 91.7 / 91.4 ms.
 * Bulk build is close: 39.9 / 41.0 versus 42.8 / 42.1 ms. See the guide.
 * Split parent storage and in-place treap insertion reduce memory traffic;
 * an iterative top-down merge showed no advantage in local ARM64 runs.
 * Those tuning axes need a dedicated x86-64 race before further changes.
 */

#include <atl/ds/detail/dynamic_forest.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	namespace detail {
		/**
		 * @brief The node pool of a @ref atl::experimental::euler_tour_tree_by_splay : an
		 *        @ref atl::sized_splay_store with the `push` and `pull` hooks, one
		 *        interleaved record per node (links, subtree size, and on the
		 *        value tiers the value, its subtree fold and the pending tag).
		 *
		 * A tag at x is already reflected in x's own value and aggregate and owed
		 * to its children (the @ref atl::treap_sequence scheme), so a subtree root's
		 * aggregate is always current. Node 0 is the null sentinel: size 0 and
		 * the identity aggregate, so `pull` folds through empty sides without
		 * branching. There is no reversal: tour order only ever rotates.
		 */
		template<typename Value, typename Op, typename Tag, typename Mapping, typename Compose>
			requires atl::detail::euler_tour_algebra<Value, Op, Tag, Mapping, Compose> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
		struct euler_tour_store_by_splay {
			static constexpr bool has_value = !std::is_void_v<Value>;
			static constexpr bool has_tags = !std::is_void_v<Tag>;

			using value_field = std::conditional_t<has_value, Value, atl::detail::euler_tour_no_field<0>>;
			using tag_field = std::conditional_t<has_tags, Tag, atl::detail::euler_tour_no_field<1>>;

			template<typename Algebra, int Id>
			using slot = std::conditional_t<std::is_void_v<Algebra>, atl::detail::euler_tour_no_field<Id>, Algebra>;

			/** @brief One tour node (a vertex or an arc): links, size, value, aggregate and pending tag. */
			struct node {
				uint32 child[2]{};
				uint32 parent = 0;
				uint32 size = 0;
				[[no_unique_address]] value_field value{};
				[[no_unique_address]] value_field aggregate{};
				[[no_unique_address]] tag_field tag{};
			};

			std::vector<node> nodes;
			[[no_unique_address]] slot<Op, 2> op;
			[[no_unique_address]] slot<Mapping, 3> mapping;
			[[no_unique_address]] slot<Compose, 4> compose;
			[[no_unique_address]] tag_field tag_identity{};

			constexpr euler_tour_store_by_splay() = default;

			constexpr euler_tour_store_by_splay(std::vector<node> values, slot<Op, 2> fold, slot<Mapping, 3> map, slot<Compose, 4> composition, tag_field tag)
				: nodes(std::move(values)), op(atl::detail::move_or_copy(fold)), mapping(atl::detail::move_or_copy(map)),
				  compose(atl::detail::move_or_copy(composition)), tag_identity(std::move(tag)) {}

			[[nodiscard]] constexpr uint32 child(uint32 x, bool side) const { return nodes[x].child[side]; }

			[[nodiscard]] constexpr uint32 parent(uint32 x) const { return nodes[x].parent; }

			constexpr void set_child(uint32 x, bool side, uint32 c) { nodes[x].child[side] = c; }

			constexpr void set_parent(uint32 x, uint32 p) { nodes[x].parent = p; }

			[[nodiscard]] constexpr uint32 size(uint32 x) const { return nodes[x].size; }

			/**
			 * @brief Reset every node: @p count detached nodes (size 1) after the
			 *        sentinel (size 0), values and aggregates @p identity, tags
			 *        @p tag_identity_value.
			 */
			constexpr void assign(std::size_t count, value_field const &identity, tag_field const &tag_identity_value) {
				tag_identity = tag_identity_value;
				node fresh{};
				fresh.size = 1;
				if constexpr (has_value) fresh.value = fresh.aggregate = identity;
				if constexpr (has_tags) fresh.tag = tag_identity;
				nodes.assign(count + 1, fresh);
				nodes[0].size = 0;
			}

			/** @brief Size and aggregate from the children, branchless through null sides. */
			constexpr void pull(uint32 x) {
				node &n = nodes[x];
				node const &l = nodes[n.child[0]], &r = nodes[n.child[1]];
				n.size = l.size + 1 + r.size;
				if constexpr (has_value) n.aggregate = op(op(l.aggregate, n.value), r.aggregate);
			}

			/**
			 * @brief Fold tag @p f into node @p x: value and aggregate now, the tag
			 *        composed for the subtree below. @pre `x != 0`
			 */
			constexpr void apply_tag(uint32 x, tag_field const &f)
				requires has_tags
			{
				node &n = nodes[x];
				n.value = mapping(f, n.value);
				n.aggregate = mapping(f, n.aggregate);
				n.tag = compose(f, n.tag);
			}

			/** @brief Settle x's pending tag one level down (called by the engine). */
			constexpr void push(uint32 x)
				requires has_tags
			{
				Tag &pending = nodes[x].tag;
				// an identity tag has nothing to settle: skip the child mappings
				if constexpr (std::equality_comparable<Tag>)
					if (pending == tag_identity) return;
				if (uint32 const l = child(x, 0); l != 0) apply_tag(l, pending);
				if (uint32 const r = child(x, 1); r != 0) apply_tag(r, pending);
				pending = tag_identity;
			}
		};
	}

	/**
	 * @brief @ref atl::euler_tour_tree with the tour in a splay tree (a client
	 *        of the public @ref atl::splay_engine): the same API and semantics,
	 *        amortized O(log n) bounds. Every query splays.
	 *
	 * Race loser (ds-euler_tour_tree, 2026-10-25, Xeon Platinum 8168, N=Q=2e5,
	 * plain -O2; treap store first): connectivity churn 180 vs 208 ms, vertex
	 * add + subtree sum 329 vs 355, the same on a static tree 158 vs 238,
	 * N - 1 links 92 vs 115, subtree add + subtree sum 724 vs 749;
	 * assign_parents within noise (42.8 vs 39.9, spreads 1.04-1.10). With
	 * -march=native the churn gap holds (173 vs 208). Splaying writes on every
	 * query and walks the access path twice; the treap's parent links answer
	 * the read-only queries with one upward walk and fold a subtree segment in
	 * place.
	 *
	 * @tparam Value   Default-initializable when non-void. Vertex value; `void` for structure only.
	 * @tparam Op      The fold, associative and commutative; `op(identity, a) == a`.
	 * @tparam Tag     Default-initializable when non-void. Update tag (needs `Value`).
	 * @tparam Mapping `mapping(f, a)`, distributing over `Op`, fixing the identity.
	 * @tparam Compose `compose(f, g)` = "g first, then f".
	 */
	template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void>
		requires atl::detail::euler_tour_algebra<Value, Op, Tag, Mapping, Compose> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
	class euler_tour_tree_by_splay {
		using store_type = detail::euler_tour_store_by_splay<Value, Op, Tag, Mapping, Compose>;
		using engine = atl::splay_engine;

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
		flat_hash_map<uint64, uint32, atl::detail::forest_edge_hash> edges_;
		std::vector<uint32> free_ids_;

		[[nodiscard]] static constexpr uint32 node_of(uint32 v) { return v + 1; }

		[[nodiscard]] constexpr uint32 arc_base(uint32 e) const { return n_ + 1 + 2 * e; }

		[[nodiscard]] static constexpr uint64 edge_key(uint32 u, uint32 v) { return atl::detail::forest_edge_key(u, v); }

		/** @brief The arc node u -> v of edge @p e (whose endpoints are u and v). */
		[[nodiscard]] constexpr uint32 arc(uint32 e, uint32 u, uint32 v) const { return arc_base(e) + (u < v ? 0 : 1); }

		/** @brief The id of edge u - v, asserted to exist. */
		[[nodiscard]] constexpr uint32 edge_id(uint32 u, uint32 v) const {
			uint32 const e = edges_.at_or(edge_key(u, v), none);
			atl_assert(e != none); // u - v is an edge
			return e;
		}

		/** @brief Cut x's child on @p side loose (x must be pushed) and return it. */
		constexpr uint32 detach(uint32 x, bool side) {
			uint32 const c = s_.child(x, side);
			if (c == 0) return 0;
			s_.set_child(x, side, 0);
			s_.set_parent(c, 0);
			s_.pull(x);
			return c;
		}

		/** @brief Root a tree at the childless node x with sides @p left and @p right. */
		constexpr uint32 attach(uint32 left, uint32 x, uint32 right) {
			s_.set_child(x, 0, left);
			if (left != 0) s_.set_parent(left, x);
			s_.set_child(x, 1, right);
			if (right != 0) s_.set_parent(right, x);
			s_.pull(x);
			return x;
		}

		/**
		 * @brief Walk from @p x up to its splay root @p y (x just splayed, then y),
		 *        visiting the tour segment that starts right after x and runs
		 *        cyclically to just before y: x's right subtree, then each ancestor
		 *        entered from its left with its right subtree, then y's left
		 *        subtree when x lies right of y (the segment wraps). @p side_visit
		 *        gets each whole subtree (possibly 0), @p node_visit each single
		 *        node, @p path_visit every node from x up to y, bottom-up, after
		 *        its parts. The splay of y leaves x at depth at most 2, so the walk
		 *        is O(1).
		 */
		template<typename SideVisit, typename NodeVisit, typename PathVisit>
			requires std::invocable<SideVisit &, uint32> && std::invocable<NodeVisit &, uint32>
				  && std::invocable<PathVisit &, uint32>
		constexpr void visit_segment(uint32 x, uint32 y, SideVisit &&side_visit, NodeVisit &&node_visit,
									 PathVisit &&path_visit) {
			side_visit(s_.child(x, 1));
			path_visit(x);
			uint32 c = x;
			for (uint32 z = s_.parent(c); z != y; c = z, z = s_.parent(z)) {
				if (s_.child(z, 0) == c) {
					node_visit(z);
					side_visit(s_.child(z, 1));
				}
				path_visit(z);
			}
			if (s_.child(y, 1) == c) side_visit(s_.child(y, 0));
			path_visit(y);
		}

		/**
		 * @brief Splay the arcs of edge v - p so the segment of v's side is
		 *        reachable from arc(p -> v); returns {arc(p -> v), arc(v -> p)}.
		 */
		constexpr std::pair<uint32, uint32> splay_side(uint32 v, uint32 p) {
			check_vertex(v);
			check_vertex(p);
			uint32 const e = edge_id(v, p);
			uint32 const in = arc(e, p, v), out = arc(e, v, p);
			engine::splay(s_, in);
			engine::splay(s_, out);
			return {in, out};
		}

		/** @brief @p n isolated vertices and 2 (n - 1) free arc nodes, all identities. */
		constexpr void init(std::size_t n, value_type const &identity, tag_type const &tag_identity) {
			// ids are 32-bit with `none` reserved: n vertex nodes and 2 (n - 1) arc nodes
			atl_assert(n <= std::size_t(graph::none) / 3);
			n_ = uint32(n);
			std::size_t const edge_slots = n > 0 ? n - 1 : 0;
			s_.assign(n + 2 * edge_slots, identity, tag_identity);
			edges_.clear();
			edges_.reserve(edge_slots);
			free_ids_.resize(edge_slots);
			// popped from the back: the lowest id first
			for (std::size_t i = 0; i < edge_slots; ++i) free_ids_[i] = uint32(edge_slots - 1 - i);
		}

		/** @brief One isolated vertex per element of @p values. */
		template<std::ranges::sized_range R>
			requires std::constructible_from<Value, atl::detail::forwarded_element_t<R>>
		constexpr void init_values(R &&values, value_type const &identity, tag_type const &tag_identity) {
			init(std::ranges::size(values), identity, tag_identity);
			uint32 x = 1;
			for (auto &&v : values) {
				auto &n = s_.nodes[x++];
				n.value = static_cast<Value>(atl::detail::forward_element<R>(v));
				n.aggregate = n.value;
			}
		}

		constexpr void check_vertex(uint32 v) const { atl_assert(v < n_); }

	public:
		/**
		 * @brief An empty forest (no vertices).
		 * @complexity O(1)
		 */
		constexpr euler_tour_tree_by_splay()
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
		constexpr explicit euler_tour_tree_by_splay(std::size_t n)
			requires(!has_value)
		{
			init(n, value_type{}, tag_type{});
		}

		/**
		 * @brief @p n isolated vertices, each holding @p identity, folded with
		 *        the commutative @p op.
		 * @pre `n <= (2^32 - 1) / 3` (asserted) @complexity O(n)
		 */
		constexpr euler_tour_tree_by_splay(std::size_t n, value_type identity, op_slot op = op_slot{})
			requires(has_value && !has_tags)
			: s_{{}, atl::detail::move_or_copy(op), {}, {}, {}} {
			init(n, identity, tag_type{});
		}

		/**
		 * @brief One isolated vertex per element of @p values. Elements are
		 *        moved out only of an rvalue range that owns them.
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires(has_value && !has_tags && std::constructible_from<value_type, atl::detail::forwarded_element_t<R>>)
		constexpr euler_tour_tree_by_splay(R &&values, std::type_identity_t<value_type> identity, op_slot op = op_slot{})
			: s_{{}, atl::detail::move_or_copy(op), {}, {}, {}} {
			init_values(std::forward<R>(values), identity, tag_type{});
		}

		/**
		 * @brief @p n isolated vertices holding @p identity, with lazy tags
		 *        (`compose(f, g)` = "g first, then f", the @ref atl::lazy_segment_tree
		 *        laws, plus `mapping(f, identity) == identity`).
		 * @complexity O(n)
		 */
		constexpr euler_tour_tree_by_splay(std::size_t n, value_type identity, tag_type tag_identity, op_slot op = op_slot{},
										   mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
			requires has_tags
			: s_{{}, atl::detail::move_or_copy(op), atl::detail::move_or_copy(mapping), atl::detail::move_or_copy(compose), {}} {
			init(n, identity, tag_identity);
		}

		/** @brief One isolated vertex per element of @p values, with lazy tags. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires(has_tags && std::constructible_from<value_type, atl::detail::forwarded_element_t<R>>)
		constexpr euler_tour_tree_by_splay(R &&values, std::type_identity_t<value_type> identity,
										   std::type_identity_t<tag_type> tag_identity, op_slot op = op_slot{},
										   mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
			: s_{{}, atl::detail::move_or_copy(op), atl::detail::move_or_copy(mapping), atl::detail::move_or_copy(compose), {}} {
			init_values(std::forward<R>(values), identity, tag_identity);
		}

		/** @brief The number of vertices. @complexity O(1) */
		[[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether there are no vertices. @complexity O(1) */
		[[nodiscard]] constexpr bool empty() const noexcept { return n_ == 0; }

		/**
		 * @brief Link every vertex v with `parents[v] != none` (and `!= v`) to
		 *        `parents[v]` (a parent-array format) in O(n):
		 *        each tree's tour is written by one DFS and linked balanced,
		 *        instead of n - 1 calls to `link`.
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
			atl_assert(atl::detail::is_parent_forest(p, n_));
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
			std::vector<uint32> tour, stack, cursor(start.begin(), start.end() - 1);
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
				engine::build_balanced(s_, tour);
			}
			// the unused ids, popped from the back lowest first (link / cut may have shuffled the old list)
			std::size_t const edge_slots = n_ > 0 ? n_ - 1 : 0;
			free_ids_.resize(edge_slots - next_id);
			for (std::size_t i = 0; i < free_ids_.size(); ++i) free_ids_[i] = uint32(edge_slots - 1 - i);
		}

		/**
		 * @brief Add the edge u - v.
		 * @pre `u, v < size()`, u and v in different trees (asserted)
		 * @complexity amortized O(log n) plus one hash insertion (expected O(1))
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
			auto const [c, d] = engine::split_before(s_, node_of(u));
			auto const [a, b] = engine::split_before(s_, node_of(v));
			engine::join(s_, attach(a, arc(e, v, u), d), attach(c, arc(e, u, v), b));
		}

		/**
		 * @brief Remove the edge u - v; its arc nodes return to the free list.
		 * @pre `u, v < size()`, u - v is an edge (asserted)
		 * @complexity amortized O(log n) plus one hash erasure (expected O(1))
		 */
		constexpr void cut(uint32 u, uint32 v) {
			check_vertex(u);
			check_vertex(v);
			uint32 const e = edge_id(u, v);
			edges_.erase(edge_key(u, v));
			free_ids_.push_back(e);
			uint32 const x = arc(e, u, v), y = arc(e, v, u);
			// tour = P1, first arc, P2, second arc, P3: P2 is one side, P3 + P1 the other
			auto const [before, from_x] = engine::split_before(s_, x);
			engine::splay(s_, y);
			uint32 outer_front, outer_back;
			if (s_.parent(x) == 0) { // y left x's tree alone: y precedes x
				outer_front = detach(y, 0);
				detach(y, 1); // P2, already its own tree
				outer_back = detach(from_x, 1);
			} else {		  // y is the root of x's tree, x its first node
				detach(y, 0); // x, P2
				outer_back = detach(y, 1);
				engine::splay(s_, x); // x comes first in the detached part: no left child
				detach(x, 1);		  // P2
				outer_front = before;
			}
			engine::join(s_, outer_front, outer_back);
		}

		/**
		 * @brief Whether u and v are in the same tree.
		 * @pre `u, v < size()` (asserted) @complexity amortized O(log n)
		 */
		[[nodiscard]] constexpr bool connected(uint32 u, uint32 v) {
			check_vertex(u);
			check_vertex(v);
			if (u == v) return true;
			engine::splay(s_, node_of(u));
			engine::splay(s_, node_of(v));
			return s_.parent(node_of(u)) != 0; // u is no longer a root iff v's splay passed it
		}

		/**
		 * @brief The number of vertices in v's tree.
		 * @pre `v < size()` (asserted) @complexity amortized O(log n)
		 */
		[[nodiscard]] constexpr std::size_t component_size(uint32 v) {
			check_vertex(v);
			engine::splay(s_, node_of(v));
			return (s_.size(node_of(v)) + 2) / 3; // k vertices and k - 1 edges: 3k - 2 tour nodes
		}

		/**
		 * @brief The fold of every vertex value in v's tree.
		 * @pre `v < size()` (asserted) @complexity amortized O(log n)
		 */
		[[nodiscard]] constexpr value_type component_fold(uint32 v)
			requires has_value
		{
			check_vertex(v);
			engine::splay(s_, node_of(v));
			return s_.nodes[node_of(v)].aggregate;
		}

		/**
		 * @brief Apply tag @p f to every vertex value in v's tree.
		 * @pre `v < size()` (asserted) @complexity amortized O(log n)
		 */
		constexpr void component_apply(uint32 v, tag_type const &f)
			requires has_tags
		{
			check_vertex(v);
			engine::splay(s_, node_of(v));
			s_.apply_tag(node_of(v), f);
		}

		/**
		 * @brief The fold of the vertex values on v's side of the edge
		 *        v - @p parent: the subtree of v when the tree is rooted at
		 *        @p parent's side. No re-rooting.
		 * @pre `v, parent < size()`, v - parent is an edge (asserted)
		 * @complexity amortized O(log n) plus a hash lookup
		 */
		[[nodiscard]] constexpr value_type subtree_fold(uint32 v, uint32 parent)
			requires has_value
		{
			auto const [x, y] = splay_side(v, parent);
			value_type acc = s_.nodes[0].aggregate; // the identity
			visit_segment(
				x, y, [&](uint32 t) { acc = s_.op(acc, s_.nodes[t].aggregate); },
				[&](uint32 z) { acc = s_.op(acc, s_.nodes[z].value); }, [](uint32) {});
			return acc;
		}

		/**
		 * @brief Apply tag @p f to every vertex value on v's side of the edge
		 *        v - @p parent. No re-rooting.
		 * @pre `v, parent < size()`, v - parent is an edge (asserted)
		 * @complexity amortized O(log n) plus a hash lookup
		 */
		constexpr void subtree_apply(uint32 v, uint32 parent, tag_type const &f)
			requires has_tags
		{
			auto const [x, y] = splay_side(v, parent);
			visit_segment(
				x, y,
				[&](uint32 t) {
					if (t != 0) s_.apply_tag(t, f);
				},
				[&](uint32 z) { s_.nodes[z].value = s_.mapping(f, s_.nodes[z].value); }, [&](uint32 z) { s_.pull(z); });
		}

		/**
		 * @brief The value of vertex @p v. O(1) without tags; with tags, by value
		 *        after settling the tags above it (amortized O(log n)).
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
			engine::splay(s_, node_of(v));
			return s_.nodes[node_of(v)].value;
		}

		/**
		 * @brief Replace the value of vertex @p v.
		 * @pre `v < size()` (asserted) @complexity amortized O(log n)
		 */
		template<typename NewValue>
			requires has_value && std::assignable_from<Value &, NewValue>
		constexpr void set(uint32 v, NewValue &&a) {
			check_vertex(v);
			uint32 const x = node_of(v);
			engine::splay(s_, x);
			s_.nodes[x].value = std::forward<NewValue>(a);
			s_.pull(x);
		}

		/**
		 * @brief Replace the value `a` of vertex @p v by `f(a)`.
		 * @pre `v < size()` (asserted) @complexity amortized O(log n)
		 */
		template<update_function<value_type> Update>
			requires has_value
		constexpr void update(uint32 v, Update &&f) {
			check_vertex(v);
			uint32 const x = node_of(v);
			engine::splay(s_, x);
			s_.nodes[x].value = std::invoke(f, std::as_const(s_.nodes[x].value));
			s_.pull(x);
		}
	};

	/** @brief Structure only: the empty forest or @p n isolated vertices. */
	euler_tour_tree_by_splay() -> euler_tour_tree_by_splay<>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	euler_tour_tree_by_splay(std::size_t) -> euler_tour_tree_by_splay<>;
	/** @brief Vertex values: `Value` from the identity. */
	template<std::copyable Value, binary_operation<Value> Op>
	euler_tour_tree_by_splay(std::size_t, Value, Op) -> euler_tour_tree_by_splay<Value, Op>;
	/** @brief `Value` from the identity; the elements must convert to it without narrowing. */
	template<std::ranges::sized_range R, std::copyable Value, binary_operation<Value> Op>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	euler_tour_tree_by_splay(R &&, Value, Op) -> euler_tour_tree_by_splay<Value, Op>;
	/** @brief Lazy updates: `Value` and `Tag` from the identities. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	euler_tour_tree_by_splay(std::size_t, Value, Tag, Op, Mapping, Compose) -> euler_tour_tree_by_splay<Value, Op, Tag, Mapping, Compose>;
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op,
			 action<Tag, Value> Mapping, binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	euler_tour_tree_by_splay(R &&, Value, Tag, Op, Mapping, Compose) -> euler_tour_tree_by_splay<Value, Op, Tag, Mapping, Compose>;
}
