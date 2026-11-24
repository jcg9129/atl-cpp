#pragma once

/** @file
 * @brief Link-cut trees without the edge registry; include link_cut_tree.hpp for edge values.
 */
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/dynamic_forest_core.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief The template-argument contract of @ref atl::link_cut_tree : tiers
		 *        switched on by non-`void` arguments (values with their fold,
		 *        then lazy tags), each tier's algebra well-formed; the fold
		 *        order and the edge tier need values.
		 */
		template<typename Value, typename Op, typename Tag, typename Mapping, typename Compose, typename FoldOrder,
				 typename ValuePlacement>
		concept link_cut_algebra = (std::is_void_v<Value> == std::is_void_v<Op>)
								&& (std::is_void_v<Value> || (std::copyable<Value> && binary_operation<Op, Value>))
								&& (std::is_void_v<Tag> == std::is_void_v<Mapping>) && (std::is_void_v<Tag> == std::is_void_v<Compose>)
								&& (std::is_void_v<Tag>
									|| (!std::is_void_v<Value> && std::copyable<Tag> && action<Mapping, Tag, Value>
										&& binary_operation<Compose, Tag>))
								&& fold_order<FoldOrder> && (std::is_void_v<FoldOrder> || !std::is_void_v<Value>)
								&& value_placement<ValuePlacement> && (std::is_void_v<ValuePlacement> || !std::is_void_v<Value>);

		/**
		 * @brief The subtree-aggregate contract (@ref atl::subtree_link_cut_tree):
		 *        `Inverse` is `void` (off), or the inverse of a commutative group
		 *        on the values, which then carry no tags and fold in the
		 *        commutative mode.
		 */
		template<typename Value, typename Tag, typename FoldOrder, typename Inverse>
		concept link_cut_subtree_algebra = std::is_void_v<Inverse>
										|| (!std::is_void_v<Value> && std::is_void_v<Tag> && std::same_as<FoldOrder, commutative_fold_t>
											&& inverse_operation<Inverse, Value>);

		/** @brief An absent node field or algebra slot (distinct per @p Id, so absent fields overlap). */
		template<int Id>
		struct link_cut_no_field {};

		/**
		 * @brief The node pool of a @ref atl::link_cut_tree : a @ref atl::splay_store
		 *        (sized when @p IsSized) with the `push`, `pull` and
		 *        `is_splay_root` hooks the engine drives, in the chosen layout.
		 *
		 * Per node: links, the subtree size, a pending-reversal mark, and on the
		 * value tiers the value, its subtree fold in in-order and (unless the
		 * fold commutes) in reversed order, and the pending tag. A mark or a tag
		 * at x is already reflected in x's own aggregates and owed to its
		 * children (the @ref atl::treap_sequence scheme), so a subtree root's
		 * aggregates are always current. Node 0 is the null sentinel: size 0 and
		 * the identity in both aggregates, so `pull` folds through empty sides
		 * without branching. `is_splay_root` is the path-parent test: a parent
		 * that does not list x as a child is the tree parent of x's whole path.
		 *
		 * With a non-`void` @p Inverse, the subtree variant of
		 * @ref atl::subtree_link_cut_tree adds per node the fold and the vertex
		 * count of its virtual children's whole subtrees (the trees hanging off
		 * x by path-parent pointers), and the splay subtree's total: every
		 * vertex of the splay subtree with everything hanging off it. Both are
		 * reversal-invariant (the fold commutes), so a flip leaves them alone.
		 */
		template<splay_layout Layout, bool IsSized, typename Value, typename Op, typename Tag, typename Mapping, typename Compose,
				 typename FoldOrder, typename Inverse = void>
			requires link_cut_algebra<Value, Op, Tag, Mapping, Compose, FoldOrder, void> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
				  && link_cut_subtree_algebra<Value, Tag, FoldOrder, Inverse>
		struct link_cut_store {
			static constexpr bool has_value = !std::is_void_v<Value>;
			static constexpr bool has_tags = !std::is_void_v<Tag>;
			static constexpr bool has_reversed_aggregate = has_value && !std::same_as<FoldOrder, commutative_fold_t>;
			static constexpr bool is_columnar = std::same_as<Layout, splay_layout_columnar>;
			static constexpr bool is_interleaved = std::same_as<Layout, splay_layout_interleaved>;
			static constexpr bool has_subtree = !std::is_void_v<Inverse>;

			using value_field = std::conditional_t<has_value, Value, link_cut_no_field<0>>;
			using aggregate_field = std::conditional_t<has_value, Value, link_cut_no_field<1>>;
			using reversed_field = std::conditional_t<has_reversed_aggregate, Value, link_cut_no_field<2>>;
			using tag_field = std::conditional_t<has_tags, Tag, link_cut_no_field<3>>;
			using size_field = std::conditional_t<IsSized, uint32, link_cut_no_field<4>>;
			using virtual_field = std::conditional_t<has_subtree, Value, link_cut_no_field<9>>;
			using total_field = std::conditional_t<has_subtree, Value, link_cut_no_field<10>>;
			using virtual_size_field = std::conditional_t<has_subtree, uint32, link_cut_no_field<11>>;
			using total_size_field = std::conditional_t<has_subtree, uint32, link_cut_no_field<12>>;

			template<typename Algebra, int Id>
			using slot = std::conditional_t<std::is_void_v<Algebra>, link_cut_no_field<Id>, Algebra>;

			/** @brief A node's links and (when sized) subtree size: the split layout's record. */
			struct link {
				uint32 child[2]{};
				uint32 parent = 0;
				[[no_unique_address]] size_field size{};
			};

			/** @brief A node's value, aggregates, tag and subtree totals, each present only on its tier. */
			struct payload {
				[[no_unique_address]] value_field value{};
				[[no_unique_address]] aggregate_field aggregate{};
				[[no_unique_address]] reversed_field reversed_aggregate{};
				[[no_unique_address]] tag_field tag{};
				[[no_unique_address]] virtual_field virtual_aggregate{}; ///< fold of the virtual children's subtrees
				[[no_unique_address]] total_field total_aggregate{};	 ///< the splay subtree and all hanging off it
				[[no_unique_address]] virtual_size_field virtual_size{}; ///< vertices in the virtual children's subtrees
				[[no_unique_address]] total_size_field total_size{};	 ///< vertices total_aggregate covers
				bool is_reversed = false;								 ///< the children are owed a swap (x's aggregates already swapped)
			};

			/** @brief The interleaved layout's one record per node. */
			struct node : link, payload {};

			/** @brief Interleaved layout: one node array. */
			struct interleaved_storage {
				std::vector<node> nodes;
			};

			/** @brief Split layout: links and payloads in two parallel arrays. */
			struct split_storage {
				std::vector<link> links;
				std::vector<payload> payloads;
			};

			/** @brief One array per field; absent fields stay empty. */
			struct columnar_storage {
				std::array<std::vector<uint32>, 2> children;
				std::vector<uint32> parents, sizes;
				std::vector<value_field> values;
				std::vector<aggregate_field> aggregates;
				std::vector<reversed_field> reversed_aggregates;
				std::vector<tag_field> tags;
				std::vector<virtual_field> virtual_aggregates;
				std::vector<total_field> total_aggregates;
				std::vector<uint32> virtual_sizes, total_sizes;
				std::vector<uint8> reversed_flags;
			};

			using storage = std::conditional_t<is_interleaved, interleaved_storage,
											   std::conditional_t<is_columnar, columnar_storage, split_storage>>;

			storage store;
			[[no_unique_address]] slot<Op, 5> op;
			[[no_unique_address]] slot<Mapping, 6> mapping;
			[[no_unique_address]] slot<Compose, 7> compose;
			[[no_unique_address]] slot<Inverse, 13> inverse;
			[[no_unique_address]] tag_field tag_identity{};

			constexpr link_cut_store() = default;

			constexpr link_cut_store(storage nodes, slot<Op, 5> fold, slot<Mapping, 6> map,
									 slot<Compose, 7> composition, slot<Inverse, 13> invert, tag_field tag)
				: store(std::move(nodes)), op(detail::move_or_copy(fold)), mapping(detail::move_or_copy(map)),
				  compose(detail::move_or_copy(composition)), inverse(detail::move_or_copy(invert)), tag_identity(std::move(tag)) {}

			constexpr link &link_at(uint32 x)
				requires(!is_columnar)
			{
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.links[x];
			}

			constexpr link const &link_at(uint32 x) const
				requires(!is_columnar)
			{
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.links[x];
			}

			constexpr payload &payload_at(uint32 x)
				requires(!is_columnar)
			{
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.payloads[x];
			}

			/** @brief x's child on @p side (0 left, 1 right), in any layout. */
			[[nodiscard]] constexpr uint32 child(uint32 x, bool side) const {
				if constexpr (is_columnar)
					return store.children[side][x];
				else
					return link_at(x).child[side];
			}

			/** @brief x's parent: in its splay tree, or the path-parent at a splay root. */
			[[nodiscard]] constexpr uint32 parent(uint32 x) const {
				if constexpr (is_columnar)
					return store.parents[x];
				else
					return link_at(x).parent;
			}

			/** @brief Set x's child on @p side; one link, nothing else. */
			constexpr void set_child(uint32 x, bool side, uint32 c) {
				if constexpr (is_columnar)
					store.children[side][x] = c;
				else
					link_at(x).child[side] = c;
			}

			/** @brief Set x's parent; one link, nothing else. */
			constexpr void set_parent(uint32 x, uint32 p) {
				if constexpr (is_columnar)
					store.parents[x] = p;
				else
					link_at(x).parent = p;
			}

			[[nodiscard]] constexpr uint32 size(uint32 x) const
				requires IsSized
			{
				if constexpr (is_columnar)
					return store.sizes[x];
				else
					return link_at(x).size;
			}

			constexpr void set_size(uint32 x, uint32 n)
				requires IsSized
			{
				if constexpr (is_columnar)
					store.sizes[x] = n;
				else
					link_at(x).size = n;
			}

			[[nodiscard]] constexpr value_field &value(uint32 x)
				requires has_value
			{
				if constexpr (is_columnar)
					return store.values[x];
				else
					return payload_at(x).value;
			}

			[[nodiscard]] constexpr aggregate_field &aggregate(uint32 x)
				requires has_value
			{
				if constexpr (is_columnar)
					return store.aggregates[x];
				else
					return payload_at(x).aggregate;
			}

			[[nodiscard]] constexpr reversed_field &reversed_aggregate(uint32 x)
				requires has_reversed_aggregate
			{
				if constexpr (is_columnar)
					return store.reversed_aggregates[x];
				else
					return payload_at(x).reversed_aggregate;
			}

			/**
			 * @brief The fold of x's subtree read right to left: the reversed aggregate,
			 *        or the aggregate itself when the fold commutes.
			 */
			[[nodiscard]] constexpr aggregate_field &backward_aggregate(uint32 x)
				requires has_value
			{
				if constexpr (has_reversed_aggregate)
					return reversed_aggregate(x);
				else
					return aggregate(x);
			}

			[[nodiscard]] constexpr tag_field &tag(uint32 x)
				requires has_tags
			{
				if constexpr (is_columnar)
					return store.tags[x];
				else
					return payload_at(x).tag;
			}

			/** @brief Subtree variant: the fold of x's virtual children's whole subtrees. */
			[[nodiscard]] constexpr virtual_field &virtual_aggregate(uint32 x)
				requires has_subtree
			{
				if constexpr (is_columnar)
					return store.virtual_aggregates[x];
				else
					return payload_at(x).virtual_aggregate;
			}

			/** @brief Subtree variant: the fold over x's splay subtree of every value and virtual fold. */
			[[nodiscard]] constexpr total_field &total_aggregate(uint32 x)
				requires has_subtree
			{
				if constexpr (is_columnar)
					return store.total_aggregates[x];
				else
					return payload_at(x).total_aggregate;
			}

			/** @brief Subtree variant: the vertex count of x's virtual children's subtrees. */
			[[nodiscard]] constexpr virtual_size_field &virtual_size(uint32 x)
				requires has_subtree
			{
				if constexpr (is_columnar)
					return store.virtual_sizes[x];
				else
					return payload_at(x).virtual_size;
			}

			/** @brief Subtree variant: the vertex count total_aggregate covers. */
			[[nodiscard]] constexpr total_size_field &total_size(uint32 x)
				requires has_subtree
			{
				if constexpr (is_columnar)
					return store.total_sizes[x];
				else
					return payload_at(x).total_size;
			}

			/**
			 * @brief Move children of x between preferred and virtual (subtree
			 *        variant): @p out (a former preferred child, or 0) becomes
			 *        virtual, @p in (a former virtual child, or 0) stops being one.
			 *        Branchless: node 0 holds the identity and size 0, and the
			 *        inverse of the identity is the identity. x is not pulled.
			 */
			constexpr void exchange_virtual(uint32 x, uint32 out, uint32 in)
				requires has_subtree
			{
				virtual_aggregate(x) = op(op(virtual_aggregate(x), total_aggregate(out)), inverse(total_aggregate(in)));
				virtual_size(x) = virtual_size(x) - total_size(in) + total_size(out); // in is counted: no wraparound
			}

			/** @brief The fold of x's own value and its virtual children's subtrees. */
			[[nodiscard]] constexpr Value own_subtree_aggregate(uint32 x)
				requires has_subtree
			{
				return op(value(x), virtual_aggregate(x));
			}

			/** @brief Whether x's children are owed a swap. */
			[[nodiscard]] constexpr bool is_reversed(uint32 x) const {
				if constexpr (is_columnar)
					return store.reversed_flags[x] != 0;
				else if constexpr (is_interleaved)
					return store.nodes[x].is_reversed;
				else
					return store.payloads[x].is_reversed;
			}

			/** @brief Set or clear x's pending-reversal flag. */
			constexpr void set_reversed(uint32 x, bool is_set) {
				if constexpr (is_columnar)
					store.reversed_flags[x] = uint8(is_set);
				else
					payload_at(x).is_reversed = is_set;
			}

			/**
			 * @brief Reset every node: @p nodes detached nodes (size 1) after the sentinel
			 *        (size 0), every value and aggregate @p identity, every tag @p tag_identity.
			 */
			constexpr void assign(std::size_t nodes, value_field const &identity, tag_field const &tag_identity_value) {
				tag_identity = tag_identity_value;
				std::size_t const slots = nodes + 1;
				if constexpr (is_columnar) {
					store.children[0].assign(slots, 0);
					store.children[1].assign(slots, 0);
					store.parents.assign(slots, 0);
					if constexpr (IsSized) store.sizes.assign(slots, 1);
					if constexpr (has_value) {
						store.values.assign(slots, identity);
						store.aggregates.assign(slots, identity);
					}
					if constexpr (has_reversed_aggregate) store.reversed_aggregates.assign(slots, identity);
					if constexpr (has_tags) store.tags.assign(slots, tag_identity);
					if constexpr (has_subtree) {
						store.virtual_aggregates.assign(slots, identity);
						store.total_aggregates.assign(slots, identity);
						store.virtual_sizes.assign(slots, 0);
						store.total_sizes.assign(slots, 1);
					}
					store.reversed_flags.assign(slots, 0);
				} else {
					payload p{};
					if constexpr (has_value) p.value = p.aggregate = identity;
					if constexpr (has_reversed_aggregate) p.reversed_aggregate = identity;
					if constexpr (has_tags) p.tag = tag_identity;
					if constexpr (has_subtree) {
						p.virtual_aggregate = p.total_aggregate = identity;
						p.total_size = 1;
					}
					link l{};
					if constexpr (IsSized) l.size = 1;
					if constexpr (is_interleaved) {
						node n{};
						static_cast<link &>(n) = l;
						static_cast<payload &>(n) = p;
						store.nodes.assign(slots, n);
					} else {
						store.links.assign(slots, l);
						store.payloads.assign(slots, p);
					}
				}
				if constexpr (IsSized) set_size(0, 0);
				if constexpr (has_subtree) total_size(0) = 0;
			}

			/** @brief Give the childless node @p x the value @p v (both aggregates too). */
			template<typename NewValue>
				requires has_value && std::assignable_from<Value &, NewValue>
			constexpr void assign_value(uint32 x, NewValue &&v) {
				value(x) = std::forward<NewValue>(v);
				aggregate(x) = value(x);
				if constexpr (has_reversed_aggregate) reversed_aggregate(x) = value(x);
				if constexpr (has_subtree) total_aggregate(x) = own_subtree_aggregate(x);
			}

			/** @brief Size and aggregates from the children, branchless through null sides. */
			constexpr void pull(uint32 x) {
				uint32 const l = child(x, 0), r = child(x, 1);
				if constexpr (IsSized) set_size(x, size(l) + 1 + size(r));
				if constexpr (has_value) {
					aggregate(x) = op(op(aggregate(l), value(x)), aggregate(r));
					if constexpr (has_reversed_aggregate)
						reversed_aggregate(x) = op(op(reversed_aggregate(r), value(x)), reversed_aggregate(l));
				}
				if constexpr (has_subtree) {
					total_aggregate(x) = op(op(total_aggregate(l), own_subtree_aggregate(x)), total_aggregate(r));
					total_size(x) = total_size(l) + 1 + virtual_size(x) + total_size(r);
				}
			}

			/**
			 * @brief Mark subtree @p x reversed: its aggregates swap now, the children's
			 *        swap stays owed. @pre `x != 0`
			 */
			constexpr void flip(uint32 x) {
				set_reversed(x, !is_reversed(x));
				if constexpr (has_reversed_aggregate) std::ranges::swap(aggregate(x), reversed_aggregate(x));
			}

			/** @brief Drop x's mark alone (the parity-walk push strategy settles marks in bulk). */
			constexpr void clear_reversed(uint32 x)
				requires(!has_tags)
			{
				set_reversed(x, false);
			}

			/**
			 * @brief Fold tag @p f into node @p x: value and aggregates now, the tag
			 *        composed for the subtree below (the lazy tree's convention).
			 */
			constexpr void apply_tag(uint32 x, tag_field const &f)
				requires has_tags
			{
				value(x) = mapping(f, value(x));
				aggregate(x) = mapping(f, aggregate(x));
				if constexpr (has_reversed_aggregate) reversed_aggregate(x) = mapping(f, reversed_aggregate(x));
				tag(x) = compose(f, tag(x));
			}

			/**
			 * @brief Settle x's pending work one level down: a reversal mark swaps
			 *        x's children and flips each of them; a non-identity tag is
			 *        applied to each child and reset. Called by the engine before
			 *        x's children are read or relinked.
			 */
			constexpr void push(uint32 x) {
				if (is_reversed(x)) {
					uint32 const l = child(x, 0), r = child(x, 1);
					set_child(x, 0, r);
					set_child(x, 1, l);
					if (l != 0) flip(l);
					if (r != 0) flip(r);
					set_reversed(x, false);
				}
				if constexpr (has_tags) {
					Tag &pending = tag(x);
					// an identity tag has nothing to settle: skip the child mappings
					if constexpr (std::equality_comparable<Tag>)
						if (pending == tag_identity) return;
					if (uint32 const l = child(x, 0); l != 0) apply_tag(l, pending);
					if (uint32 const r = child(x, 1); r != 0) apply_tag(r, pending);
					pending = tag_identity;
				}
			}

			/** @brief Engine hook, the splay stop test: x has no parent, or a path-parent that does not list x as a child. */
			[[nodiscard]] constexpr bool is_splay_root(uint32 x) const {
				uint32 const p = parent(x);
				return p == 0 || (child(p, 0) != x && child(p, 1) != x);
			}
		};

		/** @brief The descent probe of a predecessor search: always right. */
		struct link_cut_rightmost_probe {
			/** @brief Always step right. */
			constexpr splay_step operator()(uint32) const { return splay_step::right; }
		};

		/**
		 * @brief An engine that can drive @p Store as the link-cut tree does:
		 *        handle splays and descents, plus rank splays on a sized store.
		 */
		template<typename Engine, typename Store>
		concept link_cut_engine_for = splay_store<Store> && requires(Store &s, uint32 x, link_cut_rightmost_probe probe) {
			Engine::splay(s, x);
			{ Engine::descend(s, x, probe) } -> std::convertible_to<uint32>;
		} && (!sized_splay_store<Store> || requires(Store &s, uint32 x) {
										  { Engine::splay_at(s, x, x) } -> std::convertible_to<uint32>;
									  });

		/**
		 * @brief The whole @ref atl::link_cut_tree over any splay engine, node layout,
		 *        size field and path-operation form, so the experimental
		 *        contenders share one implementation with the library class.
		 *
		 * @tparam Engine      @ref atl::splay_engine or an experimental contender.
		 * @tparam Layout      A `splay_layout`.
		 * @tparam IsSized     Keep subtree sizes (`depth`, `distance`, `ancestor`,
		 *                     `path_vertex` need them).
		 * @tparam IsEvertForm Path folds and updates by re-rooting at u and back
		 *                     (the contender) instead of the two-access form.
		 * @tparam Inverse     `void`, or the group inverse that switches on the
		 *                     subtree aggregates of @ref atl::subtree_link_cut_tree
		 *                     (vertex values only).
		 */
		template<typename Engine, splay_layout Layout, bool IsSized, bool IsEvertForm, typename Value, typename Op, typename Tag,
				 typename Mapping, typename Compose, typename FoldOrder, typename ValuePlacement, typename Inverse = void>
			requires link_cut_algebra<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
				  && link_cut_subtree_algebra<Value, Tag, FoldOrder, Inverse> && (std::is_void_v<Inverse> || std::is_void_v<ValuePlacement>)
				  && link_cut_engine_for<Engine, link_cut_store<Layout, IsSized, Value, Op, Tag, Mapping, Compose, FoldOrder, Inverse>>
		class link_cut_tree_impl {
		protected:
			using store_type = link_cut_store<Layout, IsSized, Value, Op, Tag, Mapping, Compose, FoldOrder, Inverse>;

			static constexpr bool has_value = store_type::has_value;
			static constexpr bool has_tags = store_type::has_tags;
			static constexpr bool has_subtree = store_type::has_subtree;
			static constexpr bool is_commutative = std::same_as<FoldOrder, commutative_fold_t>;
			static constexpr bool has_edge_values = std::same_as<ValuePlacement, edge_values_t>;
			/** @brief Nodes per tree edge on a root path: an edge tier path alternates vertex and edge nodes. */
			static constexpr uint32 stride = has_edge_values ? 2 : 1;

		public:
			/** @brief "No vertex / no edge": equal to @ref atl::graph::none. */
			static constexpr uint32 none = graph::none;

			/** @brief The value type (a placeholder on the structure-only tier). */
			using value_type = typename store_type::value_field;
			/** @brief The tag type (a placeholder below the lazy tier). */
			using tag_type = typename store_type::tag_field;

		protected:
			using op_slot = typename store_type::template slot<Op, 5>;
			using mapping_slot = typename store_type::template slot<Mapping, 6>;
			using compose_slot = typename store_type::template slot<Compose, 7>;
			using inverse_slot = typename store_type::template slot<Inverse, 13>;

			/** @brief The edge tier's bookkeeping (shared with @ref atl::top_tree). */
			using edge_registry = forest_edge_registry;

			store_type s_;
			uint32 n_ = 0;
			[[no_unique_address]] std::conditional_t<has_edge_values, edge_registry, link_cut_no_field<8>> edges_;

			/** @brief What `expose` returns: where the climb ended and what its last step cut off. */
			struct exposure {
				uint32 top;			 ///< the last splay root the climb reached: the splay root of the whole root path
				uint32 former_right; ///< top's right subtree before the climb replaced it
			};

			[[nodiscard]] static constexpr uint32 node_of(uint32 v) { return v + 1; }

			[[nodiscard]] static constexpr uint32 vertex_of(uint32 x) { return x - 1; }

			[[nodiscard]] constexpr uint32 edge_node(uint32 e) const { return n_ + 1 + e; }

			[[nodiscard]] static constexpr uint64 edge_key(uint32 u, uint32 v) { return forest_edge_key(u, v); }

			/**
			 * @brief Make root..x one preferred path, without the final splay of x.
			 *        Returns the top splay root, which is the splay root of that
			 *        whole path, and its right subtree just before the climb's last
			 *        step replaced it. After an exposure of u, exposing v returns
			 *        top = lca(u, v) when they are connected, with former_right the
			 *        splay tree of the path strictly between it and u.
			 */
			constexpr exposure expose(uint32 x) {
				exposure e{0, 0};
				for (uint32 last = 0, y = x; y != 0; last = y, y = s_.parent(y)) {
					Engine::splay(s_, y);
					e = {y, s_.child(y, 1)};
					if constexpr (has_subtree) s_.exchange_virtual(y, e.former_right, last);
					s_.set_child(y, 1, last); // the former right child keeps its parent: now a path-parent pointer
					s_.pull(y);
				}
				return e;
			}

			/** @brief Expose x and splay it to the top: x's splay tree is root..x, x last. */
			constexpr void access(uint32 x) {
				expose(x);
				Engine::splay(s_, x);
			}

			/**
			 * @brief Splay the rightmost node of the subtree at @p root to the top of its
			 *        splay tree (a predecessor search, kept amortized) and return it.
			 */
			constexpr uint32 splay_rightmost(uint32 root) {
				return Engine::descend(s_, root, link_cut_rightmost_probe{});
			}

			/** @brief Make node x its tree's root: access, then mark x's splay tree reversed. */
			constexpr void evert(uint32 x) {
				access(x);
				s_.flip(x);
			}

			/** @brief Root node of x's tree, splayed to the top of the root path. */
			constexpr uint32 find_root(uint32 x) {
				access(x);
				return Engine::descend(s_, x, [](uint32) { return splay_step::left; });
			}

			/**
			 * @brief After `expose(x)` and `{w, c} = expose(y)`: are x and y in one
			 *        tree? Either x is the meeting point, or the cut-off part
			 *        holding x hangs below w (x's splay root has a path-parent),
			 *        while an unconnected x tops its own root path.
			 */
			constexpr bool is_exposure_connected(uint32 x, uint32 w) {
				if (x == w) return true;
				Engine::splay(s_, x);
				return s_.parent(x) != 0;
			}

			/** @brief Are nodes x and y in one tree? (two exposures and at most one splay) */
			constexpr bool nodes_connected(uint32 x, uint32 y) {
				if (x == y) return true;
				expose(x);
				return is_exposure_connected(x, expose(y).top);
			}

			/**
			 * @brief Cut x from its parent node when that parent is @p expected; false
			 *        (nothing changed but the splay shape) otherwise.
			 */
			constexpr bool detach_from(uint32 x, uint32 expected) {
				access(x);
				uint32 const l = s_.child(x, 0);
				if (l == 0) return false;
				uint32 const p = splay_rightmost(l);
				if (p != expected) return false;
				// p's in-order successor is x, which has no right child after the access
				s_.set_child(p, 1, 0);
				s_.set_parent(x, 0);
				s_.pull(p);
				return true;
			}

			/** @brief Cut x from its parent node, whatever it is. @pre x has a parent (asserted) */
			constexpr uint32 detach_parent_path(uint32 x) {
				access(x);
				uint32 const l = s_.child(x, 0);
				atl_assert(l != 0); // x is not a root
				s_.set_child(x, 0, 0);
				s_.set_parent(l, 0);
				s_.pull(x);
				return l;
			}

			/**
			 * @brief Forget the already unlinked edge node @p y: its map entry goes and its
			 *        id returns to the free list.
			 */
			constexpr void release_edge(uint32 y) {
				uint32 const e = y - n_ - 1;
				auto const [a, b] = edges_.endpoints[e];
				edges_.ids.erase(edge_key(a, b));
				edges_.endpoints[e] = {none, none};
				edges_.free_ids.push_back(e);
			}

			/**
			 * @brief Remove edge node @p y from the forest: cut it from its parent, then
			 *        cut its one child from it.
			 */
			constexpr void cut_edge_node(uint32 y) {
				uint32 const e = y - n_ - 1;
				auto const [a, b] = edges_.endpoints[e];
				access(y);
				uint32 const l = s_.child(y, 0);
				atl_assert(l != 0); // an edge node always has a parent
				uint32 const p = splay_rightmost(l);
				s_.set_child(p, 1, 0);
				s_.set_parent(y, 0);
				s_.pull(p);
				// y now roots the part holding the other endpoint, its only child
				uint32 const c = p == node_of(a) ? node_of(b) : node_of(a);
				access(c);
				s_.set_child(c, 0, 0);
				s_.set_parent(y, 0);
				s_.pull(c);
				release_edge(y);
			}

			/** @brief O(n) acyclicity check of a parent array of size() vertices (assert-only). @tparam Parents A forward vertex range iterable through const. */
			template<typename Parents>
				requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const>
			[[nodiscard]] constexpr bool is_forest(Parents const &parents) const {
				return is_parent_forest(std::vector<uint32>(std::ranges::begin(parents), std::ranges::end(parents)), n_);
			}

			/** @brief No vertex has an edge (assert-only, O(n)). */
			[[nodiscard]] constexpr bool is_all_isolated() const {
				for (uint32 x = 1; x <= n_; ++x)
					if (s_.parent(x) != 0 || s_.child(x, 0) != 0 || s_.child(x, 1) != 0) return false;
				if constexpr (has_edge_values)
					if (edges_.ids.size() != 0) return false;
				return true;
			}

			/** @brief n isolated vertices (and n - 1 free edge nodes on the edge tier), all identities. */
			constexpr void init(std::size_t n, value_type const &identity, tag_type const &tag_identity) {
				// ids are 32-bit with `none` reserved; the edge tier holds up to 2n - 1 nodes
				atl_assert(n < (has_edge_values ? std::size_t(1) << 31 : std::size_t((std::numeric_limits<uint32>::max)())));
				n_ = uint32(n);
				std::size_t const edge_slots = has_edge_values && n > 0 ? n - 1 : 0;
				s_.assign(n + edge_slots, identity, tag_identity);
				if constexpr (has_edge_values) {
					edges_.ids.clear();
					edges_.ids.reserve(edge_slots);
					edges_.endpoints.assign(edge_slots, {none, none});
					reset_free_ids();
				}
			}

			/** @brief Every edge id free, handed out lowest first. */
			constexpr void reset_free_ids()
				requires has_edge_values
			{
				std::size_t const edge_slots = edges_.endpoints.size();
				edges_.free_ids.resize(edge_slots);
				// popped from the back: the lowest id first
				for (std::size_t i = 0; i < edge_slots; ++i) edges_.free_ids[i] = uint32(edge_slots - 1 - i);
			}

			/** @brief One isolated vertex per element of @p values. */
			template<std::ranges::sized_range R>
				requires std::constructible_from<Value, forwarded_element_t<R>>
			constexpr void init_values(R &&values, value_type const &identity, tag_type const &tag_identity) {
				init(std::ranges::size(values), identity, tag_identity);
				uint32 x = 1;
				for (auto &&v : values) s_.assign_value(x++, static_cast<Value>(forward_element<R>(v)));
			}

			/**
			 * @brief Bring node x where a change of its value reaches every aggregate
			 *        holding it by one pull: the top of its splay tree, and with
			 *        subtree aggregates the top of its whole tree (an access), since
			 *        the virtual aggregates above hold its splay tree's total.
			 */
			constexpr void raise_for_update(uint32 x) {
				if constexpr (has_subtree)
					access(x);
				else
					Engine::splay(s_, x);
			}

			/**
			 * @brief Subtree variant of assign_parents: every vertex is its own splay
			 *        tree hanging off its parent, so each parent's virtual aggregate
			 *        folds its children's totals, children first (topological order on the
			 *        child counts, no recursion). @pre the path-parent links are set
			 * @tparam Parents A forward vertex range iterable through const.
			 */
			template<typename Parents>
				requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const> && has_subtree
			constexpr void fold_virtual_children(Parents const &parents) {
				std::vector<uint32> p(std::ranges::begin(parents), std::ranges::end(parents));
				std::vector<uint32> pending(n_, 0), ready;
				ready.reserve(n_);
				for (uint32 v = 0; v < n_; ++v)
					if (p[v] != none && p[v] != v) ++pending[p[v]];
				for (uint32 v = 0; v < n_; ++v)
					if (pending[v] == 0) ready.push_back(v);
				for (std::size_t i = 0; i < ready.size(); ++i) {
					uint32 const v = ready[i];
					s_.pull(node_of(v)); // every child is in: v's total is final
					if (p[v] == none || p[v] == v) continue;
					s_.exchange_virtual(node_of(p[v]), node_of(v), 0);
					if (--pending[p[v]] == 0) ready.push_back(p[v]);
				}
			}

			/** @brief Assert that @p v is a vertex id. */
			constexpr void check_vertex(uint32 v) const { atl_assert(v < n_); }

			/** @brief Assert that @p e is a live edge id; return its node. */
			constexpr uint32 check_edge(uint32 e) const
				requires has_edge_values
			{
				atl_assert(e < edges_.endpoints.size() && edges_.endpoints[e].first != none); // a live edge id
				return edge_node(e);
			}

			/** @brief The node holding the value of @p id: a vertex, or an edge on the edge tier. */
			constexpr uint32 value_node(uint32 id) const {
				if constexpr (has_edge_values) {
					return check_edge(id);
				} else {
					check_vertex(id);
					return node_of(id);
				}
			}

		public:
			/**
			 * @brief An empty forest (no vertices).
			 * @complexity O(1)
			 */
			constexpr link_cut_tree_impl()
				requires std::default_initializable<op_slot> && std::default_initializable<mapping_slot>
					  && std::default_initializable<compose_slot> && std::default_initializable<value_type>
					  && std::default_initializable<tag_type>
			{
				init(0, value_type{}, tag_type{});
			}

			/**
			 * @brief @p n isolated vertices (structure-only tier).
			 * @pre `n < 2^32 - 1` (asserted) @complexity O(n)
			 */
			constexpr explicit link_cut_tree_impl(std::size_t n)
				requires(!has_value)
			{
				init(n, value_type{}, tag_type{});
			}

			/**
			 * @brief @p n isolated vertices, each holding @p identity (on the edge
			 *        tier: no edges yet), folded with @p op.
			 * @complexity O(n)
			 */
			constexpr link_cut_tree_impl(std::size_t n, value_type identity, op_slot op = op_slot{})
				requires(has_value && !has_tags && !has_subtree)
				: s_{{}, detail::move_or_copy(op), {}, {}, {}, {}} {
				init(n, identity, tag_type{});
			}

			/**
			 * @brief One isolated vertex per element of @p values (vertex tier).
			 *        Elements are moved out only of an rvalue range that owns them.
			 * @complexity O(n)
			 */
			template<std::ranges::sized_range R>
				requires(has_value && !has_tags && !has_edge_values && !has_subtree
						 && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
			constexpr link_cut_tree_impl(R &&values, std::type_identity_t<value_type> identity, op_slot op = op_slot{})
				: s_{{}, detail::move_or_copy(op), {}, {}, {}, {}} {
				init_values(std::forward<R>(values), identity, tag_type{});
			}

			/**
			 * @brief @p n isolated vertices holding @p identity, with subtree
			 *        aggregates over the group (op, inverse).
			 * @complexity O(n)
			 */
			constexpr link_cut_tree_impl(std::size_t n, value_type identity, op_slot op, inverse_slot inverse)
				requires has_subtree
				: s_{{}, detail::move_or_copy(op), {}, {}, detail::move_or_copy(inverse), {}} {
				init(n, identity, tag_type{});
			}

			/** @brief One isolated vertex per element of @p values, with subtree aggregates. @complexity O(n) */
			template<std::ranges::sized_range R>
				requires(has_subtree && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
			constexpr link_cut_tree_impl(R &&values, std::type_identity_t<value_type> identity, op_slot op, inverse_slot inverse)
				: s_{{}, detail::move_or_copy(op), {}, {}, detail::move_or_copy(inverse), {}} {
				init_values(std::forward<R>(values), identity, tag_type{});
			}

			/**
			 * @brief @p n isolated vertices holding @p identity, with lazy path
			 *        tags (`compose(f, g)` = "g first, then f", the
			 *        @ref atl::lazy_segment_tree laws).
			 * @complexity O(n)
			 */
			constexpr link_cut_tree_impl(std::size_t n, value_type identity, tag_type tag_identity, op_slot op = op_slot{},
										 mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
				requires has_tags
				: s_{{}, detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), {}, {}} {
				init(n, identity, tag_identity);
			}

			/** @brief One isolated vertex per element of @p values, with lazy path tags. @complexity O(n) */
			template<std::ranges::sized_range R>
				requires(has_tags && !has_edge_values && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
			constexpr link_cut_tree_impl(R &&values, std::type_identity_t<value_type> identity,
										 std::type_identity_t<tag_type> tag_identity, op_slot op = op_slot{},
										 mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
				: s_{{}, detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), {}, {}} {
				init_values(std::forward<R>(values), identity, tag_identity);
			}

			/** @brief As the untagged form, in the commutative-fold mode. */
			constexpr link_cut_tree_impl(commutative_fold_t, std::size_t n, value_type identity, op_slot op = op_slot{})
				requires(is_commutative && !has_tags)
				: link_cut_tree_impl(n, std::move(identity), detail::move_or_copy(op)) {}

			/** @brief As the untagged form, in the commutative-fold mode. */
			template<std::ranges::sized_range R>
				requires(is_commutative && !has_tags && !has_edge_values
						 && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
			constexpr link_cut_tree_impl(commutative_fold_t, R &&values, std::type_identity_t<value_type> identity,
										 op_slot op = op_slot{})
				: link_cut_tree_impl(std::forward<R>(values), std::move(identity), detail::move_or_copy(op)) {}

			/** @brief As the untagged form, in the commutative-fold mode. */
			constexpr link_cut_tree_impl(commutative_fold_t, std::size_t n, value_type identity, tag_type tag_identity,
										 op_slot op = op_slot{}, mapping_slot mapping = mapping_slot{},
										 compose_slot compose = compose_slot{})
				requires(is_commutative && has_tags)
				: link_cut_tree_impl(n, std::move(identity), std::move(tag_identity), detail::move_or_copy(op), detail::move_or_copy(mapping),
									 detail::move_or_copy(compose)) {}

			/** @brief As the untagged form, in the commutative-fold mode. */
			template<std::ranges::sized_range R>
				requires(is_commutative && has_tags && !has_edge_values
						 && std::constructible_from<value_type, detail::forwarded_element_t<R>>)
			constexpr link_cut_tree_impl(commutative_fold_t, R &&values, std::type_identity_t<value_type> identity,
										 std::type_identity_t<tag_type> tag_identity, op_slot op = op_slot{},
										 mapping_slot mapping = mapping_slot{}, compose_slot compose = compose_slot{})
				: link_cut_tree_impl(std::forward<R>(values), std::move(identity), std::move(tag_identity), detail::move_or_copy(op),
									 detail::move_or_copy(mapping), detail::move_or_copy(compose)) {}

			/** @brief As the untagged form, on the edge tier. */
			constexpr link_cut_tree_impl(edge_values_t, std::size_t n, value_type identity, op_slot op = op_slot{})
				requires(has_edge_values && !has_tags)
				: link_cut_tree_impl(n, std::move(identity), detail::move_or_copy(op)) {}

			/** @brief As the untagged form, on the edge tier. */
			constexpr link_cut_tree_impl(edge_values_t, std::size_t n, value_type identity, tag_type tag_identity,
										 op_slot op = op_slot{}, mapping_slot mapping = mapping_slot{},
										 compose_slot compose = compose_slot{})
				requires(has_edge_values && has_tags)
				: link_cut_tree_impl(n, std::move(identity), std::move(tag_identity), detail::move_or_copy(op), detail::move_or_copy(mapping),
									 detail::move_or_copy(compose)) {}

			/** @brief As the untagged form, on the edge tier in the commutative-fold mode. */
			constexpr link_cut_tree_impl(edge_values_t, commutative_fold_t, std::size_t n, value_type identity,
										 op_slot op = op_slot{})
				requires(has_edge_values && is_commutative && !has_tags)
				: link_cut_tree_impl(n, std::move(identity), detail::move_or_copy(op)) {}

			/** @brief As the untagged form, on the edge tier in the commutative-fold mode. */
			constexpr link_cut_tree_impl(edge_values_t, commutative_fold_t, std::size_t n, value_type identity,
										 tag_type tag_identity, op_slot op = op_slot{}, mapping_slot mapping = mapping_slot{},
										 compose_slot compose = compose_slot{})
				requires(has_edge_values && is_commutative && has_tags)
				: link_cut_tree_impl(n, std::move(identity), std::move(tag_identity), detail::move_or_copy(op), detail::move_or_copy(mapping),
									 detail::move_or_copy(compose)) {}

			/** @brief The number of vertices. @complexity O(1) */
			constexpr std::size_t size() const noexcept { return n_; }

			/** @brief Whether there are no vertices. @complexity O(1) */
			constexpr bool empty() const noexcept { return n_ == 0; }

			/**
			 * @brief Make @p v the root of its tree.
			 * @pre `v < size()` (asserted) @complexity amortized O(log n)
			 */
			constexpr void make_root(uint32 v) {
				check_vertex(v);
				evert(node_of(v));
			}

			/**
			 * @brief Add the edge u - v: u's tree hangs below v, so v's root stays
			 *        the root and u's side is re-rooted at u.
			 * @pre `u, v < size()`, u and v in different trees (asserted)
			 * @complexity amortized O(log n)
			 */
			constexpr void link(uint32 u, uint32 v)
				requires(!has_edge_values)
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				evert(xu);
				atl_assert(find_root(xv) != xu); // different trees (u is its tree's root now)
				if constexpr (has_subtree) {
					// xu's whole tree becomes a virtual child of xv: only xv's own aggregates
					// may change, so xv must top its tree (nothing above it to update)
					access(xv);
					s_.exchange_virtual(xv, xu, 0);
					s_.set_parent(xu, xv);
					s_.pull(xv);
				} else {
					s_.set_parent(xu, xv);
				}
			}

			/**
			 * @brief Add the edge u - v with value @p w and return its id, in
			 *        `[0, size() - 1)` (ids of cut edges are reused, the most
			 *        recently freed first); u's tree hangs below v as in the
			 *        vertex tier.
			 * @pre `u, v < size()`, u and v in different trees (asserted)
			 * @complexity amortized O(log n) plus one hash insertion (expected O(1))
			 */
			template<typename EdgeValue>
				requires has_edge_values && std::constructible_from<Value, EdgeValue>
			constexpr uint32 link(uint32 u, uint32 v, EdgeValue &&w) {
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				evert(xu);
				atl_assert(find_root(xv) != xu); // different trees (u is its tree's root now)
				uint32 const e = edges_.free_ids.back();
				edges_.free_ids.pop_back();
				uint32 const y = edge_node(e);
				s_.set_child(y, 0, 0);
				s_.set_child(y, 1, 0);
				s_.set_reversed(y, false);
				if constexpr (IsSized) s_.set_size(y, 1);
				if constexpr (has_tags) s_.tag(y) = s_.tag_identity;
				s_.assign_value(y, Value(std::forward<EdgeValue>(w)));
				s_.set_parent(y, xv);
				s_.set_parent(xu, y);
				edges_.ids.insert(edge_key(u, v), e);
				edges_.endpoints[e] = {u, v};
				return e;
			}

			/**
			 * @brief Remove the edge u - v. The side that held the root keeps it;
			 *        the other side is rooted at its endpoint.
			 * @pre u - v is an edge (asserted)
			 * @complexity amortized O(log n) (plus a hash lookup on the edge tier)
			 */
			constexpr void cut(uint32 u, uint32 v) {
				check_vertex(u);
				check_vertex(v);
				if constexpr (has_edge_values) {
					uint32 const e = edges_.ids.at_or(edge_key(u, v), none);
					atl_assert(e != none); // u - v is an edge
					cut_edge_node(edge_node(e));
				} else {
					uint32 const xu = node_of(u), xv = node_of(v);
					if (detach_from(xu, xv)) return;
					[[maybe_unused]] bool const is_cut = detach_from(xv, xu);
					atl_assert(is_cut); // u - v is an edge
				}
			}

			/**
			 * @brief Remove edge @p e (edge tier); its id becomes free.
			 * @pre @p e is a live edge id (asserted) @complexity amortized O(log n)
			 */
			constexpr void cut(uint32 e)
				requires has_edge_values
			{
				cut_edge_node(check_edge(e));
			}

			/**
			 * @brief Remove the edge from @p v to its parent under the current
			 *        root; v becomes the root of its new tree.
			 * @pre v is not a root (asserted) @complexity amortized O(log n)
			 */
			constexpr void cut_from_parent(uint32 v) {
				check_vertex(v);
				uint32 const l = detach_parent_path(node_of(v));
				if constexpr (has_edge_values) {
					// the cut-off root path ends with v's edge node: drop it
					uint32 const y = splay_rightmost(l);
					if (uint32 const above = s_.child(y, 0); above != 0) {
						s_.set_child(y, 0, 0);
						s_.set_parent(above, 0);
						s_.pull(y);
					}
					release_edge(y);
				}
			}

			/**
			 * @brief Whether u and v are in the same tree.
			 * @pre `u, v < size()` (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr bool connected(uint32 u, uint32 v) {
				check_vertex(u);
				check_vertex(v);
				return nodes_connected(node_of(u), node_of(v));
			}

			/** @brief The root of v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr uint32 root(uint32 v) {
				check_vertex(v);
				return vertex_of(find_root(node_of(v)));
			}

			/** @brief v's parent under the current root, `none` for a root. @complexity amortized O(log n) */
			[[nodiscard]] constexpr uint32 parent(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				uint32 const l = s_.child(x, 0);
				if (l == 0) return none;
				uint32 p = splay_rightmost(l);
				if constexpr (has_edge_values) p = splay_rightmost(s_.child(p, 0)); // past v's edge node
				return vertex_of(p);
			}

			/**
			 * @brief The lowest common ancestor under the current root, `none`
			 *        across trees.
			 * @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 lca(uint32 u, uint32 v) {
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				expose(xu);
				uint32 const w = expose(xv).top;
				return is_exposure_connected(xu, w) ? vertex_of(w) : none;
			}

			/** @brief The number of edges from the root to @p v. @complexity amortized O(log n) */
			[[nodiscard]] constexpr uint32 depth(uint32 v)
				requires IsSized
			{
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				return s_.size(s_.child(x, 0)) / stride;
			}

			/** @brief The number of edges on the u - v path, `none` across trees. @complexity amortized O(log n) */
			[[nodiscard]] constexpr uint32 distance(uint32 u, uint32 v)
				requires IsSized
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				expose(xu);
				auto const [w, below] = expose(xv);
				// below: the path strictly between w and u, whose size the check's splay keeps
				uint32 const u_side = w == xu ? 0 : s_.size(below);
				if (!is_exposure_connected(xu, w)) return none;
				return (u_side + s_.size(s_.child(w, 1))) / stride;
			}

			/**
			 * @brief The k-th ancestor of @p v (`ancestor(v, 0) == v`), `none` past
			 *        the root.
			 * @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 ancestor(uint32 v, uint32 k)
				requires IsSized
			{
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				uint32 const above = s_.size(s_.child(x, 0));
				uint64 const steps = uint64(k) * stride;
				if (steps > above) return none;
				return vertex_of(Engine::splay_at(s_, x, uint32(above - steps)));
			}

			/**
			 * @brief The k-th vertex on the u -> v path (`path_vertex(u, v, 0) == u`),
			 *        `none` past v or across trees.
			 * @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr uint32 path_vertex(uint32 u, uint32 v, uint32 k)
				requires IsSized
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				expose(xu);
				uint32 const w = expose(xv).top;
				if (!is_exposure_connected(xu, w)) return none;
				// after the check's splay, xu roots the u side (when it is not w)
				uint32 const u_side = w == xu ? 0 : s_.size(xu);
				uint32 const v_side = s_.size(s_.child(w, 1));
				uint64 const position = uint64(k) * stride;
				if (position < u_side) return vertex_of(Engine::splay_at(s_, xu, uint32(u_side - 1 - position)));
				if (position == u_side) return vertex_of(w);
				if (position > uint64(u_side) + v_side) return none;
				return vertex_of(Engine::splay_at(s_, s_.child(w, 1), uint32(position - u_side - 1)));
			}

			/**
			 * @brief The value of vertex @p id (edge @p id on the edge tier).
			 *        O(1) without tags; with tags, by value after settling the tags
			 *        above it (amortized O(log n)).
			 */
			[[nodiscard]] constexpr value_type const &at(uint32 id)
				requires(has_value && !has_tags)
			{
				return s_.value(value_node(id));
			}

			/** @copydoc at */
			[[nodiscard]] constexpr value_type at(uint32 id)
				requires has_tags
			{
				uint32 const x = value_node(id);
				Engine::splay(s_, x);
				return s_.value(x);
			}

			/**
			 * @brief Replace the value of vertex @p id (edge @p id on the edge tier).
			 * @complexity amortized O(log n)
			 */
			template<typename NewValue>
				requires has_value && std::assignable_from<Value &, NewValue>
			constexpr void set(uint32 id, NewValue &&x) {
				uint32 const node = value_node(id);
				raise_for_update(node);
				s_.value(node) = std::forward<NewValue>(x);
				s_.pull(node);
			}

			/**
			 * @brief Replace the value `a` of vertex @p id (edge @p id on the edge
			 *        tier) by `f(a)`.
			 * @complexity amortized O(log n)
			 */
			template<update_function<value_type> Update>
				requires has_value
			constexpr void update(uint32 id, Update &&f) {
				uint32 const node = value_node(id);
				raise_for_update(node);
				s_.value(node) = std::invoke(f, std::as_const(s_.value(node)));
				s_.pull(node);
			}

			/**
			 * @brief The fold of the u - v path in path order from u to v,
			 *        endpoints included: vertex values, or on the edge tier the
			 *        edge values (the identity when u == v). The root is unchanged.
			 * @pre u and v connected (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr value_type fold(uint32 u, uint32 v)
				requires has_value
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				atl_assert(nodes_connected(xu, xv));
				if constexpr (IsEvertForm) {
					uint32 const saved_root = find_root(xu);
					evert(xu);
					access(xv);
					value_type result = s_.aggregate(xv);
					evert(saved_root);
					return result;
				} else {
					expose(xu);
					auto const [w, below] = expose(xv);
					// below runs top-down from w's child to u: read backwards; node 0 folds as the identity
					return s_.op(s_.op(s_.backward_aggregate(below), s_.value(w)), s_.aggregate(s_.child(w, 1)));
				}
			}

			/**
			 * @brief `fold` when u and v are connected, `std::nullopt`
			 *        otherwise: one pair of exposures answers both, where
			 *        `connected` then `fold` costs two pairs.
			 * @pre `u, v < size()` (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr std::optional<value_type> try_fold(uint32 u, uint32 v)
				requires has_value
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				if constexpr (IsEvertForm) {
					if (!nodes_connected(xu, xv)) return std::nullopt;
					return fold(u, v);
				} else {
					expose(xu);
					uint32 const w = expose(xv).top;
					if (!is_exposure_connected(xu, w)) return std::nullopt;
					// after the check's splay, xu roots the path strictly below w down to u (empty when u is w)
					uint32 const below = w == xu ? 0 : xu;
					return s_.op(s_.op(s_.backward_aggregate(below), s_.value(w)), s_.aggregate(s_.child(w, 1)));
				}
			}

			/**
			 * @brief Apply tag @p f to every vertex value on the u - v path (edge
			 *        values on the edge tier). The root is unchanged.
			 * @pre u and v connected (asserted) @complexity amortized O(log n)
			 */
			constexpr void apply(uint32 u, uint32 v, tag_type const &f)
				requires has_tags
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				atl_assert(nodes_connected(xu, xv));
				if constexpr (IsEvertForm) {
					uint32 const saved_root = find_root(xu);
					evert(xu);
					access(xv);
					s_.apply_tag(xv, f);
					evert(saved_root);
				} else {
					expose(xu);
					auto const [w, below] = expose(xv);
					if (below != 0) s_.apply_tag(below, f);
					s_.value(w) = s_.mapping(f, s_.value(w));
					if (uint32 const r = s_.child(w, 1); r != 0) s_.apply_tag(r, f);
					s_.pull(w);
				}
			}

			/**
			 * @brief Link every vertex v with `parents[v] != none` (and `!= v`)
			 *        below `parents[v]` (a parent-array format),
			 *        rooting each tree at its parentless vertex.
			 * @pre every vertex isolated; `parents` has size() entries and forms a
			 *      forest (all asserted) @complexity O(n)
			 * @tparam Parents A forward vertex range iterable through const.
			 */
			template<typename Parents>
				requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const> && (!has_edge_values)
			constexpr void assign_parents(Parents const &parents) {
				atl_assert(is_all_isolated());
				atl_assert(is_forest(parents));
				uint32 v = 0;
				for (auto const p : parents) {
					atl_assert(v < n_);
					if (uint32(p) != none && uint32(p) != v) s_.set_parent(node_of(v), node_of(uint32(p)));
					++v;
				}
				atl_assert(v == n_);
				if constexpr (has_subtree) fold_virtual_children(parents);
			}

			/**
			 * @brief Edge tier: as above, the edge v - parents[v] holding
			 *        `weights[v]` (entries of roots ignored). Edge ids are
			 *        assigned 0, 1, ... in increasing v.
			 * @pre no edges; parents has size() entries forming a forest, and
			 *      weights has at least size() entries (asserted); further weights are ignored
			 * @complexity O(n) plus n hash insertions
			 * @tparam Parents A forward vertex range iterable through const.
			 */
			template<typename Parents, std::ranges::input_range Weights>
				requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const> && has_edge_values && std::constructible_from<Value, std::ranges::range_reference_t<Weights>>
			constexpr void assign_parents(Parents const &parents, Weights &&weights) {
				atl_assert(is_all_isolated());
				atl_assert(is_forest(parents));
				reset_free_ids();
				auto weight = std::ranges::begin(weights);
				uint32 v = 0;
				for (auto const p : parents) {
					atl_assert(v < n_ && weight != std::ranges::end(weights));
					if (uint32(p) != none && uint32(p) != v) {
						uint32 const e = edges_.free_ids.back();
						edges_.free_ids.pop_back();
						uint32 const y = edge_node(e);
						s_.assign_value(y, Value(*weight));
						s_.set_parent(node_of(v), y);
						s_.set_parent(y, node_of(uint32(p)));
						edges_.ids.insert(edge_key(v, uint32(p)), e);
						edges_.endpoints[e] = {v, uint32(p)};
					}
					++weight;
					++v;
				}
				atl_assert(v == n_);
			}

			/**
			 * @brief The id of edge u - v (edge tier), `none` when there is none.
			 * @complexity expected O(1)
			 */
			[[nodiscard]] constexpr uint32 edge(uint32 u, uint32 v) const
				requires has_edge_values
			{
				check_vertex(u);
				check_vertex(v);
				return edges_.ids.at_or(edge_key(u, v), none);
			}

			/** @brief The number of edges (edge tier). @complexity O(1) */
			[[nodiscard]] constexpr std::size_t edge_count() const noexcept
				requires has_edge_values
			{
				return edges_.ids.size();
			}
		};
	}

	/**
	 * @brief Link-cut tree: a forest under link, cut and re-rooting, with
	 *        connectivity, rooted queries (lca, parent, depth, ancestor,
	 *        distance, path_vertex) and path folds and lazy path updates over a
	 *        monoid, each amortized O(log n). A client of the public
	 *        @ref atl::splay_engine.
	 *
	 * Tiers are switched on by non-`void` template arguments, in the order of
	 * @ref atl::treap_sequence and with the @ref atl::lazy_segment_tree names: structure
	 * only; `Value` + `Op` (vertex values, path folds); + `Tag`, `Mapping`,
	 * `Compose` (lazy path updates). `FoldOrder = commutative_fold_t` drops the
	 * reversed-order aggregate when the fold commutes (sum, min, max, xor). `ValuePlacement =
	 * edge_values_t` is the edge tier: values on edges (the edge-as-vertex
	 * trick, built in), `link(u, v, w)` returning an edge id, folds over the
	 * path's edges, `cut(u, v)` or `cut(edge)`.
	 *
	 * The root of each tree changes only by `make_root`, `link` (u's side joins
	 * below v's root) and `cut` (the detached side is rooted at its endpoint):
	 * path folds and updates keep it, so rooted and unrooted problems share the
	 * class. Structural and path queries mutate the auxiliary trees;
	 * untagged value reads and edge lookups are const. Vertex ids are `uint32` in
	 * `[0, size())`; `none` answers "no vertex".
	 *
	 * @tparam Value          Default-initializable when non-void. Vertex (or edge) value; `void` for structure only.
	 * @tparam Op             The fold, associative, with a two-sided identity.
	 * @tparam Tag            Default-initializable when non-void. Path-update tag (needs `Value`).
	 * @tparam Mapping        `mapping(f, a)`, distributing over `Op`.
	 * @tparam Compose        `compose(f, g)` = "g first, then f"; tag_identity is its two-sided identity.
	 * @tparam FoldOrder      `void` or @ref atl::commutative_fold_t.
	 * @tparam ValuePlacement `void` (vertex values) or `atl::edge_values_t`.
	 */
	template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
			 typename FoldOrder = void, typename ValuePlacement = void>
		requires detail::link_cut_algebra<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
	class link_cut_tree : public detail::link_cut_tree_impl<splay_engine, detail::splay_layout_interleaved, true, false, Value, Op,
															Tag, Mapping, Compose, FoldOrder, ValuePlacement> {
		using base = detail::link_cut_tree_impl<splay_engine, detail::splay_layout_interleaved, true, false, Value, Op, Tag,
												Mapping, Compose, FoldOrder, ValuePlacement>;

	public:
		using base::base;
	};

	/** @brief The edge tier spelled as a type: @ref atl::link_cut_tree with `atl::edge_values_t`. */
	template<typename Value, typename Op, typename Tag = void, typename Mapping = void, typename Compose = void,
			 typename FoldOrder = void>
	using edge_link_cut_tree = link_cut_tree<Value, Op, Tag, Mapping, Compose, FoldOrder, edge_values_t>;

	/** @brief Structure only: the empty forest or @p n isolated vertices. */
	link_cut_tree() -> link_cut_tree<>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	link_cut_tree(std::size_t) -> link_cut_tree<>;
	/** @brief Vertex values: `Value` from the identity. */
	template<std::copyable Value, binary_operation<Value> Op>
	link_cut_tree(std::size_t, Value, Op) -> link_cut_tree<Value, Op>;
	/** @brief `Value` from the identity; the elements must convert to it without narrowing. */
	template<std::ranges::sized_range R, std::copyable Value, binary_operation<Value> Op>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	link_cut_tree(R &&, Value, Op) -> link_cut_tree<Value, Op>;
	/** @brief Lazy path updates: `Value` and `Tag` from the identities. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	link_cut_tree(std::size_t, Value, Tag, Op, Mapping, Compose) -> link_cut_tree<Value, Op, Tag, Mapping, Compose>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op,
			 action<Tag, Value> Mapping, binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	link_cut_tree(R &&, Value, Tag, Op, Mapping, Compose) -> link_cut_tree<Value, Op, Tag, Mapping, Compose>;

	/** @brief The commutative-fold mode, deduced from the leading @ref atl::commutative_fold argument. */
	template<std::copyable Value, binary_operation<Value> Op>
	link_cut_tree(commutative_fold_t, std::size_t, Value, Op) -> link_cut_tree<Value, Op, void, void, void, commutative_fold_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Value, binary_operation<Value> Op>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	link_cut_tree(commutative_fold_t, R &&, Value, Op) -> link_cut_tree<Value, Op, void, void, void, commutative_fold_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	link_cut_tree(commutative_fold_t, std::size_t, Value, Tag, Op, Mapping, Compose)
		-> link_cut_tree<Value, Op, Tag, Mapping, Compose, commutative_fold_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op,
			 action<Tag, Value> Mapping, binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	link_cut_tree(commutative_fold_t, R &&, Value, Tag, Op, Mapping, Compose)
		-> link_cut_tree<Value, Op, Tag, Mapping, Compose, commutative_fold_t>;

	/** @brief The edge tier, deduced from the leading `atl::edge_values` argument. */
	template<std::copyable Value, binary_operation<Value> Op>
	link_cut_tree(edge_values_t, std::size_t, Value, Op) -> link_cut_tree<Value, Op, void, void, void, void, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	link_cut_tree(edge_values_t, std::size_t, Value, Tag, Op, Mapping, Compose)
		-> link_cut_tree<Value, Op, Tag, Mapping, Compose, void, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, binary_operation<Value> Op>
	link_cut_tree(edge_values_t, commutative_fold_t, std::size_t, Value, Op)
		-> link_cut_tree<Value, Op, void, void, void, commutative_fold_t, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
	link_cut_tree(edge_values_t, commutative_fold_t, std::size_t, Value, Tag, Op, Mapping, Compose)
		-> link_cut_tree<Value, Op, Tag, Mapping, Compose, commutative_fold_t, edge_values_t>;
}
