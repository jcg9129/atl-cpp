#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/dynamic_forest.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/ds/top_clusters.hpp>
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

namespace atl {
	namespace detail {
		/**
		 * @brief The `Reverse` slot of @ref atl::top_tree : `void` (the tree keeps a
		 *        reversed path cluster per node), @ref atl::commutative_fold_t (path
		 *        clusters are symmetric: nothing to reverse), or a callable
		 *        `Path(Path const &)` swapping a cluster's ends in O(1).
		 */
		template<typename Reverse, typename Path>
		concept top_path_reversal = std::is_void_v<Reverse> || std::same_as<Reverse, commutative_fold_t>
								 || (std::regular_invocable<Reverse const &, Path const &>
									 && std::convertible_to<std::invoke_result_t<Reverse const &, Path const &>, Path>);

		/** @brief Whether the `Reverse` slot is a user callable (neither `void` nor @ref atl::commutative_fold_t). */
		template<typename Reverse>
		inline constexpr bool is_top_reverse_callable = !std::is_void_v<Reverse> && !std::same_as<Reverse, commutative_fold_t>;

		/**
		 * @brief The template-argument contract of @ref atl::top_tree : the cluster
		 *        algebra (two merges, two conversions, the orientation slot),
		 *        then the lazy tier switched on as a whole by a non-`void` `Tag`.
		 */
		template<typename Value, typename Path, typename Point, typename Compress, typename Rake, typename AddVertex,
				 typename AddEdge, typename Reverse, typename Tag, typename Mapping, typename MapPath, typename MapPoint,
				 typename Compose, typename ValuePlacement>
		concept top_tree_algebra = std::copyable<Value> && std::copyable<Path> && std::copyable<Point>
								&& binary_operation<Compress, Path> && binary_operation<Rake, Point>
								&& top_vertex_operation<AddVertex, Point, Value, Path> && top_edge_operation<AddEdge, Path, Point>
								&& top_path_reversal<Reverse, Path> && (std::is_void_v<Tag> == std::is_void_v<Mapping>)
								&& (std::is_void_v<Tag> == std::is_void_v<MapPath>) && (std::is_void_v<Tag> == std::is_void_v<MapPoint>)
								&& (std::is_void_v<Tag> == std::is_void_v<Compose>)
								&& (std::is_void_v<Tag>
									|| (std::copyable<Tag> && action<Mapping, Tag, Value> && path_action<MapPath, Tag, Path>
										&& action<MapPoint, Tag, Point> && binary_operation<Compose, Tag>))
								&& value_placement<ValuePlacement>;

		/** @brief The id bit marking a rake node: compress ids are below it, so one test tells the kinds apart. */
		inline constexpr uint32 top_rake_bit = uint32(1) << 31;

		/** @brief An absent node field or algebra slot (distinct per @p Id, so absent fields overlap). */
		template<int Id>
		struct top_no_field {};

		/**
		 * @brief The node pools of a @ref atl::top_tree and the two
		 *        @ref atl::splay_store views the engine drives over them.
		 *
		 * **Compress nodes** (ids `[1, 2^31)`), one per vertex (and per edge on
		 * the edge tier): a splay tree of them is a preferred path, in order
		 * from top to bottom. Per node: links, `mid` (the root of its rake
		 * tree, 0 when it has no light child), the splay subtree's node count,
		 * the value, the path cluster of the splay subtree (its spine segment
		 * and everything hanging off it), the reversed cluster when `Reverse`
		 * is `void`, a pending reversal mark and the pending (spine, light)
		 * tag pair. **Rake nodes** (ids with `top_rake_bit` set), one per
		 * light child: a splay tree of them holds a vertex's light children,
		 * each rake node's `mid` the compress root of one child's path. Per
		 * node: links, `mid`, the point cluster of the rake subtree and a
		 * pending tag.
		 *
		 * A compress root's parent is the rake node holding it (or 0 at a
		 * tree's root path), a rake root's parent the compress node owning the
		 * rake tree: so `is_splay_root` is one bit test, and the engine's
		 * rotation keeps that parent at the new root; the client repoints the
		 * owner's `mid`. As in @ref atl::link_cut_tree, marks and tags at a node are
		 * already reflected in its own clusters and owed below (its children
		 * and its `mid`). Index 0 of each pool is the null sentinel holding the
		 * identity cluster, so pulls fold through empty sides without
		 * branching (a rake id masks to its index; 0 masks to 0).
		 */
		template<splay_layout Layout, bool IsOwnCached, typename Value, typename Path, typename Point, typename Compress,
				 typename Rake, typename AddVertex, typename AddEdge, typename Reverse, typename Tag, typename Mapping,
				 typename MapPath, typename MapPoint, typename Compose>
			requires top_tree_algebra<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
									  MapPoint, Compose, void>
		struct top_pool {
			static constexpr bool has_tags = !std::is_void_v<Tag>;
			static constexpr bool has_reversed_path = std::is_void_v<Reverse>;
			static constexpr bool has_reverse_callable = is_top_reverse_callable<Reverse>;
			static constexpr bool is_interleaved = std::same_as<Layout, splay_layout_interleaved>;
			static constexpr uint32 rake_mask = ~top_rake_bit;

			using reversed_field = std::conditional_t<has_reversed_path, Path, top_no_field<0>>;
			using own_field = std::conditional_t<IsOwnCached, Path, top_no_field<9>>;
			using tag_field = std::conditional_t<has_tags, Tag, top_no_field<1>>;
			using light_tag_field = std::conditional_t<has_tags, Tag, top_no_field<2>>;
			using rake_tag_field = std::conditional_t<has_tags, Tag, top_no_field<3>>;

			/** @brief An optional algebra member: the callable, or an empty placeholder. */
			template<typename Algebra, int Id>
			using slot = std::conditional_t<std::is_void_v<Algebra> || std::same_as<Algebra, commutative_fold_t>, top_no_field<Id>,
											Algebra>;

			using reverse_slot = slot<Reverse, 4>;
			using mapping_slot = slot<Mapping, 5>;
			using map_path_slot = slot<MapPath, 6>;
			using map_point_slot = slot<MapPoint, 7>;
			using compose_slot = slot<Compose, 8>;

			/** @brief A compress node's links, rake-tree root and splay subtree size. */
			struct compress_link {
				uint32 child[2]{};
				uint32 parent = 0;
				uint32 mid = 0;	 ///< the root of the rake tree of light children, 0 when none
				uint32 size = 0; ///< compress nodes in the splay subtree
			};

			/** @brief A compress node's value, clusters, mark and tags. */
			struct compress_payload {
				Value value;
				Path path;											///< the splay subtree's cluster, top to bottom
				[[no_unique_address]] reversed_field reversed_path; ///< the same, bottom to top
				[[no_unique_address]] own_field own;				///< cached `add_vertex(point(mid), value)` (IsOwnCached)
				[[no_unique_address]] tag_field spine_tag;			///< owed to the children's spines
				[[no_unique_address]] light_tag_field light_tag;	///< owed to everything hanging off them
				bool is_reversed = false;							///< the children are owed a swap
			};

			/** @brief The interleaved layout's compress record. */
			struct compress_node : compress_link, compress_payload {};

			/** @brief A rake node's links and the compress root it holds. */
			struct rake_link {
				uint32 child[2]{};
				uint32 parent = 0;
				uint32 mid = 0; ///< the compress root of the light child's path
			};

			/** @brief A rake node's cluster and tag. */
			struct rake_payload {
				Point point;							  ///< the rake subtree's hanging set
				[[no_unique_address]] rake_tag_field tag; ///< owed to the children and to mid
			};

			/** @brief The interleaved layout's rake record. */
			struct rake_node : rake_link, rake_payload {};

			/** @brief Interleaved layout: one record per node. */
			struct interleaved_storage {
				std::vector<compress_node> compress;
				std::vector<rake_node> rake;
			};

			/** @brief Split layout: links apart from payloads. */
			struct split_storage {
				std::vector<compress_link> compress_links;
				std::vector<compress_payload> compress_payloads;
				std::vector<rake_link> rake_links;
				std::vector<rake_payload> rake_payloads;
			};

			std::conditional_t<is_interleaved, interleaved_storage, split_storage> store;
			std::vector<uint32> free_rakes; ///< recycled rake indices
			Compress compress;
			Rake rake;
			AddVertex add_vertex;
			AddEdge add_edge;
			[[no_unique_address]] reverse_slot reverse;
			[[no_unique_address]] mapping_slot mapping;
			[[no_unique_address]] map_path_slot map_path;
			[[no_unique_address]] map_point_slot map_point;
			[[no_unique_address]] compose_slot compose;
			[[no_unique_address]] tag_field tag_identity;

			/** @brief Install the algebra; the pools stay empty. */
			constexpr top_pool(Compress compress_op, Rake rake_op, AddVertex add_vertex_op, AddEdge add_edge_op,
							   reverse_slot reverse_op, mapping_slot mapping_op, map_path_slot map_path_op,
							   map_point_slot map_point_op, compose_slot compose_op, tag_field tag_identity_value)
				: compress(detail::move_or_copy(compress_op)), rake(detail::move_or_copy(rake_op)), add_vertex(detail::move_or_copy(add_vertex_op)),
				  add_edge(detail::move_or_copy(add_edge_op)), reverse(detail::move_or_copy(reverse_op)), mapping(detail::move_or_copy(mapping_op)),
				  map_path(detail::move_or_copy(map_path_op)), map_point(detail::move_or_copy(map_point_op)), compose(detail::move_or_copy(compose_op)),
				  tag_identity(std::move(tag_identity_value)) {}

			/** @brief Compress node @p x's links (0 is the sentinel). */
			constexpr compress_link &cl(uint32 x) {
				if constexpr (is_interleaved)
					return store.compress[x];
				else
					return store.compress_links[x];
			}

			/** @copydoc cl(uint32) */
			constexpr compress_link const &cl(uint32 x) const {
				if constexpr (is_interleaved)
					return store.compress[x];
				else
					return store.compress_links[x];
			}

			/** @brief Compress node @p x's payload (0 holds the identity clusters). */
			constexpr compress_payload &cp(uint32 x) {
				if constexpr (is_interleaved)
					return store.compress[x];
				else
					return store.compress_payloads[x];
			}

			/** @copydoc cp(uint32) */
			constexpr compress_payload const &cp(uint32 x) const {
				if constexpr (is_interleaved)
					return store.compress[x];
				else
					return store.compress_payloads[x];
			}

			/** @brief Rake node @p y's links (@p y with or without the rake bit; 0 is the sentinel). */
			constexpr rake_link &rl(uint32 y) {
				if constexpr (is_interleaved)
					return store.rake[y & rake_mask];
				else
					return store.rake_links[y & rake_mask];
			}

			constexpr rake_link const &rl(uint32 y) const {
				if constexpr (is_interleaved)
					return store.rake[y & rake_mask];
				else
					return store.rake_links[y & rake_mask];
			}

			constexpr rake_payload &rp(uint32 y) {
				if constexpr (is_interleaved)
					return store.rake[y & rake_mask];
				else
					return store.rake_payloads[y & rake_mask];
			}

			constexpr rake_payload const &rp(uint32 y) const {
				if constexpr (is_interleaved)
					return store.rake[y & rake_mask];
				else
					return store.rake_payloads[y & rake_mask];
			}

			/** @brief The number of compress slots, the sentinel included. */
			[[nodiscard]] constexpr std::size_t compress_slots() const {
				if constexpr (is_interleaved)
					return store.compress.size();
				else
					return store.compress_links.size();
			}

			/** @brief The light-tag field's identity (a placeholder below the lazy tier). */
			[[nodiscard]] constexpr light_tag_field light_tag_identity() const {
				if constexpr (has_tags)
					return tag_identity;
				else
					return {};
			}

			/** @brief The rake-tag field's identity (a placeholder below the lazy tier). */
			[[nodiscard]] constexpr rake_tag_field rake_tag_identity() const {
				if constexpr (has_tags)
					return tag_identity;
				else
					return {};
			}

			/** @brief A payload holding @p value with cluster @p path (both orientations), no mark, identity tags. */
			[[nodiscard]] constexpr compress_payload make_payload(Value value, Path const &path) const {
				reversed_field reversed{};
				if constexpr (has_reversed_path) reversed = path;
				own_field own{};
				if constexpr (IsOwnCached) own = path;
				return {std::move(value), path, std::move(reversed), std::move(own), tag_identity, light_tag_identity(), false};
			}

			/** @brief Append a compress record (links @p link, payload @p payload). */
			constexpr void push_compress(compress_link const &link, compress_payload payload) {
				if constexpr (is_interleaved) {
					store.compress.push_back(compress_node{link, std::move(payload)});
				} else {
					store.compress_links.push_back(link);
					store.compress_payloads.push_back(std::move(payload));
				}
			}

			/** @brief Append a rake record. */
			constexpr void push_rake(rake_link const &link, rake_payload payload) {
				if constexpr (is_interleaved) {
					store.rake.push_back(rake_node{link, std::move(payload)});
				} else {
					store.rake_links.push_back(link);
					store.rake_payloads.push_back(std::move(payload));
				}
			}

			/** @brief Empty both pools (no sentinels) and reserve room for @p compress_nodes nodes and their sentinel. */
			constexpr void clear(std::size_t compress_nodes) {
				if constexpr (is_interleaved) {
					store.compress.clear();
					store.rake.clear();
					store.compress.reserve(compress_nodes + 1);
				} else {
					store.compress_links.clear();
					store.compress_payloads.clear();
					store.rake_links.clear();
					store.rake_payloads.clear();
					store.compress_links.reserve(compress_nodes + 1);
					store.compress_payloads.reserve(compress_nodes + 1);
				}
				free_rakes.clear();
			}

			/** @brief A fresh rake node (links 0, identity tag; the caller sets `mid` and pulls) and its id. */
			constexpr uint32 new_rake() {
				if (!free_rakes.empty()) {
					uint32 const y = free_rakes.back() | top_rake_bit;
					free_rakes.pop_back();
					rl(y) = rake_link{};
					if constexpr (has_tags) rp(y).tag = tag_identity;
					return y;
				}
				Point identity = rp(0).point;
				std::size_t const index = [this] {
					if constexpr (is_interleaved)
						return store.rake.size();
					else
						return store.rake_links.size();
				}();
				push_rake(rake_link{}, rake_payload{std::move(identity), rake_tag_identity()});
				return uint32(index) | top_rake_bit; // fewer rake than compress nodes: below the bit
			}

			/** @brief Recycle rake node @p y. */
			constexpr void free_rake(uint32 y) { free_rakes.push_back(y & rake_mask); }

			/** @brief Compress node x as a one-vertex cluster: its value over its light children. */
			[[nodiscard]] constexpr Path own(uint32 x) const {
				if constexpr (IsOwnCached)
					return cp(x).own;
				else
					return add_vertex(rp(cl(x).mid).point, cp(x).value);
			}

			/** @brief Recompute x's cached own cluster after its value or its rake root changed (no-op uncached). */
			constexpr void refresh_own(uint32 x) {
				if constexpr (IsOwnCached) cp(x).own = add_vertex(rp(cl(x).mid).point, cp(x).value);
			}

			/** @brief Size and clusters of compress node x from its children and its rake tree. */
			constexpr void cpull(uint32 x) {
				compress_link &lx = cl(x);
				uint32 const l = lx.child[0], r = lx.child[1];
				lx.size = cl(l).size + 1 + cl(r).size;
				compress_payload &px = cp(x);
				if constexpr (IsOwnCached) {
					px.path = compress(compress(cp(l).path, px.own), cp(r).path);
					if constexpr (has_reversed_path)
						px.reversed_path = compress(compress(cp(r).reversed_path, px.own), cp(l).reversed_path);
				} else {
					Path const o = own(x);
					px.path = compress(compress(cp(l).path, o), cp(r).path);
					if constexpr (has_reversed_path) px.reversed_path = compress(compress(cp(r).reversed_path, o), cp(l).reversed_path);
				}
			}

			/** @brief The hanging set of rake node y's subtree. */
			constexpr void rpull(uint32 y) {
				rake_link const &ly = rl(y);
				rp(y).point = rake(rake(rp(ly.child[0]).point, add_edge(cp(ly.mid).path)), rp(ly.child[1]).point);
			}

			/** @brief Mark compress subtree x reversed: its clusters swap ends now, the children's stay owed. */
			constexpr void cflip(uint32 x) {
				compress_payload &px = cp(x);
				px.is_reversed = !px.is_reversed;
				if constexpr (has_reversed_path)
					std::ranges::swap(px.path, px.reversed_path);
				else if constexpr (has_reverse_callable)
					px.path = reverse(std::as_const(px.path));
			}

			/** @brief Tag compress subtree x: @p spine on its spine, @p light on everything hanging off it. */
			constexpr void capply(uint32 x, tag_field const &spine, tag_field const &light)
				requires has_tags
			{
				compress_payload &px = cp(x);
				px.value = mapping(spine, px.value);
				px.path = map_path(spine, light, px.path);
				if constexpr (IsOwnCached) px.own = map_path(spine, light, px.own);
				if constexpr (has_reversed_path) px.reversed_path = map_path(spine, light, px.reversed_path);
				px.spine_tag = compose(spine, px.spine_tag);
				px.light_tag = compose(light, px.light_tag);
			}

			/** @brief Tag every vertex below rake subtree y. */
			constexpr void rapply(uint32 y, tag_field const &f)
				requires has_tags
			{
				rake_payload &py = rp(y);
				py.point = map_point(f, py.point);
				py.tag = compose(f, py.tag);
			}

			/** @brief Settle compress node x's mark and tags one level down (children and rake tree). */
			constexpr void cpush(uint32 x) {
				compress_link &lx = cl(x);
				compress_payload &px = cp(x);
				if (px.is_reversed) {
					std::ranges::swap(lx.child[0], lx.child[1]);
					if (lx.child[0] != 0) cflip(lx.child[0]);
					if (lx.child[1] != 0) cflip(lx.child[1]);
					px.is_reversed = false;
				}
				if constexpr (has_tags) {
					if constexpr (std::equality_comparable<Tag>)
						if (px.spine_tag == tag_identity && px.light_tag == tag_identity) return;
					if (lx.child[0] != 0) capply(lx.child[0], px.spine_tag, px.light_tag);
					if (lx.child[1] != 0) capply(lx.child[1], px.spine_tag, px.light_tag);
					if (lx.mid != 0) rapply(lx.mid, px.light_tag);
					px.spine_tag = tag_identity;
					px.light_tag = tag_identity;
				}
			}

			/** @brief Settle rake node y's tag one level down: its children and its child's whole path. */
			constexpr void rpush(uint32 y)
				requires has_tags
			{
				rake_payload &py = rp(y);
				if constexpr (std::equality_comparable<Tag>)
					if (py.tag == tag_identity) return;
				rake_link const &ly = rl(y);
				if (ly.child[0] != 0) rapply(ly.child[0], py.tag);
				if (ly.child[1] != 0) rapply(ly.child[1], py.tag);
				capply(ly.mid, py.tag, py.tag);
				py.tag = tag_identity;
			}

			/** @brief The compress splay trees as a @ref atl::sized_splay_store (with push, pull and the root test). */
			struct compress_view {
				top_pool *pool;

				/** @brief Engine hook: the child on @p side. */
				[[nodiscard]] constexpr uint32 child(uint32 x, bool side) const { return pool->cl(x).child[side]; }

				/** @brief Engine hook: the parent (a holder or owner at a splay root). */
				[[nodiscard]] constexpr uint32 parent(uint32 x) const { return pool->cl(x).parent; }

				/** @brief Engine hook: set one child link. */
				constexpr void set_child(uint32 x, bool side, uint32 c) { pool->cl(x).child[side] = c; }

				/** @brief Engine hook: set the parent link. */
				constexpr void set_parent(uint32 x, uint32 p) { pool->cl(x).parent = p; }

				/** @brief Engine hook: the splay subtree size. */
				[[nodiscard]] constexpr uint32 size(uint32 x) const { return pool->cl(x).size; }

				/** @brief Engine hook: settle pending work one level down. */
				constexpr void push(uint32 x) { pool->cpush(x); }

				/** @brief Engine hook: recompute from the children. */
				constexpr void pull(uint32 x) { pool->cpull(x); }

				/** @brief A compress root hangs from a rake node or from nothing. */
				[[nodiscard]] constexpr bool is_splay_root(uint32 x) const {
					uint32 const p = pool->cl(x).parent;
					return p == 0 || (p & top_rake_bit) != 0;
				}
			};

			/** @brief The rake splay trees as a @ref atl::splay_store (push only on the lazy tier). */
			struct rake_view {
				top_pool *pool;

				/** @brief Engine hook: the child on @p side. */
				[[nodiscard]] constexpr uint32 child(uint32 y, bool side) const { return pool->rl(y).child[side]; }

				/** @brief Engine hook: the parent (a holder or owner at a splay root). */
				[[nodiscard]] constexpr uint32 parent(uint32 y) const { return pool->rl(y).parent; }

				/** @brief Engine hook: set one child link. */
				constexpr void set_child(uint32 y, bool side, uint32 c) { pool->rl(y).child[side] = c; }

				/** @brief Engine hook: set the parent link. */
				constexpr void set_parent(uint32 y, uint32 p) { pool->rl(y).parent = p; }

				/** @brief Engine hook: settle pending work one level down. */
				constexpr void push(uint32 y)
					requires has_tags
				{
					pool->rpush(y);
				}

				/** @brief Engine hook: recompute from the children. */
				constexpr void pull(uint32 y) { pool->rpull(y); }

				/** @brief A rake root hangs from its owning compress node. */
				[[nodiscard]] constexpr bool is_splay_root(uint32 y) const { return (pool->rl(y).parent & top_rake_bit) == 0; }
			};
		};

		/**
		 * @brief The named members of the ready-made cluster sets
		 *        (@ref atl::top_clusters), each switched on by its set's path type
		 *        and written once over the generic members of @p Derived.
		 */
		template<typename Derived, typename Value, typename Path, typename Tag>
			requires std::copyable<Value> && std::copyable<Path>
		class top_named_members {
			constexpr Derived &self() { return static_cast<Derived &>(*this); }

			/** @brief The tag adding @p x (the summing sets). */
			static constexpr Tag add_tag(Value const &x)
				requires top_clusters::detail::sum_family<Path>
			{
				return Tag::add_of(x);
			}

		protected:
			top_named_members() = default;

		public:
			/** @brief The sum of the vertex values on the u - v path. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value path_sum(uint32 u, uint32 v)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().fold(u, v).spine.sum;
			}

			/** @brief The sum over v's subtree under the current root. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_sum(uint32 v)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().subtree_fold(v).total().sum;
			}

			/** @brief The sum over v's side of the edge {v, p}. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_sum(uint32 v, uint32 p)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().subtree_fold(v, p).total().sum;
			}

			/** @brief The sum over v's whole tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value component_sum(uint32 v)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().component_fold(v).total().sum;
			}

			/** @brief The number of vertices in v's subtree under the current root. @complexity amortized O(log n) */
			[[nodiscard]] constexpr std::size_t subtree_size(uint32 v)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().subtree_fold(v).total().count;
			}

			/** @brief The number of vertices on v's side of the edge {v, p}. @complexity amortized O(log n) */
			[[nodiscard]] constexpr std::size_t subtree_size(uint32 v, uint32 p)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().subtree_fold(v, p).total().count;
			}

			/** @brief The number of vertices in v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr std::size_t component_size(uint32 v)
				requires top_clusters::detail::sum_family<Path>
			{
				return self().component_fold(v).total().count;
			}

			/** @brief Add @p x to every vertex on the u - v path. @complexity amortized O(log n) */
			constexpr void path_add(uint32 u, uint32 v, Value const &x)
				requires top_clusters::detail::sum_family<Path>
			{
				self().apply(u, v, add_tag(x));
			}

			/** @brief Add @p x to every vertex of v's subtree. @complexity amortized O(log n) */
			constexpr void subtree_add(uint32 v, Value const &x)
				requires top_clusters::detail::sum_family<Path>
			{
				self().subtree_apply(v, add_tag(x));
			}

			/** @brief Add @p x to every vertex on v's side of the edge {v, p}. @complexity amortized O(log n) */
			constexpr void subtree_add(uint32 v, uint32 p, Value const &x)
				requires top_clusters::detail::sum_family<Path>
			{
				self().subtree_apply(v, p, add_tag(x));
			}

			/** @brief The minimum vertex value on the u - v path. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value path_min(uint32 u, uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().fold(u, v).spine.min;
			}

			/** @brief The maximum vertex value on the u - v path. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value path_max(uint32 u, uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().fold(u, v).spine.max;
			}

			/** @brief The minimum over v's subtree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_min(uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().subtree_fold(v).total().min;
			}

			/** @brief The minimum over v's side of the edge {v, p}. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_min(uint32 v, uint32 p)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().subtree_fold(v, p).total().min;
			}

			/** @brief The maximum over v's subtree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_max(uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().subtree_fold(v).total().max;
			}

			/** @brief The maximum over v's side of the edge {v, p}. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value subtree_max(uint32 v, uint32 p)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().subtree_fold(v, p).total().max;
			}

			/** @brief The minimum over v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value component_min(uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().component_fold(v).total().min;
			}

			/** @brief The maximum over v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr Value component_max(uint32 v)
				requires top_clusters::detail::extremes_family<Path>
			{
				return self().component_fold(v).total().max;
			}

			/** @brief Set every vertex on the u - v path to @p x. @complexity amortized O(log n) */
			constexpr void path_assign(uint32 u, uint32 v, Value const &x)
				requires top_clusters::detail::extremes_family<Path>
			{
				self().apply(u, v, Tag::assign_of(x));
			}

			/** @brief Set every vertex of v's subtree to @p x. @complexity amortized O(log n) */
			constexpr void subtree_assign(uint32 v, Value const &x)
				requires top_clusters::detail::extremes_family<Path>
			{
				self().subtree_apply(v, Tag::assign_of(x));
			}

			/** @brief Set every vertex on v's side of the edge {v, p} to @p x. @complexity amortized O(log n) */
			constexpr void subtree_assign(uint32 v, uint32 p, Value const &x)
				requires top_clusters::detail::extremes_family<Path>
			{
				self().subtree_apply(v, p, Tag::assign_of(x));
			}

			/** @brief The length of the longest path in v's tree (0 for a lone vertex). @complexity amortized O(log n) */
			[[nodiscard]] constexpr auto diameter(uint32 v)
				requires top_clusters::detail::diameter_family<Path>
			{
				return self().component_fold(v).diameter.length;
			}

			/** @brief The endpoints of a longest path in v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr std::pair<uint32, uint32> diameter_endpoints(uint32 v)
				requires top_clusters::detail::diameter_family<Path>
			{
				auto const span = self().component_fold(v).diameter;
				return {span.first, span.second};
			}

			/** @brief The distance from v to the farthest vertex of its tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr auto eccentricity(uint32 v)
				requires top_clusters::detail::diameter_family<Path>
			{
				return self().component_fold(v).from_bottom.distance;
			}

			/** @brief The farthest vertex from v in its tree, with its distance. @complexity amortized O(log n) */
			[[nodiscard]] constexpr auto farthest(uint32 v)
				requires top_clusters::detail::diameter_family<Path>
			{
				return self().component_fold(v).from_bottom;
			}

			/** @brief The length of the u - v path. @pre connected (asserted) @complexity amortized O(log n) */
			[[nodiscard]] constexpr auto path_length(uint32 u, uint32 v)
				requires top_clusters::detail::diameter_family<Path>
			{
				return self().fold(u, v).length;
			}

			/** @brief Mark or unmark vertex @p v. @complexity amortized O(log n) */
			constexpr void set_marked(uint32 v, bool is_marked)
				requires top_clusters::detail::marked_family<Path>
			{
				self().update_vertex(v, [is_marked](Value x) {
					x.is_marked = is_marked;
					return x;
				});
			}

			/** @brief Whether vertex @p v is marked. @complexity O(1) */
			[[nodiscard]] constexpr bool is_marked(uint32 v)
				requires top_clusters::detail::marked_family<Path>
			{
				return self().at_vertex(v).is_marked;
			}

			/** @brief The sum of the distances from v to every marked vertex of its tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr auto marked_distance_sum(uint32 v)
				requires top_clusters::detail::marked_family<Path>
			{
				return self().component_fold(v).bottom_sum;
			}

			/** @brief The number of marked vertices in v's tree. @complexity amortized O(log n) */
			[[nodiscard]] constexpr std::size_t marked_count(uint32 v)
				requires top_clusters::detail::marked_family<Path>
			{
				return self().component_fold(v).count;
			}
		};

		/**
		 * @brief The whole @ref atl::top_tree in any node layout, chain-push mode and
		 *        rake insertion, so the experimental variants share one
		 *        implementation with the library class.
		 *
		 * @tparam Layout           `splay_layout_split` (default) or interleaved.
		 * @tparam IsChainPushed    After one top-down push of the whole chain,
		 *                          splay without the engine's per-tree push pass
		 *                          (default); false re-pushes per splay (per-splay push).
		 * @tparam IsRakeRootInsert A new light child becomes the rake root
		 *                          (default); false joins it at the end.
		 * @tparam IsEdgeLeaf       Edge tier only: edges are leaves of the
		 *                          compress trees, never splayed (the
		 *                          edge-cluster shape);
		 *                          false makes them ordinary compress nodes.
		 * @tparam IsEvertForm      Path and edge-side operations by re-rooting
		 *                          and back (by rerooting) instead of
		 *                          two accesses around the lca (library).
		 * @tparam IsOwnCached      Each compress node caches its one-vertex
		 *                          cluster `add_vertex(point(mid), value)`, so a
		 *                          pull reads neither the rake pool nor calls
		 *                          `add_vertex`.
		 */
		template<splay_layout Layout, bool IsChainPushed, bool IsRakeRootInsert, bool IsEdgeLeaf, bool IsEvertForm,
				 bool IsOwnCached, typename Value, typename Path,
				 typename Point, typename Compress, typename Rake, typename AddVertex, typename AddEdge, typename Reverse,
				 typename Tag, typename Mapping, typename MapPath, typename MapPoint, typename Compose, typename ValuePlacement>
			requires top_tree_algebra<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
									  MapPoint, Compose, ValuePlacement>
				  && (!IsEdgeLeaf || std::same_as<ValuePlacement, edge_values_t>)
		class top_tree_impl
			: public top_named_members<top_tree_impl<Layout, IsChainPushed, IsRakeRootInsert, IsEdgeLeaf, IsEvertForm, IsOwnCached,
													 Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping,
													 MapPath, MapPoint, Compose, ValuePlacement>,
									   Value, Path, std::conditional_t<std::is_void_v<Tag>, top_no_field<1>, Tag>> {
		protected:
			using pool_type = top_pool<Layout, IsOwnCached, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping,
									   MapPath, MapPoint, Compose>;
			using compress_view = typename pool_type::compress_view;
			using rake_view = typename pool_type::rake_view;

			/** @brief The lazy tier is on. */
			static constexpr bool has_tags = pool_type::has_tags;
			/** @brief The edge tier is on. */
			static constexpr bool has_edge_values = std::same_as<ValuePlacement, edge_values_t>;
			/** @brief `Reverse` is a user callable. */
			static constexpr bool has_reverse_callable = pool_type::has_reverse_callable;
			/** @brief `Reverse` is commutative_fold_t. */
			static constexpr bool is_commutative = std::same_as<Reverse, commutative_fold_t>;
			/** @brief Edges are compress-tree leaves: sizes count vertices, edge updates go through an endpoint. */
			static constexpr bool has_edge_leaves = IsEdgeLeaf;
			/**
			 * @brief Sized nodes per tree edge on a root path: an edge tier path
			 *        alternates vertex and edge nodes (edge leaves have size 0).
			 */
			static constexpr uint32 stride = has_edge_values && !has_edge_leaves ? 2 : 1;

		public:
			/** @brief "No vertex / no edge": equal to @ref atl::graph::none. */
			static constexpr uint32 none = graph::none;

			using value_type = Value;
			using path_type = Path;
			using point_type = Point;
			/** @brief The tag type (a placeholder below the lazy tier). */
			using tag_type = typename pool_type::tag_field;

		protected:
			using reverse_slot = typename pool_type::reverse_slot;
			using mapping_slot = typename pool_type::mapping_slot;
			using map_path_slot = typename pool_type::map_path_slot;
			using map_point_slot = typename pool_type::map_point_slot;
			using compose_slot = typename pool_type::compose_slot;

			pool_type pool_;
			uint32 n_ = 0;
			[[no_unique_address]] std::conditional_t<has_edge_values, forest_edge_registry, top_no_field<9>> edges_;

			/** @brief What `expose` returns: where the climb ended and what its last step cut off. */
			struct exposure {
				uint32 top;			 ///< the last compress node the climb reached: the root of the root path's splay tree
				uint32 former_right; ///< top's right subtree before the climb replaced it
			};

			/** @brief The compress node of vertex @p v. */
			[[nodiscard]] static constexpr uint32 node_of(uint32 v) { return v + 1; }

			/** @brief The vertex of compress node @p x. */
			[[nodiscard]] static constexpr uint32 vertex_of(uint32 x) { return x - 1; }

			/** @brief The compress node of edge @p e (edge tier). */
			[[nodiscard]] constexpr uint32 edge_node(uint32 e) const { return n_ + 1 + e; }

			[[nodiscard]] constexpr compress_view cview() { return compress_view{&pool_}; }

			[[nodiscard]] constexpr rake_view rview() { return rake_view{&pool_}; }

			/** @brief After a splay made @p x its compress tree's root: the holding rake node points at it. */
			constexpr void repoint_compress(uint32 x) {
				if (uint32 const y = pool_.cl(x).parent; y != 0) pool_.rl(y).mid = x;
			}

			/** @brief Splay compress node x, whose chain is pushed (`push_chain`). */
			constexpr void splay_compress_pushed(uint32 x) {
				compress_view view = cview();
				if constexpr (IsChainPushed)
					splay_engine::splay_pushed(view, x);
				else
					splay_engine::splay(view, x);
				repoint_compress(x);
			}

			/** @brief Splay rake node y, whose chain is pushed; its owner points at it. */
			constexpr void splay_rake_pushed(uint32 y) {
				rake_view view = rview();
				if constexpr (IsChainPushed)
					splay_engine::splay_pushed(view, y);
				else
					splay_engine::splay(view, y);
				pool_.cl(pool_.rl(y).parent).mid = y;
			}

			/** @brief Splay compress node x with the engine's own push pass (pending tags above x's tree are harmless). */
			constexpr void splay_compress(uint32 x) {
				compress_view view = cview();
				splay_engine::splay(view, x);
				repoint_compress(x);
			}

			/** @brief Push every node from x's tree root (across all levels) down to x, top-down, recording in @p path. */
			constexpr void push_chain_with(uint32 x, std::vector<uint32> &path) {
				path.clear();
				for (uint32 y = x; y != 0; y = (y & top_rake_bit) != 0 ? pool_.rl(y).parent : pool_.cl(y).parent) path.push_back(y);
				for (std::size_t i = path.size(); i-- > 0;) {
					uint32 const y = path[i];
					if ((y & top_rake_bit) == 0)
						pool_.cpush(y);
					else if constexpr (has_tags)
						pool_.rpush(y);
				}
			}

			/**
			 * @brief Push the whole chain above compress node x: a tag on a compress
			 *        node reaches its rake tree and a rake node's tag its child's
			 *        path, so per-tree push passes would leave cross-level tags owed.
			 */
			constexpr void push_chain(uint32 x) {
				if consteval {
					std::vector<uint32> path;
					push_chain_with(x, path);
				} else {
					push_chain_with(x, splay_engine::ancestor_buffer());
				}
			}

			/** @brief Join two detached rake trees (either may be 0) and return the root. */
			constexpr uint32 join_rakes(uint32 a, uint32 b) {
				rake_view view = rview();
				return splay_engine::join(view, a, b);
			}

			/** @brief Hang the compress root @p c as a new light child of the pushed compress node @p x (x not pulled). */
			constexpr void hang_light(uint32 x, uint32 c) {
				uint32 const y = pool_.new_rake();
				pool_.rl(y).mid = c;
				pool_.cl(c).parent = y;
				uint32 const old = pool_.cl(x).mid;
				if constexpr (IsRakeRootInsert) {
					pool_.rl(y).child[0] = old;
					if (old != 0) pool_.rl(old).parent = y;
					pool_.rl(y).parent = x;
					pool_.rpull(y);
					pool_.cl(x).mid = y;
				} else {
					pool_.rpull(y);
					uint32 root = y;
					if (old != 0) {
						pool_.rl(old).parent = 0;
						root = join_rakes(old, y);
					}
					pool_.rl(root).parent = x;
					pool_.cl(x).mid = root;
				}
			}

			/** @brief Remove rake node @p y, the root of @p z's rake tree, and recycle it. */
			constexpr void drop_rake_root(uint32 z, uint32 y) {
				uint32 const a = pool_.rl(y).child[0], b = pool_.rl(y).child[1];
				if (a != 0) pool_.rl(a).parent = 0;
				if (b != 0) pool_.rl(b).parent = 0;
				uint32 const joined = join_rakes(a, b);
				pool_.cl(z).mid = joined;
				if (joined != 0) pool_.rl(joined).parent = z;
				pool_.free_rake(y);
			}

			/**
			 * @brief Make root..x one preferred path, without the final splay of x;
			 *        x's lower part becomes a light child. Returns the top compress
			 *        node, the root of that path's splay tree, and its right subtree
			 *        just before the climb's last step replaced it: after an
			 *        exposure of u, exposing v returns top = lca(u, v) when they are
			 *        connected, former_right the path strictly between it and u.
			 *
			 * Each climb step: x's tree hangs from rake node y in z's rake tree;
			 * splay y there and z in its own tree, then z's lower path takes y's
			 * place as a light child (y is dropped when z has none) and x's path
			 * becomes z's preferred continuation.
			 */
			constexpr exposure expose(uint32 x) {
				push_chain(x);
				splay_compress_pushed(x);
				uint32 const below = pool_.cl(x).child[1];
				if (below != 0) {
					pool_.cl(x).child[1] = 0;
					hang_light(x, below);
					pool_.refresh_own(x);
					pool_.cpull(x);
				}
				exposure e{x, below};
				for (uint32 y; (y = pool_.cl(x).parent) != 0;) {
					splay_rake_pushed(y);
					uint32 const z = pool_.rl(y).parent;
					splay_compress_pushed(z);
					uint32 const r = pool_.cl(z).child[1];
					if (r != 0) {
						pool_.rl(y).mid = r;
						pool_.cl(r).parent = y;
						pool_.rpull(y);
					} else {
						drop_rake_root(z, y);
					}
					pool_.refresh_own(z);
					pool_.cl(z).child[1] = x;
					pool_.cl(x).parent = z;
					pool_.cpull(z);
					e = {z, r};
					x = z;
				}
				return e;
			}

			/** @brief Expose x and splay it to the top: x roots the root path's splay tree, x last. */
			constexpr void access(uint32 x) {
				expose(x);
				splay_compress_pushed(x);
			}

			/** @brief Make node x its tree's root. */
			constexpr void evert(uint32 x) {
				access(x);
				pool_.cflip(x);
			}

			/**
			 * @brief The rightmost node of the subtree at @p root (with edge leaves:
			 *        the rightmost vertex, the leaves never splay), splayed to the top
			 *        of its compress tree.
			 */
			constexpr uint32 splay_rightmost(uint32 root) {
				compress_view view = cview();
				uint32 const r = splay_engine::descend(view, root, [this](uint32 t) {
					uint32 const right = pool_.cl(t).child[1];
					if constexpr (has_edge_leaves)
						if (right > n_) return splay_step::stop; // an edge leaf: t is the rightmost vertex
					return right == 0 ? splay_step::stop : splay_step::right;
				});
				repoint_compress(r);
				return r;
			}

			/** @brief An edge leaf's size and clusters: one node, size 0 (sizes count vertices). */
			constexpr void pull_edge_leaf(uint32 y)
				requires has_edge_leaves
			{
				pool_.cl(y).size = 0;
				typename pool_type::compress_payload &py = pool_.cp(y);
				py.path = pool_.add_vertex(pool_.rp(0).point, py.value);
				if constexpr (pool_type::has_reversed_path) py.reversed_path = py.path;
			}

			/**
			 * @brief Edge leaves: bring edge leaf @p y's lower endpoint c to the top
			 *        with its parent vertex p as the compress root: then p's right
			 *        child is c, whose left child is y. Returns {c, p}.
			 */
			constexpr std::pair<uint32, uint32> raise_edge_leaf(uint32 y)
				requires has_edge_leaves
			{
				auto const [a, b] = edges_.endpoints[y - n_ - 1];
				uint32 c = node_of(a);
				access(c);
				uint32 p = node_above(c);
				if (p == 0 || pool_.cl(c).child[0] != y) {
					c = node_of(b);
					access(c);
					p = node_above(c);
				}
				atl_assert(p != 0 && pool_.cl(c).child[0] == y); // y is the edge above c
				return {c, p};
			}

			/**
			 * @brief Replace the value of node @p x by `modify(value)` and settle every
			 *        cluster holding it: x tops the root path (an access), or for an
			 *        edge leaf its lower endpoint and parent do.
			 */
			template<typename Modify>
				requires std::invocable<Modify &, Value &>
			constexpr void modify_value(uint32 x, Modify &&modify) {
				if constexpr (has_edge_leaves) {
					if (x > n_) {
						auto const [c, p] = raise_edge_leaf(x);
						std::invoke(modify, pool_.cp(x).value);
						pull_edge_leaf(x);
						pool_.refresh_own(x);
						pool_.cpull(c);
						pool_.cpull(p);
						return;
					}
				}
				access(x);
				std::invoke(modify, pool_.cp(x).value);
				pool_.refresh_own(x);
				pool_.cpull(x);
			}

			/** @brief Root node of x's tree, splayed to the top of the root path. */
			constexpr uint32 find_root(uint32 x) {
				access(x);
				compress_view view = cview();
				return splay_engine::descend(view, x, [](uint32) { return splay_step::left; });
			}

			/** @brief After `access(x)`: the node right above x on its root path, 0 at the root. */
			constexpr uint32 node_above(uint32 x) {
				uint32 const l = pool_.cl(x).child[0];
				return l == 0 ? 0 : splay_rightmost(l);
			}

			/**
			 * @brief After `expose(x)` and `{w, c} = expose(y)`: are x and y in one
			 *        tree? Either x is the meeting point, or x's part was cut off
			 *        below w and hangs as a light child, while an unconnected x
			 *        tops its own tree.
			 */
			constexpr bool is_exposure_connected(uint32 x, uint32 w) {
				if (x == w) return true;
				splay_compress(x);
				return pool_.cl(x).parent != 0;
			}

			/** @brief Are nodes x and y in one tree? */
			constexpr bool nodes_connected(uint32 x, uint32 y) {
				if (x == y) return true;
				expose(x);
				return is_exposure_connected(x, expose(y).top);
			}

			/** @brief Cut x from the node above it when that is @p expected; false (nothing cut) otherwise. */
			constexpr bool detach_from(uint32 x, uint32 expected) {
				access(x);
				uint32 const p = node_above(x);
				if (p == 0 || p != expected) return false;
				// p is the compress root and x, its successor, its right child with no children
				pool_.cl(p).child[1] = 0;
				pool_.cl(x).parent = 0;
				pool_.cpull(p);
				return true;
			}

			/** @brief Reset compress node y to a detached node holding @p value. */
			constexpr void reset_node(uint32 y, Value value) {
				pool_.cl(y) = {{0, 0}, 0, 0, 1};
				typename pool_type::compress_payload &py = pool_.cp(y);
				py.value = std::move(value);
				py.is_reversed = false;
				if constexpr (has_tags) py.spine_tag = py.light_tag = pool_.tag_identity;
				pool_.refresh_own(y);
				if constexpr (has_edge_leaves)
					pull_edge_leaf(y);
				else
					pool_.cpull(y);
			}

			/** @brief Remove edge node @p y (edge tier): cut it from the node above, then its one child from it. */
			constexpr void cut_edge_node(uint32 y) {
				uint32 const e = y - n_ - 1;
				if constexpr (has_edge_leaves) {
					auto const [c, p] = raise_edge_leaf(y);
					pool_.cl(p).child[1] = 0;
					pool_.cl(c).parent = 0;
					pool_.cl(c).child[0] = 0;
					pool_.cpull(c);
					pool_.cpull(p);
					edges_.release(e);
					return;
				}
				auto const [a, b] = edges_.endpoints[e];
				access(y);
				uint32 const p = node_above(y);
				atl_assert(p != 0); // an edge node always has a node above it
				pool_.cl(p).child[1] = 0;
				pool_.cl(y).parent = 0;
				pool_.cpull(p);
				// y now roots the part holding the other endpoint, its only child
				uint32 const c = p == node_of(a) ? node_of(b) : node_of(a);
				access(c);
				pool_.cl(c).child[0] = 0;
				pool_.cl(y).parent = 0;
				pool_.cpull(c);
				edges_.release(e);
			}

			/** @brief Assert that @p v is a vertex id. */
			constexpr void check_vertex(uint32 v) const { atl_assert(v < n_); }

			/** @brief Assert that @p e is a live edge id; return its node. */
			constexpr uint32 check_edge(uint32 e) const
				requires has_edge_values
			{
				atl_assert(edges_.is_live(e)); // a live edge id
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

			/**
			 * @brief The node above v when p is v's parent: p's node, or on the
			 *        edge tier the node of edge {v, p}. @pre {v, p} is an edge
			 *        (asserted on the edge tier here, elsewhere by the caller)
			 */
			constexpr uint32 parent_side_node(uint32 v, uint32 p) const {
				if constexpr (has_edge_values) {
					uint32 const e = edges_.find(v, p);
					atl_assert(e != none); // {v, p} is an edge
					return has_edge_leaves ? node_of(p) : edge_node(e);
				} else {
					return node_of(p);
				}
			}

			/** @brief The cluster of compress subtree @p x read bottom to top (top = its last node). */
			[[nodiscard]] constexpr Path backward_path(uint32 x) const {
				if constexpr (pool_type::has_reversed_path)
					return pool_.cp(x).reversed_path;
				else if constexpr (has_reverse_callable)
					return pool_.reverse(pool_.cp(x).path);
				else
					return pool_.cp(x).path;
			}

			/** @brief Splay rake node y with the engine's push pass; its owner points at it. */
			constexpr void splay_rake(uint32 y) {
				rake_view view = rview();
				splay_engine::splay(view, y);
				pool_.cl(pool_.rl(y).parent).mid = y;
			}

			/**
			 * @brief After `expose(u)` and `{w, below} = expose(v)`: the rake node of w
			 *        holding u's segment (the path strictly below w down to u, hung
			 *        there by the climb), made w's rake root; 0 when u is w. Read
			 *        before anything splays u's segment.
			 */
			constexpr uint32 segment_rake(uint32 xu, uint32 w, uint32 below) {
				if (w == xu) return 0;
				uint32 const y = pool_.cl(below).parent;
				if (pool_.cl(w).mid != y) splay_rake(y); // only after a rake append
				return y;
			}

			/**
			 * @brief The cluster of compress node x as a one-vertex cluster that also
			 *        holds the root side (x's left subtree, read upwards, seen from x
			 *        as a hanging child) and every light child but rake node @p y's
			 *        own (all of them when y is 0). x tops the root path; y is x's
			 *        rake root.
			 */
			[[nodiscard]] constexpr Path own_with_root_side(uint32 x, uint32 y) const {
				Point hanging = [&] {
					if (y == 0) return Point(pool_.rp(pool_.cl(x).mid).point);
					auto const &ly = pool_.rl(y);
					return Point(pool_.rake(pool_.rp(ly.child[0]).point, pool_.rp(ly.child[1]).point));
				}();
				if (uint32 const above = pool_.cl(x).child[0]; above != 0) hanging = pool_.rake(hanging, pool_.add_edge(backward_path(above)));
				return pool_.add_vertex(hanging, pool_.cp(x).value);
			}

			/**
			 * @brief The u - v path in two accesses, the root untouched: after
			 *        `expose(u)`, `expose(v)` meets u's root path at w = lca(u, v),
			 *        with w's right subtree the path below w to v and u's segment
			 *        hung as w's light child (`segment_rake`). Returns
			 *        {w, the segment's rake node or 0}. @pre connected (asserted)
			 */
			constexpr std::pair<uint32, uint32> expose_path(uint32 xu, uint32 xv) {
				expose(xu);
				auto const [w, below] = expose(xv);
				uint32 const y = segment_rake(xu, w, below);
				atl_assert(is_exposure_connected(xu, w)); // u and v connected
				return {w, y};
			}

			/** @brief Where `locate_edge_side` left v's side of an edge. */
			struct edge_side {
				uint32 x;		   ///< v's node, the top of the root path
				uint32 rake;	   ///< p is v's child: the rake node holding p's side, v's rake root; else 0
				uint32 saved_root; ///< evert form only: the root to restore, else 0
			};

			/**
			 * @brief The compress node a light child p of v enters v's rake tree by:
			 *        p itself, or on the edge-as-vertex tier the edge node {v, p}.
			 */
			constexpr uint32 light_child_top(uint32 v, uint32 p) const {
				if constexpr (has_edge_values && !has_edge_leaves) {
					uint32 const e = edges_.find(v, p);
					atl_assert(e != none); // {v, p} is an edge
					return edge_node(e);
				} else {
					return node_of(p);
				}
			}

			/**
			 * @brief After an access that left v's node on top: whether light child
			 *        @p top heads its own path (the first node of its compress tree;
			 *        with edge leaves, p's left subtree is exactly the edge leaf).
			 */
			constexpr bool heads_path(uint32 v, uint32 p, uint32 top) const {
				if constexpr (has_edge_leaves) {
					uint32 const e = edges_.find(v, p);
					return e != none && pool_.cl(top).child[0] == edge_node(e);
				} else {
					return pool_.cl(top).child[0] == 0;
				}
			}

			/**
			 * @brief Bring v's node to the top of the root path with v's side of the
			 *        edge {v, p} readable: p is v's parent (v's subtree), or p is v's
			 *        child, whose rake node is made v's rake root (the side is
			 *        everything else: `own_with_root_side`). The evert form
			 *        re-roots at p instead. @pre {v, p} is an edge (asserted)
			 */
			constexpr edge_side locate_edge_side(uint32 v, uint32 p) {
				check_vertex(v);
				check_vertex(p);
				uint32 const xv = node_of(v), expected = parent_side_node(v, p);
				access(xv);
				bool const is_parent = node_above(xv) == expected;
				splay_compress(xv);
				if (is_parent) return {xv, 0, 0};
				if constexpr (IsEvertForm) {
					uint32 const saved_root = find_root(xv);
					evert(node_of(p));
					access(xv);
					atl_assert(node_above(xv) == expected); // {v, p} is an edge
					splay_compress(xv);
					return {xv, 0, saved_root};
				} else {
					uint32 const top = light_child_top(v, p);
					splay_compress(top);
					uint32 const y = pool_.cl(top).parent;
					atl_assert(heads_path(v, p, top) && (y & top_rake_bit) != 0); // p is a light child...
					splay_rake(y);
					atl_assert(pool_.rl(y).parent == xv); // ... of v: {v, p} is an edge
					return {xv, y, 0};
				}
			}

			/** @brief Undo the evert form's re-rooting. */
			constexpr void restore_root(uint32 saved_root) {
				if (saved_root != 0) evert(saved_root);
			}

			/**
			 * @brief Evert form: bring the u - v path to the top as the root path's
			 *        splay tree, u first, v at the root; returns the root to restore
			 *        (`0` when u is already the root).
			 */
			constexpr uint32 raise_path(uint32 xu, uint32 xv)
				requires IsEvertForm
			{
				atl_assert(nodes_connected(xu, xv)); // u and v connected
				uint32 const saved_root = find_root(xu);
				if (saved_root == xu) {
					access(xv);
					return 0;
				}
				evert(xu);
				access(xv);
				return saved_root;
			}

			/** @brief Tag x's value and light subtrees with @p f: x's subtree once x tops its tree; x not pulled. */
			constexpr void apply_below(uint32 x, tag_type const &f)
				requires has_tags
			{
				typename pool_type::compress_payload &px = pool_.cp(x);
				px.value = pool_.mapping(f, px.value);
				if (uint32 const m = pool_.cl(x).mid; m != 0) pool_.rapply(m, f);
				pool_.refresh_own(x);
			}

			/** @brief No vertex has an edge (assert-only, O(n)). */
			[[nodiscard]] constexpr bool is_all_isolated() const {
				for (uint32 x = 1; x <= n_; ++x) {
					auto const &l = pool_.cl(x);
					if (l.parent != 0 || l.child[0] != 0 || l.child[1] != 0 || l.mid != 0) return false;
				}
				if constexpr (has_edge_values)
					if (!edges_.ids.empty()) return false;
				return true;
			}

			/**
			 * @brief n isolated vertices, the k-th holding `next()` (called n times in
			 *        order), and the identities in the sentinels; the edge tier adds
			 *        n - 1 free edge nodes.
			 */
			template<typename NextValue>
				requires std::invocable<NextValue &> && std::convertible_to<std::invoke_result_t<NextValue &>, Value>
			constexpr void init(std::size_t n, NextValue &&next, Path const &path_identity, Point const &point_identity) {
				// compress ids stay below the rake bit; the edge tier holds up to 2n - 1 nodes
				atl_assert(n < (has_edge_values ? std::size_t(1) << 30 : std::size_t(top_rake_bit)));
				n_ = uint32(n);
				std::size_t const edge_slots = has_edge_values && n > 0 ? n - 1 : 0;
				pool_.clear(n + edge_slots);
				if constexpr (has_edge_values) edges_.reset(edge_slots);
				if (n == 0) return;
				Value first = std::invoke(next);
				pool_.push_compress({}, pool_.make_payload(first, path_identity));
				pool_.push_rake({}, {point_identity, pool_.rake_tag_identity()});
				typename pool_type::compress_link const lone{{0, 0}, 0, 0, 1};
				for (std::size_t v = 0; v < n; ++v) {
					Value value = v == 0 ? first : Value(std::invoke(next));
					Path const path = pool_.add_vertex(point_identity, value);
					pool_.push_compress(lone, pool_.make_payload(std::move(value), path));
				}
				for (std::size_t e = 0; e < edge_slots; ++e) pool_.push_compress(lone, pool_.make_payload(first, path_identity));
			}

			/** @brief The pool over the given algebra (no nodes yet). */
			static constexpr pool_type make_pool(Compress compress, Rake rake, AddVertex add_vertex, AddEdge add_edge,
												 reverse_slot reverse, tag_type tag_identity, mapping_slot mapping,
												 map_path_slot map_path, map_point_slot map_point, compose_slot compose) {
				return pool_type(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), detail::move_or_copy(reverse),
								 detail::move_or_copy(mapping), detail::move_or_copy(map_path), detail::move_or_copy(map_point), detail::move_or_copy(compose),
								 std::move(tag_identity));
			}

			/** @brief A generator handing out the elements of @p values in order (forwarded per the library rule). */
			template<std::ranges::input_range R>
				requires std::constructible_from<Value, forwarded_element_t<R>>
			static constexpr auto values_source(std::remove_reference_t<R> &values) {
				return [it = std::ranges::begin(values)]() mutable -> Value {
					Value value(forward_element<R>(*it));
					++it;
					return value;
				};
			}

			/** @brief The starting value of vertex @p v for a ready-made set: `Value::of_vertex(v)`, else `Value{}`. */
			static constexpr Value ready_made_value(uint32 v) {
				if constexpr (top_clusters::detail::per_vertex_value<Value>)
					return Value::of_vertex(v);
				else
					return Value{};
			}

			/**
			 * @brief Link the light children listed per parent, children first:
			 *        vertex v's children hang from a balanced rake tree in
			 *        `children[start[v], start[v + 1])`, each child c entering as
			 *        a compress tree over c and (edge tier) `above[c]`, the edge node
			 *        to its parent: the edge node on top of c, or with edge leaves
			 *        c on top with the leaf as its left child. @p order lists
			 *        children before parents.
			 */
			constexpr void build_rake_trees(std::vector<uint32> const &order, std::vector<uint32> const &start,
											std::vector<uint32> const &children, std::vector<uint32> const &above) {
				auto const top = [&](uint32 c) { return has_edge_values && !has_edge_leaves && above[c] != 0 ? above[c] : node_of(c); };
				std::vector<uint32> rakes;
				for (uint32 const v : order) {
					uint32 const x = node_of(v);
					uint32 const first = start[v], last = start[v + 1];
					if (first != last) {
						rakes.clear();
						for (uint32 i = first; i < last; ++i) {
							uint32 const y = pool_.new_rake(), c = top(children[i]);
							pool_.rl(y).mid = c;
							pool_.cl(c).parent = y;
							rakes.push_back(y);
						}
						uint32 root = rakes[0];
						if (rakes.size() == 1) {
							pool_.rpull(root);
						} else {
							rake_view view = rview();
							root = splay_engine::build_balanced(view, rakes);
						}
						pool_.rl(root).parent = x;
						pool_.cl(x).mid = root;
						pool_.refresh_own(x);
					}
					if constexpr (has_edge_leaves) {
						if (uint32 const y = above[v]; y != 0) {
							pool_.cl(x).child[0] = y;
							pool_.cl(y).parent = x;
						}
					}
					pool_.cpull(x);
					if constexpr (has_edge_values && !has_edge_leaves) {
						if (uint32 const y = above[v]; y != 0) { // v's edge node: v is its preferred child
							pool_.cl(y).child[1] = x;
							pool_.cl(x).parent = y;
							pool_.cpull(y);
						}
					}
				}
			}

			/**
			 * @brief The children-first order and per-parent child lists of a
			 *        parent array (topological order on the child counts, no recursion),
			 *        then `build_rake_trees`.
			 */
			constexpr void link_parent_array(std::vector<uint32> const &p, std::vector<uint32> const &above) {
				std::vector<uint32> start(n_ + 1, 0), pending(n_, 0);
				for (uint32 v = 0; v < n_; ++v)
					if (p[v] != none && p[v] != v) {
						++start[p[v] + 1];
						++pending[p[v]];
					}
				for (uint32 v = 0; v < n_; ++v) start[v + 1] += start[v];
				std::vector<uint32> children(start[n_]), fill(start.begin(), start.end() - 1);
				for (uint32 v = 0; v < n_; ++v)
					if (p[v] != none && p[v] != v) children[fill[p[v]]++] = v;
				std::vector<uint32> order;
				order.reserve(n_);
				for (uint32 v = 0; v < n_; ++v)
					if (pending[v] == 0) order.push_back(v);
				for (std::size_t i = 0; i < order.size(); ++i) {
					uint32 const v = order[i];
					if (p[v] != none && p[v] != v && --pending[p[v]] == 0) order.push_back(p[v]);
				}
				build_rake_trees(order, start, children, above);
			}

		public:
			/** @brief An empty forest (no vertices). @complexity O(1) */
			constexpr top_tree_impl()
				requires std::default_initializable<Compress> && std::default_initializable<Rake>
					  && std::default_initializable<AddVertex> && std::default_initializable<AddEdge>
					  && std::default_initializable<reverse_slot> && std::default_initializable<tag_type>
					  && std::default_initializable<mapping_slot> && std::default_initializable<map_path_slot>
					  && std::default_initializable<map_point_slot> && std::default_initializable<compose_slot>
				: pool_(make_pool({}, {}, {}, {}, {}, {}, {}, {}, {}, {})) {}

			/**
			 * @brief @p n isolated vertices, each holding @p value: a one-vertex
			 *        cluster `add_vertex(point_identity, value)`.
			 * @pre `n < 2^31` (`2^30` on the edge tier; asserted) @complexity O(n)
			 */
			constexpr top_tree_impl(std::size_t n, Value value, Path path_identity, Point point_identity, Compress compress = {},
									Rake rake = {}, AddVertex add_vertex = {}, AddEdge add_edge = {})
				requires(!has_tags && !has_reverse_callable)
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), {}, {}, {}, {}, {},
								  {})) {
				init(n, [&value] { return value; }, path_identity, point_identity);
			}

			/** @brief As above, with a callable `reverse` swapping a path cluster's ends. @complexity O(n) */
			constexpr top_tree_impl(std::size_t n, Value value, Path path_identity, Point point_identity, Compress compress,
									Rake rake, AddVertex add_vertex, AddEdge add_edge, reverse_slot reverse)
				requires(!has_tags && has_reverse_callable)
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), detail::move_or_copy(reverse),
								  {}, {}, {}, {}, {})) {
				init(n, [&value] { return value; }, path_identity, point_identity);
			}

			/**
			 * @brief As above with the lazy tier: tags (identity @p tag_identity),
			 *        `mapping(f, value)`, `map_path(spine, light, path)`,
			 *        `map_point(f, point)` and `compose(f, g)` = "g first, then f".
			 * @complexity O(n)
			 */
			constexpr top_tree_impl(std::size_t n, Value value, Path path_identity, Point point_identity, Compress compress,
									Rake rake, AddVertex add_vertex, AddEdge add_edge, tag_type tag_identity,
									mapping_slot mapping = {}, map_path_slot map_path = {}, map_point_slot map_point = {},
									compose_slot compose = {})
				requires(has_tags && !has_reverse_callable)
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), {},
								  std::move(tag_identity), detail::move_or_copy(mapping), detail::move_or_copy(map_path), detail::move_or_copy(map_point),
								  detail::move_or_copy(compose))) {
				init(n, [&value] { return value; }, path_identity, point_identity);
			}

			/** @brief The lazy tier with a callable `reverse`. @complexity O(n) */
			constexpr top_tree_impl(std::size_t n, Value value, Path path_identity, Point point_identity, Compress compress,
									Rake rake, AddVertex add_vertex, AddEdge add_edge, reverse_slot reverse, tag_type tag_identity,
									mapping_slot mapping = {}, map_path_slot map_path = {}, map_point_slot map_point = {},
									compose_slot compose = {})
				requires(has_tags && has_reverse_callable)
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), detail::move_or_copy(reverse),
								  std::move(tag_identity), detail::move_or_copy(mapping), detail::move_or_copy(map_path), detail::move_or_copy(map_point),
								  detail::move_or_copy(compose))) {
				init(n, [&value] { return value; }, path_identity, point_identity);
			}

			/**
			 * @brief One isolated vertex per element of @p values (the vertex
			 *        values; on the edge tier the values of the vertex nodes).
			 *        Elements are moved out only of an rvalue range that owns them.
			 * @complexity O(n)
			 */
			template<std::ranges::sized_range R>
				requires(!has_tags && !has_reverse_callable && std::constructible_from<Value, forwarded_element_t<R>>)
			constexpr top_tree_impl(R &&values, std::type_identity_t<Path> path_identity, std::type_identity_t<Point> point_identity,
									std::type_identity_t<Compress> compress = {}, std::type_identity_t<Rake> rake = {},
									std::type_identity_t<AddVertex> add_vertex = {}, std::type_identity_t<AddEdge> add_edge = {})
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), {}, {}, {}, {}, {},
								  {})) {
				init(std::ranges::size(values), values_source<R>(values), path_identity, point_identity);
			}

			/** @copydoc top_tree_impl(R &&, std::type_identity_t<Path>, std::type_identity_t<Point>, std::type_identity_t<Compress>, std::type_identity_t<Rake>, std::type_identity_t<AddVertex>, std::type_identity_t<AddEdge>) */
			template<std::ranges::sized_range R>
				requires(!has_tags && has_reverse_callable && std::constructible_from<Value, forwarded_element_t<R>>)
			constexpr top_tree_impl(R &&values, std::type_identity_t<Path> path_identity, std::type_identity_t<Point> point_identity,
									std::type_identity_t<Compress> compress, std::type_identity_t<Rake> rake,
									std::type_identity_t<AddVertex> add_vertex, std::type_identity_t<AddEdge> add_edge,
									std::type_identity_t<reverse_slot> reverse)
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), detail::move_or_copy(reverse),
								  {}, {}, {}, {}, {})) {
				init(std::ranges::size(values), values_source<R>(values), path_identity, point_identity);
			}

			/** @brief One vertex per element of @p values, with the lazy tier. @complexity O(n) */
			template<std::ranges::sized_range R>
				requires(has_tags && !has_reverse_callable && std::constructible_from<Value, forwarded_element_t<R>>)
			constexpr top_tree_impl(R &&values, std::type_identity_t<Path> path_identity, std::type_identity_t<Point> point_identity,
									std::type_identity_t<Compress> compress, std::type_identity_t<Rake> rake,
									std::type_identity_t<AddVertex> add_vertex, std::type_identity_t<AddEdge> add_edge,
									std::type_identity_t<tag_type> tag_identity, std::type_identity_t<mapping_slot> mapping = {},
									std::type_identity_t<map_path_slot> map_path = {},
									std::type_identity_t<map_point_slot> map_point = {},
									std::type_identity_t<compose_slot> compose = {})
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), {},
								  std::move(tag_identity), detail::move_or_copy(mapping), detail::move_or_copy(map_path), detail::move_or_copy(map_point),
								  detail::move_or_copy(compose))) {
				init(std::ranges::size(values), values_source<R>(values), path_identity, point_identity);
			}

			/** @brief One vertex per element of @p values, lazy tier with a callable `reverse`. @complexity O(n) */
			template<std::ranges::sized_range R>
				requires(has_tags && has_reverse_callable && std::constructible_from<Value, forwarded_element_t<R>>)
			constexpr top_tree_impl(R &&values, std::type_identity_t<Path> path_identity, std::type_identity_t<Point> point_identity,
									std::type_identity_t<Compress> compress, std::type_identity_t<Rake> rake,
									std::type_identity_t<AddVertex> add_vertex, std::type_identity_t<AddEdge> add_edge,
									std::type_identity_t<reverse_slot> reverse, std::type_identity_t<tag_type> tag_identity,
									std::type_identity_t<mapping_slot> mapping = {},
									std::type_identity_t<map_path_slot> map_path = {},
									std::type_identity_t<map_point_slot> map_point = {},
									std::type_identity_t<compose_slot> compose = {})
				: pool_(make_pool(detail::move_or_copy(compress), detail::move_or_copy(rake), detail::move_or_copy(add_vertex), detail::move_or_copy(add_edge), detail::move_or_copy(reverse),
								  std::move(tag_identity), detail::move_or_copy(mapping), detail::move_or_copy(map_path), detail::move_or_copy(map_point),
								  detail::move_or_copy(compose))) {
				init(std::ranges::size(values), values_source<R>(values), path_identity, point_identity);
			}

			/**
			 * @brief A ready-made set (@ref atl::top_clusters): @p n isolated vertices,
			 *        identities from the types; each vertex starts as
			 *        `Value::of_vertex(v)` when the set defines it, else `Value{}`.
			 * @complexity O(n)
			 */
			constexpr explicit top_tree_impl(std::size_t n)
				requires top_clusters::is_ready_made<Path> && std::default_initializable<Value>
					  && std::default_initializable<Compress> && std::default_initializable<Rake>
					  && std::default_initializable<AddVertex> && std::default_initializable<AddEdge>
					  && std::default_initializable<reverse_slot> && std::default_initializable<tag_type>
					  && std::default_initializable<mapping_slot> && std::default_initializable<map_path_slot>
					  && std::default_initializable<map_point_slot> && std::default_initializable<compose_slot>
				: pool_(make_pool({}, {}, {}, {}, {}, {}, {}, {}, {}, {})) {
				init(n, [v = uint32(0)]() mutable { return ready_made_value(v++); }, Path{}, Point{});
			}

			/** @brief A ready-made set over the vertex values @p values. @complexity O(n) */
			template<std::ranges::sized_range R>
				requires top_clusters::is_ready_made<Path> && (!has_edge_values)
					  && std::constructible_from<Value, forwarded_element_t<R>> && std::default_initializable<Compress>
					  && std::default_initializable<Rake> && std::default_initializable<AddVertex>
					  && std::default_initializable<AddEdge> && std::default_initializable<reverse_slot>
					  && std::default_initializable<tag_type> && std::default_initializable<mapping_slot>
					  && std::default_initializable<map_path_slot> && std::default_initializable<map_point_slot>
					  && std::default_initializable<compose_slot>
			constexpr explicit top_tree_impl(R &&values) : pool_(make_pool({}, {}, {}, {}, {}, {}, {}, {}, {}, {})) {
				init(std::ranges::size(values), values_source<R>(values), Path{}, Point{});
			}

			/** @brief The forms above in the commutative mode, deduced from the leading @ref atl::commutative_fold. */
			template<typename... Args>
				requires is_commutative && std::constructible_from<top_tree_impl, Args...>
			constexpr explicit top_tree_impl(commutative_fold_t, Args &&...args) : top_tree_impl(std::forward<Args>(args)...) {}

			/** @brief The forms above on the edge tier, deduced from the leading @ref atl::edge_values. */
			template<typename... Args>
				requires has_edge_values && std::constructible_from<top_tree_impl, Args...>
			constexpr explicit top_tree_impl(edge_values_t, Args &&...args) : top_tree_impl(std::forward<Args>(args)...) {}

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
				access(xv);
				// u's path continues v's root path: no rake node needed
				pool_.cl(xv).child[1] = xu;
				pool_.cl(xu).parent = xv;
				pool_.cpull(xv);
			}

			/**
			 * @brief Add the edge u - v holding @p w (edge tier) and return its id,
			 *        in `[0, size() - 1)` (ids of cut edges are reused, the most
			 *        recently freed first); u's tree hangs below v.
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
				uint32 const e = edges_.acquire(u, v), y = edge_node(e);
				reset_node(y, Value(std::forward<EdgeValue>(w)));
				if constexpr (has_edge_leaves) {
					access(xv);
					pool_.cpush(xu); // u's flip settled: u comes first, its left side empty
					pool_.cl(xu).child[0] = y;
					pool_.cl(y).parent = xu;
					pool_.cpull(xu);
					pool_.cl(xv).child[1] = xu;
					pool_.cl(xu).parent = xv;
				} else {
					pool_.cl(y).child[1] = xu;
					pool_.cl(xu).parent = y;
					pool_.cpull(y);
					access(xv);
					pool_.cl(xv).child[1] = y;
					pool_.cl(y).parent = xv;
				}
				pool_.cpull(xv);
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
					uint32 const e = edges_.find(u, v);
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
				uint32 const x = node_of(v);
				access(x);
				if constexpr (has_edge_leaves) {
					uint32 const p = node_above(x);
					atl_assert(p != 0); // v is not a root
					uint32 const y = pool_.cl(x).child[0];
					pool_.cl(p).child[1] = 0;
					pool_.cl(x).parent = 0;
					pool_.cl(x).child[0] = 0;
					pool_.cpull(x);
					pool_.cpull(p);
					edges_.release(y - n_ - 1);
					return;
				}
				uint32 const l = pool_.cl(x).child[0];
				atl_assert(l != 0); // v is not a root
				pool_.cl(x).child[0] = 0;
				pool_.cl(l).parent = 0;
				pool_.cpull(x);
				if constexpr (has_edge_values) {
					// the cut-off root path ends with v's edge node, which has no child left: drop it
					uint32 const y = splay_rightmost(l);
					if (uint32 const above = pool_.cl(y).child[0]; above != 0) {
						pool_.cl(y).child[0] = 0;
						pool_.cl(above).parent = 0;
						pool_.cpull(y);
					}
					edges_.release(y - n_ - 1);
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
				uint32 p = node_above(x);
				if (p == 0) return none;
				if constexpr (has_edge_values && !has_edge_leaves) p = splay_rightmost(pool_.cl(p).child[0]); // past v's edge node
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
			[[nodiscard]] constexpr uint32 depth(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				return pool_.cl(pool_.cl(x).child[0]).size / stride;
			}

			/** @brief The number of edges on the u - v path, `none` across trees. @complexity amortized O(log n) */
			[[nodiscard]] constexpr uint32 distance(uint32 u, uint32 v) {
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				expose(xu);
				auto const [w, below] = expose(xv);
				// below: the path strictly between w and u, read before the check's splay reshapes it
				uint32 const u_side = w == xu ? 0 : pool_.cl(below).size;
				if (!is_exposure_connected(xu, w)) return none;
				return (u_side + pool_.cl(pool_.cl(w).child[1]).size) / stride;
			}

			/**
			 * @brief The value of vertex @p id (edge @p id on the edge tier).
			 *        O(1) without tags; with tags, by value after settling the tags
			 *        above it (amortized O(log n)).
			 */
			[[nodiscard]] constexpr Value const &at(uint32 id) const
				requires(!has_tags)
			{
				return pool_.cp(value_node(id)).value;
			}

			/** @copydoc at */
			[[nodiscard]] constexpr Value at(uint32 id)
				requires has_tags
			{
				Value result = pool_.cp(0).value;
				modify_value(value_node(id), [&result](Value &value) { result = value; });
				return result;
			}

			/**
			 * @brief Replace the value of vertex @p id (edge @p id on the edge tier).
			 * @complexity amortized O(log n)
			 */
			template<typename NewValue>
				requires std::assignable_from<Value &, NewValue>
			constexpr void set(uint32 id, NewValue &&x) {
				modify_value(value_node(id), [&x](Value &value) { value = std::forward<NewValue>(x); });
			}

			/**
			 * @brief Replace the value `a` of vertex @p id (edge @p id on the edge
			 *        tier) by `f(a)`.
			 * @complexity amortized O(log n)
			 */
			template<update_function<Value> Update>
			constexpr void update(uint32 id, Update &&f) {
				modify_value(value_node(id), [&f](Value &value) { value = std::invoke(f, std::as_const(value)); });
			}

			/** @brief The value of vertex @p v's node (edge tier). @complexity as `at` */
			[[nodiscard]] constexpr Value const &at_vertex(uint32 v) const
				requires has_edge_values && (!has_tags)
			{
				check_vertex(v);
				return pool_.cp(node_of(v)).value;
			}

			/** @copydoc at_vertex */
			[[nodiscard]] constexpr Value at_vertex(uint32 v)
				requires has_edge_values && has_tags
			{
				check_vertex(v);
				access(node_of(v));
				return pool_.cp(node_of(v)).value;
			}

			/** @brief Replace the value of vertex @p v's node (edge tier). @complexity amortized O(log n) */
			template<typename NewValue>
				requires has_edge_values && std::assignable_from<Value &, NewValue>
			constexpr void set_vertex(uint32 v, NewValue &&x) {
				check_vertex(v);
				modify_value(node_of(v), [&x](Value &value) { value = std::forward<NewValue>(x); });
			}

			/** @brief Replace the value `a` of vertex @p v's node by `f(a)` (edge tier). @complexity amortized O(log n) */
			template<update_function<Value> Update>
				requires has_edge_values
			constexpr void update_vertex(uint32 v, Update &&f) {
				check_vertex(v);
				modify_value(node_of(v), [&f](Value &value) { value = std::invoke(f, std::as_const(value)); });
			}

			/** @brief Vertex tier: `update` on vertex @p v (the spelling the edge tier shares). */
			template<update_function<Value> Update>
				requires(!has_edge_values)
			constexpr void update_vertex(uint32 v, Update &&f) {
				update(v, std::forward<Update>(f));
			}

			/** @brief Vertex tier: `at` on vertex @p v (the spelling the edge tier shares). */
			[[nodiscard]] constexpr decltype(auto) at_vertex(uint32 v)
				requires(!has_edge_values)
			{
				return at(v);
			}

			/**
			 * @brief The whole tree of u and v as a path cluster with boundary u
			 *        (top) and v (bottom): the spine u..v and everything hanging
			 *        off it. The root is unchanged: two accesses meet at the lca,
			 *        the root side read as a cluster hanging off it.
			 * @pre u and v connected (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr Path fold(uint32 u, uint32 v) {
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				if constexpr (IsEvertForm) {
					uint32 const saved_root = raise_path(xu, xv);
					Path result = pool_.cp(xv).path;
					if (saved_root != 0) evert(saved_root);
					return result;
				} else {
					auto const [w, y] = expose_path(xu, xv);
					// u's segment (read upwards), w with everything else hanging, then w's path down to v
					Path result = own_with_root_side(w, y);
					if (y != 0) result = pool_.compress(backward_path(pool_.rl(y).mid), result);
					if (uint32 const below = pool_.cl(w).child[1]; below != 0) result = pool_.compress(result, pool_.cp(below).path);
					return result;
				}
			}

			/**
			 * @brief v's subtree under the current root as a one-vertex cluster at
			 *        v: `add_vertex(light subtrees of v, value of v)`.
			 * @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr Path subtree_fold(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				return pool_.own(x);
			}

			/**
			 * @brief v's side of the edge {v, p} as a one-vertex cluster at v,
			 *        whatever the root (when p is v's child: everything but p's
			 *        side, the root side read as a hanging cluster; no re-rooting).
			 * @pre {v, p} is an edge (asserted) @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr Path subtree_fold(uint32 v, uint32 p) {
				auto const [x, y, saved_root] = locate_edge_side(v, p);
				if (y != 0) return own_with_root_side(x, y);
				Path result = pool_.own(x);
				restore_root(saved_root);
				return result;
			}

			/**
			 * @brief v's whole tree as a path cluster with boundary (root, v).
			 * @complexity amortized O(log n)
			 */
			[[nodiscard]] constexpr Path component_fold(uint32 v) {
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				return pool_.cp(x).path;
			}

			/**
			 * @brief Apply tag @p f to every value on the u - v path (the spine; on
			 *        the edge tier its vertex and edge nodes). The root is unchanged.
			 * @pre u and v connected (asserted) @complexity amortized O(log n)
			 */
			constexpr void apply(uint32 u, uint32 v, tag_type const &f)
				requires has_tags
			{
				check_vertex(u);
				check_vertex(v);
				uint32 const xu = node_of(u), xv = node_of(v);
				if constexpr (IsEvertForm) {
					uint32 const saved_root = raise_path(xu, xv);
					pool_.capply(xv, f, pool_.tag_identity);
					if (saved_root != 0) evert(saved_root);
				} else {
					auto const [w, y] = expose_path(xu, xv);
					if (y != 0) {
						pool_.capply(pool_.rl(y).mid, f, pool_.tag_identity);
						pool_.rpull(y);
					}
					pool_.cp(w).value = pool_.mapping(f, pool_.cp(w).value);
					pool_.refresh_own(w);
					if (uint32 const below = pool_.cl(w).child[1]; below != 0) pool_.capply(below, f, pool_.tag_identity);
					pool_.cpull(w);
				}
			}

			/** @brief Apply tag @p f to every value in v's subtree under the current root. @complexity amortized O(log n) */
			constexpr void subtree_apply(uint32 v, tag_type const &f)
				requires has_tags
			{
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				apply_below(x, f);
				pool_.cpull(x);
			}

			/**
			 * @brief Apply tag @p f to every value on v's side of the edge {v, p}.
			 * @pre {v, p} is an edge (asserted) @complexity amortized O(log n)
			 */
			constexpr void subtree_apply(uint32 v, uint32 p, tag_type const &f)
				requires has_tags
			{
				auto const [x, y, saved_root] = locate_edge_side(v, p);
				if (y != 0) {
					// everything but p's side: x, its other light children, the root side
					typename pool_type::compress_payload &px = pool_.cp(x);
					px.value = pool_.mapping(f, px.value);
					auto const &ly = pool_.rl(y);
					if (ly.child[0] != 0) pool_.rapply(ly.child[0], f);
					if (ly.child[1] != 0) pool_.rapply(ly.child[1], f);
					pool_.rpull(y);
					pool_.refresh_own(x);
					if (uint32 const above = pool_.cl(x).child[0]; above != 0) pool_.capply(above, f, f);
				} else {
					apply_below(x, f);
				}
				pool_.cpull(x);
				restore_root(saved_root);
			}

			/** @brief Apply tag @p f to every value in v's tree. @complexity amortized O(log n) */
			constexpr void component_apply(uint32 v, tag_type const &f)
				requires has_tags
			{
				check_vertex(v);
				uint32 const x = node_of(v);
				access(x);
				pool_.capply(x, f, f);
			}

			/**
			 * @brief Link every vertex v with `parents[v] != none` (and `!= v`)
			 *        below `parents[v]` (a parent-array format),
			 *        rooting each tree at its parentless vertex. Every child enters
			 *        as a light child, the children of a vertex in a balanced rake
			 *        tree.
			 * @pre every vertex isolated; `parents` has size() entries and forms a
			 *      forest (all asserted) @complexity O(n)
			 * @tparam Parents A forward vertex range iterable through const.
			 */
			template<typename Parents>
				requires std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const> && (!has_edge_values)
			constexpr void assign_parents(Parents const &parents) {
				atl_assert(is_all_isolated());
				std::vector<uint32> p(std::ranges::begin(parents), std::ranges::end(parents));
				atl_assert(is_parent_forest(p, n_));
				link_parent_array(p, std::vector<uint32>(n_, 0));
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
				std::vector<uint32> p(std::ranges::begin(parents), std::ranges::end(parents));
				atl_assert(is_parent_forest(p, n_));
				edges_.reset(edges_.endpoints.size());
				std::vector<uint32> above(n_, 0);
				auto weight = std::ranges::begin(weights);
				for (uint32 v = 0; v < n_; ++v, ++weight) {
					atl_assert(weight != std::ranges::end(weights));
					if (p[v] != none && p[v] != v) {
						uint32 const y = edge_node(edges_.acquire(v, p[v]));
						reset_node(y, Value(*weight));
						above[v] = y;
					}
				}
				link_parent_array(p, above);
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
				return edges_.find(u, v);
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
	 * @brief Top tree: a forest under link, cut and re-rooting where every
	 *        tree is summarized by a user cluster algebra, so path folds and
	 *        updates, subtree folds and updates (no inverse needed: min, max,
	 *        gcd), and whole-tree shape aggregates (diameter, farthest vertex,
	 *        distance sums) all hold at once, each amortized O(log n). A
	 *        client of the public @ref atl::splay_engine.
	 *
	 * A *path cluster* (`Path`) summarizes a connected piece of a tree with two
	 * boundary vertices, top and bottom: its spine (the vertices between) and
	 * everything hanging off it. A *point cluster* (`Point`) summarizes the
	 * subtrees hanging at one vertex, the vertex excluded. The user supplies:
	 *  - `compress(upper, lower)`: join two path clusters, `upper`'s bottom
	 *    adjacent to `lower`'s top (associative; the path identity two-sided);
	 *  - `rake(a, b)`: merge two hanging sets (associative AND commutative:
	 *    light children are kept in no particular order);
	 *  - `add_vertex(light, value)`: a vertex over its hanging set as a
	 *    one-vertex path cluster (which must equal its own reversal);
	 *  - `add_edge(child)`: a child's path cluster (top = the child) seen from
	 *    its parent as a hanging set;
	 *  - `Reverse`: `void` (a reversed cluster is kept per node, always
	 *    correct), @ref atl::commutative_fold_t (clusters are symmetric: sum, max),
	 *    or a callable swapping a cluster's ends in O(1) (diameter: swap the
	 *    farthest-from-top and farthest-from-bottom entries).
	 * The lazy tier (a non-`void` `Tag`) adds `mapping(f, value)`,
	 * `map_path(spine, light, path)` (tag `spine` on the spine's vertices,
	 * `light` on everything hanging off), `map_point(f, point)` and
	 * `compose(f, g)` = "g first, then f": a path update is the pair (f, id),
	 * a subtree update (f, f), so non-commuting path and subtree updates mix.
	 *
	 * The ready-made sets of @ref atl::top_clusters (sum, extremes, diameter,
	 * marked_distance) fill every slot and switch on named members; see the
	 * aliases @ref atl::sum_top_tree, @ref atl::extremes_top_tree, @ref atl::diameter_top_tree
	 * and @ref atl::marked_distance_top_tree.
	 *
	 * The root of each tree changes only by `make_root`, `link` (u's side joins
	 * below v's root) and `cut` (the detached side is rooted at its endpoint):
	 * path and edge-side operations leave it alone. Aggregate and structural
	 * queries splay; untagged value reads and edge lookups are const.
	 * Vertex ids are `uint32` in `[0, size())`;
	 * `none` answers "no vertex".
	 *
	 * @tparam Value          Vertex value (on the edge tier: of vertex and edge nodes).
	 * @tparam Path           Path cluster.
	 * @tparam Point          Point cluster (a hanging set).
	 * @tparam Compress       `Path(Path const &upper, Path const &lower)`.
	 * @tparam Rake           `Point(Point const &, Point const &)`.
	 * @tparam AddVertex      `Path(Point const &light, Value const &)`.
	 * @tparam AddEdge        `Point(Path const &child)`.
	 * @tparam Reverse        `void`, @ref atl::commutative_fold_t or `Path(Path const &)`.
	 * @tparam Tag            Lazy tag (`void`: no lazy tier).
	 * @tparam Mapping        `Value(Tag const &, Value const &)`.
	 * @tparam MapPath        `Path(Tag const &spine, Tag const &light, Path const &)`.
	 * @tparam MapPoint       `Point(Tag const &, Point const &)`.
	 * @tparam Compose        `Tag(Tag const &f, Tag const &g)`, g first.
	 * @tparam ValuePlacement `void` (vertex values) or @ref atl::edge_values_t.
	 */
	template<typename Value, typename Path, typename Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, typename Reverse = void, typename Tag = void, typename Mapping = void, typename MapPath = void,
			 typename MapPoint = void, typename Compose = void, typename ValuePlacement = void>
		requires detail::top_tree_algebra<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
										  MapPoint, Compose, ValuePlacement>
	class top_tree : public detail::top_tree_impl<splay_layout_split, true, true, false, false, false, Value, Path, Point,
												  Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint,
												  Compose, ValuePlacement> {
		using base = detail::top_tree_impl<splay_layout_split, true, true, false, false, false, Value, Path, Point, Compress, Rake,
										   AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint, Compose, ValuePlacement>;

	public:
		using base::base;
	};

	namespace detail {
		/** @brief A @ref atl::top_tree over the ready-made set @p Set (every slot from the set's members). */
		template<typename Set, typename ValuePlacement = void>
		using top_tree_of_set =
			top_tree<typename Set::value_type, typename Set::path_type, typename Set::point_type, typename Set::compress,
					 typename Set::rake, typename Set::add_vertex, typename Set::add_edge, typename Set::reverse,
					 typename Set::tag_type, typename Set::mapping, typename Set::map_path, typename Set::map_point,
					 typename Set::compose, ValuePlacement>;
	}

	/** @brief Vertex values with path / subtree / component sums and sizes and lazy adds (@ref atl::top_clusters::sum). */
	template<typename T>
	using sum_top_tree = detail::top_tree_of_set<top_clusters::sum<T>>;

	/** @brief Sums, minima and maxima with lazy add and assign (@ref atl::top_clusters::extremes). */
	template<typename T>
	using extremes_top_tree = detail::top_tree_of_set<top_clusters::extremes<T>>;

	/** @brief Edge lengths with diameter, eccentricity, farthest vertex, path length (@ref atl::top_clusters::diameter). */
	template<typename W>
	using diameter_top_tree = detail::top_tree_of_set<top_clusters::diameter<W>, edge_values_t>;

	/** @brief Marked vertices with distance sums to them (@ref atl::top_clusters::marked_distance). */
	template<typename W>
	using marked_distance_top_tree = detail::top_tree_of_set<top_clusters::marked_distance<W>, edge_values_t>;

	/** @brief Vertex values: every type deduced from the arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge>
	top_tree(std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge>;
	/** @brief With a callable `reverse`. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, typename Reverse>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse>;
	/** @brief The lazy tier. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint, typename Compose>
	top_tree(std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping, MapPath, MapPoint, Compose)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, void, Tag, Mapping, MapPath, MapPoint, Compose>;
	/** @brief The lazy tier with a callable `reverse`. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, typename Reverse, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint,
			 typename Compose>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint,
			 Compose) -> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint,
								  Compose>;

	/** @brief One vertex per element: `Value` is the range's value type. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge>
	top_tree(R &&, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge, typename Reverse>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(R &&, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint,
			 typename Compose>
	top_tree(R &&, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping, MapPath, MapPoint, Compose)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge, void, Tag, Mapping, MapPath,
					MapPoint, Compose>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge, typename Reverse, std::copyable Tag, typename Mapping, typename MapPath,
			 typename MapPoint, typename Compose>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(R &&, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint, Compose)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
					MapPoint, Compose>;

	/** @brief The commutative mode, deduced from the leading @ref atl::commutative_fold argument. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge>
	top_tree(commutative_fold_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint, typename Compose>
	top_tree(commutative_fold_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping, MapPath,
			 MapPoint, Compose) -> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t, Tag, Mapping,
											MapPath, MapPoint, Compose>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge>
	top_tree(commutative_fold_t, R &&, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::ranges::sized_range R, std::copyable Path, std::copyable Point, typename Compress, typename Rake,
			 typename AddVertex, typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint,
			 typename Compose>
	top_tree(commutative_fold_t, R &&, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping, MapPath, MapPoint, Compose)
		-> top_tree<std::ranges::range_value_t<R>, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t, Tag,
					Mapping, MapPath, MapPoint, Compose>;

	/** @brief The edge tier, deduced from the leading @ref atl::edge_values argument. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge>
	top_tree(edge_values_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, void, void, void, void, void, void, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, typename Reverse>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(edge_values_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, void, void, void, void, void, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint, typename Compose>
	top_tree(edge_values_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping, MapPath, MapPoint,
			 Compose) -> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, void, Tag, Mapping, MapPath, MapPoint,
								  Compose, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, typename Reverse, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint,
			 typename Compose>
		requires detail::is_top_reverse_callable<Reverse>
	top_tree(edge_values_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
			 MapPoint, Compose) -> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath,
											MapPoint, Compose, edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge>
	top_tree(edge_values_t, commutative_fold_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge)
		-> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t, void, void, void, void, void,
					edge_values_t>;
	/** @brief Deduce the forest algebra and value tier from the supplied constructor arguments. */
	template<std::copyable Value, std::copyable Path, std::copyable Point, typename Compress, typename Rake, typename AddVertex,
			 typename AddEdge, std::copyable Tag, typename Mapping, typename MapPath, typename MapPoint, typename Compose>
	top_tree(edge_values_t, commutative_fold_t, std::size_t, Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Tag, Mapping,
			 MapPath, MapPoint, Compose) -> top_tree<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, commutative_fold_t, Tag,
													 Mapping, MapPath, MapPoint, Compose, edge_values_t>;
}
