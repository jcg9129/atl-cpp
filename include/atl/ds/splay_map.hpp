#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/ds/detail/key_value_entry.hpp>
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/tags.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief The template-argument contract shared with `treap_map:` a
		 *        strict weak order over the keys, a semiregular mapped type (or
		 *        `void` for a set), and an optional fold over what the map folds
		 *        (mapped values for a map, keys for a set).
		 */
		template<typename Key, typename Value, typename Compare, typename Op>
		concept splay_map_parameters =
			std::strict_weak_order<Compare const &, Key const &, Key const &> && (std::is_void_v<Value> || std::semiregular<Value>)
			&& (std::is_void_v<Op> || (std::is_void_v<Value> ? binary_operation<Op, Key> : binary_operation<Op, Value>));

		/** @brief The node layouts a splay map supports (the columnar one would split the key from the links twice over). */
		template<typename Layout>
		concept splay_map_layout = splay_selectable_layout<Layout>;

		/**
		 * @brief Node pool of a splay map: keys, mapped values and aggregates
		 *        in the chosen layout, recycling, the comparator and the fold,
		 *        and the `sized_splay_store` interface (with `pull`; no
		 *        `push`, a keyed tree has no pending tags) the engine drives.
		 *
		 * Node 0 is the null sentinel: size 0 and, on the fold tier, the identity
		 * as its aggregate, so `pull` folds through empty sides without branching.
		 */
		template<typename Layout, std::semiregular Key, typename Value, typename Compare, typename Op>
			requires splay_map_layout<Layout> && splay_map_parameters<Key, Value, Compare, Op>
		struct splay_map_pool {
			static constexpr bool has_value = !std::is_void_v<Value>;
			static constexpr bool has_op = !std::is_void_v<Op>;
			static constexpr bool is_interleaved = std::same_as<Layout, splay_layout_interleaved>;
			using fold_type = std::conditional_t<has_value, Value, Key>;

			struct no_slot {};

			template<typename X>
			using slot = std::conditional_t<std::is_void_v<X>, no_slot, X>;

			struct no_value_slot {};

			struct no_aggregate_slot {};

			struct value_slot {
				slot<Value> value{};
			};

			struct aggregate_slot {
				fold_type aggregate{};
			};

			struct payload : std::conditional_t<has_value, value_slot, no_value_slot>,
							 std::conditional_t<has_op, aggregate_slot, no_aggregate_slot> {
				Key key{};
			};

			/** @brief A node's structure: children, parent, subtree size (16 bytes). */
			struct link {
				uint32 child[2]{};
				uint32 parent = 0, size = 0;
			};

			struct node : link, payload {};

			struct interleaved_storage {
				std::vector<node> nodes;
			};

			struct split_storage {
				std::vector<link> links;
				std::vector<payload> payloads;
			};

			/** @brief Node storage; id 0 is the null sentinel. */
			std::conditional_t<is_interleaved, interleaved_storage, split_storage> store;
			/** @brief Free ids, reused before the storage grows. */
			std::vector<uint32> recycled;
			[[no_unique_address]] Compare comp;
			[[no_unique_address]] slot<Op> op;
			/** @brief The fold's identity, also node 0's aggregate (fold tier). */
			[[no_unique_address]] std::conditional_t<has_op, fold_type, no_slot> identity;

			splay_map_pool()
				requires std::default_initializable<Compare> && std::default_initializable<slot<Op>>
				: comp{}, op{}, identity{} {
				append_node();
				if constexpr (has_op) payloads(0).aggregate = identity;
			}

			splay_map_pool(Compare compare, slot<Op> fold = {}, std::conditional_t<has_op, fold_type, no_slot> unit = {})
				: comp(detail::move_or_copy(compare)), op(detail::move_or_copy(fold)), identity(std::move(unit)) {
				append_node();
				if constexpr (has_op) payloads(0).aggregate = identity;
			}

			decltype(auto) fold_values(auto &&a, auto &&b) const
				requires has_op && requires { detail::invoke_algebra<fold_type>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<fold_type>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			template<typename Left, typename Right>
				requires std::predicate<Compare const &, Left const &, Right const &>
			bool compare_keys(Left const &a, Right const &b) const { return std::invoke(comp, a, b); }

			[[nodiscard]] link &links(uint32 x) {
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.links[x];
			}

			[[nodiscard]] link const &links(uint32 x) const {
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.links[x];
			}

			[[nodiscard]] payload &payloads(uint32 x) {
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.payloads[x];
			}

			[[nodiscard]] payload const &payloads(uint32 x) const {
				if constexpr (is_interleaved)
					return store.nodes[x];
				else
					return store.payloads[x];
			}

			[[nodiscard]] uint32 child(uint32 x, bool side) const { return links(x).child[side]; }

			[[nodiscard]] uint32 parent(uint32 x) const { return links(x).parent; }

			[[nodiscard]] uint32 size(uint32 x) const { return links(x).size; }

			void set_child(uint32 x, bool side, uint32 c) { links(x).child[side] = c; }

			void set_parent(uint32 x, uint32 p) { links(x).parent = p; }

			[[nodiscard]] Key const &key(uint32 x) const { return payloads(x).key; }

			[[nodiscard]] slot<Value> &value(uint32 x)
				requires has_value
			{
				return payloads(x).value;
			}

			[[nodiscard]] fold_type const &aggregate(uint32 x) const
				requires has_op
			{
				return payloads(x).aggregate;
			}

			/** @brief What the fold reads from a node: its value (map) or key (set). */
			[[nodiscard]] static fold_type const &fold_of(payload const &pl)
				requires has_op
			{
				if constexpr (has_value)
					return pl.value;
				else
					return pl.key;
			}

			[[nodiscard]] fold_type const &fold_of(uint32 x) const
				requires has_op
			{
				return fold_of(payloads(x));
			}

			/** @brief Size and aggregate from the children; branchless through null sides. */
			void pull(uint32 x) {
				link &l = links(x);
				uint32 const left = l.child[0], right = l.child[1];
				l.size = links(left).size + 1 + links(right).size;
				if constexpr (has_op) {
					payload &pl = payloads(x);
					pl.aggregate = fold_values(fold_values(payloads(left).aggregate, fold_of(pl)), payloads(right).aggregate);
				}
			}

			uint32 append_node() {
				if constexpr (is_interleaved) {
					store.nodes.emplace_back();
					return uint32(store.nodes.size() - 1);
				} else {
					store.links.emplace_back();
					store.payloads.emplace_back();
					return uint32(store.links.size() - 1);
				}
			}

			[[nodiscard]] std::size_t slot_count() const {
				if constexpr (is_interleaved)
					return store.nodes.size();
				else
					return store.links.size();
			}

			/** @brief Room for @p elements more nodes beyond the live ones and the recycled ids. */
			void reserve(std::size_t elements) {
				if (elements <= recycled.size()) return;
				atl_assert(elements - recycled.size() <= (std::numeric_limits<uint32>::max)() - slot_count());
				std::size_t const total = slot_count() + (elements - recycled.size());
				if constexpr (is_interleaved) {
					store.nodes.reserve(total);
				} else {
					store.links.reserve(total);
					store.payloads.reserve(total);
				}
			}

			/** @brief A fresh detached node holding @p k (and @p v on the map tier). */
			uint32 make(Key k, slot<Value> v = {}) {
				uint32 id;
				if (!recycled.empty()) {
					id = recycled.back();
					recycled.pop_back();
				} else {
					// ids are 32-bit; id 0 is the null sentinel
					atl_assert(slot_count() < (std::numeric_limits<uint32>::max)());
					id = append_node();
				}
				links(id) = {{0, 0}, 0, 1};
				payload &pl = payloads(id);
				pl.key = std::move(k);
				if constexpr (has_value) pl.value = std::move(v);
				if constexpr (has_op) pl.aggregate = fold_of(pl);
				return id;
			}

			/** @brief Return the single node @p x to the free list (its links are dead). */
			void free_node(uint32 x) {
				// a heap-owning key or value is released now, not when the id is reused
				if constexpr (!std::is_trivially_destructible_v<payload>) payloads(x) = payload{};
				recycled.push_back(x);
			}

			/**
			 * @brief Return every node of subtree @p root to the free list: a stackless
			 * preorder walk by parent pointers, links left intact until reuse.
			 */
			void recycle(uint32 root) {
				for (uint32 x = root; x != 0;) {
					if constexpr (!std::is_trivially_destructible_v<payload>) payloads(x) = payload{};
					recycled.push_back(x);
					if (uint32 const l = child(x, 0); l != 0) {
						x = l;
						continue;
					}
					if (uint32 const r = child(x, 1); r != 0) {
						x = r;
						continue;
					}
					for (;;) { // climb to the next unvisited right subtree
						if (x == root) return;
						uint32 const p = parent(x), r = child(p, 1);
						if (r != 0 && r != x) {
							x = r;
							break;
						}
						x = p;
					}
				}
			}
		};

		/**
		 * @brief The engine operations a splay map calls: `descend`, `splay`,
		 *        `splay_at`, `join` and `for_each` over its pool.
		 */
		template<typename Engine, typename Store>
		concept splay_map_engine = requires(Store &s, uint32 x) {
			{
				Engine::descend(s, x, [](uint32) { return splay_step::stop; })
			} -> std::same_as<uint32>;
			{ Engine::splay_at(s, x, x) } -> std::same_as<uint32>;
			{ Engine::join(s, x, x) } -> std::same_as<uint32>;
			Engine::splay(s, x);
			Engine::for_each(s, x, [](uint32) {});
			{ Engine::build_balanced(s, std::vector<uint32>{}) } -> std::same_as<uint32>;
		};

		struct splay_map_adopt_t {
			explicit splay_map_adopt_t() = default;
		};

		/**
		 * @brief The whole `atl::splay_map`, over any engine and node layout
		 *        (CRTP: @p Derived is the class handed back by `split` and
		 *        `sibling`), so the experimental contenders share every line.
		 */
		template<typename Derived, typename Engine, typename Layout, std::semiregular Key, typename Value, typename Compare,
				 typename Op>
			requires splay_map_layout<Layout> && splay_map_parameters<Key, Value, Compare, Op>
				  && splay_map_engine<Engine, splay_map_pool<Layout, Key, Value, Compare, Op>>
		class splay_map_impl {
			using pool = splay_map_pool<Layout, Key, Value, Compare, Op>;
			static constexpr bool has_value = pool::has_value;
			static constexpr bool has_op = pool::has_op;
			using fold_type = typename pool::fold_type;

			/** @brief The pool's algebra-member slot. */
			template<typename X>
			using slot = typename pool::template slot<X>;

			std::shared_ptr<pool> pool_;
			uint32 root_ = 0;

		public:
			/** @brief Stored key type. */
			using key_type = Key;
			/** @brief Type used for positions and element counts. */
			using size_type = std::size_t;

			/** @brief A key without a mapped value: what the set tier's `kth`/`next`/`prev` return. */
			struct set_entry {
				Key const &key; ///< Stored key; invalidated when its node is erased or the pool reallocates.
			};

			/** @brief A key with its mapped value: what the map tier's `kth`/`next`/`prev` return. */
			struct map_entry {
				Key const &key;			  ///< Stored key; invalidated when its node is erased or the pool reallocates.
				slot<Value> const &value; ///< Mapped value; shares the key reference's lifetime.
			};

			/**
			 * @brief A stored entry, by reference into the pool: `{key}` for a set,
			 *        `{key, value}` for a map. Invalidated like any returned reference
			 *        (see the class note).
			 */
			using entry = std::conditional_t<has_value, map_entry, set_entry>;

		private:
			/**
			 * @brief The keys in `[lo, hi)` after `isolate:` `low` (the root) if
			 *        `is_low_inside`, then the whole left subtree of `high` (the
			 *        root's right child, 0 when there is none), then `high` if
			 *        `is_high_inside`.
			 */
			struct key_range {
				uint32 low = 0, high = 0;
				bool is_low_inside = false, is_high_inside = false;
			};

			/**
			 * @brief @p k as the comparator should see it: by reference when it is a
			 *        `Key` or a heterogeneous key, otherwise converted to one `Key`.
			 */
			template<ordered_lookup_key<Key, Compare> Lookup>
			static decltype(auto) as_lookup(Lookup const &k) {
				if constexpr (std::same_as<Lookup, Key> || heterogeneous_key_for<Lookup, Key, Compare>)
					return (k);
				else
					return Key(k);
			}

			/** @brief The entry of node @p x, by reference into the pool. */
			entry entry_of(uint32 x) const {
				if constexpr (has_value)
					return {pool_->key(x), pool_->value(x)};
				else
					return {pool_->key(x)};
			}

			/**
			 * @brief Search for @p k and splay the last node visited: the node
			 *        holding @p k if present (returned), otherwise its in-order
			 *        neighbour where the search fell off (0 returned).
			 */
			template<ordered_lookup_key<Key, Compare> Lookup>
			uint32 splay_exact(Lookup const &k) {
				pool &p = *pool_;
				uint32 hit = 0;
				root_ = Engine::descend(p, root_, [&p, &k, &hit](uint32 x) {
					Key const &key = p.key(x);
					if (p.compare_keys(k, key)) return splay_step::left;
					if (p.compare_keys(key, k)) return splay_step::right;
					hit = x;
					return splay_step::stop;
				});
				return hit;
			}

			/**
			 * @brief Splay the node holding @p k, or else the neighbour of @p k
			 *        the search fell off at: either the largest key `< k` or the
			 *        smallest key `> k`. Returns the new root (0 when empty).
			 */
			template<ordered_lookup_key<Key, Compare> Lookup>
			uint32 splay_bound(uint32 root, Lookup const &k) {
				pool &p = *pool_;
				return Engine::descend(p, root, [&p, &k](uint32 x) {
					Key const &key = p.key(x);
					if (p.compare_keys(key, k)) return splay_step::right;
					return p.compare_keys(k, key) ? splay_step::left : splay_step::stop;
				});
			}

			/** @brief Cut x's child on @p side loose and return it. */
			uint32 detach(uint32 x, bool side) {
				pool &p = *pool_;
				uint32 const c = p.child(x, side);
				if (c == 0) return 0;
				p.set_child(x, side, 0);
				p.set_parent(c, 0);
				p.pull(x);
				return c;
			}

			/**
			 * @brief Make the fresh node @p x the root, after a search for its key
			 *        missed: the old root (the neighbour the search fell off at)
			 *        becomes its child, the old root's subtree beyond x's key moves
			 *        under x.
			 */
			void link_root(uint32 x) {
				pool &p = *pool_;
				if (uint32 const r = root_; r != 0) {
					bool const side = p.compare_keys(p.key(r), p.key(x)); // r before x: r goes left
					uint32 const beyond = p.child(r, side);
					p.set_child(r, side, 0);
					p.set_child(x, !side, r);
					p.set_parent(r, x);
					p.set_child(x, side, beyond);
					if (beyond != 0) p.set_parent(beyond, x);
					p.pull(r);
				}
				p.pull(x);
				root_ = x;
			}

			/** @brief Remove the root node, joining its two subtrees. */
			void erase_root() {
				pool &p = *pool_;
				uint32 const x = root_, l = p.child(x, 0), r = p.child(x, 1);
				if (l != 0) p.set_parent(l, 0);
				if (r != 0) p.set_parent(r, 0);
				root_ = Engine::join(p, l, r);
				p.free_node(x);
			}

			/**
			 * @brief Gather the keys in `[lo, hi)` with two splays: the bound of
			 *        @p lo to the root, then the bound of @p hi to the top of the
			 *        root's right subtree (searched as a tree of its own). The set
			 *        of nodes under each touched node is unchanged, so nothing
			 *        needs a pull until the caller cuts something out.
			 */
			template<ordered_lookup_key<Key, Compare> Lo, ordered_lookup_key<Key, Compare> Hi>
			key_range isolate(Lo const &lo, Hi const &hi) {
				pool &p = *pool_;
				key_range range;
				root_ = splay_bound(root_, lo);
				if (root_ == 0) return range;
				range.low = root_;
				if (!p.compare_keys(p.key(root_), lo)) { // the root is the smallest key >= lo
					if (!p.compare_keys(p.key(root_), hi)) return range;
					range.is_low_inside = true;
				}
				// either way the right subtree holds exactly the other keys >= lo
				uint32 const rest = p.child(root_, 1);
				if (rest == 0) return range;
				p.set_parent(rest, 0); // the splay below must stop at rest's root
				uint32 const high = splay_bound(rest, hi);
				p.set_child(root_, 1, high);
				p.set_parent(high, root_);
				range.high = high;
				range.is_high_inside = p.compare_keys(p.key(high), hi);
				return range;
			}

			/** @brief Visit node @p x: `visit(key)` (set) or `visit(key, value)` (map). */
			template<typename Visit>
				requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &>
								   : std::invocable<Visit &, Key const &>)
			void visit_node(Visit &visit, uint32 x) const {
				if constexpr (has_value)
					std::invoke(visit, pool_->key(x), std::as_const(pool_->value(x)));
				else
					std::invoke(visit, pool_->key(x));
			}

			/** @brief A copy of node @p x of the shared pool (copied out before the pool may grow). */
			uint32 copy_node(uint32 x) {
				pool &p = *pool_;
				Key k = p.key(x);
				if constexpr (has_value) {
					slot<Value> v = p.value(x);
					return p.make(std::move(k), std::move(v));
				} else {
					return p.make(std::move(k));
				}
			}

			/** @brief A balanced copy of @p o's tree in this map's pool. @complexity O(n) */
			uint32 clone_from(splay_map_impl const &o) {
				std::vector<uint32> ids;
				ids.reserve(o.size());
				// collected first: copying allocates, and the two maps share the pool
				Engine::for_each(*o.pool_, o.root_, [&ids](uint32 x) { ids.push_back(x); });
				for (uint32 &x : ids) x = copy_node(x);
				return Engine::build_balanced(*pool_, ids);
			}

			/**
			 * @brief Linear-time build from a range sorted by the comparator with no
			 *        duplicates: the nodes in order, linked perfectly balanced.
			 */
			template<std::ranges::input_range R>
				requires(has_value ? key_value_entry<forwarded_element_t<R>, Key, slot<Value>> : std::convertible_to<forwarded_element_t<R>, Key>)
			void build_sorted(R &&entries) {
				pool &p = *pool_;
				std::vector<uint32> ids;
				if constexpr (std::ranges::sized_range<R>) {
					p.reserve(std::ranges::size(entries));
					ids.reserve(std::ranges::size(entries));
				}
				for (auto &&e : entries) {
					auto &&element = forward_element<R>(std::forward<decltype(e)>(e));
					uint32 id;
					if constexpr (has_value)
						id = p.make(Key(std::get<0>(std::forward<decltype(element)>(element))),
									Value(std::get<1>(std::forward<decltype(element)>(element))));
					else
						id = p.make(Key(std::forward<decltype(element)>(element)));
					atl_assert(ids.empty() || p.compare_keys(p.key(ids.back()), p.key(id)));
					ids.push_back(id);
				}
				root_ = Engine::build_balanced(p, ids);
			}

		public:
			/** @brief Adopt subtree @p root of @p shared (the `split`/`sibling` path). @note Not constexpr (uses a shared_ptr node pool). */
			splay_map_impl(splay_map_adopt_t, std::shared_ptr<pool> shared, uint32 root)
				: pool_(std::move(shared)), root_(root) {}

			/**
			 * @brief Not available on the fold tier: a fold needs its identity, which
			 * only the `(identity, op)` constructor supplies.
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			splay_map_impl()
				requires(!has_op && std::default_initializable<Compare>)
				: pool_(std::make_shared<pool>()) {}

			/** @brief Construct an empty tree with the supplied comparator. @param comp Key order. @note Not constexpr (uses a shared_ptr node pool). */
			explicit splay_map_impl(Compare comp)
				requires(!has_op)
				: pool_(std::make_shared<pool>(detail::move_or_copy(comp))) {}

			/**
			 * @brief Fold tier: supply the monoid `(op, identity)` the key-range
			 *        queries fold with. The template only gates the tier; both
			 *        parameters are non-deduced, so arguments convert as usual.
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename OpArg = Op>
				requires(has_op && std::same_as<OpArg, Op>)
			splay_map_impl(fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})
				: pool_(std::make_shared<pool>(detail::move_or_copy(comp), detail::move_or_copy(op), std::move(identity))) {}

			/**
			 * @brief Build from @p entries — keys for a set, `(key, value)` pairs for a
			 *        map — already sorted by @p comp with no duplicate keys, in O(n)
			 *        instead of n inserts, perfectly balanced. Elements are moved out
			 *        of an owning rvalue range and copied otherwise.
			 * @pre @p entries is strictly increasing under @p comp (asserted).
			 * @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<std::ranges::input_range R>
				requires(!has_op
						 && (has_value ? key_value_entry<forwarded_element_t<R>, Key, slot<Value>>
									   : std::convertible_to<forwarded_element_t<R>, Key>))
			splay_map_impl(sorted_unique_t, R &&entries, Compare comp = {}) : splay_map_impl(detail::move_or_copy(comp)) {
				build_sorted(std::forward<R>(entries));
			}

			/**
			 * @brief Build from a sorted unique range on the fold tier.
			 * Fold tier: also supplies the monoid, as the `(identity, op)` constructor.
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<std::ranges::input_range R, typename OpArg = Op>
				requires(has_op && std::same_as<OpArg, Op>
						 && (has_value ? key_value_entry<forwarded_element_t<R>, Key, slot<Value>>
									   : std::convertible_to<forwarded_element_t<R>, Key>))
			splay_map_impl(sorted_unique_t, R &&entries, fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})
				: splay_map_impl(std::move(identity), detail::move_or_copy(op), detail::move_or_copy(comp)) {
				build_sorted(std::forward<R>(entries));
			}

			/** @brief Clone the source nodes into its shared pool. @param o Source handle. @complexity O(source size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_map_impl(splay_map_impl const &o) : pool_(o.pool_), root_(o.pool_ ? clone_from(o) : 0) {}

			/** @brief Transfer the pool and tree root. @param o Source, empty on return. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			splay_map_impl(splay_map_impl &&o) noexcept : pool_(std::move(o.pool_)), root_(o.root_) { o.root_ = 0; }

			/** @brief Recycle this tree, then clone the source into its pool. @param o Source tree. @return `*this`. @complexity O(destination size + source size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_map_impl &operator=(splay_map_impl const &o) {
				if (this != &o) {
					if (pool_) pool_->recycle(root_);
					pool_ = o.pool_;
					root_ = pool_ ? clone_from(o) : 0;
				}
				return *this;
			}

			/** @brief Recycle this tree, then transfer the source handle. @param o Source, empty on return. @return `*this`. @complexity O(destination size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_map_impl &operator=(splay_map_impl &&o) noexcept {
				if (this != &o) {
					if (pool_) pool_->recycle(root_);
					pool_ = std::move(o.pool_);
					root_ = o.root_;
					o.root_ = 0;
				}
				return *this;
			}

			/** @brief Recycle this map's nodes into the shared pool. @note Not constexpr (uses a shared_ptr node pool). */
			~splay_map_impl() {
				if (pool_) pool_->recycle(root_);
			}

			/** @brief Number of elements in this handle; zero for a moved-from handle. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			std::size_t size() const noexcept { return pool_ ? pool_->size(root_) : 0; }

			/** @brief Whether this handle contains no elements. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			bool empty() const noexcept { return size() == 0; }

			/**
			 * @brief Make room in the (shared) pool for @p elements more keys, so the
			 *        next @p elements insertions never regrow the node storage.
			 * @complexity O(pool size) when it reallocates
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void reserve(std::size_t elements) {
				atl_assert(pool_ != nullptr);
				pool_->reserve(elements);
			}

			/** @brief Whether @p k is present; splays. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			[[nodiscard]] bool contains(Lookup const &k) {
				return pool_ && splay_exact(as_lookup(k)) != 0;
			}

			/**
			 * @brief Insert @p key (set tier); returns true if newly added.
			 * @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			bool insert(Key key)
				requires(!has_value)
			{
				atl_assert(pool_ != nullptr);
				if (splay_exact(key) != 0) return false;
				link_root(pool_->make(std::move(key)));
				return true;
			}

			/**
			 * @brief Insert `{key, value}` (map tier); an existing value is kept. Returns
			 * true if newly inserted. @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename V = Value>
				requires(has_value && std::same_as<V, Value>)
			bool insert(Key key, std::type_identity_t<V> value) {
				atl_assert(pool_ != nullptr);
				if (splay_exact(key) != 0) return false;
				link_root(pool_->make(std::move(key), std::move(value)));
				return true;
			}

			/**
			 * @brief Insert or overwrite (map tier); returns true if newly inserted.
			 * @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename V = Value>
				requires(has_value && std::same_as<V, Value>)
			bool insert_or_assign(Key key, std::type_identity_t<V> value) {
				atl_assert(pool_ != nullptr);
				if (uint32 const hit = splay_exact(key); hit != 0) {
					pool_->value(hit) = std::move(value);
					if constexpr (has_op) pool_->pull(hit); // the hit is the root: only its aggregate is stale
					return false;
				}
				link_root(pool_->make(std::move(key), std::move(value)));
				return true;
			}

			/**
			 * @brief Replace the value at @p key by `f(value)` (map tier), inserting
			 *        `f(Value{})` when @p key is absent — `m[k] += d` in one
			 *        descent. Returns the new value.
			 * @param f Invoked exactly once.
			 * @complexity amortized O(log n)
			 */
			template<typename Update, typename V = Value>
				requires(has_value && std::same_as<V, Value> && update_function<Update, V>)
			V const &update(Key key, Update &&f) {
				atl_assert(pool_ != nullptr);
				if (uint32 const hit = splay_exact(key); hit != 0) {
					pool_->value(hit) = std::invoke(f, std::as_const(pool_->value(hit)));
					if constexpr (has_op) pool_->pull(hit);
					return pool_->value(hit);
				}
				Value const fresh{};
				Value value = std::invoke(f, fresh);
				link_root(pool_->make(std::move(key), std::move(value)));
				return pool_->value(root_);
			}

			/**
			 * @brief The mapped value at @p k (map tier); splays. @pre `contains(k)`
			 * (asserted) @complexity amortized O(log n)
			 */
			template<ordered_lookup_key<Key, Compare> Lookup = Key, typename V = Value>
				requires(has_value && std::same_as<V, Value>)
			[[nodiscard]] V const &at(Lookup const &k) {
				atl_assert(pool_ != nullptr);
				uint32 const hit = splay_exact(as_lookup(k));
				atl_assert(hit != 0);
				return pool_->value(hit);
			}

			/**
			 * @brief The mapped value at @p k (map tier), or `nullptr` if absent — the
			 *        non-asserting counterpart of `at`; splays.
			 * @complexity amortized O(log n)
			 */
			template<ordered_lookup_key<Key, Compare> Lookup = Key, typename V = Value>
				requires(has_value && std::same_as<V, Value>)
			[[nodiscard]] V const *find(Lookup const &k) {
				if (!pool_) return nullptr;
				uint32 const hit = splay_exact(as_lookup(k));
				return hit != 0 ? &pool_->value(hit) : nullptr;
			}

			/** @brief Remove @p k; returns true if it was present. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			bool erase(Lookup const &k) {
				atl_assert(pool_ != nullptr);
				if (splay_exact(as_lookup(k)) == 0) return false;
				erase_root();
				return true;
			}

			/**
			 * @brief Remove every key in `[lo, hi)`; returns how many were removed
			 *        (0 when the range is empty).
			 * @complexity amortized O(log n + removed)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<ordered_lookup_key<Key, Compare> Lo = Key, ordered_lookup_key<Key, Compare> Hi = Key>
			std::size_t erase(Lo const &lo, Hi const &hi) {
				atl_assert(pool_ != nullptr);
				pool &p = *pool_;
				// no lo < hi test: lo and hi may be two different heterogeneous types
				// the comparator cannot order against each other
				key_range const range = isolate(as_lookup(lo), as_lookup(hi));
				if (range.low == 0) return 0;
				std::size_t removed = 0;
				if (uint32 const high = range.high; high != 0) {
					if (uint32 const inner = p.child(high, 0); inner != 0) {
						removed += p.size(inner);
						p.set_child(high, 0, 0);
						p.recycle(inner);
					}
					if (range.is_high_inside) {
						uint32 const beyond = p.child(high, 1);
						p.set_child(range.low, 1, beyond);
						if (beyond != 0) p.set_parent(beyond, range.low);
						p.free_node(high);
						++removed;
					} else {
						p.pull(high);
					}
				}
				if (range.is_low_inside) {
					erase_root();
					++removed;
				} else {
					p.pull(range.low);
				}
				return removed;
			}

			/**
			 * @brief Cut this map at @p k: keys `< k` stay, keys `>= k` move to the
			 *        returned map, which shares this map's pool.
			 * @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			[[nodiscard]] Derived split(Lookup const &k) {
				atl_assert(pool_ != nullptr);
				auto const &key = as_lookup(k);
				root_ = splay_bound(root_, key);
				if (root_ == 0) return Derived(splay_map_adopt_t{}, pool_, 0);
				if (pool_->compare_keys(pool_->key(root_), key)) return Derived(splay_map_adopt_t{}, pool_, detach(root_, 1));
				uint32 const upper = root_;
				root_ = detach(upper, 0);
				return Derived(splay_map_adopt_t{}, pool_, upper);
			}

			/**
			 * @brief Number of keys strictly before @p k (also `k`'s index if present);
			 * splays. @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			[[nodiscard]] std::size_t rank(Lookup const &k) {
				if (!pool_) return 0;
				auto const &key = as_lookup(k);
				root_ = splay_bound(root_, key);
				if (root_ == 0) return 0;
				return pool_->size(pool_->child(root_, 0)) + std::size_t(pool_->compare_keys(pool_->key(root_), key));
			}

			/**
			 * @brief The @p i-th smallest entry (`.key`, plus `.value` on the map
			 *        tier); splays it to the root.
			 * @pre `i < size()` (asserted) @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			[[nodiscard]] entry kth(std::size_t i) {
				atl_assert(pool_ && i < size());
				root_ = Engine::splay_at(*pool_, root_, uint32(i));
				return entry_of(root_);
			}

			/** @brief The entry with the smallest key `>= k`, if any; splays it. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			[[nodiscard]] std::optional<entry> next(Lookup const &k) {
				if (!pool_) return std::nullopt;
				pool &p = *pool_;
				auto const &key = as_lookup(k);
				uint32 best = 0;
				root_ = Engine::descend(p, root_, [&p, &key, &best](uint32 x) {
					if (p.compare_keys(p.key(x), key)) return splay_step::right;
					best = x;
					return p.compare_keys(key, p.key(x)) ? splay_step::left : splay_step::stop;
				});
				if (best == 0) return std::nullopt;
				if (best != root_) { // the search fell off right of best's predecessor
					Engine::splay(p, best);
					root_ = best;
				}
				return entry_of(best);
			}

			/** @brief The entry with the largest key `<= k`, if any; splays it. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			template<ordered_lookup_key<Key, Compare> Lookup = Key>
			[[nodiscard]] std::optional<entry> prev(Lookup const &k) {
				if (!pool_) return std::nullopt;
				pool &p = *pool_;
				auto const &key = as_lookup(k);
				uint32 best = 0;
				root_ = Engine::descend(p, root_, [&p, &key, &best](uint32 x) {
					if (p.compare_keys(key, p.key(x))) return splay_step::left;
					best = x;
					return p.compare_keys(p.key(x), key) ? splay_step::right : splay_step::stop;
				});
				if (best == 0) return std::nullopt;
				if (best != root_) {
					Engine::splay(p, best);
					root_ = best;
				}
				return entry_of(best);
			}

			/**
			 * @brief Fold of the values (keys, for a set) of every key in `[lo, hi)`, or
			 * the identity if none; two splays. @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<ordered_lookup_key<Key, Compare> Lo = Key, ordered_lookup_key<Key, Compare> Hi = Key>
				requires(has_op)
			[[nodiscard]] fold_type fold(Lo const &lo, Hi const &hi) {
				atl_assert(pool_ != nullptr);
				pool &p = *pool_;
				key_range const range = isolate(as_lookup(lo), as_lookup(hi));
				if (range.high == 0) return range.is_low_inside ? p.fold_of(range.low) : p.identity;
				// the null sentinel's aggregate is the identity
				fold_type inner = p.aggregate(p.child(range.high, 0));
				if (range.is_low_inside) inner = p.fold_values(p.fold_of(range.low), inner);
				if (range.is_high_inside) inner = p.fold_values(inner, p.fold_of(range.high));
				return inner;
			}

			/** @brief Fold of every value (key, for a set). @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			[[nodiscard]] fold_type fold() const
				requires(has_op)
			{
				atl_assert(pool_ != nullptr);
				return pool_->aggregate(root_); // the null sentinel holds the identity
			}

			/**
			 * @brief In-order visit: `visit(key)` (set) or `visit(key, value)` (map); const,
			 * stackless. @p visit must not touch this map. @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename Visit>
				requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &>
								   : std::invocable<Visit &, Key const &>)
			void for_each(Visit visit) const {
				if (!pool_ || root_ == 0) return;
				Engine::for_each(*pool_, root_, [&](uint32 x) { visit_node(visit, x); });
			}

			/**
			 * @brief In-order visit of the keys in `[lo, hi)` only: `visit(key)` (set)
			 *        or `visit(key, value)` (map), after two splays; subtrees outside
			 *        the range are never entered. @p visit must not touch this map.
			 * @complexity amortized O(log n + visited)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<ordered_lookup_key<Key, Compare> Lo, ordered_lookup_key<Key, Compare> Hi, typename Visit>
				requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &>
								   : std::invocable<Visit &, Key const &>)
			void for_each(Lo const &lo, Hi const &hi, Visit visit) {
				if (!pool_) return;
				key_range const range = isolate(as_lookup(lo), as_lookup(hi));
				if (range.is_low_inside) visit_node(visit, range.low);
				if (range.high == 0) return;
				if (uint32 const inner = pool_->child(range.high, 0); inner != 0)
					Engine::for_each(*pool_, inner, [&](uint32 x) { visit_node(visit, x); });
				if (range.is_high_inside) visit_node(visit, range.high);
			}

			/** @brief Empty the map, recycling its nodes. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
			void clear() {
				if (pool_) pool_->recycle(root_);
				root_ = 0;
			}

			/** @brief An empty map sharing this map's pool, in O(1). @note Not constexpr (uses a shared_ptr node pool). */
			[[nodiscard]] Derived sibling() const {
				atl_assert(pool_ != nullptr);
				return Derived(splay_map_adopt_t{}, pool_, 0);
			}
		};

		/** @brief Split storage separates navigation from payload; measurements are in docs/splay-map.md. */
		using splay_map_default_layout = splay_layout_split;
	}

	/**
	 * @brief Splay-tree ordered map: order statistics and value folds over key
	 *        ranges in amortized O(log n), adapting to the access pattern — the
	 *        drop-in counterpart of `treap_map`.
	 *
	 * The same template, tiers and members as `treap_map` (`merge_union`
	 * aside), so switching is a type change. Every access splays the touched
	 * key to the root, which adapts to skewed access patterns and sequential key walks.
	 * Uniformly random accesses also pay for restructuring. Deterministic: no seed.
	 *
	 * Differences from `treap_map`, stated up front: every lookup restructures
	 * the tree, so `contains`, `find`, `at`, `rank`, `kth`, `next`, `prev`,
	 * `fold(lo, hi)` and `for_each(lo, hi, visit)` are non-const; `size`,
	 * `empty`, `fold()` and `for_each(visit)` stay const. A copy is rebuilt
	 * perfectly balanced. `merge_union` is not provided.
	 *
	 * Lookups accept any key-like argument; when `Compare::is_transparent` is
	 * present (the default `std::ranges::less` has it) an argument the
	 * comparator orders against `Key` directly is compared without building a
	 * `Key`, except a mixed pair of arithmetic types, which converts to `Key`.
	 *
	 * `splay_set<Key>` is the mapped-value-free alias.
	 *
	 * @note Maps sharing a pool (copies do; `sibling` and `split` results
	 *       do) share one node array: a reference returned by `at`, `find`,
	 *       `kth`, `next`, `prev` or `update` is invalidated by any
	 *       node-allocating call on any of them, and by erasing its key.
	 *
	 * @tparam Key     The key type.
	 * @tparam Value   The mapped type, or `void` for a set.
	 * @tparam Compare A strict weak order over `Key` (default `<`).
	 * @tparam Op      Optional fold over mapped values (values for a map, keys
	 *                 for a set): any `op(T, T)` convertible to `T`.
	 * @tparam Layout  The node layout: `splay_layout_split` (default) or
	 *                 `splay_layout_interleaved` (see docs/splay-map.md).
	 */
	template<std::semiregular Key, typename Value = void, typename Compare = std::ranges::less, typename Op = void,
			 typename Layout = detail::splay_map_default_layout>
		requires detail::splay_map_parameters<Key, Value, Compare, Op> && detail::splay_map_layout<Layout>
	class splay_map
		: public detail::splay_map_impl<splay_map<Key, Value, Compare, Op, Layout>, splay_engine, Layout, Key, Value, Compare, Op> {
		using base = detail::splay_map_impl<splay_map, splay_engine, Layout, Key, Value, Compare, Op>;

	public:
		/** @brief Inherit the tier-specific constructors. */
		using base::base;
	};

	/** @brief Ordered set of keys: `splay_map` without mapped values. */
	template<typename Key, typename Compare = std::ranges::less, typename Op = void,
			 detail::splay_map_layout Layout = detail::splay_map_default_layout>
	using splay_set = splay_map<Key, void, Compare, Op, Layout>;
}
