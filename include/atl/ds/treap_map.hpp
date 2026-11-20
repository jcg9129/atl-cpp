#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/ds/detail/key_value_entry.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/tags.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
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
	/**
	 * @brief Keyed treap: an ordered map with order statistics and value folds
	 *        over key ranges, in expected O(log n).
	 *
	 * The ordered-set workhorse for keys `hierarchical_bitset` cannot serve
	 * (arbitrary ordered types, unbounded universes): insert/erase/lookup by key,
	 * `rank`/`kth`, predecessor/successor, key-range split/erase/visit, and —
	 * when an `Op` is supplied — `fold(lo, hi)` folding the *mapped values* of
	 * every key in `[lo, hi)`, read-only. Keys are unique (map semantics). Built
	 * on the same `detail::treap_base` engine as `treap_sequence`.
	 *
	 * Lookups accept any key-like argument; when `Compare::is_transparent` is
	 * present (the default `std::ranges::less` has it) an argument the
	 * comparator orders against `Key` directly is compared without building a
	 * `Key` (a `std::string_view` against `std::string` keys).
	 *
	 * `treap_set<Key>` is the mapped-value-free alias.
	 *
	 * @note Maps sharing a pool (copies do: their cloned nodes go to the
	 *       source's pool; `sibling` and `split` results do) share one
	 *       node array: a reference returned by `at`,
	 *       `find`, `kth`, `next`, `prev` or `update` is invalidated by any
	 *       node-allocating call on any of them.
	 *
	 * @tparam Key     The key type.
	 * @tparam Value   The mapped type, or `void` for a set.
	 * @tparam Compare A strict weak order over `Key` (default `<`).
	 * @tparam Op      Optional fold over mapped values (values for a map, keys
	 *                 for a set): any `op(T, T)` convertible to `T`.
	 */
	template<std::semiregular Key, typename Value = void, typename Compare = std::ranges::less, typename Op = void>
		requires(std::strict_weak_order<Compare const &, Key const &, Key const &>
				 && (std::is_void_v<Value> || std::semiregular<Value>)
				 && (std::is_void_v<Op>
					 || (std::is_void_v<Value> ? binary_operation<Op, Key> : binary_operation<Op, Value>)))
	class treap_map {
		static constexpr bool has_value = !std::is_void_v<Value>;
		static constexpr bool has_op = !std::is_void_v<Op>;

		/** @brief What Op folds over: mapped values for a map, the keys for a set. */
		using fold_type = std::conditional_t<has_value, std::conditional_t<has_value, Value, Key>, Key>;

		struct no_slot {};

		struct no_value_slot {};

		struct no_aggregate_slot {};

		struct value_slot {
			std::conditional_t<has_value, Value, no_slot> value;
		};

		struct aggregate_slot {
			std::conditional_t<has_op, fold_type, no_slot> aggregate;
		};

		struct node : std::conditional_t<has_value, value_slot, no_value_slot>,
					  std::conditional_t<has_op, aggregate_slot, no_aggregate_slot> {
			Key key;
			uint32 left = 0, right = 0, size = 1, priority = 0;
		};

		template<typename X>
		using slot = std::conditional_t<std::is_void_v<X>, no_slot, X>;

		struct pool {
			std::vector<node> nodes;
			std::vector<uint32> recycled;
			uint64 rng;
			[[no_unique_address]] Compare comp;
			[[no_unique_address]] slot<Op> op;
			[[no_unique_address]] std::conditional_t<has_op, fold_type, no_slot> id;

			pool()
				requires std::default_initializable<Compare> && std::default_initializable<slot<Op>>
				: rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), comp{}, op{}, id{} {
				nodes.emplace_back();
				nodes[0].size = 0;
				if constexpr (has_op) nodes[0].aggregate = id;
			}

			explicit pool(Compare compare, slot<Op> fold = {}, std::conditional_t<has_op, fold_type, no_slot> identity = {})
				: rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), comp(detail::move_or_copy(compare)), op(detail::move_or_copy(fold)), id(std::move(identity)) {
				nodes.emplace_back(); // id 0 = null: size 0
				nodes[0].size = 0;
				if constexpr (has_op) nodes[0].aggregate = id;
			}

			fold_type const &fold_type_of(node const &nd) const
				requires(has_op)
			{
				if constexpr (has_value)
					return nd.value;
				else
					return nd.key;
			}

			decltype(auto) fold_values(auto &&a, auto &&b) const
				requires has_op && requires { detail::invoke_algebra<fold_type>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<fold_type>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			template<typename Left, typename Right>
				requires std::predicate<Compare const &, Left const &, Right const &>
			bool compare_keys(Left const &a, Right const &b) const { return std::invoke(comp, a, b); }

			uint32 left(uint32 t) const { return nodes[t].left; }

			uint32 right(uint32 t) const { return nodes[t].right; }

			uint32 size(uint32 t) const { return nodes[t].size; }

			uint32 priority(uint32 t) const { return nodes[t].priority; }

			void pull(uint32 t) {
				node &nd = nodes[t];
				nd.size = 1 + nodes[nd.left].size + nodes[nd.right].size;
				if constexpr (has_op)
					nd.aggregate = fold_values(fold_values(nodes[nd.left].aggregate, fold_type_of(nd)), nodes[nd.right].aggregate);
			}

			uint32 rebind(uint32 t, uint32 l, uint32 r) {
				nodes[t].left = l;
				nodes[t].right = r;
				pull(t);
				return t;
			}

			uint32 rebind_left(uint32 t, uint32 l) {
				nodes[t].left = l;
				pull(t);
				return t;
			}

			uint32 rebind_right(uint32 t, uint32 r) {
				nodes[t].right = r;
				pull(t);
				return t;
			}

			uint32 push(uint32 t) { return t; } // no tags in the keyed treap

			/** @brief Room for @p extra more nodes beyond the live ones and the recycled ids. */
			void reserve(std::size_t extra) {
				if (extra > recycled.size()) {
					atl_assert(extra - recycled.size() <= (std::numeric_limits<uint32>::max)() - nodes.size());
					nodes.reserve(nodes.size() + (extra - recycled.size()));
				}
			}

			uint32 make(Key key) {
				uint32 nid;
				if (!recycled.empty()) {
					nid = recycled.back();
					recycled.pop_back();
				} else {
					// ids are 32-bit; id 0 is the null sentinel
					atl_assert(nodes.size() < (std::numeric_limits<uint32>::max)());
					nodes.emplace_back();
					nid = uint32(nodes.size() - 1);
				}
				node &nd = nodes[nid];
				nd.key = std::move(key);
				nd.left = nd.right = 0;
				nd.size = 1;
				nd.priority = uint32(detail::splitmix64(rng += 0x9e3779b97f4a7c15ULL) >> 32);
				if constexpr (has_op && !has_value) nd.aggregate = nd.key;
				return nid;
			}

			void recycle(uint32 t) {
				if (t == 0) return;
				recycle(nodes[t].left);
				recycle(nodes[t].right);
				recycled.push_back(t);
			}

			uint32 clone(uint32 t) {
				if (t == 0) return 0;
				uint32 const l = clone(nodes[t].left), r = clone(nodes[t].right);
				uint32 const nid = make(nodes[t].key);
				node &nd = nodes[nid];
				nd.priority = nodes[t].priority;
				if constexpr (has_value) nd.value = nodes[t].value;
				nd.left = l;
				nd.right = r;
				pull(nid);
				return nid;
			}
		};

		using base = detail::treap_base;

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		treap_map(std::shared_ptr<pool> shared, uint32 root) : pool_(std::move(shared)), root_(root) {}

		/**
		 * @brief @p k as the comparator should see it: by reference when it is a
		 *        `Key` or a heterogeneous key, otherwise converted to one `Key`.
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		static decltype(auto) as_lookup(Lookup const &k) {
			if constexpr (std::same_as<Lookup, Key> || detail::heterogeneous_key_for<Lookup, Key, Compare>)
				return (k);
			else
				return Key(k);
		}

		/** @brief Split subtree @p t into (keys < k, keys >= k). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		std::pair<uint32, uint32> split_key(uint32 t, Lookup const &k) {
			return base::split(*pool_, t, [&](pool &p, uint32 x) { return p.compare_keys(p.nodes[x].key, k); });
		}

		/** @brief Split subtree @p t into (keys <= k, keys > k). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		std::pair<uint32, uint32> split_key_inclusive(uint32 t, Lookup const &k) {
			return base::split(*pool_, t, [&](pool &p, uint32 x) { return !p.compare_keys(k, p.nodes[x].key); });
		}

		/** @brief Node id holding exactly @p k, or 0. Read-only descent. */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		uint32 find_node(Lookup const &k) const {
			if (!pool_) return 0;
			uint32 t = root_;
			while (t != 0) {
				node const &nd = pool_->nodes[t];
				if (pool_->compare_keys(k, nd.key))
					t = nd.left;
				else if (pool_->compare_keys(nd.key, k))
					t = nd.right;
				else
					return t;
			}
			return 0;
		}

		/** @brief Fold of values with keys >= lo inside subtree t (whole right subtrees read their aggregate). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		fold_type fold_from(uint32 t, Lookup const &lo) const
			requires(has_op)
		{
			if (t == 0) return pool_->id;
			node const &nd = pool_->nodes[t];
			if (pool_->compare_keys(nd.key, lo)) return fold_from(nd.right, lo);
			return pool_->fold_values(pool_->fold_values(fold_from(nd.left, lo), pool_->fold_type_of(nd)), pool_->nodes[nd.right].aggregate);
		}

		/** @brief Fold of values with keys < hi inside subtree t. */
		template<detail::ordered_lookup_key<Key, Compare> Lookup>
		fold_type fold_until(uint32 t, Lookup const &hi) const
			requires(has_op)
		{
			if (t == 0) return pool_->id;
			node const &nd = pool_->nodes[t];
			if (!pool_->compare_keys(nd.key, hi)) return fold_until(nd.left, hi);
			return pool_->fold_values(pool_->fold_values(pool_->nodes[nd.left].aggregate, pool_->fold_type_of(nd)), fold_until(nd.right, hi));
		}

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
		entry entry_of(uint32 t) const {
			node const &nd = pool_->nodes[t];
			if constexpr (has_value)
				return {nd.key, nd.value};
			else
				return {nd.key};
		}

		/**
		 * @brief Linear-time build from a range sorted by the comparator with no
		 *        duplicates: a Cartesian tree over the stored priorities kept on
		 *        a right-spine stack, each node pulled when it leaves the stack
		 *        (its subtree is then final).
		 */
		template<std::ranges::input_range R>
			requires(has_value ? detail::key_value_entry<detail::forwarded_element_t<R>, Key, slot<Value>> : std::convertible_to<detail::forwarded_element_t<R>, Key>)
		void build_sorted(R &&entries) {
			if constexpr (std::ranges::sized_range<R>) pool_->reserve(std::ranges::size(entries));
			std::vector<uint32> spine;
			uint32 last_id = 0;
			for (auto &&e : entries) {
				auto &&element = detail::forward_element<R>(std::forward<decltype(e)>(e));
				uint32 nid;
				if constexpr (has_value) {
					nid = pool_->make(Key(std::get<0>(std::forward<decltype(element)>(element))));
					pool_->nodes[nid].value = Value(std::get<1>(std::forward<decltype(element)>(element)));
				} else {
					nid = pool_->make(Key(std::forward<decltype(element)>(element)));
				}
				atl_assert(last_id == 0 || pool_->compare_keys(pool_->nodes[last_id].key, pool_->nodes[nid].key));
				last_id = nid;
				uint32 last = 0;
				while (!spine.empty() && pool_->priority(spine.back()) < pool_->priority(nid)) {
					last = spine.back();
					spine.pop_back();
					pool_->pull(last);
				}
				pool_->nodes[nid].left = last;
				if (!spine.empty()) pool_->nodes[spine.back()].right = nid;
				spine.push_back(nid);
			}
			for (uint32 const id : std::views::reverse(spine)) pool_->pull(id);
			root_ = spine.empty() ? 0 : spine.front();
		}

	public:
		/**
		 * @brief Not available on the fold tier: a fold needs its identity, which
		 * only the `(identity, op)` constructor supplies.
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		treap_map()
			requires(!has_op && std::default_initializable<Compare>)
			: pool_(std::make_shared<pool>()) {}

		/** @brief Construct an empty tree with the supplied comparator. @param comp Key order. @note Not constexpr (uses a shared_ptr node pool). */
		explicit treap_map(Compare comp)
			requires(!has_op)
			: pool_(std::make_shared<pool>(detail::move_or_copy(comp))) {}

		/**
		 * @brief Fold tier: supply the value monoid `(op, identity)` the key-range
		 * queries fold with. The template only gates the tier; both
		 * parameters are non-deduced, so arguments convert as usual.
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename OpArg = Op>
			requires(has_op && std::same_as<OpArg, Op>)
		treap_map(fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {}) : pool_(std::make_shared<pool>(detail::move_or_copy(comp), detail::move_or_copy(op), std::move(identity))) {}

		/**
		 * @brief Build from @p entries — keys for a set, `(key, value)` pairs for a
		 *        map — already sorted by @p comp with no duplicate keys, in O(n)
		 *        instead of n inserts. Elements are moved out of an owning rvalue
		 *        range and copied otherwise.
		 * @pre @p entries is strictly increasing under @p comp (asserted).
		 * @complexity O(n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<std::ranges::input_range R>
			requires(!has_op
					 && (has_value ? detail::key_value_entry<detail::forwarded_element_t<R>, Key, slot<Value>>
								   : std::convertible_to<detail::forwarded_element_t<R>, Key>))
		treap_map(sorted_unique_t, R &&entries, Compare comp = {}) : treap_map(detail::move_or_copy(comp)) {
			build_sorted(std::forward<R>(entries));
		}

		/**
		 * @brief Build from a sorted unique range on the fold tier.
		 * Fold tier: also supplies the value monoid, as the `(identity, op)` constructor.
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<std::ranges::input_range R, typename OpArg = Op>
			requires(has_op && std::same_as<OpArg, Op>
					 && (has_value ? detail::key_value_entry<detail::forwarded_element_t<R>, Key, slot<Value>>
								   : std::convertible_to<detail::forwarded_element_t<R>, Key>))
		treap_map(sorted_unique_t, R &&entries, fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})
			: treap_map(std::move(identity), detail::move_or_copy(op), detail::move_or_copy(comp)) {
			build_sorted(std::forward<R>(entries));
		}

		/** @brief Clone the source nodes into its shared pool. @param o Source handle. @complexity O(source size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_map(treap_map const &o) : pool_(o.pool_), root_(o.pool_ ? o.pool_->clone(o.root_) : 0) {}

		/** @brief Transfer the pool and tree root. @param o Source, empty on return. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		treap_map(treap_map &&o) noexcept : pool_(std::move(o.pool_)), root_(o.root_) { o.root_ = 0; }

		/** @brief Recycle this tree, then clone the source into its pool. @param o Source tree. @return `*this`. @complexity O(destination size + source size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_map &operator=(treap_map const &o) {
			if (this != &o) {
				if (pool_) pool_->recycle(root_);
				pool_ = o.pool_;
				root_ = pool_ ? pool_->clone(o.root_) : 0;
			}
			return *this;
		}

		/** @brief Recycle this tree, then transfer the source handle. @param o Source, empty on return. @return `*this`. @complexity O(destination size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_map &operator=(treap_map &&o) noexcept {
			if (this != &o) {
				if (pool_) pool_->recycle(root_);
				pool_ = std::move(o.pool_);
				root_ = o.root_;
				o.root_ = 0;
			}
			return *this;
		}

		/** @brief Recycle this handle's nodes into its shared pool. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		~treap_map() {
			if (pool_) pool_->recycle(root_);
		}

		/** @brief Number of elements in this handle; zero for a moved-from handle. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		std::size_t size() const noexcept { return pool_ ? pool_->nodes[root_].size : 0; }

		/** @brief Whether this handle contains no elements. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief Make room in the (shared) pool for @p n more keys, so the next
		 *        @p n insertions never regrow the node array.
		 * @complexity O(pool size) when it reallocates
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		void reserve(std::size_t n) {
			atl_assert(pool_ != nullptr);
			pool_->reserve(n);
		}

		/** @brief Whether @p k is present. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		[[nodiscard]] bool contains(Lookup const &k) const {
			return find_node(as_lookup(k)) != 0;
		}

		/**
		 * @brief Insert @p key (set tier); returns true if newly added.
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		bool insert(Key key)
			requires(!has_value)
		{
			atl_assert(pool_ != nullptr);
			if (find_node(key) != 0) return false;
			auto [l, r] = split_key(root_, key);
			root_ = base::merge(*pool_, base::merge(*pool_, l, pool_->make(std::move(key))), r);
			return true;
		}

		/**
		 * @brief Insert `{key, value}` (map tier); existing value kept. Returns true if
		 * newly inserted. @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename V = Value>
			requires(has_value && std::same_as<V, Value>)
		bool insert(Key key, std::type_identity_t<V> value) {
			atl_assert(pool_ != nullptr);
			if (find_node(key) != 0) return false;
			auto [l, r] = split_key(root_, key);
			uint32 const nid = pool_->make(std::move(key));
			pool_->nodes[nid].value = std::move(value);
			pool_->pull(nid);
			root_ = base::merge(*pool_, base::merge(*pool_, l, nid), r);
			return true;
		}

		/**
		 * @brief Insert or overwrite (map tier); returns true if newly inserted.
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename V = Value>
			requires(has_value && std::same_as<V, Value>)
		bool insert_or_assign(Key key, std::type_identity_t<V> value) {
			atl_assert(pool_ != nullptr);
			if (uint32 const t = find_node(key); t != 0) {
				pool_->nodes[t].value = std::move(value);
				if constexpr (has_op) refresh_path(key);
				return false;
			}
			return insert(std::move(key), std::move(value));
		}

		/**
		 * @brief Replace the value at @p key by `f(value)` (map tier), inserting
		 *        `f(Value{})` when @p key is absent — `m[k] += d` in one
		 *        descent, with one pull-up of the path. Returns the new value.
		 * @param key Key to update, or insert when absent.
		 * @param f Invoked exactly once.
		 * @complexity expected O(log n)
		 */
		template<typename Update, typename V = Value>
			requires(has_value && std::same_as<V, Value> && update_function<Update, V>)
		V const &update(Key key, Update &&f) {
			atl_assert(pool_ != nullptr);
			auto const result = upsert(root_, key, f);
			root_ = result.first;
			return pool_->nodes[result.second].value;
		}

		/**
		 * @brief The mapped value at @p k (map tier). @pre `contains(k)`
		 * @complexity expected O(log n)
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key, typename V = Value>
			requires(has_value && std::same_as<V, Value>)
		[[nodiscard]] V const &at(Lookup const &k) const {
			uint32 const t = find_node(as_lookup(k));
			atl_assert(t != 0);
			return pool_->nodes[t].value;
		}

		/**
		 * @brief The mapped value at @p k (map tier), or `nullptr` if absent — the
		 *        non-asserting counterpart of `at`.
		 * @complexity expected O(log n)
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key, typename V = Value>
			requires(has_value && std::same_as<V, Value>)
		[[nodiscard]] V const *find(Lookup const &k) const {
			uint32 const t = find_node(as_lookup(k));
			return t != 0 ? &pool_->nodes[t].value : nullptr;
		}

		/** @brief Remove @p k; returns true if it was present. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		bool erase(Lookup const &k) {
			atl_assert(pool_ != nullptr);
			auto const &key = as_lookup(k);
			if (find_node(key) == 0) return false;
			auto [l, rest] = split_key(root_, key);
			auto [mid, r] = split_key_inclusive(rest, key);
			pool_->recycle(mid);
			root_ = base::merge(*pool_, l, r);
			return true;
		}

		/**
		 * @brief Remove every key in `[lo, hi)`; returns how many were removed
		 *        (0 when the range is empty).
		 * @complexity expected O(log n + removed)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lo = Key, detail::ordered_lookup_key<Key, Compare> Hi = Key>
		std::size_t erase(Lo const &lo, Hi const &hi) {
			atl_assert(pool_ != nullptr);
			auto const &low = as_lookup(lo);
			auto const &high = as_lookup(hi);
			// no lo < hi test: lo and hi may be two different heterogeneous types
			// the comparator cannot order against each other, and an empty range
			// already yields an empty middle part
			auto [l, rest] = split_key(root_, low);
			auto [mid, r] = split_key(rest, high);
			std::size_t const removed = pool_->size(mid);
			pool_->recycle(mid);
			root_ = base::merge(*pool_, l, r);
			return removed;
		}

		/**
		 * @brief Cut this map at @p k: keys `< k` stay, keys `>= k` move to the
		 *        returned map, which shares this map's pool (so it can be
		 *        `merge_union`'d back) — the inverse of `merge_union`.
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		[[nodiscard]] treap_map split(Lookup const &k) {
			atl_assert(pool_ != nullptr);
			auto [l, r] = split_key(root_, as_lookup(k));
			root_ = l;
			return treap_map(pool_, r);
		}

		/**
		 * @brief Number of keys strictly before @p k (also `k`'s index if present).
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		[[nodiscard]] std::size_t rank(Lookup const &k) const {
			if (!pool_) return 0;
			auto const &key = as_lookup(k);
			std::size_t count = 0;
			uint32 t = root_;
			while (t != 0) {
				node const &nd = pool_->nodes[t];
				if (pool_->compare_keys(nd.key, key)) {
					count += pool_->nodes[nd.left].size + 1;
					t = nd.right;
				} else {
					t = nd.left;
				}
			}
			return count;
		}

		/**
		 * @brief The @p i-th smallest entry (`.key`, plus `.value` on the map tier).
		 * @pre `i < size()` @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] entry kth(std::size_t i) const {
			atl_assert(pool_ && i < size());
			uint32 t = root_;
			auto k = uint32(i);
			for (;;) {
				node const &nd = pool_->nodes[t];
				uint32 const ls = pool_->nodes[nd.left].size;
				if (k < ls)
					t = nd.left;
				else if (k == ls)
					return entry_of(t);
				else
					k -= ls + 1, t = nd.right;
			}
		}

		/** @brief The entry with the smallest key `>= k`, if any. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		[[nodiscard]] std::optional<entry> next(Lookup const &k) const {
			if (!pool_) return std::nullopt;
			auto const &key = as_lookup(k);
			uint32 best = 0, t = root_;
			while (t != 0) {
				node const &nd = pool_->nodes[t];
				if (pool_->compare_keys(nd.key, key)) {
					t = nd.right;
				} else {
					best = t;
					t = nd.left;
				}
			}
			if (best == 0) return std::nullopt;
			return entry_of(best);
		}

		/** @brief The entry with the largest key `<= k`, if any. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<detail::ordered_lookup_key<Key, Compare> Lookup = Key>
		[[nodiscard]] std::optional<entry> prev(Lookup const &k) const {
			if (!pool_) return std::nullopt;
			auto const &key = as_lookup(k);
			uint32 best = 0, t = root_;
			while (t != 0) {
				node const &nd = pool_->nodes[t];
				if (pool_->compare_keys(key, nd.key)) {
					t = nd.left;
				} else {
					best = t;
					t = nd.right;
				}
			}
			if (best == 0) return std::nullopt;
			return entry_of(best);
		}

		/**
		 * @brief Fold of the values (keys, for a set) of every key in `[lo, hi)` —
		 * read-only. @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lo = Key, detail::ordered_lookup_key<Key, Compare> Hi = Key>
			requires(has_op)
		[[nodiscard]] fold_type fold(Lo const &lo, Hi const &hi) const {
			atl_assert(pool_ != nullptr);
			auto const &low = as_lookup(lo);
			auto const &high = as_lookup(hi);
			// descend to the highest node inside [lo, hi) (none when the range is
			// empty); its left arm folds keys >= lo, its right arm keys < hi
			uint32 t = root_;
			while (t != 0) {
				node const &nd = pool_->nodes[t];
				if (!pool_->compare_keys(nd.key, low) && pool_->compare_keys(nd.key, high)) {
					return pool_->fold_values(pool_->fold_values(fold_from(nd.left, low), pool_->fold_type_of(nd)),
											  fold_until(nd.right, high));
				}
				t = pool_->compare_keys(nd.key, low) ? nd.right : nd.left;
			}
			return pool_->id;
		}

		/** @brief Fold of every value (key, for a set). @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		[[nodiscard]] fold_type fold() const
			requires(has_op)
		{
			atl_assert(pool_ != nullptr);
			return root_ != 0 ? pool_->nodes[root_].aggregate : pool_->id;
		}

		/** @brief In-order visit: `visit(key)` (set) or `visit(key, value)` (map). @note Not constexpr (uses a shared_ptr node pool). */
		template<typename Visit>
			requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &>
							   : std::invocable<Visit &, Key const &>)
		void for_each(Visit visit) const {
			if (pool_) visit_subtree(root_, visit);
		}

		/**
		 * @brief In-order visit of the keys in `[lo, hi)` only: `visit(key)` (set)
		 *        or `visit(key, value)` (map); subtrees outside the range are
		 *        never entered.
		 * @complexity expected O(log n + visited)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<detail::ordered_lookup_key<Key, Compare> Lo, detail::ordered_lookup_key<Key, Compare> Hi, typename Visit>
			requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &>
							   : std::invocable<Visit &, Key const &>)
		void for_each(Lo const &lo, Hi const &hi, Visit visit) const {
			if (!pool_) return;
			auto const &low = as_lookup(lo);
			auto const &high = as_lookup(hi);
			visit_range(root_, low, high, visit);
		}

		/** @brief Release this handle's nodes to its shared pool. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		void clear() {
			if (pool_) pool_->recycle(root_);
			root_ = 0;
		}

		/**
		 * @brief An empty map sharing this map's pool — the prerequisite for
		 * `merge_union` (and O(1), unlike copying-then-clearing).
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] treap_map sibling() const {
			atl_assert(pool_ != nullptr);
			return treap_map(pool_, 0);
		}

		/**
		 * @brief Absorb @p other (distinct source on the same pool; emptied by the call): the union of the two
		 * key sets, keeping **this** map's value on duplicate keys. Expected
		 * O(m log(1 + n/m)) for sizes n >= m — asymptotically better than inserting
		 * the smaller map element by element when the key ranges interleave
		 * little. @complexity expected O(m log(1 + n/m))
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		void merge_union(treap_map &&other) {
			atl_assert(this != &other && pool_ && other.pool_ == pool_);
			root_ = merge_trees(root_, other.root_, true);
			other.root_ = 0;
		}

	private:
		template<typename Visit>
			requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &> : std::invocable<Visit &, Key const &>)
		void visit_subtree(uint32 t, Visit &visit) const {
			if (t == 0) return;
			node const &nd = pool_->nodes[t];
			visit_subtree(nd.left, visit);
			if constexpr (has_value)
				std::invoke(visit, nd.key, nd.value);
			else
				std::invoke(visit, nd.key);
			visit_subtree(nd.right, visit);
		}

		template<detail::ordered_lookup_key<Key, Compare> Lo, detail::ordered_lookup_key<Key, Compare> Hi, typename Visit>
			requires(has_value ? std::invocable<Visit &, Key const &, slot<Value> const &> : std::invocable<Visit &, Key const &>)
		void visit_range(uint32 t, Lo const &lo, Hi const &hi, Visit &visit) const {
			if (t == 0) return;
			node const &nd = pool_->nodes[t];
			if (pool_->compare_keys(nd.key, lo)) return visit_range(nd.right, lo, hi, visit);
			if (!pool_->compare_keys(nd.key, hi)) return visit_range(nd.left, lo, hi, visit);
			visit_range(nd.left, lo, hi, visit);
			if constexpr (has_value)
				std::invoke(visit, nd.key, nd.value);
			else
				std::invoke(visit, nd.key);
			visit_range(nd.right, lo, hi, visit);
		}

		void refresh_path_from(uint32 t, Key const &k)
			requires has_op
		{
			if (t == 0) return;
			node const &nd = pool_->nodes[t];
			if (pool_->compare_keys(k, nd.key))
				refresh_path_from(nd.left, k);
			else if (pool_->compare_keys(nd.key, k))
				refresh_path_from(nd.right, k);
			pool_->pull(t);
		}

		/**
		 * @brief The `update` descent: find @p key under @p t and replace its
		 *        value, or insert it at the leaf and rotate it up past every
		 *        lower-priority ancestor (the rotate-insert form of the treap,
		 *        so finding and placing share one root-to-leaf walk). Returns
		 *        the subtree's new root and the touched node by value.
		 */
		template<typename Update>
			requires update_function<Update, Value>
		std::pair<uint32, uint32> upsert(uint32 t, Key &key, Update &f) {
			if (t == 0) {
				uint32 const nid = pool_->make(std::move(key));
				Value const fresh{};
				pool_->nodes[nid].value = std::invoke(f, fresh);
				pool_->pull(nid);
				return {nid, nid};
			}
			if (pool_->compare_keys(key, pool_->nodes[t].key)) {
				auto const result = upsert(pool_->nodes[t].left, key, f);
				uint32 const l = result.first;
				if (pool_->priority(l) > pool_->priority(t)) { // l was just placed: rotate right
					pool_->rebind_left(t, pool_->nodes[l].right);
					return {pool_->rebind_right(l, t), result.second};
				}
				return {pool_->rebind_left(t, l), result.second};
			}
			if (pool_->compare_keys(pool_->nodes[t].key, key)) {
				auto const result = upsert(pool_->nodes[t].right, key, f);
				uint32 const r = result.first;
				if (pool_->priority(r) > pool_->priority(t)) { // mirror: rotate left
					pool_->rebind_right(t, pool_->nodes[r].left);
					return {pool_->rebind_left(r, t), result.second};
				}
				return {pool_->rebind_right(t, r), result.second};
			}
			pool_->nodes[t].value = std::invoke(f, std::as_const(pool_->nodes[t].value));
			if constexpr (has_op) pool_->pull(t);
			return {t, t};
		}

		// Union of two subtrees; `a_wins` says whose value survives a duplicate
		// key (tracked across the priority swap, so semantics stay deterministic).
		uint32 merge_trees(uint32 a, uint32 b, bool a_wins) {
			if (a == 0 || b == 0) return a | b;
			if (pool_->priority(a) < pool_->priority(b)) {
				std::swap(a, b);
				a_wins = !a_wins;
			}
			Key const k = pool_->nodes[a].key; // copied out: splits may reallocate
			auto [bl, rest] = split_key(b, k);
			auto [dup, br] = split_key_inclusive(rest, k);
			if (dup != 0) {
				if constexpr (has_value) {
					if (!a_wins) pool_->nodes[a].value = std::move(pool_->nodes[dup].value);
				}
				pool_->recycled.push_back(dup); // a single node: no subtree walk needed
			}
			uint32 const l = merge_trees(pool_->left(a), bl, a_wins);
			uint32 const r = merge_trees(pool_->right(a), br, a_wins);
			return pool_->rebind(a, l, r);
		}

		// Recompute the aggregates on k's root path after an in-place value write.
		void refresh_path(Key const &k)
			requires(has_op)
		{
			refresh_path_from(root_, k);
		}
	};

	/** @brief Ordered set of keys: `treap_map` without mapped values. */
	template<typename Key, typename Compare = std::ranges::less, typename Op = void>
	using treap_set = treap_map<Key, void, Compare, Op>;
}
