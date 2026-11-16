#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/hash_table.hpp>
#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/ds/detail/key_value_entry.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <tuple>
#include <utility>

namespace atl {
	/**
	 * @brief Unique-key hash container with flat slots and linear probing.
	 *
	 * Lookup and erasure take expected O(1), insertion expected amortized O(1),
	 * assuming constant-cost hashing and equality with well-distributed hashes.
	 * A worst-case probe and full iteration take O(capacity()). Rehash invalidates
	 * all references and iterators; erase invalidates the erased element; clear
	 * invalidates all elements but retains slots, including their stored resources.
	 * Hash and equality must remain stable, with equal keys hashing equally.
	 * Default seeded hashing reads runtime process state: build tables from main
	 * or later, after static seed initialization; set_hash_seed must precede use.
	 * Custom constexpr hash/equality functors permit constant evaluation.
	 * Hash invocation and home-slot conversion/mixing are constrained to be noexcept.
	 * No exception guarantee is provided for throwing element operations during mutation.
	 * Key arguments must remain valid across any rehash performed by insertion.
	 * Capacity arithmetic must fit size_t and its next power of two; backing
	 * allocations must be addressable. See reserve for its element-count bound.
	 *
	 * @tparam Key  Key type — default-constructible and copyable.
	 * @tparam T    Mapped type — default-constructible and movable.
	 * @tparam Hash Hash functor (default @ref hash, seeded; pass
	 *              @ref identity_hash for keys that are already uniformly random).
	 * @tparam Eq   Key equality.
	 *
	 * @note Iterators expose `pair<Key, T>&` — modify `.second` freely, but never
	 *       `.first` (it would corrupt the table).
	 */
	template<typename Key, typename T, typename Hash = hash<Key>, typename Eq = std::equal_to<>>
		requires std::default_initializable<Key> && std::copyable<Key> && std::default_initializable<T>
			  && std::movable<T> && detail::nothrow_hasher_for<Hash, Key> && detail::key_eq_for<Eq, Key> && std::default_initializable<Hash>
			  && std::default_initializable<Eq>
	class flat_hash_map {
		using slot = std::pair<Key, T>;
		using table = detail::hash_table<slot, Key, detail::map_key_of<Key, T>, Hash, Eq>;
		table table_;

	public:
		/** @brief Stored key type. */
		using key_type = Key;
		/** @brief Stored mapped type. */
		using mapped_type = T;
		/** @brief Stored element type. */
		using value_type = std::pair<Key, T>;
		/** @brief Element and slot count type. */
		using size_type = std::size_t;
		/** @brief Mutable forward iterator over live entries; keys must not be changed. */
		using iterator = typename table::iterator;
		/** @brief Read-only forward iterator over live elements. */
		using const_iterator = typename table::const_iterator;

		/** @brief An empty container with no allocated slots. */
		constexpr flat_hash_map() = default;

		/** @brief Reserve for expected_elements distinct insertions. @pre Same bounds as reserve. */
		constexpr explicit flat_hash_map(std::size_t expected_elements) : table_(expected_elements) {}

		/**
		 * @brief Build from a range of `(key, value)` pairs, as @ref insert(R &&)
		 *        does; on duplicate keys the first occurrence wins.
		 * @complexity Expected amortized O(n), plus a scan of old capacity if reservation grows.
		 */
		template<std::ranges::input_range R>
			requires detail::key_value_entry<detail::forwarded_element_t<R>, Key, T>
		constexpr explicit flat_hash_map(R &&entries) {
			insert(std::forward<R>(entries));
		}

		/** @brief Reference to the value for @p key, inserting a default-constructed one if absent. */
		constexpr T &operator[](Key const &key) {
			auto [i, was_inserted] = table_.find_or_prepare(key);
			if (was_inserted) table_.slot(i).second = T{};
			return table_.slot(i).second;
		}

		/** @brief As above; @p key is moved into the table if newly inserted. */
		constexpr T &operator[](Key &&key) {
			auto [i, was_inserted] = table_.find_or_prepare(std::move(key));
			if (was_inserted) table_.slot(i).second = T{};
			return table_.slot(i).second;
		}

		/** @brief Reference to the mapped value. @pre key is present (asserted). */
		[[nodiscard]] constexpr T &at(Key const &key) {
			std::size_t const i = table_.find(key);
			atl_assert(i != table::npos);
			return table_.slot(i).second;
		}

		/** @brief Read-only mapped reference. @pre key is present (asserted). */
		[[nodiscard]] constexpr T const &at(Key const &key) const {
			std::size_t const i = table_.find(key);
			atl_assert(i != table::npos);
			return table_.slot(i).second;
		}

		/**
		 * @brief Mapped reference for a transparent lookup key.
		 * @pre key is present (asserted); equal mixed keys hash equally.
		 */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr T &at(K const &key) {
			std::size_t const i = table_.find(key);
			atl_assert(i != table::npos);
			return table_.slot(i).second;
		}

		/** @brief Read-only mapped reference for a transparent lookup key. @pre key is present (asserted). */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr T const &at(K const &key) const {
			std::size_t const i = table_.find(key);
			atl_assert(i != table::npos);
			return table_.slot(i).second;
		}

		/**
		 * @brief The value for @p key, or @p fallback if absent — the non-asserting,
		 * never-inserting counterpart of @ref at. Returns by value (so the
		 * result never dangles on a temporary fallback, unlike a reference-
		 * returning form); the fallback is untouched on a hit and *moved*,
		 * not copied, on a miss when passed as an rvalue.
		 * @note Requires implicit conversion from `T const &` and the forwarded fallback to `T`.
		 * @complexity O(1) expected
		 */
		template<typename U = T>
			requires std::convertible_to<U &&, T> && std::convertible_to<T const &, T>
		[[nodiscard]] constexpr T at_or(Key const &key, U &&fallback) const {
			std::size_t const i = table_.find(key);
			if (i != table::npos) return table_.slot(i).second;
			return std::forward<U>(fallback);
		}

		/**
		 * @brief Copy the mapped value for a transparent key, or forward fallback on a miss.
		 * @note Requires implicit conversion from `T const &` and the forwarded fallback to `T`.
		 * @pre Equal mixed keys hash equally.
		 */
		template<typename K, typename U = T>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K> && std::convertible_to<U &&, T> && std::convertible_to<T const &, T>
		[[nodiscard]] constexpr T at_or(K const &key, U &&fallback) const {
			std::size_t const i = table_.find(key);
			if (i != table::npos) return table_.slot(i).second;
			return std::forward<U>(fallback);
		}

		/**
		 * @brief Insert `{key, value}`; returns true if newly inserted (existing value kept).
		 * Key and value are moved in when passed as rvalues (and the value only
		 * when actually inserted).
		 */
		template<typename V = T>
			requires std::convertible_to<V &&, T>
		constexpr bool insert(Key const &key, V &&value) {
			auto [i, was_inserted] = table_.find_or_prepare(key);
			if (was_inserted) table_.slot(i).second = static_cast<T>(std::forward<V>(value));
			return was_inserted;
		}

		/** @brief Insert by moving key if absent; keep the existing value on a hit. @return Whether newly inserted. */
		template<typename V = T>
			requires std::convertible_to<V &&, T>
		constexpr bool insert(Key &&key, V &&value) {
			auto [i, was_inserted] = table_.find_or_prepare(std::move(key));
			if (was_inserted) table_.slot(i).second = static_cast<T>(std::forward<V>(value));
			return was_inserted;
		}

		/**
		 * @brief Insert every `(key, value)` pair of @p entries, keeping the
		 *        existing value on a duplicate key (like @ref insert(Key const &, V &&)).
		 *        Reserves up front when the range is sized; elements are moved out
		 *        of an owning rvalue range and copied otherwise.
		 * @complexity Expected amortized O(n), plus a scan of old capacity if reservation grows.
		 */
		template<std::ranges::input_range R>
			requires detail::key_value_entry<detail::forwarded_element_t<R>, Key, T>
		constexpr void insert(R &&entries) {
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(entries) <= (std::numeric_limits<std::size_t>::max)() - table_.size());
				table_.reserve(table_.size() + std::ranges::size(entries));
			}
			for (auto &&e : entries) {
				auto &&element = detail::forward_element<R>(std::forward<decltype(e)>(e));
				using element_type = decltype(element);
				// the key is converted first so a duplicate never touches the value
				auto [i, was_inserted] = table_.find_or_prepare(Key(std::get<0>(std::forward<element_type>(element))));
				if (was_inserted) table_.slot(i).second = T(std::get<1>(std::forward<element_type>(element)));
			}
		}

		/**
		 * @brief Insert @p key with a value built from @p args, constructing it
		 *        only when the key is absent (an existing value is kept and no `T`
		 *        is built). Returns true if newly inserted.
		 * @complexity Expected amortized O(1).
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr bool try_emplace(Key const &key, Args &&...args) {
			auto [i, was_inserted] = table_.find_or_prepare(key);
			if (was_inserted) table_.slot(i).second = T(std::forward<Args>(args)...);
			return was_inserted;
		}

		/**
		 * @brief Insert by moving key and constructing its mapped value only on a miss.
		 * @return Whether the key was newly inserted.
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr bool try_emplace(Key &&key, Args &&...args) {
			auto [i, was_inserted] = table_.find_or_prepare(std::move(key));
			if (was_inserted) table_.slot(i).second = T(std::forward<Args>(args)...);
			return was_inserted;
		}

		/**
		 * @brief Build a `pair<Key, T>` from @p args and insert it by move,
		 *        keeping an existing value; returns true if newly inserted. The
		 *        pair is built (and its value discarded) even when the key is
		 *        present — @ref try_emplace avoids that.
		 * @complexity Expected amortized O(1).
		 */
		template<typename... Args>
			requires std::constructible_from<value_type, Args...>
		constexpr bool emplace(Args &&...args) {
			value_type entry(std::forward<Args>(args)...);
			auto [i, was_inserted] = table_.find_or_prepare(std::move(entry.first));
			if (was_inserted) table_.slot(i).second = std::move(entry.second);
			return was_inserted;
		}

		/** @brief Insert or overwrite; returns true if the key was newly inserted. */
		constexpr bool insert_or_assign(Key const &key, T value) {
			auto [i, was_inserted] = table_.find_or_prepare(key);
			table_.slot(i).second = std::move(value);
			return was_inserted;
		}

		/** @brief Insert or overwrite, moving key on insertion. @return Whether newly inserted. */
		constexpr bool insert_or_assign(Key &&key, T value) {
			auto [i, was_inserted] = table_.find_or_prepare(std::move(key));
			table_.slot(i).second = std::move(value);
			return was_inserted;
		}

		/** @brief Whether key is present. @return Membership under Eq. */
		[[nodiscard]] constexpr bool contains(Key const &key) const { return table_.contains(key); }

		/**
		 * @brief Heterogeneous @ref contains, @ref count, @ref find and @ref erase: look up
		 *        by any @p K without first building a `Key` (no `std::string` temporary
		 *        for a `string_view` or literal). They exist only when `Hash` and `Eq` both
		 *        declare `is_transparent` and accept @p K (`atl::hash<std::string>` with
		 *        the default `std::equal_to<>` do), and `hash(k)` must equal the hash of
		 *        the equal `Key`. `erase` also excludes anything convertible to an
		 *        iterator, so an iterator argument is never taken as a key.
		 */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr bool contains(K const &key) const {
			return table_.contains(key);
		}

		/** @brief Return 1 if key is present, otherwise 0. */
		[[nodiscard]] constexpr std::size_t count(Key const &key) const { return table_.contains(key) ? 1 : 0; }

		/** @brief Return 1 if a transparent key is present, otherwise 0. */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr std::size_t count(K const &key) const {
			return table_.contains(key) ? 1 : 0;
		}

		/** @brief Iterator to key, or end() if absent. */
		[[nodiscard]] constexpr iterator find(Key const &key) {
			std::size_t const i = table_.find(key);
			return i == table::npos ? table_.end() : table_.iter_at(i);
		}

		/** @brief Read-only iterator to key, or end() if absent. */
		[[nodiscard]] constexpr const_iterator find(Key const &key) const {
			std::size_t const i = table_.find(key);
			return i == table::npos ? table_.end() : table_.iter_at(i);
		}

		/** @brief Iterator to a transparent key, or end() if absent. */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr iterator find(K const &key) {
			std::size_t const i = table_.find(key);
			return i == table::npos ? table_.end() : table_.iter_at(i);
		}

		/** @brief Iterator to a transparent key, or end() if absent. */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K>
		[[nodiscard]] constexpr const_iterator find(K const &key) const {
			std::size_t const i = table_.find(key);
			return i == table::npos ? table_.end() : table_.iter_at(i);
		}

		/** @brief Remove @p key; returns true if it was present. */
		constexpr bool erase(Key const &key) { return table_.erase(key); }

		/** @brief Remove a transparent key. @return Whether it was present. */
		template<typename K>
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K> && (!std::convertible_to<K, iterator>) && (!std::convertible_to<K, const_iterator>)
		constexpr bool erase(K const &key) {
			return table_.erase(key);
		}

		/** @brief Remove all keys; retain slots and their stored resources. @complexity O(capacity()) */
		constexpr void clear() { table_.clear(); }

		/** @brief Reserve for n total elements, without shrinking. @param n Element count, not slot count.
		 * @pre n <= size_t maximum / 4; the next power of two of n * 4 / 3 + 1 is representable.
		 * @complexity Expected O(old capacity + new capacity) when growing; O(1) otherwise. */
		constexpr void reserve(std::size_t n) { table_.reserve(n); }

		/** @brief Number of live elements. */
		constexpr std::size_t size() const noexcept { return table_.size(); }

		/** @brief Whether there are no live elements. */
		constexpr bool empty() const noexcept { return table_.empty(); }

		/** @brief Number of allocated slots, including empty and deleted slots. */
		constexpr std::size_t capacity() const noexcept { return table_.capacity(); }

		/** @brief Iterator to the first live entry. @complexity O(capacity()) worst case. */
		constexpr iterator begin() { return table_.begin(); }

		/** @brief Past-the-end iterator. */
		constexpr iterator end() { return table_.end(); }

		/** @brief Read-only iterator to the first live entry. @complexity O(capacity()) worst case. */
		constexpr const_iterator begin() const { return table_.begin(); }

		/** @brief Read-only past-the-end iterator. */
		constexpr const_iterator end() const { return table_.end(); }
	};
}
