#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/hash_table.hpp>
#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
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
	 * @tparam Key  The element type — default-constructible and copyable.
	 * @tparam Hash Hash functor (default @ref hash, which is seeded; pass
	 *              @ref identity_hash for keys that are already uniformly random).
	 * @tparam Eq   Key equality.
	 */
	template<typename Key, typename Hash = hash<Key>, typename Eq = std::equal_to<>>
		requires std::default_initializable<Key> && std::copyable<Key> && detail::nothrow_hasher_for<Hash, Key>
			  && detail::key_eq_for<Eq, Key> && std::default_initializable<Hash>
			  && std::default_initializable<Eq>
	class flat_hash_set {
		using table = detail::hash_table<Key, Key, detail::set_key_of, Hash, Eq>;
		table table_;

	public:
		/** @brief Stored key type. */
		using key_type = Key;
		/** @brief Stored element type. */
		using value_type = Key;
		/** @brief Element and slot count type. */
		using size_type = std::size_t;
		/** @brief Read-only forward iterator over live elements. */
		using const_iterator = typename table::const_iterator;
		/** @brief Iterator over live elements; set elements are immutable. */
		using iterator = const_iterator;

		/** @brief An empty container with no allocated slots. */
		constexpr flat_hash_set() = default;

		/** @brief Pre-size for @p expected_elements insertions without a rehash. */
		constexpr explicit flat_hash_set(std::size_t expected_elements) : table_(expected_elements) {}

		/**
		 * @brief Build from the elements of @p keys, as @ref insert(R &&) does.
		 * @complexity Expected amortized O(n), plus a scan of old capacity if reservation grows.
		 */
		template<std::ranges::input_range R>
			requires(!std::same_as<std::remove_cvref_t<R>, Key>
					 && std::convertible_to<detail::forwarded_element_t<R>, Key>)
		constexpr explicit flat_hash_set(R &&keys) {
			insert(std::forward<R>(keys));
		}

		/** @brief Insert @p key; returns true if it was newly added. */
		constexpr bool insert(Key const &key) { return table_.find_or_prepare(key).second; }

		/** @brief Insert @p key by move; returns true if it was newly added. */
		constexpr bool insert(Key &&key) { return table_.find_or_prepare(std::move(key)).second; }

		/**
		 * @brief Insert every element of @p keys. Reserves up front when the range
		 *        is sized; elements are moved out of an owning rvalue range and
		 *        copied otherwise.
		 * @complexity Expected amortized O(n), plus a scan of old capacity if reservation grows.
		 */
		template<std::ranges::input_range R>
			requires(!std::same_as<std::remove_cvref_t<R>, Key>
					 && std::convertible_to<detail::forwarded_element_t<R>, Key>)
		constexpr void insert(R &&keys) {
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(keys) <= (std::numeric_limits<std::size_t>::max)() - table_.size());
				table_.reserve(table_.size() + std::ranges::size(keys));
			}
			for (auto &&e : keys) table_.find_or_prepare(Key(detail::forward_element<R>(std::forward<decltype(e)>(e))));
		}

		/**
		 * @brief Build a `Key` from @p args and insert it by move; returns true if it
		 * was newly added. The key is built (and then moved) even when it is
		 * already present.
		 */
		template<typename... Args>
			requires std::constructible_from<Key, Args...>
		constexpr bool emplace(Args &&...args) {
			return insert(Key(std::forward<Args>(args)...));
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

		/** @brief Iterator to @p key, or `end()` if absent. */
		[[nodiscard]] constexpr const_iterator find(Key const &key) const {
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
			requires detail::hashed_heterogeneous_key_for<K, Key, Hash, Eq> && detail::nothrow_hasher_for<Hash, K> && (!std::convertible_to<K, const_iterator>)
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

		/** @brief Read-only iterator to the first live entry. @complexity O(capacity()) worst case. */
		constexpr const_iterator begin() const { return table_.begin(); }

		/** @brief Read-only past-the-end iterator. */
		constexpr const_iterator end() const { return table_.end(); }
	};
}
