#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief A hash functor usable on @p Key: callable, returning something size_t-like. */
	template<typename H, typename Key>
	concept hasher_for = requires(H const &h, Key const &k) {
		{ h(k) } -> std::convertible_to<std::size_t>;
	};

	/** @brief An equality functor usable on @p Key. */
	template<typename E, typename Key>
	concept key_eq_for = requires(E const &e, Key const &a, Key const &b) {
		{ e(a, b) } -> std::convertible_to<bool>;
	};

	struct set_key_of {
		template<std::movable Key>
		constexpr Key &operator()(Key &s) const {
			return s;
		}

		template<std::movable Key>
		constexpr Key const &operator()(Key const &s) const {
			// callers always pass stored slots, never temporaries
			// NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter)
			return s;
		}
	};

	template<std::copyable Key, std::movable Mapped>
	struct map_key_of {
		constexpr Key &operator()(std::pair<Key, Mapped> &s) const {
			return s.first;
		}

		constexpr Key const &operator()(std::pair<Key, Mapped> const &s) const {
			return s.first;
		}
	};

	/**
	 * @brief Whether hash_table may take a hasher's low bits as the home slot unmixed:
	 *        true for the avalanching `atl::hash` and for `atl::identity_hash`, whose
	 *        contract is "the keys are already random". Any other hasher (`std::hash`,
	 *        the identity on integers in libstdc++, or a user functor) is passed
	 *        through a Fibonacci multiply first, so structured keys such as
	 *        `(u << 32) | v` do not pile into one probe run.
	 */
	template<typename Hash>
	inline constexpr bool hash_low_bits_usable = false;
	template<typename Key>
	inline constexpr bool hash_low_bits_usable<atl::hash<Key>> = true;
	template<>
	inline constexpr bool hash_low_bits_usable<atl::identity_hash> = true;

	/** @brief A hasher whose call and home-slot conversion/mixing cannot throw. */
	template<typename Hash, typename Key>
	concept nothrow_hasher_for = hasher_for<Hash, Key> && requires(Hash const &h, Key const &k) {
		{ h(k) } noexcept;
	} && ((hash_low_bits_usable<Hash> && requires(Hash const &h, Key const &k, std::size_t m) {
									 { h(k) & m } noexcept;
									 // home() returns this expression implicitly; an explicit cast may choose a different conversion.
									 requires std::is_nothrow_convertible_v<decltype(h(k) & m), std::size_t>;
								 }) || (!hash_low_bits_usable<Hash> && requires(Hash const &h, Key const &k, std::size_t m) {
									 { std::size_t((uint64(h(k)) * 0x9E3779B97F4A7C15ull) >> std::countl_zero(uint64(m))) } noexcept;
								 }));

	/** @brief A deleted slot keeps its probe chain reachable. */
	enum slot_state : uint8 { slot_empty = 0,
							  slot_full = 1,
							  slot_deleted = 2 };

	/**
	 * @brief Open-addressing hash table with linear probing — the shared engine behind
	 *        @ref flat_hash_set and @ref flat_hash_map.
	 *
	 * One flat slot array plus a parallel control byte per slot (empty/full/deleted).
	 * Power-of-two capacity with masked indexing, tombstone deletion, and growth at a
	 * 0.75 load factor. Probes are expected O(1); insertion is expected amortized O(1).
	 * Iteration and clearing scan O(capacity) slots.
	 *
	 * @tparam Slot  Stored element: `Key` for a set, `pair<Key, T>` for a map.
	 * @tparam Key   The lookup key type.
	 * @tparam KeyOf Extracts a `Key&` from a `Slot`.
	 * @tparam Hash  Hash functor with no-throw home-slot conversion/mixing.
	 * @tparam Eq    Key equality.
	 */
	template<typename Slot, typename Key, typename KeyOf, nothrow_hasher_for<Key> Hash, key_eq_for<Key> Eq>
		requires std::default_initializable<Slot> && std::movable<Slot> && std::default_initializable<KeyOf>
			  && std::default_initializable<Hash> && std::default_initializable<Eq>
			  && std::copyable<Key>
			  && requires(KeyOf const &key_of, Slot &slot, Slot const &const_slot) {
					 { key_of(slot) } -> std::same_as<Key &>;
					 { key_of(const_slot) } -> std::same_as<Key const &>;
				 }
	class hash_table {
	public:
		static constexpr std::size_t npos = std::size_t(-1);

		constexpr hash_table() = default;

		constexpr explicit hash_table(std::size_t expected_elements) { reserve(expected_elements); }

		constexpr std::size_t size() const noexcept { return size_; }

		constexpr bool empty() const noexcept { return size_ == 0; }

		constexpr std::size_t capacity() const noexcept { return slots_.size(); }

		constexpr void clear() {
			if (control_.empty()) return;
			std::ranges::fill(control_, uint8(slot_empty));
			size_ = 0;
			tombstones_ = 0;
		}

		constexpr void reserve(std::size_t n) {
			// the n-th insert grows when 4n >= 3 capacity, so stay strictly above 4n / 3
			atl_assert(n <= (std::numeric_limits<std::size_t>::max)() / 4);
			std::size_t const need = round_up(n * 4 / 3 + 1);
			if (need > slots_.size()) rehash(need);
		}

		/** @brief Locate @p key (a `Key`, or any @ref hashed_heterogeneous_key_for the caller has vetted); `npos` if absent. */
		template<typename K>
			requires nothrow_hasher_for<Hash, K> && requires(Eq const &eq, Key const &stored, K const &key) { { eq(stored, key) } -> std::convertible_to<bool>; }
		constexpr std::size_t find(K const &key) const {
			if (slots_.empty()) return npos;
			std::size_t const m = mask();
			for (std::size_t i = home(key, m);; i = (i + 1) & m) {
				uint8 const st = control_[i];
				if (st == slot_empty) return npos;
				if (st == slot_full && eq_(key_of_(slots_[i]), key)) return i;
			}
		}

		template<typename K>
			requires nothrow_hasher_for<Hash, K> && requires(Eq const &eq, Key const &stored, K const &key) { { eq(stored, key) } -> std::convertible_to<bool>; }
		constexpr bool contains(K const &key) const { return find(key) != npos; }

		/**
		 * @brief Locate @p key, or reserve a slot and write the key into it. Returns
		 * `(index, inserted)`; the caller fills any mapped value when `inserted`.
		 */
		constexpr std::pair<std::size_t, bool> find_or_prepare(Key const &key) { return find_or_prepare_impl(key); }

		/** @brief As above, but the key is moved into the slot when newly inserted. */
		constexpr std::pair<std::size_t, bool> find_or_prepare(Key &&key) { return find_or_prepare_impl(std::move(key)); }

		template<typename K>
			requires nothrow_hasher_for<Hash, K> && requires(Eq const &eq, Key const &stored, K const &key) { { eq(stored, key) } -> std::convertible_to<bool>; }
		constexpr bool erase(K const &key) {
			std::size_t const i = find(key);
			if (i == npos) return false;
			control_[i] = slot_deleted;
			--size_;
			++tombstones_;
			return true;
		}

		constexpr Slot &slot(std::size_t i) { return slots_[i]; }

		constexpr Slot const &slot(std::size_t i) const { return slots_[i]; }

		/** @brief Forward iterator over the occupied slots (skips empty/deleted). */
		template<bool IsConst>
		class basic_iterator {
			using table_ptr = std::conditional_t<IsConst, hash_table const *, hash_table *>;
			table_ptr t_ = nullptr;
			std::size_t i_ = 0;

			constexpr void skip() {
				while (i_ < t_->control_.size() && t_->control_[i_] != slot_full) ++i_;
			}

		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = Slot;
			using difference_type = std::ptrdiff_t;
			using reference = std::conditional_t<IsConst, Slot const &, Slot &>;
			using pointer = std::conditional_t<IsConst, Slot const *, Slot *>;
			constexpr basic_iterator() = default;

			constexpr basic_iterator(table_ptr t, std::size_t i) : t_(t), i_(i) { skip(); }

			constexpr reference operator*() const { return t_->slots_[i_]; }

			constexpr pointer operator->() const { return &t_->slots_[i_]; }

			constexpr basic_iterator &operator++() {
				++i_;
				skip();
				return *this;
			}

			constexpr basic_iterator operator++(int) {
				basic_iterator tmp = *this;
				++*this;
				return tmp;
			}

			constexpr bool operator==(basic_iterator const &o) const { return i_ == o.i_; }
		};

		using iterator = basic_iterator<false>;
		using const_iterator = basic_iterator<true>;

		constexpr iterator begin() { return iterator(this, 0); }

		constexpr iterator end() { return iterator(this, slots_.size()); }

		constexpr const_iterator begin() const { return const_iterator(this, 0); }

		constexpr const_iterator end() const { return const_iterator(this, slots_.size()); }

		constexpr iterator iter_at(std::size_t i) { return iterator(this, i); }

		constexpr const_iterator iter_at(std::size_t i) const { return const_iterator(this, i); }

	private:
		std::vector<Slot> slots_;
		std::vector<uint8> control_;
		std::size_t size_ = 0;
		std::size_t tombstones_ = 0;
		[[no_unique_address]] Hash hash_;
		[[no_unique_address]] Eq eq_;
		[[no_unique_address]] KeyOf key_of_;

		/**
		 * @brief Shared body of both @ref find_or_prepare overloads: probe for
		 *        @p key, reusing the first tombstone on the path when inserting.
		 *
		 * The key is only forwarded at the single insertion write, after every
		 * probe comparison on the path has already read it.
		 */
		template<typename K>
			requires std::same_as<std::remove_cvref_t<K>, Key>
		constexpr std::pair<std::size_t, bool> find_or_prepare_impl(K &&key) {
			// probe before any growth check: a hit never rehashes, and reusing a tombstone
			// does not raise the load, so only a fresh slot can trigger a grow
			std::size_t at = npos;
			if (!slots_.empty()) {
				std::size_t const m = mask();
				std::size_t reuse = npos;
				for (std::size_t i = home(key, m);; i = (i + 1) & m) {
					uint8 const st = control_[i];
					if (st == slot_empty) {
						at = reuse != npos ? reuse : i;
						break;
					}
					if (st == slot_full) {
						if (eq_(key_of_(slots_[i]), key)) return {i, false};
					} else if (reuse == npos) {
						reuse = i;
					}
				}
				if (reuse != npos)
					--tombstones_;
				else if (size_ + tombstones_ + 1 >= slots_.size() - slots_.size() / 4)
					at = npos;
			}
			if (at == npos) {
				grow();
				at = empty_slot_for(key);
			}
			key_of_(slots_[at]) = std::forward<K>(key);
			control_[at] = slot_full;
			++size_;
			return {at, true};
		}

		/** @brief First empty slot on @p key's probe path; the table holds no tombstones. */
		constexpr std::size_t empty_slot_for(Key const &key) const {
			std::size_t const m = mask();
			std::size_t i = home(key, m);
			while (control_[i] != slot_empty) i = (i + 1) & m;
			return i;
		}

		constexpr std::size_t mask() const noexcept { return slots_.size() - 1; }

		template<typename K>
			requires nothrow_hasher_for<Hash, K>
		constexpr std::size_t home(K const &k, std::size_t m) const noexcept {
			if constexpr (hash_low_bits_usable<Hash>)
				return hash_(k) & m;
			else // the top log2(capacity) bits of a Fibonacci multiply; m + 1 >= 2
				return std::size_t((uint64(hash_(k)) * 0x9E3779B97F4A7C15ull) >> std::countl_zero(uint64(m)));
		}

		static constexpr std::size_t round_up(std::size_t n) {
			atl_assert(n <= (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1)));
			return n < 2 ? 2 : std::bit_ceil(n);
		}

		/**
		 * @brief Rehash to at least twice the live size, never below the current capacity (a
		 * tombstone-triggered rehash must not shrink below a reservation).
		 */
		constexpr void grow() {
			atl_assert(size_ < (std::numeric_limits<std::size_t>::max)() / 2);
			rehash((std::max)(slots_.size(), round_up((size_ + 1) * 2)));
		}

		/** @brief Rebuild into @p new_cap slots (a power of two), dropping tombstones. */
		constexpr void rehash(std::size_t new_cap) {
			std::vector<Slot> old_slots = std::move(slots_);
			std::vector<uint8> old_ctrl = std::move(control_);
			// resize, not assign: default-constructs in place, so a move-only Slot
			// (e.g. a unique_ptr mapped value) works — Slot is movable, not copyable
			slots_.clear();
			slots_.resize(new_cap);
			control_.assign(new_cap, uint8(slot_empty));
			size_ = 0;
			tombstones_ = 0;
			for (std::size_t i = 0; i < old_slots.size(); ++i)
				if (old_ctrl[i] == slot_full) reinsert(std::move(old_slots[i]));
		}

		/**
		 * @brief Place a slot known to be absent into a table with no
		 *        tombstones: first empty slot on its probe path, no comparisons.
		 */
		constexpr void reinsert(Slot &&s) {
			std::size_t const i = empty_slot_for(key_of_(s));
			slots_[i] = std::move(s);
			control_[i] = slot_full;
			++size_;
		}
	};
}
