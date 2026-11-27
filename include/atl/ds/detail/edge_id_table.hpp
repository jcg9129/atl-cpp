#pragma once

#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash_seed.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Seeded at runtime; unseeded during constant evaluation, which cannot read runtime state. */
	[[nodiscard]] constexpr std::size_t edge_id_hash(uint64 key) noexcept {
		if consteval {
			return std::size_t(splitmix64(key));
		} else {
			return std::size_t(splitmix64(key + hash_seed()));
		}
	}

	/**
	 * @brief Integer keys to small trivial values, without iteration. Linear probing
	 *        at at most half load gives expected O(1) lookup, insertion and erasure;
	 *        doubling makes growth amortized. Backward-shift deletion closes each
	 *        probe chain immediately, so link/cut churn cannot accumulate tombstones.
	 */
	template<std::regular Value>
		requires std::is_trivially_copyable_v<Value>
	class integer_key_table {
		struct slot {
			uint64 key = 0;
			Value value{};
			bool is_occupied = false;
		};

		// Small tables keep occupancy beside the payload to avoid a second allocation.
		// From 4096 slots, a bitmap lets unsuccessful probes skip random payload
		// cache lines. The choice depends only on capacity and never on key bits.
		struct occupancy {
			std::vector<uint64> words;

			constexpr occupancy() = default;

			constexpr explicit occupancy(std::size_t capacity) : words(capacity < 4096 ? 0 : capacity / 64) {}

			[[nodiscard]] constexpr bool test(slot const &s, std::size_t i) const noexcept {
				return words.empty() ? s.is_occupied : ((words[i / 64] >> (i % 64)) & 1) != 0;
			}

			constexpr void set(slot &s, std::size_t i, bool value) noexcept {
				if (words.empty()) {
					s.is_occupied = value;
				} else {
					uint64 const bit = uint64(1) << (i % 64);
					if (value)
						words[i / 64] |= bit;
					else
						words[i / 64] &= ~bit;
				}
			}

			constexpr void reset(std::vector<slot> &slots) noexcept {
				if (words.empty()) {
					for (auto &s : slots) s.is_occupied = false;
				} else {
					for (auto &word : words) word = 0;
				}
			}
		};

		std::vector<slot> slots_;
		occupancy occupancy_;
		std::size_t entry_count_ = 0;

		[[nodiscard]] constexpr std::size_t locate(uint64 key) const noexcept {
			std::size_t const mask = slots_.size() - 1;
			std::size_t i = edge_id_hash(key) & mask;
			while (occupancy_.test(slots_[i], i) && slots_[i].key != key) i = (i + 1) & mask;
			return i;
		}

	public:
		[[nodiscard]] constexpr std::size_t entry_count() const noexcept { return entry_count_; }

		/** @brief Forget all entries, retaining storage. @complexity O(capacity) */
		constexpr void reset_entries() {
			occupancy_.reset(slots_);
			entry_count_ = 0;
		}

		/** @brief Reserve for @p entries live pairs. @complexity Expected O(entries) when growing, O(1) otherwise. */
		constexpr void reserve_entries(std::size_t entries) {
			atl_assert(entries <= (std::numeric_limits<std::size_t>::max)() / 4);
			if (entries <= slots_.size() / 2) return;
			std::size_t capacity = 8;
			while (capacity / 2 < entries) capacity *= 2;
			std::vector<slot> replacement(capacity);
			occupancy replacement_occupancy(capacity);
			auto old = std::move(slots_);
			auto old_occupancy = std::move(occupancy_);
			slots_ = std::move(replacement);
			occupancy_ = std::move(replacement_occupancy);
			for (std::size_t i = 0; i < old.size(); ++i)
				if (old_occupancy.test(old[i], i)) {
					std::size_t const destination = locate(old[i].key);
					slots_[destination] = old[i];
					occupancy_.set(slots_[destination], destination, true);
				}
		}

		/** @brief Existing id or @p fallback; all key and value bit patterns are allowed. */
		[[nodiscard]] constexpr Value lookup_id(uint64 key, Value fallback) const noexcept {
			if (slots_.empty()) return fallback;
			std::size_t const i = locate(key);
			return occupancy_.test(slots_[i], i) ? slots_[i].value : fallback;
		}

		/** @brief Value, or null when absent. Invalidated by growth or erasure. */
		[[nodiscard]] constexpr Value const *entry_value(uint64 key) const noexcept {
			if (slots_.empty()) return nullptr;
			std::size_t const i = locate(key);
			return occupancy_.test(slots_[i], i) ? &slots_[i].value : nullptr;
		}

		/** @brief Mutable value, or null when absent. Invalidated by growth or erasure. */
		[[nodiscard]] constexpr Value *entry_value(uint64 key) noexcept {
			if (slots_.empty()) return nullptr;
			std::size_t const i = locate(key);
			return occupancy_.test(slots_[i], i) ? &slots_[i].value : nullptr;
		}

		/** @brief Insert a pair, retaining the value of an existing key. */
		constexpr void insert_entry(uint64 key, Value value) {
			reserve_entries(entry_count_ + 1);
			std::size_t const i = locate(key);
			if (occupancy_.test(slots_[i], i)) return;
			slots_[i] = {key, value};
			occupancy_.set(slots_[i], i, true);
			++entry_count_;
		}

		constexpr void erase_entry(uint64 key) noexcept {
			if (slots_.empty()) return;
			std::size_t hole = locate(key);
			if (!occupancy_.test(slots_[hole], hole)) return;
			std::size_t const mask = slots_.size() - 1;
			for (std::size_t i = (hole + 1) & mask; occupancy_.test(slots_[i], i); i = (i + 1) & mask) {
				std::size_t const home = edge_id_hash(slots_[i].key) & mask;
				// Move only when the hole lies on this entry's circular probe path.
				// Capacity is bounded by reserve_entries, so these sums cannot overflow.
				std::size_t const offset = slots_.size() - home;
				if (((hole + offset) & mask) < ((i + offset) & mask)) {
					slots_[hole] = slots_[i];
					hole = i;
				}
			}
			occupancy_.set(slots_[hole], hole, false);
			--entry_count_;
		}
	};

	using edge_id_table = integer_key_table<uint32>;
}
