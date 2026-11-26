#pragma once

#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash_seed.hpp>
#include <cstddef>
#include <limits>
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
	 * @brief Packed endpoint pairs to edge ids, without iteration. Linear probing
	 *        at at most half load gives expected O(1) lookup, insertion and erasure;
	 *        doubling makes growth amortized. Backward-shift deletion closes each
	 *        probe chain immediately, so link/cut churn cannot accumulate tombstones.
	 */
	class edge_id_table {
		struct slot {
			uint64 key = 0;
			uint32 value = 0;
			bool is_occupied = false;
		};

		std::vector<slot> slots_;
		std::size_t entry_count_ = 0;

		[[nodiscard]] constexpr std::size_t locate(uint64 key) const noexcept {
			std::size_t const mask = slots_.size() - 1;
			std::size_t i = edge_id_hash(key) & mask;
			while (slots_[i].is_occupied && slots_[i].key != key) i = (i + 1) & mask;
			return i;
		}

	public:
		[[nodiscard]] constexpr std::size_t entry_count() const noexcept { return entry_count_; }

		/** @brief Forget all entries, retaining storage. @complexity O(capacity) */
		constexpr void reset_entries() {
			for (auto &s : slots_) s.is_occupied = false;
			entry_count_ = 0;
		}

		/** @brief Reserve for @p entries live pairs. @complexity Expected O(entries) when growing, O(1) otherwise. */
		constexpr void reserve_entries(std::size_t entries) {
			atl_assert(entries <= (std::numeric_limits<std::size_t>::max)() / 4);
			if (entries <= slots_.size() / 2) return;
			std::size_t capacity = 8;
			while (capacity / 2 < entries) capacity *= 2;
			std::vector<slot> replacement(capacity);
			auto old = std::move(slots_);
			slots_ = std::move(replacement);
			for (auto const &s : old)
				if (s.is_occupied) slots_[locate(s.key)] = s;
		}

		/** @brief Existing id or @p fallback; all key and value bit patterns are allowed. */
		[[nodiscard]] constexpr uint32 lookup_id(uint64 key, uint32 fallback) const noexcept {
			if (slots_.empty()) return fallback;
			auto const &s = slots_[locate(key)];
			return s.is_occupied ? s.value : fallback;
		}

		/** @brief Insert a pair, retaining the id of an existing key. */
		constexpr void insert_entry(uint64 key, uint32 value) {
			reserve_entries(entry_count_ + 1);
			auto &s = slots_[locate(key)];
			if (s.is_occupied) return;
			s = {key, value, true};
			++entry_count_;
		}

		constexpr void erase_entry(uint64 key) noexcept {
			if (slots_.empty()) return;
			std::size_t hole = locate(key);
			if (!slots_[hole].is_occupied) return;
			std::size_t const mask = slots_.size() - 1;
			for (std::size_t i = (hole + 1) & mask; slots_[i].is_occupied; i = (i + 1) & mask) {
				std::size_t const home = edge_id_hash(slots_[i].key) & mask;
				// Move only when the hole lies on this entry's circular probe path.
				// Capacity is bounded by reserve_entries, so these sums cannot overflow.
				std::size_t const offset = slots_.size() - home;
				if (((hole + offset) & mask) < ((i + offset) & mask)) {
					slots_[hole] = slots_[i];
					hole = i;
				}
			}
			slots_[hole].is_occupied = false;
			--entry_count_;
		}
	};
}
