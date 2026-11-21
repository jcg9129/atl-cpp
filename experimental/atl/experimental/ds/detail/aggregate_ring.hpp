#pragma once

#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace atl::experimental::detail {
	/**
	 * @brief Growable FIFO ring of optional slots, so Value needs no default constructor. Growth relocates the O(n)
	 * live values; reserve counts simultaneously live values.
	 * @pre Value transfers during mutation do not throw.
	 */
	template<std::copyable Value>
	class aggregate_ring {
		std::vector<std::optional<Value>> slots_; ///< Engaged exactly at the live positions.
		std::size_t head_ = 0;					  ///< Slot of logical position zero.
		std::size_t size_ = 0;

		/** @brief Translate a logical offset without overflowing head + offset. */
		[[nodiscard]] constexpr std::size_t position(std::size_t offset) const {
			atl_assert(offset < slots_.size());
			auto const tail = slots_.size() - head_;
			return offset < tail ? head_ + offset : offset - tail;
		}

	public:
		constexpr aggregate_ring() = default;

		/** @brief Copy only the live values, into a ring of exactly size() slots starting at slot zero. */
		constexpr aggregate_ring(aggregate_ring const &other) {
			reserve(other.size());
			for (std::size_t i = 0; i < other.size(); ++i) push_back(other[i]);
		}

		/** @brief The source keeps its offsets without slots: it may only be destroyed or assigned to. */
		constexpr aggregate_ring(aggregate_ring &&) = default;

		/** @brief Copy through a temporary, so a failed allocation leaves this ring unchanged. */
		constexpr aggregate_ring &operator=(aggregate_ring const &other) {
			if (this != &other) *this = aggregate_ring(other);
			return *this;
		}

		constexpr aggregate_ring &operator=(aggregate_ring &&) = default;

		constexpr std::size_t size() const noexcept { return size_; }

		constexpr bool empty() const noexcept { return size_ == 0; }

		[[nodiscard]] constexpr std::size_t max_size() const noexcept { return slots_.max_size(); }

		[[nodiscard]] constexpr Value &operator[](std::size_t offset) {
			atl_assert(offset < size_);
			return *slots_[position(offset)];
		}

		[[nodiscard]] constexpr Value const &operator[](std::size_t offset) const {
			atl_assert(offset < size_);
			return *slots_[position(offset)];
		}

		/** @brief Hold @p elements simultaneously live values; the new slots are allocated before any value moves. */
		constexpr void reserve(std::size_t elements) {
			atl_assert(elements <= max_size());
			if (elements <= slots_.size()) return;
			std::vector<std::optional<Value>> replacement(elements);
			for (std::size_t i = 0; i < size_; ++i) replacement[i].emplace(std::move((*this)[i]));
			slots_.swap(replacement);
			head_ = 0;
		}

		/** @brief Make room for @p elements values, at least doubling the slots but never beyond @p limit. */
		constexpr void prepare(std::size_t elements, std::size_t limit) {
			atl_assert(elements <= limit && limit <= max_size());
			if (elements <= slots_.size()) return;
			auto const capacity = slots_.size();
			auto const grown = capacity > limit - capacity ? limit : capacity + capacity;
			reserve(elements > grown ? elements : grown);
		}

		/** @brief Append into a free slot; never allocates. */
		constexpr void push_back(Value value) {
			atl_assert(size_ < slots_.size());
			slots_[position(size_)].emplace(std::move(value));
			++size_;
		}

		constexpr void pop_front() {
			atl_assert(!empty());
			slots_[head_].reset();
			head_ = head_ == slots_.size() - 1 ? 0 : head_ + 1;
			--size_;
		}

		constexpr void pop_back() {
			atl_assert(!empty());
			slots_[position(size_ - 1)].reset();
			--size_;
		}

		/** @brief Destroy the live values in O(n), keeping the slots. */
		constexpr void clear() {
			while (!empty()) pop_front();
			head_ = 0;
		}
	};
}
