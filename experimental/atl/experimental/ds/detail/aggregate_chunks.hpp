#pragma once

#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace atl::experimental::detail {
	/**
	 * @brief FIFO of fixed-size chunks that never relocates a live record; chunks emptied at the front are reused at
	 * the back, and cursors stay valid while their record lives.
	 * @pre Value transfers do not throw.
	 */
	template<std::copyable Value>
	class aggregate_chunks {
		/** @brief Records per chunk; not tuned. */
		static constexpr std::size_t chunk_size = 32;

		/** @brief Ownership runs through owned_next, separately from the active and free links, so relinking a chunk never moves it. */
		struct block {
			std::array<std::optional<Value>, chunk_size> values; ///< Engaged only for live records.
			std::unique_ptr<block> owned_next;
			block *next = nullptr; ///< Next active chunk, or next free chunk.
			block *previous = nullptr;

			// User-provided: MSVC does not treat the implicit destructor of this self-owning type as constexpr.
			constexpr ~block() {} // NOLINT(modernize-use-equals-default)
		};

	public:
		/** @brief A position; offset < chunk_size, so each position, the end included, has one representation and == compares positions. */
		struct cursor {
			block *chunk;
			std::size_t offset;
			[[nodiscard]] friend constexpr bool operator==(cursor, cursor) = default;
		};

	private:
		std::unique_ptr<block> owned_; ///< Every allocated chunk, active or free.
		block *free_ = nullptr;
		cursor first_{};
		cursor last_{}; ///< Past the end; its chunk always exists.
		std::size_t size_ = 0;
		std::size_t block_count_ = 0;

		/** @brief Add one chunk to the free list; a failed allocation changes nothing. */
		constexpr void allocate_block() {
			atl_assert(block_count_ < (std::numeric_limits<std::size_t>::max)() / sizeof(block));
			auto fresh = std::make_unique<block>();
			fresh->owned_next = std::move(owned_);
			fresh->next = free_;
			free_ = fresh.get();
			owned_ = std::move(fresh);
			++block_count_;
		}

		[[nodiscard]] constexpr block *take_block() {
			if (!free_) allocate_block();
			auto *result = free_;
			free_ = result->next;
			result->next = result->previous = nullptr;
			return result;
		}

		constexpr void swap(aggregate_chunks &other) noexcept {
			owned_.swap(other.owned_);
			std::swap(free_, other.free_);
			std::swap(first_, other.first_);
			std::swap(last_, other.last_);
			std::swap(size_, other.size_);
			std::swap(block_count_, other.block_count_);
		}

	public:
		/** @brief Allocates the chunk of the end cursor. */
		constexpr aggregate_chunks() { first_ = last_ = {take_block(), 0}; }

		/** @brief Copy only the live values, starting at offset zero of the first chunk. */
		constexpr aggregate_chunks(aggregate_chunks const &other) : aggregate_chunks() {
			reserve(other.size());
			for (auto position = other.begin(); position != other.end(); position = next(position)) {
				prepare_push();
				push_back(other.at(position));
			}
		}

		/** @brief The source keeps dangling cursors without chunks: it may only be destroyed or assigned to. */
		constexpr aggregate_chunks(aggregate_chunks &&) = default;

		/** @brief Iterative, so destruction does not recurse once per chunk through owned_next. */
		constexpr ~aggregate_chunks() {
			while (owned_) {
				auto next_owned = std::move(owned_->owned_next);
				owned_ = std::move(next_owned);
			}
		}

		/** @brief Copy through a temporary, so a failed allocation leaves this FIFO unchanged. */
		constexpr aggregate_chunks &operator=(aggregate_chunks const &other) {
			if (this != &other) {
				aggregate_chunks copy(other);
				swap(copy);
			}
			return *this;
		}

		/** @brief Take the chunks of @p other; the chunks this FIFO held are destroyed iteratively. */
		constexpr aggregate_chunks &operator=(aggregate_chunks &&other) noexcept {
			if (this != &other) {
				aggregate_chunks moved(std::move(other));
				swap(moved);
			}
			return *this;
		}

		constexpr std::size_t size() const noexcept { return size_; }

		constexpr bool empty() const noexcept { return size_ == 0; }

		/** @brief A record limit that keeps the chunk count, two spare chunks included, representable in bytes. */
		[[nodiscard]] constexpr std::size_t max_size() const noexcept {
			auto const blocks = (std::numeric_limits<std::size_t>::max)() / sizeof(block);
			return blocks > 2 ? (blocks - 2) * chunk_size : 0;
		}

		[[nodiscard]] constexpr cursor begin() const { return first_; }

		[[nodiscard]] constexpr cursor end() const { return last_; }

		/** @brief The next position; at a chunk boundary the successor chunk must already be linked. */
		[[nodiscard]] static constexpr cursor next(cursor position) {
			atl_assert(position.chunk);
			return position.offset == chunk_size - 1 ? cursor{position.chunk->next, 0} : cursor{position.chunk, position.offset + 1};
		}

		[[nodiscard]] static constexpr cursor previous(cursor position) {
			atl_assert(position.chunk);
			if (position.offset != 0) return {position.chunk, position.offset - 1};
			atl_assert(position.chunk->previous);
			return {position.chunk->previous, chunk_size - 1};
		}

		[[nodiscard]] constexpr Value &at(cursor position) {
			atl_assert(position.chunk && position.offset < chunk_size && position.chunk->values[position.offset]);
			return *position.chunk->values[position.offset];
		}

		[[nodiscard]] constexpr Value const &at(cursor position) const {
			atl_assert(position.chunk && position.offset < chunk_size && position.chunk->values[position.offset]);
			return *position.chunk->values[position.offset];
		}

		/** @brief Hold @p elements simultaneously live records: a partly consumed first chunk and the end chunk need two chunks beyond elements / chunk_size. */
		constexpr void reserve(std::size_t elements) {
			atl_assert(elements <= max_size());
			auto const needed = elements / chunk_size + 2;
			while (block_count_ < needed) allocate_block();
		}

		/** @brief Link the chunk the next push_back needs, before anything changes, so a failed allocation changes nothing. */
		constexpr void prepare_push() {
			atl_assert(size_ < max_size());
			if (last_.offset == chunk_size - 1 && !last_.chunk->next) {
				auto *successor = take_block();
				successor->previous = last_.chunk;
				last_.chunk->next = successor;
			}
		}

		/** @brief Append after prepare_push(); never allocates. */
		constexpr void push_back(Value value) {
			atl_assert(size_ < max_size());
			atl_assert(last_.offset != chunk_size - 1 || last_.chunk->next);
			last_.chunk->values[last_.offset].emplace(std::move(value));
			last_ = next(last_);
			++size_;
		}

		/** @brief Destroy the first record; a chunk it empties goes to the free list. */
		constexpr void pop_front() {
			atl_assert(!empty());
			auto *old = first_.chunk;
			first_.chunk->values[first_.offset].reset();
			first_ = next(first_);
			--size_;
			if (first_.chunk != old) {
				first_.chunk->previous = nullptr;
				old->next = free_;
				free_ = old;
			}
		}

		/** @brief Destroy the live records in O(n), keeping every chunk. */
		constexpr void clear() {
			while (!empty()) pop_front();
		}
	};

	/** @brief The chunk FIFO with the four DABA Lite boundaries, which a copy rebinds to its own chunks. */
	template<std::copyable Value>
	struct daba_chunk_storage : aggregate_chunks<Value> {
		using base = aggregate_chunks<Value>;
		using cursor = typename base::cursor;
		cursor left = this->begin();
		cursor right = left;
		cursor accumulator = left;
		cursor back = left;

		constexpr daba_chunk_storage() = default;

		/** @brief Copy the records and rebind the four boundaries in one O(n) walk. */
		constexpr daba_chunk_storage(daba_chunk_storage const &other) : base(other) {
			auto destination = this->begin();
			for (auto source = other.begin();; source = base::next(source), destination = base::next(destination)) {
				if (source == other.left) left = destination;
				if (source == other.right) right = destination;
				if (source == other.accumulator) accumulator = destination;
				if (source == other.back) back = destination;
				if (source == other.end()) break;
			}
		}

		/** @brief Chunks keep their addresses, so the moved cursors stay valid. */
		constexpr daba_chunk_storage(daba_chunk_storage &&) = default;

		/** @brief Copy through a temporary, so a failed allocation leaves this storage unchanged. */
		constexpr daba_chunk_storage &operator=(daba_chunk_storage const &other) {
			if (this != &other) *this = daba_chunk_storage(other);
			return *this;
		}

		constexpr daba_chunk_storage &operator=(daba_chunk_storage &&) = default;
	};
}
