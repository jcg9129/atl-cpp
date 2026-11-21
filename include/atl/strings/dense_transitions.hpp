#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/** @brief Internal factory for adopting the automaton fill's completed allocation. */
		struct dense_transitions_builder;
	}

	/**
	 * @brief A deterministic transition table over the dense alphabet
	 *        `[0, alphabet_size())`, one flat row-major `uint32` table: row `v`
	 *        holds the targets of state `v`, contiguously. A cell is either a
	 *        state or @ref atl::dense_transitions::none, so partial tables (a trie, a suffix automaton)
	 *        are representable as well as complete functions. Produced by
	 *        @ref atl::aho_corasick::transition_table, whose tables are complete, for
	 *        automaton DPs that read every (state, symbol) pair.
	 * @note Spans refer to owned cells and survive const operations. Assignment and
	 *       destruction invalidate them; moving transfers their storage. Copies own
	 *       independent cells. Copying is O(cells), moving is O(1) apart from releasing
	 *       the destination allocation. Cell lookup and row/count access are O(1).
	 *       Allocation exceptions propagate; construction validates every target under ATL_ASSERT.
	 */
	class dense_transitions {
		std::vector<uint32> next_;
		// Keep vector adoption O(1); the automaton fill instead adopts an
		// overwrite allocation once every cell has been initialized.
		std::unique_ptr<uint32[]> overwritten_;
		std::size_t cell_count_ = 0;
		uint32 const *data_ = nullptr;
		std::size_t alphabet_size_ = 1;

		/** @brief The initialized cells, whichever allocation owns them. */
		[[nodiscard]] constexpr std::span<uint32 const> cells() const {
			return {data_, cell_count_};
		}

		/**
		 * @brief Adopt an initialized array of @p cell_count complete row-major targets.
		 * @pre The allocation holds exactly @p cell_count initialized cells
		 *      (documented). It is non-null, alphabet_size >= 1,
		 *      cell_count >= alphabet_size and a multiple of it, and each
		 *      target is less than state_count() (asserted).
		 * @complexity O(1), excluding the assertion scan.
		 */
		constexpr dense_transitions(std::unique_ptr<uint32[]> next, std::size_t const cell_count, std::size_t const alphabet_size)
			: overwritten_(std::move(next)), cell_count_(cell_count), data_(overwritten_.get()), alphabet_size_(alphabet_size) {
			atl_assert(overwritten_ != nullptr && alphabet_size_ >= 1 && cell_count >= alphabet_size_ && cell_count % alphabet_size_ == 0);
			atl_assert(state_count() <= none);
			atl_assert(std::ranges::all_of(cells(), [&](uint32 const t) { return t < state_count(); }));
		}

		friend struct detail::dense_transitions_builder;

	public:
		/** @brief The target of a missing transition; never a valid state. */
		static constexpr uint32 none = uint32(-1);

		/** @brief An empty table: no states, alphabet size 1. */
		constexpr dense_transitions() = default;

		/** @brief Copy the initialized cells into an independent snapshot. */
		constexpr dense_transitions(dense_transitions const &other)
			: next_(other.cells().begin(), other.cells().end()), cell_count_(next_.size()), data_(next_.data()), alphabet_size_(other.alphabet_size_) {}

		/** @brief Replace this snapshot with an independent copy. */
		constexpr dense_transitions &operator=(dense_transitions const &other) {
			if (this != &other) *this = dense_transitions(other);
			return *this;
		}

		/** @brief Transfer the allocation, leaving an empty table. */
		constexpr dense_transitions(dense_transitions &&other) noexcept
			: next_(std::move(other.next_)), overwritten_(std::move(other.overwritten_)),
			  cell_count_(std::exchange(other.cell_count_, 0)), data_(std::exchange(other.data_, nullptr)), alphabet_size_(other.alphabet_size_) {}

		/** @brief Transfer the allocation, leaving an empty table. */
		constexpr dense_transitions &operator=(dense_transitions &&other) noexcept {
			if (this != &other) {
				next_ = std::move(other.next_);
				overwritten_ = std::move(other.overwritten_);
				cell_count_ = std::exchange(other.cell_count_, 0);
				data_ = std::exchange(other.data_, nullptr);
				alphabet_size_ = other.alphabet_size_;
			}
			return *this;
		}

		/**
		 * @brief Adopt @p next, row-major `state_count × alphabet_size` targets.
		 * @pre `alphabet_size >= 1`, `next.size()` a multiple of it, and every
		 *      at most 2^32 - 1 states, and every entry either @ref atl::dense_transitions::none or `< next.size() / alphabet_size`
		 *      (all asserted)
		 */
		constexpr dense_transitions(std::vector<uint32> next, std::size_t const alphabet_size)
			: next_(std::move(next)), cell_count_(next_.size()), data_(next_.data()), alphabet_size_(alphabet_size) {
			atl_assert(alphabet_size_ >= 1 && next_.size() % alphabet_size_ == 0);
			atl_assert(state_count() <= none);
			atl_assert(std::ranges::all_of(next_, [&](uint32 const t) { return t == none || t < state_count(); }));
		}

		/**
		 * @brief The state reached from @p v on symbol @p c, or @ref atl::dense_transitions::none if
		 *        there is no such transition.
		 */
		[[nodiscard]] constexpr uint32 operator()(uint32 const v, uint32 const c) const {
			atl_assert(v < state_count() && c < alphabet_size_);
			return cells()[std::size_t(v) * alphabet_size_ + c];
		}

		/** @brief All targets of @p v, indexed by symbol (`none` where absent). */
		[[nodiscard]] constexpr std::span<uint32 const> row(uint32 const v) const {
			atl_assert(v < state_count());
			return cells().subspan(std::size_t(v) * alphabet_size_, alphabet_size_);
		}

		/** @brief Number of states (rows). */
		[[nodiscard]] constexpr std::size_t state_count() const { return cells().size() / alphabet_size_; }

		/** @brief Number of symbols (columns); symbols are `[0, alphabet_size())`. */
		[[nodiscard]] constexpr std::size_t alphabet_size() const { return alphabet_size_; }
	};

	namespace detail {
		/** @brief Adopt a fully filled automaton table without exposing a storage-specific public constructor. */
		struct dense_transitions_builder {
			/**
			 * @brief Transfer a complete initialized row-major allocation.
			 * @pre Exactly cell_count initialized cells (documented); non-null,
			 *      at least one row, positive alphabet_size dividing cell_count,
			 *      and every target in range (asserted by dense_transitions).
			 */
			[[nodiscard]] static constexpr dense_transitions adopt(std::unique_ptr<uint32[]> next, std::size_t cell_count, std::size_t alphabet_size) {
				return {std::move(next), cell_count, alphabet_size};
			}
		};
	}

}
