#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Sparse table: O(1) range folds of an IDEMPOTENT operation over
	 *        static data — min, max, gcd, bitwise and/or: any semilattice
	 *        (associative + commutative + idempotent). The laws are a
	 *        precondition on the caller, like a sort comparator's ordering.
	 *
	 * A query covers `[l, r)` with two overlapping power-of-two windows, so
	 * elements may be combined twice — which is exactly what idempotency
	 * licenses. For non-idempotent (or non-commutative) associative ops use
	 * @ref atl::disjoint_sparse_table; for plain integer min/argmin
	 * @ref atl::direct_rmq builds in O(n) and also returns the position; for
	 * updates use @ref atl::segment_tree.
	 *
	 * Storage is one flat level-major array with rows padded to `n` (layout
	 * raced, see docs): a query touches two entries of one row.
	 * No identity element is required — queries must be non-empty.
	 * @tparam T Copyable stored element.
	 * @tparam Op Const-invocable closed binary operation.
	 * @pre All operation intermediates are valid and representable. The padded
	 *      table size must fit vector<T>::max_size() (asserted before multiplication).
	 */
	template<std::copyable T, binary_operation<T> Op>
	class sparse_table {
		[[no_unique_address]] Op op_;
		std::vector<T> table_; // Fixed row stride avoids per-level offset lookup.
		std::size_t n_ = 0;

	public:
		/** @brief Construct an empty table; no folds are valid. */
		constexpr sparse_table() = default;

		/**
		 * @brief Build over @p values. Row 0 moves the elements out of an rvalue range
		 * that owns them and copies them otherwise, including lvalue views. @complexity O(n log n) time and space
		 */
		template<std::ranges::sized_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		constexpr explicit sparse_table(R &&values, Op op = {}) : op_(std::move(op)) {
			atl_assert(std::ranges::size(values) <= table_.max_size());
			n_ = std::size_t(std::ranges::size(values));
			if (n_ == 0) return;
			std::size_t const levels = std::bit_width(n_);
			atl_assert(n_ <= table_.max_size() / levels);
			table_.reserve(levels * n_);
			for (auto &&v : values) table_.emplace_back(detail::forward_element<R>(v));
			table_.resize(levels * n_, table_[0]); // padding slots, never read
			for (std::size_t k = 1; k < levels; ++k) {
				std::size_t const half = std::size_t(1) << (k - 1);
				T const *prev = table_.data() + (k - 1) * n_;
				T *row = table_.data() + k * n_;
				for (std::size_t i = 0; i + 2 * half <= n_; ++i) row[i] = op_(prev[i], prev[i + half]);
			}
		}

		/** @brief Fold of `[l, r)`. @pre `l < r <= size()` @complexity O(1) */
		constexpr T fold(std::size_t l, std::size_t r) const {
			atl_assert(l < r && r <= n_);
			std::size_t const k = std::size_t(std::bit_width(r - l)) - 1;
			T const *row = table_.data() + k * n_;
			return op_(row[l], row[r - (std::size_t(1) << k)]);
		}

		/** @brief Number of input elements. */
		constexpr std::size_t size() const { return n_; }

		/** @brief Whether the input has no elements. */
		constexpr bool empty() const { return n_ == 0; }
	};

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::ranges::sized_range R, typename Op>
	sparse_table(R &&, Op) -> sparse_table<std::ranges::range_value_t<R>, Op>;

	/**
	 * @brief Disjoint sparse table: O(1) range folds of ANY associative
	 *        operation over static data — non-commutative and non-idempotent
	 *        included (sums, products, affine or matrix composition).
	 *
	 * Level `k` splits the array into blocks of `2^(k+1)` and stores, inside
	 * each block, suffix folds running down to the block midpoint and prefix
	 * folds running up from it. A query `[l, r)` picks the level of the
	 * highest bit where `l` and `r-1` differ: the two ends then straddle one
	 * midpoint, and the answer is one application of the op — every element
	 * combined exactly once, in order. Row 0 holds the original values, so
	 * single-element queries are a load.
	 *
	 * Same conventions as @ref atl::sparse_table, flat padded level-major
	 * storage, no identity, non-empty queries only.
	 * @tparam T Copyable stored element.
	 * @tparam Op Const-invocable closed associative operation.
	 * @pre Every operation is valid and representable. The padded table size
	 *      must fit vector<T>::max_size() (asserted before multiplication).
	 */
	template<std::copyable T, binary_operation<T> Op>
	class disjoint_sparse_table {
		[[no_unique_address]] Op op_;
		std::vector<T> table_; // Fixed row stride avoids per-level offset lookup.
		std::size_t n_ = 0;

	public:
		/** @brief Construct an empty table; no folds are valid. */
		constexpr disjoint_sparse_table() = default;

		/**
		 * @brief Build over @p values. Row 0 moves the elements out of an rvalue range
		 * that owns them and copies them otherwise, including lvalue views. @complexity O(n log n) time and space
		 */
		template<std::ranges::sized_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		constexpr explicit disjoint_sparse_table(R &&values, Op op = {}) : op_(std::move(op)) {
			atl_assert(std::ranges::size(values) <= table_.max_size());
			n_ = std::size_t(std::ranges::size(values));
			if (n_ == 0) return;
			std::size_t const levels = (std::max<std::size_t>)(1, std::bit_width(n_ - 1));
			atl_assert(n_ <= table_.max_size() / levels);
			table_.reserve(levels * n_);
			for (auto &&v : values) table_.emplace_back(detail::forward_element<R>(v));
			table_.resize(levels * n_, table_[0]); // padding slots, never read
			T const *base = table_.data();
			for (std::size_t k = 1; k < levels; ++k) {
				std::size_t const half = std::size_t(1) << k;
				T *row = table_.data() + k * n_;
				for (std::size_t block = 0; block < n_; block += 2 * half) {
					std::size_t const mid = (std::min)(block + half, n_);
					// Suffixes preserve left-to-right operand order.
					row[mid - 1] = base[mid - 1];
					for (std::size_t i = mid - 1; i-- > block;) row[i] = op_(base[i], row[i + 1]);
					// Prefixes preserve left-to-right operand order.
					if (mid == n_) continue;
					row[mid] = base[mid];
					std::size_t const end = (std::min)(block + 2 * half, n_);
					for (std::size_t i = mid + 1; i < end; ++i) row[i] = op_(row[i - 1], base[i]);
				}
			}
		}

		/**
		 * @brief Fold of `[l, r)`, elements combined in order.
		 * @pre `l < r <= size()` @complexity O(1)
		 */
		constexpr T fold(std::size_t l, std::size_t r) const {
			atl_assert(l < r && r <= n_);
			std::size_t const last = r - 1;
			if (l == last) return table_[l];
			std::size_t const k = std::size_t(std::bit_width(l ^ last)) - 1;
			T const *row = table_.data() + k * n_;
			return op_(row[l], row[last]);
		}

		/** @brief Number of input elements. */
		constexpr std::size_t size() const { return n_; }

		/** @brief Whether the input has no elements. */
		constexpr bool empty() const { return n_ == 0; }
	};

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::ranges::sized_range R, typename Op>
	disjoint_sparse_table(R &&, Op) -> disjoint_sparse_table<std::ranges::range_value_t<R>, Op>;
}
