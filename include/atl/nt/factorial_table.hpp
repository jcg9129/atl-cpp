#pragma once

#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace atl {
	/**
	 * @brief Tables of factorials, inverse factorials and integer inverses in a prime field, for indices
	 *        `0..maximum_index()`; `extend` grows them.
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @complexity O(maximum_index) field operations and storage, plus one modular inversion per construction or
	 *             growing `extend`.
	 * @note Every accessor returns by value, so growth invalidates no reference, pointer or iterator a caller holds.
	 */
	template<static_prime_mod_int Coefficient>
	class factorial_table {
		std::vector<Coefficient> factorials_{Coefficient(1)};
		std::vector<Coefficient> inverse_factorials_{Coefficient(1)};
		// Entry zero is unused.
		std::vector<Coefficient> inverse_integers_{Coefficient(0)};

	public:
		/**
		 * @brief Build every entry for indices `0..maximum_index` with one inversion.
		 * @param maximum_index Largest queryable index.
		 * @pre `maximum_index < Coefficient::mod()`, because p! is zero and cannot be inverted (asserted).
		 */
		explicit constexpr factorial_table(std::size_t maximum_index) { extend(maximum_index); }

		/**
		 * @brief Ensure every entry for indices `0..maximum_index` exists; a smaller index changes nothing.
		 *
		 * Only the new suffix is computed: factorials forward from the old top, one inversion of the new top
		 * factorial, and inverse factorials and integer inverses backward down to the old top. Existing entries
		 * keep their values.
		 * @param maximum_index Largest index to make queryable.
		 * @pre `maximum_index < Coefficient::mod()`, because p! is zero and cannot be inverted, and
		 *      `maximum_index < std::vector<Coefficient>::max_size()` (both asserted when the table grows).
		 * @throws std::bad_alloc (or what the allocator throws) if the storage cannot grow; the table then keeps
		 *         its previous entries and `maximum_index()`.
		 * @complexity O(maximum_index - previous maximum_index()) field operations, plus one modular inversion
		 *             when the table grows. Capacity grows geometrically, so the copying caused by any sequence of
		 *             extensions is linear in the final size.
		 */
		constexpr void extend(std::size_t maximum_index) {
			std::size_t const previous = this->maximum_index();
			if (maximum_index <= previous) return;
			atl_assert(maximum_index < Coefficient::mod());
			atl_assert(maximum_index < (std::numeric_limits<std::size_t>::max)());
			std::size_t const size = maximum_index + 1;
			std::size_t const limit = factorials_.max_size();
			atl_assert(size <= limit);
			// Geometric capacity keeps a sequence of small extensions linear. A failed reserve leaves its vector
			// unchanged, and once all three succeed the resizes below cannot reallocate.
			auto const grow = [size, limit](std::vector<Coefficient> &entries) {
				if (size > entries.capacity()) entries.reserve(entries.capacity() <= limit / 2 ? (std::max)(size, 2 * entries.capacity()) : size);
			};
			grow(factorials_);
			grow(inverse_factorials_);
			grow(inverse_integers_);
			factorials_.resize(size);
			inverse_factorials_.resize(size);
			inverse_integers_.resize(size);
			for (std::size_t i = previous + 1; i <= maximum_index; ++i) factorials_[i] = factorials_[i - 1] * Coefficient(i);
			inverse_factorials_[maximum_index] = factorials_[maximum_index].inverse();
			for (std::size_t i = maximum_index; i > previous + 1; --i) inverse_factorials_[i - 1] = inverse_factorials_[i] * Coefficient(i);
			for (std::size_t i = previous + 1; i <= maximum_index; ++i) inverse_integers_[i] = inverse_factorials_[i] * factorials_[i - 1];
		}

		/** @brief Largest stored index: the constructor's argument, or the largest one `extend` was given. */
		constexpr std::size_t maximum_index() const { return factorials_.size() - 1; }

		/**
		 * @brief The factorial `i!`, with `0! = 1`.
		 * @pre `i <= maximum_index()` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient factorial(std::size_t i) const {
			atl_assert(i <= maximum_index());
			return factorials_[i];
		}

		/**
		 * @brief The inverse factorial `1 / i!`.
		 * @pre `i <= maximum_index()` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient inverse_factorial(std::size_t i) const {
			atl_assert(i <= maximum_index());
			return inverse_factorials_[i];
		}

		/**
		 * @brief The inverse `1 / i` of a positive integer.
		 * @pre `1 <= i <= maximum_index()` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient inverse_integer(std::size_t i) const {
			atl_assert(i >= 1);
			atl_assert(i <= maximum_index());
			return inverse_integers_[i];
		}

		/**
		 * @brief The binomial coefficient `n choose k`, zero when `k > n`.
		 * @pre `n <= maximum_index()` (asserted, also when `k > n`).
		 */
		[[nodiscard]] constexpr Coefficient binomial(std::size_t n, std::size_t k) const {
			atl_assert(n <= maximum_index());
			if (k > n) return Coefficient(0);
			return factorials_[n] * inverse_factorials_[k] * inverse_factorials_[n - k];
		}
	};
}
