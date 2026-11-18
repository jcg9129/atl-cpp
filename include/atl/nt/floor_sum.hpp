#pragma once

/**
 * @file
 * @brief Euclid-like recursions over the line `y = (a·x + b) / m`: the floor sum
 *        `Σ floor((a·i + b) / m)`, the monoid product along the line's staircase
 *        (the "universal Euclidean algorithm"), and the first `x` whose residue
 *        `(a·x + b) mod m` lands in a range. Each runs in O(log m) steps.
 *
 * References: Jin Bin, "Applications of the Euclidean Algorithm" (2009), §4.2
 * (the floor-sum identity); Dai Jiangqi (2022), §3.3 (the staircase word);
 * Yang Xinhe, "On a Class of Linear Congruence Inequality Problems" (2025),
 * §3 (the least solution of `L <= a·x mod m <= R`). Implemented from the ideas,
 * every function checked against a brute force.
 *
 * The wide intermediates use atl::uint128 on every target, including its
 * software implementation.
 *
 * All three functions take the line's `m, a, b` in that order (modulus, step,
 * offset), following the length `n` where there is one.
 */

#include <atl/config.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace atl {
	namespace detail {
		/**
		 * @brief `Σ_{i<n} floor((a·i + b) / m)` for `a, b < m`, all arithmetic in the
		 *        unsigned type `W`.
		 *
		 * Each round pulls the integer parts of `a/m` out of the sum, then counts the
		 * same lattice points row by row instead of column by column: with
		 * `t = a·n + b`, the sum equals the floor sum over `floor(t / m)` terms with
		 * step `m`, modulus `a` and offset `t mod m` (Jin Bin's identity (7)). The
		 * pair `(m, a)` follows the Euclidean algorithm, so there are O(log m) rounds.
		 *
		 * @pre `a < m`, `b < m`, and `a·n + b` and the result fit `W`: the result is
		 *      at most `n·(n - 1) / 2`, and every partial sum is a part of it.
		 * @complexity O(log m) divisions in `W`
		 */
		template<unsigned_integral W>
		constexpr W floor_sum_reduced(W n, W m, W a, W b) {
			W sum = 0;
			while (true) {
				if (a >= m) {
					// n·(n-1)/2 halved on the even factor, so it never overflows
					W const pairs = (n % 2 == 0) ? (n / 2) * (n - 1) : n * ((n - 1) / 2);
					sum += (a / m) * pairs;
					a %= m;
				}
				if (b >= m) {
					sum += (b / m) * n;
					b %= m;
				}
				W const top = a * n + b;
				if (top < m) return sum;
				n = top / m;
				b = top % m;
				std::swap(m, a);
			}
		}

		struct quotient_remainder {
			uint64 quotient, remainder;
		};

		/**
		 * @brief `floor((x·y + z) / d)` and `(x·y + z) mod d` with no type wider than
		 *        64 bits: a double-and-add arithmetic oracle.
		 *
		 * The integer parts of `x / d` and `z / d` are pulled out first; the rest,
		 * `(x mod d)·y`, is accumulated by double-and-add over the bits of `y` with the
		 * running remainder kept below `d` (by modular double-and-add).
		 *
		 * @pre `d >= 1` (asserted); the quotient fits `uint64`.
		 * @complexity O(log y)
		 */
		constexpr quotient_remainder mul_add_divmod(uint64 x, uint64 y, uint64 z, uint64 d) {
			atl_assert(d >= 1);
			uint64 const outer = (x / d) * y + z / d;
			x %= d;
			z %= d;
			uint64 quotient = 0, remainder = 0;
			auto const add = [&](uint64 v) { // v < d
				if (remainder >= d - v) {
					remainder -= d - v;
					++quotient;
				} else {
					remainder += v;
				}
			};
			for (int bit = int(std::bit_width(y)) - 1; bit >= 0; --bit) {
				quotient <<= 1;
				add(remainder);
				if ((y >> bit) & 1) add(x);
			}
			add(z);
			return {outer + quotient, remainder};
		}

		/**
		 * @brief Reduced floor sum that starts in 128-bit divisions and drops to
		 *        64-bit ones as soon as the running `n` and `m` are both below 2^32
		 *        (they only shrink, and then `a·n + b < 2^64` and the rest of the sum
		 *        is below 2^63).
		 *
		 * Against always-128-bit and a 64/128 choice made once at entry
		 * (benchmarks/nt/floor_sum.cpp, Xeon 8358, plain -O2, 1e5 queries, 2026-10-15):
		 * within 7% of the entry choice at n, m < 1e9 (28.4 vs 26.6 ms) and 1.4-1.7x
		 * ahead of both once the inputs start wide (n, m < 1e18: 64.2 vs 92.3 / 107 ms;
		 * n < 1e18, m < 1e9: 36.6 vs 46.2 / 53.7 ms).
		 *
		 * @pre `a < m`, `b < m`, `n, m < 2^63`.
		 * @complexity O(log m) divisions
		 */
		constexpr uint128 floor_sum_narrowing(uint128 n, uint128 m, uint128 a, uint128 b) {
			constexpr uint128 narrow = uint128(1) << 32;
			uint128 sum = 0;
			while (n >= narrow || m >= narrow) {
				// invariant: a, b < m
				uint128 const top = a * n + b;
				if (top < m) return sum;
				n = top / m;
				b = top % m;
				std::swap(m, a); // top >= m forces a > 0, so the denominator stays positive
				uint128 const pairs = (n % 2 == 0) ? (n / 2) * (n - 1) : n * ((n - 1) / 2);
				sum += (a / m) * pairs;
				a %= m;
				sum += (b / m) * n;
				b %= m;
			}
			return sum + floor_sum_reduced<uint64>(uint64(n), uint64(m), uint64(a), uint64(b));
		}

		/** @brief The counters of @ref atl::floor_staircase_product: wide enough for `step·count + offset`. */
		using staircase_word = uint128;

		/**
		 * @brief `floor((x·y + z) / d)` with wide intermediates.
		 * @pre `x, y, z < 2^64`, `d >= 1`, and the quotient fits `uint64`.
		 */
		constexpr staircase_word staircase_mul_add_div(staircase_word x, staircase_word y, staircase_word z,
													   staircase_word d) {
			return (x * y + z) / d;
		}

		/** @brief Copyable accumulator with assignable products and swappable letters. */
		template<typename Monoid, typename Op>
		concept staircase_monoid = binary_operation<Op, Monoid> && std::is_copy_constructible_v<Monoid>
								&& std::is_move_constructible_v<Monoid> && std::is_move_assignable_v<Monoid>
								&& requires(Monoid &value, Op const &operation, Monoid const &other) {
									   value = std::invoke(operation, other, other);
									   { std::invoke(operation, other, other) } -> std::convertible_to<Monoid>;
									   std::swap(value, value);
								   };

		/**
		 * @brief `x^e` under the associative @p op with @p identity, by binary
		 *        exponentiation: O(log e) applications of @p op.
		 */
		template<typename Monoid, typename Op>
			requires detail::staircase_monoid<Monoid, Op>
		constexpr Monoid monoid_power(Monoid x, staircase_word e, Monoid const &identity, Op const &op) {
			Monoid result = identity;
			while (e > 0) {
				if (e & 1) result = std::invoke(op, result, x);
				e >>= 1;
				if (e > 0) x = std::invoke(op, x, x);
			}
			return result;
		}

		/**
		 * @brief Least `x` with `lo <= a·x mod m <= hi`, or UINT64_MAX if absent.
		 * The sentinel cannot be a solution: each solution is below `m`. Scalar
		 * recursion also permits constant evaluation on MSVC.
		 *
		 * The points `(x, floor(a·x / m))` fall into rows of equal `y`, and within a
		 * row the residues `a·x - m·y` climb by `a`. Row 0 is checked directly. When it
		 * misses, `hi - lo + 1 < a`, so each row holds at most one hit, and row `y` has
		 * one exactly when `(-m·y) mod a` lies in `[lo mod a, hi mod a]` (an interval
		 * that does not wrap, since `[lo, hi]` holds no multiple of `a`): the same
		 * problem for `y`, with modulus `a`. Reflecting `a -> m - a` (and the range
		 * with it) when `2a > m` keeps `a <= m / 2`, so the modulus at least halves
		 * every round and there are O(log m) rounds.
		 *
		 * @pre `a < m` and `lo <= hi < m`.
		 * @complexity O(log m)
		 */
		constexpr uint64 first_multiple_in_range(uint64 a, uint64 m, uint64 lo, uint64 hi) {
			if (lo == 0) return uint64(0);
			if (a == 0) return (std::numeric_limits<uint64>::max)();
			// a·x ≡ s <=> (m - a)·x ≡ m - s, and 0 is in neither range since lo >= 1
			if (a > m - a) return first_multiple_in_range(m - a, m, m - hi, m - lo);
			uint64 const first = lo / a + (lo % a != 0);
			if (uint128(first) * a <= hi) return first;
			auto const row = first_multiple_in_range((a - m % a) % a, a, lo % a, hi % a);
			if (row == (std::numeric_limits<uint64>::max)()) return row;
			uint128 const bottom = uint128(lo) + uint128(m) * row;
			return uint64(bottom / a + (bottom % a != 0));
		}
	}

	/**
	 * @brief `Σ_{i=0}^{n-1} floor((a·i + b) / m)`, exactly, for signed `a` and `b`.
	 *
	 * Floors round toward negative infinity. With `a, b >= 0` this counts the lattice
	 * points `(i, j)` with `0 <= i < n` and `1 <= j <= (a·i + b) / m`: the points
	 * under a line segment.
	 *
	 * `a` and `b` are first reduced into `[0, m)` (their quotients contribute
	 * `floor(a/m)·n·(n-1)/2 + floor(b/m)·n`); the reduced sum runs in 128-bit
	 * divisions until `n` and `m` shrink below 2^32, then in 64-bit ones.
	 *
	 * @param n Number of terms.
	 * @param m Denominator.
	 * @param a Step of the numerator.
	 * @param b Offset of the numerator.
	 * @return The exact sum, as `int128`.
	 * @pre `n >= 0` and `m >= 1` (asserted); the exact result fits `int128`
	 *      (asserted without overflowing an intermediate). A sufficient bound is
	 *      `|floor(a/m)|·n·(n-1)/2 + |floor(b/m)|·n + n·(n-1)/2 <= 2^127-1`.
	 *      In particular it fits when `|a|, |b| < m`.
	 *
	 * @complexity O(log m)
	 */
	[[nodiscard]] constexpr int128 floor_sum(int64 n, int64 m, int64 a, int64 b) {
		atl_assert(n >= 0);
		atl_assert(m >= 1);
		// floor division, so the reduced step and offset lie in [0, m)
		int64 a_quotient = a / m, a_rem = a % m;
		if (a_rem < 0) a_rem += m, --a_quotient;
		int64 b_quotient = b / m, b_rem = b % m;
		if (b_rem < 0) b_rem += m, --b_quotient;

		// 0 <= reduced <= n·(n-1)/2 < 2^125
		int128 const reduced =
			int128(detail::floor_sum_narrowing(uint128(n), uint128(m), uint128(a_rem), uint128(b_rem)));

		uint128 const pairs = uint128(n) * uint128(n == 0 ? 0 : n - 1) / 2;
		int128 const tail = int128(b_quotient) * n + reduced;
		uint128 const sign_bit = uint128(1) << 127;
		uint128 const tail_magnitude = tail < 0 ? uint128(-tail) : uint128(tail);
		bool const is_negative_step = a_quotient < 0;
		uint128 const a_magnitude = is_negative_step ? uint128(-int128(a_quotient)) : uint128(a_quotient);
		// Bound the step using the exact tail, allowing cancellation without
		// ever constructing an overflowing signed product or partial sum.
		uint128 const limit = is_negative_step
								? (tail < 0 ? sign_bit - tail_magnitude : sign_bit + tail_magnitude)
								: (tail < 0 ? sign_bit - 1 + tail_magnitude : sign_bit - 1 - tail_magnitude);
		atl_assert(pairs == 0 || a_magnitude <= limit / pairs);
		uint128 const step_magnitude = a_magnitude * pairs;
		bool const is_negative_tail = tail < 0;
		bool const is_negative = step_magnitude == 0				  ? is_negative_tail
							   : is_negative_step == is_negative_tail ? is_negative_step
							   : step_magnitude >= tail_magnitude	  ? is_negative_step
																	  : is_negative_tail;
		uint128 const magnitude = is_negative_step == is_negative_tail ? step_magnitude + tail_magnitude
								: step_magnitude >= tail_magnitude	   ? step_magnitude - tail_magnitude
																	   : tail_magnitude - step_magnitude;
		// The negative endpoint has no positive signed counterpart.
		return is_negative && magnitude != 0 ? -int128(magnitude - 1) - 1 : int128(magnitude);
	}

	/**
	 * @brief The product, under a monoid, of the staircase word of the line
	 *        `y = (a·x + b) / m` over `0 <= x <= n` (the "universal Euclidean
	 *        algorithm").
	 *
	 * With `f(x) = floor((a·x + b) / m)`, the word is `up^f(0)`, then for each
	 * `x = 1..n` the letters `up^(f(x) - f(x-1))` followed by one `right`: walk the
	 * staircase under the line from `(0, 0)`, writing `up` for each unit climbed and
	 * `right` for each unit stepped right (at a lattice point on the line the climb
	 * comes first). It holds `f(n)` copies of `up` and `n` of `right`, and the
	 * function returns their product under @p op, left to right.
	 *
	 * Sums such as `Σ f(x)`, `Σ x·f(x)`, `Σ f(x)²` or `Σ A^x·B^f(x)` for matrices
	 * are such products, each with its own small monoid.
	 *
	 * @tparam Monoid The element type.
	 * @tparam Op An associative binary operation on `Monoid` with identity
	 *                @p identity (a precondition, as for the reducing structures);
	 *                it need not be commutative.
	 * @param n Rightmost `x`.
	 * @param m Denominator.
	 * @param a Step of the numerator.
	 * @param b Offset of the numerator.
	 * @param up The element for one unit climbed.
	 * @param right The element for one unit stepped right.
	 * @param identity The identity of @p op (the product of the empty word).
	 * @param op The monoid operation, invoked through std::invoke.
	 * @return The ordered staircase product.
	 * @pre Monoid is copy-constructible, move-assignable and swappable (constrained).
	 *      Every evaluated operation is valid; any numeric fields fit their types.
	 * @pre `m >= 1` (asserted).
	 *
	 * @complexity O(log max(m, a, n) + log(b / m)) applications of @p op: each
	 *             Euclidean round raises its letters to powers whose logarithms
	 *             telescope. Wide arithmetic uses atl::uint128 on every target.
	 */
	template<typename Monoid, typename Op>
		requires detail::staircase_monoid<Monoid, Op>
	[[nodiscard]] constexpr Monoid floor_staircase_product(
		uint64 n, uint64 m, uint64 a, uint64 b, Monoid up, Monoid right,
		Monoid const &identity, Op op) {
		atl_assert(m >= 1);
		Monoid prefix = detail::monoid_power(up, b / m, identity, op);
		Monoid suffix = identity;
		detail::staircase_word count = n, denominator = m, step = a, offset = b % m;
		// invariant: answer = prefix · (word of the current line) · suffix, offset < denominator
		while (true) {
			if (step >= denominator) {
				right = std::invoke(op, detail::monoid_power(up, step / denominator, identity, op), right);
				step %= denominator;
			}
			detail::staircase_word const climbs = detail::staircase_mul_add_div(step, count, offset, denominator);
			if (climbs == 0) {
				prefix = std::invoke(op, prefix, detail::monoid_power(right, count, identity, op));
				return std::invoke(op, prefix, suffix);
			}
			// The j-th up (j = 1..climbs) follows h(j-1) rights, where
			// h(k) = floor((denominator·k + flipped) / step): the transposed line.
			detail::staircase_word const flipped = denominator - offset - 1;
			detail::staircase_word const first_run = flipped / step;
			detail::staircase_word const last_run =
				count - detail::staircase_mul_add_div(denominator, climbs - 1, flipped, step);
			prefix = std::invoke(op, prefix, detail::monoid_power(right, first_run, identity, op));
			prefix = std::invoke(op, prefix, up);
			suffix = std::invoke(op, detail::monoid_power(right, last_run, identity, op), suffix);
			count = climbs - 1;
			offset = flipped % step;
			std::swap(denominator, step);
			std::swap(up, right);
		}
	}

	/**
	 * @brief The least `x >= 0` with `lo <= (a·x + b) mod m <= hi`, or none.
	 *
	 * The residues `(a·x + b) mod m` repeat with period `m / gcd(a, m)`, so a solution,
	 * when one exists, is below `m`. A Euclid-like recursion on `(a, m)` finds it
	 * without enumerating (Yang Xinhe 2025, §3).
	 *
	 * @param m Modulus.
	 * @param a Step (any value; reduced mod m).
	 * @param b Offset (any value; reduced mod m).
	 * @param lo Lower end of the target range, inclusive.
	 * @param hi Upper end of the target range, inclusive.
	 * @return The least such `x`, which is below `m`, or `std::nullopt` when no `x`
	 *         lands in the range.
	 * @pre `m >= 1` and `lo <= hi < m` (asserted).
	 *
	 * @complexity O(log m)
	 */
	[[nodiscard]] constexpr std::optional<uint64> first_residue_in_range(
		uint64 m, uint64 a, uint64 b, uint64 lo, uint64 hi) {
		atl_assert(m >= 1);
		atl_assert(lo <= hi && hi < m);
		a %= m;
		b %= m;
		// a·x mod m must land in [lo - b, hi - b] mod m; when that range wraps it
		// contains 0, so x = 0 is the answer
		uint64 const shifted_lo = lo >= b ? lo - b : lo + (m - b);
		uint64 const shifted_hi = hi >= b ? hi - b : hi + (m - b);
		if (shifted_lo > shifted_hi) return uint64(0);
		uint64 const answer = detail::first_multiple_in_range(a, m, shifted_lo, shifted_hi);
		if (answer == (std::numeric_limits<uint64>::max)()) return std::nullopt;
		return answer;
	}
}
