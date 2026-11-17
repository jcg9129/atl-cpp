#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/detail/checked_arithmetic.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Segment tree beats: @ref atl::segment_tree_beats, the lazy tree whose
 *        tag application may fail and descend, with the library algebras of
 *        @ref atl::beats (clamp + add over extremes, modulo + assign, floor
 *        sqrt / divide + add, bitwise and / or over max), and the everyday class
 *        @ref atl::beats_segment_tree, hand-written for clamp + add.
 */

namespace atl {
	namespace detail {
		/** @brief `a + b`, asserting (under `ATL_ASSERT`) that it does not overflow. */
		template<integral T>
		[[nodiscard]] constexpr T beats_add(T a, T b) noexcept {
			T result{};
			atl_assert(!add_overflows(a, b, result));
			return T(a + b);
		}

		/** @brief `a - b`, asserting (under `ATL_ASSERT`) that it does not overflow. */
		template<integral T>
		[[nodiscard]] constexpr T beats_sub(T a, T b) noexcept {
			T result{};
			atl_assert(!sub_overflows(a, b, result));
			return T(a - b);
		}

		/** @brief `a * b`, asserting (under `ATL_ASSERT`) that it does not overflow. */
		template<integral T>
		[[nodiscard]] constexpr T beats_mul(T a, T b) noexcept {
			T result{};
			atl_assert(!mul_overflows(a, b, result));
			return T(a * b);
		}

		/**
		 * @brief An element value shifted by @p delta: overflow-checked, and the
		 *        result asserted to stay strictly inside the type's range, whose two
		 *        ends are the "no value" sentinels of @ref atl::beats::extremes.
		 */
		template<signed_integral T>
		[[nodiscard]] constexpr T beats_shift_value(T value, T delta) noexcept {
			T const r = beats_add(value, delta);
			atl_assert((std::numeric_limits<T>::lowest)() < r && r < (std::numeric_limits<T>::max)());
			return r;
		}

		/**
		 * @brief A clamp bound of @ref atl::beats::clamp_add shifted by @p delta,
		 *        saturating: the sentinels ("no bound") stay, and a bound pushed past
		 *        either end of the type becomes that end's sentinel, which every valid
		 *        value already lies inside.
		 */
		template<signed_integral T>
		[[nodiscard]] constexpr T beats_shift_bound(T bound, T delta) noexcept {
			constexpr T lowest = (std::numeric_limits<T>::lowest)(), highest = (std::numeric_limits<T>::max)();
			if (bound == lowest || bound == highest) return bound;
			T result{};
			if (add_overflows(bound, delta, result)) return delta > 0 ? highest : lowest;
			return result;
		}

		/** @brief `floor(a / d)` for `d > 0` (C++ `/` truncates toward zero). */
		template<integral T>
		[[nodiscard]] constexpr T beats_floor_divide(T a, T d) noexcept {
			T q = T(a / d);
			if constexpr (signed_integral<T>)
				if (a % d != 0 && a < 0) --q;
			return q;
		}

		/**
		 * @brief `floor(sqrt(v))` for `v >= 0`. At run time, types up to 64 bits take
		 *        the rounded double root and correct it by at most a step either way;
		 *        constant evaluation and 128-bit types take integer Newton from a
		 *        power of two above the root, which decreases to it monotonically.
		 */
		template<signed_integral T>
		[[nodiscard]] constexpr T beats_floor_sqrt(T v) noexcept {
			atl_assert(v >= 0);
			if (v < 2) return v;
			// Every branch returns, with nothing after it: MSVC reports C4702 for code after
			// an `if !consteval` whose runtime branch always returns.
			auto const newton = [v] {
				int bits = 0;
				for (T t = v; t != 0; t >>= 1) ++bits;
				T x = T(1) << ((bits + 1) / 2);
				for (;;) {
					T const y = T((x + v / x) / 2);
					if (y >= x) return x;
					x = y;
				}
			};
			if consteval {
				return newton();
			} else {
				if constexpr (sizeof(T) <= 8) {
					// v < 2^63, so r <= 3037000500 and (r + 1)^2 fits uint64
					uint64 const u = uint64(v);
					uint64 r = uint64(std::sqrt(double(u)));
					while (r * r > u) --r;
					while ((r + 1) * (r + 1) <= u) ++r;
					return T(r);
				} else {
					return newton();
				}
			}
		}
	}

	/**
	 * @brief The library algebras of `segment_tree_beats`: value (summary) and
	 *        tag types, and the three callables `merge`, `mapping`, `compose` that
	 *        are the engine's default `Op`, `Mapping` and `Compose`.
	 *
	 * Four sets ship, each a summary type with the tag it is meant for:
	 *
	 * | summary | tag | named members it switches on |
	 * |---|---|---|
	 * | @ref atl::beats::extremes | @ref atl::beats::clamp_add | `chmin`, `chmax`, `add`, `assign`; `sum`, `min`, `max` |
	 * | @ref atl::beats::mod_sum | @ref atl::beats::assign | `modulo`, `assign`; `sum`, `min`, `max` |
	 * | @ref atl::beats::sqrt_add_sum | @ref atl::beats::add | `floor_sqrt`, `divide`, `add`; `sum`, `min`, `max` |
	 * | @ref atl::beats::bits_max | @ref atl::beats::bit_mask | `bit_and`, `bit_or`; `max` |
	 *
	 * Identities come from the types (`T::identity()`), and a summary builds its
	 * leaf from an element (`Value::of(x)`), so the engine needs no identity
	 * arguments for them (@ref atl::beats::summary, @ref atl::beats::tag).
	 */
	namespace beats {
		/**
		 * @brief A summary type that knows its identity and how to summarize one
		 *        element: `Value::identity()`, `Value::of(element_type)`. Switches on
		 *        the identity-free constructors of @ref atl::segment_tree_beats and its
		 *        element-level members.
		 */
		template<typename Value>
		concept summary = std::copyable<Value> && requires { typename Value::element_type; { Value::identity() } -> std::same_as<Value>; }
					   && requires(typename Value::element_type const &x) { { Value::of(x) } -> std::same_as<Value>; };

		/** @brief A tag type that knows its identity: `Tag::identity()`. */
		template<typename Tag>
		concept tag = std::copyable<Tag> && requires { { Tag::identity() } -> std::same_as<Tag>; };

		/**
		 * @brief The summary of the range chmin / chmax / add / assign algebra:
		 *        maximum, strict second maximum and the maximum's multiplicity, the
		 *        same for the minimum, the sum and the number of elements.
		 *
		 * The two ends of `Element`'s range are sentinels: `lowest()` is "no second
		 * maximum", `max()` is "no second minimum", and the identity (an empty
		 * range, `width == 0`) holds both. Values are asserted to lie strictly
		 * between them.
		 *
		 * @tparam Element The element type (signed: chmax and add need negatives).
		 * @tparam Sum     The sum type, at least as wide as `Element`; `int128` is
		 *                 the opt-in for sums past 64 bits.
		 */
		template<signed_integral Element, signed_integral Sum = Element>
			requires(std::numeric_limits<Sum>::digits >= std::numeric_limits<Element>::digits)
		struct extremes {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;
			/** @brief Type used for aggregate sums. */
			using sum_type = Sum;

			Element max = (std::numeric_limits<Element>::lowest)();		   ///< largest element
			Element second_max = (std::numeric_limits<Element>::lowest)(); ///< largest element below `max`, or `lowest()`
			Element min = (std::numeric_limits<Element>::max)();		   ///< smallest element
			Element second_min = (std::numeric_limits<Element>::max)();	   ///< smallest element above `min`, or `(max)()`
			Sum sum = 0;												   ///< sum of the elements
			uint32 max_count = 0;										   ///< elements equal to `max`
			uint32 min_count = 0;										   ///< elements equal to `min`
			uint32 width = 0;											   ///< number of elements

			/** @brief The summary of an empty range. */
			[[nodiscard]] static constexpr extremes identity() noexcept { return {}; }

			/** @brief The summary of the single element @p x. @pre `x` strictly inside `Element`'s range */
			[[nodiscard]] static constexpr extremes of(Element x) noexcept {
				atl_assert((std::numeric_limits<Element>::lowest)() < x && x < (std::numeric_limits<Element>::max)());
				return {x, (std::numeric_limits<Element>::lowest)(), x, (std::numeric_limits<Element>::max)(), Sum(x), 1, 1, 1};
			}

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(extremes const &) const = default;
		};

		/**
		 * @brief The tag `v -> min(max(v + add, lower), upper)`: add, then clamp. Closed
		 *        under composition, so range add, chmin (`upper`), chmax (`lower`) and
		 *        assign (`lower == upper`) share it. The sentinels `lowest()` /
		 *        `max()` mean "no bound".
		 *
		 * @tparam Element The element type.
		 */
		template<signed_integral Element>
		struct clamp_add {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;

			Element add = 0;										  ///< added first
			Element lower = (std::numeric_limits<Element>::lowest)(); ///< then raised to at least this
			Element upper = (std::numeric_limits<Element>::max)();	  ///< then lowered to at most this

			/** @brief The identity map. */
			[[nodiscard]] static constexpr clamp_add identity() noexcept { return {}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(clamp_add const &) const = default;
		};

		/**
		 * @brief The summary of the range modulo / range assign algebra: sum, maximum,
		 *        minimum and number of elements. The maximum decides when a modulo is a
		 *        no-op; `max == min` makes it an assignment.
		 *
		 * @tparam Element The element type.
		 * @tparam Sum     The sum type, at least as wide as `Element`.
		 */
		template<integral Element, integral Sum = Element>
			requires(std::numeric_limits<Sum>::digits >= std::numeric_limits<Element>::digits)
		struct mod_sum {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;
			/** @brief Type used for aggregate sums. */
			using sum_type = Sum;

			Sum sum = 0;											///< sum of the elements
			Element max = (std::numeric_limits<Element>::lowest)(); ///< largest element (`lowest()` when empty)
			Element min = (std::numeric_limits<Element>::max)();	///< smallest element (`(max)()` when empty)
			uint32 width = 0;										///< number of elements

			/** @brief The summary of an empty range. */
			[[nodiscard]] static constexpr mod_sum identity() noexcept { return {}; }

			/** @brief The summary of the single element @p x. */
			[[nodiscard]] static constexpr mod_sum of(Element x) noexcept { return {Sum(x), x, x, 1}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(mod_sum const &) const = default;
		};

		/**
		 * @brief The tag "assign `value`" (when `has_value`) or the identity.
		 *
		 * @tparam Element The element type.
		 */
		template<integral Element>
		struct assign {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;

			Element value = 0;		///< the value assigned
			bool has_value = false; ///< false: the identity (and `value` is 0)

			/** @brief The identity map. */
			[[nodiscard]] static constexpr assign identity() noexcept { return {}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(assign const &) const = default;
		};

		/**
		 * @brief The summary of the range floor-sqrt / floor-divide / add algebra:
		 *        sum, maximum, minimum and number of elements. A floor sqrt or division
		 *        is an add on every node whose maximum and minimum drop by the same
		 *        amount.
		 *
		 * @tparam Element The element type.
		 * @tparam Sum     The sum type, at least as wide as `Element`.
		 */
		template<signed_integral Element, signed_integral Sum = Element>
			requires(std::numeric_limits<Sum>::digits >= std::numeric_limits<Element>::digits)
		struct sqrt_add_sum {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;
			/** @brief Type used for aggregate sums. */
			using sum_type = Sum;

			Sum sum = 0;											///< sum of the elements
			Element max = (std::numeric_limits<Element>::lowest)(); ///< largest element (`lowest()` when empty)
			Element min = (std::numeric_limits<Element>::max)();	///< smallest element (`(max)()` when empty)
			uint32 width = 0;										///< number of elements

			/** @brief The summary of an empty range. */
			[[nodiscard]] static constexpr sqrt_add_sum identity() noexcept { return {}; }

			/** @brief The summary of the single element @p x. */
			[[nodiscard]] static constexpr sqrt_add_sum of(Element x) noexcept { return {Sum(x), x, x, 1}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(sqrt_add_sum const &) const = default;
		};

		/**
		 * @brief The tag "add `delta`".
		 *
		 * @tparam Element The element type.
		 */
		template<signed_integral Element>
		struct add {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;

			Element delta = 0; ///< the amount added

			/** @brief The identity map. */
			[[nodiscard]] static constexpr add identity() noexcept { return {}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(add const &) const = default;
		};

		/**
		 * @brief The summary of the range bitwise and / or algebra: the bits set in
		 *        every element, the bits set in some element, and the maximum.
		 *
		 * The identity (an empty range) is the only summary with a bit in
		 * `all_ones` that is not in `any_ones`.
		 *
		 * @tparam Element The element type (unsigned, so that the order is the
		 *                 order of the bit patterns).
		 */
		template<unsigned_integral Element>
		struct bits_max {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;

			Element all_ones = Element(~Element(0)); ///< bits set in every element
			Element any_ones = 0;					 ///< bits set in at least one element
			Element max = 0;						 ///< largest element (0 when empty)

			/** @brief The summary of an empty range. */
			[[nodiscard]] static constexpr bits_max identity() noexcept { return {}; }

			/** @brief The summary of the single element @p x. */
			[[nodiscard]] static constexpr bits_max of(Element x) noexcept { return {x, x, x}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(bits_max const &) const = default;
		};

		/**
		 * @brief The tag `v -> (v & and_mask) | or_mask`, closed under composition:
		 *        `bit_and(x)` is `{x, 0}`, `bit_or(x)` is `{~0, x}`.
		 *
		 * @tparam Element The element type.
		 */
		template<unsigned_integral Element>
		struct bit_mask {
			/** @brief Single-element type summarized by this algebra. */
			using element_type = Element;

			Element and_mask = Element(~Element(0)); ///< bits kept
			Element or_mask = 0;					 ///< bits then set

			/** @brief The identity map. */
			[[nodiscard]] static constexpr bit_mask identity() noexcept { return {}; }

			/** @brief Compare every stored field. @complexity O(1) */
			constexpr bool operator==(bit_mask const &) const = default;
		};

		/** @brief The element operation of every library summary (the engine's default `Op`). */
		struct merge {
			/** @brief Extremes: each side's top two values and counts, sums and widths added. */
			template<signed_integral Element, signed_integral Sum>
			[[nodiscard]] constexpr extremes<Element, Sum> operator()(extremes<Element, Sum> const &a, extremes<Element, Sum> const &b) const noexcept {
				extremes<Element, Sum> r;
				if (a.max > b.max) {
					r.max = a.max, r.max_count = a.max_count, r.second_max = (std::max)(a.second_max, b.max);
				} else if (a.max < b.max) {
					r.max = b.max, r.max_count = b.max_count, r.second_max = (std::max)(a.max, b.second_max);
				} else {
					r.max = a.max, r.max_count = a.max_count + b.max_count, r.second_max = (std::max)(a.second_max, b.second_max);
				}
				if (a.min < b.min) {
					r.min = a.min, r.min_count = a.min_count, r.second_min = (std::min)(a.second_min, b.min);
				} else if (a.min > b.min) {
					r.min = b.min, r.min_count = b.min_count, r.second_min = (std::min)(a.min, b.second_min);
				} else {
					r.min = a.min, r.min_count = a.min_count + b.min_count, r.second_min = (std::min)(a.second_min, b.second_min);
				}
				r.sum = detail::beats_add(a.sum, b.sum);
				r.width = a.width + b.width;
				return r;
			}

			/** @brief Modulo summary: sums and widths added, extremes taken. */
			template<integral Element, integral Sum>
			[[nodiscard]] constexpr mod_sum<Element, Sum> operator()(mod_sum<Element, Sum> const &a, mod_sum<Element, Sum> const &b) const noexcept {
				return {detail::beats_add(a.sum, b.sum), (std::max)(a.max, b.max), (std::min)(a.min, b.min), a.width + b.width};
			}

			/** @brief Sqrt summary: sums and widths added, extremes taken. */
			template<signed_integral Element, signed_integral Sum>
			[[nodiscard]] constexpr sqrt_add_sum<Element, Sum> operator()(sqrt_add_sum<Element, Sum> const &a, sqrt_add_sum<Element, Sum> const &b) const noexcept {
				return {detail::beats_add(a.sum, b.sum), (std::max)(a.max, b.max), (std::min)(a.min, b.min), a.width + b.width};
			}

			/** @brief Bits summary: and of the all-ones, or of the any-ones, max of the maxima. */
			template<unsigned_integral Element>
			[[nodiscard]] constexpr bits_max<Element> operator()(bits_max<Element> const &a, bits_max<Element> const &b) const noexcept {
				return {Element(a.all_ones & b.all_ones), Element(a.any_ones | b.any_ones), (std::max)(a.max, b.max)};
			}
		};

		/**
		 * @brief The fallible tag application of every library pair (the engine's
		 *        default `Mapping`): `true` when the summary absorbed the tag, `false`
		 *        when the engine must descend.
		 */
		struct mapping {
			/**
			 * @brief Clamp-add on extremes. The add always applies. The clamp then
			 *        succeeds when it is constant on `[min, max]` (it flattens the node
			 *        into one value), or when it moves only the minimum's class (the
			 *        lower bound lies below the second minimum) and only the maximum's
			 *        (the upper bound lies above the second maximum). A single element
			 *        never fails, and a tag that succeeded on a node succeeds on its
			 *        children: they hold a subset of its values.
			 */
			template<signed_integral Element, signed_integral Sum>
			constexpr bool operator()(clamp_add<Element> const &f, extremes<Element, Sum> &s) const noexcept {
				atl_assert(f.lower <= f.upper);
				if (s.width == 0) return true;
				if (f.add != 0) {
					s.max = detail::beats_shift_value(s.max, f.add);
					s.min = detail::beats_shift_value(s.min, f.add);
					if (s.second_max != (std::numeric_limits<Element>::lowest)()) s.second_max = detail::beats_shift_value(s.second_max, f.add);
					if (s.second_min != (std::numeric_limits<Element>::max)()) s.second_min = detail::beats_shift_value(s.second_min, f.add);
					s.sum = detail::beats_add(s.sum, detail::beats_mul(Sum(f.add), Sum(s.width)));
				}
				// A clamp that is constant on [min, max] flattens the node. Checked before the
				// one-class moves: a pushed composite (say chmin to below everything, then
				// chmax) can be constant on a child that holds several values at either end.
				if (f.upper <= s.min) {
					flatten(s, f.upper);
					return true;
				}
				if (f.lower >= s.max || f.lower == f.upper) {
					flatten(s, f.lower);
					return true;
				}
				if (f.lower > s.min) {
					if (f.lower >= s.second_min) return false;
					s.sum = detail::beats_add(s.sum, detail::beats_mul(detail::beats_sub(Sum(f.lower), Sum(s.min)), Sum(s.min_count)));
					if (s.second_max == s.min) s.second_max = f.lower; // exactly two distinct values
					s.min = f.lower;
				}
				if (f.upper < s.max) {
					if (f.upper <= s.second_max) return false;
					s.sum = detail::beats_sub(s.sum, detail::beats_mul(detail::beats_sub(Sum(s.max), Sum(f.upper)), Sum(s.max_count)));
					if (s.second_min == s.max) s.second_min = f.upper;
					s.max = f.upper;
				}
				return true;
			}

			/** @brief Assign on the modulo summary; never fails. */
			template<integral Element, integral Sum>
			constexpr bool operator()(assign<Element> const &f, mod_sum<Element, Sum> &s) const noexcept {
				if (f.has_value && s.width != 0) s.sum = detail::beats_mul(Sum(f.value), Sum(s.width)), s.max = s.min = f.value;
				return true;
			}

			/** @brief Add on the sqrt summary; never fails. */
			template<signed_integral Element, signed_integral Sum>
			constexpr bool operator()(add<Element> const &f, sqrt_add_sum<Element, Sum> &s) const noexcept {
				if (s.width == 0 || f.delta == 0) return true;
				s.sum = detail::beats_add(s.sum, detail::beats_mul(Sum(f.delta), Sum(s.width)));
				s.max = detail::beats_add(s.max, f.delta);
				s.min = detail::beats_add(s.min, f.delta);
				return true;
			}

			/**
			 * @brief And-or on the bits summary: succeeds when every element agrees on
			 *        every bit the tag changes, which makes it the same shift for all of
			 *        them (so the maximum stays the maximum). A single element never
			 *        fails.
			 */
			template<unsigned_integral Element>
			constexpr bool operator()(bit_mask<Element> const &f, bits_max<Element> &s) const noexcept {
				if ((s.all_ones & Element(~s.any_ones)) != 0) return true; // the identity
				Element const changed = Element(Element(~f.and_mask) | f.or_mask);
				if (((s.all_ones ^ s.any_ones) & changed) != 0) return false;
				s.all_ones = Element((s.all_ones & f.and_mask) | f.or_mask);
				s.any_ones = Element((s.any_ones & f.and_mask) | f.or_mask);
				s.max = Element((s.max & f.and_mask) | f.or_mask);
				return true;
			}

		private:
			/** @brief Every element of @p s becomes @p x. */
			template<signed_integral Element, signed_integral Sum>
			static constexpr void flatten(extremes<Element, Sum> &s, Element x) noexcept {
				s.max = s.min = x;
				s.second_max = (std::numeric_limits<Element>::lowest)();
				s.second_min = (std::numeric_limits<Element>::max)();
				s.max_count = s.min_count = s.width;
				s.sum = detail::beats_mul(Sum(x), Sum(s.width));
			}
		};

		/** @brief The tag composition of every library tag, `compose(f, g)` = "`g` first, then `f`" (the engine's default `Compose`). */
		struct compose {
			/**
			 * @brief Clamp-add after clamp-add: the adds sum; `g`'s bounds, shifted by
			 *        `f.add` (saturating at the sentinels), are clamped into `f`'s. A
			 *        constant result drops its add, so assignments never accumulate
			 *        one.
			 */
			template<signed_integral Element>
			[[nodiscard]] constexpr clamp_add<Element> operator()(clamp_add<Element> const &f, clamp_add<Element> const &g) const noexcept {
				Element const lower = detail::beats_shift_bound(g.lower, f.add), upper = detail::beats_shift_bound(g.upper, f.add);
				clamp_add<Element> r;
				r.lower = (std::min)((std::max)(lower, f.lower), f.upper);
				r.upper = (std::min)((std::max)(upper, f.lower), f.upper);
				r.add = r.lower == r.upper ? Element(0) : detail::beats_add(g.add, f.add);
				return r;
			}

			/** @brief Assign after assign: the newer assignment wins. */
			template<integral Element>
			[[nodiscard]] constexpr assign<Element> operator()(assign<Element> const &f, assign<Element> const &g) const noexcept {
				return f.has_value ? f : g;
			}

			/** @brief Add after add. */
			template<signed_integral Element>
			[[nodiscard]] constexpr add<Element> operator()(add<Element> const &f, add<Element> const &g) const noexcept {
				return {detail::beats_add(g.delta, f.delta)};
			}

			/** @brief And-or after and-or. */
			template<unsigned_integral Element>
			[[nodiscard]] constexpr bit_mask<Element> operator()(bit_mask<Element> const &f, bit_mask<Element> const &g) const noexcept {
				return {Element(g.and_mask & f.and_mask), Element((g.or_mask & f.and_mask) | f.or_mask)};
			}
		};
	}

	namespace detail {
		/** @brief Stands in for a missing `element_type`, so member signatures stay well-formed. */
		struct beats_no_element {};

		template<typename T>
		struct beats_element {
			using type = beats_no_element;
		};

		template<typename T>
			requires requires { typename T::element_type; }
		struct beats_element<T> {
			using type = typename T::element_type;
		};

		template<typename Tag>
		inline constexpr bool is_beats_clamp_add = false;

		template<typename Element>
		inline constexpr bool is_beats_clamp_add<beats::clamp_add<Element>> = true;

		template<typename Tag>
		inline constexpr bool is_beats_add = false;

		template<typename Element>
		inline constexpr bool is_beats_add<beats::add<Element>> = true;

		template<typename Tag>
		inline constexpr bool is_beats_assign = false;

		template<typename Element>
		inline constexpr bool is_beats_assign<beats::assign<Element>> = true;

		template<typename Tag>
		inline constexpr bool is_beats_bit_mask = false;

		template<typename Element>
		inline constexpr bool is_beats_bit_mask<beats::bit_mask<Element>> = true;

		template<typename Value>
		inline constexpr bool is_beats_mod_sum = false;

		template<integral Element, integral Sum>
		inline constexpr bool is_beats_mod_sum<beats::mod_sum<Element, Sum>> = true;

		template<typename Value>
		inline constexpr bool is_beats_sqrt_add_sum = false;

		template<signed_integral Element, signed_integral Sum>
		inline constexpr bool is_beats_sqrt_add_sum<beats::sqrt_add_sum<Element, Sum>> = true;
	}

	/**
	 * @brief A per-node tag rule `rule(s)`: the tag a node with summary @p s absorbs
	 *        whole (`std::optional<Tag>` holding it), or `std::nullopt` to descend —
	 *        the node-dependent update of `segment_tree_beats::apply_derived`.
	 *
	 * @tparam Rule  The callable type.
	 * @tparam Value The summary type.
	 * @tparam Tag   The tag type.
	 */
	template<typename Rule, typename Value, typename Tag>
	concept beats_tag_rule = std::regular_invocable<Rule const &, Value const &>
						  && std::same_as<std::invoke_result_t<Rule const &, Value const &>, std::optional<Tag>>;

	/**
	 * @brief Segment tree beats: a lazy segment tree whose tag application may fail,
	 *        in which case the update descends ("beats") into the node's children.
	 *
	 * The algebra is @ref atl::lazy_segment_tree's, with one change: `mapping(f, s)`
	 * modifies the summary in place and returns whether it could absorb the tag
	 * from the summary alone (@ref atl::fallible_action). On `false` the tree pushes the
	 * node's pending tag, recurses into both children and recomputes the node. This
	 * is what range chmin with range sum needs: `a[i] = min(a[i], x)` changes a
	 * sum by an amount the sum does not determine, but a summary with the maximum,
	 * the strict second maximum and the maximum's count determines it whenever
	 * `x` lies above the second maximum.
	 *
	 *  - `(Value, op, identity)` is the element monoid;
	 *  - `(Tag, compose, tag_identity)` is the tag monoid, `compose(f, g)` meaning
	 *    "`g` first, then `f`";
	 *  - `mapping(f, s)` laws (documented, not checkable): a single element never
	 *    fails; a tag that succeeded on a node succeeds on both children's current
	 *    summaries (so pushes never fail — asserted); where it succeeds it agrees
	 *    with applying `f` to every element and folding.
	 *
	 * Node-dependent updates (floor sqrt, modulo: the tag a node can absorb depends
	 * on its values) go through `apply_derived`, whose rule computes the tag per
	 * node.
	 *
	 * **Complexity rule.** Each update costs O(log n) plus its descents. The
	 * descents are paid for by a potential: a quantity of the whole tree that is
	 * O((n + q) · log n · c) in total, drops on every failed node an update visits,
	 * and rises by at most O(log n · c) per update (the beats potential argument). An algebra without
	 * such a potential has no bound beyond O(n) per update. Library sets: clamp-add
	 * O(log² n) amortized per update, O(log n) without add; modulo with assign
	 * O(log n · log V); floor sqrt with add O(log n · log log V); divide with add
	 * O(log n · log V); and / or O(log n · w) for `w`-bit elements.
	 *
	 * The library sets (@ref atl::beats) switch on named members: `chmin`, `chmax`,
	 * `add`, `assign`, `modulo`, `floor_sqrt`, `divide`, `bit_and`, `bit_or` by the
	 * tag type, `sum`, `min`, `max`, `element` by the summary type; with those
	 * types the identities come from the types (`Value::identity()`), and the
	 * tree is built from plain elements. For chmin / chmax / add / assign +
	 * sum / min / max alone, @ref atl::beats_segment_tree is the faster hand-written
	 * class; this engine over @ref atl::beats::extremes and @ref atl::beats::clamp_add is for
	 * anyone also needing `apply`, `fold` or another generic member on that algebra.
	 *
	 * Iterative over the shared `detail::segment_tree_base`: pushes along the
	 * boundary paths, the canonical decomposition, and a recursive descent below
	 * the canonical nodes whose mapping fails.
	 *
	 * @tparam Value   The summary type.
	 * @tparam Tag     The tag type.
	 * @tparam Op      Summary fold: any `op(Value, Value)` convertible to `Value`.
	 * @tparam Mapping Fallible in-place tag application `bool(Tag const &, Value &)`.
	 * @tparam Compose Tag composition: any `compose(Tag, Tag)` convertible to `Tag`.
	 */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op = beats::merge, fallible_action<Tag, Value> Mapping = beats::mapping,
			 binary_operation<Tag> Compose = beats::compose>
	class segment_tree_beats : detail::segment_tree_base<Value> {
		using base = detail::segment_tree_base<Value>;
		using base::ascend_boundaries;
		using base::build;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::recompute_up;
		using base::size_;

		/** @brief The element type of a library tag, for the named updates. */
		using tag_element = typename detail::beats_element<Tag>::type;

		/** @brief The element type of a library summary, for the element-level members. */
		using value_element = typename detail::beats_element<Value>::type;

		[[no_unique_address]] Op op_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		Value element_identity_;
		Tag tag_identity_;
		std::vector<Tag> pending_tags_; /**< pending tag per internal node: applied to the node
											 itself already, still owed to its subtree */

		/** @brief Set the geometry for @p n elements, all the identity. @pre `n < 2^32` */
		constexpr void reset(std::size_t n) {
			atl_assert(n < (atl::uint64(1) << 32) - 1);
			base::reset(n, element_identity_);
			pending_tags_.assign(size_, tag_identity_);
		}

		/**
		 * @brief Whether @p x may sit at a leaf: for a library summary, exactly the summary
		 *        of one element (so a hand-built `extremes` with a wrong count or width is
		 *        rejected); anything for a user type. For assertions.
		 */
		[[nodiscard]] static constexpr bool is_leaf_summary(Value const &x) {
			if constexpr (beats::summary<Value> && std::equality_comparable<Value> && requires { x.max; })
				return x == Value::of(x.max);
			else
				return true;
		}

		/** @brief Recompute node @p k from its children. */
		constexpr void pull(std::size_t k) { nodes_[k] = std::invoke(std::as_const(op_), nodes_[2 * k], nodes_[2 * k + 1]); }

		/**
		 * @brief Settle node @p k's pending tag onto its two children; an identity tag
		 * is skipped when `Tag` is equality-comparable (the lazy tree's measured skip).
		 * A failing child mapping breaks the push law and is asserted.
		 */
		constexpr void push_down(std::size_t k) {
			if constexpr (std::equality_comparable<Tag>)
				if (pending_tags_[k] == tag_identity_) return;
			settle(2 * k, pending_tags_[k]);
			settle(2 * k + 1, pending_tags_[k]);
			pending_tags_[k] = tag_identity_;
		}

		/** @brief Apply a pushed tag @p f to child node @p c, which must absorb it. */
		constexpr void settle(std::size_t c, Tag const &f) {
			bool const absorbed = std::invoke(std::as_const(mapping_), f, nodes_[c]);
			atl_assert(absorbed);
			if (c < size_) pending_tags_[c] = std::invoke(std::as_const(compose_), f, pending_tags_[c]);
		}

		/** @brief Settle every pending tag above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief Apply @p f to the subtree of node @p k, descending wherever the mapping fails. */
		constexpr void beat(std::size_t k, Tag const &f) {
			if (std::invoke(std::as_const(mapping_), f, nodes_[k])) {
				if (k < size_) pending_tags_[k] = std::invoke(std::as_const(compose_), f, pending_tags_[k]);
				return;
			}
			atl_assert(k < size_); // a single element never fails
			push_down(k);
			beat(2 * k, f);
			beat(2 * k + 1, f);
			pull(k);
		}

		/** @brief Apply the tag @p rule derives to the subtree of node @p k, descending on `nullopt`. */
		template<beats_tag_rule<Value, Tag> Rule>
		constexpr void beat_derived(std::size_t k, Rule const &rule) {
			if (std::optional<Tag> const f = std::invoke(rule, nodes_[k])) {
				bool const absorbed = std::invoke(std::as_const(mapping_), *f, nodes_[k]);
				atl_assert(absorbed);
				if (k < size_) pending_tags_[k] = std::invoke(std::as_const(compose_), *f, pending_tags_[k]);
				return;
			}
			atl_assert(k < size_); // the rule must accept a single element
			push_down(k);
			beat_derived(2 * k, rule);
			beat_derived(2 * k + 1, rule);
			pull(k);
		}

		/**
		 * @brief Push the boundary paths of `[l, r)`, then visit its canonical nodes
		 *        with @p visit (any order: every caller folds commutatively or writes).
		 */
		template<std::invocable<std::size_t> Visit>
		constexpr void visit_canonical(std::size_t l, std::size_t r, Visit visit) {
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, visit, visit);
		}

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = Value;
		/** @brief Range action type. */
		using tag_type = Tag;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/**
		 * @brief An empty tree with value-initialized identities and callables — a
		 *        placeholder for a member or global assigned a real tree later.
		 * @complexity O(1)
		 */
		constexpr segment_tree_beats()
			requires std::default_initializable<Value> && std::default_initializable<Tag> && std::default_initializable<Op>
				  && std::default_initializable<Mapping> && std::default_initializable<Compose>
			: segment_tree_beats(0, Value{}, Tag{}) {}

		/** @brief @p n elements, all equal to @p identity. @pre `n < 2^32 - 1` @complexity O(n) */
		constexpr segment_tree_beats(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{},
									 Compose compose = Compose{})
			: op_(std::move(op)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  element_identity_(std::move(identity)),
			  tag_identity_(std::move(tag_identity)) {
			reset(n);
		}

		/**
		 * @brief From initial summaries. Elements are moved out only of an rvalue range
		 * that owns them, copied otherwise (`detail::forward_element`). `Value` is
		 * non-deduced here so class deduction goes through the guide below, which
		 * rejects narrowing. @pre `size(values) < 2^32 - 1` @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr segment_tree_beats(R &&values, std::type_identity_t<Value> identity, Tag tag_identity, Op op = Op{},
									 Mapping mapping = Mapping{}, Compose compose = Compose{})
			: op_(std::move(op)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  element_identity_(std::move(identity)),
			  tag_identity_(std::move(tag_identity)) {
			reset(std::ranges::size(values));
			std::size_t i = 0;
			for (auto &&v : values) {
				nodes_[leaf(i)] = static_cast<Value>(detail::forward_element<R>(v));
				atl_assert(is_leaf_summary(nodes_[leaf(i)]));
				++i;
			}
			build(op_);
		}

		/**
		 * @brief Library types: @p n elements equal to `element_type{}` (zero), the
		 *        identities from the types. @pre `n < 2^32 - 1` @complexity O(n)
		 */
		constexpr explicit segment_tree_beats(std::size_t n)
			requires beats::summary<Value> && beats::tag<Tag> && std::default_initializable<Op> && std::default_initializable<Mapping>
					  && std::default_initializable<Compose>
			: element_identity_(Value::identity()), tag_identity_(Tag::identity()) {
			reset(n);
			Value const zero = Value::of(value_element{});
			for (std::size_t i = 0; i < n; ++i) nodes_[leaf(i)] = zero;
			build(op_);
		}

		/**
		 * @brief Library types: from initial elements (not summaries), the identities
		 *        from the types. @pre `size(elements) < 2^32 - 1`, and each element
		 *        valid for `Value::of` @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires beats::summary<Value> && beats::tag<Tag> && std::default_initializable<Op> && std::default_initializable<Mapping>
					  && std::default_initializable<Compose> && requires { static_cast<value_element>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr explicit segment_tree_beats(R &&elements) : element_identity_(Value::identity()), tag_identity_(Tag::identity()) {
			reset(std::ranges::size(elements));
			std::size_t i = 0;
			for (auto &&x : elements) nodes_[leaf(i++)] = Value::of(static_cast<value_element>(detail::forward_element<R>(x)));
			build(op_);
		}

		/**
		 * @brief Assign the summary @p x to position @p i. @pre `i < size()`; for a library
		 * summary, `x` is one element's summary (asserted) @complexity O(log n)
		 */
		constexpr void set(std::size_t i, Value x) {
			atl_assert(i < n_ && is_leaf_summary(x));
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = std::move(x);
			recompute_up(k, op_);
		}

		/** @brief Library summaries: assign the element @p x to position @p i. @pre `i < size()` @complexity O(log n) */
		constexpr void set(std::size_t i, value_element x)
			requires beats::summary<Value>
		{
			set(i, Value::of(x));
		}

		/**
		 * @brief The summary at position @p i (a reference into the tree, valid until
		 * the next mutation). Settles the tags above `i`, hence not const.
		 * @pre `i < size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value const &at(std::size_t i) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			return nodes_[k];
		}

		/** @brief Library summaries: the element at position @p i. @pre `i < size()` @complexity O(log n) */
		[[nodiscard]] constexpr auto element(std::size_t i)
			requires beats::summary<Value> && requires(Value const &v) { v.max; }
		{
			return at(i).max;
		}

		/** @brief Apply tag @p f to the single element at @p i. @pre `i < size()` @complexity O(log n) */
		constexpr void apply(std::size_t i, Tag const &f) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			bool const absorbed = std::invoke(std::as_const(mapping_), f, nodes_[k]);
			atl_assert(absorbed); // a single element never fails
			recompute_up(k, op_);
		}

		/**
		 * @brief Apply tag @p f to every element of `[l, r)`. @pre `l <= r <= size()`
		 * @complexity O(log n) plus the descents below failing nodes, amortized by the
		 * algebra's potential (class docs)
		 */
		constexpr void apply(std::size_t l, std::size_t r, Tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, [&](std::size_t k) { beat(k, f); }, [&](std::size_t k) { beat(k, f); });
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

		/**
		 * @brief A node-dependent update of `[l, r)`: each canonical node, and below
		 * it each node the rule declines, gets the tag `rule(summary)` derives for it
		 * (`std::nullopt` descends). The rule must accept a single element, and a
		 * derived tag must be one the mapping absorbs (both asserted).
		 * @pre `l <= r <= size()` @complexity as `apply`
		 */
		template<beats_tag_rule<Value, Tag> Rule>
		constexpr void apply_derived(std::size_t l, std::size_t r, Rule const &rule) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, [&](std::size_t k) { beat_derived(k, rule); }, [&](std::size_t k) { beat_derived(k, rule); });
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

		/**
		 * @brief Fold of `[l, r)`, or the identity if empty. Settles boundary tags,
		 * hence not const. @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value fold(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return element_identity_;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			Value sl = element_identity_, sr = element_identity_;
			decompose(l, r, [&](std::size_t k) { sl = std::invoke(std::as_const(op_), sl, nodes_[k]); }, [&](std::size_t k) { sr = std::invoke(std::as_const(op_), nodes_[k], sr); });
			return std::invoke(std::as_const(op_), sl, sr);
		}

		/** @brief Fold of the whole range; the root is always current. @complexity O(1) */
		[[nodiscard]] constexpr Value const &fold() const { return nodes_[1]; }

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires
		 * `pred` monotone in `r` and `pred(identity)`. @pre `l <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<Value const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) {
			atl_assert(l <= n_ && std::invoke(pred, element_identity_));
			if (l == n_) return n_;
			l += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down(l >> i);
			Value sm = element_identity_;
			do {
				while (l % 2 == 0) l >>= 1;
				if (!std::invoke(pred, std::invoke(std::as_const(op_), sm, nodes_[l]))) {
					while (l < size_) {
						push_down(l);
						l *= 2;
						if (Value t = std::invoke(std::as_const(op_), sm, nodes_[l]); std::invoke(pred, t)) sm = std::move(t), ++l;
					}
					return l - size_;
				}
				sm = std::invoke(std::as_const(op_), sm, nodes_[l]);
				++l;
			} while (!std::has_single_bit(l));
			return n_;
		}

		/**
		 * @brief Smallest `l` in `[0, r]` with `pred(fold(l, r))` true. Requires `pred`
		 * monotone in `l` and `pred(identity)`. @pre `r <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<Value const &> Pred>
		[[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred) {
			atl_assert(r <= n_ && std::invoke(pred, element_identity_));
			if (r == 0) return 0;
			r += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down((r - 1) >> i);
			Value sm = element_identity_;
			do {
				--r;
				while (r > 1 && (r & 1)) r >>= 1;
				if (!std::invoke(pred, std::invoke(std::as_const(op_), nodes_[r], sm))) {
					while (r < size_) {
						push_down(r);
						r = 2 * r + 1;
						if (Value t = std::invoke(std::as_const(op_), nodes_[r], sm); std::invoke(pred, t)) sm = std::move(t), --r;
					}
					return r + 1 - size_;
				}
				sm = std::invoke(std::as_const(op_), nodes_[r], sm);
			} while (!std::has_single_bit(r));
			return 0;
		}

		/**
		 * @brief Clamp-add tag: `a[i] = min(a[i], x)` on `[l, r)`.
		 * @pre `l <= r <= size()`, `x` strictly inside `Element`'s range
		 * @complexity O(log n) amortized; O(log² n) amortized once adds are mixed in
		 */
		constexpr void chmin(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_clamp_add<Tag>
		{
			atl_assert((std::numeric_limits<tag_element>::lowest)() < x && x < (std::numeric_limits<tag_element>::max)());
			apply(l, r, Tag{0, (std::numeric_limits<tag_element>::lowest)(), x});
		}

		/**
		 * @brief Clamp-add tag: `a[i] = max(a[i], x)` on `[l, r)`.
		 * @pre `l <= r <= size()`, `x` strictly inside `Element`'s range
		 * @complexity as `chmin`
		 */
		constexpr void chmax(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_clamp_add<Tag>
		{
			atl_assert((std::numeric_limits<tag_element>::lowest)() < x && x < (std::numeric_limits<tag_element>::max)());
			apply(l, r, Tag{0, x, (std::numeric_limits<tag_element>::max)()});
		}

		/**
		 * @brief Clamp-add or add tag: `a[i] += x` on `[l, r)`.
		 * @pre `l <= r <= size()`; the results stay inside the summary's bounds
		 * @complexity O(log n)
		 */
		constexpr void add(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_clamp_add<Tag> || detail::is_beats_add<Tag>
		{
			if constexpr (detail::is_beats_clamp_add<Tag>)
				apply(l, r, Tag{x, (std::numeric_limits<tag_element>::lowest)(), (std::numeric_limits<tag_element>::max)()});
			else
				apply(l, r, Tag{x});
		}

		/**
		 * @brief Clamp-add or assign tag: `a[i] = x` on `[l, r)` (a clamp to `[x, x]`,
		 * which never descends). @pre `l <= r <= size()`; for clamp-add, `x` strictly
		 * inside `Element`'s range @complexity O(log n)
		 */
		constexpr void assign(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_clamp_add<Tag> || detail::is_beats_assign<Tag>
		{
			if constexpr (detail::is_beats_clamp_add<Tag>) {
				atl_assert((std::numeric_limits<tag_element>::lowest)() < x && x < (std::numeric_limits<tag_element>::max)());
				apply(l, r, Tag{0, x, x});
			} else {
				apply(l, r, Tag{x, true});
			}
		}

		/**
		 * @brief Modulo summary + assign tag: `a[i] = a[i] % m` on `[l, r)`. A node whose
		 * maximum is below `m` is left alone, one holding a single value is assigned.
		 * @pre `l <= r <= size()`, `m >= 1`, every `a[i]` in `[l, r)` nonnegative
		 * (asserted on the nodes visited, which cover the range)
		 * @complexity O(log n · log V) amortized, V the largest value
		 */
		constexpr void modulo(std::size_t l, std::size_t r, tag_element m)
			requires detail::is_beats_mod_sum<Value> && detail::is_beats_assign<Tag>
		{
			atl_assert(m >= 1);
			apply_derived(l, r, [m](Value const &s) -> std::optional<Tag> {
				atl_assert(s.width == 0 || s.min >= 0);
				if (s.max < m) return Tag::identity();
				if (s.max == s.min) return Tag{tag_element(s.max % m), true};
				return std::nullopt;
			});
		}

		/**
		 * @brief Sqrt summary + add tag: `a[i] = floor(sqrt(a[i]))` on `[l, r)`. A node
		 * whose maximum and minimum drop by the same amount takes that drop as an add.
		 * @pre `l <= r <= size()`, every `a[i]` in `[l, r)` nonnegative (asserted on the
		 * nodes visited) @complexity O(log n · log log V) amortized, V the largest value
		 */
		constexpr void floor_sqrt(std::size_t l, std::size_t r)
			requires detail::is_beats_sqrt_add_sum<Value> && detail::is_beats_add<Tag>
		{
			apply_derived(l, r, [](Value const &s) -> std::optional<Tag> {
				if (s.width == 0) return Tag::identity();
				atl_assert(s.min >= 0);
				tag_element const top = detail::beats_floor_sqrt(s.max);
				tag_element const bottom = s.max == s.min ? top : detail::beats_floor_sqrt(s.min);
				if (s.max - top != s.min - bottom) return std::nullopt;
				return Tag{tag_element(top - s.max)};
			});
		}

		/**
		 * @brief Sqrt summary + add tag: `a[i] = floor(a[i] / d)` on `[l, r)` (rounding
		 * toward minus infinity). @pre `l <= r <= size()`, `d >= 1`
		 * @complexity O(log n · log V) amortized, V the largest absolute value
		 */
		constexpr void divide(std::size_t l, std::size_t r, tag_element d)
			requires detail::is_beats_sqrt_add_sum<Value> && detail::is_beats_add<Tag>
		{
			atl_assert(d >= 1);
			if (d == 1) return;
			apply_derived(l, r, [d](Value const &s) -> std::optional<Tag> {
				if (s.width == 0) return Tag::identity();
				tag_element const top = detail::beats_floor_divide(s.max, d);
				tag_element const bottom = s.max == s.min ? top : detail::beats_floor_divide(s.min, d);
				if (s.max - top != s.min - bottom) return std::nullopt;
				return Tag{tag_element(top - s.max)};
			});
		}

		/**
		 * @brief And-or tag: `a[i] &= x` on `[l, r)`. @pre `l <= r <= size()`
		 * @complexity O(log n · w) amortized for `w`-bit elements
		 */
		constexpr void bit_and(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_bit_mask<Tag>
		{
			apply(l, r, Tag{x, 0});
		}

		/**
		 * @brief And-or tag: `a[i] |= x` on `[l, r)`. @pre `l <= r <= size()`
		 * @complexity O(log n · w) amortized for `w`-bit elements
		 */
		constexpr void bit_or(std::size_t l, std::size_t r, tag_element x)
			requires detail::is_beats_bit_mask<Tag>
		{
			apply(l, r, Tag{tag_element(~tag_element(0)), x});
		}

		/**
		 * @brief Library summaries with a sum: the sum of `[l, r)` (0 if empty), read off
		 * the canonical nodes without folding whole summaries. @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr auto sum(std::size_t l, std::size_t r)
			requires beats::summary<Value> && requires(Value const &v) { v.sum; }
		{
			atl_assert(l <= r && r <= n_);
			decltype(Value::sum) total{};
			if (l == r) return total;
			visit_canonical(l, r, [&](std::size_t k) { total = detail::beats_add(total, nodes_[k].sum); });
			return total;
		}

		/**
		 * @brief Library summaries with a minimum: the minimum of `[l, r)`.
		 * @pre `l < r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr auto(min)(std::size_t l, std::size_t r)
			requires beats::summary<Value> && requires(Value const &v) { v.min; }
		{
			atl_assert(l < r && r <= n_);
			auto best = element_identity_.min;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::min)(best, nodes_[k].min); });
			return best;
		}

		/**
		 * @brief Library summaries with a maximum: the maximum of `[l, r)`.
		 * @pre `l < r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr auto(max)(std::size_t l, std::size_t r)
			requires beats::summary<Value> && requires(Value const &v) { v.max; }
		{
			atl_assert(l < r && r <= n_);
			auto best = element_identity_.max;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::max)(best, nodes_[k].max); });
			return best;
		}

		/** @brief Number of elements. @complexity O(1) */
		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether there are no elements. @complexity O(1) */
		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }
	};

	/** @brief Deduces the algebra from the identities and the callables. */
	template<typename Value, typename Tag, binary_operation<Value> Op, fallible_action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	segment_tree_beats(std::size_t, Value, Tag, Op, Mapping, Compose) -> segment_tree_beats<Value, Tag, Op, Mapping, Compose>;

	/**
	 * @brief Deduces `Value` from the element identity; rejected if that would narrow the
	 * range's values — spell the class arguments explicitly if the conversion is
	 * intended.
	 */
	template<std::ranges::sized_range R, typename Value, typename Tag, binary_operation<Value> Op, fallible_action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
		requires std::convertible_to<std::ranges::range_value_t<R>, Value> && requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	segment_tree_beats(R &&, Value, Tag, Op, Mapping, Compose) -> segment_tree_beats<Value, Tag, Op, Mapping, Compose>;

	/**
	 * @brief The everyday beats tree: range `chmin` / `chmax` / `add` / `assign` and
	 *        range `sum` / `min` / `max`, hand-written for this one algebra.
	 *
	 * Same named members and same answers as @ref atl::segment_tree_beats over
	 * @ref atl::beats::extremes and @ref atl::beats::clamp_add, but only an add is pending per
	 * internal node: the clamps are implied by the node's own extremes, and a push
	 * clamps each child to its parent's maximum and minimum. Against that engine it
	 * measured 18% to 46% faster on every shape of the race, at both sizes, under
	 * `-O2` and `-march=native` (ds-segment_tree_beats, 2026-11-01, Xeon 8280).
	 *
	 * The generic engine members (`apply`, `apply_derived`, `fold`, `at`, the
	 * summary-valued `set`) are not here: for those, spell the engine type,
	 * `segment_tree_beats<beats::extremes<Element, Sum>, beats::clamp_add<Element>>`.
	 *
	 * **Complexity.** O(log² n) amortized per update with `add` in the mix, O(log n)
	 * amortized without. A chmin or chmax descends below a node exactly where the
	 * engine's mapping fails (the bound reaches the second extreme without flattening
	 * the node), so the engine's potential pays for these descents too (the beats potential argument).
	 * Queries and the binary searches are O(log n).
	 *
	 * Bounds (documented; violations are caught by overflow asserts under
	 * `ATL_ASSERT`): let `M` bound every `|a[i]|` at all times plus the total `|x|`
	 * of the `add` calls; `2M` must fit `Element` strictly, and `n · 2M` must fit
	 * `Sum`. A stream keeping `|a[i]| <= A` at all times needs less, however large its
	 * adds sum up to: `3A` strictly inside `Element` and `4nA` inside `Sum`, because a
	 * multi-valued node's net pending add is at most `2A` (its extreme classes move only
	 * by adds and by clamps pointing inward) and a single-valued node's is void. Element
	 * values and clamp arguments lie strictly inside `Element`'s range, whose two ends
	 * are sentinels (@ref atl::beats::extremes).
	 *
	 * @tparam Element The element type (signed).
	 * @tparam Sum     The sum type, at least as wide as `Element`; `Element` by
	 *                 default, `int128` as the opt-in.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
		requires(std::numeric_limits<Sum>::digits >= std::numeric_limits<Element>::digits)
	class beats_segment_tree : detail::segment_tree_base<beats::extremes<Element, Sum>> {
		using node = beats::extremes<Element, Sum>;
		using base = detail::segment_tree_base<node>;
		using base::ascend_boundaries;
		using base::build;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::size_;

		static constexpr Element lowest = (std::numeric_limits<Element>::lowest)(), highest = (std::numeric_limits<Element>::max)();

		/** @brief Pending add per internal node: applied to the node, owed to its subtree. Void on a single-valued node. */
		std::vector<Element> adds_;

		/** @brief Set the geometry for @p n elements, all the identity. @pre `n < 2^32 - 1` */
		constexpr void reset(std::size_t n) {
			atl_assert(n < (atl::uint64(1) << 32) - 1);
			base::reset(n, node::identity());
			adds_.assign(size_, 0);
		}

		/** @brief Recompute node @p k from its children, in place. */
		constexpr void pull(std::size_t k) {
			node &s = nodes_[k];
			node const &a = nodes_[2 * k], &b = nodes_[2 * k + 1];
			if (a.max > b.max) {
				s.max = a.max, s.max_count = a.max_count, s.second_max = (std::max)(a.second_max, b.max);
			} else if (a.max < b.max) {
				s.max = b.max, s.max_count = b.max_count, s.second_max = (std::max)(a.max, b.second_max);
			} else {
				s.max = a.max, s.max_count = a.max_count + b.max_count, s.second_max = (std::max)(a.second_max, b.second_max);
			}
			if (a.min < b.min) {
				s.min = a.min, s.min_count = a.min_count, s.second_min = (std::min)(a.second_min, b.min);
			} else if (a.min > b.min) {
				s.min = b.min, s.min_count = b.min_count, s.second_min = (std::min)(a.min, b.second_min);
			} else {
				s.min = a.min, s.min_count = a.min_count + b.min_count, s.second_min = (std::min)(a.second_min, b.second_min);
			}
			s.sum = detail::beats_add(a.sum, b.sum);
			s.width = a.width + b.width;
		}

		/** @brief Every element of node @p k becomes @p x; its pending add is dropped. */
		constexpr void flatten(std::size_t k, Element x) {
			node &s = nodes_[k];
			s.max = s.min = x;
			s.second_max = lowest, s.second_min = highest;
			s.max_count = s.min_count = s.width;
			s.sum = detail::beats_mul(Sum(x), Sum(s.width));
			if (k < size_) adds_[k] = 0;
		}

		/** @brief Lower node @p k's maximum class to @p x. @pre `second_max < x < max` */
		constexpr void lower_max(std::size_t k, Element x) {
			node &s = nodes_[k];
			// A single-valued child reached by a push can sit far above x after several chmins
			// its parent absorbed one by one, so max - x need not fit Sum even though the sum does.
			if (s.min == s.max) {
				s.sum = detail::beats_mul(Sum(x), Sum(s.width));
				s.min = x;
			} else {
				s.sum = detail::beats_sub(s.sum, detail::beats_mul(detail::beats_sub(Sum(s.max), Sum(x)), Sum(s.max_count)));
				if (s.second_min == s.max) s.second_min = x;
			}
			s.max = x;
		}

		/** @brief Raise node @p k's minimum class to @p x (the mirror of `lower_max`). @pre `min < x < second_min` */
		constexpr void raise_min(std::size_t k, Element x) {
			node &s = nodes_[k];
			if (s.max == s.min) {
				s.sum = detail::beats_mul(Sum(x), Sum(s.width));
				s.max = x;
			} else {
				s.sum = detail::beats_add(s.sum, detail::beats_mul(detail::beats_sub(Sum(x), Sum(s.min)), Sum(s.min_count)));
				if (s.second_max == s.min) s.second_max = x;
			}
			s.min = x;
		}

		/** @brief Add @p x to every element of node @p k, owing it to the subtree. */
		constexpr void shift(std::size_t k, Element x) {
			node &s = nodes_[k];
			if (s.width == 0) return;
			s.max = detail::beats_shift_value(s.max, x);
			s.min = detail::beats_shift_value(s.min, x);
			if (s.second_max != lowest) s.second_max = detail::beats_shift_value(s.second_max, x);
			if (s.second_min != highest) s.second_min = detail::beats_shift_value(s.second_min, x);
			s.sum = detail::beats_add(s.sum, detail::beats_mul(Sum(x), Sum(s.width)));
			if (k < size_) {
				Element result{};
				atl_assert(!detail::add_overflows(adds_[k], x, result) || s.max == s.min);
				using Unsigned = traits::make_unsigned_t<Element>;
				adds_[k] = Element(Unsigned(Unsigned(adds_[k]) + Unsigned(x)));
			}
		}

		/**
		 * @brief Settle node @p k onto its children. Under a single-valued node the
		 *        children simply take its value: its add is void, and their stale values
		 *        plus that add could overflow. Otherwise the add goes down, then each
		 *        child's maximum is lowered to the parent's and its minimum raised to the
		 *        parent's: a parent's extreme moves only within its own class, so only
		 *        a child's extreme class can lie outside.
		 */
		constexpr void push_down(std::size_t k) {
			std::size_t const a = 2 * k, b = 2 * k + 1;
			Element const top = nodes_[k].max, bottom = nodes_[k].min;
			if (top == bottom) {
				for (std::size_t const c : {a, b})
					if (nodes_[c].width != 0 && (nodes_[c].max != top || nodes_[c].min != top)) flatten(c, top);
				adds_[k] = 0;
				return;
			}
			if (adds_[k] != 0) {
				shift(a, adds_[k]);
				shift(b, adds_[k]);
				adds_[k] = 0;
			}
			for (std::size_t const c : {a, b}) {
				if (nodes_[c].max > top) lower_max(c, top);
				if (nodes_[c].min < bottom) raise_min(c, bottom);
			}
		}

		/** @brief Settle everything pending above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief An update of `[l, r)` running @p on_node on the canonical nodes. */
		template<std::invocable<std::size_t> OnNode>
		constexpr void update(std::size_t l, std::size_t r, OnNode on_node) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, on_node, on_node);
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

		/** @brief Push the boundary paths of `[l, r)`, then visit its canonical nodes. */
		template<std::invocable<std::size_t> Visit>
		constexpr void visit_canonical(std::size_t l, std::size_t r, Visit visit) {
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, visit, visit);
		}

		/** @brief Chmin below node @p k, descending where @p x reaches the second maximum without flattening. */
		constexpr void chmin_node(std::size_t k, Element x) {
			if (x >= nodes_[k].max) return;
			if (x <= nodes_[k].min) {
				flatten(k, x);
			} else if (x > nodes_[k].second_max) {
				lower_max(k, x);
			} else {
				push_down(k);
				chmin_node(2 * k, x);
				chmin_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Chmax below node @p k, descending where @p x reaches the second minimum without flattening. */
		constexpr void chmax_node(std::size_t k, Element x) {
			if (x <= nodes_[k].min) return;
			if (x >= nodes_[k].max) {
				flatten(k, x);
			} else if (x < nodes_[k].second_min) {
				raise_min(k, x);
			} else {
				push_down(k);
				chmax_node(2 * k, x);
				chmax_node(2 * k + 1, x);
				pull(k);
			}
		}

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = beats::extremes<Element, Sum>;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/** @brief An empty tree. @complexity O(1) */
		constexpr beats_segment_tree() : beats_segment_tree(std::size_t(0)) {}

		/** @brief @p n zeros. @pre `n < 2^32 - 1` @complexity O(n) */
		constexpr explicit beats_segment_tree(std::size_t n) {
			reset(n);
			node const zero = node::of(Element{});
			for (std::size_t i = 0; i < n; ++i) nodes_[leaf(i)] = zero;
			build(beats::merge{});
		}

		/**
		 * @brief From initial elements. @pre `size(elements) < 2^32 - 1`, and each
		 * element strictly inside `Element`'s range @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr explicit beats_segment_tree(R &&elements) {
			reset(std::ranges::size(elements));
			std::size_t i = 0;
			for (auto &&x : elements) nodes_[leaf(i++)] = node::of(static_cast<Element>(detail::forward_element<R>(x)));
			build(beats::merge{});
		}

		/**
		 * @brief `a[i] = x`. @pre `i < size()`, `x` strictly inside `Element`'s range
		 * @complexity O(log n)
		 */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			std::size_t k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = node::of(x);
			while (k >>= 1) pull(k);
		}

		/**
		 * @brief The element at position @p i. Settles the path above it, hence not const.
		 * @pre `i < size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)].max;
		}

		/**
		 * @brief The summary of position @p i. Settles the path above it, hence not const.
		 * @pre `i < size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr node const &at(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)];
		}

		/**
		 * @brief The summary of `[l, r)`. Settles the boundary paths, hence not const.
		 * @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr node fold(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return node::identity();
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			node sl = node::identity(), sr = node::identity();
			decompose(l, r, [&](std::size_t k) { sl = beats::merge{}(sl, nodes_[k]); }, [&](std::size_t k) { sr = beats::merge{}(nodes_[k], sr); });
			return beats::merge{}(sl, sr);
		}

		/** @brief The summary of the whole range; the root is always current. @complexity O(1) */
		[[nodiscard]] constexpr node const &fold() const { return nodes_[1]; }

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true, the fold being
		 * the @ref atl::beats::extremes summary of `[l, r)`. Requires `pred` monotone in `r`
		 * and `pred(identity)`. @pre `l <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<beats::extremes<Element, Sum> const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) {
			atl_assert(l <= n_ && std::invoke(pred, node::identity()));
			if (l == n_) return n_;
			l += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down(l >> i);
			node sm = node::identity();
			do {
				while (l % 2 == 0) l >>= 1;
				node const whole = beats::merge{}(sm, nodes_[l]);
				if (!std::invoke(pred, whole)) {
					while (l < size_) {
						push_down(l);
						l *= 2;
						if (node const t = beats::merge{}(sm, nodes_[l]); std::invoke(pred, t)) sm = t, ++l;
					}
					return l - size_;
				}
				sm = whole;
				++l;
			} while (!std::has_single_bit(l));
			return n_;
		}

		/**
		 * @brief Smallest `l` in `[0, r]` with `pred(fold(l, r))` true, the fold being the
		 * @ref atl::beats::extremes summary of `[l, r)`. Requires `pred` monotone in `l` and
		 * `pred(identity)`. @pre `r <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<beats::extremes<Element, Sum> const &> Pred>
		[[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred) {
			atl_assert(r <= n_ && std::invoke(pred, node::identity()));
			if (r == 0) return 0;
			r += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down((r - 1) >> i);
			node sm = node::identity();
			do {
				--r;
				while (r > 1 && (r & 1)) r >>= 1;
				node const whole = beats::merge{}(nodes_[r], sm);
				if (!std::invoke(pred, whole)) {
					while (r < size_) {
						push_down(r);
						r = 2 * r + 1;
						if (node const t = beats::merge{}(nodes_[r], sm); std::invoke(pred, t)) sm = t, --r;
					}
					return r + 1 - size_;
				}
				sm = whole;
			} while (!std::has_single_bit(r));
			return 0;
		}

		/**
		 * @brief `a[i] = min(a[i], x)` on `[l, r)`.
		 * @pre `l <= r <= size()`, `x` strictly inside `Element`'s range
		 * @complexity O(log n) amortized; O(log² n) amortized once adds are mixed in
		 */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) {
			atl_assert(lowest < x && x < highest);
			update(l, r, [&](std::size_t k) { chmin_node(k, x); });
		}

		/**
		 * @brief `a[i] = max(a[i], x)` on `[l, r)`.
		 * @pre `l <= r <= size()`, `x` strictly inside `Element`'s range
		 * @complexity as `chmin`
		 */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) {
			atl_assert(lowest < x && x < highest);
			update(l, r, [&](std::size_t k) { chmax_node(k, x); });
		}

		/**
		 * @brief `a[i] += x` on `[l, r)`.
		 * @pre `l <= r <= size()`; the results stay strictly inside `Element`'s range
		 * @complexity O(log n), plus what it recharges for later chmins and chmaxes
		 */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { shift(k, x); });
		}

		/**
		 * @brief `a[i] = x` on `[l, r)`; never descends.
		 * @pre `l <= r <= size()`, `x` strictly inside `Element`'s range
		 * @complexity O(log n)
		 */
		constexpr void assign(std::size_t l, std::size_t r, Element x) {
			atl_assert(lowest < x && x < highest);
			update(l, r, [&](std::size_t k) { flatten(k, x); });
		}

		/** @brief The sum of `[l, r)` (0 if empty). @pre `l <= r <= size()` @complexity O(log n) */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total{};
			if (l == r) return total;
			visit_canonical(l, r, [&](std::size_t k) { total = detail::beats_add(total, nodes_[k].sum); });
			return total;
		}

		/** @brief The minimum of `[l, r)`. @pre `l < r <= size()` @complexity O(log n) */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = highest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::min)(best, nodes_[k].min); });
			return best;
		}

		/** @brief The maximum of `[l, r)`. @pre `l < r <= size()` @complexity O(log n) */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = lowest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::max)(best, nodes_[k].max); });
			return best;
		}

		/** @brief Number of elements. @complexity O(1) */
		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether there are no elements. @complexity O(1) */
		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }
	};
}
