#pragma once

/** @file
 * @brief Floating, string, pointer and composite-key extensions of hash_core.hpp.
 */
#include <atl/types/alias.hpp>
#include <atl/utils/hash_core.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Seeded hash of a raw byte range: each 8-byte block is folded
		 *        into the running state and the state is passed through the
		 *        full splitmix64 bijection before the next block.
		 *
		 * The state therefore depends on the SECRET seed non-linearly at
		 * every step: what a block must be to cancel a change in an earlier
		 * one depends on that state. The design intent is that no fixed pair
		 * of inputs collides for every seed (argued, not proven). Mixing each
		 * block on its own and combining with xor-and-multiply demonstrably
		 * fails this: flipping the top bit of two consecutive mixed blocks
		 * cancels whatever the seed.
		 */
		inline uint64 hash_bytes(void const *data, std::size_t len, uint64 seed) noexcept {
			auto const *p = static_cast<unsigned char const *>(data);
			uint64 h = splitmix64(seed ^ (uint64(len) * 0x9ddfea08eb382d69ULL));
			for (; len >= 8; p += 8, len -= 8) {
				uint64 k;
				std::memcpy(&k, p, 8);
				h = splitmix64(h ^ k);
			}
			if (len) {
				uint64 k = 0;
				std::memcpy(&k, p, len);
				h = splitmix64(h ^ k);
			}
			return h;
		}

		/** @brief Order-dependent combine of two sub-hashes for composite keys. */
		constexpr uint64 hash_combine(uint64 a, uint64 b) noexcept {
			return splitmix64(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
		}

		/**
		 * @brief Element types whose value equality coincides with equality of
		 *        the object bytes, so a contiguous buffer of them is hashed in one
		 *        pass. Integers qualify (no padding bits on any supported target);
		 *        floats (`-0.0 == 0.0`) and `bool` (`vector<bool>` has no `data()`)
		 *        do not.
		 */
		template<typename T>
		concept byte_hashable = hashable_integer<T> && !std::same_as<std::remove_cv_t<T>, bool> && !std::is_volatile_v<T>;
	}

	namespace detail {
		template<std::floating_point T>
		inline constexpr bool has_x87_value_bytes =
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
			std::numeric_limits<T>::radix == 2 && std::numeric_limits<T>::max_exponent == 16384 && std::numeric_limits<T>::digits == 64 && (sizeof(T) == 12 || sizeof(T) == 16);
#else
			false;
#endif

		template<std::floating_point T>
		inline constexpr bool has_binary128_value_bytes =
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__x86_64__) || defined(__i386__) || defined(__powerpc__) || defined(__s390x__) || defined(__riscv)
			std::numeric_limits<T>::radix == 2 && std::numeric_limits<T>::max_exponent == 16384 && std::numeric_limits<T>::digits == 113 && sizeof(T) == 16;
#else
			false;
#endif

		template<std::floating_point T>
		inline constexpr bool uses_float_value_fallback = sizeof(T) > 8 &&
#ifdef ATL_FORCE_FALLBACKS
														  true;
#else
														  !has_x87_value_bytes<T> && !has_binary128_value_bytes<T>;
#endif

		// Arithmetic decomposition avoids depending on any unknown object layout.
		// A fixed number of binary digits also bounds work on non-binary formats;
		// collisions there are allowed, and equal values still decompose identically.
		template<std::floating_point T>
		uint64 hash_float_value(T x, uint64 seed) noexcept {
			if (x == T(0)) return splitmix64(seed);
			if (std::isnan(x)) return splitmix64(seed ^ 0x7ff8000000000000ULL);
			if (std::isinf(x)) return splitmix64(seed ^ (std::signbit(x) ? 0xfff0000000000000ULL : 0x7ff0000000000000ULL));
			int exponent = 0;
			T fraction = std::frexp(std::fabs(x), &exponent);
			uint64 h = hash_combine(seed, uint64(exponent));
			h = hash_combine(h, uint64(std::signbit(x)));
			for (int i = 0; i < std::numeric_limits<T>::digits; ++i) {
				fraction *= T(2);
				bool const bit = fraction >= T(1);
				if (bit) fraction -= T(1);
				h = hash_combine(h, uint64(bit));
			}
			return h;
		}
	}

	/**
	 * @brief Floating point with signed zeros hashed alike and NaNs canonicalized.
	 * Known x87 layouts hash only their ten value bytes; binary128 uses all
	 * sixteen. Unknown wide layouts use value decomposition, never padding.
	 */
	template<std::floating_point T>
	struct hash<T> {
		/**
		 * @brief Hash a floating-point value, normalizing signed zeros and NaNs.
		 * @param x Floating-point key; padding bytes are excluded.
		 * @return Seeded value hash reduced to `std::size_t`.
		 * @complexity O(1) for known layouts; O(std::numeric_limits<T>::digits) for value decomposition.
		 */
		std::size_t operator()(T x) const noexcept {
			T const v = x == T(0) ? T(0) : x;
			if (std::isnan(v)) return static_cast<std::size_t>(detail::splitmix64(detail::hash_seed() ^ 0x7ff8000000000000ULL));
			if constexpr (sizeof(T) <= 8) {
				uint64 bits = 0;
				std::memcpy(&bits, &v, sizeof(T));
				return static_cast<std::size_t>(detail::splitmix64(bits + detail::hash_seed()));
			} else if constexpr (detail::uses_float_value_fallback<T>) {
				return static_cast<std::size_t>(detail::hash_float_value(v, detail::hash_seed()));
			} else if constexpr (detail::has_x87_value_bytes<T>) {
				return static_cast<std::size_t>(detail::hash_bytes(&v, 10, detail::hash_seed()));
			} else {
				return static_cast<std::size_t>(detail::hash_bytes(&v, 16, detail::hash_seed()));
			}
		}
	};

	/**
	 * @brief String hash, transparent: called with a `std::string`, a
	 *        `std::string_view` or a `char const *` (a NUL-terminated string) of
	 *        equal contents it returns the same hash (every argument converts to
	 *        one `string_view` path), so a table keyed on `std::string` can be
	 *        probed with either of the others without building a temporary.
	 *        `hash<char const *>` itself is the pointer hash: it hashes the
	 *        address, not the contents.
	 */
	template<>
	struct hash<std::string_view> {
		/** @brief Enable heterogeneous lookup through implicit `std::string_view` conversion. */
		using is_transparent = void;

		/**
		 * @brief Hash string contents, including embedded NUL bytes.
		 * @param s Characters to hash; implicit C-string conversion requires NUL termination.
		 * @return The same seeded hash for equal character sequences.
		 * @complexity O(s.size()).
		 */
		std::size_t operator()(std::string_view s) const noexcept {
			return static_cast<std::size_t>(detail::hash_bytes(s.data(), s.size(), detail::hash_seed()));
		}
	};

	/** @brief As `hash<std::string_view>`: equal contents give equal hashes. */
	template<>
	struct hash<std::string> {
		/** @brief Enable heterogeneous lookup through implicit `std::string_view` conversion. */
		using is_transparent = void;

		/**
		 * @brief Hash a string or string view by its character sequence.
		 * @param s Characters to hash; implicit C-string conversion requires NUL termination.
		 * @return The same hash as `hash<std::string_view>` for these contents.
		 * @complexity O(s.size()).
		 */
		std::size_t operator()(std::string_view s) const noexcept { return hash<std::string_view>{}(s); }
	};

	/** @brief Pointers, by address (never by the pointee). */
	template<typename T>
	struct hash<T *> {
		/**
		 * @brief Hash a pointer address without accessing the pointee.
		 * @param p Pointer value, including null.
		 * @return Seeded address hash reduced to `std::size_t`.
		 * @complexity O(1).
		 */
		std::size_t operator()(T *p) const noexcept {
			return static_cast<std::size_t>(detail::splitmix64(uint64(reinterpret_cast<std::uintptr_t>(p)) + detail::hash_seed()));
		}
	};

	namespace detail {
		template<typename Key, typename Argument = Key const &>
		concept nothrow_hashable = requires(Argument value) {
			{ hash<Key>{}(value) } -> std::convertible_to<uint64>;
			{ hash_combine(0, hash<Key>{}(value)) } noexcept;
		};

		template<typename A, typename B>
		concept packable_hash_pair = std::integral<A> && std::integral<B>
								  && !std::same_as<std::remove_cv_t<A>, bool> && !std::same_as<std::remove_cv_t<B>, bool>
								  && sizeof(A) + sizeof(B) <= sizeof(uint64);

		template<typename T>
		using tuple_hash_argument_t = std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<T>>>;
		template<typename T>
		using tuple_hash_key_t = std::decay_t<tuple_hash_argument_t<T>>;
	}

	/**
	 * @brief Pairs, order-dependent: two non-bool integers that fit one 64-bit
	 *        word together are packed and mixed in one round; anything else
	 *        combines `hash<A>` and `hash<B>`, which must exist.
	 */
	template<typename A, typename B>
		requires(detail::packable_hash_pair<A, B> || (detail::nothrow_hashable<A> && detail::nothrow_hashable<B>))
	struct hash<std::pair<A, B>> {
		/**
		 * @brief Hash the ordered pair, packing small integer elements when possible.
		 * @param p Pair whose first and second elements participate in order.
		 * @return Seeded pair hash.
		 * @complexity O(1) for packed integers; otherwise the sum of the element hash costs.
		 */
		std::size_t operator()(std::pair<A, B> const &p) const noexcept {
			if constexpr (detail::packable_hash_pair<A, B>) {
				// both keys fit one word: pack them (injectively) and mix once, as
				// an integer key is, instead of three rounds (measured 21-29%
				// faster flat_hash_set inserts and lookups, benchmarks/ds/hash.cpp)
				using UA = std::make_unsigned_t<A>;
				using UB = std::make_unsigned_t<B>;
				uint64 const packed = (uint64(UA(p.first)) << (8 * sizeof(B))) | uint64(UB(p.second));
				return static_cast<std::size_t>(detail::splitmix64(packed + detail::hash_seed()));
			} else {
				return static_cast<std::size_t>(detail::hash_combine(hash<A>{}(p.first), hash<B>{}(p.second)));
			}
		}
	};

	/** @brief Tuples, order-dependent, combining `hash<T>` of each decayed element type. */
	template<typename... Ts>
		requires(detail::nothrow_hashable<detail::tuple_hash_key_t<Ts>, detail::tuple_hash_argument_t<Ts>> && ...)
	struct hash<std::tuple<Ts...>> {
		/**
		 * @brief Fold tuple element hashes in their declared order.
		 * @param t Tuple to hash using decayed element hashers.
		 * @return Seeded tuple hash, including for the empty tuple.
		 * @complexity O(sizeof...(Ts)) combines plus the element hash costs.
		 */
		std::size_t operator()(std::tuple<Ts...> const &t) const noexcept {
			uint64 h = detail::hash_seed();
			std::apply([&](auto const &...xs) { ((h = detail::hash_combine(h, hash<std::decay_t<decltype(xs)>>{}(xs))), ...); },
					   t);
			return static_cast<std::size_t>(h);
		}
	};

	/**
	 * @brief Vectors, length included: a buffer of integers other than `bool`
	 *        as bytes in one pass, anything else folding `hash<T>` per element
	 *        (which respects value equality, e.g. `-0.0 == 0.0`). O(size).
	 */
	template<typename T>
		requires(detail::byte_hashable<T> || detail::nothrow_hashable<T>)
	struct hash<std::vector<T>> {
		/**
		 * @brief Hash a vector by length and ordered element values.
		 * @param v Sequence to hash; integer buffers use a byte pass.
		 * @return Seeded sequence hash respecting element equality.
		 * @complexity O(v.size()) combines plus element hash costs, or O(v.size() * sizeof(T)) bytes.
		 */
		std::size_t operator()(std::vector<T> const &v) const noexcept {
			if constexpr (detail::byte_hashable<T>) {
				return static_cast<std::size_t>(detail::hash_bytes(v.data(), v.size() * sizeof(T), detail::hash_seed()));
			} else {
				uint64 h = detail::hash_seed() ^ (v.size() * 0x9ddfea08eb382d69ULL);
				for (T const &e : v) h = detail::hash_combine(h, hash<T>{}(e));
				return static_cast<std::size_t>(h);
			}
		}
	};

	/** @brief Arrays, as the vector hash with the length `N`. O(N). */
	template<typename T, std::size_t N>
		requires(detail::byte_hashable<T> || detail::nothrow_hashable<T>)
	struct hash<std::array<T, N>> {
		/**
		 * @brief Hash an array by length and ordered element values.
		 * @param a Array to hash; integer buffers use a byte pass.
		 * @return The vector hash for the same sequence of values.
		 * @complexity O(N) combines plus element hash costs, or O(N * sizeof(T)) bytes.
		 */
		std::size_t operator()(std::array<T, N> const &a) const noexcept {
			if constexpr (detail::byte_hashable<T>) {
				return static_cast<std::size_t>(detail::hash_bytes(a.data(), N * sizeof(T), detail::hash_seed()));
			} else {
				uint64 h = detail::hash_seed() ^ (N * 0x9ddfea08eb382d69ULL);
				for (T const &e : a) h = detail::hash_combine(h, hash<T>{}(e));
				return static_cast<std::size_t>(h);
			}
		}
	};
}
