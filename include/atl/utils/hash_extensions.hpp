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
		template<typename T>
		concept byte_hashable = hash_buffer_element<T>;
		template<typename Key, typename Argument = Key const &>
		concept nothrow_hashable = hash_element<Key, Argument>;
		template<typename A, typename B>
		concept packable_hash_pair = hash_packed_elements<A, B>;
		template<typename T>
		using tuple_hash_argument_t = hash_const_argument_t<T>;
		template<typename T>
		using tuple_hash_key_t = hash_decayed_key_t<T>;
	}

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

	template<std::floating_point T>
	struct detail::hash_extension<T, void> {
		static std::size_t call(T x) noexcept {
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

	template<typename T>
		requires std::same_as<T, std::string_view>
	struct detail::hash_extension<T, void> {
		static std::size_t call(std::string_view s) noexcept {
			return static_cast<std::size_t>(detail::hash_bytes(s.data(), s.size(), detail::hash_seed()));
		}
	};

	template<typename T>
		requires std::same_as<T, std::string>
	struct detail::hash_extension<T, void> {
		static std::size_t call(std::string_view s) noexcept { return hash<std::string_view>{}(s); }
	};

	template<typename T>
	struct detail::hash_extension<T *, void> {
		static std::size_t call(T *p) noexcept {
			return static_cast<std::size_t>(detail::splitmix64(uint64(reinterpret_cast<std::uintptr_t>(p)) + detail::hash_seed()));
		}
	};

	template<typename A, typename B>
		requires(detail::hash_packed_elements<A, B> || (detail::hash_element<A> && detail::hash_element<B>))
	struct detail::hash_extension<std::pair<A, B>, void> {
		static std::size_t call(std::pair<A, B> const &p) noexcept {
			if constexpr (detail::hash_packed_elements<A, B>) {
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

	template<typename... Ts>
		requires(detail::hash_element<detail::hash_decayed_key_t<Ts>, detail::hash_const_argument_t<Ts>> && ...)
	struct detail::hash_extension<std::tuple<Ts...>, void> {
		static std::size_t call(std::tuple<Ts...> const &t) noexcept {
			uint64 h = detail::hash_seed();
			std::apply([&](auto const &...xs) { ((h = detail::hash_combine(h, hash<std::decay_t<decltype(xs)>>{}(xs))), ...); },
					   t);
			return static_cast<std::size_t>(h);
		}
	};

	template<typename T>
		requires(detail::hash_buffer_element<T> || detail::hash_element<T>)
	struct detail::hash_extension<std::vector<T>, void> {
		static std::size_t call(std::vector<T> const &v) noexcept {
			if constexpr (detail::hash_buffer_element<T>) {
				return static_cast<std::size_t>(detail::hash_bytes(v.data(), v.size() * sizeof(T), detail::hash_seed()));
			} else {
				uint64 h = detail::hash_seed() ^ (v.size() * 0x9ddfea08eb382d69ULL);
				for (T const &e : v) h = detail::hash_combine(h, hash<T>{}(e));
				return static_cast<std::size_t>(h);
			}
		}
	};

	template<typename T, std::size_t N>
		requires(detail::hash_buffer_element<T> || detail::hash_element<T>)
	struct detail::hash_extension<std::array<T, N>, void> {
		static std::size_t call(std::array<T, N> const &a) noexcept {
			if constexpr (detail::hash_buffer_element<T>) {
				return static_cast<std::size_t>(detail::hash_bytes(a.data(), N * sizeof(T), detail::hash_seed()));
			} else {
				uint64 h = detail::hash_seed() ^ (N * 0x9ddfea08eb382d69ULL);
				for (T const &e : a) h = detail::hash_combine(h, hash<T>{}(e));
				return static_cast<std::size_t>(h);
			}
		}
	};

}
