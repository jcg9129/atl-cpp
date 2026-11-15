#pragma once
#include <atl/config.hpp>
#include <type_traits>
#include <utility>

namespace atl {
	// std::forward_like ships from libstdc++ 13 / libc++ 16. Older toolchains
	// that accept -std=c++23 lack it, and since the name is looked up when a
	// class template is parsed, merely mentioning it in a header is a
	// parse-time error there, before any instantiation. atl::forward_like is
	// the spelling every atl header uses instead: std's when available, an
	// identical cast otherwise. Define ATL_NO_STD_FORWARD_LIKE to force the
	// fallback on a conforming compiler.
#if defined(__cpp_lib_forward_like) && !defined(ATL_NO_STD_FORWARD_LIKE) && !defined(ATL_FORCE_FALLBACKS)
	using std::forward_like;

	namespace detail {
		inline constexpr bool uses_forward_like_fallback = false;
	}
#else
	namespace detail {
		inline constexpr bool uses_forward_like_fallback = true;
	}

	/**
	 * @brief Casts @p x to the value category and constness `T&&` would impose
	 *        on a member of an object of type `T`: the C++23 std::forward_like
	 *        semantics: const is added when `T` is const, and @p x keeps its own const.
	 */
	template<typename T, typename U>
	[[nodiscard]] constexpr auto &&forward_like(U &&x) noexcept {
		constexpr bool adds_const = std::is_const_v<std::remove_reference_t<T>>;
		if constexpr (std::is_lvalue_reference_v<T &&>) {
			if constexpr (adds_const)
				return std::as_const(x);
			else
				return static_cast<U &>(x);
		} else {
			if constexpr (adds_const)
				return std::move(std::as_const(x));
			else
				return std::move(x);
		}
	}
#endif
}
