#pragma once
#include <atl/config.hpp>

namespace atl {
	namespace detail {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ATL_FORCE_FALLBACKS)
		inline constexpr bool uses_prefetch_fallback = false;
#else
		inline constexpr bool uses_prefetch_fallback = true;
#endif
	}

	/**
	 * @brief A cache hint for the line holding @p p, to be read (default) or
	 *        written soon. Never faults, even on an invalid address.
	 *
	 * A no-op under constant evaluation and on compilers without the GNU
	 * `__builtin_prefetch`, or with `ATL_FORCE_FALLBACKS`.
	 *
	 * @tparam IsWrite Whether the line is about to be written, so the hint
	 *         requests it in an exclusive state.
	 */
	template<bool IsWrite = false>
	constexpr void prefetch(void const *const p) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ATL_FORCE_FALLBACKS)
		if !consteval {
			__builtin_prefetch(p, IsWrite ? 1 : 0);
		}
#else
		static_cast<void>(p);
#endif
	}
}
