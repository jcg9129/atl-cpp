#pragma once

/**
 * @file
 * @brief Configuration macros. Pulls in `<version>` for the `__cpp_lib_*`
 *        feature-test macros that guard optional C++ features behind portable
 *        fallbacks.
 */
#include <version>

/**
 * @brief 1 when a native 128-bit integer (`__int128`) is available, else 0.
 *
 * `__int128` is a GCC/Clang extension present on 64-bit targets but absent on
 * MSVC and many 32-bit/embedded toolchains. When 0, the `int128`/`uint128`
 * aliases and their traits are not defined and the operations that need a
 * 128-bit intermediate use portable software fallbacks. Define `ATL_NO_INT128`
 * to force that path on a 128-bit-capable compiler. `ATL_FORCE_FALLBACKS`
 * selects every portable fallback, including this one.
 */
#if defined(__SIZEOF_INT128__) && !defined(ATL_NO_INT128) && !defined(ATL_FORCE_FALLBACKS)
#define ATL_HAS_INT128 1
#else
#define ATL_HAS_INT128 0
#endif
