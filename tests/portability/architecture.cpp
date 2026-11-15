#include <cstdio>

#if defined(_M_ARM64) || defined(__aarch64__)
#define ATL_DETECTED_arm64 1
#else
#define ATL_DETECTED_arm64 0
#endif
#if defined(_M_IX86) || defined(__i386__)
#define ATL_DETECTED_x86 1
#else
#define ATL_DETECTED_x86 0
#endif
#if defined(_M_X64) || defined(__x86_64__)
#define ATL_DETECTED_x64 1
#else
#define ATL_DETECTED_x64 0
#endif

#if defined(ATL_EXPECT_arm64)
static_assert(ATL_DETECTED_arm64 && !ATL_DETECTED_x64 && !ATL_DETECTED_x86);
#elif defined(ATL_EXPECT_x86)
static_assert(ATL_DETECTED_x86 && !ATL_DETECTED_x64 && !ATL_DETECTED_arm64);
#elif defined(ATL_EXPECT_x64)
static_assert(ATL_DETECTED_x64 && !ATL_DETECTED_x86 && !ATL_DETECTED_arm64);
#endif

int main() {
	std::printf("architecture macros: _M_ARM64/__aarch64__=%d _M_IX86/__i386__=%d _M_X64/__x86_64__=%d\n",
				ATL_DETECTED_arm64, ATL_DETECTED_x86, ATL_DETECTED_x64);
	return 0;
}
