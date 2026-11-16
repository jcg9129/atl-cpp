#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
// Other backends also build on toolchains without the optional <print> header.
#if defined(IO_PRINTLN)
#include <print>
#endif

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(IO_ATL)
#include <atl/io/fast_io.hpp>
namespace aio = atl::io;

static inline u32 rd() { return aio::read<u32>(); }

static inline void wr(u32 x) { aio::writeln(x); }

static inline void wr64(u64 x) { aio::writeln(x); }

static inline void io_flush() { aio::flush(); }

static inline void io_setup() {}

#elif defined(IO_GETCHAR)

// Windows exposes unlocked stdio under different names.
static inline int input_char() {
#ifdef _WIN32
	return _getchar_nolock();
#else
	return getchar_unlocked();
#endif
}

static inline void output_char(int c) {
#ifdef _WIN32
	_putchar_nolock(c);
#else
	putchar_unlocked(c);
#endif
}

static inline u32 rd() {
	int c = input_char();
	while (c != EOF && (c < '0' || c > '9')) c = input_char();
	if (c == EOF) std::exit(1);
	u32 x = 0;
	while (c >= '0' && c <= '9') x = x * 10 + u32(c - '0'), c = input_char();
	return x;
}

static inline void wr_impl(u64 x) {
	char b[24];
	int i = 0;
	if (!x) b[i++] = '0';
	while (x) b[i++] = char('0' + x % 10), x /= 10;
	while (i) output_char(b[--i]);
	output_char('\n');
}

static inline void wr(u32 x) { wr_impl(x); }

static inline void wr64(u64 x) { wr_impl(x); }

static inline void io_flush() { std::fflush(stdout); }

static inline void io_setup() {}

#elif defined(IO_PRINTLN)

static inline u32 rd() {
	u32 x;
	std::cin >> x;
	return x;
}

static inline void wr(u32 x) { std::println("{}", x); }

static inline void wr64(u64 x) { std::println("{}", x); }

static inline void io_flush() { std::fflush(stdout); }

static inline void io_setup() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
}

#else // IO_CIN

static inline u32 rd() {
	u32 x;
	std::cin >> x;
	return x;
}

static inline void wr(u32 x) { std::cout << x << '\n'; }

static inline void wr64(u64 x) { std::cout << x << '\n'; }

static inline void io_flush() { std::cout.flush(); }

static inline void io_setup() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
}
#endif
