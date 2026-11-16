#pragma once

/**
 * @file
 * @brief Buffered whitespace-separated input with eight-digit SWAR integer parsing.
 *
 * The default fread refill is for batch input: on a pipe it can wait for a full
 * buffer or EOF. Define ATL_IO_INTERACTIVE for short pipe reads on POSIX and
 * Windows, and flush each reply. ATL_USE_MMAP optionally maps regular files on
 * POSIX; other streams and platforms retain the buffered path.
 */

#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// Mapping is opt-in: it needs file-sized address space instead of a fixed buffer.
// Platform-only headers must stay guarded at file scope for amalgamation.
#if defined(ATL_USE_MMAP) && defined(__has_include)
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#define ATL_HAS_MMAP 1
#endif
#endif
/**
 * @brief 1 when the reader maps regular-file input with mmap: `ATL_USE_MMAP` is defined and the POSIX
 *        mapping headers exist; else 0 (buffered `fread` path).
 */
#ifndef ATL_HAS_MMAP
#define ATL_HAS_MMAP 0
#endif

#if ATL_HAS_MMAP
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Interactive refills must use short reads: falling back to fread can deadlock
// when a peer waits for a reply before sending another token.
#if defined(ATL_IO_INTERACTIVE)
#if defined(_WIN32)
#define ATL_HAS_INTERACTIVE_READ 1
#include <fcntl.h>
#include <io.h>
#elif defined(__has_include)
#if __has_include(<unistd.h>)
#define ATL_HAS_INTERACTIVE_READ 1
#include <unistd.h>
#endif
#endif
#ifndef ATL_HAS_INTERACTIVE_READ
#error "ATL_IO_INTERACTIVE needs Windows _read or POSIX read"
#endif
#endif
/**
 * @brief 1 when `ATL_IO_INTERACTIVE` is defined and refills use one short descriptor read (POSIX `read`,
 *        Windows `_read`); else 0 (refills fill the whole buffer with `fread`).
 */
#ifndef ATL_HAS_INTERACTIVE_READ
#define ATL_HAS_INTERACTIVE_READ 0
#endif

namespace atl::io {
	/**
	 * @brief The types the reader knows how to parse: integers (including 128-bit),
	 *        `bool`, `char`, `std::string` tokens, types modeling
	 *        @ref atl::modular_integer (read as a uint64 magnitude and reduced), and —
	 *        where `std::from_chars` supports it — floating point.
	 * @tparam T Candidate element type.
	 */
	template<typename T>
	concept readable = atl::integral<T> || std::same_as<T, bool> || std::same_as<T, char>
					|| std::same_as<T, std::string> || atl::modular_integer<T>
#if defined(__cpp_lib_to_chars)
					|| std::floating_point<T>
#endif
		;

	/**
	 * @brief Readable types whose destinations can be assigned a parsed value.
	 * @tparam T Candidate destination type, including cv-qualification.
	 */
	template<typename T>
	concept readable_into = readable<T> && requires(T &dst, T (*produce)()) { dst = produce(); };

	/**
	 * @brief Buffered whitespace-separated token reader over a `FILE *` (stdin by
	 *        default), with SWAR integer parsing.
	 *
	 * @pre Well-formed input: whitespace-separated tokens, each numeric token at
	 *      most 64 bytes, with a representable magnitude (uint64 for modular input).
	 *      Signed input may start with '-'; unsigned input has no sign. String
	 *      tokens may have any length. The source is nonnull, readable, and remains
	 *      open during read operations; do not mix other reads with this reader.
	 * @note Bytes <= space are delimiters. Numeric overflow is unchecked. At EOF,
	 *       reads return zero, an empty string, or a modular zero; try_read leaves
	 *       its argument unchanged. I/O errors are treated as EOF.
	 * @complexity Linear in bytes consumed, with O(2^18) buffered storage or an
	 *             O(file size) virtual mapping under ATL_USE_MMAP.
	 * @note Not constexpr (reads a runtime FILE stream and may map a file).
	 */
	class reader {
		static constexpr std::size_t buffer_capacity = 1 << 18;
		std::FILE *source_;
		char *buffer_ = nullptr;
		char const *cursor_ = nullptr;
		char const *end_ = nullptr;
		bool is_mmapped_ = false;
#if ATL_HAS_MMAP
		void *map_ = nullptr;
		std::size_t map_size_ = 0;
#endif

		/**
		 * @brief Preserve unread bytes contiguously; interactive refills must return
		 *        after one short descriptor read so a waiting peer can receive a reply.
		 */
		void refill() {
			if (is_mmapped_) return;
			std::size_t rem = std::size_t(end_ - cursor_);
			if (rem && cursor_ != buffer_) std::memmove(buffer_, cursor_, rem);
#if ATL_HAS_INTERACTIVE_READ
#ifdef _WIN32
			int got;
			do got = ::_read(::_fileno(source_), buffer_ + rem, static_cast<unsigned>(buffer_capacity - rem));
#else
			::ssize_t got;
			do got = ::read(::fileno(source_), buffer_ + rem, buffer_capacity - rem);
#endif
			while (got < 0 && errno == EINTR);
			if (got < 0) got = 0; // a read error ends the input, like EOF
			std::size_t const added = std::size_t(got);
#else
			std::size_t const added = std::fread(buffer_ + rem, 1, buffer_capacity - rem, source_);
#endif
			cursor_ = buffer_;
			end_ = buffer_ + rem + added;
		}

		/** @brief Guarantee a full token (<= 64 bytes) is contiguous unless at EOF. */
		void ensure_contiguous_token() {
#if ATL_HAS_INTERACTIVE_READ
			// a buffered delimiter already ends the token: reading on could block on
			// a peer that waits for our reply. A short read may deliver a partial
			// token, so keep reading until one ends, the window fills, or EOF
			auto const is_delimiter = [](char c) { return static_cast<unsigned char>(c) <= ' '; };
			while (std::size_t(end_ - cursor_) < 64 && std::ranges::find_if(cursor_, end_, is_delimiter) == end_) {
				std::size_t const old_size = std::size_t(end_ - cursor_);
				refill();
				if (std::size_t(end_ - cursor_) == old_size) return; // EOF (or mmapped)
			}
#else
			if (!is_mmapped_ && std::size_t(end_ - cursor_) < 64) refill();
#endif
		}

		void skip_whitespace() {
			for (;;) {
				while (cursor_ != end_ && static_cast<unsigned char>(*cursor_) <= ' ') ++cursor_;
				if (cursor_ != end_ || is_mmapped_) return;
				refill();
				if (cursor_ == end_) return;
			}
		}

		/** @brief Parse an unsigned integer assuming `cursor_` points at its first digit. */
		template<atl::unsigned_integral U>
		U parse_uint() {
			std::remove_cv_t<U> x = 0;
			while (end_ - cursor_ >= 8) {
				std::uint64_t v;
				std::memcpy(&v, cursor_, 8);
				if constexpr (std::endian::native == std::endian::big) v = std::byteswap(v);
				v ^= 0x3030303030303030ull;
				// a digit byte is now 0..9. A zero high nibble alone also
				// admits ':' .. '?' (10..15); adding 6 carries exactly those
				// into the high nibble, and cannot carry between bytes
				if ((v | (v + 0x0606060606060606ull)) & 0xf0f0f0f0f0f0f0f0ull) break;
				v = (v * 10 + (v >> 8)) & 0x00ff00ff00ff00ffull;
				v = (v * 100 + (v >> 16)) & 0x0000ffff0000ffffull;
				v = (v * 10000 + (v >> 32)) & 0x00000000ffffffffull;
				x = U(x * 100'000'000u + U(v));
				cursor_ += 8;
			}
			if (end_ - cursor_ >= 4) {
				std::uint32_t v;
				std::memcpy(&v, cursor_, 4);
				if constexpr (std::endian::native == std::endian::big) v = std::byteswap(v);
				v ^= 0x30303030u;
				if (!((v | (v + 0x06060606u)) & 0xf0f0f0f0u)) {
					v = (v * 10 + (v >> 8)) & 0x00ff00ffu;
					v = (v * 100 + (v >> 16)) & 0x0000ffffu;
					x = U(x * 10000u + U(v));
					cursor_ += 4;
				}
			}
			while (cursor_ != end_) {
				unsigned c = static_cast<unsigned char>(*cursor_);
				if (c - '0' > 9u) break;
				x = U(x * 10 + (c - '0'));
				++cursor_;
			}
			return x;
		}

		/**
		 * @brief Consume a malformed token whole. A token with no digit where one
		 *        is expected ("abc", or the '-' of a negative number read as
		 *        unsigned) reads as 0 and is skipped, so a read-until-EOF loop
		 *        always advances. If digits were already consumed, leave the cursor
		 *        unchanged so callers can parse punctuation separately.
		 * @param digits_begin Cursor position before parsing digits; consume a suffix only if no digit was read.
		 * @complexity Linear in the skipped suffix, including refills.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		void skip_malformed_token(char const *digits_begin) {
			if (cursor_ != digits_begin) return;
			for (;;) {
				while (cursor_ != end_ && static_cast<unsigned char>(*cursor_) > ' ') ++cursor_;
				if (cursor_ != end_ || is_mmapped_) return;
				refill();
				if (cursor_ == end_) return;
			}
		}

	public:
		/**
		 * @brief Read from @p src. Nothing is read until the first request, so a
		 *        program that never reads never blocks on the source.
		 * @param src Borrowed readable stream; nonnull and open during reads.
		 * @complexity O(1) initialization plus buffer allocation or mapping setup.
		 * @note Not constexpr (uses a runtime FILE stream). With `ATL_IO_INTERACTIVE` on Windows the
		 *       source is switched to binary mode (`_setmode`), so CR is read as a delimiter byte.
		 */
		explicit reader(std::FILE *src = stdin) : source_(src) {
#if ATL_HAS_MMAP
			int fd = ::fileno(source_);
			struct ::stat st;
			long const offset = std::ftell(source_);
			// Both the mapping size and pointer differences must fit on 32-bit targets.
			if (fd >= 0 && offset >= 0 && ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0
				&& offset <= st.st_size && std::uintmax_t(st.st_size) <= std::uintmax_t((std::numeric_limits<std::ptrdiff_t>::max)())) {
				void *p = ::mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
				if (p != MAP_FAILED) {
					map_ = p;
					map_size_ = std::size_t(st.st_size);
					is_mmapped_ = true;
					cursor_ = static_cast<char const *>(p) + offset;
					end_ = static_cast<char const *>(p) + map_size_;
					return;
				}
			}
#endif
			buffer_ = new char[buffer_capacity];
			cursor_ = end_ = buffer_;
#if ATL_HAS_INTERACTIVE_READ && defined(_WIN32)
			// In text mode _read blocks after a trailing CR to look for LF, so a peer that sends
			// "123\r" and waits would deadlock. Every byte <= ' ' is a delimiter, so binary mode
			// changes no parse result.
			::_setmode(::_fileno(source_), _O_BINARY);
#endif
		}

		/** @brief Readers exclusively own their buffer or mapping and cannot be copied. */
		reader(reader const &) = delete;
		/** @brief Readers cannot share ownership by copy assignment. */
		reader &operator=(reader const &) = delete;

		/** @brief Release the buffer or mapping without closing the source. @note Not constexpr (releases runtime storage). */
		~reader() {
#if ATL_HAS_MMAP
			if (is_mmapped_) {
				::munmap(map_, map_size_);
				return;
			}
#endif
			delete[] buffer_;
		}

		/**
		 * @brief Read the next token as an unsigned integer (0 past EOF or if malformed).
		 * @tparam U Unsigned integer type.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<atl::unsigned_integral U>
		U read_uint() {
			skip_whitespace();
			ensure_contiguous_token();
			char const *const digits_begin = cursor_;
			U const v = parse_uint<U>();
			skip_malformed_token(digits_begin);
			return v;
		}

		/**
		 * @brief Read the next token as a signed integer (0 past EOF or if malformed).
		 * @tparam T Element type accepted by the declared constraint.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<atl::signed_integral T>
		T read_int() {
			skip_whitespace();
			ensure_contiguous_token();
			bool is_negative = (cursor_ != end_ && *cursor_ == '-');
			if (is_negative) ++cursor_;
			using U = std::remove_cv_t<atl::traits::make_unsigned_t<T>>;
			char const *const digits_begin = cursor_;
			U v = parse_uint<U>();
			skip_malformed_token(digits_begin);
			return is_negative ? T(U(0) - v) : T(v);
		}

		/**
		 * @brief Read the next token as a residue of `Z::mod()`: "-v" reads as
		 *        `-(v mod M)`, so "-1" is `M - 1` for every modulus width
		 *        (0 past EOF or if malformed).
		 * @tparam Z Model of atl::modular_integer; construction reduces the magnitude.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<atl::modular_integer Z>
		Z read_modular() {
			skip_whitespace();
			ensure_contiguous_token();
			bool const is_negative = (cursor_ != end_ && *cursor_ == '-');
			if (is_negative) ++cursor_;
			char const *const digits_begin = cursor_;
			// sign and magnitude apart: a 64-bit modulus needs the full uint64
			// range, where reading through int64 would wrap a negative mod 2^64
			Z const magnitude(parse_uint<uint64>());
			skip_malformed_token(digits_begin);
			return is_negative ? -magnitude : magnitude;
		}

		/**
		 * @brief Read the next whitespace-delimited token (empty past EOF).
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		std::string read_token() {
			skip_whitespace();
			std::string s;
			for (;;) {
				char const *p = cursor_;
				while (p != end_ && static_cast<unsigned char>(*p) > ' ') ++p;
				s.append(cursor_, p);
				cursor_ = p;
				if (p != end_ || is_mmapped_) break;
				refill();
				if (cursor_ == end_) break;
			}
			return s;
		}

#if defined(__cpp_lib_to_chars)
		/**
		 * @brief Read the next token as a floating-point value via `std::from_chars` (correctly rounded).
		 * @tparam F Floating-point type supported by charconv.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<std::floating_point F>
		F read_float() {
			skip_whitespace();
			ensure_contiguous_token();
			char const *p = cursor_;
			while (p != end_ && static_cast<unsigned char>(*p) > ' ') ++p;
			std::remove_cv_t<F> val = 0;
			if (p != end_ || is_mmapped_) {
				std::from_chars(cursor_, p, val);
				cursor_ = p;
			} else { // rare: token runs to the buffer end in fread mode
				std::string tok = read_token();
				std::from_chars(tok.data(), tok.data() + tok.size(), val);
			}
			return val;
		}
#endif

		/**
		 * @brief Read the next token as a @p T.
		 * @tparam T Element type accepted by the declared constraint.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable T>
		T read_one() {
			if constexpr (atl::unsigned_integral<T>)
				return read_uint<T>();
			else if constexpr (atl::signed_integral<T>)
				return read_int<T>();
			else if constexpr (std::is_same_v<T, bool>)
				return read_uint<uint32>() != 0;
			else if constexpr (std::is_same_v<T, char>) {
				skip_whitespace();
				ensure_contiguous_token();
				return cursor_ != end_ ? *cursor_++ : char(0);
			} else if constexpr (std::is_same_v<T, std::string>)
				return read_token();
			else if constexpr (atl::modular_integer<T>)
				return read_modular<T>();
#if defined(__cpp_lib_to_chars)
			else if constexpr (std::is_floating_point_v<T>)
				return read_float<T>();
#endif
			else
				static_assert(sizeof(T) == 0, "atl::io: unsupported read type (float I/O needs __cpp_lib_to_chars)");
		}

		/**
		 * @brief Stream-style read into @p x.
		 * @tparam T A readable type assignable from a parsed value.
		 * @param x Destination value.
		 * @return This stream instance.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable_into T>
		reader &operator>>(T &x) {
			x = read_one<T>();
			return *this;
		}

		/**
		 * @brief Whether no token remains (only trailing whitespace before EOF).
		 * @return True if no non-whitespace byte remains; may block on a live pipe.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		bool eof() {
			skip_whitespace();
			return cursor_ == end_;
		}

		/**
		 * @brief One value: `auto x = r.read<int>();` (0 past EOF for numbers).
		 * @tparam T Element type accepted by the declared constraint.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable T>
		T read() {
			return read_one<T>();
		}

		/**
		 * @brief Several values for structured bindings, read strictly left to
		 *        right: `auto [n, q] = r.read<int, int>();`
		 * @tparam R Additional readable value types.
		 * @tparam B Second readable value type.
		 * @tparam A First readable value type.
		 * @return The parsed result, or zero/empty at EOF.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable A, readable B, readable... R>
		std::tuple<A, B, R...> read() {
			return std::tuple<A, B, R...>{read_one<A>(), read_one<B>(), read_one<R>()...};
		}

		/**
		 * @brief Read one value into @p x; returns false (x untouched) if input is exhausted.
		 * @tparam T A readable type assignable from a parsed value.
		 * @param x Destination value.
		 * @return True if a value was read; false at EOF with x unchanged.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable_into T>
		bool try_read(T &x) {
			if (eof()) return false;
			x = read_one<T>();
			return true;
		}

		/**
		 * @brief Read into existing lvalues; returns false as soon as input is
		 *        exhausted, so it doubles as a read-until-EOF loop (see the free
		 *        `io::read(Ts&...)`).
		 * @tparam Ts Readable types assignable from parsed values.
		 * @param xs Destinations, read left to right until EOF.
		 * @return False at EOF; earlier successful assignments remain. An empty pack returns true.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable_into... Ts>
		bool read(Ts &...xs) {
			return (try_read(xs) && ...);
		}

		/**
		 * @brief Read @p n values into a vector: `auto a = r.read_n<int>(n);`
		 * @tparam T A cv-unqualified readable vector element type.
		 * @param n Number of values to read, including zero/empty values after EOF.
		 * @return Exactly n values.
		 * @complexity O(n + bytes consumed) time and O(n) result storage.
		 * @note Not constexpr (uses a runtime FILE stream).
		 */
		template<readable T>
			requires std::same_as<T, std::remove_cv_t<T>>
		std::vector<T> read_n(std::size_t n) {
			std::vector<T> v;
			v.reserve(n);
			for (std::size_t i = 0; i < n; ++i) v.push_back(read_one<T>());
			return v;
		}
	};

	/**
	 * @brief The global stdin reader behind the free functions.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	// NOLINTNEXTLINE(bugprone-throwing-static-initialization): the io singleton by design
	inline reader in;

	/**
	 * @brief Whether stdin has no more tokens.
	 * @return True if no non-whitespace byte remains; may block on a live pipe.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	inline bool eof() { return in.eof(); }

	/**
	 * @brief One value from stdin: `auto x = read<int>();` (0 past EOF for numbers).
	 * @tparam T Element type accepted by the declared constraint.
	 * @return The parsed result, or zero/empty at EOF.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	template<readable T>
	T read() {
		return in.read<T>();
	}

	/**
	 * @brief Several values from stdin for structured bindings:
	 *        `auto [n, q] = read<int, int>();`. Elements are read strictly left to
	 *        right.
	 * @tparam R Additional readable value types.
	 * @tparam B Second readable value type.
	 * @tparam A First readable value type.
	 * @return The parsed result, or zero/empty at EOF.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	template<readable A, readable B, readable... R>
	std::tuple<A, B, R...> read() {
		return in.read<A, B, R...>();
	}

	/**
	 * @brief Read stdin into existing lvalues; returns false as soon as input is
	 *        exhausted, so it doubles as a read-until-EOF loop:
	 *
	 *     int x; while (read(x)) { ... } // process every value until EOF
	 *     read(a, b, c); // trust the input, ignore the bool
	 * @tparam Ts Readable types assignable from parsed values.
	 * @param xs Destinations, read left to right until EOF.
	 * @return False at EOF; earlier successful assignments remain. An empty pack returns true.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	template<readable_into... Ts>
	bool read(Ts &...xs) {
		return (in.try_read(xs) && ...);
	}

	/**
	 * @brief Read @p n values from stdin into a vector: `auto a = read_n<int>(n);`
	 * @tparam T A cv-unqualified readable vector element type.
	 * @param n Number of values to read, including zero/empty values after EOF.
	 * @return Exactly n values.
	 * @complexity O(n + bytes consumed) time and O(n) result storage.
	 * @note Not constexpr (uses a runtime FILE stream).
	 */
	template<readable T>
		requires std::same_as<T, std::remove_cv_t<T>>
	std::vector<T> read_n(std::size_t n) {
		return in.read_n<T>(n);
	}
}
