# `atl::io` — fast I/O

Buffered whitespace-separated input and decimal output for batch processing.
Use it when token parsing or integer formatting dominates runtime. Input must
be well formed and within the requested type's range; use another interface
when you need validation or recoverable I/O errors. Native and software
`atl::int128` / `atl::uint128` work on every supported target.

```cpp
#include <atl/io/fast_io.hpp>
#include <atl/io/reader.hpp>
#include <atl/io/writer.hpp>
```

The umbrella includes both individual headers. This complete example writes its
own temporary input, then reads it without depending on stdin:

\include fast-io/roundtrip.cpp

Expected stdout:

\include fast-io/roundtrip.out

## API

`reader` and `writer` are noncopyable, own their buffers, and borrow a `FILE*`.
They never close that stream. All stream operations are runtime only.
`inline reader in` and `inline writer out` bind the free functions to stdin
and stdout. `in` does not read until a request arrives; `out` flushes on destruction.

`readable<T>` accepts atl integers, `bool`, `char`, `std::string`, and models
of `atl::modular_integer`, including `mod_int` and `dynamic_mod_int`. The integer
is read and its constructor reduces it. Floating types additionally
require `__cpp_lib_to_chars`. `readable_into<T>` additionally checks
`requires(T &dst, T (*produce)()) { dst = produce(); }`, so destination-taking
reads reject types such as `const int` during overload resolution.
`writable<T>` checks that `writer::put(T const&)` is valid: integers, booleans, characters, strings/string views, NUL-terminated
character pointers, modular integers (`mod_int`, `dynamic_mod_int`, or other
models, printed as `value()`) and supported floats.

Reader members (template arguments are constrained by `readable` unless a
more specific constraint is shown). `operator>>`, `try_read`, and
`read(Ts &...xs)` constrain every destination type with `readable_into`.
`read_n<T>` additionally requires `std::same_as<T, std::remove_cv_t<T>>` for
vector storage. Value-returning `read<T>()` and `read_one<T>()` retain the
`readable` constraint, so `read<const int>()` remains valid:

| Signature | Effect / return |
|---|---|
| `explicit reader(std::FILE *src = stdin)` | Borrow the source; optionally map a regular file at its current position |
| `~reader()` | Release storage; leave source open |
| `T read_one()`, `T read()` | Read one value; zero/empty at EOF |
| `std::tuple<A, B, R...> read()` | Read at least two values, strictly left to right |
| `reader &operator>>(T &x)` | Assign a value; return this reader |
| `bool try_read(T &x)` | False at EOF, leaving `x` untouched; otherwise assign and return true |
| `bool read(Ts &...xs)` | Short-circuit left to right; earlier assignments remain if a later read reaches EOF; empty pack returns true |
| `std::vector<T> read_n(std::size_t n)` | Read exactly `n` values, including zero/empty values if EOF arrives early |
| `bool eof()` | Consume whitespace and report whether any input remains; may block on an open pipe |
| `U read_uint()` | Unsigned decimal (`atl::unsigned_integral`) |
| `T read_int()` | Signed decimal (`atl::signed_integral`), optional leading minus |
| `Z read_modular()` | `atl::modular_integer`: parse a uint64 magnitude, construct/reduce, then negate if needed |
| `std::string read_token()` | Read a token of any length |
| `F read_float()` | `std::floating_point`, optional charconv support; parse with `std::from_chars` |

A modular integer reads its sign and digits separately: "-v" is `-(v mod M)`,
so "-1" is `M - 1` for every modulus width, including a 64-bit modulus whose
residues need the full uint64 range. `mod_int`'s `operator>>` reads the same way.

The free `T read<T>()`, `std::tuple<A, B, R...> read()`,
`bool read(Ts &...xs)`, `std::vector<T> read_n(std::size_t n)`, and `bool eof()`
use `in` with the same constraints and behavior.

Writer members:

| Signature | Effect / return |
|---|---|
| `explicit writer(std::FILE *sink = stdout)` | Borrow the sink |
| `~writer()` | Flush and release storage; leave sink open |
| `void put(char c)`, `void put(bool is_true)` | Emit a character or `0`/`1` |
| `void put(std::string_view s)`, `void put(char const *s)`, `void put(std::string const &s)` | Emit bytes verbatim; pointer overload requires a NUL terminator |
| `void put(T x)` | Signed/unsigned atl integer: emit decimal, including signed minimum |
| `void put(Z const &x)` | Modular model: emit `value()` |
| `void put(F x)` | Float: shortest round-tripping representation via charconv |
| `void put_fixed(F x, int precision)` | Float: fixed notation with `precision` fractional digits, correctly rounded |
| `writer &operator<<(T const &x)` | Emit a writable value and return this writer |
| `void write()`, `void write(T const &x, Ts const &...xs)` | Writable values, space-separated, no newline; empty call emits nothing |
| `void writeln(Ts const &...xs)` | Writable values plus newline; empty call emits a blank line |
| `void writeln(R const &r)` | Const-iterable range of writable elements, space-separated, plus newline; strings use the value overload |
| `void write_fixed(F x, int precision)`, `void writeln_fixed(F x, int precision)` | Fixed float output, without/with newline |
| `void flush()` | Spill this buffer and call `std::fflush` on the sink |

The free `void write()`, `void write(T const &x, Ts const &...xs)`,
`void writeln(Ts const &...xs)`, `void writeln(R const &r)`,
`void write_fixed(F x, int precision)`, `void writeln_fixed(F x, int precision)`,
and `void flush()` forward to `out`. Floating overloads require
`std::floating_point<F>` and `__cpp_lib_to_chars`. Range overloads require
`std::ranges::input_range<R const>`,
`requires(writer &w, std::ranges::range_reference_t<R const> x) { w.put(x); }`,
and no conversion from `R const&` to `std::string_view`. A range iterable only
when non-const is rejected; a range iterable only when const is accepted.
The member defers the element check through its defaulted `Self = writer`
template parameter.

## Complexity

Reading and writing take O(bytes consumed/emitted), with a 256 KiB buffer per
instance. Strings and `read_n` also allocate their returned storage.
`ATL_USE_MMAP` uses O(file size) virtual address space for a regular POSIX file;
physical residency follows operating-system paging. Unsupported platforms,
empty files, unrepresentable mapping sizes and mapping failures use the buffer.
Fixed float output may allocate O(precision + maximum decimal exponent) storage.

## Preconditions

Asserted under `ATL_ASSERT`: charconv output fits the internal buffer selected
for it. Fixed output passes precision through to charconv; negative precision
uses its printf-style default of six fractional digits.

Documented only:

- Streams are nonnull and have the requested read/write mode. A source stays
  open during reads. A writer must be destroyed before `fclose`,
  even if it has already been flushed. Do not interleave reads through stdio or
  other readers; interactive reads use the underlying descriptor directly.
- Numeric tokens are at most 64 bytes and fit the destination. A signed
  minimum's unsigned magnitude is supported. Modular magnitudes fit uint64.
  No overflow check is performed. Strings may cross arbitrarily many refills.
- Whitespace is any byte at most ASCII space. Unsigned tokens have no sign;
  signed integers allow `-`, and floating syntax follows `std::from_chars`.
  Boolean input reads a uint32 and tests whether it is nonzero.
- Modular constructors must reduce magnitudes, `value()` must return a residue,
  and unary minus must implement modular negation; syntax is checked by the concept.
- A mapped file must not be truncated while the reader exists. Stream positions
  are not synchronized with the reader's buffered/mapped cursor.

There is no error state. Read errors act like EOF; write errors are ignored.
An integer read stops before the first non-digit. If there is no initial digit,
it returns zero and consumes the malformed token, allowing EOF loops to advance.
A consumed minus sign is allowed before this check. Float conversion errors leave
zero, with token consumption. These conveniences do not validate input.

## Measured performance

`benchmarks/io/writer.cpp` races ways of formatting unsigned decimal integers
into memory. All produce the same digits; they differ in where the digits go.
"Scratch + copy" writes them backward into a scratch buffer and copies them out;
for `uint128` that means 128-bit divisions by 100 ("128-bit digit pairs"). "In
place" counts the digits first and writes them backward directly into the
output, which is what `io::writer` does for up to 64 bits. "10^19 chunks" splits
a `uint128` into 19-digit chunks so only the splitting divisions are 128-bit.
The library row runs `io::writer` on the same values into the null device, so it
also includes the buffered stdio flush. Each row formats a whole vector,
space-separated, so it measures throughput.

Xeon Platinum 8280, GCC 15.2, RelWithDebInfo without assertions, 2026-10-15.
Minimum of 3 runs, milliseconds per vector:

| shape | flags | library (io::writer to null device) | scratch + copy | in place (count digits first) |
|---|---|---:|---:|---:|
| 1e7 uint32, mixed lengths | `-O2` | 119 | 152 | 121 |
| 1e7 uint32, mixed lengths | `-O2 -march=native` | 136 | 144 | 136 |
| 1e7 uint32, nine digits | `-O2` | n/a | 96.8 | 73.1 |
| 1e7 uint32, nine digits | `-O2 -march=native` | n/a | 94.2 | 68.9 |
| 1e7 uint64, mixed lengths | `-O2` | 186 | 183 | 171 |
| 1e7 uint64, mixed lengths | `-O2 -march=native` | 177 | 192 | 170 |

| shape | flags | library (io::writer to null device) | 128-bit digit pairs | 10^19 chunks |
|---|---|---:|---:|---:|
| 1e6 uint128, full width | `-O2` | 83.2 | 161 | 83.8 |
| 1e6 uint128, full width | `-O2 -march=native` | 82.4 | 181 | 83.0 |

Placing the digits in the output is never slower than scratch + copy: the
scratch + copy form is 6-37% slower on 32- and 64-bit values. 128-bit digit
pairs are 1.9-2.2x slower than 10^19 chunks, which tie the library.

No measurements are recorded for software int128 or the interactive backend.
The experimental harness in `experimental/io_bench/` compares complete backends
and pure input/output workloads; its range-sum workload has no historical timings.

## Pitfalls and notes

The default whole-buffer `fread` is for batch input: on a pipe/terminal it can
wait for 256 KiB or EOF. Define `ATL_IO_INTERACTIVE` consistently before all
includes for interactive protocols. It uses short POSIX `read` or Windows CRT
`_read`, and refills a partial token until its delimiter arrives. Send a
delimiter after every token and call `flush()` after each reply. On Windows,
use binary pipe streams when byte-exact behavior is required. Unsupported
short-read platforms diagnose the opt-in macro at compile time.

`ATL_USE_MMAP` is optional and POSIX-only; Windows keeps the buffered path.
`ATL_IO_BIG_LUT` chooses a 40 KiB, four-digit table with scratch-and-copy over
the default 200-byte, two-digit table. It was not part of the recorded race;
measure on your target before enabling it. All switches must match across
translation units because they change inline definitions.

## How it works

The reader compacts a short unread tail before refilling. It makes at most
64 token bytes contiguous, then parses eight decimal digits per SWAR step,
four per shorter step, and the remainder individually. Digit validation rejects
punctuation sharing the digits' high nibble. Byte swapping normalizes the SWAR
loads on big-endian targets. Whitespace and string scans cross refills; mmap
instead exposes the current-position suffix of the file directly.

The writer estimates decimal length from bit width and corrects it with a
powers-of-ten table. It writes digit pairs directly into the output buffer,
avoiding scratch storage and a second copy. A 128-bit value is split into at
most three base-10^19 chunks, with lower chunks padded to nineteen digits.
The native arithmetic fast path is retained; software uint128 uses the
published wide-word quotient/remainder primitive, so it needs no native
128-bit division. Signed magnitudes are negated in unsigned arithmetic to
support the minimum value. Spilling uses `fwrite`; explicit flush and destruction
also flush the stdio buffer so an interactive peer receives each reply.

See also: [Types and utilities](types-and-utilities.md), [128-bit integers](int128.md),
[Modular arithmetic](modular-arithmetic.md).
