// A separate executable exercises opt-in mappings without replacing fread tests.
#define ATL_USE_MMAP
#include <atl/io/fast_io.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdio>
#include <limits>

TEST_CASE("io: mmap reads from the current file position and handles an unterminated token", "[io]") {
	std::FILE *file = std::tmpfile();
	REQUIRE(file != nullptr);
	std::fputs("skip 340282366920938463463374607431768211455 -170141183460469231731687303715884105728", file);
	REQUIRE(std::fflush(file) == 0);
	REQUIRE(std::fseek(file, 5, SEEK_SET) == 0);
	{
		// On Windows the same API intentionally uses fread: mmap is POSIX-only.
		atl::io::reader r(file);
		REQUIRE(r.read<atl::uint128>() == (std::numeric_limits<atl::uint128>::max)());
		REQUIRE(r.read<atl::int128>() == (std::numeric_limits<atl::int128>::min)());
		REQUIRE(r.eof());
	}
	std::fclose(file);
}

TEST_CASE("io: mmap opt-in handles empty files and an initial EOF position", "[io]") {
	std::FILE *file = std::tmpfile();
	REQUIRE(file != nullptr);
	{
		atl::io::reader r(file);
		REQUIRE(r.eof());
		REQUIRE(r.read<int>() == 0);
	}
	std::fputs("1", file);
	REQUIRE(std::fflush(file) == 0);
	{
		atl::io::reader r(file);
		REQUIRE(r.eof());
	}
	std::fclose(file);
}
