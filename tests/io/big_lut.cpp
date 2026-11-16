// Keep the optional output table exercised in its own executable.
#define ATL_IO_BIG_LUT
#include <atl/io/fast_io.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

TEST_CASE("io: four-digit table formats integer and chunk boundaries", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	atl::uint128 const chunk = 10'000'000'000'000'000'000ull;
	{
		atl::io::writer w(fp);
		for (unsigned value : {0u, 9u, 10u, 99u, 100u, 999u, 1000u, 9999u, 10000u}) w.writeln(value);
		w.writeln(chunk - 1, chunk, chunk + 1, chunk * chunk, chunk * chunk + 7);
		w.writeln((std::numeric_limits<atl::uint128>::max)(), (std::numeric_limits<atl::int128>::min)());
		w.put(std::string_view{});
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		for (std::string_view const expected : {"0", "9", "10", "99", "100", "999", "1000", "9999", "10000",
												"9999999999999999999", "10000000000000000000", "10000000000000000001", "100000000000000000000000000000000000000",
												"100000000000000000000000000000000000007", "340282366920938463463374607431768211455", "-170141183460469231731687303715884105728"}) {
			REQUIRE(r.read_token() == expected);
		}
		REQUIRE(r.eof());
	}
	std::fclose(fp);
}
