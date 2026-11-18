#include <atl/nt/mod_int.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdint>
#include <string>

using atl::uint64;

template<typename Base>
concept powerable = requires(Base b, uint64 e) { atl::power(b, e); };
static_assert(powerable<int>);
static_assert(powerable<atl::mod_int32<1'000'000'007>>);
static_assert(!powerable<std::string>);

TEST_CASE("power: integers and mod_int", "[power]") {
	REQUIRE(atl::power(int64_t(3), uint64(0)) == 1);
	REQUIRE(atl::power(int64_t(3), uint64(13)) == 1594323);
	REQUIRE(atl::power(int64_t(2), uint64(62)) == int64_t(1) << 62);
	using mint = atl::mod_int32<1'000'000'007>;
	REQUIRE(atl::power(mint(2), uint64(30)) == mint(1 << 30));
	REQUIRE(atl::power(mint(5), uint64(1'000'000'006)) == mint(1));
}
