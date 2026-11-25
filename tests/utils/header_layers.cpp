#include <atl/ds/flat_hash_map_core.hpp>
#include <atl/nt/mod_int_core.hpp>
#include <array>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template<typename Key>
constexpr bool stable_hash_interface() {
	return atl::detail::hasher_for<atl::hash<Key>, Key>
		&& atl::detail::nothrow_hasher_for<atl::hash<Key>, Key>
		&& std::is_default_constructible_v<atl::hash<Key>>
		&& std::is_nothrow_invocable_r_v<std::size_t, atl::hash<Key>, Key const &>
		&& requires { typename atl::flat_hash_map<Key, int>; };
}

struct unsupported_key {};

static_assert(!atl::detail::hasher_for<atl::hash<unsupported_key>, unsupported_key>);
static_assert(stable_hash_interface<std::string>());
static_assert(stable_hash_interface<std::string_view>());
static_assert(stable_hash_interface<double>());
static_assert(stable_hash_interface<int *>());
static_assert(stable_hash_interface<std::pair<std::string, double>>());
static_assert(stable_hash_interface<std::tuple<std::string, double>>());
static_assert(stable_hash_interface<std::vector<std::string>>());
static_assert(stable_hash_interface<std::array<std::string, 2>>());
using mint = atl::mod_int32<17>;
static_assert(requires(std::ostream &os, std::istream &is, mint &m) { os << m; is >> m; });

#include <atl/ds/flat_hash_map.hpp>
#include <atl/nt/mod_int.hpp>
#include <catch2/catch.hpp>
#include <sstream>

static_assert(stable_hash_interface<std::string>());
static_assert(stable_hash_interface<std::pair<std::string, double>>());
static_assert(!atl::detail::hasher_for<atl::hash<unsupported_key>, unsupported_key>);

template<typename T>
struct stream_box {
	T value{3};

	operator T() const { return value; }
};

template<typename T>
struct stream_ref {
	T value{};

	operator T &() { return value; }
};

template<atl::static_mod_int T>
std::ostream &operator<<(std::ostream &os, T const &) { return os << "generic"; }

TEST_CASE("hash and stream interfaces survive late extensions", "[hash][mod_int]") {
	atl::flat_hash_map<std::string, int> map;
	map["ok"] = 7;
	REQUIRE(map.at("ok") == 7);
	REQUIRE(atl::hash<std::string>{}("ok") == atl::hash<std::string_view>{}("ok"));
	REQUIRE(atl::hash<double>{}(0.) == atl::hash<double>{}(-0.));
	REQUIRE(atl::hash<std::pair<std::string, double>>{}({"ok", 0.}) == atl::hash<std::pair<std::string, double>>{}({"ok", -0.}));
	REQUIRE(atl::hash<std::tuple<std::string, double>>{}({"ok", 0.}) == atl::hash<std::tuple<std::string, double>>{}({"ok", -0.}));
	REQUIRE(atl::hash<std::vector<std::string>>{}({"ok"}) == atl::hash<std::array<std::string, 1>>{}({"ok"}));
	int x{};
	REQUIRE(atl::hash<int *>{}(&x) == atl::hash<int *>{}(&x));
	std::ostringstream os;
	os << stream_box<mint>{} << ' ' << mint(4);
	REQUIRE(os.str() == "3 4");
	std::istringstream is("20");
	stream_ref<mint> box;
	is >> box;
	REQUIRE(box.value == mint(3));
}
