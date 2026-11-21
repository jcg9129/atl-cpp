#include <atl/strings/string_hash.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash.hpp>
#include <catch2/catch.hpp>
#include <cctype>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint64;

TEST_CASE("string_hash: basic queries", "[string_hash]") {
	std::string const s = "abracadabra";
	atl::string_hash h(s, 131);

	REQUIRE(h.size() == s.size());
	REQUIRE(h.base() == 131);
	REQUIRE(h.equal(0, 4, 7, 11));							   // "abra" == "abra"
	REQUIRE(!h.equal(0, 4, 1, 5));							   // "abra" != "brac"
	REQUIRE(h.substring_hash(0, 0) == h.substring_hash(5, 5)); // empty ranges agree
	REQUIRE(h.full_hash() == h.substring_hash(0, s.size()));
}

TEST_CASE("string_hash: substring equality matches content", "[string_hash]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::string s;
	for (int i = 0; i < 400; ++i) s.push_back(char('a' + std::size_t(rng() % 4))); // small alphabet -> many collisions to check
	atl::string_hash h(s, uint64(2 + (rng() % (atl::string_hash::mod - 2))));

	for (int t = 0; t < 3000; ++t) {
		std::size_t l1 = std::size_t(rng() % s.size()), r1 = l1 + std::size_t(rng() % (s.size() - l1 + 1));
		std::size_t l2 = std::size_t(rng() % s.size()), r2 = l2 + std::size_t(rng() % (s.size() - l2 + 1));
		bool const same = s.substr(l1, r1 - l1) == s.substr(l2, r2 - l2);
		REQUIRE(h.equal(l1, r1, l2, r2) == same);
	}
}

TEST_CASE("string_hash: same base is comparable across instances", "[string_hash]") {
	std::string const a = "the quick brown fox";
	std::string const b = "lazy dog quick brown";
	atl::string_hash ha(a, 137);
	atl::string_hash hb(b, 137);
	// "quick brown" appears in both
	REQUIRE(ha.substring_hash(4, 15) == hb.substring_hash(9, 20));
	REQUIRE(ha.substring_hash(0, 3) != hb.substring_hash(0, 3)); // "the" vs "laz"

	atl::string_hash hb2(b, 138); // different base -> not comparable
	REQUIRE(ha.substring_hash(4, 15) != hb2.substring_hash(9, 20));
}

TEST_CASE("string_hash: generic element type", "[string_hash]") {
	std::vector<int> v = {5, 1, 4, 1, 5, 9, 2, 6, 1, 4};
	atl::string_hash h(v, 998244353);
	REQUIRE(h.equal(1, 3, 8, 10)); // {1,4} == {1,4}
	REQUIRE(!h.equal(0, 2, 1, 3)); // {5,1} != {1,4}
}

TEST_CASE("string_hash: one default base, pinned by set_hash_seed", "[string_hash]") {
	std::string const s = "reproducible";
	std::vector<int> const v(s.begin(), s.end());
	atl::string_hash const text(s);
	atl::string_hash const ints(v);
	REQUIRE(text.base() == ints.base());
	REQUIRE(text.full_hash() == ints.full_hash());
	REQUIRE(text.base() >= 2);
	REQUIRE(text.base() < atl::string_hash::mod);

	atl::set_hash_seed(20240101);
	uint64 const pinned = atl::string_hash::default_base();
	atl::string_hash const h1(s);
	atl::set_hash_seed(20240101);
	atl::string_hash const h2(s);
	REQUIRE(h1.base() == pinned);
	REQUIRE(h2.base() == pinned);
	REQUIRE(h1.full_hash() == h2.full_hash());
}

TEST_CASE("string_hash: projection, push_back, hash_of", "[string_hash]") {
	std::string const s = "Mississippi";
	auto const lower = [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); };
	atl::string_hash const projected(s, 131, lower);
	atl::string_hash const plain(std::string("mississippi"), 131);
	REQUIRE(projected.full_hash() == plain.full_hash());

	atl::string_hash online(131);
	REQUIRE(online.size() == 0);
	REQUIRE(online.full_hash() == 0);
	online.reserve(s.size());
	for (char const c : std::string("mississippi")) online.push_back(c);
	REQUIRE(online.size() == plain.size());
	for (std::size_t l = 0; l <= s.size(); ++l)
		for (std::size_t r = l; r <= s.size(); ++r) REQUIRE(online.substring_hash(l, r) == plain.substring_hash(l, r));

	REQUIRE(atl::hash_of(std::string("mississippi"), 131) == plain.full_hash());
	REQUIRE(atl::hash_of(s, 131, lower) == plain.full_hash());
	REQUIRE(atl::hash_of(std::string("ssi"), 131) == plain.substring_hash(2, 5));
	REQUIRE(atl::hash_of(std::string(), 131) == 0);
	REQUIRE(atl::hash_of(s) == atl::string_hash(s).full_hash());
}

TEST_CASE("string_hash: concat and lcp match a brute oracle", "[string_hash]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::string a, b;
	for (int i = 0; i < 120; ++i) a.push_back(char('a' + std::size_t(rng() % 2))); // sigma 2: long common prefixes
	for (int i = 0; i < 80; ++i) b.push_back(char('a' + std::size_t(rng() % 2)));
	uint64 const base = 2 + (rng() % (atl::string_hash::mod - 2));
	atl::string_hash const ha(a, base), hb(b, base);

	auto const brute_lcp = [](std::string const &x, std::size_t i, std::string const &y, std::size_t j) {
		std::size_t k = 0;
		while (i + k < x.size() && j + k < y.size() && x[i + k] == y[j + k]) ++k;
		return k;
	};
	for (int t = 0; t < 2000; ++t) {
		std::size_t const i = std::size_t(rng() % (a.size() + 1)), j = std::size_t(rng() % (a.size() + 1)), k = std::size_t(rng() % (b.size() + 1));
		REQUIRE(ha.lcp(i, j) == brute_lcp(a, i, a, j));
		REQUIRE(ha.lcp(i, hb, k) == brute_lcp(a, i, b, k));

		std::size_t const l1 = std::size_t(rng() % (a.size() + 1)), r1 = l1 + std::size_t(rng() % (a.size() - l1 + 1));
		std::size_t const l2 = std::size_t(rng() % (b.size() + 1)), r2 = l2 + std::size_t(rng() % (b.size() - l2 + 1));
		std::string const joined = a.substr(l1, r1 - l1) + b.substr(l2, r2 - l2);
		REQUIRE(ha.concat(ha.substring_hash(l1, r1), hb.substring_hash(l2, r2), r2 - l2) == atl::hash_of(joined, base));
	}
	// right part longer than the instance: the power falls back to squaring
	atl::string_hash const tiny(std::string("ab"), base);
	REQUIRE(tiny.concat(tiny.full_hash(), ha.full_hash(), a.size()) == atl::hash_of("ab" + a, base));

	atl::string_hash const other_base(b, base == 2 ? 3 : base - 1);
	CHECK(atl_test::aborts([&] { (void)ha.lcp(0, other_base, 0); })); // bases must agree
}
