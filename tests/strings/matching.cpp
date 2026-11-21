#include <atl/strings/matching.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cctype>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "../expect_abort.hpp"

using atl::border_lengths;
using atl::common_prefix_lengths;
using atl::occurrences;

namespace {
	std::uint32_t brute_border(std::string const &p) {
		for (std::uint32_t len = std::uint32_t(p.size()) - 1;; --len) {
			if (p.compare(0, len, p, p.size() - len, len) == 0) return len;
			if (len == 0) return 0;
		}
	}

	std::vector<std::uint32_t> brute_occurrences(std::string const &t, std::string const &p) {
		std::vector<std::uint32_t> out;
		if (p.empty()) return out;
		for (std::uint32_t i = 0; i + p.size() <= t.size(); ++i)
			if (t.compare(i, p.size(), p) == 0) out.push_back(i);
		return out;
	}
}

TEST_CASE("matching: border and common-prefix lengths match brute force", "[strings]") {
	auto seed = GENERATE(take(60, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::string s(std::size_t(rng() % 40), 'a'); // includes the empty text
	for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
	std::size_t const n = s.size();

	auto const border = border_lengths(s);
	auto const lcp = common_prefix_lengths(s);
	REQUIRE(border.size() == n);
	REQUIRE(lcp.size() == n);
	for (std::uint32_t i = 0; i < n; ++i) {
		REQUIRE(border[i] == brute_border(s.substr(0, i + 1)));
		std::uint32_t k = 0;
		while (i + k < n && s[k] == s[i + k]) ++k;
		REQUIRE(lcp[i] == k);
	}
	if (n > 0) REQUIRE(lcp[0] == n);

	// the documented identity: smallest period of the text = n - border[n-1]
	if (n > 0) {
		std::uint32_t const period = std::uint32_t(n) - border[n - 1];
		std::uint32_t smallest = 1;
		while (smallest < n) {
			bool ok = true;
			for (std::uint32_t i = smallest; i < n && ok; ++i) ok = s[i] == s[i - smallest];
			if (ok) break;
			++smallest;
		}
		REQUIRE(period == smallest);
	}
}

TEST_CASE("matching: occurrences matches brute force, overlaps included", "[strings]") {
	auto seed = GENERATE(take(60, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::string t(std::size_t(rng() % 50), 'a');
	for (auto &c : t) c = char('a' + std::size_t(rng() % sigma));

	for (int probe = 0; probe < 20; ++probe) {
		std::string p(1 + std::size_t(rng() % 8), 'a');
		for (auto &c : p) c = char('a' + std::size_t(rng() % sigma));
		if (std::size_t(rng() % 3) == 0 && p.size() <= t.size() && !t.empty()) {
			// plant a real substring so hits are frequent
			auto const at = std::size_t(rng() % (t.size() - p.size() + 1));
			p = t.substr(at, p.size());
		}
		auto const got = occurrences(t, p, border_lengths(p));
		REQUIRE(got == brute_occurrences(t, p));
	}
}

TEST_CASE("matching: array conversions invert each other and the text versions", "[strings]") {
	using atl::to_border_lengths;
	using atl::to_common_prefix_lengths;

	// exhaustive over every small string — the sharpest check of the
	// conversions' propagation/break subtleties
	for (std::size_t sigma : {2, 3}) {
		std::size_t const max_len = sigma == 2 ? 12 : 8;
		for (std::size_t len = 0; len <= max_len; ++len) {
			std::size_t count = 1;
			for (std::size_t i = 0; i < len; ++i) count *= sigma;
			for (std::size_t code = 0; code < count; ++code) {
				std::string s(len, 'a');
				for (std::size_t i = 0, c = code; i < len; ++i, c /= sigma) s[i] = char('a' + c % sigma);
				auto const border = border_lengths(s);
				auto const lcp = common_prefix_lengths(s);
				REQUIRE(to_border_lengths(lcp) == border);
				REQUIRE(to_common_prefix_lengths(border) == lcp);
			}
		}
	}

	std::mt19937_64 rng(20260817);
	for (int round = 0; round < 200; ++round) {
		std::size_t const sigma = 1 + std::size_t(rng() % 3);
		std::string s(1 + std::size_t(rng() % 300), 'a');
		for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
		REQUIRE(to_border_lengths(common_prefix_lengths(s)) == border_lengths(s));
		REQUIRE(to_common_prefix_lengths(border_lengths(s)) == common_prefix_lengths(s));
	}
}

TEST_CASE("matching: edges, projections, and mixed range types", "[strings]") {
	SECTION("empty and oversized patterns") {
		std::string const t{"abc"};
		REQUIRE(occurrences(t, std::string{}, std::vector<std::uint32_t>{}).empty());
		REQUIRE(occurrences(std::string{}, t, border_lengths(t)).empty());
		REQUIRE(occurrences(t, std::string{"abcd"}, border_lengths(std::string{"abcd"})).empty());
		REQUIRE(border_lengths(std::string{}).empty());
		REQUIRE(common_prefix_lengths(std::string{}).empty());
	}
	SECTION("overlapping matches") {
		std::string const t{"aaaaa"}, p{"aaa"};
		REQUIRE(occurrences(t, p, border_lengths(p)) == std::vector<std::uint32_t>{0, 1, 2});
	}
	SECTION("projections and non-char symbols") {
		auto const lower = [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); };
		std::string const t{"AbaBAb"}, p{"ab"}; // lowercased: "ababab"
		REQUIRE(occurrences(t, p, border_lengths(p, lower), lower) == std::vector<std::uint32_t>{0, 2, 4});

		std::vector<int> const nums{7, 3, 7, 3, 7};
		REQUIRE(border_lengths(nums) == std::vector<std::uint32_t>{0, 0, 1, 2, 3});
		REQUIRE(common_prefix_lengths(nums) == std::vector<std::uint32_t>{5, 0, 3, 0, 1});
	}
	SECTION("text and pattern of different range types") {
		std::vector<char> const t{'x', 'y', 'x', 'y'};
		std::string const p{"xy"};
		REQUIRE(occurrences(t, p, border_lengths(p)) == std::vector<std::uint32_t>{0, 2});
	}
}

TEST_CASE("matching: the conversions assert a valid input array", "[strings]") {
	// as a Z-array, z_1 = 5 runs past the end; as a prefix function,
	// b_0 = 3 exceeds the longest proper border of a one-symbol prefix
	std::vector<std::uint32_t> const invalid{3, 5, 0};
	CHECK(atl_test::aborts([&] { (void)atl::to_border_lengths(invalid); }));
	CHECK(atl_test::aborts([&] { (void)atl::to_common_prefix_lengths(invalid); }));
}

TEST_CASE("matching: symbols of different signedness are equal only by value", "[strings]") {
	std::vector<int> const text{-1, 0, -1};
	std::vector<unsigned> const wrapped{(std::numeric_limits<unsigned>::max)()};
	std::vector<unsigned> const zero{0};
	REQUIRE(occurrences(text, wrapped, border_lengths(wrapped)).empty());
	REQUIRE(occurrences(text, zero, border_lengths(zero)) == std::vector<std::uint32_t>{1});
	std::vector<signed char> const bytes{-1, 0, -1};
	std::vector<unsigned char> const high_byte{255};
	REQUIRE(occurrences(bytes, high_byte, border_lengths(high_byte)).empty());
}
