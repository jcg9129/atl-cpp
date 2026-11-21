#include <atl/strings/lyndon.hpp>
#include <atl/strings/matching.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cctype>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using atl::lyndon_factorization;
using atl::minimal_rotation_index;
using atl::runs;

namespace {
	bool is_lyndon(std::string const &w) {
		for (std::size_t d = 1; d < w.size(); ++d)
			if (w.substr(d) <= w) return false;
		return true;
	}

	// smallest period = len - longest proper border
	std::uint32_t min_period(std::string const &s) {
		auto const b = atl::border_lengths(s);
		return std::uint32_t(s.size() - b.back());
	}

	std::vector<atl::run> brute_runs(std::string const &s) {
		std::vector<atl::run> out;
		auto const period_of = [&](std::size_t l, std::size_t r) { return min_period(s.substr(l, r - l)); };
		for (std::size_t l = 0; l < s.size(); ++l)
			for (std::size_t r = l + 2; r <= s.size(); ++r) {
				std::uint32_t const p = period_of(l, r);
				if (r - l < 2 * p) continue;
				if (l > 0 && period_of(l - 1, r) == p) continue;
				if (r < s.size() && period_of(l, r + 1) == p) continue;
				out.push_back({p, std::uint32_t(l), std::uint32_t(r)});
			}
		std::ranges::sort(out);
		return out;
	}

	void check(std::string const &s) {
		// Duval: boundaries split into Lyndon factors, non-increasing —
		// properties that characterize the factorization uniquely
		auto const b = lyndon_factorization(s);
		REQUIRE(b.front() == 0);
		REQUIRE(b.back() == s.size());
		std::string preceding_factor;
		for (std::size_t i = 0; i + 1 < b.size(); ++i) {
			REQUIRE(b[i] < b[i + 1]);
			auto const factor = s.substr(b[i], b[i + 1] - b[i]);
			REQUIRE(is_lyndon(factor));
			if (i > 0) REQUIRE(preceding_factor >= factor);
			preceding_factor = factor;
		}

		// minimal rotation: the rotated string must be the least of all
		if (!s.empty()) {
			auto const at = minimal_rotation_index(s);
			REQUIRE(at < s.size());
			auto const rotate = [&](std::size_t k) { return s.substr(k) + s.substr(0, k); };
			auto const got = rotate(at);
			for (std::size_t k = 0; k < s.size(); ++k) REQUIRE(got <= rotate(k));
		}

		REQUIRE(runs(s) == brute_runs(s));
	}
}

TEST_CASE("lyndon/rotation/runs: exhaustive small strings", "[strings]") {
	for (std::size_t sigma : {1, 2, 3}) {
		std::size_t const max_len = sigma <= 2 ? 13 : 8;
		for (std::size_t len = 0; len <= max_len; ++len) {
			std::size_t count = 1;
			for (std::size_t i = 0; i < len; ++i) count *= sigma;
			for (std::size_t code = 0; code < count; ++code) {
				std::string s(len, 'a');
				for (std::size_t i = 0, c = code; i < len; ++i, c /= sigma) s[i] = char('a' + c % sigma);
				check(s);
			}
		}
	}
}

TEST_CASE("lyndon/rotation/runs: random larger strings", "[strings]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 26);
	std::string s(1 + std::size_t(rng() % 120), 'a');
	for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
	check(s);
}

TEST_CASE("lyndon/rotation/runs: projections and non-char symbols", "[strings]") {
	auto const lower = [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); };
	REQUIRE(lyndon_factorization(std::string{"BAba"}, lower)
			== lyndon_factorization(std::string{"baba"}));
	REQUIRE(minimal_rotation_index(std::string{"CAB"}, lower) == minimal_rotation_index(std::string{"cab"}));

	std::vector<atl::int64> const v{5, -3, 5, -3, 5};
	auto const r = runs(v);
	REQUIRE(r.size() == 1);
	REQUIRE(r[0] == atl::run{2, 0, 5});		 // period 2 over the whole thing
	REQUIRE(minimal_rotation_index(v) == 1); // least rotation starts at a -3

	// integral symbols whose RANGE exceeds uint32: must fall through to
	// compression, not silently truncate (regression for the offset guard)
	std::vector<atl::int64> wide(v.size());
	for (std::size_t i = 0; i < v.size(); ++i) wide[i] = v[i] * 1'000'000'000'000'000'000LL;
	REQUIRE(runs(wide) == r);
	REQUIRE(minimal_rotation_index(wide) == 1);
	REQUIRE(lyndon_factorization(wide) == lyndon_factorization(v));
}

TEST_CASE("lyndon: sparse narrow and full-width integer alphabets", "[strings]") {
	std::vector<atl::uint32> const sparse{0, 3'000'000'000u, 0, 3'000'000'000u};
	REQUIRE(runs(sparse) == std::vector<atl::run>{{2, 0, 4}});
	std::vector<atl::int8> const narrow{-128, 127, -128, 127};
	REQUIRE(runs(narrow) == std::vector<atl::run>{{2, 0, 4}});
	std::vector<atl::int64> const extreme{std::numeric_limits<atl::int64>::lowest(), (std::numeric_limits<atl::int64>::max)(), std::numeric_limits<atl::int64>::lowest()};
	REQUIRE(runs(extreme).empty());
}
