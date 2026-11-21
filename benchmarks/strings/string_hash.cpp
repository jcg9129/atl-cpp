/** @file
 * Measures rolling-hash throughput: one 10^6-symbol build, 10^5 ten-symbol
 * builds, a forward/reverse pair and 10^6 substring queries. No contender is
 * raced; the explicit base excludes seed initialization. Presized indexed
 * writes did not win consistently across construction shapes; push_back is
 * retained. The supplied table contains only library timings. Intel Xeon Platinum
 * 8358, GCC 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG, with/without
 * -march=native, 2026-10-15: the long build takes 4.8/5.0 ms and substring
 * queries take 14.4/12.5 ms. Full tables: docs/string-hash.md.
 */
#include <atl/strings/string_hash.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <format>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::uint64;

TEST_CASE("string_hash benchmarks", "[string_hash]") {
	std::mt19937_64 rng(1);
	constexpr std::size_t n = 1'000'000, queries = 1'000'000;
	constexpr uint64 base = 131;
	std::string s(n, 'a');
	for (auto &c : s) c = char('a' + rng() % 26);
	std::string const reversed(s.rbegin(), s.rend());
	std::vector<std::string> words(100'000, std::string(10, 'a'));
	for (auto &w : words)
		for (auto &c : w) c = char('a' + rng() % 26);
	std::vector<std::pair<std::size_t, std::size_t>> ranges(queries);
	for (auto &[l, r] : ranges) {
		l = std::size_t(rng() % n);
		r = l + std::size_t(rng() % (n - l + 1));
	}

	BENCHMARK("build n=1e6") { return atl::string_hash(s, base).size(); };
	BENCHMARK("build 1e5 instances x10") {
		uint64 acc = 0;
		for (auto const &w : words) acc += atl::string_hash(w, base).substring_hash(0, 10);
		return acc;
	};
	BENCHMARK("build forward+reverse n=1e6") {
		return atl::string_hash(s, base).size() + atl::string_hash(reversed, base).size();
	};
	atl::string_hash const h(s, base);
	BENCHMARK(std::format("{} substring hashes", queries)) {
		uint64 acc = 0;
		for (auto [l, r] : ranges) acc ^= h.substring_hash(l, r);
		return acc;
	};
}
