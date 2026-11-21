/** @file
 * Races palindrome-radius construction throughput at 10^6 and 10^7 symbols:
 * unary, alternating, Fibonacci and random texts. The library expands virtual
 * positions; by parity step expands real pairs. Intel Xeon Platinum 8358,
 * GCC 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG, with/without -march=native,
 * 2026-10-15: unary at 10^7 takes 145/145 ms (library), 158/159 ms
 * (by parity step). Full tables: docs/palindrome-lengths.md.
 */
#include <atl/strings/palindrome_lengths.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	/**
	 * @brief The contender: Manacher's build over the same 2n + 1
	 *        virtual centers, expanding one real symbol pair per step (a radius
	 *        has its center's parity), so separator steps cost nothing.
	 *        Returns the same virtual-center radius representation as the library.
	 */
	std::vector<atl::uint32> palindrome_lengths_by_parity_step(std::string const &text) {
		using atl::uint32;
		atl_assert(text.size() < (atl::uint64(1) << 31));
		uint32 const n = uint32(text.size());
		uint32 const m = 2 * n + 1;
		std::vector<uint32> length(m, 0);
		uint32 l = 0, r = 0;
		for (uint32 j = 1; j + 1 < m; ++j) {
			uint32 k = j < r ? (std::min)(length[l + (r - j)], r - j) : 0;
			k += (k ^ j) & 1;
			if (k < j && k < m - j - 1) {
				uint32 a = (j - k - 1) / 2, b = (j + k + 1) / 2;
				while (text[a] == text[b]) {
					k += 2;
					if (a == 0 || b + 1 == n) break;
					--a;
					++b;
				}
			}
			length[j] = k;
			if (j + k > r) {
				l = j - k;
				r = j + k;
			}
		}
		return length;
	}
}

TEST_CASE("palindrome_lengths build", "[palindrome_lengths]") {
	auto const measure = []<typename Contender>(std::string const &label, Contender contender)
		requires std::invocable<Contender &>
	{
		BENCHMARK(std::string(label)) { return std::invoke(contender); };
	};
	std::mt19937_64 rng(1);
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000)}) {
		std::vector<std::pair<std::string, std::string>> texts;
		texts.emplace_back("all-a", std::string(n, 'a'));
		std::string abab(n, 'a');
		for (std::size_t i = 1; i < n; i += 2) abab[i] = 'b';
		texts.emplace_back("abab", abab);
		std::string a = "a", b = "ab";
		while (b.size() < n) {
			std::string next = b + a;
			a = std::move(b);
			b = std::move(next);
		}
		b.resize(n);
		texts.emplace_back("fib", b);
		for (int const sigma : {2, 26}) {
			std::string t(n, 'a');
			for (auto &c : t) c = char('a' + rng() % sigma);
			texts.emplace_back("random" + std::to_string(sigma), t);
		}
		for (auto const &[name, text] : texts) {
			std::string const shape = name + " n=" + std::to_string(n);
			atl::palindrome_lengths const library(text);
			auto const contender = palindrome_lengths_by_parity_step(text);
			for (atl::uint32 c = 0; c < n; ++c) REQUIRE(library.odd_length(c) == contender[2 * c + 1]);
			for (atl::uint32 g = 1; g < n; ++g) REQUIRE(library.even_length(g) == contender[2 * g]);
			measure(shape + " | library", [&] { return atl::palindrome_lengths(text).size(); });
			measure(shape + " | by parity step", [&] { return palindrome_lengths_by_parity_step(text).size(); });
		}
	}
}
