#include <atl/bits/xor_basis.hpp>
#include <atl/experimental/bits/xor_basis_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <tuple>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using atl::uint8;
namespace ex = atl::experimental;

namespace {
	template<typename Word>
	concept accepts_basis_word = requires { typename atl::xor_basis<Word>; };

	template<typename Word>
	concept accepts_stamped_word = requires { typename atl::stamped_xor_basis<Word>; };

	template<typename Word>
	concept accepts_echelon_word = requires { typename ex::echelon_xor_basis<Word>; };

	template<typename Word>
	concept accepts_scanning_word = requires { typename ex::scanning_xor_basis<Word>; };

	template<typename Word>
	concept accepts_all_basis_words = accepts_basis_word<Word> && accepts_stamped_word<Word>
								   && accepts_echelon_word<Word> && accepts_scanning_word<Word>;

	template<typename Word>
	concept rejects_all_basis_words = !accepts_basis_word<Word> && !accepts_stamped_word<Word>
								   && !accepts_echelon_word<Word> && !accepts_scanning_word<Word>;

	static_assert(accepts_all_basis_words<uint32>);
	static_assert(accepts_all_basis_words<atl::uint128>);
	static_assert(rejects_all_basis_words<uint32 const>);
	static_assert(rejects_all_basis_words<uint32 volatile>);
	static_assert(rejects_all_basis_words<uint32 const volatile>);
	static_assert(rejects_all_basis_words<atl::uint128 const>);
	static_assert(rejects_all_basis_words<atl::uint128 volatile>);
	static_assert(rejects_all_basis_words<atl::uint128 const volatile>);

	/** @brief The span of a word list, by closure: every subset XOR. */
	template<atl::unsigned_integral Word>
	std::vector<Word> brute_span(std::vector<Word> const &words) {
		std::set<Word> span{Word(0)};
		for (Word const w : words) {
			if (span.contains(w)) continue;
			std::vector<Word> const old(span.begin(), span.end());
			for (Word const v : old) span.insert(Word(v ^ w));
		}
		return {span.begin(), span.end()};
	}

	/** @brief Random words drawn from a random subspace of dimension <= 10,
	 *         so the brute span stays small at any width. */
	template<atl::unsigned_integral Word>
	std::vector<Word> subspace_words(std::mt19937_64 &rng, std::size_t count) {
		std::size_t const dim = std::size_t(rng() % 11);
		std::vector<Word> gens(dim);
		for (auto &g : gens) {
			Word x = 0;
			x = Word(rng());
			if constexpr (sizeof(Word) > 8) x = Word((x << 64) | Word(rng()));
			// sparse generators sometimes, so pivots collide and reduce
			if (rng() % 3 == 0) x &= Word(rng()) & Word(rng());
			g = x;
		}
		std::vector<Word> words(count);
		for (auto &w : words) {
			w = 0;
			for (Word const g : gens)
				if (rng() & 1) w ^= g;
		}
		return words;
	}

	template<atl::unsigned_integral Word>
	void check_reduced_form(atl::xor_basis<Word> const &basis) {
		Word const mask = basis.pivot_mask();
		std::size_t count = 0;
		for (std::size_t b = 0; b < basis.bits; ++b) {
			Word const v = basis.vector(b);
			bool const is_pivot = bool((mask >> b) & 1);
			REQUIRE((v != 0) == is_pivot);
			if (!is_pivot) continue;
			++count;
			REQUIRE(atl::detail::top_bit_index(v) == b);
			REQUIRE((v & mask) == Word(Word(1) << b));
		}
		REQUIRE(count == basis.rank());
	}

	template<atl::unsigned_integral Word>
	void oracle_round(std::mt19937_64 &rng) {
		auto const words = subspace_words<Word>(rng, std::size_t(rng() % 40));
		atl::xor_basis<Word> basis;
		std::vector<Word> independent, span_so_far{Word(0)}; // recomputed only when the span grows
		for (Word const w : words) {
			bool const was_in = std::ranges::binary_search(span_so_far, w);
			REQUIRE(basis.insert(w) == !was_in);
			if (!was_in) independent.push_back(w), span_so_far = brute_span(independent);
		}
		check_reduced_form(basis);
		auto const span = brute_span(words);
		REQUIRE(span.size() == (std::size_t(1) << basis.rank()));
		REQUIRE(basis.representable_count() == span.size());
		REQUIRE(basis.empty() == (span.size() == 1));
		for (std::size_t k = 0; k < span.size(); ++k) REQUIRE(basis.kth_smallest(Word(k)) == span[k]);
		REQUIRE(basis.max_xor() == span.back());
		if (span.size() > 1) REQUIRE(basis.min_nonzero() == span[1]);
		for (Word const v : span) REQUIRE(basis.contains(v));
		for (int probe = 0; probe < 30; ++probe) {
			Word start = probe < 15 ? Word(rng()) : Word(span[std::size_t(rng() % span.size())] ^ Word(rng() % 16));
			if constexpr (sizeof(Word) > 8) start = Word((start << 64) | Word(rng()));
			Word best = 0, worst = (std::numeric_limits<Word>::max)();
			for (Word const v : span) best = (std::max)(best, Word(start ^ v)), worst = (std::min)(worst, Word(start ^ v));
			REQUIRE(basis.max_xor(start) == best);
			REQUIRE(basis.min_xor(start) == worst);
			REQUIRE(basis.reduce(start) == worst);
			REQUIRE(basis.contains(start) == std::ranges::binary_search(span, start));
			auto const below = std::size_t(std::ranges::lower_bound(span, start) - span.begin());
			REQUIRE(basis.count_below(start) == Word(below));
			REQUIRE(ex::greedy_count_below(basis, start) == Word(below));
		}
		REQUIRE(basis.count_below(0) == 0);
		REQUIRE(basis.count_below((std::numeric_limits<Word>::max)()) == Word(span.size() - (span.back() == (std::numeric_limits<Word>::max)())));

		ex::echelon_xor_basis<Word> echelon;
		ex::scanning_xor_basis<Word> scanning;
		for (Word const w : words) {
			bool const grew = echelon.insert(w);
			REQUIRE(scanning.insert(w) == grew);
		}
		REQUIRE(echelon.rank() == basis.rank());
		REQUIRE(echelon.reduced() == basis);
		REQUIRE(scanning.reduced() == basis);
		for (int probe = 0; probe < 10; ++probe) {
			Word const x = probe % 2 ? Word(rng()) : span[std::size_t(rng() % span.size())];
			REQUIRE(echelon.contains(x) == basis.contains(x));
			REQUIRE(scanning.contains(x) == basis.contains(x));
			REQUIRE(echelon.max_xor(x) == basis.max_xor(x));
			REQUIRE(scanning.max_xor(x) == basis.max_xor(x));
		}
	}
}

namespace {
	using basis_words = std::tuple<uint8, uint32, uint64, atl::uint128>;
}

TEMPLATE_LIST_TEST_CASE("xor_basis: every query against the brute-force span", "[xor_basis]", basis_words) {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 4; ++round) oracle_round<TestType>(rng);
}

TEST_CASE("xor_basis: exhaustive spans of 8-bit words", "[xor_basis]") {
	for (uint32 a = 0; a < 256; a += 3)
		for (uint32 b = 0; b < 256; b += 5) {
			std::vector<uint8> const words{uint8(a), uint8(b), uint8(a * 7 + b)};
			atl::xor_basis<uint8> const basis(words);
			auto const span = brute_span(words);
			REQUIRE(span.size() == (std::size_t(1) << basis.rank()));
			for (std::size_t k = 0; k < span.size(); ++k) REQUIRE(basis.kth_smallest(uint8(k)) == span[k]);
			for (uint32 x = 0; x < 256; x += 17) {
				REQUIRE(basis.contains(uint8(x)) == std::ranges::binary_search(span, uint8(x)));
				REQUIRE(basis.count_below(uint8(x)) == std::size_t(std::ranges::lower_bound(span, uint8(x)) - span.begin()));
			}
		}
}

TEST_CASE("xor_basis: full rank, merge, equality and edge cases", "[xor_basis]") {
	atl::xor_basis<uint64> basis;
	REQUIRE(basis.empty());
	REQUIRE(basis.rank() == 0);
	REQUIRE(basis.max_xor() == 0);
	REQUIRE(basis.max_xor(5) == 5);
	REQUIRE(basis.kth_smallest(0) == 0);
	REQUIRE(basis.count_below(1) == 1);
	REQUIRE(basis.representable_count() == 1);
	REQUIRE_FALSE(basis.insert(0));
	REQUIRE(basis.contains(0));
	REQUIRE_FALSE(basis.contains(1));

	// full rank: every 64-bit value representable, k-th smallest is k
	for (std::size_t b = 0; b < 64; ++b) REQUIRE(basis.insert((uint64(1) << b) | (uint64(1) << 63)));
	REQUIRE(basis.rank() == 64);
	REQUIRE(basis.pivot_mask() == ~uint64(0));
	REQUIRE(basis.max_xor() == ~uint64(0));
	REQUIRE(basis.max_xor(12345) == ~uint64(0));
	REQUIRE(basis.min_nonzero() == 1);
	REQUIRE(basis.kth_smallest(987654321) == 987654321);
	REQUIRE(basis.kth_smallest(~uint64(0)) == ~uint64(0));
	REQUIRE(basis.count_below(~uint64(0)) == ~uint64(0));
	REQUIRE_FALSE(basis.insert(42));

	std::mt19937_64 rng(7);
	for (int round = 0; round < 200; ++round) {
		auto const a = subspace_words<uint32>(rng, std::size_t(rng() % 12)), b = subspace_words<uint32>(rng, std::size_t(rng() % 12));
		atl::xor_basis<uint32> left(a), right(b);
		auto both = a;
		both.insert(both.end(), b.begin(), b.end());
		left.merge(right);
		REQUIRE(left == atl::xor_basis<uint32>(both));
		auto shuffled = both;
		std::ranges::shuffle(shuffled, rng);
		REQUIRE(atl::xor_basis(shuffled) == left);
	}
	atl::xor_basis<uint32> cleared(std::vector<uint32>{3, 5});
	cleared.clear();
	REQUIRE(cleared == atl::xor_basis<uint32>());
}

TEST_CASE("xor_basis: preconditions", "[xor_basis]") {
	atl::xor_basis<uint32> basis;
	CHECK(atl_test::aborts([&] { (void)basis.min_nonzero(); }));
	CHECK(atl_test::aborts([&] { (void)basis.kth_smallest(1); }));
	CHECK(atl_test::aborts([&] { (void)basis.vector(32); }));
	basis.insert(6);
	CHECK(atl_test::aborts([&] { (void)basis.kth_smallest(2); }));
	REQUIRE(basis.kth_smallest(1) == 6);
	atl::xor_basis<uint64> full;
	for (std::size_t b = 0; b < 64; ++b) full.insert(uint64(1) << b);
	CHECK(atl_test::aborts([&] { (void)full.representable_count(); }));
}

namespace {
	/** @brief The literal update of the 2025 paper's section 3.4 ("XOR a_i
	 *         onto w_j"), kept to pin the erratum: it must be w_j onto a_i. */
	struct literal_paper_prefix_basis {
		std::array<uint32, 8> w{};
		std::array<int, 8> p{};

		literal_paper_prefix_basis() { p.fill((std::numeric_limits<int>::min)()); }

		void insert(uint32 a, int t) {
			for (int j = 7; j >= 0; --j) {
				if (!((a >> j) & 1)) continue;
				if (p[std::size_t(j)] < t) std::swap(w[std::size_t(j)], a), std::swap(p[std::size_t(j)], t);
				w[std::size_t(j)] ^= a;
			}
		}

		std::vector<uint32> span_from(int from) const {
			std::vector<uint32> words;
			for (std::size_t j = 0; j < 8; ++j)
				if (w[j] != 0 && p[j] >= from) words.push_back(w[j]);
			return brute_span(words);
		}
	};
}

TEST_CASE("stamped_xor_basis: array ranges against the brute force", "[xor_basis]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 30);
	auto const a = subspace_words<uint32>(rng, n);
	atl::stamped_xor_basis<uint32> basis;
	for (std::size_t r = 1; r <= n; ++r) {
		std::vector<uint32> prefix(a.begin(), a.begin() + std::ptrdiff_t(r - 1));
		bool const grew = brute_span(prefix).size() != brute_span(std::vector<uint32>(a.begin(), a.begin() + std::ptrdiff_t(r))).size();
		REQUIRE(basis.insert(a[r - 1], uint32(r - 1)) == grew);
		for (std::size_t l = 0; l <= r; ++l) {
			std::vector<uint32> const range(a.begin() + std::ptrdiff_t(l), a.begin() + std::ptrdiff_t(r));
			auto const span = brute_span(range);
			REQUIRE((std::size_t(1) << basis.rank(uint32(l))) == span.size());
			REQUIRE(basis.max_xor(uint32(l)) == span.back());
			REQUIRE(basis.basis_from(uint32(l)) == atl::xor_basis<uint32>(range));
			for (int probe = 0; probe < 8; ++probe) {
				uint32 const x = probe % 2 ? uint32(rng()) : span[std::size_t(rng() % span.size())];
				REQUIRE(basis.contains(x, uint32(l)) == std::ranges::binary_search(span, x));
				uint32 best = 0;
				for (uint32 const v : span) best = (std::max)(best, x ^ v);
				REQUIRE(basis.max_xor(uint32(l), x) == best);
			}
		}
	}
	REQUIRE(basis.rank() == basis.rank(0));
}

TEST_CASE("stamped_xor_basis: arbitrary stamp order (offline deletion)", "[xor_basis]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(rng() % 25);
	auto const words = subspace_words<uint64>(rng, n);
	std::vector<int> stamps(n);
	for (auto &s : stamps) s = int(rng() % 12) - 6;
	atl::stamped_xor_basis<uint64, int> basis;
	for (std::size_t i = 0; i < n; ++i) basis.insert(words[i], stamps[i]);
	for (int from = -7; from <= 7; ++from) {
		std::vector<uint64> chosen;
		for (std::size_t i = 0; i < n; ++i)
			if (stamps[i] >= from) chosen.push_back(words[i]);
		atl::xor_basis<uint64> const want(chosen);
		REQUIRE(basis.basis_from(from) == want);
		REQUIRE(basis.rank(from) == want.rank());
		REQUIRE(basis.max_xor(from) == want.max_xor());
		for (int probe = 0; probe < 6; ++probe) {
			uint64 const x = probe % 2 ? rng() : (chosen.empty() ? 0 : chosen[std::size_t(rng() % chosen.size())]);
			REQUIRE(basis.contains(x, from) == want.contains(x));
			REQUIRE(basis.max_xor(from, x) == want.max_xor(x));
		}
	}
	for (std::size_t b = 0; b < 64; ++b)
		if (basis.vector(b) != 0) REQUIRE(atl::detail::top_bit_index(basis.vector(b)) == b);
	CHECK(atl_test::aborts([&] { (void)basis.stamp(64); }));
}

TEST_CASE("stamped_xor_basis: the paper's literal section 3.4 update is wrong", "[xor_basis]") {
	std::mt19937_64 rng(11);
	std::size_t wrong = 0;
	for (int round = 0; round < 200; ++round) {
		std::size_t const n = 1 + std::size_t(rng() % 10);
		std::vector<uint32> a(n);
		for (auto &x : a) x = uint32(rng() % 256);
		literal_paper_prefix_basis paper;
		atl::stamped_xor_basis<uint32, int> fixed;
		for (std::size_t i = 0; i < n; ++i) paper.insert(a[i], int(i)), fixed.insert(a[i], int(i));
		for (std::size_t l = 0; l < n; ++l) {
			std::vector<uint32> const range(a.begin() + std::ptrdiff_t(l), a.end());
			auto const span = brute_span(range);
			REQUIRE((std::size_t(1) << fixed.rank(int(l))) == span.size());
			wrong += paper.span_from(int(l)) != span;
		}
	}
	REQUIRE(wrong > 0);
}
