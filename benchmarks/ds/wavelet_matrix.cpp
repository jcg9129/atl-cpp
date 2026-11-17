/**
 * @file
 * @brief Races wavelet construction by two-pass versus one-pass stable partition,
 * rank by split arrays versus interleaved cells, branching versus branchless query descents,
 * and raw values versus rank compression. Measures build and query throughput separately
 * on balanced, query-heavy, build-heavy, small-alphabet, narrow-range, rank-space and k=0 shapes.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): balanced raw build
 * takes 10.0 / 8.4 ms, versus two-pass partition 52.3 / 51.5; docs/wavelet-matrix.md.
 * Prefix count versions of persistent_segment_tree provide the comparison for static order statistics.
 */

#include <atl/ds/persistent_segment_tree.hpp>
#include <atl/ds/wavelet_matrix.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint32;

namespace {
	/** @brief Order statistics by prefix count versions and a lockstep persistent-segment-tree descent. */
	struct kth_by_persistent_segment_tree {
		using pst = atl::persistent_segment_tree<uint32, std::plus<>>;
		std::vector<uint32> sorted;
		std::vector<pst> vs;

		explicit kth_by_persistent_segment_tree(std::vector<uint32> const &a) : sorted(a) {
			std::ranges::sort(sorted);
			sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());
			vs.reserve(a.size() + 1);
			vs.emplace_back(sorted.size(), uint32(0), std::plus<>{});
			vs.front().reserve(a.size());
			for (uint32 const x : a) {
				std::size_t const pos = std::size_t(std::ranges::lower_bound(sorted, x) - sorted.begin());
				vs.push_back(vs.back().update(pos, [](uint32 const &c) { return c + 1; }));
			}
		}

		uint32 kth(std::size_t l, std::size_t r, std::size_t k) const {
			std::size_t const pos = pst::descend(vs[l], vs[r], [k = k](uint32 const &cl, uint32 const &cr) mutable {
				uint32 const c = cr - cl;
				if (k < c) return true;
				k -= c;
				return false;
			});
			return sorted[pos];
		}

		std::size_t count_less(std::size_t l, std::size_t r, uint32 x) const {
			std::size_t const pos = std::size_t(std::ranges::lower_bound(sorted, x) - sorted.begin());
			return vs[r].fold(0, pos) - vs[l].fold(0, pos);
		}

		std::size_t nodes() const { return vs.front().arena_size(); }

		/** @brief Arena nodes (aggregate + two child ids each) plus the version handles and the sorted universe. */
		std::size_t estimated_bytes() const {
			return nodes() * (sizeof(uint32) * 3) + vs.capacity() * sizeof(pst) + sorted.capacity() * sizeof(uint32);
		}
	};

	struct scenario {
		char const *name;
		std::size_t n, q;
		uint32 vmax;	  // values drawn from [0, vmax]
		std::size_t span; // 0 = unrestricted range width
	};

	void run_scenario(scenario const &sc) {
		std::mt19937 rng(7);
		std::vector<uint32> a(sc.n);
		for (auto &v : a) v = uint32(rng() % (sc.vmax + 1));

		struct qq {
			std::size_t l, r, k;
			uint32 x;
		};

		std::vector<qq> qs(sc.q);
		for (auto &[l, r, k, x] : qs) {
			l = rng() % sc.n;
			std::size_t const hi = sc.span == 0 ? sc.n : std::min(sc.n, l + sc.span);
			r = l + 1 + rng() % (hi - l);
			k = rng() % (r - l);
			x = uint32(rng() % (sc.vmax + 1));
		}

		atl::wavelet_matrix wm(a);
		kth_by_persistent_segment_tree ch(a);
		// Wavelet: per level, packed words (one sentinel word) plus 32-bit rank prefixes.
		std::size_t const wm_levels = sc.vmax == 0 ? 1 : std::size_t(std::bit_width(sc.vmax));
		std::size_t const wm_bytes = wm_levels * ((sc.n / 64 + 2) * 8 + (sc.n / 64 + 3) * 4);
		std::size_t const ch_bytes = ch.estimated_bytes();
		std::printf("[%s] estimated storage (payload only, no allocator overhead): wavelet %.1f MB, by persistent segment tree %.1f MB (%.2fx)\n", sc.name,
					double(wm_bytes) / 1e6, double(ch_bytes) / 1e6, double(ch_bytes) / double(wm_bytes));
		for (auto const &[l, r, k, x] : qs) {
			REQUIRE(ch.kth(l, r, k) == wm.kth(l, r, k));
			REQUIRE(ch.count_less(l, r, x) == wm.count_less(l, r, x));
		}
		BENCHMARK(std::string(sc.name) + " | build by persistent segment tree") { return kth_by_persistent_segment_tree(a).nodes(); };
		BENCHMARK(std::string(sc.name) + " | kth by persistent segment tree") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs) acc += ch.kth(l, r, k);
			return acc;
		};
		BENCHMARK(std::string(sc.name) + " | count_less by persistent segment tree") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs) acc += ch.count_less(l, r, x);
			return acc;
		};
		BENCHMARK(std::string(sc.name) + " | build library") { return atl::wavelet_matrix(a).size(); };
		BENCHMARK(std::string(sc.name) + " | kth library") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs) acc += wm.kth(l, r, k);
			return acc;
		};
		BENCHMARK(std::string(sc.name) + " | count_less library") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs) acc += wm.count_less(l, r, x);
			return acc;
		};

		// Rank space needs only log(distinct) levels instead of log(max).
		std::vector<uint32> sorted(a);
		std::ranges::sort(sorted);
		sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());
		std::vector<uint32> ranks(sc.n);
		for (std::size_t i = 0; i < sc.n; ++i)
			ranks[i] = uint32(std::ranges::lower_bound(sorted, a[i]) - sorted.begin());
		atl::wavelet_matrix wmc(ranks);
		for (auto const &[l, r, k, x] : qs) {
			REQUIRE(sorted[wmc.kth(l, r, k)] == wm.kth(l, r, k));
			REQUIRE(wmc.count_less(l, r, uint32(std::ranges::lower_bound(sorted, x) - sorted.begin())) == wm.count_less(l, r, x));
		}

		BENCHMARK(std::string(sc.name) + " | kth by rank compression") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs) acc += sorted[wmc.kth(l, r, k)];
			return acc;
		};
		BENCHMARK(std::string(sc.name) + " | count_less by rank compression") {
			atl::uint64 acc = 0;
			for (auto const &[l, r, k, x] : qs)
				acc += wmc.count_less(l, r, uint32(std::ranges::lower_bound(sorted, x) - sorted.begin()));
			return acc;
		};
	}
}

TEST_CASE("wavelet_matrix raw values and rank compression", "[wavelet_matrix]") {
	SECTION("balanced range queries") { run_scenario({"balanced n=q=2e5", 200'000, 200'000, 1'000'000'000, 0}); }
	SECTION("query-heavy") { run_scenario({"query-heavy n=5e4 q=1e6", 50'000, 1'000'000, 1'000'000'000, 0}); }
	SECTION("build-heavy") { run_scenario({"build-heavy n=1e6 q=1e4", 1'000'000, 10'000, 1'000'000'000, 0}); }
	SECTION("small alphabet") { run_scenario({"alphabet<256 n=q=2e5", 200'000, 200'000, 255, 0}); }
	SECTION("narrow ranges") { run_scenario({"narrow r-l<=64 n=q=2e5", 200'000, 200'000, 1'000'000'000, 64}); }
}

namespace race {
	/** @brief Rank by split arrays keeps 64-bit words and their prefix ranks in separate vectors. */
	class rank_by_split_arrays {
		std::vector<atl::uint64> words_;
		std::vector<uint32> rank_;

	public:
		rank_by_split_arrays() = default;

		explicit rank_by_split_arrays(std::size_t n) : words_(n / 64 + (n % 64 != 0) + 1, 0) {}

		void set(std::size_t i) { words_[i / 64] |= atl::uint64(1) << (i % 64); }

		void set_word(std::size_t w, atl::uint64 bits) { words_[w] = bits; }

		void finish() {
			rank_.resize(words_.size() + 1);
			uint32 acc = 0;
			for (std::size_t w = 0; w < words_.size(); ++w) {
				rank_[w] = acc;
				acc += uint32(std::popcount(words_[w]));
			}
			rank_[words_.size()] = acc;
		}

		bool at(std::size_t i) const { return (words_[i / 64] >> (i % 64)) & 1; }

		uint32 rank1(std::size_t i) const {
			return rank_[i / 64] + uint32(std::popcount(words_[i / 64] & ((atl::uint64(1) << (i % 64)) - 1)));
		}

		uint32 rank0(std::size_t i) const { return uint32(i) - rank1(i); }
	};

	/** @brief Rank by interleaved cells puts each word beside its prefix rank in a 16-byte cell. */
	class rank_by_interleaved_cells {
		struct cell {
			atl::uint64 word;
			uint32 rank;
		};

		std::vector<cell> cells_;

	public:
		rank_by_interleaved_cells() = default;

		explicit rank_by_interleaved_cells(std::size_t n) : cells_(n / 64 + (n % 64 != 0) + 1, cell{0, 0}) {}

		void set(std::size_t i) { cells_[i / 64].word |= atl::uint64(1) << (i % 64); }

		void set_word(std::size_t w, atl::uint64 bits) { cells_[w].word = bits; }

		void finish() {
			uint32 acc = 0;
			for (auto &c : cells_) c.rank = acc, acc += uint32(std::popcount(c.word));
		}

		bool at(std::size_t i) const { return (cells_[i / 64].word >> (i % 64)) & 1; }

		uint32 rank1(std::size_t i) const {
			cell const &c = cells_[i / 64];
			return c.rank + uint32(std::popcount(c.word & ((atl::uint64(1) << (i % 64)) - 1)));
		}

		uint32 rank0(std::size_t i) const { return uint32(i) - rank1(i); }
	};

	/**
	 * @brief Wavelet matrix over uint32 with switchable build and query forms:
	 *        @p UsesBranchlessBuild selects one stable-partition pass, words assembled in a
	 *        register; @p UsesBranchlessQuery selects instead of branching per level.
	 */
	template<class Bits, bool UsesBranchlessBuild, bool UsesBranchlessQuery>
		requires std::constructible_from<Bits, std::size_t> && requires(Bits &bits, std::size_t i, atl::uint64 word) { bits.set(i); bits.set_word(i, word); bits.finish(); { bits.rank0(i) } -> std::convertible_to<uint32>; }
	class wavelet_matrix_by_partition_and_rank {
		std::vector<Bits> levels_;
		std::vector<uint32> zeros_;

	public:
		explicit wavelet_matrix_by_partition_and_rank(std::vector<uint32> cur) {
			std::size_t const n = cur.size();
			uint32 mx = 0;
			for (uint32 const v : cur) mx = std::max(mx, v);
			int const width = mx == 0 ? 1 : std::bit_width(mx);
			levels_.resize(std::size_t(width));
			zeros_.resize(std::size_t(width));
			std::vector<uint32> next(n), ones(UsesBranchlessBuild ? n : 0);
			for (int lvl = width - 1; lvl >= 0; --lvl) {
				Bits &bv = levels_[std::size_t(lvl)] = Bits(n);
				std::size_t lo = 0;
				if constexpr (UsesBranchlessBuild) {
					std::size_t hi = 0;
					atl::uint64 word = 0;
					for (std::size_t i = 0; i < n; ++i) {
						uint32 const v = cur[i];
						atl::uint64 const bit = v >> lvl & 1;
						next[lo] = v;
						ones[hi] = v;
						lo += std::size_t(bit ^ 1);
						hi += std::size_t(bit);
						word |= bit << (i % 64);
						if (i % 64 == 63) bv.set_word(i / 64, word), word = 0;
					}
					if (n % 64) bv.set_word(n / 64, word);
					std::ranges::copy(ones.begin(), ones.begin() + std::ptrdiff_t(hi), next.begin() + std::ptrdiff_t(lo));
				} else {
					for (std::size_t i = 0; i < n; ++i)
						if ((cur[i] >> lvl & 1) == 0)
							next[lo++] = cur[i];
						else
							bv.set(i);
				}
				bv.finish();
				zeros_[std::size_t(lvl)] = uint32(lo);
				if constexpr (!UsesBranchlessBuild)
					for (std::size_t i = 0; i < n; ++i)
						if (cur[i] >> lvl & 1) next[lo++] = cur[i];
				cur.swap(next);
			}
		}

		uint32 kth(std::size_t l, std::size_t r, std::size_t k) const {
			uint32 v = 0;
			for (std::size_t lvl = levels_.size(); lvl-- > 0;) {
				Bits const &bv = levels_[lvl];
				uint32 const l0 = bv.rank0(l), r0 = bv.rank0(r), z = r0 - l0;
				if constexpr (UsesBranchlessQuery) {
					bool const one = k >= z;
					k -= one ? z : 0;
					v |= uint32(one) << lvl;
					l = one ? zeros_[lvl] + (uint32(l) - l0) : l0;
					r = one ? zeros_[lvl] + (uint32(r) - r0) : r0;
				} else if (k < z) {
					l = l0;
					r = r0;
				} else {
					k -= z;
					v |= uint32(1) << lvl;
					l = zeros_[lvl] + (uint32(l) - l0);
					r = zeros_[lvl] + (uint32(r) - r0);
				}
			}
			return v;
		}

		std::size_t count_less(std::size_t l, std::size_t r, uint32 upper) const {
			if (levels_.size() < std::size_t(std::bit_width(upper))) return r - l;
			std::size_t res = 0;
			for (std::size_t lvl = levels_.size(); lvl-- > 0;) {
				Bits const &bv = levels_[lvl];
				uint32 const l0 = bv.rank0(l), r0 = bv.rank0(r);
				if constexpr (UsesBranchlessQuery) {
					bool const one = upper >> lvl & 1;
					res += one ? r0 - l0 : 0;
					l = one ? zeros_[lvl] + (uint32(l) - l0) : l0;
					r = one ? zeros_[lvl] + (uint32(r) - r0) : r0;
				} else if (upper >> lvl & 1) {
					res += r0 - l0;
					l = zeros_[lvl] + (uint32(l) - l0);
					r = zeros_[lvl] + (uint32(r) - r0);
				} else {
					l = l0;
					r = r0;
				}
			}
			return res;
		}
	};

	/** @brief Two partition passes and branching queries. */
	using wavelet_matrix_by_two_pass_partition = wavelet_matrix_by_partition_and_rank<rank_by_split_arrays, false, false>;
	/** @brief One partition pass and branching queries. */
	using wavelet_matrix_by_one_pass_partition = wavelet_matrix_by_partition_and_rank<rank_by_split_arrays, true, false>;
	/** @brief One partition pass with branchless rank-based queries. */
	using wavelet_matrix_by_branchless_queries = wavelet_matrix_by_partition_and_rank<rank_by_split_arrays, true, true>;
	/** @brief Branchless queries and interleaved rank cells. */
	using wavelet_matrix_by_branchless_queries_and_cells = wavelet_matrix_by_partition_and_rank<rank_by_interleaved_cells, true, true>;
	/** @brief Branching queries and interleaved rank cells. */
	using wavelet_matrix_by_interleaved_cells = wavelet_matrix_by_partition_and_rank<rank_by_interleaved_cells, true, false>;

	struct qq {
		std::size_t l, r, k;
		uint32 x;
	};

	template<class Query>
		requires std::is_invocable_r_v<uint32, Query &, std::size_t, std::size_t, std::size_t>
	atl::uint64 kth_stream(Query query, std::vector<qq> const &qs) {
		atl::uint64 acc = 0;
		for (auto const &q : qs) acc += query(q.l, q.r, q.k);
		return acc;
	}

	template<class Query>
		requires std::is_invocable_r_v<std::size_t, Query &, std::size_t, std::size_t, uint32>
	atl::uint64 count_stream(Query query, std::vector<qq> const &qs) {
		atl::uint64 acc = 0;
		for (auto const &q : qs) acc += query(q.l, q.r, q.x);
		return acc;
	}

	/** @brief Every contender on one shape; checks they agree with the library matrix first. */
	void race_shape(char const *name, std::size_t n, std::size_t q, uint32 vmax, bool is_minimum_query) {
		std::mt19937 rng(9);
		std::vector<uint32> a(n);
		for (auto &v : a) v = uint32(rng() % (vmax + 1));
		std::vector<qq> qs(q);
		for (std::size_t i = 0; i < q; ++i) {
			auto &[l, r, k, x] = qs[i];
			l = rng() % n;
			r = l + 1 + rng() % (n - l);
			k = is_minimum_query ? 0 : rng() % (r - l); // is_minimum_query: always the minimum, predictable branches
			x = uint32(rng() % (vmax + 1));
		}
		atl::wavelet_matrix const library(a);
		wavelet_matrix_by_two_pass_partition const w0(a);
		wavelet_matrix_by_one_pass_partition const w1(a);
		wavelet_matrix_by_branchless_queries const w2(a);
		wavelet_matrix_by_branchless_queries_and_cells const w3(a);
		wavelet_matrix_by_interleaved_cells const w4(a);
		for (std::size_t i = 0; i < std::min<std::size_t>(q, 20000); ++i) {
			auto const &[l, r, k, x] = qs[i];
			uint32 const want = library.kth(l, r, k);
			std::size_t const want_c = library.count_less(l, r, x);
			if (w0.kth(l, r, k) != want || w1.kth(l, r, k) != want || w2.kth(l, r, k) != want || w3.kth(l, r, k) != want || w4.kth(l, r, k) != want || w0.count_less(l, r, x) != want_c || w1.count_less(l, r, x) != want_c || w2.count_less(l, r, x) != want_c || w3.count_less(l, r, x) != want_c || w4.count_less(l, r, x) != want_c)
				FAIL("wavelet contenders disagree with the library matrix");
		}
		std::string const s = std::string("race ") + name + " | ";
		BENCHMARK(s + "build library") { return atl::wavelet_matrix(a).size(); };
		BENCHMARK(s + "build by two pass partition") { return wavelet_matrix_by_two_pass_partition(a).kth(0, 1, 0); };
		BENCHMARK(s + "build by one pass partition") { return wavelet_matrix_by_one_pass_partition(a).kth(0, 1, 0); };
		BENCHMARK(s + "build by one pass partition and interleaved cells") { return wavelet_matrix_by_branchless_queries_and_cells(a).kth(0, 1, 0); };
		BENCHMARK(s + "kth library") {
			return kth_stream([&](std::size_t l, std::size_t r, std::size_t k) { return library.kth(l, r, k); }, qs);
		};
		BENCHMARK(s + "kth by two pass partition") {
			return kth_stream([&](std::size_t l, std::size_t r, std::size_t k) { return w0.kth(l, r, k); }, qs);
		};
		BENCHMARK(s + "kth by branchless queries") {
			return kth_stream([&](std::size_t l, std::size_t r, std::size_t k) { return w2.kth(l, r, k); }, qs);
		};
		BENCHMARK(s + "kth by interleaved cells") {
			return kth_stream([&](std::size_t l, std::size_t r, std::size_t k) { return w4.kth(l, r, k); }, qs);
		};
		BENCHMARK(s + "kth by branchless queries and cells") {
			return kth_stream([&](std::size_t l, std::size_t r, std::size_t k) { return w3.kth(l, r, k); }, qs);
		};
		BENCHMARK(s + "count_less library") {
			return count_stream([&](std::size_t l, std::size_t r, uint32 x) { return library.count_less(l, r, x); }, qs);
		};
		BENCHMARK(s + "count_less by two pass partition") {
			return count_stream([&](std::size_t l, std::size_t r, uint32 x) { return w0.count_less(l, r, x); }, qs);
		};
		BENCHMARK(s + "count_less by branchless queries") {
			return count_stream([&](std::size_t l, std::size_t r, uint32 x) { return w2.count_less(l, r, x); }, qs);
		};
		BENCHMARK(s + "count_less by interleaved cells") {
			return count_stream([&](std::size_t l, std::size_t r, uint32 x) { return w4.count_less(l, r, x); }, qs);
		};
		BENCHMARK(s + "count_less by branchless queries and cells") {
			return count_stream([&](std::size_t l, std::size_t r, uint32 x) { return w3.count_less(l, r, x); }, qs);
		};
	}
}

TEST_CASE("wavelet_matrix build, rank layout and descent comparisons", "[wavelet_matrix][race]") {
	race::race_shape("n=2e5 q=2e5 30-bit", 200'000, 200'000, 1'000'000'000, false);
	race::race_shape("n=1e6 q=2e5 30-bit", 1'000'000, 200'000, 1'000'000'000, false);
	race::race_shape("n=2e5 q=2e5 rank-space 2e5", 200'000, 200'000, 199'999, false);
	race::race_shape("n=2e5 q=2e5 30-bit, k=0", 200'000, 200'000, 1'000'000'000, true);
}
