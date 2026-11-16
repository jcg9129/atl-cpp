/**
 * @file
 * Race between `atl::hierarchical_bitset` (the library, heap offsets and early
 * summary propagation), `hierarchical_bitset_by_full_propagation` (writes every
 * summary level), `hierarchical_bitset_by_inline_offsets` (stores offsets inline),
 * and `std::set` (ordered tree). Loops measure throughput of one million random
 * inserts, insert/erase pairs, and next/prev queries in dense (one million) and
 * sparse (2^26) universes, including construction for mutation loops.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): dense insertion takes 5.2/5.2 ms for the library, 7.9/6.0 ms
 * by full propagation, and 4.9/4.7 ms by inline offsets; query results vary
 * by shape and flags. Numbers: docs/hierarchical-bitset.md.
 */
#include <atl/ds/hierarchical_bitset.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

namespace {
	/** @brief hierarchical_bitset variants: offsets inline or in a heap vector; insert
	 *         stopping early or writing every level (by full propagation). */
	template<bool HasInlineOffsets, bool HasEarlyInsert>
	class hierarchical_bitset_by_offset_and_propagation {
	public:
		static constexpr std::size_t npos = std::size_t(-1);

		explicit hierarchical_bitset_by_offset_and_propagation(std::size_t universe) {
			std::size_t m = universe ? universe : 1;
			if constexpr (!HasInlineOffsets) offset_.push_back(0);
			std::size_t total = 0;
			do {
				std::size_t const words = m / 64 + (m % 64 != 0);
				total += words;
				if constexpr (HasInlineOffsets)
					offset_[++levels_] = total;
				else
					offset_.push_back(total);
				m = words;
			} while (m > 1);
			if constexpr (!HasInlineOffsets) levels_ = offset_.size() - 1;
			words_.assign(total, 0);
		}

		bool contains(std::size_t x) const { return words_[x >> 6] >> (x & 63) & 1; }

		void insert(std::size_t x) {
			if (contains(x)) return;
			if constexpr (HasEarlyInsert) {
				for (std::size_t h = 0; h < levels_; ++h) {
					atl::uint64 &word = words_[offset_[h] + (x >> 6)];
					atl::uint64 const old = word;
					word = old | atl::uint64(1) << (x & 63);
					if (old) return;
					x >>= 6;
				}
			} else {
				for (std::size_t h = 0; h < levels_; ++h) {
					words_[offset_[h] + (x >> 6)] |= atl::uint64(1) << (x & 63);
					x >>= 6;
				}
			}
		}

		void erase(std::size_t x) {
			if (!contains(x)) return;
			for (std::size_t h = 0; h < levels_; ++h) {
				std::size_t const w = offset_[h] + (x >> 6);
				words_[w] &= ~(atl::uint64(1) << (x & 63));
				if (words_[w]) break;
				x >>= 6;
			}
		}

		std::size_t next(std::size_t x) const {
			for (std::size_t h = 0; h < levels_; ++h) {
				std::size_t const w = x >> 6;
				if (w >= offset_[h + 1] - offset_[h]) return npos;
				atl::uint64 const d = words_[offset_[h] + w] >> (x & 63);
				if (d) {
					x += std::size_t(std::countr_zero(d));
					for (std::size_t g = h; g-- > 0;) {
						x <<= 6;
						x += std::size_t(std::countr_zero(words_[offset_[g] + (x >> 6)]));
					}
					return x;
				}
				x = w + 1;
			}
			return npos;
		}

		std::size_t prev(std::size_t x) const {
			for (std::size_t h = 0; h < levels_; ++h) {
				atl::uint64 const d = words_[offset_[h] + (x >> 6)] << (63 - (x & 63));
				if (d) {
					x -= std::size_t(std::countl_zero(d));
					for (std::size_t g = h; g-- > 0;) {
						x <<= 6;
						x += 63 - std::size_t(std::countl_zero(words_[offset_[g] + (x >> 6)]));
					}
					return x;
				}
				if ((x >> 6) == 0) return npos;
				x = (x >> 6) - 1;
			}
			return npos;
		}

	private:
		std::size_t levels_ = 0;
		std::conditional_t<HasInlineOffsets, std::array<std::size_t, 12>, std::vector<std::size_t>> offset_{};
		std::vector<atl::uint64> words_;
	};

	/** @brief Keep heap offsets but propagate every inserted bit through all summaries. */
	using hierarchical_bitset_by_full_propagation = hierarchical_bitset_by_offset_and_propagation<false, false>;
	/** @brief Keep early propagation but store level offsets inline. */
	using hierarchical_bitset_by_inline_offsets = hierarchical_bitset_by_offset_and_propagation<true, true>;

	/** @brief n random keys below @p universe. */
	std::vector<std::size_t> random_keys(std::size_t n, std::size_t universe, std::uint64_t seed) {
		std::mt19937_64 rng(seed);
		std::vector<std::size_t> keys(n);
		for (auto &k : keys) k = std::size_t(rng() % universe);
		return keys;
	}

	/** @brief Insert every key into a fresh set; return a checksum. */
	template<typename Set>
		requires std::constructible_from<Set, std::size_t> && requires(Set &set, Set const &read, std::size_t x) {
			set.insert(x);
			set.erase(x);
			{ read.contains(x) } -> std::convertible_to<bool>;
			{ read.next(x) } -> std::convertible_to<std::size_t>;
			{ read.prev(x) } -> std::convertible_to<std::size_t>;
			{ Set::npos } -> std::convertible_to<std::size_t>;
		}
	std::size_t insert_all(std::size_t universe, std::vector<std::size_t> const &keys) {
		Set t(universe);
		for (auto k : keys) t.insert(k);
		return std::size_t(t.contains(keys[0]));
	}

	/** @brief Alternate insert(a) / erase(b) over a set already half full. */
	template<typename Set>
		requires std::constructible_from<Set, std::size_t> && requires(Set &set, Set const &read, std::size_t x) {
			set.insert(x);
			set.erase(x);
			{ read.contains(x) } -> std::convertible_to<bool>;
			{ read.next(x) } -> std::convertible_to<std::size_t>;
			{ read.prev(x) } -> std::convertible_to<std::size_t>;
			{ Set::npos } -> std::convertible_to<std::size_t>;
		}
	std::size_t insert_erase_keys(std::size_t universe, std::vector<std::size_t> const &keys) {
		Set t(universe);
		for (std::size_t i = 0; i < keys.size() / 2; ++i) t.insert(keys[i]);
		for (std::size_t i = 0; i + 1 < keys.size(); i += 2) {
			t.insert(keys[i]);
			t.erase(keys[i + 1]);
		}
		return std::size_t(t.contains(keys[0]));
	}

	/** @brief Sum of next(k) over the query keys. */
	template<typename Set>
		requires std::constructible_from<Set, std::size_t> && requires(Set &set, Set const &read, std::size_t x) {
			set.insert(x);
			set.erase(x);
			{ read.contains(x) } -> std::convertible_to<bool>;
			{ read.next(x) } -> std::convertible_to<std::size_t>;
			{ read.prev(x) } -> std::convertible_to<std::size_t>;
			{ Set::npos } -> std::convertible_to<std::size_t>;
		}
	std::size_t sum_next_queries(Set const &t, std::vector<std::size_t> const &queries) {
		std::size_t acc = 0;
		for (auto k : queries) {
			std::size_t const r = t.next(k);
			acc += r == Set::npos ? 0 : r;
		}
		return acc;
	}

	/** @brief Sum of prev(k) over the query keys. */
	template<typename Set>
		requires std::constructible_from<Set, std::size_t> && requires(Set &set, Set const &read, std::size_t x) {
			set.insert(x);
			set.erase(x);
			{ read.contains(x) } -> std::convertible_to<bool>;
			{ read.next(x) } -> std::convertible_to<std::size_t>;
			{ read.prev(x) } -> std::convertible_to<std::size_t>;
			{ Set::npos } -> std::convertible_to<std::size_t>;
		}
	std::size_t sum_prev_queries(Set const &t, std::vector<std::size_t> const &queries) {
		std::size_t acc = 0;
		for (auto k : queries) {
			std::size_t const r = t.prev(k);
			acc += r == Set::npos ? 0 : r;
		}
		return acc;
	}
}

TEST_CASE("hierarchical_bitset benchmarks", "[hierarchical_bitset]") {
	constexpr std::size_t n = 1'000'000;
	constexpr std::size_t dense = 1'000'000, sparse = std::size_t(1) << 26;
	auto const dense_keys = random_keys(n, dense, 1), sparse_keys = random_keys(n, sparse, 2);
	auto const dense_queries = random_keys(n, dense, 3), sparse_queries = random_keys(n, sparse, 4);

	BENCHMARK("insert 1e6, universe 1e6 (library)") { return insert_all<atl::hierarchical_bitset>(dense, dense_keys); };
	BENCHMARK("insert 1e6, universe 1e6 (by full propagation)") { return insert_all<hierarchical_bitset_by_full_propagation>(dense, dense_keys); };
	BENCHMARK("insert 1e6, universe 1e6 (by inline offsets)") { return insert_all<hierarchical_bitset_by_inline_offsets>(dense, dense_keys); };
	BENCHMARK("insert 1e6, universe 2^26 (library)") { return insert_all<atl::hierarchical_bitset>(sparse, sparse_keys); };
	BENCHMARK("insert 1e6, universe 2^26 (by full propagation)") { return insert_all<hierarchical_bitset_by_full_propagation>(sparse, sparse_keys); };
	BENCHMARK("std::set insert 1e6, universe 1e6") {
		std::set<std::size_t> s;
		for (auto k : dense_keys) s.insert(k);
		return s.size();
	};
	BENCHMARK("insert/erase churn 1e6, universe 1e6 (library)") { return insert_erase_keys<atl::hierarchical_bitset>(dense, dense_keys); };
	BENCHMARK("insert/erase churn 1e6, universe 1e6 (by full propagation)") { return insert_erase_keys<hierarchical_bitset_by_full_propagation>(dense, dense_keys); };

	BENCHMARK("insert/erase churn 1e6, universe 1e6 (by inline offsets)") { return insert_erase_keys<hierarchical_bitset_by_inline_offsets>(dense, dense_keys); };

	atl::hierarchical_bitset const td(dense, dense_keys), ts(sparse, sparse_keys);
	hierarchical_bitset_by_inline_offsets id(dense), is(sparse);
	for (auto k : dense_keys) id.insert(k);
	for (auto k : sparse_keys) is.insert(k);
	hierarchical_bitset_by_full_propagation ld(dense), ls(sparse);
	for (auto k : dense_keys) ld.insert(k);
	for (auto k : sparse_keys) ls.insert(k);
	std::set<std::size_t> const s(dense_keys.begin(), dense_keys.end());
	for (auto k : dense_queries) {
		auto const next = s.lower_bound(k);
		REQUIRE(td.next(k) == (next == s.end() ? atl::hierarchical_bitset::npos : *next));
		REQUIRE(td.next(k) == ld.next(k));
		REQUIRE(td.next(k) == id.next(k));
		REQUIRE(td.prev(k) == ld.prev(k));
		REQUIRE(td.prev(k) == id.prev(k));
	}
	for (auto k : sparse_queries) {
		REQUIRE(ts.next(k) == ls.next(k));
		REQUIRE(ts.next(k) == is.next(k));
		REQUIRE(ts.prev(k) == ls.prev(k));
		REQUIRE(ts.prev(k) == is.prev(k));
	}
	REQUIRE(insert_all<atl::hierarchical_bitset>(dense, dense_keys) == insert_all<hierarchical_bitset_by_full_propagation>(dense, dense_keys));
	REQUIRE(insert_all<atl::hierarchical_bitset>(dense, dense_keys) == insert_all<hierarchical_bitset_by_inline_offsets>(dense, dense_keys));
	REQUIRE(insert_all<atl::hierarchical_bitset>(sparse, sparse_keys) == insert_all<hierarchical_bitset_by_full_propagation>(sparse, sparse_keys));
	REQUIRE(insert_erase_keys<atl::hierarchical_bitset>(dense, dense_keys) == insert_erase_keys<hierarchical_bitset_by_full_propagation>(dense, dense_keys));
	REQUIRE(insert_erase_keys<atl::hierarchical_bitset>(dense, dense_keys) == insert_erase_keys<hierarchical_bitset_by_inline_offsets>(dense, dense_keys));

	BENCHMARK("next 1e6, universe 1e6 (library)") { return sum_next_queries(td, dense_queries); };
	BENCHMARK("next 1e6, universe 1e6 (by full propagation)") { return sum_next_queries(ld, dense_queries); };
	BENCHMARK("next 1e6, universe 1e6 (by inline offsets)") { return sum_next_queries(id, dense_queries); };
	BENCHMARK("next 1e6, universe 2^26 (library)") { return sum_next_queries(ts, sparse_queries); };
	BENCHMARK("next 1e6, universe 2^26 (by full propagation)") { return sum_next_queries(ls, sparse_queries); };
	BENCHMARK("next 1e6, universe 2^26 (by inline offsets)") { return sum_next_queries(is, sparse_queries); };
	BENCHMARK("prev 1e6, universe 1e6 (library)") { return sum_prev_queries(td, dense_queries); };
	BENCHMARK("prev 1e6, universe 1e6 (by full propagation)") { return sum_prev_queries(ld, dense_queries); };
	BENCHMARK("prev 1e6, universe 1e6 (by inline offsets)") { return sum_prev_queries(id, dense_queries); };
	BENCHMARK("prev 1e6, universe 2^26 (library)") { return sum_prev_queries(ts, sparse_queries); };
	BENCHMARK("prev 1e6, universe 2^26 (by full propagation)") { return sum_prev_queries(ls, sparse_queries); };
	BENCHMARK("prev 1e6, universe 2^26 (by inline offsets)") { return sum_prev_queries(is, sparse_queries); };
	BENCHMARK("std::set lower_bound 1e6, universe 1e6") {
		std::size_t acc = 0;
		for (auto k : dense_queries) {
			auto it = s.lower_bound(k);
			acc += it == s.end() ? 0 : *it;
		}
		return acc;
	};
}
