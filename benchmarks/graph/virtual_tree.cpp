/**
 * @file
 * @brief Virtual-tree batch throughput on random, path and star trees, with LCA
 * construction excluded. Contenders change one/two sorting passes, key encoding,
 * LCA backend, stack representation, frame ranking or scratch lifetime. The
 * shared loop hashes outputs; untimed runs require matching hashes.
 * Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2 / -O2 -march=native, 2026-11-01: 64 queries of n/128 keys at n=5e5
 * take 32.0 / 30.2 ms with bitmap sorting and 53.3 / 49.0 ms with counting.
 * Construction/backend comparisons are not measured yet in the supplied records.
 * Full table: docs/virtual-tree.md.
 */
#include <atl/experimental/graph/virtual_tree_variants.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace {
	constexpr std::size_t n = 500'000;

	/** @brief Parent array of one shape on @p size vertices with shuffled labels. */
	std::vector<uint32> make_tree(std::string const &shape, std::mt19937_64 &rng, std::size_t size = n) {
		std::vector<uint32> parent(size, g::none);
		for (uint32 v = 1; v < size; ++v)
			parent[v] = shape == "path" ? v - 1 : shape == "star" ? 0
																  : uint32(rng() % v);
		std::vector<uint32> label(size);
		std::iota(label.begin(), label.end(), 0u);
		std::ranges::shuffle(label, rng);
		std::vector<uint32> relabelled(size);
		for (uint32 v = 0; v < size; ++v) relabelled[label[v]] = parent[v] == g::none ? g::none : label[parent[v]];
		return relabelled;
	}

	/** @brief A batch of queries flattened: query q is `keys[offset[q] .. offset[q + 1])`. */
	struct batch {
		std::vector<uint32> keys, offset{0};

		std::span<uint32 const> query(std::size_t q) const {
			return std::span<uint32 const>(keys).subspan(offset[q], offset[q + 1] - offset[q]);
		}

		std::size_t size() const { return offset.size() - 1; }
	};

	/** @brief @p queries random queries of `k_min..k_max` keys uniform below @p size (duplicates possible). */
	batch make_batch(std::size_t queries, std::size_t k_min, std::size_t k_max, std::mt19937_64 &rng,
					 std::size_t size = n) {
		batch b;
		for (std::size_t q = 0; q < queries; ++q) {
			std::size_t const k = k_min + rng() % (k_max - k_min + 1);
			for (std::size_t i = 0; i < k; ++i) b.keys.push_back(uint32(rng() % size));
			atl_assert(b.keys.size() < (uint64(1) << 32));
			b.offset.push_back(uint32(b.keys.size()));
		}
		return b;
	}

	/** @brief A checksum of one result; the polynomial hash wraps modulo 2^64 on purpose. */
	uint64 fold(g::virtual_tree const &vt) {
		uint64 h = vt.vertices.size();
		for (std::size_t i = 0; i < vt.vertices.size(); ++i) h = h * 1'000'003 + vt.vertices[i] * 31 + vt.parent[i];
		return h;
	}

	/** @brief Runs @p build(query, out) over the batch, one result reused. */
	template<typename Build>
		requires std::invocable<Build const &, std::span<uint32 const>, g::virtual_tree &>
	uint64 run(batch const &b, Build const &build) {
		g::virtual_tree out;
		uint64 h = 0;
		for (std::size_t q = 0; q < b.size(); ++q) {
			build(b.query(q), out);
			h += fold(out);
		}
		return h;
	}

	/** @brief A raced contender over the batch, its workspace reused across queries. */
	template<ex::virtual_tree_algorithm Algorithm, ex::virtual_tree_key_policy Policy>
		requires(Algorithm == ex::virtual_tree_algorithm::stack_sweep || Algorithm == ex::virtual_tree_algorithm::two_sort)
	uint64 run_variant(batch const &b, Policy const &p, bool is_counting = false) {
		ex::virtual_tree_workspace<typename Policy::key_type> w;
		return run(b, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
			ex::virtual_tree_by_policy<Algorithm>(p, keys, w, out, is_counting);
		});
	}

	/** @brief The depth-frame contender over the batch, its workspace reused across queries. */
	uint64 run_depth(batch const &b, g::lca const &tree, bool is_counting = false) {
		ex::virtual_tree_workspace<uint64> w;
		return run(b, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
			ex::virtual_tree_by_depth(tree, keys, w, out, is_counting);
		});
	}
}

namespace {
	/** @brief The raw-array-stack contender over the batch, its workspace reused across queries. */
	uint64 run_raw_array(batch const &b, g::lca const &tree) {
		ex::virtual_tree_workspace<uint64> w;
		return run(b, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
			ex::virtual_tree_by_raw_array(tree, keys, w, out);
		});
	}
}

TEST_CASE("virtual_tree race", "[virtual_tree]") {
	using enum ex::virtual_tree_algorithm;
	for (std::string const tree_shape : {"random", "path", "star"}) {
		std::mt19937_64 rng(2026'10'24);
		auto const parent = make_tree(tree_shape, rng);
		g::lca const tree(parent);
		ex::lca_by_entry_sparse_table const sparse(tree);
		ex::packed_entry_keys const packed{tree};
		ex::projected_keys const projected{tree};
		ex::entry_keys const entry{tree};
		ex::sparse_entry_keys const sparse_entry{sparse};

		/** @brief A query workload; `is_large` adds the counting-sort rows. */
		struct shape {
			std::string name;
			batch queries;
			bool is_large;
		};

		std::vector<shape> shapes;
		shapes.push_back({"3e5 queries k in [2,4]", make_batch(300'000, 2, 4, rng), false});
		shapes.push_back({"1e4 queries k = 100", make_batch(10'000, 100, 100, rng), false});
		shapes.push_back({"1 query k = n/2", make_batch(1, n / 2, n / 2, rng), true});
		shapes.push_back({"8 queries k = n/16", make_batch(8, n / 16, n / 16, rng), true});
		shapes.push_back({"16 queries k = n/32", make_batch(16, n / 32, n / 32, rng), true});
		shapes.push_back({"32 queries k = n/64", make_batch(32, n / 64, n / 64, rng), true});
		shapes.push_back({"128 queries k = n/256", make_batch(128, n / 256, n / 256, rng), true});
		shapes.push_back({"512 queries k = n/1024", make_batch(512, n / 1024, n / 1024, rng), true});

		BENCHMARK(tree_shape + " | build lca_by_entry_sparse_table (on top of the lca)") {
			return ex::lca_by_entry_sparse_table(tree).lca_entry(0, 1);
		};
		for (auto const &[shape_name, queries, is_large] : shapes) {
			std::string const name = tree_shape + " | " + shape_name;
			uint64 const want = run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
				out = g::virtual_tree_of(tree, keys);
			});
			REQUIRE(run_variant<stack_sweep>(queries, packed) == want);
			REQUIRE(run_variant<stack_sweep>(queries, projected) == want);
			REQUIRE(run_variant<stack_sweep>(queries, entry) == want);
			REQUIRE(run_variant<stack_sweep>(queries, sparse_entry) == want);
			REQUIRE(run_variant<two_sort>(queries, packed) == want);
			REQUIRE(run_depth(queries, tree) == want);
			REQUIRE(run_raw_array(queries, tree) == want);
			REQUIRE(run_variant<two_sort>(queries, sparse_entry) == want);

			BENCHMARK(name + " | library fresh") {
				return run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
					out = g::virtual_tree_of(tree, keys);
				});
			};
			BENCHMARK(name + " | library refill") {
				return run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
					g::virtual_tree_of(tree, keys, out);
				});
			};
			BENCHMARK(name + " | by stack sweep packed SV") { return run_variant<stack_sweep>(queries, packed); };
			if (!is_large) {
				BENCHMARK(name + " | by stack sweep packed SV, scratch allocated per query") {
					return run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
						ex::virtual_tree_workspace<uint64> w;
						ex::virtual_tree_by_policy<stack_sweep>(packed, keys, w, out);
					});
				};
			}
			BENCHMARK(name + " | by stack sweep packed SV depth frames") { return run_depth(queries, tree); };
			BENCHMARK(name + " | by stack sweep packed SV raw-array stack") { return run_raw_array(queries, tree); };
			BENCHMARK(name + " | by stack sweep projected SV") { return run_variant<stack_sweep>(queries, projected); };
			BENCHMARK(name + " | by stack sweep entry SV") { return run_variant<stack_sweep>(queries, entry); };
			BENCHMARK(name + " | by stack sweep entry sparse") { return run_variant<stack_sweep>(queries, sparse_entry); };
			BENCHMARK(name + " | by two sort packed SV") { return run_variant<two_sort>(queries, packed); };
			BENCHMARK(name + " | by two sort entry sparse") { return run_variant<two_sort>(queries, sparse_entry); };
			if (is_large) {
				REQUIRE(run_variant<stack_sweep>(queries, packed, true) == want);
				REQUIRE(run_variant<stack_sweep>(queries, entry, true) == want);
				BENCHMARK(name + " | by stack sweep packed SV counting") { return run_variant<stack_sweep>(queries, packed, true); };
				BENCHMARK(name + " | by stack sweep packed SV depth frames counting") { return run_depth(queries, tree, true); };
				BENCHMARK(name + " | by stack sweep entry SV counting") { return run_variant<stack_sweep>(queries, entry, true); };
				BENCHMARK(name + " | by stack sweep entry sparse counting") {
					return run_variant<stack_sweep>(queries, sparse_entry, true);
				};
			}
		}
	}
}

TEST_CASE("virtual_tree key sort race", "[virtual_tree]") {
	for (std::size_t const size : {n, std::size_t(200'000)}) {
		std::mt19937_64 rng(2026'10'29);
		auto const parent = make_tree("random", rng, size);
		g::lca const tree(parent);
		for (std::size_t const divisor : {2, 8, 32, 128, 1024}) {
			batch const queries = make_batch(divisor / 2, size / divisor, size / divisor, rng, size);
			std::string const name =
				"n=" + std::to_string(size) + " | " + std::to_string(divisor / 2) + " queries k = n/" + std::to_string(divisor);
			atl::graph::detail::virtual_tree_scratch s;
			auto const sorted_by = [&](ex::virtual_tree_sort method) {
				return run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
					ex::virtual_tree_by_sort(tree, keys, method, s, out);
				});
			};
			uint64 const want = run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
				g::virtual_tree_of(tree, keys, out);
			});
			REQUIRE(sorted_by(ex::virtual_tree_sort::counting_dispatch) == want);
			REQUIRE(sorted_by(ex::virtual_tree_sort::counting) == want);
			REQUIRE(sorted_by(ex::virtual_tree_sort::comparison) == want);
			BENCHMARK(name + " | key sort library bitmap") {
				return run(queries, [&](std::span<uint32 const> keys, g::virtual_tree &out) {
					g::virtual_tree_of(tree, keys, out);
				});
			};
			BENCHMARK(name + " | key sort by counting dispatch") { return sorted_by(ex::virtual_tree_sort::counting_dispatch); };
			BENCHMARK(name + " | key sort counting") { return sorted_by(ex::virtual_tree_sort::counting); };
			BENCHMARK(name + " | key sort comparison") { return sorted_by(ex::virtual_tree_sort::comparison); };
		}
	}
}
