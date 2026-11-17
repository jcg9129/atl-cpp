/**
 * @file
 * @brief Races prefix-count order statistics by the library versus parallel raw arrays,
 * and point sets by child-id reads versus whole-node copies. Measures build-plus-query
 * throughput for n=q=2e5 and 2e5 updates on 2x2 matrices, 24-character string maxima
 * and uint32 sums. Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): prefix
 * build plus queries library 317 / 326 ms, raw arrays 355 / 373.
 * See docs/persistent-segment-tree.md for the full table.
 */

#include <atl/ds/persistent_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	/** @brief Time a complete workload through a direct, constrained callable. */
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}
}

using atl::uint32;

namespace {
	/** @brief Prefix counts by three raw arrays with bump allocation and a specialized sum fold. */
	struct kth_by_parallel_arrays {
		std::vector<uint32> lc, rc, cnt;
		std::size_t n;

		explicit kth_by_parallel_arrays(std::size_t m, std::size_t updates) : n(m) {
			std::size_t const levels = m > 4 ? std::size_t(std::bit_width(m - 1)) : 2;
			lc.reserve(updates * (levels + 1) + 2 * m);
			rc.reserve(updates * (levels + 1) + 2 * m);
			cnt.reserve(updates * (levels + 1) + 2 * m);
			make(0, 0, 0); // sentinel
		}

		uint32 make(uint32 l, uint32 r, uint32 c) {
			atl_assert(lc.size() <= (std::numeric_limits<uint32>::max)());
			lc.push_back(l);
			rc.push_back(r);
			cnt.push_back(c);
			return uint32(lc.size() - 1);
		}

		uint32 build(std::size_t lo, std::size_t hi) {
			if (hi - lo == 1) return make(0, 0, 0);
			std::size_t const mid = lo + (hi - lo) / 2;
			uint32 const l = build(lo, mid), r = build(mid, hi);
			return make(l, r, 0);
		}

		uint32 add(uint32 x, std::size_t lo, std::size_t hi, std::size_t i) {
			if (hi - lo == 1) return make(0, 0, cnt[x] + 1);
			std::size_t const mid = lo + (hi - lo) / 2;
			if (i < mid) {
				uint32 const l = add(lc[x], lo, mid, i);
				return make(l, rc[x], cnt[x] + 1);
			}
			uint32 const r = add(rc[x], mid, hi, i);
			return make(lc[x], r, cnt[x] + 1);
		}

		std::size_t kth(uint32 a, uint32 b, std::size_t k) const {
			std::size_t lo = 0, hi = n;
			while (hi - lo > 1) {
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const c = cnt[lc[b]] - cnt[lc[a]];
				if (k < c) {
					a = lc[a];
					b = lc[b];
					hi = mid;
				} else {
					k -= c;
					a = rc[a];
					b = rc[b];
					lo = mid;
				}
			}
			return lo;
		}
	};
}

TEST_CASE("persistent_segment_tree benchmarks", "[persistent_segment_tree]") {
	constexpr std::size_t n = 200'000, q = 200'000;
	std::mt19937 rng(1);
	std::vector<std::size_t> a(n);
	for (auto &v : a) v = rng() % n;

	struct qq {
		std::size_t l, r, k;
	};

	std::vector<qq> qs(q);
	for (auto &[l, r, k] : qs) {
		l = rng() % n;
		r = l + 1 + rng() % (n - l);
		k = rng() % (r - l);
	}

	auto const run_library = [&] {
		using pst = atl::persistent_segment_tree<uint32, std::plus<>>;
		std::vector<pst> vs;
		vs.reserve(n + 1);
		vs.emplace_back(n, uint32(0), std::plus<>{});
		vs.front().reserve(n);
		for (std::size_t i = 0; i < n; ++i)
			vs.push_back(vs.back().update(a[i], [](uint32 const &c) { return c + 1; }));
		std::size_t acc = 0;
		for (auto const &[l, r, k] : qs)
			acc += pst::descend(vs[l], vs[r], [k = k](uint32 const &cl, uint32 const &cr) mutable {
				uint32 const c = cr - cl;
				if (k < c) return true;
				k -= c;
				return false;
			});
		return acc;
	};

	auto const run_by_parallel_arrays = [&] {
		kth_by_parallel_arrays t(n, n);
		std::vector<uint32> roots{t.make(0, 0, 0)};
		roots[0] = t.build(0, n);
		for (std::size_t i = 0; i < n; ++i) roots.push_back(t.add(roots.back(), 0, n, a[i]));
		std::size_t acc = 0;
		for (auto const &[l, r, k] : qs) acc += t.kth(roots[l], roots[r], k);
		return acc;
	};
	REQUIRE(run_library() == run_by_parallel_arrays());
	measure("library: build prefixes + kth queries", run_library);
	measure("by parallel arrays", run_by_parallel_arrays);
}

namespace {
	/**
	 * @brief Arena model of the persistent point update comparison: @p CopyNode copies the whole node (aggregate included)
	 *        out at every level; otherwise
	 *        only child ids are read and the leaf aggregate is consumed
	 *        in place (library).
	 */
	template<std::copyable S, atl::binary_operation<S> Op, bool CopyNode>
	struct set_by_node_copy {
		struct node {
			S aggregate;
			atl::uint32 left, right;
		};

		Op op;
		std::vector<node> nodes;

		atl::uint32 make(S a, atl::uint32 l, atl::uint32 r) {
			atl_assert(nodes.size() <= (std::numeric_limits<atl::uint32>::max)());
			nodes.push_back({std::move(a), l, r});
			return atl::uint32(nodes.size() - 1);
		}

		atl::uint32 build(std::vector<S> const &v, std::size_t lo, std::size_t hi) {
			if (hi - lo == 1) return make(v[lo], 0, 0);
			std::size_t const mid = lo + (hi - lo) / 2;
			atl::uint32 const l = build(v, lo, mid), r = build(v, mid, hi);
			return make(op(nodes[l].aggregate, nodes[r].aggregate), l, r);
		}

		atl::uint32 set(atl::uint32 x, std::size_t lo, std::size_t hi, std::size_t i, S const &value) {
			atl::uint32 lc, rc;
			if constexpr (CopyNode) {
				node const nd = nodes[x];
				if (hi - lo == 1) return make(value, 0, 0);
				lc = nd.left, rc = nd.right;
			} else {
				if (hi - lo == 1) return make(value, 0, 0);
				lc = nodes[x].left, rc = nodes[x].right;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (i < mid)
				lc = set(lc, lo, mid, i, value);
			else
				rc = set(rc, mid, hi, i, value);
			return make(op(nodes[lc].aggregate, nodes[rc].aggregate), lc, rc);
		}
	};

	using mat = std::array<atl::uint64, 4>;

	struct mat_op {
		mat operator()(mat const &a, mat const &b) const {
			constexpr atl::uint64 p = 998244353;
			return {(a[0] * b[0] + a[1] * b[2]) % p, (a[0] * b[1] + a[1] * b[3]) % p, (a[2] * b[0] + a[3] * b[2]) % p,
					(a[2] * b[1] + a[3] * b[3]) % p};
		}
	};

	struct max_string_op {
		std::string operator()(std::string const &a, std::string const &b) const { return a < b ? b : a; }
	};

	/** @brief Certify that copy strategies preserve every updated aggregate. */
	template<std::copyable S, atl::binary_operation<S> Op>
		requires std::equality_comparable<S> && std::default_initializable<Op>
	void verify_updates(std::vector<S> const &init, std::vector<std::pair<std::size_t, S>> const &updates) {
		set_by_node_copy<S, Op, false> ids{Op{}, {}};
		set_by_node_copy<S, Op, true> whole{Op{}, {}};
		auto a = ids.build(init, 0, init.size());
		auto b = whole.build(init, 0, init.size());
		for (auto const &[i, value] : updates) {
			a = ids.set(a, 0, init.size(), i, value);
			b = whole.set(b, 0, init.size(), i, value);
			REQUIRE(ids.nodes[a].aggregate == whole.nodes[b].aggregate);
		}
		REQUIRE(ids.nodes.size() == whole.nodes.size());
	}

	template<std::copyable S, atl::binary_operation<S> Op, bool CopyNode>
		requires std::default_initializable<Op>
	std::size_t run_updates(std::vector<S> const &init, std::vector<std::pair<std::size_t, S>> const &ups) {
		set_by_node_copy<S, Op, CopyNode> m{Op{}, {}};
		m.nodes.reserve(2 * init.size() + ups.size() * 20);
		atl::uint32 root = m.build(init, 0, init.size());
		for (auto const &[i, v] : ups) root = m.set(root, 0, init.size(), i, v);
		return m.nodes.size() + root;
	}
}

TEST_CASE("persistent_segment_tree child-id reads vs whole-node copy", "[persistent_segment_tree][race]") {
	constexpr std::size_t n = 200'000, q = 200'000;
	std::mt19937_64 rng(8);
	std::vector<mat> im(n);
	std::vector<std::string> is(n);
	std::vector<std::pair<std::size_t, mat>> um(q);
	std::vector<std::pair<std::size_t, std::string>> us(q);
	auto rand_string = [&] {
		std::string s(24, 'a');
		for (auto &c : s) c = char('a' + rng() % 26);
		return s;
	};
	for (auto &m : im)
		for (auto &e : m) e = rng() % 998244353;
	for (auto &s : is) s = rand_string();
	for (auto &[i, m] : um) {
		i = rng() % n;
		for (auto &e : m) e = rng() % 998244353;
	}
	for (auto &[i, s] : us) i = rng() % n, s = rand_string();
	verify_updates<mat, mat_op>(im, um);
	verify_updates<std::string, max_string_op>(is, us);
	std::vector<atl::uint32> check_initial(n, 1);
	std::vector<std::pair<std::size_t, atl::uint32>> check_updates(q);
	for (std::size_t k = 0; k < q; ++k) check_updates[k] = {um[k].first, atl::uint32(k)};
	verify_updates<atl::uint32, std::plus<>>(check_initial, check_updates);
	measure("2e5 sets, 2x2 matrix, by child ids", [&] { return run_updates<mat, mat_op, false>(im, um); });
	measure("2e5 sets, 2x2 matrix, by whole-node copy", [&] { return run_updates<mat, mat_op, true>(im, um); });
	measure("2e5 sets, 24-char string max, by child ids", [&] {
		return run_updates<std::string, max_string_op, false>(is, us);
	});
	measure("2e5 sets, 24-char string max, by whole-node copy", [&] {
		return run_updates<std::string, max_string_op, true>(is, us);
	});
	measure("2e5 sets, uint32 plus, by child ids", [&] {
		std::vector<atl::uint32> iu(n, 1);
		std::vector<std::pair<std::size_t, atl::uint32>> uu(q);
		for (std::size_t k = 0; k < q; ++k) uu[k] = {um[k].first, atl::uint32(k)};
		return run_updates<atl::uint32, std::plus<>, false>(iu, uu);
	});
	measure("2e5 sets, uint32 plus, by whole-node copy", [&] {
		std::vector<atl::uint32> iu(n, 1);
		std::vector<std::pair<std::size_t, atl::uint32>> uu(q);
		for (std::size_t k = 0; k < q; ++k) uu[k] = {um[k].first, atl::uint32(k)};
		return run_updates<atl::uint32, std::plus<>, true>(iu, uu);
	});
}
