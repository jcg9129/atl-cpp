/**
 * @file
 * @brief Point assignment, range fold, boundary search and construction throughput.
 *
 * The recursive contender uses top-down traversal; the full-build contender combines padding and recomputes accepted search folds. Independent query loops and set streams use one million elements; build shapes straddle a power of two.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, -O2 and -O2 -march=native, 2026-10-15: library set 124/121 ms versus recursion 175/170 ms.
 * Full measurements: docs/segment-tree.md.
 */
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/ds/segment_tree.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>

using ll = long long;

/** @brief Point assignment and range sums by top-down recursion in a 4n array. */
struct segment_tree_by_recursion {
	int n;
	std::vector<ll> t;

	explicit segment_tree_by_recursion(int size) : n(size), t(4 * size, 0) {}

	void upd(int nd, int lo, int hi, int i, ll v) {
		if (hi - lo == 1) {
			t[nd] = v;
			return;
		}
		int const m = (lo + hi) / 2;
		if (i < m)
			upd(2 * nd, lo, m, i, v);
		else
			upd(2 * nd + 1, m, hi, i, v);
		t[nd] = t[2 * nd] + t[2 * nd + 1];
	}

	ll qry(int nd, int lo, int hi, int l, int r) const {
		if (r <= lo || hi <= l) return 0;
		if (l <= lo && hi <= r) return t[nd];
		int const m = (lo + hi) / 2;
		return qry(2 * nd, lo, m, l, r) + qry(2 * nd + 1, m, hi, l, r);
	}

	void set(int i, ll v) { upd(1, 0, n, i, v); }

	ll fold(int l, int r) const { return qry(1, 0, n, l, r); }
};

// Iterative (atl::segment_tree) vs a recursive segment tree — point set and range sum.
// The iterative sweep visits fewer nodes with no call overhead, so it wins, most on queries.
TEST_CASE("segment_tree benchmarks", "[segment_tree]") {
	constexpr int n = 1'000'000, m = 1'000'000;
	std::mt19937 rng(1);
	std::vector<int> pos(m), lo(m), hi(m);
	std::vector<ll> val(m);
	for (int i = 0; i < m; ++i) {
		pos[i] = int(rng() % n);
		val[i] = ll(rng() % 1000);
		int a = int(rng() % n), b = int(rng() % n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
	}

	atl::segment_tree it(std::vector<ll>(n, 0), 0LL, std::plus<>{});
	segment_tree_by_recursion rc(n);

	BENCHMARK("library -- 1e6 set") {
		for (int i = 0; i < m; ++i) it.set(std::size_t(pos[i]), val[i]);
		return it.fold();
	};
	BENCHMARK("by recursion -- 1e6 set") {
		for (int i = 0; i < m; ++i) rc.set(pos[i], val[i]);
		return rc.fold(0, n);
	};
	BENCHMARK("library -- 1e6 range query") {
		ll acc = 0;
		for (int i = 0; i < m; ++i) acc += it.fold(std::size_t(lo[i]), std::size_t(hi[i]));
		return acc;
	};
	BENCHMARK("by recursion -- 1e6 range query") {
		ll acc = 0;
		for (int i = 0; i < m; ++i) acc += rc.fold(lo[i], hi[i]);
		return acc;
	};
}

namespace {
	/** @brief Full-capacity
	 *         build and two `op` calls per accepted ascent node. */
	template<std::copyable T, atl::binary_operation<T> Op>
	class segment_tree_by_full_build_and_recomputed_search : atl::detail::segment_tree_base<T> {
		using base = atl::detail::segment_tree_base<T>;
		using base::leaf;
		using base::n_;
		using base::nodes_;
		using base::size_;
		Op op_;
		T identity_;

	public:
		segment_tree_by_full_build_and_recomputed_search(std::vector<T> const &values, T identity, Op op = Op{}) : op_(op), identity_(std::move(identity)) {
			this->reset(values.size(), identity_);
			for (std::size_t i = 0; i < values.size(); ++i) nodes_[leaf(i)] = values[i];
			for (std::size_t k = size_; k-- > 1;) nodes_[k] = op_(nodes_[2 * k], nodes_[2 * k + 1]);
		}

		T const &fold() const { return nodes_[1]; }

		template<std::predicate<T const &> Pred>
		std::size_t max_right(std::size_t l, Pred pred) const {
			if (l == n_) return n_;
			l += size_;
			T sm = identity_;
			do {
				while (l % 2 == 0) l >>= 1;
				if (!pred(op_(sm, nodes_[l]))) {
					while (l < size_) {
						l *= 2;
						if (T t = op_(sm, nodes_[l]); pred(t)) sm = std::move(t), ++l;
					}
					return l - size_;
				}
				sm = op_(sm, nodes_[l]);
				++l;
			} while (!std::has_single_bit(l));
			return n_;
		}
	};

	constexpr std::uint64_t mod = 998244353;

	/** @brief A running sum beside a 2x2 matrix mod p: the sum drives max_right, the matrix makes op costly. */
	struct sum_mat {
		std::uint64_t sum = 0;
		std::array<std::uint64_t, 4> m{1, 0, 0, 1};
	};

	struct sum_mat_op {
		sum_mat operator()(sum_mat const &a, sum_mat const &b) const {
			return {a.sum + b.sum,
					{(a.m[0] * b.m[0] + a.m[1] * b.m[2]) % mod, (a.m[0] * b.m[1] + a.m[1] * b.m[3]) % mod,
					 (a.m[2] * b.m[0] + a.m[3] * b.m[2]) % mod, (a.m[2] * b.m[1] + a.m[3] * b.m[3]) % mod}};
		}
	};

	struct gcd_op {
		std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const { return std::gcd(a, b); }
	};

	struct concat_op {
		std::string operator()(std::string const &a, std::string const &b) const { return a + b; }
	};
}

TEST_CASE("segment_tree search, fold and build comparisons", "[segment_tree][race]") {
	constexpr std::size_t n = 1'000'000, q = 1'000'000;
	std::mt19937_64 rng(7);
	std::vector<long long> vll(n);
	std::vector<sum_mat> vmat(n);
	std::vector<std::uint64_t> vgcd(n);
	std::vector<std::string> vstr(n);
	for (std::size_t i = 0; i < n; ++i) {
		vll[i] = (long long)(rng() % 1000);
		vmat[i].sum = rng() % 1000;
		for (auto &e : vmat[i].m) e = rng() % mod;
		vgcd[i] = rng() % 50000 == 0 ? rng() % 1000 + 1 : 6 * (rng() % 1000 + 1);
		vstr[i] = std::string(1 + rng() % 4, char('a' + rng() % 26));
	}
	std::vector<std::size_t> ls(q), short_r(q), uni_l(q), uni_r(q);
	for (std::size_t i = 0; i < q; ++i) {
		ls[i] = rng() % n;
		short_r[i] = (std::min)(n, ls[i] + 1 + rng() % 8);
		std::size_t a = rng() % n, b = rng() % n;
		if (a > b) std::swap(a, b);
		uni_l[i] = a, uni_r[i] = b + 1;
	}

	atl::segment_tree<long long, std::plus<>> library(vll, 0LL);
	segment_tree_by_full_build_and_recomputed_search<long long, std::plus<>> recomputed_ll(vll, 0LL);
	atl::segment_tree<sum_mat, sum_mat_op> library_mat(vmat, sum_mat{});
	segment_tree_by_full_build_and_recomputed_search<sum_mat, sum_mat_op> recomputed_mat(vmat, sum_mat{});
	atl::segment_tree<std::uint64_t, gcd_op> library_gcd(vgcd, std::uint64_t(0));
	segment_tree_by_full_build_and_recomputed_search<std::uint64_t, gcd_op> recomputed_gcd(vgcd, std::uint64_t(0));
	atl::segment_tree<std::string, concat_op> library_str(vstr, std::string());

	for (std::size_t k = 0; k < 100; ++k) {
		REQUIRE(library.max_right(ls[k], [](long long v) { return v <= 25'000'000; }) == recomputed_ll.max_right(ls[k], [](long long v) { return v <= 25'000'000; }));
		REQUIRE(library_mat.max_right(ls[k], [](sum_mat const &v) { return v.sum <= 25'000'000; }) == recomputed_mat.max_right(ls[k], [](sum_mat const &v) { return v.sum <= 25'000'000; }));
		REQUIRE(library_gcd.max_right(ls[k], [](std::uint64_t v) { return v != 1; }) == recomputed_gcd.max_right(ls[k], [](std::uint64_t v) { return v != 1; }));
	}
	constexpr long long ll_budget = 25'000'000;
	BENCHMARK("max_right 1e6, ll sum (library)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += library.max_right(l, [](long long s) { return s <= ll_budget; });
		return acc;
	};
	BENCHMARK("max_right 1e6, ll sum (by full build and recomputed search)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += recomputed_ll.max_right(l, [](long long s) { return s <= ll_budget; });
		return acc;
	};
	BENCHMARK("max_right 1e6, sum+2x2 matrix (library)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += library_mat.max_right(l, [](sum_mat const &s) { return s.sum <= ll_budget; });
		return acc;
	};
	BENCHMARK("max_right 1e6, sum+2x2 matrix (by full build and recomputed search)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += recomputed_mat.max_right(l, [](sum_mat const &s) { return s.sum <= ll_budget; });
		return acc;
	};
	BENCHMARK("max_right 1e6, gcd (library)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += library_gcd.max_right(l, [](std::uint64_t g) { return g != 1; });
		return acc;
	};
	BENCHMARK("max_right 1e6, gcd (by full build and recomputed search)") {
		std::size_t acc = 0;
		for (auto l : ls) acc += recomputed_gcd.max_right(l, [](std::uint64_t g) { return g != 1; });
		return acc;
	};

	BENCHMARK("fold len 1-8, ll sum (library)") {
		long long acc = 0;
		for (std::size_t i = 0; i < q; ++i) acc += library.fold(ls[i], short_r[i]);
		return acc;
	};
	BENCHMARK("fold uniform, ll sum (library)") {
		long long acc = 0;
		for (std::size_t i = 0; i < q; ++i) acc += library.fold(uni_l[i], uni_r[i]);
		return acc;
	};
	BENCHMARK("fold len 1-8, 2x2 matrix (library)") {
		std::uint64_t acc = 0;
		for (std::size_t i = 0; i < q; ++i) acc += library_mat.fold(ls[i], short_r[i]).m[1];
		return acc;
	};
	BENCHMARK("fold uniform, 2x2 matrix (library)") {
		std::uint64_t acc = 0;
		for (std::size_t i = 0; i < q; ++i) acc += library_mat.fold(uni_l[i], uni_r[i]).m[1];
		return acc;
	};
	BENCHMARK("fold len 1-8, string concat (library)") {
		std::size_t acc = 0;
		for (std::size_t i = 0; i < q; ++i) acc += library_str.fold(ls[i], short_r[i]).size();
		return acc;
	};

	for (std::size_t const m : {std::size_t(1) << 20, (std::size_t(1) << 20) + 1}) {
		std::vector<long long> bl(m);
		std::vector<std::uint64_t> bg(m);
		for (std::size_t i = 0; i < m; ++i) {
			bl[i] = vll[i % vll.size()];
			bg[i] = vgcd[i % vgcd.size()];
		}
		std::string const tag = m == (std::size_t(1) << 20) ? "2^20" : "2^20+1";
		REQUIRE((atl::segment_tree<long long, std::plus<>>(bl, 0LL).fold() == segment_tree_by_full_build_and_recomputed_search<long long, std::plus<>>(bl, 0LL).fold()));
		REQUIRE((atl::segment_tree<std::uint64_t, gcd_op>(bg, std::uint64_t(0)).fold() == segment_tree_by_full_build_and_recomputed_search<std::uint64_t, gcd_op>(bg, std::uint64_t(0)).fold()));
		BENCHMARK("build n=" + tag + ", ll sum (library)") { return atl::segment_tree<long long, std::plus<>>(bl, 0LL).fold(); };
		BENCHMARK("build n=" + tag + ", ll sum (by full build and recomputed search)") { return segment_tree_by_full_build_and_recomputed_search<long long, std::plus<>>(bl, 0LL).fold(); };
		BENCHMARK("build n=" + tag + ", gcd (library)") { return atl::segment_tree<std::uint64_t, gcd_op>(bg, std::uint64_t(0)).fold(); };
		BENCHMARK("build n=" + tag + ", gcd (by full build and recomputed search)") { return segment_tree_by_full_build_and_recomputed_search<std::uint64_t, gcd_op>(bg, std::uint64_t(0)).fold(); };
	}
}
