/**
 * @file
 * @brief Range-action and range-fold stream throughput.
 *
 * Recursion uses top-down propagation; the push-policy contender visits identity tags. Each sample constructs a fresh tree and replays add/sum or affine operations, including apply-heavy, query-heavy, point and boundary-search streams at n=1e3 and n=1e6.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, -O2 and -O2 -march=native, 2026-10-15: mixed library 470/444 ms versus recursion 751/750 ms.
 * Full measurements: docs/lazy-segment-tree.md.
 */
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using ll = long long;

/** @brief Range addition and sums by top-down recursion in a 4n array. */
struct lazy_segment_tree_by_recursion {
	int n;
	std::vector<ll> t, lz;

	explicit lazy_segment_tree_by_recursion(int size) : n(size), t(4 * size, 0), lz(4 * size, 0) {}

	void push(int nd, int lo, int hi) {
		if (!lz[nd]) return;
		int const m = (lo + hi) / 2;
		t[2 * nd] += lz[nd] * (m - lo);
		lz[2 * nd] += lz[nd];
		t[2 * nd + 1] += lz[nd] * (hi - m);
		lz[2 * nd + 1] += lz[nd];
		lz[nd] = 0;
	}

	void upd(int nd, int lo, int hi, int l, int r, ll f) {
		if (r <= lo || hi <= l) return;
		if (l <= lo && hi <= r) {
			t[nd] += f * (hi - lo);
			lz[nd] += f;
			return;
		}
		push(nd, lo, hi);
		int const m = (lo + hi) / 2;
		upd(2 * nd, lo, m, l, r, f);
		upd(2 * nd + 1, m, hi, l, r, f);
		t[nd] = t[2 * nd] + t[2 * nd + 1];
	}

	ll qry(int nd, int lo, int hi, int l, int r) {
		if (r <= lo || hi <= l) return 0;
		if (l <= lo && hi <= r) return t[nd];
		push(nd, lo, hi);
		int const m = (lo + hi) / 2;
		return qry(2 * nd, lo, m, l, r) + qry(2 * nd + 1, m, hi, l, r);
	}

	void apply(int l, int r, ll f) { upd(1, 0, n, l, r, f); }

	ll fold(int l, int r) { return qry(1, 0, n, l, r); }
};

// Iterative (atl::lazy_segment_tree) vs a recursive lazy tree, range add / range
// sum. Each sample builds a fresh tree and runs the same mixed 50/50 op stream:
// lazy-tree cost depends on the pending-tag state, so applies and queries must be
// interleaved from a known state to measure identical work per sample.
TEST_CASE("lazy_segment_tree benchmarks", "[lazy_segment_tree]") {
	constexpr int n = 1'000'000, m = 1'000'000;
	std::mt19937 rng(1);
	std::vector<int> lo(m), hi(m);
	std::vector<ll> val(m);
	std::vector<char> is_apply(m);
	for (int i = 0; i < m; ++i) {
		int a = int(rng() % n), b = int(rng() % n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
		val[i] = ll(rng() % 100);
		is_apply[i] = char(rng() % 2);
	}

	using node = std::pair<ll, ll>; // (sum, width)

	BENCHMARK("library -- 1e6 mixed apply/query") {
		atl::lazy_segment_tree st(
			std::vector<node>(n, node{0, 1}), node{0, 0}, ll(0),
			[](node const &x, node const &y) { return node{x.first + y.first, x.second + y.second}; },
			[](ll f, node const &s) { return node{s.first + f * s.second, s.second}; },
			[](ll f, ll g) { return f + g; });
		std::uint64_t acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				st.apply(std::size_t(lo[i]), std::size_t(hi[i]), val[i]);
			else
				acc += static_cast<std::uint64_t>(st.fold(std::size_t(lo[i]), std::size_t(hi[i])).first);
		return acc;
	};
	BENCHMARK("by recursion -- 1e6 mixed apply/query") {
		lazy_segment_tree_by_recursion st(n);
		std::uint64_t acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				st.apply(lo[i], hi[i], val[i]);
			else
				acc += static_cast<std::uint64_t>(st.fold(lo[i], hi[i]));
		return acc;
	};
}

namespace {
	/**
	 * @brief lazy_segment_tree engine with an identity-tag switch: @p SkipIdentity makes
	 *        push_down return early on an identity tag (library).
	 */
	template<bool SkipIdentity, std::copyable S, std::copyable F, atl::binary_operation<S> Op, atl::action<F, S> Mapping, atl::binary_operation<F> Compose>
	class lazy_segment_tree_by_push_policy : atl::detail::segment_tree_base<S> {
		using base = atl::detail::segment_tree_base<S>;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::size_;
		Op op_;
		Mapping mapping_;
		Compose compose_;
		S e_;
		F id_;
		std::vector<F> tags_;

		void apply_tag(std::size_t k, F const &f) {
			nodes_[k] = mapping_(f, nodes_[k]);
			if (k < size_) tags_[k] = compose_(f, tags_[k]);
		}

		void push_down(std::size_t k) {
			if constexpr (SkipIdentity)
				if (tags_[k] == id_) return;
			apply_tag(2 * k, tags_[k]);
			apply_tag(2 * k + 1, tags_[k]);
			tags_[k] = id_;
		}

		void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		void recompute(std::size_t k) { nodes_[k] = op_(nodes_[2 * k], nodes_[2 * k + 1]); }

	public:
		lazy_segment_tree_by_push_policy(std::vector<S> const &values, S e, F id) : e_(std::move(e)), id_(std::move(id)) {
			this->reset(values.size(), e_);
			tags_.assign(size_, id_);
			for (std::size_t i = 0; i < values.size(); ++i) nodes_[leaf(i)] = values[i];
			for (std::size_t k = size_; k-- > 1;) recompute(k);
		}

		void set(std::size_t i, S x) {
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = std::move(x);
			this->recompute_up(k, op_);
		}

		S const &at(std::size_t i) {
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)];
		}

		void apply(std::size_t i, F const &f) {
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = mapping_(f, nodes_[k]);
			this->recompute_up(k, op_);
		}

		void apply(std::size_t l, std::size_t r, F const &f) {
			if (l == r) return;
			this->descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			this->decompose(l, r, [&](std::size_t k) { apply_tag(k, f); }, [&](std::size_t k) { apply_tag(k, f); });
			this->ascend_boundaries(l, r, [&](std::size_t k) { recompute(k); });
		}

		S fold(std::size_t l, std::size_t r) {
			if (l == r) return e_;
			this->descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			S sl = e_, sr = e_;
			this->decompose(l, r, [&](std::size_t k) { sl = op_(sl, nodes_[k]); }, [&](std::size_t k) { sr = op_(nodes_[k], sr); });
			return op_(sl, sr);
		}

		template<std::predicate<S const &> Pred>
		std::size_t max_right(std::size_t l, Pred pred) {
			if (l == n_) return n_;
			l += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down(l >> i);
			S sm = e_;
			do {
				while (l % 2 == 0) l >>= 1;
				if (!pred(op_(sm, nodes_[l]))) {
					while (l < size_) {
						push_down(l);
						l *= 2;
						if (S t = op_(sm, nodes_[l]); pred(t)) sm = std::move(t), ++l;
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

	/** @brief Sum and width for additive range actions. */
	struct add_node {
		ll sum = 0, width = 0;
	};

	struct add_op {
		add_node operator()(add_node const &x, add_node const &y) const { return {x.sum + y.sum, x.width + y.width}; }
	};

	struct add_map {
		add_node operator()(ll f, add_node const &s) const { return {s.sum + f * s.width, s.width}; }
	};

	struct add_compose {
		ll operator()(ll f, ll g) const { return f + g; }
	};

	/** @brief Sum modulo p and width for affine range actions. */
	struct aff_node {
		std::uint64_t sum = 0, width = 0;
	};

	struct affine {
		std::uint64_t a = 1, b = 0;
		bool operator==(affine const &) const = default;
	};

	struct aff_op {
		aff_node operator()(aff_node const &x, aff_node const &y) const { return {(x.sum + y.sum) % mod, x.width + y.width}; }
	};

	struct aff_map {
		aff_node operator()(affine const &f, aff_node const &s) const { return {(f.a * s.sum + f.b * s.width) % mod, s.width}; }
	};

	struct aff_compose {
		affine operator()(affine const &f, affine const &g) const { return {f.a * g.a % mod, (f.a * g.b + f.b) % mod}; }
	};

	/** @brief Stream operation: range apply/fold, point set/read/apply, or boundary search. */
	struct op_t {
		int kind;
		std::size_t l, r;
		std::uint64_t v;
	};

	/** @brief @p m operations on `[0, n)`; @p mix gives the cumulative per-mille of each kind. */
	std::vector<op_t> make_stream(std::size_t n, std::size_t m, std::vector<int> const &mix, std::uint64_t seed) {
		std::mt19937_64 rng(seed);
		std::vector<op_t> ops(m);
		for (auto &o : ops) {
			int const roll = int(rng() % 1000);
			o.kind = 0;
			while (roll >= mix[std::size_t(o.kind)]) ++o.kind;
			std::size_t a = rng() % n, b = rng() % n;
			if (a > b) std::swap(a, b);
			o.l = a, o.r = b + 1, o.v = 1 + rng() % 100;
		}
		return ops;
	}

	template<class Node, class Tag, class Factory, class MakeTag, class MakeNode>
		requires std::invocable<Factory &> && std::is_invocable_r_v<Tag, MakeTag &, std::uint64_t>
			  && std::is_invocable_r_v<Node, MakeNode &, std::uint64_t>
			  && requires(std::invoke_result_t<Factory &> &tree, Node node, Tag tag) {
					 tree.apply(0, 1, tag);
					 tree.apply(0, tag);
					 tree.set(0, node);
					 { tree.fold(0, 1).sum } -> std::convertible_to<std::uint64_t>;
					 { tree.at(0).sum } -> std::convertible_to<std::uint64_t>;
					 tree.max_right(0, [](Node const &) { return true; });
				 }
	std::uint64_t replay_lazy(std::vector<op_t> const &ops, Factory factory, MakeTag make_tag, MakeNode make_node) {
		auto t = std::invoke(factory);
		std::uint64_t acc = 0;
		for (auto const &o : ops) switch (o.kind) {
			case 0:
				t.apply(o.l, o.r, make_tag(o.v));
				break;
			case 1:
				acc += std::uint64_t(t.fold(o.l, o.r).sum);
				break;
			case 2:
				t.set(o.l, make_node(o.v));
				break;
			case 3:
				acc += std::uint64_t(t.at(o.l).sum);
				break;
			case 4:
				t.apply(o.l, make_tag(o.v));
				break;
			default:
				acc += t.max_right(o.l, [&](Node const &s) { return std::uint64_t(s.sum) <= o.v * 20000; });
				break;
			}
		return acc;
	}
}

TEST_CASE("lazy_segment_tree identity-tag push comparison", "[lazy_segment_tree][race]") {
	struct shape {
		std::string name;
		std::size_t n, m;
		std::vector<int> mix;
	};

	std::vector<shape> const shapes{
		{"n=1e6 50/50 apply/fold", 1'000'000, 1'000'000, {500, 1000}},
		{"n=1e6 10/90 apply/fold", 1'000'000, 1'000'000, {100, 1000}},
		{"n=1e6 point set/at/apply + 10% range apply", 1'000'000, 1'000'000, {100, 100, 400, 700, 1000}},
		{"n=1e6 max_right + 10% range apply", 1'000'000, 1'000'000, {100, 100, 100, 100, 100, 1000}},
		{"n=1e3 50/50 apply/fold", 1'000, 1'000'000, {500, 1000}},
	};
	for (auto const &sh : shapes) {
		auto const ops = make_stream(sh.n, sh.m, sh.mix, 11);
		std::vector<add_node> const init_add(sh.n, add_node{0, 1});
		std::vector<aff_node> const init_aff(sh.n, aff_node{1, 1});
		auto add_tag = [](std::uint64_t v) { return ll(v); };
		auto add_leaf = [](std::uint64_t v) { return add_node{ll(v), 1}; };
		auto aff_tag = [](std::uint64_t v) { return affine{v, v + 1}; };
		auto aff_leaf = [](std::uint64_t v) { return aff_node{v, 1}; };
		auto library = [&] { return atl::lazy_segment_tree(init_add, add_node{}, ll(0), add_op{}, add_map{}, add_compose{}); };
		auto by_unconditional_push = [&] { return lazy_segment_tree_by_push_policy<false, add_node, ll, add_op, add_map, add_compose>(init_add, add_node{}, ll(0)); };
		REQUIRE((replay_lazy<add_node, ll>(ops, library, add_tag, add_leaf) == replay_lazy<add_node, ll>(ops, by_unconditional_push, add_tag, add_leaf)));
		BENCHMARK("add/sum " + sh.name + " (library)") {
			return replay_lazy<add_node, ll>(ops, library, add_tag, add_leaf);
		};
		BENCHMARK("add/sum " + sh.name + " (by unconditional push)") {
			return replay_lazy<add_node, ll>(ops, by_unconditional_push, add_tag, add_leaf);
		};
		if (sh.n < 1'000'000 || sh.mix.size() > 2) continue;
		auto affine_library = [&] { return atl::lazy_segment_tree(init_aff, aff_node{}, affine{}, aff_op{}, aff_map{}, aff_compose{}); };
		auto affine_by_unconditional_push = [&] { return lazy_segment_tree_by_push_policy<false, aff_node, affine, aff_op, aff_map, aff_compose>(init_aff, aff_node{}, affine{}); };
		REQUIRE((replay_lazy<aff_node, affine>(ops, affine_library, aff_tag, aff_leaf) == replay_lazy<aff_node, affine>(ops, affine_by_unconditional_push, aff_tag, aff_leaf)));
		BENCHMARK("affine mod p " + sh.name + " (library)") {
			return replay_lazy<aff_node, affine>(ops, affine_library, aff_tag, aff_leaf);
		};
		BENCHMARK("affine mod p " + sh.name + " (by unconditional push)") {
			return replay_lazy<aff_node, affine>(ops, affine_by_unconditional_push, aff_tag, aff_leaf);
		};
	}
}
