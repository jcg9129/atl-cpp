#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/experimental/ds/link_cut_tree_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	namespace ex = atl::experimental;
	using atl::detail::splay_layout_columnar;
	using atl::detail::splay_layout_interleaved;
	using atl::detail::splay_layout_split;

	constexpr uint32 none = atl::graph::none;
	constexpr ll mod = 1'000'003;

	/** @brief x -> a x + b mod p; folds compose in path order ("f, then g"), so order matters. */
	struct affine {
		ll a = 1, b = 0;

		bool operator==(affine const &) const = default;
	};

	constexpr affine then(affine const &f, affine const &g) { return {g.a * f.a % mod, (g.a * f.b + g.b) % mod}; }

	struct affine_then {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(f, g); }
	};

	affine power(affine f, ll k) {
		affine r{};
		for (; k > 0; k >>= 1, f = then(f, f))
			if (k & 1) r = then(r, f);
		return r;
	}

	/** @brief Tier T1: a non-commutative composite with a width, tags assigning one map to every vertex. */
	struct counted {
		affine f;
		ll count = 0;

		bool operator==(counted const &) const = default;
	};

	struct counted_op {
		counted operator()(counted const &x, counted const &y) const { return {then(x.f, y.f), x.count + y.count}; }
	};

	struct assign_tag {
		bool is_set = false;
		affine c;

		bool operator==(assign_tag const &) const = default;
	};

	struct assign_mapping {
		counted operator()(assign_tag const &t, counted const &x) const { return t.is_set ? counted{power(t.c, x.count), x.count} : x; }
	};

	struct assign_compose {
		assign_tag operator()(assign_tag const &f, assign_tag const &g) const { return f.is_set ? f : g; }
	};

	/** @brief Tier T2: sum with a width, affine tags (Tree II). */
	struct sum_count {
		ll sum = 0, count = 0;

		bool operator==(sum_count const &) const = default;
	};

	struct sum_op {
		constexpr sum_count operator()(sum_count const &x, sum_count const &y) const {
			return {(x.sum + y.sum) % mod, x.count + y.count};
		}
	};

	struct affine_mapping {
		constexpr sum_count operator()(affine const &f, sum_count const &x) const {
			return {(f.a * x.sum + f.b * x.count) % mod, x.count};
		}
	};

	/** @brief Tag composition, "g first, then f". */
	struct affine_compose {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(g, f); }
	};

	struct max_op {
		constexpr ll operator()(ll a, ll b) const { return (std::max)(a, b); }
	};

	/** @brief A dynamic forest under test: vertex count and connectivity at least. */
	template<typename Forest>
	concept forest_under_test = requires(Forest &t, uint32 u) {
		{ t.size() } -> std::convertible_to<std::size_t>;
		{ t.connected(u, u) } -> std::same_as<bool>;
	};

	/** @brief A contender wrapper: `type<tier arguments...>` names the class under test. */
	template<typename Kind>
	concept contender = requires { typename Kind::template type<>; };

	/**
	 * The certifying oracle: a parent array, explicit roots (parent `none`),
	 * values per vertex; make_root reverses the root path; paths by marking
	 * ancestors; folds and tags vertex by vertex in path order. O(n) per op.
	 */
	template<std::copyable V>
	struct naive_forest {
		std::vector<uint32> par;
		std::vector<V> val;

		naive_forest(uint32 n, V fill) : par(n, none), val(n, fill) {}

		uint32 root(uint32 v) const {
			while (par[v] != none) v = par[v];
			return v;
		}

		bool connected(uint32 u, uint32 v) const { return root(u) == root(v); }

		uint32 depth(uint32 v) const {
			uint32 d = 0;
			for (; par[v] != none; v = par[v]) ++d;
			return d;
		}

		void make_root(uint32 v) {
			uint32 prev = none;
			while (v != none) {
				uint32 const next = par[v];
				par[v] = prev;
				prev = v;
				v = next;
			}
		}

		void link(uint32 u, uint32 v) {
			make_root(u);
			par[u] = v;
		}

		bool has_edge(uint32 u, uint32 v) const { return par[u] == v || par[v] == u; }

		void cut(uint32 u, uint32 v) {
			if (par[u] == v)
				par[u] = none;
			else
				par[v] = none;
		}

		uint32 lca(uint32 u, uint32 v) const {
			if (!connected(u, v)) return none;
			std::vector<char> mark(par.size(), 0);
			for (uint32 x = u; x != none; x = par[x]) mark[x] = 1;
			for (uint32 x = v;; x = par[x])
				if (mark[x]) return x;
		}

		/** @brief The u -> v path's vertices in order. @pre connected */
		std::vector<uint32> path(uint32 u, uint32 v) const {
			uint32 const w = lca(u, v);
			std::vector<uint32> up, down;
			for (uint32 x = u; x != w; x = par[x]) up.push_back(x);
			up.push_back(w);
			for (uint32 x = v; x != w; x = par[x]) down.push_back(x);
			up.insert(up.end(), down.rbegin(), down.rend());
			return up;
		}

		uint32 ancestor(uint32 v, uint32 k) const {
			for (; k > 0 && v != none; --k) v = par[v];
			return v;
		}

		uint32 distance(uint32 u, uint32 v) const { return connected(u, v) ? uint32(path(u, v).size() - 1) : none; }

		uint32 path_vertex(uint32 u, uint32 v, uint32 k) const {
			if (!connected(u, v)) return none;
			auto const p = path(u, v);
			return k < p.size() ? p[k] : none;
		}

		template<atl::binary_operation<V> Op>
		V fold(uint32 u, uint32 v, V acc, Op const &op) const {
			for (uint32 const x : path(u, v)) acc = op(acc, val[x]);
			return acc;
		}

		template<atl::update_function<V> F>
		void apply(uint32 u, uint32 v, F const &f) {
			for (uint32 const x : path(u, v)) val[x] = f(val[x]);
		}

		std::vector<uint32> non_roots() const {
			std::vector<uint32> r;
			for (uint32 v = 0; v < par.size(); ++v)
				if (par[v] != none) r.push_back(v);
			return r;
		}
	};

	/** @brief The algebra a vertex-tier test runs: the oracle's fold and tag application, random values and tags. */
	template<std::copyable V, atl::binary_operation<V> Op, typename Tag = void, typename Mapping = void>
		requires(std::is_void_v<Tag> == std::is_void_v<Mapping>) && (std::is_void_v<Tag> || atl::action<Mapping, Tag, V>)
	struct algebra {
		V identity;
		Op op;
		std::conditional_t<std::is_void_v<Tag>, int, Tag> tag_identity{};
		std::conditional_t<std::is_void_v<Mapping>, int, Mapping> mapping{};
	};

	affine random_affine(std::mt19937_64 &rng) { return {ll(rng() % mod), ll(rng() % mod)}; }

	/**
	 * Random mixed operations on @p t against @p nf: every member the tier has,
	 * make_root interleaved with the rooted queries.
	 */
	template<forest_under_test Lct, std::copyable V, atl::binary_operation<V> Op, std::invocable<std::mt19937_64 &> RandomValue,
			 std::invocable<std::mt19937_64 &> RandomTag, std::copy_constructible Mapping>
	void drive_vertices(Lct &t, naive_forest<V> &nf, std::mt19937_64 &rng, int ops, V identity, Op const &op, RandomValue &&rv,
						RandomTag &&rt, Mapping const &mapping) {
		uint32 const n = uint32(nf.par.size());
		constexpr bool has_values = requires { t.fold(0u, 0u); };
		constexpr bool has_tags = requires { t.apply(0u, 0u, rt(rng)); };
		constexpr bool is_sized = requires { t.depth(0u); };
		auto pick = [&] { return uint32(rng() % n); };
		for (int i = 0; i < ops; ++i) {
			uint32 const u = pick(), v = pick();
			switch (rng() % 14) {
			case 0:
			case 1:
				if (!nf.connected(u, v)) {
					t.link(u, v);
					nf.link(u, v);
				} else {
					REQUIRE(t.connected(u, v));
				}
				break;
			case 2: {
				auto const children = nf.non_roots();
				if (children.empty()) break;
				uint32 const c = children[rng() % children.size()], p = nf.par[c];
				switch (rng() % 3) {
				case 0:
					t.cut(c, p);
					break;
				case 1:
					t.cut(p, c);
					break;
				default:
					t.cut_from_parent(c);
				}
				nf.cut(c, p);
				break;
			}
			case 3:
				t.make_root(u);
				nf.make_root(u);
				break;
			case 4:
				REQUIRE(t.connected(u, v) == nf.connected(u, v));
				break;
			case 5:
				REQUIRE(t.root(u) == nf.root(u));
				break;
			case 6:
				REQUIRE(t.parent(u) == nf.par[u]);
				break;
			case 7:
				REQUIRE(t.lca(u, v) == nf.lca(u, v));
				break;
			case 8:
				if constexpr (is_sized) {
					uint32 const k = uint32(rng() % (nf.depth(u) + 3));
					switch (rng() % 4) {
					case 0:
						REQUIRE(t.depth(u) == nf.depth(u));
						break;
					case 1:
						REQUIRE(t.distance(u, v) == nf.distance(u, v));
						break;
					case 2:
						REQUIRE(t.ancestor(u, k) == nf.ancestor(u, k));
						break;
					default: {
						uint32 const d = nf.distance(u, v);
						uint32 const j = uint32(rng() % (d == none ? 3 : d + 3));
						REQUIRE(t.path_vertex(u, v, j) == nf.path_vertex(u, v, j));
					}
					}
				}
				break;
			case 9:
			case 10:
				if constexpr (has_values) {
					switch (rng() % 3) {
					case 0:
						REQUIRE(t.at(u) == nf.val[u]);
						break;
					case 1: {
						V const x = rv(rng);
						t.set(u, x);
						nf.val[u] = x;
						break;
					}
					default: {
						V const x = rv(rng);
						auto const f = [&](V const &a) { return op(a, x); };
						t.update(u, f);
						nf.val[u] = f(nf.val[u]);
					}
					}
				}
				break;
			case 11:
				if constexpr (has_values)
					if (nf.connected(u, v)) REQUIRE(t.fold(u, v) == nf.fold(u, v, identity, op));
				break;
			case 12:
				if constexpr (has_values) {
					auto const folded = t.try_fold(u, v);
					REQUIRE(folded.has_value() == nf.connected(u, v));
					if (folded) REQUIRE(*folded == nf.fold(u, v, identity, op));
				}
				break;
			default:
				if constexpr (has_tags) {
					if (nf.connected(u, v)) {
						auto const f = rt(rng);
						t.apply(u, v, f);
						nf.apply(u, v, [&](V const &a) { return mapping(f, a); });
					}
				}
			}
		}
		// a final sweep: every vertex and every rooted relation
		for (uint32 v = 0; v < n; ++v) {
			REQUIRE(t.parent(v) == nf.par[v]);
			if constexpr (is_sized) REQUIRE(t.depth(v) == nf.depth(v));
			if constexpr (has_values) REQUIRE(t.at(v) == nf.val[v]);
		}
	}

	/** @brief A random forest as a parent array (roughly a quarter of the vertices roots). */
	std::vector<uint32> random_parents(uint32 n, std::mt19937_64 &rng) {
		std::vector<uint32> order(n);
		for (uint32 i = 0; i < n; ++i) order[i] = i;
		std::ranges::shuffle(order, rng);
		std::vector<uint32> parents(n, none);
		for (uint32 i = 1; i < n; ++i)
			if (rng() % 4 != 0) parents[order[i]] = order[rng() % i];
		return parents;
	}

	/** @brief Run one kind (a template taking the tier arguments) on every tier it supports. */
	template<contender Kind, bool HasTags = true>
	void run_vertex_tiers(int seeds, int ops) {
		for (int seed = 0; seed < seeds; ++seed) {
			std::mt19937_64 rng(uint64(seed) * 7919 + 1);
			uint32 const n = 1 + uint32(rng() % 200);
			auto const no_value = [](std::mt19937_64 &) { return 0; };
			auto const no_mapping = [](int, int) { return 0; };
			// structure only
			{
				typename Kind::template type<> t(n);
				naive_forest<int> nf(n, 0);
				if (seed % 2 == 0) {
					auto const parents = random_parents(n, rng);
					t.assign_parents(parents);
					for (uint32 v = 0; v < n; ++v) nf.par[v] = parents[v];
				}
				drive_vertices(t, nf, rng, ops, 0, std::plus<>{}, no_value, no_value, no_mapping);
			}
			// non-commutative values: affine composite
			{
				std::vector<affine> values(n);
				for (auto &x : values) x = random_affine(rng);
				typename Kind::template type<affine, affine_then> t(values, affine{}, affine_then{});
				naive_forest<affine> nf(n, affine{});
				nf.val = values;
				if (seed % 3 == 0) {
					auto const parents = random_parents(n, rng);
					t.assign_parents(parents);
					for (uint32 v = 0; v < n; ++v) nf.par[v] = parents[v];
				}
				drive_vertices(t, nf, rng, ops, affine{}, affine_then{}, random_affine, no_value,
							   [](int, affine const &a) { return a; });
			}
			// commutative values in both fold-order modes
			{
				std::vector<ll> values(n);
				for (auto &x : values) x = ll(rng() % 1000);
				typename Kind::template type<ll, std::plus<>> t(values, 0LL, std::plus<>{});
				typename Kind::template type<ll, std::plus<>, void, void, void, atl::commutative_fold_t> c(values, 0LL, std::plus<>{});
				naive_forest<ll> nf(n, 0), nc(n, 0);
				nf.val = nc.val = values;
				auto const rv = [](std::mt19937_64 &r) { return ll(r() % 1000); };
				std::mt19937_64 rng2 = rng;
				drive_vertices(t, nf, rng, ops, 0LL, std::plus<>{}, rv, no_value, [](int, ll a) { return a; });
				drive_vertices(c, nc, rng2, ops, 0LL, std::plus<>{}, rv, no_value, [](int, ll a) { return a; });
			}
			if constexpr (HasTags) {
				// T1: non-commutative composite + assign tags
				{
					std::vector<counted> values(n);
					for (auto &x : values) x = {random_affine(rng), 1};
					typename Kind::template type<counted, counted_op, assign_tag, assign_mapping, assign_compose> t(values, counted{}, assign_tag{});
					naive_forest<counted> nf(n, counted{});
					nf.val = values;
					auto const rv = [](std::mt19937_64 &r) { return counted{random_affine(r), 1}; };
					auto const rt = [](std::mt19937_64 &r) {
						return r() % 4 == 0 ? assign_tag{} : assign_tag{true, random_affine(r)};
					};
					drive_vertices(t, nf, rng, ops, counted{}, counted_op{}, rv, rt, assign_mapping{});
				}
				// T2: sum + affine tags, both fold-order modes
				{
					std::vector<sum_count> values(n);
					for (auto &x : values) x = {ll(rng() % mod), 1};
					typename Kind::template type<sum_count, sum_op, affine, affine_mapping, affine_compose> t(values, sum_count{}, affine{});
					typename Kind::template type<sum_count, sum_op, affine, affine_mapping, affine_compose, atl::commutative_fold_t> c(values, sum_count{},
																																	   affine{});
					naive_forest<sum_count> nf(n, sum_count{}), nc(n, sum_count{});
					nf.val = nc.val = values;
					auto const rv = [](std::mt19937_64 &r) { return sum_count{ll(r() % mod), 1}; };
					auto const rt = [](std::mt19937_64 &r) { return random_affine(r); };
					std::mt19937_64 rng2 = rng;
					drive_vertices(t, nf, rng, ops, sum_count{}, sum_op{}, rv, rt, affine_mapping{});
					drive_vertices(c, nc, rng2, ops, sum_count{}, sum_op{}, rv, rt, affine_mapping{});
				}
			}
		}
	}

	/** @brief A contender: `type<tier arguments...>` is the class under test. */
	struct library {
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
				 typename FoldOrder = void, typename ValuePlacement = void>
		using type = atl::link_cut_tree<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement>;
	};

	template<typename Engine, atl::detail::splay_layout Layout, bool IsSized, bool IsEvertForm>
	struct variant {
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
				 typename FoldOrder = void, typename ValuePlacement = void>
		using type = ex::link_cut_tree_by_configuration<Engine, Layout, IsSized, IsEvertForm, Value, Op, Tag, Mapping, Compose, FoldOrder,
														ValuePlacement>;
	};

	using evert_form = variant<atl::splay_engine, splay_layout_interleaved, true, true>;
	using unsized = variant<atl::splay_engine, splay_layout_interleaved, false, false>;
	using split_layout = variant<atl::splay_engine, splay_layout_split, true, false>;
	using columnar_layout = variant<atl::splay_engine, splay_layout_columnar, true, false>;
	using parity_walk = variant<ex::link_cut_parity_engine, splay_layout_interleaved, true, false>;
	using pointer_reversal = variant<ex::link_cut_pointer_reversal_engine, splay_layout_interleaved, true, false>;
}

TEST_CASE("link_cut_tree: every tier against the naive forest", "[link_cut_tree]") { run_vertex_tiers<library>(60, 3000); }

TEST_CASE("link_cut_tree contenders against the naive forest", "[link_cut_tree]") {
	run_vertex_tiers<evert_form>(15, 2000);
	run_vertex_tiers<unsized>(15, 2000);
	run_vertex_tiers<split_layout>(15, 2000);
	run_vertex_tiers<columnar_layout>(15, 2000);
	run_vertex_tiers<parity_walk, false>(15, 2000);
	run_vertex_tiers<pointer_reversal>(15, 2000);
}

namespace {
	/**
	 * The edge tier's oracle: a parent array with the weight and the id of
	 * each vertex's parent edge; make_root moves both along the reversed path.
	 */
	template<std::copyable V>
	struct naive_edge_forest : naive_forest<int> {
		std::vector<V> weight; ///< weight[v]: the edge v - par[v]
		std::vector<uint32> id;

		explicit naive_edge_forest(uint32 n) : naive_forest<int>(n, 0), weight(n), id(n, none) {}

		void make_root(uint32 v) {
			uint32 prev = none;
			V w{};
			uint32 e = none;
			while (v != none) {
				uint32 const next = par[v];
				V const nw = weight[v];
				uint32 const ne = id[v];
				par[v] = prev;
				weight[v] = w;
				id[v] = e;
				prev = v;
				w = nw;
				e = ne;
				v = next;
			}
		}

		void link(uint32 u, uint32 v, V w, uint32 e) {
			make_root(u);
			par[u] = v;
			weight[u] = w;
			id[u] = e;
		}

		/** @brief The child endpoint of edge u - v. @pre it exists */
		uint32 child_of(uint32 u, uint32 v) const { return par[u] == v ? u : v; }

		template<atl::binary_operation<V> Op>
		V fold(uint32 u, uint32 v, V acc, Op const &op) const {
			auto const p = path(u, v);
			for (std::size_t i = 0; i + 1 < p.size(); ++i) acc = op(acc, weight[child_of(p[i], p[i + 1])]);
			return acc;
		}

		template<atl::update_function<V> F>
		void apply(uint32 u, uint32 v, F const &f) {
			auto const p = path(u, v);
			for (std::size_t i = 0; i + 1 < p.size(); ++i) {
				uint32 const c = child_of(p[i], p[i + 1]);
				weight[c] = f(weight[c]);
			}
		}

		uint32 child_of_edge(uint32 e) const {
			for (uint32 v = 0; v < par.size(); ++v)
				if (par[v] != none && id[v] == e) return v;
			return none;
		}
	};

	template<forest_under_test Lct, std::copyable V, atl::binary_operation<V> Op, std::invocable<std::mt19937_64 &> RandomValue,
			 std::invocable<std::mt19937_64 &> RandomTag, std::copy_constructible Mapping>
	void drive_edges(Lct &t, naive_edge_forest<V> &nf, std::mt19937_64 &rng, int ops, V identity, Op const &op,
					 RandomValue &&rv, RandomTag &&rt, Mapping const &mapping) {
		uint32 const n = uint32(nf.par.size());
		constexpr bool has_tags = requires { t.apply(0u, 0u, rt(rng)); };
		auto pick = [&] { return uint32(rng() % n); };
		for (int i = 0; i < ops; ++i) {
			uint32 const u = pick(), v = pick();
			switch (rng() % 14) {
			case 0:
			case 1:
				if (!nf.connected(u, v)) {
					V const w = rv(rng);
					uint32 const e = t.link(u, v, w);
					REQUIRE(e < n - 1);
					REQUIRE(nf.child_of_edge(e) == none); // a free id
					nf.link(u, v, w, e);
				}
				break;
			case 2: {
				auto const children = nf.non_roots();
				if (children.empty()) break;
				uint32 const c = children[rng() % children.size()], p = nf.par[c];
				REQUIRE(t.edge(c, p) == nf.id[c]);
				REQUIRE(t.edge(p, c) == nf.id[c]);
				switch (rng() % 4) {
				case 0:
					t.cut(c, p);
					break;
				case 1:
					t.cut(p, c);
					break;
				case 2:
					t.cut(nf.id[c]);
					break;
				default:
					t.cut_from_parent(c);
				}
				nf.cut(c, p);
				nf.id[c] = none;
				REQUIRE(t.edge(c, p) == none);
				break;
			}
			case 3:
				t.make_root(u);
				nf.make_root(u);
				break;
			case 4:
				REQUIRE(t.connected(u, v) == nf.connected(u, v));
				REQUIRE(t.edge_count() == nf.non_roots().size());
				break;
			case 5:
				REQUIRE(t.root(u) == nf.root(u));
				REQUIRE(t.parent(u) == nf.par[u]);
				break;
			case 6:
				REQUIRE(t.lca(u, v) == nf.lca(u, v));
				break;
			case 7: {
				uint32 const k = uint32(rng() % (nf.depth(u) + 3));
				uint32 const d = nf.distance(u, v);
				uint32 const j = uint32(rng() % (d == none ? 3 : d + 3));
				REQUIRE(t.depth(u) == nf.depth(u));
				REQUIRE(t.distance(u, v) == d);
				REQUIRE(t.ancestor(u, k) == nf.ancestor(u, k));
				REQUIRE(t.path_vertex(u, v, j) == nf.path_vertex(u, v, j));
				break;
			}
			case 8:
			case 9: {
				auto const children = nf.non_roots();
				if (children.empty()) break;
				uint32 const c = children[rng() % children.size()], e = nf.id[c];
				switch (rng() % 3) {
				case 0:
					REQUIRE(t.at(e) == nf.weight[c]);
					break;
				case 1: {
					V const x = rv(rng);
					t.set(e, x);
					nf.weight[c] = x;
					break;
				}
				default: {
					V const x = rv(rng);
					auto const f = [&](V const &a) { return op(a, x); };
					t.update(e, f);
					nf.weight[c] = f(nf.weight[c]);
				}
				}
				break;
			}
			case 10:
				if (nf.connected(u, v)) REQUIRE(t.fold(u, v) == nf.fold(u, v, identity, op));
				break;
			case 11: {
				auto const folded = t.try_fold(u, v);
				REQUIRE(folded.has_value() == nf.connected(u, v));
				if (folded) REQUIRE(*folded == nf.fold(u, v, identity, op));
				break;
			}
			default:
				if constexpr (has_tags) {
					if (nf.connected(u, v)) {
						auto const f = rt(rng);
						t.apply(u, v, f);
						nf.apply(u, v, [&](V const &a) { return mapping(f, a); });
					}
				}
			}
		}
		for (uint32 v = 0; v < n; ++v) {
			REQUIRE(t.parent(v) == nf.par[v]);
			REQUIRE(t.depth(v) == nf.depth(v));
			if (nf.par[v] != none) REQUIRE(t.at(nf.id[v]) == nf.weight[v]);
		}
	}

	template<contender Kind>
	void run_edge_tiers(int seeds, int ops) {
		for (int seed = 0; seed < seeds; ++seed) {
			std::mt19937_64 rng(uint64(seed) * 104729 + 3);
			uint32 const n = 1 + uint32(rng() % 200);
			auto const no_tag = [](std::mt19937_64 &) { return 0; };
			// non-commutative edge composite
			{
				typename Kind::template type<affine, affine_then, void, void, void, void, atl::edge_values_t> t(n, affine{}, affine_then{});
				naive_edge_forest<affine> nf(n);
				if (seed % 2 == 0) {
					auto const parents = random_parents(n, rng);
					std::vector<affine> weights(n);
					for (auto &w : weights) w = random_affine(rng);
					t.assign_parents(parents, weights);
					uint32 next = 0;
					for (uint32 v = 0; v < n; ++v)
						if (parents[v] != none) {
							nf.par[v] = parents[v];
							nf.weight[v] = weights[v];
							nf.id[v] = next++;
						}
				}
				drive_edges(t, nf, rng, ops, affine{}, affine_then{}, random_affine, no_tag,
							[](int, affine const &a) { return a; });
			}
			// path max, commutative (the dynamic MST shape)
			{
				typename Kind::template type<ll, max_op, void, void, void, atl::commutative_fold_t, atl::edge_values_t> t(n, ll(-1), max_op{});
				naive_edge_forest<ll> nf(n);
				auto const rv = [](std::mt19937_64 &r) { return ll(r() % 1'000'000); };
				drive_edges(t, nf, rng, ops, ll(-1), max_op{}, rv, no_tag, [](int, ll a) { return a; });
			}
			// sum + affine tags on edges (mapping fixes the identity {0, 0})
			{
				typename Kind::template type<sum_count, sum_op, affine, affine_mapping, affine_compose, void, atl::edge_values_t> t(n, sum_count{},
																																	affine{});
				naive_edge_forest<sum_count> nf(n);
				auto const rv = [](std::mt19937_64 &r) { return sum_count{ll(r() % mod), 1}; };
				auto const rt = [](std::mt19937_64 &r) { return random_affine(r); };
				drive_edges(t, nf, rng, ops, sum_count{}, sum_op{}, rv, rt, affine_mapping{});
			}
		}
	}
}

TEST_CASE("link_cut_tree edge tier against the naive edge-weighted forest", "[link_cut_tree]") {
	run_edge_tiers<library>(60, 3000);
	run_edge_tiers<evert_form>(10, 2000);
	run_edge_tiers<split_layout>(10, 2000);
	run_edge_tiers<columnar_layout>(10, 2000);
}

TEST_CASE("link_cut_tree: deduction, small cases, empty forest", "[link_cut_tree]") {
	atl::link_cut_tree empty;
	CHECK(empty.empty());
	CHECK(empty.size() == 0);
	static_assert(std::same_as<decltype(empty), atl::link_cut_tree<>>);

	atl::link_cut_tree one(1);
	CHECK(one.root(0) == 0);
	CHECK(one.parent(0) == none);
	CHECK(one.depth(0) == 0);
	CHECK(one.lca(0, 0) == 0);
	CHECK(one.ancestor(0, 1) == none);
	CHECK(one.path_vertex(0, 0, 0) == 0);
	CHECK(one.path_vertex(0, 0, 1) == none);

	std::vector<int> const values{3, 1, 4, 1, 5};
	atl::link_cut_tree sums(values, ll(0), std::plus<>{});
	static_assert(std::same_as<decltype(sums), atl::link_cut_tree<ll, std::plus<>>>);
	sums.link(0, 1);
	sums.link(1, 2);
	sums.link(4, 2);
	CHECK(sums.fold(0, 4) == 3 + 1 + 4 + 5);
	CHECK(sums.try_fold(4, 0) == std::optional<ll>(5 + 4 + 1 + 3));
	CHECK(sums.try_fold(3, 3) == std::optional<ll>(1));
	CHECK(!sums.try_fold(3, 0).has_value());
	CHECK(!sums.try_fold(0, 3).has_value());
	CHECK(sums.root(0) == 2); // u's side joins below v: link keeps v's root
	CHECK(sums.distance(3, 0) == none);
	CHECK(sums.lca(3, 0) == none);
	CHECK(sums.path_vertex(3, 0, 0) == none);

	atl::link_cut_tree commutative(atl::commutative_fold, std::size_t(3), ll(0), std::plus<>{});
	static_assert(
		std::same_as<decltype(commutative), atl::link_cut_tree<ll, std::plus<>, void, void, void, atl::commutative_fold_t>>);

	atl::link_cut_tree tagged(std::vector<sum_count>(4, {0, 1}), sum_count{}, affine{}, sum_op{}, affine_mapping{}, affine_compose{});
	static_assert(std::same_as<decltype(tagged), atl::link_cut_tree<sum_count, sum_op, affine, affine_mapping, affine_compose>>);
	tagged.link(0, 1);
	tagged.link(2, 1);
	tagged.apply(0, 2, affine{1, 5});
	CHECK(tagged.fold(0, 2) == sum_count{15, 3});
	CHECK(tagged.at(3) == sum_count{0, 1});

	atl::link_cut_tree edges(atl::edge_values, 3, ll(0), std::plus<>{});
	static_assert(std::same_as<decltype(edges), atl::edge_link_cut_tree<ll, std::plus<>>>);
	uint32 const e01 = edges.link(0, 1, 10);
	uint32 const e12 = edges.link(1, 2, 20);
	CHECK(e01 == 0);
	CHECK(e12 == 1);
	CHECK(edges.fold(0, 2) == 30);
	CHECK(edges.fold(1, 1) == 0);
	CHECK(edges.try_fold(2, 0) == std::optional<ll>(30));
	CHECK(edges.try_fold(1, 1) == std::optional<ll>(0));
	CHECK(edges.edge(2, 1) == e12);
	CHECK(edges.edge(0, 2) == none);
	edges.cut(e01);
	CHECK(edges.edge_count() == 1);
	CHECK(edges.link(2, 0, 7) == e01); // a freed id is reused
	CHECK(edges.fold(0, 1) == 27);

	atl::link_cut_tree edge_comm(atl::edge_values, atl::commutative_fold, 2, ll(0), max_op{});
	static_assert(std::same_as<decltype(edge_comm), atl::link_cut_tree<ll, max_op, void, void, void, atl::commutative_fold_t,
																	   atl::edge_values_t>>);
}

TEST_CASE("link_cut_tree: a 1e6-vertex path (no recursion, deep splays)", "[link_cut_tree]") {
	uint32 const n = 1'000'000;
	std::vector<uint32> parents(n);
	for (uint32 v = 0; v < n; ++v) parents[v] = v == 0 ? none : v - 1;
	atl::link_cut_tree<ll, std::plus<>> t(std::vector<ll>(n, 1), 0LL, std::plus<>{});
	t.assign_parents(parents);
	CHECK(t.depth(n - 1) == n - 1);
	CHECK(t.fold(n - 1, 0) == ll(n));
	CHECK(t.ancestor(n - 1, n - 1) == 0);
	CHECK(t.path_vertex(0, n - 1, n / 2) == n / 2);
	t.make_root(n - 1);
	CHECK(t.depth(0) == n - 1);
	CHECK(t.root(0) == n - 1);
	CHECK(t.lca(0, n / 3) == n / 3);
	t.cut(n / 2, n / 2 + 1);
	CHECK(!t.connected(0, n - 1));
	t.link(n / 2, n - 1); // a cycle-free relink of the halves
	CHECK(t.distance(0, n / 2 + 1) == n / 2 + n / 2 - 1);

	// built by links, far ends alternating
	atl::link_cut_tree s(n);
	for (uint32 v = 1; v < n; ++v) s.link(v, v - 1);
	for (int i = 0; i < 20; ++i) {
		CHECK(s.depth(i % 2 ? n - 1 : 1) == (i % 2 ? n - 1 : 1));
		s.make_root(i % 2 ? n - 1 : 0);
		s.make_root(0);
	}

	atl::link_cut_tree e(atl::edge_values, n, ll(0), std::plus<>{});
	std::vector<ll> weights(n, 2);
	e.assign_parents(parents, weights);
	CHECK(e.fold(0, n - 1) == 2 * ll(n - 1));
	CHECK(e.depth(n - 1) == n - 1);
}

TEST_CASE("link_cut_tree: preconditions abort", "[link_cut_tree]") {
	auto const make = [] {
		atl::link_cut_tree<ll, std::plus<>> t(std::size_t(4), 0LL);
		t.link(0, 1);
		t.link(1, 2);
		return t;
	};
	CHECK(atl_test::aborts([&] { make().link(0, 2); }));										 // same tree
	CHECK(atl_test::aborts([&] { make().link(0, 0); }));										 // same vertex
	CHECK(atl_test::aborts([&] { make().cut(0, 2); }));											 // not an edge
	CHECK(atl_test::aborts([&] { make().cut(0, 3); }));											 // across trees
	CHECK(atl_test::aborts([&] { make().cut_from_parent(2); }));								 // the root
	CHECK(atl_test::aborts([&] { (void)make().fold(0, 3); }));									 // across trees
	CHECK(atl_test::aborts([&] { (void)make().root(4); }));										 // out of range
	CHECK(atl_test::aborts([&] { make().make_root(7); }));										 // out of range
	CHECK(atl_test::aborts([&] { make().assign_parents(std::vector<uint32>{none, 0, 1, 2}); })); // not isolated
	CHECK(atl_test::aborts([] {
		atl::link_cut_tree t(3);
		t.assign_parents(std::vector<uint32>{1, 2, 0}); // a cycle
	}));
	CHECK(atl_test::aborts([] {
		atl::link_cut_tree t(3);
		t.assign_parents(std::vector<uint32>{none, 0}); // too short
	}));
	auto const make_edges = [] {
		atl::link_cut_tree t(atl::edge_values, 4, 0LL, std::plus<>{});
		t.link(0, 1, 5);
		return t;
	};
	CHECK(atl_test::aborts([&] { make_edges().cut(1, 2); }));	  // not an edge
	CHECK(atl_test::aborts([&] { make_edges().cut(2u); }));		  // a free id
	CHECK(atl_test::aborts([&] { (void)make_edges().at(1); }));	  // a free id
	CHECK(atl_test::aborts([&] { make_edges().link(1, 0, 3); })); // same tree
}
