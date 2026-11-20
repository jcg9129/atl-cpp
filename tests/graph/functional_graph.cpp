#include <atl/experimental/graph/functional_graph_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/functional_graph.hpp>
#include <atl/graph/scc.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace e = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Finite transformations of three elements, including noninvertible maps. */
	struct transformation {
		std::array<uint32, 3> image;
		constexpr bool operator==(transformation const &) const = default;
	};

	/** @brief Traverse a first, then b; an order-sensitive finite monoid. */
	struct compose {
		constexpr transformation operator()(transformation const &a, transformation const &b) const {
			return {{b.image[a.image[0]], b.image[a.image[1]], b.image[a.image[2]]}};
		}
	};

	constexpr transformation transformation_identity{{0, 1, 2}};
	static_assert(compose{}({{1, 1, 1}}, {{2, 0, 0}}) != compose{}({{2, 0, 0}}, {{1, 1, 1}}));

	/** @brief Certifying oracle: enumerate (vertex, accumulated transformation) until repetition, then skip whole periods.
	 * No decomposition, doubling or monoid powering is used.
	 */
	g::functional_graph_walk_result<transformation> oracle(std::vector<uint32> const &successors,
														   std::vector<transformation> const &values, uint32 start, uint64 steps) {
		std::vector<uint64> seen(successors.size() * 27, (std::numeric_limits<uint64>::max)());
		std::vector<g::functional_graph_walk_result<transformation>> states;
		uint32 v = start;
		transformation fold = transformation_identity;
		while (true) {
			if (states.size() == steps) return {v, fold};
			auto const code = fold.image[0] + 3 * fold.image[1] + 9 * fold.image[2];
			auto &first = seen[27 * v + code];
			if (first != (std::numeric_limits<uint64>::max)()) {
				auto const index = first + (steps - first) % (states.size() - first);
				return states[std::size_t(index)];
			}
			first = states.size();
			states.push_back({v, fold});
			fold = compose{}(fold, values[v]);
			v = successors[v];
		}
	}

	/** @brief Independent per-source first-seen walks, normalized by minimum cycle vertex. */
	g::functional_graph_decomposition metadata_oracle(std::vector<uint32> const &successors) {
		auto const n = successors.size();
		g::functional_graph_decomposition out{std::vector<uint32>(n), std::vector<uint32>(n, g::none), std::vector<uint32>(n), std::vector<uint32>(n), {}};
		std::vector<uint32> minimum(n);
		for (uint32 start = 0; start < n; ++start) {
			std::vector<uint32> first(n, g::none), path;
			uint32 v = start;
			while (first[v] == g::none) {
				first[v] = uint32(path.size());
				path.push_back(v);
				v = successors[v];
			}
			uint32 const depth = first[v];
			out.depth_to_cycle[start] = depth;
			out.cycle_root[start] = v;
			minimum[start] = *std::ranges::min_element(path.begin() + depth, path.end());
		}
		std::vector<uint32> id(n, g::none);
		for (uint32 v = 0; v < n; ++v) {
			if (minimum[v] != v || out.depth_to_cycle[v]) continue;
			id[v] = uint32(out.cycles.count());
			uint32 at = v, position = 0;
			do {
				out.cycle_position[at] = position++;
				out.cycles.members.push_back(at);
				at = successors[at];
			} while (at != v);
			out.cycles.offsets.push_back(uint32(out.cycles.members.size()));
		}
		for (uint32 v = 0; v < n; ++v) out.cycle_id[v] = id[minimum[v]];
		return out;
	}

	void same_metadata(g::functional_graph_decomposition const &a, g::functional_graph_decomposition const &b) {
		REQUIRE(a.cycle_id == b.cycle_id);
		REQUIRE(a.cycle_position == b.cycle_position);
		REQUIRE(a.depth_to_cycle == b.depth_to_cycle);
		REQUIRE(a.cycle_root == b.cycle_root);
		REQUIRE(a.cycles.offsets == b.cycles.offsets);
		REQUIRE(a.cycles.members == b.cycles.members);
	}

	void check(std::vector<uint32> const &successors, std::vector<uint64> const &steps, bool should_check_scc = true) {
		auto const n = successors.size();
		std::vector<transformation> values;
		for (uint32 v = 0; v < n; ++v)
			values.push_back({{v % 3, (v / 3 + 2) % 3, (v + 1) % 3}});
		auto const expected_metadata = metadata_oracle(successors);
		auto const metadata = g::functional_graph_decomposition_of(successors);
		same_metadata(metadata, expected_metadata);
		same_metadata(e::functional_graph_decomposition_by_marking(successors), expected_metadata);
		if (should_check_scc) {
			std::vector<g::arc<>> arcs;
			for (uint32 v = 0; v < n; ++v) arcs.push_back({v, successors[v]});
			auto const partition = g::scc(g::csr<uint32>(n, arcs));
			auto const groups = partition.grouped();
			std::size_t cycle_count = 0;
			for (std::size_t id = 0; id < groups.count(); ++id) {
				auto const group = groups[id];
				bool const is_cycle = group.size() > 1 || successors[group[0]] == group[0];
				if (is_cycle) {
					++cycle_count;
					for (auto v : group) {
						REQUIRE(metadata.depth_to_cycle[v] == 0);
						REQUIRE(metadata.cycle_id[v] == metadata.cycle_id[group[0]]);
					}
				} else
					REQUIRE(metadata.depth_to_cycle[group[0]] > 0);
			}
			REQUIRE(cycle_count == metadata.cycles.count());
		}
		g::functional_graph_successor_index const endpoint(successors);
		e::functional_graph_successor_index_by_lifting const lifting_endpoint(successors);
		g::functional_graph_walk_index<transformation, compose> const public_walk(successors, values, transformation_identity, compose{});
		e::functional_graph_walk_index_by_lifting<transformation, compose> const lifting_walk(successors, values, transformation_identity, compose{});
		e::functional_graph_walk_index_by_chain_prefix<transformation, compose> const prefix_walk(successors, values, transformation_identity, compose{});
		REQUIRE(endpoint.size() == n);
		REQUIRE(public_walk.size() == n);
		same_metadata(endpoint.decomposition(), metadata);
		same_metadata(lifting_endpoint.decomposition(), metadata);
		same_metadata(public_walk.decomposition(), metadata);
		same_metadata(lifting_walk.decomposition(), metadata);
		same_metadata(prefix_walk.decomposition(), metadata);
		std::vector<g::functional_graph_successor_query> queries;
		for (uint32 v = 0; v < n; ++v)
			for (uint64 k : steps) queries.push_back({v, k});
		if (!steps.empty())
			for (uint32 v = 0; v < n; ++v) {
				uint64 const depth = expected_metadata.depth_to_cycle[v];
				uint64 const length = expected_metadata.cycles[expected_metadata.cycle_id[v]].size();
				for (uint64 k : {depth, depth + 1, depth + length, depth + 2 * length, depth + 2 * length + 1, length - 1, length, length + 1, 2 * length}) queries.push_back({v, k});
				if (depth) queries.push_back({v, depth - 1});
			}
		if (!queries.empty()) queries.push_back(queries.front());
		std::ranges::reverse(queries);
		auto const offline_endpoints = e::functional_graph_successors_by_reverse_dfs(successors, queries);
		auto const offline_walks = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, transformation_identity, compose{});
		REQUIRE(offline_walks.size() == queries.size());
		REQUIRE(offline_endpoints.size() == queries.size());
		for (std::size_t i = 0; i < queries.size(); ++i) {
			auto const [v, k] = queries[i];
			auto const want = oracle(successors, values, v, k);
			if (k <= 256) {
				auto at = v;
				auto fold = transformation_identity;
				for (uint64 step = 0; step < k; ++step) {
					fold = compose{}(fold, values[at]);
					at = successors[at];
				}
				REQUIRE(want.vertex == at);
				REQUIRE(want.fold == fold);
			}
			REQUIRE(endpoint.successor(v, k) == want.vertex);
			REQUIRE(lifting_endpoint.successor(v, k) == want.vertex);
			REQUIRE(public_walk.successor(v, k) == want.vertex);
			REQUIRE(lifting_walk.successor(v, k) == want.vertex);
			REQUIRE(prefix_walk.successor(v, k) == want.vertex);
			auto const a = public_walk.walk(v, k), b = lifting_walk.walk(v, k), c = prefix_walk.walk(v, k);
			REQUIRE(a.vertex == want.vertex);
			REQUIRE(b.vertex == want.vertex);
			REQUIRE(c.vertex == want.vertex);
			REQUIRE(a.fold == want.fold);
			REQUIRE(b.fold == want.fold);
			REQUIRE(c.fold == want.fold);
			REQUIRE(offline_endpoints[i] == want.vertex);
			REQUIRE(offline_walks[i].vertex == want.vertex);
			REQUIRE(offline_walks[i].fold == want.fold);
			auto const first = public_walk.walk(v, k / 2);
			auto const second = public_walk.walk(first.vertex, k - k / 2);
			REQUIRE(second.vertex == want.vertex);
			REQUIRE(compose{}(first.fold, second.fold) == want.fold);
		}
	}

	/** @brief Genuine single-pass sized range: iterator copies share the consumption cursor. */
	template<std::copyable Element>
	struct one_pass {
		std::vector<Element> &data;
		std::size_t position = 0, begins = 0;

		struct iterator {
			using value_type = Element;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::input_iterator_tag;
			one_pass *range = nullptr;

			Element &operator*() const { return range->data[range->position]; }

			iterator &operator++() {
				++range->position;
				return *this;
			}

			void operator++(int) { ++*this; }

			bool operator==(std::default_sentinel_t) const { return range->position == range->data.size(); }
		};

		iterator begin() {
			++begins;
			return {this};
		}

		std::default_sentinel_t end() const { return {}; }

		std::size_t size() const { return data.size(); }
	};

	static_assert(g::successor_range<one_pass<uint32>>);
	static_assert(!std::ranges::forward_range<one_pass<uint32>>);

	/** @brief Noninvertible idempotent monoid for scalar cross-checks. */
	struct minimum {
		uint64 operator()(uint64 a, uint64 b) const { return (std::ranges::min)(a, b); }
	};

	/** @brief Every walk-fold engine against a direct edge-by-edge scalar oracle. */
	template<atl::binary_operation<uint64> Operation>
		requires std::copyable<Operation>
	void scalar_check(Operation operation, uint64 identity) {
		std::vector<uint32> successors{1, 2, 0, 2, 3, 3, 5, 6, 8};
		std::vector<uint64> values{5, 3, 9, 2, 7, 1, 4, 8, 6};
		g::functional_graph_walk_index<uint64, Operation> const public_index(successors, values, identity, operation);
		e::functional_graph_walk_index_by_lifting<uint64, Operation> const lift(successors, values, identity, operation);
		e::functional_graph_walk_index_by_chain_prefix<uint64, Operation> const prefix(successors, values, identity, operation);
		std::vector<g::functional_graph_successor_query> queries;
		for (uint32 v = 0; v < successors.size(); ++v)
			for (uint64 k = 0; k < 80; ++k) queries.push_back({v, k});
		auto const batch = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, identity, operation);
		for (std::size_t q = 0; q < queries.size(); ++q) {
			auto const [v, k] = queries[q];
			uint32 at = v;
			uint64 fold = identity;
			for (uint64 step = 0; step < k; ++step) {
				fold = std::invoke(std::as_const(operation), fold, values[at]);
				at = successors[at];
			}
			REQUIRE(public_index.walk(v, k).fold == fold);
			REQUIRE(lift.walk(v, k).fold == fold);
			REQUIRE(prefix.walk(v, k).fold == fold);
			REQUIRE(batch[q].fold == fold);
		}
	}

	/** @brief Sized synthetic input asserts the graph limit before dereferencing or allocating. */
	struct oversized_successors {
		uint64 reported_size = uint64(1) << 31;

		uint64 size() const { return reported_size; }

		int const *begin() const { return nullptr; }

		int const *end() const { return nullptr; }
	};

	static_assert(!std::constructible_from<g::functional_graph_successor_index, std::vector<uint32>, uint64>);
	static_assert(g::successor_range<oversized_successors>);
	static_assert(!g::successor_range<std::vector<bool>>);
	static_assert(!g::successor_range<std::vector<char>>);
	static_assert(!g::successor_range<std::vector<double>>);
	static_assert(!g::functional_graph_query_range<std::vector<std::pair<uint32, uint64>>>);

	/** @brief Non-default-constructible move probe, also usable as a monoid value. */
	struct value {
		uint32 bits;
		value() = delete;

		explicit constexpr value(uint32 x) : bits(x) {}

		constexpr value(value const &) = default;
		constexpr value &operator=(value const &) = default;

		constexpr value(value &&other) noexcept : bits(std::exchange(other.bits, 0)) {}

		constexpr value &operator=(value &&other) noexcept {
			bits = std::exchange(other.bits, 0);
			return *this;
		}
	};

	/**
	 * @brief Monoid value whose operation is a member-function pointer.
	 * @details Pointer-sized on purpose: GCC inlines std::invoke on a member-function pointer
	 * with its virtual-dispatch branch, which loads a vtable pointer from the object; for a
	 * 4-byte object -Warray-bounds flags that never-taken load.
	 */
	struct member_xor {
		uint64 bits;

		/** @brief XOR of the two values. */
		constexpr member_xor combine(member_xor const &other) const { return {bits ^ other.bits}; }
	};

	struct counted_xor {
		uint64 *calls;

		uint64 operator()(uint64 a, uint64 b) const {
			++*calls;
			return a ^ b;
		}
	};

	struct xor_value_operation {
		xor_value_operation() = delete;

		explicit xor_value_operation(int) {}

		value operator()(value const &a, value const &b) const { return value(a.bits ^ b.bits); }
	};
}

TEST_CASE("functional graphs: exhaustive mappings through five vertices", "[graph][functional_graph]") {
	check({}, {});
	for (uint32 n = 1; n <= 5; ++n) {
		uint32 mappings = 1;
		for (uint32 i = 0; i < n; ++i) mappings *= n;
		for (uint32 code = 0; code < mappings; ++code) {
			auto remaining = code;
			std::vector<uint32> successors(n);
			for (auto &v : successors) {
				v = remaining % n;
				remaining /= n;
			}
			check(successors, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 31, 32, 33, g::maximum_functional_graph_steps});
		}
	}
}

TEST_CASE("functional graphs: seeded stress, relabelings and step boundaries", "[graph][functional_graph]") {
	std::mt19937_64 rng(0x8a712d);
	std::vector<uint64> steps{0, 1, 2, 127, 128, 129, g::maximum_functional_graph_steps};
	for (uint32 bit = 1; bit <= 59; ++bit) {
		steps.push_back((uint64(1) << bit) - 1);
		steps.push_back(uint64(1) << bit);
		steps.push_back((uint64(1) << bit) + 1);
	}
	for (int trial = 0; trial < 100; ++trial) {
		uint32 const n = 1 + uint32(rng() % 80);
		std::vector<uint32> successors(n), permutation(n);
		for (uint32 v = 0; v < n; ++v) permutation[v] = v;
		std::ranges::shuffle(permutation, rng);
		for (uint32 v = 0; v < n; ++v) {
			uint32 to = 0;
			switch (trial % 7) {
			case 0:
				to = (v + 1) % n;
				break;
			case 1:
				to = (std::ranges::min)(v + 1, n - 1);
				break;
			case 2:
				to = v;
				break;
			case 3:
				to = 0;
				break;
			case 4:
				to = v ? (v - 1) / 2 : 0;
				break;
			case 5:
				to = v > n / 2 ? n / 2 : (v ? v - 1 : 0);
				break;
			default:
				to = uint32(rng() % n);
			}
			successors[permutation[v]] = permutation[to];
		}
		check(successors, steps);
		check(successors, {});
	}
}

TEST_CASE("functional graphs: tail folds and many light edges", "[graph][functional_graph]") {
	uint32 const n = 100'000;
	std::vector<uint32> successors(n), values(n, 1);
	for (uint32 v = 1; v < n; ++v) successors[v] = v - 1;
	g::functional_graph_walk_index<uint32, std::plus<uint32>> const index(successors, values, 0, {});
	e::functional_graph_walk_index_by_chain_prefix<uint32, std::plus<uint32>> const prefix(successors, values, 0, {});
	REQUIRE(index.walk(n - 1, n - 1).fold == n - 1);
	REQUIRE(prefix.walk(n - 1, n - 1).fold == n - 1);
	std::vector<g::functional_graph_successor_query> queries{{n - 1, n - 2}, {n - 1, n}, {n - 1, 0}};
	auto const batch = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, uint32(0), std::plus<uint32>{});
	REQUIRE(batch[0].vertex == 1);
	REQUIRE(batch[1].fold == n);
	REQUIRE(batch[2].fold == 0);
	for (uint32 v = 1; v < n; ++v) successors[v] = (v - 1) / 2;
	e::functional_graph_walk_index_by_chain_prefix<uint32, std::plus<uint32>> const balanced(successors, values, 0, {});
	for (uint32 v = 0; v < n; v += 97) {
		auto const depth = balanced.decomposition().depth_to_cycle[v];
		for (uint32 k = 0; k <= depth + 1; ++k) REQUIRE(balanced.walk(v, k).fold == k);
	}
}

TEST_CASE("functional graphs: ownership, views, non-default types and scalar folds", "[graph][functional_graph]") {
	std::vector<uint32> successors{1, 2, 0, 2};
	std::vector<value> values{value(1), value(2), value(4), value(8)};
	auto verify = [&]<typename Index>(Index const &index)
		requires std::copyable<Index> && requires { index.walk(3, 4).fold.bits; }
	{
		REQUIRE(index.walk(3, 4).fold.bits == (8u ^ 4u ^ 1u ^ 2u));
		auto copy = index;
		auto moved = std::move(copy);
		REQUIRE(moved.decomposition().cycle_id.data() != index.decomposition().cycle_id.data());
		REQUIRE(moved.walk(3, 4).fold.bits == index.walk(3, 4).fold.bits);
	};
	g::functional_graph_walk_index<value, xor_value_operation> const borrowed{std::span(successors), std::span(values), value(0), xor_value_operation(0)};
	REQUIRE(values[0].bits == 1);
	verify(borrowed);
	auto view = values | std::views::take(4);
	e::functional_graph_walk_index_by_lifting<value, xor_value_operation> const viewed(successors, std::move(view), value(0), xor_value_operation(0));
	REQUIRE(values[0].bits == 1);
	verify(viewed);
	e::functional_graph_walk_index_by_chain_prefix<value, xor_value_operation> const owning(std::vector<uint32>(successors), std::move(values), value(0), xor_value_operation(0));
	verify(owning);
	std::vector<g::functional_graph_successor_query> const value_queries{{3, 4}, {0, 0}};
	auto const value_batch = e::functional_graph_walks_by_reverse_dfs(successors, std::vector<value>{value(1), value(2), value(4), value(8)}, value_queries, value(0), xor_value_operation(0));
	REQUIRE(value_batch[0].fold.bits == 15);
	REQUIRE(value_batch[1].fold.bits == 0);
	g::functional_graph_successor_index const stored([] { return std::vector<uint32>{1, 0}; }());
	auto copied = stored;
	auto moved = std::move(copied);
	REQUIRE(moved.successor(0, 3) == 1);
	std::vector<uint32> scalar{5, 2, 7, 3};
	g::functional_graph_walk_index<uint64, std::plus<uint64>> const sum(successors, std::vector<uint64>{5, 2, 7, 3}, 0, {});
	g::functional_graph_walk_index<uint32, std::bit_xor<uint32>> const xored(successors, scalar, 0, {});
	for (uint32 v = 0; v < 4; ++v) {
		uint64 total = 0;
		uint32 bits = 0, at = v;
		for (uint64 k = 0; k < 100; ++k) {
			REQUIRE(sum.walk(v, k).fold == total);
			REQUIRE(xored.walk(v, k).fold == bits);
			total += scalar[at];
			bits ^= scalar[at];
			at = successors[at];
		}
	}
	g::functional_graph_walk_index<value, xor_value_operation> const empty(std::vector<uint32>{}, std::vector<value>{}, value(0), xor_value_operation(0));
	REQUIRE(empty.size() == 0);
}

TEST_CASE("functional graphs: asserted preconditions", "[graph][functional_graph][assert]") {
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_decomposition_of(std::vector<int>{-1}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_decomposition_by_marking(std::vector<uint64>{uint64(1) << 32}); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_successor_index(std::vector<uint32>{g::none}); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_successor_index(std::vector<int>{1}); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_decomposition_of(oversized_successors{}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_successor_index_by_lifting(oversized_successors{}); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_walk_index<uint32, std::plus<uint32>>(std::vector<uint32>{0}, std::vector<uint32>{}, 0, {}); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_successor_index(std::vector<uint32>{}).successor(0, 0); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_successor_index_by_lifting(std::vector<uint32>{0}).successor(1, 0); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_successor_index(std::vector<uint32>{0}).successor(0, g::maximum_functional_graph_steps + 1); }));
	REQUIRE(atl_test::aborts([] { (void)g::functional_graph_walk_index<uint32, std::plus<uint32>>(std::vector<uint32>{0}, std::vector<uint32>{1}, 0, {}).walk(0, g::maximum_functional_graph_steps + 1); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_successors_by_reverse_dfs(std::vector<uint32>{}, std::vector<g::functional_graph_successor_query>{{0, 0}}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_walks_by_reverse_dfs(std::vector<uint32>{0}, std::vector<uint32>{1}, std::vector<g::functional_graph_successor_query>{{0, g::maximum_functional_graph_steps + 1}}, uint32(0), std::plus<uint32>{}); }));
	REQUIRE(atl_test::aborts([] { (void)g::detail::functional_graph_id(g::none); }));
}

TEST_CASE("functional graphs: scalar monoids and single-pass forwarding", "[graph][functional_graph]") {
	scalar_check(std::plus<uint64>{}, 0);
	scalar_check(std::bit_xor<uint64>{}, 0);
	scalar_check(minimum{}, (std::numeric_limits<uint64>::max)());
	std::vector<uint32> successors{1, 2, 0, 2};
	std::vector<uint64> values{5, 3, 9, 2};
	std::vector<g::functional_graph_successor_query> queries{{3, 0}, {3, 8}, {3, 8}, {1, 17}};
	auto exercise = [&]<typename Build>(Build build)
		requires std::invocable<Build &, one_pass<uint32> &, one_pass<uint64> &>
	{
		one_pass<uint32> successor_input{successors};
		one_pass<uint64> value_input{values};
		auto const index = build(successor_input, value_input);
		REQUIRE(successor_input.begins == 1);
		REQUIRE(value_input.begins == 1);
		REQUIRE(index.walk(3, 1).fold == 2);
	};
	exercise([]<g::successor_range Successors, g::functional_graph_value_range<uint64> Values>(Successors &s, Values &w) { return g::functional_graph_walk_index<uint64, std::plus<uint64>>(s, w, 0, {}); });
	exercise([]<g::successor_range Successors, g::functional_graph_value_range<uint64> Values>(Successors &s, Values &w) { return e::functional_graph_walk_index_by_lifting<uint64, std::plus<uint64>>(s, w, 0, {}); });
	exercise([]<g::successor_range Successors, g::functional_graph_value_range<uint64> Values>(Successors &s, Values &w) { return e::functional_graph_walk_index_by_chain_prefix<uint64, std::plus<uint64>>(s, w, 0, {}); });
	one_pass<uint32> successor_input{successors};
	one_pass<uint64> value_input{values};
	one_pass<g::functional_graph_successor_query> query_input{queries};
	auto const batch = e::functional_graph_walks_by_reverse_dfs(successor_input, value_input, query_input, uint64(0), std::plus<uint64>{});
	REQUIRE(successor_input.begins == 1);
	REQUIRE(value_input.begins == 1);
	REQUIRE(query_input.begins == 1);
	REQUIRE(batch[0].fold == 0);
	REQUIRE(batch[1].fold == batch[2].fold);
	auto const direct_batch = e::functional_graph_successors_by_reverse_dfs(one_pass<uint32>{successors}, one_pass<g::functional_graph_successor_query>{queries});
	REQUIRE(batch[3].vertex == direct_batch[3]);
	auto const decomposition = g::functional_graph_decomposition_of(one_pass<uint32>{successors});
	same_metadata(decomposition, e::functional_graph_decomposition_by_marking(one_pass<uint32>{successors}));
	g::functional_graph_successor_index const endpoint(one_pass<uint32>{successors});
	e::functional_graph_successor_index_by_lifting const lift(one_pass<uint32>{successors});
	REQUIRE(endpoint.successor(3, 17) == lift.successor(3, 17));
	auto source = successors;
	auto const source_index = g::functional_graph_successor_index(source);
	source.assign(source.size(), 0);
	REQUIRE(source_index.successor(0, 1) == 1);
}

TEST_CASE("functional graphs: successor ignores monoid machinery", "[graph][functional_graph]") {
	std::vector<uint32> successors{1, 2, 0, 2};
	std::vector<uint64> values{5, 3, 9, 2};
	uint64 calls = 0;
	g::functional_graph_walk_index<uint64, counted_xor> const public_index(successors, values, 0, {&calls});
	e::functional_graph_walk_index_by_lifting<uint64, counted_xor> const lift(successors, values, 0, {&calls});
	e::functional_graph_walk_index_by_chain_prefix<uint64, counted_xor> const prefix(successors, values, 0, {&calls});
	calls = 0;
	for (uint64 steps : {uint64(0), uint64(1), g::maximum_functional_graph_steps}) {
		REQUIRE(public_index.successor(3, steps) == lift.successor(3, steps));
		REQUIRE(prefix.successor(3, steps) == lift.successor(3, steps));
	}
	REQUIRE(calls == 0);
}

TEST_CASE("functional graphs: invocable member-pointer monoid", "[graph][functional_graph]") {
	std::vector<uint32> successors{1, 0, 1};
	std::vector<member_xor> values{{3}, {5}, {9}};
	auto operation = &member_xor::combine;
	g::functional_graph_walk_index<member_xor, decltype(operation)> const public_index(successors, values, member_xor{0}, operation);
	e::functional_graph_walk_index_by_lifting<member_xor, decltype(operation)> const lift(successors, values, member_xor{0}, operation);
	e::functional_graph_walk_index_by_chain_prefix<member_xor, decltype(operation)> const prefix(successors, values, member_xor{0}, operation);
	std::vector<g::functional_graph_successor_query> queries{{2, 7}, {0, 0}, {1, g::maximum_functional_graph_steps}};
	auto const batch = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, member_xor{0}, operation);
	for (std::size_t i = 0; i < queries.size(); ++i) {
		auto const [v, steps] = queries[i];
		REQUIRE(public_index.walk(v, steps).fold.bits == lift.walk(v, steps).fold.bits);
		REQUIRE(prefix.walk(v, steps).fold.bits == lift.walk(v, steps).fold.bits);
		REQUIRE(batch[i].fold.bits == lift.walk(v, steps).fold.bits);
	}
}

TEST_CASE("functional graphs: all walk-fold query boundaries", "[graph][functional_graph][assert]") {
	std::vector<uint32> successors{0};
	std::vector<uint64> values{1};
	auto exercise = []<typename Index>(Index const &index)
		requires requires { index.walk(0, 0); index.successor(0, 0); }
	{
		REQUIRE(atl_test::aborts([&] { (void)index.walk(1, 0); }));
		REQUIRE(atl_test::aborts([&] { (void)index.walk(0, g::maximum_functional_graph_steps + 1); }));
		REQUIRE(atl_test::aborts([&] { (void)index.successor(1, 0); }));
		REQUIRE(atl_test::aborts([&] { (void)index.successor(0, g::maximum_functional_graph_steps + 1); }));
	};
	exercise(g::functional_graph_walk_index<uint64, std::bit_xor<uint64>>(successors, values, 0, {}));
	exercise(e::functional_graph_walk_index_by_lifting<uint64, std::bit_xor<uint64>>(successors, values, 0, {}));
	exercise(e::functional_graph_walk_index_by_chain_prefix<uint64, std::bit_xor<uint64>>(successors, values, 0, {}));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_walk_index_by_chain_prefix<uint64, std::bit_xor<uint64>>(std::vector<int>{-1}, std::vector<uint64>{1}, 0, {}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_walk_index_by_lifting<uint64, std::bit_xor<uint64>>(std::vector<uint32>{0}, std::vector<uint64>{}, 0, {}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_successors_by_reverse_dfs(oversized_successors{}, std::vector<g::functional_graph_successor_query>{}); }));
	REQUIRE(atl_test::aborts([] { (void)e::functional_graph_walks_by_reverse_dfs(std::vector<uint32>{0}, std::vector<uint64>{}, std::vector<g::functional_graph_successor_query>{}, uint64(0), std::bit_xor<uint64>{}); }));
}

TEST_CASE("functional graphs: million-vertex path and cycle use heap stacks", "[graph][functional_graph]") {
	uint32 const n = 1'000'000;
	std::vector<uint32> successors(n), values(n, 1);
	for (bool const is_cycle : {false, true}) {
		for (uint32 v = 0; v < n; ++v) successors[v] = v + 1;
		successors.back() = is_cycle ? 0u : n - 1;
		{
			auto const marking = e::functional_graph_decomposition_by_marking(successors);
			REQUIRE(marking.depth_to_cycle[0] == (is_cycle ? 0u : n - 1));
			REQUIRE(marking.cycles.members.size() == (is_cycle ? n : 1u));
		}
		{
			g::functional_graph_walk_index<uint32, std::bit_xor<uint32>> const index(successors, values, 0, {});
			REQUIRE(index.successor(0, n - 1) == n - 1);
			REQUIRE(index.walk(0, g::maximum_functional_graph_steps).fold == 0);
			REQUIRE(index.walk(0, g::maximum_functional_graph_steps).vertex == (is_cycle ? 0u : n - 1));
		}
		{
			e::functional_graph_walk_index_by_chain_prefix<uint32, std::bit_xor<uint32>> const prefix(successors, values, 0, {});
			REQUIRE(prefix.walk(0, n - 1).fold == 1);
			REQUIRE(prefix.walk(0, n - 1).vertex == n - 1);
		}
		std::array const queries{g::functional_graph_successor_query{0, n - 1}, g::functional_graph_successor_query{0, n}};
		auto const endpoints = e::functional_graph_successors_by_reverse_dfs(successors, queries);
		auto const folds = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, 0u, std::bit_xor<uint32>{});
		REQUIRE(endpoints[0] == n - 1);
		REQUIRE(endpoints[1] == (is_cycle ? 0u : n - 1));
		REQUIRE(folds[0].fold == 1);
		REQUIRE(folds[1].fold == 0);
		REQUIRE(folds[0].vertex == endpoints[0]);
		REQUIRE(folds[1].vertex == endpoints[1]);
	}
}

TEST_CASE("functional graphs: segment storage count bound", "[graph][functional_graph][assert]") {
	auto const limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
	g::detail::functional_graph_check_fold_size(limit);
	REQUIRE(atl_test::aborts([&] { g::detail::functional_graph_check_fold_size(limit + 1); }));
	if constexpr (std::numeric_limits<std::size_t>::digits == 32) {
		REQUIRE(atl_test::aborts([&] { (void)g::functional_graph_walk_index<unsigned, std::plus<unsigned>>(oversized_successors{uint64(limit) + 1}, std::array<unsigned, 0>{}, 0u, {}); }));
		REQUIRE(atl_test::aborts([&] { (void)e::functional_graph_walks_by_reverse_dfs(oversized_successors{uint64(limit) + 1}, std::array<unsigned, 0>{}, std::array<g::functional_graph_successor_query, 0>{}, 0u, std::plus<unsigned>{}); }));
	}
}
