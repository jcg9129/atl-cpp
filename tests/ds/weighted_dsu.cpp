#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/experimental/ds/weighted_dsu_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using atl::uint8;
using atl::unite_outcome;

namespace {
	using permutation = std::array<uint8, 5>;

	/** @brief (p · q)(i) = p(q(i)): non-commutative, so an operand-order bug changes answers. */
	struct compose {
		constexpr permutation operator()(permutation const &p, permutation const &q) const {
			permutation r{};
			for (std::size_t i = 0; i < 5; ++i) r[i] = p[q[i]];
			return r;
		}
	};

	/** @brief p⁻¹(p(i)) = i. */
	struct permutation_inverse {
		constexpr permutation operator()(permutation const &p) const {
			permutation r{};
			for (std::size_t i = 0; i < 5; ++i) r[p[i]] = uint8(i);
			return r;
		}
	};

	/** @brief Z/2 under xor is its own inverse. */
	struct self_inverse {
		constexpr uint32 operator()(uint32 x) const { return x; }
	};

	/** @brief Z/2 addition stored in one uint32 bit. */
	struct bit_xor_u32 {
		constexpr uint32 operator()(uint32 a, uint32 b) const { return a ^ b; }
	};

	/** @brief A group with a sampler of random elements. */
	template<std::copyable Potential, atl::binary_operation<Potential> Op, atl::inverse_operation<Potential> Inverse,
			 typename Sampler>
		requires std::semiregular<Potential> && std::equality_comparable<Potential>
			  && std::invocable<Sampler const &, std::mt19937_64 &>
			  && std::convertible_to<std::invoke_result_t<Sampler const &, std::mt19937_64 &>, Potential>
	struct group {
		Potential identity;
		Op op;
		Inverse inverse;
		Sampler random;
	};

	auto int64_group() {
		return group{atl::int64(0), std::plus<>{}, std::negate<>{},
					 [](std::mt19937_64 &rng) { return atl::int64(rng() % 2001) - 1000; }};
	}

	auto parity_group() {
		return group{uint32(0), bit_xor_u32{}, self_inverse{}, [](std::mt19937_64 &rng) { return uint32(rng() & 1); }};
	}

	auto permutation_group() {
		return group{permutation{0, 1, 2, 3, 4}, compose{}, permutation_inverse{}, [](std::mt19937_64 &rng) {
						 permutation p{0, 1, 2, 3, 4};
						 for (std::size_t i = 4; i > 0; --i) std::swap(p[i], p[rng() % (i + 1)]);
						 return p;
					 }};
	}

	/**
	 * @brief The certifying oracle: the accepted merging constraints as a forest; BFS from
	 *        `a` assigns pot(a) = identity and walks `w` forward, `inverse(w)` backward.
	 */
	template<std::semiregular Potential, atl::binary_operation<Potential> Op, atl::inverse_operation<Potential> Inverse>
		requires std::equality_comparable<Potential>
	struct constraint_forest {
		std::size_t n;
		Potential identity;
		Op op;
		Inverse inverse;
		std::vector<std::tuple<uint32, uint32, Potential>> edges; // merging constraints, oldest first

		/** @brief Potentials relative to `a` (none where unreachable). */
		std::vector<std::optional<Potential>> potentials_from(std::size_t a) const {
			std::vector<std::vector<std::pair<uint32, Potential>>> adjacent(n);
			for (auto const &[u, v, w] : edges) {
				adjacent[u].push_back({v, w});
				adjacent[v].push_back({u, Potential(std::invoke(inverse, w))});
			}
			std::vector<std::optional<Potential>> pot(n);
			pot[a] = identity;
			std::vector<uint32> queue{uint32(a)};
			for (std::size_t head = 0; head < queue.size(); ++head) {
				uint32 const u = queue[head];
				for (auto const &[v, w] : adjacent[u])
					if (!pot[v]) pot[v] = Potential(std::invoke(op, *pot[u], w)), queue.push_back(v);
			}
			return pot;
		}

		std::optional<Potential> difference(std::size_t a, std::size_t b) const { return potentials_from(a)[b]; }

		std::size_t size(std::size_t x) const {
			std::size_t s = 0;
			for (auto const &p : potentials_from(x)) s += p.has_value();
			return s;
		}

		unite_outcome unite(std::size_t a, std::size_t b, Potential w) {
			if (auto const d = difference(a, b)) return *d == w ? unite_outcome::consistent : unite_outcome::contradicts;
			edges.emplace_back(uint32(a), uint32(b), std::move(w));
			return unite_outcome::merged;
		}
	};

	template<typename Dsu>
	constexpr bool has_rollback = requires(Dsu &d) { d.rollback(d.time()); };

	/** @brief A @ref group: identity, operation, inverse and a random-element sampler. */
	template<typename Group>
	concept sampled_group = requires(Group const &g, std::mt19937_64 &rng) {
		requires std::semiregular<decltype(g.identity)> && std::equality_comparable<decltype(g.identity)>;
		requires atl::binary_operation<decltype(g.op), decltype(g.identity)>;
		requires atl::inverse_operation<decltype(g.inverse), decltype(g.identity)>;
		requires std::copy_constructible<decltype(g.op)> && std::copy_constructible<decltype(g.inverse)>;
		std::invoke(g.op, g.identity, g.identity);
		std::invoke(g.inverse, g.identity);
		{ g.random(rng) } -> std::convertible_to<decltype(g.identity)>;
	};

	template<typename Dsu, typename Potential>
	concept potential_dsu = std::copyable<Dsu> && requires(Dsu &d, std::size_t index, Potential weight) {
		{ d.unite(index, index, weight) } -> std::same_as<unite_outcome>;
		{ d.difference(index, index) } -> std::same_as<std::optional<Potential>>;
		{ d.find(index) } -> std::convertible_to<std::size_t>;
		{ d.same(index, index) } -> std::same_as<bool>;
		{ d.size(index) } -> std::convertible_to<std::size_t>;
		{ d.count() } -> std::convertible_to<std::size_t>;
	};

	/**
	 * @brief Random constraints (3/4 drawn from a hidden assignment, so many are consistent,
	 *        the rest random) against the oracle; checks every outcome, then `difference`,
	 *        `same`, `size` and `count` on random elements. Rollback types also undo.
	 */
	template<template<typename, typename, typename> typename Dsu, sampled_group Group>
		requires potential_dsu<Dsu<decltype(Group::identity), decltype(Group::op), decltype(Group::inverse)>, decltype(Group::identity)>
			  && std::constructible_from<Dsu<decltype(Group::identity), decltype(Group::op), decltype(Group::inverse)>, std::size_t, decltype(Group::identity), decltype(Group::op), decltype(Group::inverse)>
	void check_against_oracle(Group const &g, uint64 seed, std::size_t n, std::size_t steps) {
		using potential = decltype(g.identity);
		using op = decltype(g.op);
		using inverse = decltype(g.inverse);
		std::mt19937_64 rng(seed);
		Dsu<potential, op, inverse> d(n, g.identity, g.op, g.inverse);
		constraint_forest<potential, op, inverse> oracle{n, g.identity, g.op, g.inverse, {}};
		std::vector<potential> hidden(n);
		for (auto &h : hidden) h = g.random(rng);
		std::vector<std::size_t> times;
		for (std::size_t step = 0; step < steps; ++step) {
			if constexpr (has_rollback<Dsu<potential, op, inverse>>) {
				if (rng() % 8 == 0) {
					times.push_back(d.time());
					REQUIRE(d.time() == oracle.edges.size());
				}
				if (!times.empty() && rng() % 10 == 0) {
					std::size_t const t = times.back();
					times.pop_back();
					if (t <= d.time()) {
						d.rollback(t);
						oracle.edges.resize(t);
					}
				}
			}
			std::size_t const a = rng() % n, b = rng() % n;
			potential w = rng() % 4 ? potential(std::invoke(g.op, std::invoke(g.inverse, hidden[a]), hidden[b])) : g.random(rng);
			REQUIRE(d.unite(a, b, w) == oracle.unite(a, b, w));
			REQUIRE(d.count() == n - oracle.edges.size());
			std::size_t const x = rng() % n, y = rng() % n;
			auto const expected = oracle.difference(x, y);
			REQUIRE(d.difference(x, y) == expected);
			REQUIRE(d.same(x, y) == expected.has_value());
			REQUIRE(d.size(x) == oracle.size(x));
			REQUIRE(d.find(x) == d.find(d.find(x)));
		}
	}

	template<template<typename, typename, typename> typename Dsu>
		requires requires {
			check_against_oracle<Dsu>(int64_group(), uint64{}, std::size_t{}, std::size_t{});
			check_against_oracle<Dsu>(parity_group(), uint64{}, std::size_t{}, std::size_t{});
			check_against_oracle<Dsu>(permutation_group(), uint64{}, std::size_t{}, std::size_t{});
		}
	void check_every_group() {
		auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
		std::size_t const n = 2 + seed % 40;
		check_against_oracle<Dsu>(int64_group(), seed, n, 400);
		check_against_oracle<Dsu>(parity_group(), seed, n, 400);
		check_against_oracle<Dsu>(permutation_group(), seed, n, 400);
	}

	using atl::experimental::weighted_dsu_by_compression_and_layout;
	using atl::experimental::weighted_dsu_compression;
	using atl::experimental::weighted_dsu_layout;

	template<typename Potential, typename Op, typename Inverse>
	using buffer_interleaved = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_buffer, weighted_dsu_layout::interleaved, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using buffer_split = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_buffer, weighted_dsu_layout::split, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using inverse_interleaved = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_inverse, weighted_dsu_layout::interleaved, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using inverse_split = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_inverse, weighted_dsu_layout::split, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using halving_interleaved = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::halving, weighted_dsu_layout::interleaved, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using halving_split = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::halving, weighted_dsu_layout::split, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using splitting_interleaved = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::splitting, weighted_dsu_layout::interleaved, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using splitting_split = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::splitting, weighted_dsu_layout::split, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using buffer_no_exit = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_buffer, weighted_dsu_layout::interleaved, Potential, Op, Inverse, false>;
	template<typename Potential, typename Op, typename Inverse>
	using inverse_no_exit = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::two_pass_inverse, weighted_dsu_layout::split, Potential, Op, Inverse, false>;
	template<typename Potential, typename Op, typename Inverse>
	using halving_no_exit = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::halving, weighted_dsu_layout::interleaved, Potential, Op, Inverse, false>;
	template<typename Potential, typename Op, typename Inverse>
	using splitting_no_exit = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::splitting, weighted_dsu_layout::interleaved, Potential, Op, Inverse, false>;
	template<typename Potential, typename Op, typename Inverse>
	using recursive_no_exit = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::recursive, weighted_dsu_layout::split, Potential, Op, Inverse, false>;
	template<typename Potential, typename Op, typename Inverse>
	using recursive_interleaved = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::recursive, weighted_dsu_layout::interleaved, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using recursive_split = weighted_dsu_by_compression_and_layout<weighted_dsu_compression::recursive, weighted_dsu_layout::split, Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using rollback_split = atl::rollback_weighted_dsu<Potential, Op, Inverse>;
	template<typename Potential, typename Op, typename Inverse>
	using rollback_interleaved = atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<Potential, Op, Inverse>;
}

TEST_CASE("weighted_dsu: additive potential constraints", "[weighted_dsu]") {
	// The relation "a_u = a_v + x" is unite(v, u, x); "a_u - a_v" is difference(v, u)
	atl::weighted_dsu<atl::int64> d(4);
	REQUIRE(d.unite(0, 1, 9) == unite_outcome::merged);
	REQUIRE(!d.difference(2, 0));
	REQUIRE(d.unite(1, 2, 90) == unite_outcome::merged);
	REQUIRE(d.unite(0, 2, 99) == unite_outcome::consistent);
	REQUIRE(d.unite(2, 0, 123) == unite_outcome::contradicts);
	REQUIRE(d.unite(1, 3, 990) == unite_outcome::merged);
	REQUIRE(d.difference(2, 0) == atl::int64(-99));
	REQUIRE(d.difference(0, 2) == atl::int64(99));
	REQUIRE(d.difference(0, 3) == atl::int64(999));
	REQUIRE(d.difference(1, 1) == atl::int64(0));
	REQUIRE(d.count() == 1);
	REQUIRE(d.size(3) == 4);
}

TEST_CASE("weighted_dsu: edge cases and CTAD", "[weighted_dsu]") {
	atl::weighted_dsu<> empty;
	REQUIRE(empty.elements() == 0);
	REQUIRE(empty.count() == 0);
	atl::rollback_weighted_dsu<> empty_rollback;
	REQUIRE(empty_rollback.elements() == 0);
	REQUIRE(empty_rollback.time() == 0);

	atl::weighted_dsu one(1);
	static_assert(std::is_same_v<decltype(one), atl::weighted_dsu<atl::int64, std::plus<>, std::negate<>>>);
	REQUIRE(one.unite(0, 0, 0) == unite_outcome::consistent);
	REQUIRE(one.unite(0, 0, 5) == unite_outcome::contradicts);
	REQUIRE(one.difference(0, 0) == atl::int64(0));
	REQUIRE(one.count() == 1);

	atl::weighted_dsu parity(3, uint32(0), bit_xor_u32{}, self_inverse{});
	static_assert(std::is_same_v<decltype(parity), atl::weighted_dsu<uint32, bit_xor_u32, self_inverse>>);
	REQUIRE(parity.unite(0, 1, 1) == unite_outcome::merged);
	REQUIRE(parity.unite(1, 2, 1) == unite_outcome::merged);
	REQUIRE(parity.unite(0, 2, 1) == unite_outcome::contradicts); // an odd cycle
	REQUIRE(parity.difference(0, 2) == uint32(0));

	atl::rollback_weighted_dsu rollback(2);
	static_assert(std::is_same_v<decltype(rollback), atl::rollback_weighted_dsu<atl::int64, std::plus<>, std::negate<>>>);
	REQUIRE(rollback.unite(0, 1, 7) == unite_outcome::merged);
	REQUIRE(rollback.unite(1, 0, -7) == unite_outcome::consistent);
	REQUIRE(rollback.time() == 1);
	rollback.undo();
	REQUIRE(rollback.time() == 0);
	REQUIRE(!rollback.difference(0, 1));
	REQUIRE(rollback.unite(1, 0, 3) == unite_outcome::merged); // the undone weight is gone
	REQUIRE(rollback.difference(0, 1) == atl::int64(-3));

	[[maybe_unused]] std::size_t (atl::rollback_weighted_dsu<>::*time)() const noexcept = &atl::rollback_weighted_dsu<>::time;
	[[maybe_unused]] void (atl::rollback_weighted_dsu<>::*rollback_to)(std::size_t) = &atl::rollback_weighted_dsu<>::rollback;
	[[maybe_unused]] auto deduced_time = &atl::rollback_weighted_dsu<>::time;
	[[maybe_unused]] auto deduced_rollback = &atl::rollback_weighted_dsu<>::rollback;
}

TEST_CASE("weighted_dsu: deep binomial trees, every variant agrees", "[weighted_dsu]") {
	// pairs, then pairs of pairs: depth log2(n), the deepest union by size allows
	std::size_t const n = std::size_t(1) << 12;
	std::mt19937_64 rng(7);
	std::vector<std::pair<std::size_t, permutation>> plan;
	for (std::size_t s = 1; s < n; s *= 2)
		for (std::size_t i = 0; i + s < n; i += 2 * s) plan.push_back({s, permutation_group().random(rng)});
	auto run = [&]<potential_dsu<permutation> Dsu>(Dsu d) {
		std::size_t k = 0;
		for (std::size_t s = 1; s < n; s *= 2)
			for (std::size_t i = 0; i + s < n; i += 2 * s) REQUIRE(d.unite(i, i + s, plan[k++].second) == unite_outcome::merged);
		std::vector<permutation> answers;
		std::mt19937_64 probe(11);
		for (int q = 0; q < 2000; ++q) answers.push_back(*d.difference(probe() % n, probe() % n));
		return answers;
	};
	auto const g = permutation_group();
	auto const expected = run(atl::rollback_weighted_dsu(n, g.identity, g.op, g.inverse));
	REQUIRE(run(atl::weighted_dsu(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(buffer_interleaved<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(inverse_split<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(halving_split<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(recursive_interleaved<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(splitting_split<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
	REQUIRE(run(halving_no_exit<permutation, compose, permutation_inverse>(n, g.identity, g.op, g.inverse)) == expected);
}

TEST_CASE("weighted_dsu: matches the constraint-forest oracle", "[weighted_dsu]") { check_every_group<atl::weighted_dsu>(); }

TEST_CASE("rollback_weighted_dsu: matches the constraint-forest oracle with rollback", "[weighted_dsu]") {
	check_every_group<atl::rollback_weighted_dsu>();
}

TEST_CASE("weighted_dsu variants: match the constraint-forest oracle", "[weighted_dsu]") {
	check_every_group<buffer_interleaved>();
	check_every_group<buffer_split>();
	check_every_group<inverse_interleaved>();
	check_every_group<inverse_split>();
	check_every_group<halving_interleaved>();
	check_every_group<halving_split>();
	check_every_group<recursive_interleaved>();
	check_every_group<recursive_split>();
	check_every_group<splitting_interleaved>();
	check_every_group<splitting_split>();
	check_every_group<buffer_no_exit>();
	check_every_group<inverse_no_exit>();
	check_every_group<halving_no_exit>();
	check_every_group<splitting_no_exit>();
	check_every_group<recursive_no_exit>();
}

TEST_CASE("rollback_weighted_dsu variants: match the constraint-forest oracle with rollback", "[weighted_dsu]") {
	check_every_group<rollback_split>();
	check_every_group<rollback_interleaved>();
}

TEST_CASE("rollback_weighted_dsu variants: by interleaved nodes agrees with the library class on random unites and undos", "[weighted_dsu]") {
	std::mt19937_64 rng(7);
	for (std::size_t const n : {0, 1, 2, 3, 64, 1000}) {
		atl::rollback_weighted_dsu<> library(n);
		rollback_interleaved<atl::int64, std::plus<>, std::negate<>> split(n, 0);
		for (std::size_t i = 0; n && i < 3 * n; ++i) {
			std::size_t const a = rng() % n, b = rng() % n;
			atl::int64 const w = atl::int64(rng() % 5);
			REQUIRE(library.unite(a, b, w) == split.unite(a, b, w));
			REQUIRE(library.difference(b, a) == split.difference(b, a));
			REQUIRE(library.size(a) == split.size(a));
			if (rng() % 7 == 0 && library.time()) library.undo(), split.undo();
		}
		library.rollback(0), split.rollback(0);
		REQUIRE(split.count() == n);
	}
}

TEST_CASE("weighted_dsu: asserted preconditions", "[weighted_dsu]") {
	CHECK(atl_test::aborts([] { atl::weighted_dsu<> d(std::size_t((std::numeric_limits<atl::int32>::max)()) + 1); }));
	CHECK(atl_test::aborts([] { atl::rollback_weighted_dsu<> d(std::size_t((std::numeric_limits<atl::int32>::max)()) + 1); }));
	CHECK(atl_test::aborts([] {
		atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<> d(std::size_t((std::numeric_limits<atl::int32>::max)()) + 1);
	}));
	CHECK(atl_test::aborts([] {
		atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<> d(2);
		d.undo();
	}));
	CHECK(atl_test::aborts([] {
		atl::weighted_dsu<> d(3);
		(void)d.find(3);
	}));
	CHECK(atl_test::aborts([] {
		atl::rollback_weighted_dsu<> d(3);
		(void)d.difference(0, 3);
	}));
	CHECK(atl_test::aborts([] {
		atl::rollback_weighted_dsu<> d(3);
		d.undo();
	}));
	CHECK(atl_test::aborts([] {
		atl::rollback_weighted_dsu<> d(3);
		d.rollback(1);
	}));
}

TEST_CASE("weighted_dsu: noncommutative direction, swapped roots and rollback", "[weighted_dsu]") {
	permutation const identity{0, 1, 2, 3, 4};
	permutation const p{1, 0, 2, 3, 4};
	permutation const q{0, 2, 1, 3, 4};
	REQUIRE(compose{}(p, q) != compose{}(q, p));
	atl::rollback_weighted_dsu d(4, identity, compose{}, permutation_inverse{});
	REQUIRE(d.unite(1, 2, q) == unite_outcome::merged);
	auto const saved = d.time();
	REQUIRE(d.unite(0, 1, p) == unite_outcome::merged);
	REQUIRE(d.difference(0, 2) == compose{}(p, q));
	REQUIRE(d.unite(0, 2, compose{}(q, p)) == unite_outcome::contradicts);
	REQUIRE(d.time() == saved + 1);
	d.rollback(saved);
	REQUIRE(!d.difference(0, 2));
	REQUIRE(d.difference(1, 2) == q);
	REQUIRE(d.count() == 3);
}

TEST_CASE("weighted_dsu: published 128-bit potentials", "[weighted_dsu]") {
	atl::int128 const large = atl::int128(1) << 100;
	atl::weighted_dsu<atl::int128> compressed(3);
	atl::rollback_weighted_dsu<atl::int128> undoable(3);
	compressed.unite(0, 1, large);
	compressed.unite(1, 2, large);
	undoable.unite(0, 1, large);
	auto const token = undoable.time();
	undoable.unite(1, 2, large);
	REQUIRE(compressed.difference(0, 2) == large + large);
	REQUIRE(undoable.difference(0, 2) == compressed.difference(0, 2));
	REQUIRE(compressed.difference(2, 0) == -(large + large));
	undoable.rollback(token);
	REQUIRE(!undoable.difference(0, 2));
	REQUIRE(undoable.difference(0, 1) == large);
}

namespace {
	struct sum_potential {
		atl::int64 value = 0;
		constexpr bool operator==(sum_potential const &) const = default;

		constexpr sum_potential add(sum_potential other) const { return {value + other.value}; }

		constexpr sum_potential inverse() const { return {-value}; }
	};
}

TEST_CASE("weighted_dsu: operation concepts support member pointers", "[weighted_dsu]") {
	atl::weighted_dsu compressed(4, sum_potential{}, &sum_potential::add, &sum_potential::inverse);
	atl::rollback_weighted_dsu undoable(4, sum_potential{}, &sum_potential::add, &sum_potential::inverse);
	auto check = []<potential_dsu<sum_potential> Dsu>(Dsu &d) {
		REQUIRE(d.unite(1, 2, sum_potential{5}) == unite_outcome::merged);
		REQUIRE(d.unite(0, 1, sum_potential{7}) == unite_outcome::merged);
		REQUIRE(d.difference(0, 2) == sum_potential{12});
		REQUIRE(d.difference(2, 0) == sum_potential{-12});
		REQUIRE(d.unite(0, 2, sum_potential{12}) == unite_outcome::consistent);
	};
	check(compressed);
	check(undoable);
	undoable.undo();
	REQUIRE(!undoable.difference(0, 2));
	REQUIRE(undoable.difference(1, 2) == sum_potential{5});
	check_against_oracle<atl::weighted_dsu>(
		group{sum_potential{}, &sum_potential::add, &sum_potential::inverse,
			  [](std::mt19937_64 &rng) { return sum_potential{atl::int64(rng() % 2001) - 1000}; }},
		71, 20, 200);
}

namespace {
	// Keep experimental proofs beside their runtime oracles so constexpr_checks includes only published headers.
	template<atl::experimental::weighted_dsu_compression Compression, atl::experimental::weighted_dsu_layout Layout, atl::signed_integral Potential>
	constexpr bool weighted_method_constant_check() {
		atl::experimental::weighted_dsu_by_compression_and_layout<Compression, Layout, Potential> d(5);
		bool const has_merged = d.unite(0, 1, 5) == atl::unite_outcome::merged && d.unite(1, 2, 3) == atl::unite_outcome::merged
							 && d.unite(3, 4, 1) == atl::unite_outcome::merged && d.unite(2, 3, 2) == atl::unite_outcome::merged;
		bool const is_checked = d.unite(0, 4, 11) == atl::unite_outcome::consistent && d.unite(4, 0, 11) == atl::unite_outcome::contradicts;
		return has_merged && is_checked && d.difference(4, 0) == -11 && d.size(3) == 5 && d.count() == 1;
	}

	template<atl::experimental::weighted_dsu_compression Compression>
	constexpr bool weighted_layouts_constant_check() {
		return weighted_method_constant_check<Compression, atl::experimental::weighted_dsu_layout::interleaved, atl::int32>()
			&& weighted_method_constant_check<Compression, atl::experimental::weighted_dsu_layout::split, atl::int64>();
	}

	static_assert(weighted_layouts_constant_check<atl::experimental::weighted_dsu_compression::two_pass_buffer>());
	static_assert(weighted_layouts_constant_check<atl::experimental::weighted_dsu_compression::two_pass_inverse>());
	static_assert(weighted_layouts_constant_check<atl::experimental::weighted_dsu_compression::halving>());
	static_assert(weighted_layouts_constant_check<atl::experimental::weighted_dsu_compression::splitting>());
	static_assert(weighted_layouts_constant_check<atl::experimental::weighted_dsu_compression::recursive>());
	static_assert([] {
		atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<> d(3);
		d.unite(0, 1, 2);
		auto const t = d.time();
		d.unite(1, 2, 4);
		bool const is_joined = d.difference(0, 2) == 6 && d.find(2) == d.find(0);
		d.rollback(t);
		return is_joined && !d.same(0, 2) && d.difference(1, 0) == -2 && d.unite(2, 1, 1) == atl::unite_outcome::merged;
	}());
}
