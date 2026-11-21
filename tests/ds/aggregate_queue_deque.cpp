#include <atl/ds/aggregate_deque.hpp>
#include <atl/ds/aggregate_queue.hpp>
#include <atl/experimental/ds/aggregate_queue_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

namespace {
	/** @brief Affine maps x -> a x + b over Z/101Z, singular ones included: noncommutative and noninvertible. */
	struct affine {
		unsigned a, b;
		friend bool operator==(affine, affine) = default;
	};

	/** @brief The left map, then the right one. */
	struct compose {
		constexpr affine operator()(affine left, affine right) const { return {right.a * left.a % 101, (right.a * left.b + right.b) % 101}; }
	};

	/** @brief The oracle: a fresh front-to-back fold of a std::deque. */
	template<typename Container, typename Value, typename Operation>
		requires std::same_as<typename Container::value_type, Value> && atl::binary_operation<Operation, Value>
	void check(Container const &actual, std::deque<Value> const &oracle, Value identity, Operation operation) {
		Value expected = identity;
		for (auto const &value : oracle) expected = std::invoke(operation, expected, value);
		REQUIRE(actual.size() == oracle.size());
		REQUIRE(actual.empty() == oracle.empty());
		REQUIRE(actual.fold() == expected);
		if (!oracle.empty()) {
			REQUIRE(actual.front() == oracle.front());
			REQUIRE(actual.back() == oracle.back());
		}
	}

	/** @brief Every legal trace up to @p depth operations, with two distinct values. */
	template<typename Container, bool IsDeque>
		requires std::same_as<typename Container::value_type, affine>
	void exhaustive(Container const &state, std::deque<affine> const &oracle, unsigned depth) {
		check(state, oracle, affine{1, 0}, compose{});
		if (depth == 0) return;
		for (affine const value : {affine{0, 3}, affine{2, 1}}) {
			auto next = state;
			auto want = oracle;
			next.push_back(value);
			want.push_back(value);
			exhaustive<Container, IsDeque>(next, want, depth - 1);
			if constexpr (IsDeque) {
				next = state;
				want = oracle;
				next.push_front(value);
				want.push_front(value);
				exhaustive<Container, IsDeque>(next, want, depth - 1);
			}
		}
		if (!oracle.empty()) {
			auto next = state;
			auto want = oracle;
			next.pop_front();
			want.pop_front();
			exhaustive<Container, IsDeque>(next, want, depth - 1);
			if constexpr (IsDeque) {
				next = state;
				want = oracle;
				next.pop_back();
				want.pop_back();
				exhaustive<Container, IsDeque>(next, want, depth - 1);
			}
		}
	}

	/** @brief One random FIFO trace on every general engine, checked against the oracle after every step. */
	template<std::copyable Value, atl::binary_operation<Value> Operation, typename Generator>
		requires std::invocable<Generator &, std::mt19937_64 &> && std::convertible_to<std::invoke_result_t<Generator &, std::mt19937_64 &>, Value>
	void fifo_walk(Value identity, Operation operation, Generator generate) {
		atl::aggregate_queue queue(identity, operation);
		atl::aggregate_deque deque(identity, operation);
		atl::experimental::aggregate_queue_by_daba_lite_ring daba(identity, operation);
		atl::experimental::aggregate_queue_by_daba_lite_chunks chunked(identity, operation);
		std::deque<Value> oracle;
		std::mt19937_64 random(0x48414747);
		for (unsigned step = 0; step < 24000; ++step) {
			if (oracle.empty() || (oracle.size() < 128 && random() % 2)) {
				Value value = generate(random);
				queue.push_back(value);
				deque.push_back(value);
				daba.push_back(value);
				chunked.push_back(value);
				oracle.push_back(std::move(value));
			} else {
				queue.pop_front();
				deque.pop_front();
				daba.pop_front();
				chunked.pop_front();
				oracle.pop_front();
			}
			if (step % 257 == 0) {
				queue.reserve(193);
				deque.reserve(193);
				daba.reserve(193);
				chunked.reserve(193);
			}
			if (step % 997 == 0) {
				queue.clear();
				deque.clear();
				daba.clear();
				chunked.clear();
				oracle.clear();
			}
			check(queue, oracle, identity, operation);
			check(deque, oracle, identity, operation);
			check(daba, oracle, identity, operation);
			check(chunked, oracle, identity, operation);
			REQUIRE(chunked.fold() == daba.fold());
			REQUIRE(queue.fold() == daba.fold());
			REQUIRE(queue.fold() == deque.fold());
		}
	}

	/** @brief Fill to @p n from one end, then drain alternately from both ends (to empty or to n/2), eight rounds. */
	void deque_adversaries(std::size_t n, bool is_mirrored) {
		atl::aggregate_deque<affine, compose> actual({1, 0});
		std::deque<affine> oracle;
		for (unsigned round = 0; round < 8; ++round) {
			while (oracle.size() < n) {
				atl_assert(oracle.size() <= (std::numeric_limits<unsigned>::max)());
				affine value{unsigned(oracle.size() % 4), unsigned(oracle.size() % 101)};
				if (is_mirrored) {
					actual.push_front(value);
					oracle.push_front(value);
				} else {
					actual.push_back(value);
					oracle.push_back(value);
				}
				check(actual, oracle, affine{1, 0}, compose{});
			}
			std::size_t const remaining = round % 2 ? 0 : n / 2;
			while (oracle.size() > remaining) {
				if ((oracle.size() % 2 != 0) == is_mirrored) {
					actual.pop_front();
					oracle.pop_front();
				} else {
					actual.pop_back();
					oracle.pop_back();
				}
				check(actual, oracle, affine{1, 0}, compose{});
			}
		}
	}

	/** @brief Endpoint reads and pops of an empty container, and a reserve beyond max_size(), abort. */
	template<typename Container>
		requires std::constructible_from<Container, int>
	void deaths() {
		CHECK(atl_test::aborts([] { Container values(0); (void)values.front(); }));
		CHECK(atl_test::aborts([] { Container values(0); (void)values.back(); }));
		CHECK(atl_test::aborts([] { Container values(0); values.pop_front(); }));
		CHECK(atl_test::aborts([] { Container values(0); values.reserve((std::numeric_limits<std::size_t>::max)()); }));
		if constexpr (requires(Container &values) { values.pop_back(); })
			CHECK(atl_test::aborts([] { Container values(0); values.pop_back(); }));
		if constexpr (requires(Container const &values) { values.max_size(); })
			CHECK(atl_test::aborts([] { Container values(0); values.reserve(values.max_size() + 1); }));
	}

	/** @brief A value without default constructor or equality. */
	struct token {
		int value;
		token() = delete;

		constexpr explicit token(int v) : value(v) {}
	};

	/** @brief Addition that counts its calls. */
	struct counted_sum {
		std::size_t *calls;

		std::uint64_t operator()(std::uint64_t left, std::uint64_t right) const {
			++*calls;
			return left + right;
		}
	};

	/** @brief Emplace from an endpoint, copies, moves and clear, with a capturing lambda as the operation. */
	template<template<typename, typename> typename Container>
		requires std::constructible_from<Container<int, std::plus<>>, int>
	void object_contracts() {
		auto operation = [offset = 0](token const &a, token const &b) { return token(a.value + b.value + offset); };
		Container<token, decltype(operation)> values(token(0), operation);
		values.reserve(0);
		values.emplace_back(7);
		values.emplace_back(values.front());
		REQUIRE(values.fold().value == 14);
		values.push_back(values.back());
		REQUIRE(values.back().value == 7);
		values.pop_front();
		REQUIRE(values.front().value == 7);
		if constexpr (requires { values.emplace_front(values.back()); }) {
			values.emplace_front(values.back());
			values.push_front(values.front());
			values.pop_back();
			REQUIRE(values.fold().value == 21);
		}
		auto copy = values;
		copy.clear();
		REQUIRE(copy.fold().value == 0);
		REQUIRE(!values.empty());
		auto moved = std::move(values);
		REQUIRE(moved.front().value == 7);
		auto empty_destination = std::move(copy);
		REQUIRE(empty_destination.empty());
	}
}

namespace {
	/** @brief 2x2 matrices over Z/7Z, singular ones included, in row-major order. */
	using matrix2 = std::array<unsigned, 4>;

	/** @brief The ordered product left * right. */
	struct multiply2 {
		constexpr matrix2 operator()(matrix2 const &left, matrix2 const &right) const {
			return {(left[0] * right[0] + left[1] * right[2]) % 7, (left[0] * right[1] + left[1] * right[3]) % 7,
					(left[2] * right[0] + left[3] * right[2]) % 7, (left[2] * right[1] + left[3] * right[3]) % 7};
		}
	};

	/** @brief Random FIFO traces on the queue engines and random four-ended traces on the deque, with copies. */
	void matrix_walks() {
		matrix2 const identity{1, 0, 0, 1};
		std::mt19937_64 rng(0x4d415432);
		for (unsigned trial = 0; trial < 40; ++trial) {
			atl::aggregate_queue<matrix2, multiply2> queue(identity);
			atl::experimental::aggregate_queue_by_daba_lite_ring<matrix2, multiply2> ring(identity);
			atl::experimental::aggregate_queue_by_daba_lite_chunks<matrix2, multiply2> chunks(identity);
			atl::aggregate_deque<matrix2, multiply2> deque(identity);
			std::deque<matrix2> fifo, both;
			auto const push_percent = rng() % 100;
			auto const width = 1 + rng() % 200;
			for (unsigned step = 0; step < 2000; ++step) {
				matrix2 const value{unsigned(rng() % 7), unsigned(rng() % 7), unsigned(rng() % 7), unsigned(rng() % 7)};
				if (fifo.empty() || (fifo.size() < width && rng() % 100 < push_percent)) {
					queue.push_back(value);
					ring.push_back(value);
					chunks.push_back(value);
					fifo.push_back(value);
				} else {
					queue.pop_front();
					ring.pop_front();
					chunks.pop_front();
					fifo.pop_front();
				}
				bool const should_push = both.empty() || (both.size() < width && rng() % 100 < push_percent);
				bool const is_front = rng() % 2 != 0;
				if (should_push && is_front) {
					deque.push_front(value);
					both.push_front(value);
				} else if (should_push) {
					deque.push_back(value);
					both.push_back(value);
				} else if (is_front) {
					deque.pop_front();
					both.pop_front();
				} else {
					deque.pop_back();
					both.pop_back();
				}
				if (step % 499 == 0) {
					auto const ring_copy = ring;
					ring = ring_copy;
					auto chunks_copy = chunks;
					chunks = std::move(chunks_copy);
					auto deque_copy = deque;
					deque = std::move(deque_copy);
				}
				check(queue, fifo, identity, multiply2{});
				check(ring, fifo, identity, multiply2{});
				check(chunks, fifo, identity, multiply2{});
				check(deque, both, identity, multiply2{});
			}
		}
	}
}

TEST_CASE("aggregate containers: random traces against a recomputed noncommutative fold", "[ds][aggregate_queue][aggregate_deque]") { matrix_walks(); }

TEST_CASE("aggregate queues and deque: exhaustive legal affine traces", "[ds][aggregate_queue]") {
	exhaustive<atl::aggregate_queue<affine, compose>, false>(atl::aggregate_queue<affine, compose>({1, 0}), {}, 8);
	exhaustive<atl::experimental::aggregate_queue_by_daba_lite_ring<affine, compose>, false>(atl::experimental::aggregate_queue_by_daba_lite_ring<affine, compose>({1, 0}), {}, 8);
	exhaustive<atl::experimental::aggregate_queue_by_daba_lite_chunks<affine, compose>, false>(atl::experimental::aggregate_queue_by_daba_lite_chunks<affine, compose>({1, 0}), {}, 8);
	exhaustive<atl::aggregate_deque<affine, compose>, true>(atl::aggregate_deque<affine, compose>({1, 0}), {}, 6);
}

TEST_CASE("aggregate FIFO: seeded oracle and cross-engine agreement", "[ds][aggregate_queue]") {
	fifo_walk(std::uint64_t(0), std::plus<>{}, [](std::mt19937_64 &rng) { return rng() % 1000; });
	fifo_walk(std::uint64_t(0), std::bit_xor<>{}, [](std::mt19937_64 &rng) { return rng(); });
	fifo_walk(affine{1, 0}, compose{}, [](std::mt19937_64 &rng) { return affine{unsigned(rng() % 5), unsigned(rng() % 101)}; });
}

TEST_CASE("aggregate deque: boundary adversaries and seeded four-ended walk", "[ds][aggregate_deque]") {
	for (std::size_t n : {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025}) {
		deque_adversaries(n, false);
		deque_adversaries(n, true);
	}
	atl::aggregate_deque<affine, compose> actual({1, 0});
	std::deque<affine> oracle;
	std::mt19937_64 rng(0x48444551);
	for (unsigned step = 0; step < 30000; ++step) {
		auto const action = oracle.empty() ? rng() % 2 : (oracle.size() >= 256 ? 2 + rng() % 2 : rng() % 4);
		affine const value{unsigned(rng() % 5), unsigned(rng() % 101)};
		if (action == 0) {
			actual.emplace_front(value);
			oracle.push_front(value);
		}
		if (action == 1) {
			actual.emplace_back(value);
			oracle.push_back(value);
		}
		if (action == 2) {
			actual.pop_front();
			oracle.pop_front();
		}
		if (action == 3) {
			actual.pop_back();
			oracle.pop_back();
		}
		if (step % 103 == 0) actual.reserve(257);
		check(actual, oracle, affine{1, 0}, compose{});
	}
}

TEST_CASE("monotonic FIFO: exhaustive duplicate extrema and ordered inputs", "[ds][aggregate_queue]") {
	auto const exercise = []<bool IsMaximum>() {
		using compare = std::conditional_t<IsMaximum, std::greater<>, std::less<>>;
		auto const operation = [](int a, int b) { if constexpr (IsMaximum) return a > b ? a : b; else return a < b ? a : b; };
		int const identity = IsMaximum ? -1000 : 1000;
		for (unsigned code = 0; code < 6561; ++code) {
			atl::experimental::aggregate_queue_by_monotonic_candidates<int, compare> candidates(identity);
			std::deque<int> oracle;
			auto digits = code;
			for (unsigned step = 0; step < 8; ++step) {
				auto const action = digits % 3;
				digits /= 3;
				if (action == 2 && !oracle.empty()) {
					candidates.pop_front();
					oracle.pop_front();
				} else {
					int const value = action == 0 ? -1 : 1;
					candidates.push_back(value);
					oracle.push_back(value);
				}
				check(candidates, oracle, identity, operation);
			}
		}
		for (unsigned code = 0; code < 729; ++code) {
			atl::experimental::aggregate_queue_by_monotonic_candidates<int, compare> candidates(identity);
			atl::aggregate_queue reference(identity, operation);
			atl::aggregate_deque deque(identity, operation);
			atl::experimental::aggregate_queue_by_daba_lite_ring daba(identity, operation);
			atl::experimental::aggregate_queue_by_daba_lite_chunks chunked(identity, operation);
			std::deque<int> oracle;
			auto digits = code;
			for (unsigned i = 0; i < 6; ++i) {
				int const value = int(digits % 3) - 1;
				digits /= 3;
				candidates.emplace_back(value);
				reference.push_back(value);
				deque.push_back(value);
				daba.push_back(value);
				chunked.push_back(value);
				oracle.push_back(value);
				check(candidates, oracle, identity, operation);
				check(reference, oracle, identity, operation);
				check(deque, oracle, identity, operation);
				check(daba, oracle, identity, operation);
				check(chunked, oracle, identity, operation);
			}
			while (!oracle.empty()) {
				check(candidates, oracle, identity, operation);
				check(reference, oracle, identity, operation);
				check(deque, oracle, identity, operation);
				check(daba, oracle, identity, operation);
				check(chunked, oracle, identity, operation);
				candidates.pop_front();
				reference.pop_front();
				deque.pop_front();
				daba.pop_front();
				chunked.pop_front();
				oracle.pop_front();
			}
			check(candidates, oracle, identity, operation);
		}
		std::mt19937_64 rng(451);
		for (int shape = 0; shape < 4; ++shape) {
			atl::experimental::aggregate_queue_by_monotonic_candidates<int, compare> actual(identity);
			std::deque<int> oracle;
			actual.reserve(129);
			for (int step = 0; step < 6000; ++step) {
				int const value = shape == 0 ? step % 257 : shape == 1 ? 256 - step % 257
													  : shape == 2	   ? 5
																	   : int(rng() % 7);
				actual.push_back(value);
				oracle.push_back(value);
				check(actual, oracle, identity, operation);
				if (oracle.size() == 129 || rng() % 3 == 0) {
					actual.pop_front();
					oracle.pop_front();
				}
				check(actual, oracle, identity, operation);
				if (step % 1009 == 0) {
					auto copy = actual;
					actual.clear();
					REQUIRE(actual.fold() == identity);
					actual = std::move(copy);
					check(actual, oracle, identity, operation);
				}
			}
		}
	};
	exercise.template operator()<false>();
	exercise.template operator()<true>();
}

TEST_CASE("aggregate containers: objects, aliasing, no default values and move probes", "[ds][aggregate_queue]") {
	object_contracts<atl::aggregate_queue>();
	object_contracts<atl::aggregate_deque>();
	object_contracts<atl::experimental::aggregate_queue_by_daba_lite_ring>();
	object_contracts<atl::experimental::aggregate_queue_by_daba_lite_chunks>();
	using atl_test::move_probe;
	auto const operation = [](move_probe const &a, move_probe const &b) { return move_probe(a.v < b.v ? a.v : b.v); };
	auto const exercise = [&]<template<typename, typename> typename Container>()
		requires std::constructible_from<Container<int, std::plus<>>, int>
	{
		Container<move_probe, decltype(operation)> values(move_probe(move_probe::sentinel + 1), operation);
		values.reserve(8);
		move_probe::reset();
		values.push_back(move_probe(move_probe::sentinel));
		REQUIRE(move_probe::sentinel_moves > 0);
		REQUIRE(move_probe::sentinel_copies == 1);
		values.emplace_back(values.front());
		values.pop_front();
		REQUIRE(values.front().v == move_probe::sentinel);
	};
	exercise.template operator()<atl::aggregate_queue>();
	exercise.template operator()<atl::aggregate_deque>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_ring>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_chunks>();
	{
		auto compare = [offset = 0](token const &a, token const &b) { return a.value + offset < b.value + offset; };
		atl::experimental::aggregate_queue_by_monotonic_candidates values(token(1000), compare);
		values.emplace_back(3);
		values.emplace_back(values.front());
		values.emplace_back(2);
		REQUIRE(values.fold().value == 2);
		REQUIRE(values.front().value == 3);
		auto copy = values;
		values.clear();
		auto moved = std::move(copy);
		moved.pop_front();
		REQUIRE(moved.front().value == 3);
		REQUIRE(moved.back().value == 2);
		REQUIRE(values.fold().value == 1000);
	}
}

TEST_CASE("DABA Lite: per-update aggregation bound through phase changes", "[ds][aggregate_queue]") {
	auto exercise = []<template<typename, typename> typename Engine>()
		requires std::constructible_from<Engine<int, std::plus<>>, int>
	{
		std::size_t calls = 0;
		Engine<std::uint64_t, counted_sum> values(0, counted_sum{&calls});
		std::deque<std::uint64_t> oracle;
		std::mt19937_64 rng(177);
		for (unsigned step = 0; step < 100000; ++step) {
			bool const should_push = oracle.empty() || (step < 5000) || (step >= 10000 && (step % 6000 < 3000 || rng() % 3 == 0));
			calls = 0;
			if (should_push) {
				values.push_back(step % 19);
				oracle.push_back(step % 19);
				REQUIRE(calls <= 3);
			} else {
				values.pop_front();
				oracle.pop_front();
				REQUIRE(calls <= 2);
			}
			calls = 0;
			(void)std::as_const(values).fold();
			REQUIRE(calls <= 1);
			if (step % 257 == 0) check(values, oracle, std::uint64_t(0), counted_sum{&calls});
		}
	};
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_ring>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_chunks>();
}

TEST_CASE("aggregate stacks: linear total aggregation and move work on adversarial drains", "[ds][aggregate_deque]") {
	std::size_t calls = 0;
	atl::aggregate_deque<std::uint64_t, counted_sum> values(0, counted_sum{&calls});
	constexpr std::size_t n = 8193;
	values.reserve(n);
	for (std::size_t i = 0; i < n; ++i) values.push_back(1);
	for (std::size_t i = 0; i < n; ++i) {
		if (i % 2)
			values.pop_front();
		else
			values.pop_back();
	}
	REQUIRE(calls < 8 * n);
	REQUIRE(values.empty());
	REQUIRE(values.fold() == 0);
	atl::aggregate_queue<std::uint64_t, counted_sum> queue(0, counted_sum{&calls});
	calls = 0;
	for (std::size_t i = 0; i < n; ++i) queue.push_back(1);
	while (!queue.empty()) queue.pop_front();
	REQUIRE(calls < 2 * n);
	using atl_test::move_probe;
	auto const minimum = [](move_probe const &a, move_probe const &b) { return move_probe(a.v < b.v ? a.v : b.v); };
	atl::aggregate_deque probes(move_probe(move_probe::sentinel + 1), minimum);
	probes.reserve(n);
	move_probe::reset();
	for (std::size_t i = 0; i < n; ++i) probes.emplace_back(move_probe::sentinel);
	for (std::size_t i = 0; i < n; ++i) {
		if (i % 2)
			probes.pop_front();
		else
			probes.pop_back();
	}
	REQUIRE(std::cmp_less(move_probe::sentinel_copies + move_probe::sentinel_moves, 64 * n));
}

TEST_CASE("aggregate queue and deque: max_size is the smallest backing limit", "[ds][aggregate_queue][aggregate_deque]") {
	atl::aggregate_queue<int, std::plus<>> queue(0);
	atl::aggregate_deque<int, std::plus<>> deque(0);
	auto const stack_limit = std::vector<atl::detail::aggregate_entry<int>>().max_size();
	static_assert(noexcept(queue.max_size()) && noexcept(deque.max_size()));
	REQUIRE(queue.max_size() == stack_limit);
	REQUIRE(deque.max_size() == (std::min)(stack_limit, std::vector<int>().max_size()));
	REQUIRE(queue.max_size() > 0);
	queue.reserve(9);
	deque.reserve(9);
	REQUIRE(queue.max_size() >= 9);
	REQUIRE(deque.max_size() >= 9);
}

TEST_CASE("aggregate containers: asserted preconditions", "[ds][aggregate_queue]") {
	deaths<atl::aggregate_queue<int, std::plus<>>>();
	deaths<atl::aggregate_deque<int, std::plus<>>>();
	deaths<atl::experimental::aggregate_queue_by_daba_lite_ring<int, std::plus<>>>();
	deaths<atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>>>();
	deaths<atl::experimental::aggregate_queue_by_monotonic_candidates<int>>();
}
