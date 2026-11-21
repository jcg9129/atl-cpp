/**
 * @file
 * @brief Races the ordered monoid folds of atl::aggregate_queue and atl::aggregate_deque against the experimental
 * aggregate_queue contenders; not measured yet.
 *
 * Contenders: the library deque (two stacks, split in half when one side empties) on every shape; on the FIFO
 * shapes also the library queue (two stacks, whole transfer when the front empties),
 * aggregate_queue_by_daba_lite_ring (DABA Lite schedule, constant operation calls per update, growable ring),
 * aggregate_queue_by_daba_lite_chunks (the same schedule on recycled chunks that never relocate) and, for min and
 * max, aggregate_queue_by_monotonic_candidates (candidate spans instead of folds).
 *
 * Shapes: a fixed-width sliding window, sawtooth fill-and-drain, alternating-end deque drains and a random
 * four-ended walk, at widths 16, 4096 and 131072; values are xor of random words, 4x4 matrix products over Z/101Z
 * (noncommutative), and min and max over random, increasing, decreasing and plateau inputs. Every trace is
 * generated before timing, ends empty, and is checked against the library deque (fold, size, endpoints) first.
 * Each row reports end-to-end replay throughput (fold read after every update), the same with reserve(width),
 * and a reserved replay; separate untimed passes report p99, p999 and maximum update latency (clock overhead
 * included), operation calls per update and per fold, allocation calls and peak requested heap bytes.
 * No benchmark record exists, so no CPU, compiler, flags, date or winner is claimed.
 */
#include <atl/ds/aggregate_deque.hpp>
#include <atl/ds/aggregate_queue.hpp>
#include <atl/experimental/ds/aggregate_queue_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace allocation_probe {
	/** @brief Requested heap bytes of the untimed census; the census's own map allocations are excluded. */
	struct state {
		std::unordered_map<void *, std::size_t> sizes;
		std::size_t live = 0, peak = 0, allocations = 0;
	};

	thread_local state *active = nullptr;
	thread_local bool is_bookkeeping = false;

	/** @brief Record an allocation while a census is active; not constexpr (instruments dynamic allocation). */
	void allocated(void *pointer, std::size_t bytes) {
		if (!active || is_bookkeeping) return;
		is_bookkeeping = true;
		try {
			active->sizes.emplace(pointer, bytes);
			atl_assert(bytes <= (std::numeric_limits<std::size_t>::max)() - active->live);
			active->live += bytes;
			active->peak = active->live > active->peak ? active->live : active->peak;
			++active->allocations;
		} catch (...) {
			is_bookkeeping = false;
			throw;
		}
		is_bookkeeping = false;
	}

	/** @brief Retire a recorded allocation; not constexpr (instruments dynamic allocation). */
	void freed(void *pointer) noexcept {
		if (!active || is_bookkeeping) return;
		is_bookkeeping = true;
		auto it = active->sizes.find(pointer);
		if (it != active->sizes.end()) {
			active->live -= it->second;
			active->sizes.erase(it);
		}
		is_bookkeeping = false;
	}

	/**
	 * @brief malloc, over-allocated and aligned with std::align when the alignment exceeds malloc's (MSVC has no
	 * std::aligned_alloc); the original pointer sits just below the aligned one. Not constexpr (malloc).
	 */
	[[nodiscard]] void *allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
		if (bytes == 0) bytes = 1;
		void *pointer = nullptr;
		if (alignment <= alignof(std::max_align_t)) {
			pointer = std::malloc(bytes);
			if (!pointer) throw std::bad_alloc();
		} else {
			auto const overhead = alignment - 1 + sizeof(void *);
			if (bytes > (std::numeric_limits<std::size_t>::max)() - overhead) throw std::bad_alloc();
			auto space = bytes + overhead;
			void *const allocation = std::malloc(space);
			if (!allocation) throw std::bad_alloc();
			pointer = static_cast<void **>(allocation) + 1;
			space -= sizeof(void *);
			if (!std::align(alignment, bytes, pointer, space)) {
				std::free(allocation);
				throw std::bad_alloc();
			}
			static_cast<void **>(pointer)[-1] = allocation;
		}
		try {
			allocated(pointer, bytes);
		} catch (...) {
			std::free(alignment <= alignof(std::max_align_t) ? pointer : static_cast<void **>(pointer)[-1]);
			throw;
		}
		return pointer;
	}

	/** @brief Release storage from allocate() with the same alignment; not constexpr (free). */
	void release(void *pointer, std::size_t alignment = alignof(std::max_align_t)) noexcept {
		if (!pointer) return;
		freed(pointer);
		std::free(alignment <= alignof(std::max_align_t) ? pointer : static_cast<void **>(pointer)[-1]);
	}
}

// Global allocation hooks: outside a census they add one branch per call and do no bookkeeping, so timed passes
// see plain malloc and free.

/** @brief Instrumented allocation; not constexpr (malloc). */
void *operator new(std::size_t bytes) { return allocation_probe::allocate(bytes); }

/** @brief Instrumented array allocation; not constexpr (malloc). */
void *operator new[](std::size_t bytes) { return allocation_probe::allocate(bytes); }

/** @brief Instrumented over-aligned allocation; not constexpr (malloc). */
void *operator new(std::size_t bytes, std::align_val_t alignment) { return allocation_probe::allocate(bytes, std::size_t(alignment)); }

/** @brief Instrumented over-aligned array allocation; not constexpr (malloc). */
void *operator new[](std::size_t bytes, std::align_val_t alignment) { return allocation_probe::allocate(bytes, std::size_t(alignment)); }

/** @brief Instrumented release; not constexpr (free). */
void operator delete(void *pointer) noexcept { allocation_probe::release(pointer); }

/** @brief Instrumented array release; not constexpr (free). */
void operator delete[](void *pointer) noexcept { allocation_probe::release(pointer); }

/** @brief Instrumented sized release; not constexpr (free). */
void operator delete(void *pointer, std::size_t) noexcept { allocation_probe::release(pointer); }

/** @brief Instrumented sized array release; not constexpr (free). */
void operator delete[](void *pointer, std::size_t) noexcept { allocation_probe::release(pointer); }

/** @brief Instrumented over-aligned release; not constexpr (free). */
void operator delete(void *pointer, std::align_val_t alignment) noexcept { allocation_probe::release(pointer, std::size_t(alignment)); }

/** @brief Instrumented over-aligned array release; not constexpr (free). */
void operator delete[](void *pointer, std::align_val_t alignment) noexcept { allocation_probe::release(pointer, std::size_t(alignment)); }

/** @brief Instrumented sized over-aligned release; not constexpr (free). */
void operator delete(void *pointer, std::size_t, std::align_val_t alignment) noexcept { allocation_probe::release(pointer, std::size_t(alignment)); }

/** @brief Instrumented sized over-aligned array release; not constexpr (free). */
void operator delete[](void *pointer, std::size_t, std::align_val_t alignment) noexcept { allocation_probe::release(pointer, std::size_t(alignment)); }

namespace {
	/** @brief 4x4 matrices over Z/101Z, singular ones included: a noncommutative monoid of fixed size. */
	using matrix = std::array<std::uint64_t, 16>;

	/** @brief The ordered product left * right. */
	struct multiply {
		matrix operator()(matrix const &left, matrix const &right) const {
			matrix result{};
			for (std::size_t row = 0; row < 4; ++row)
				for (std::size_t col = 0; col < 4; ++col) {
					for (std::size_t k = 0; k < 4; ++k) result[row * 4 + col] += left[row * 4 + k] * right[k * 4 + col];
					result[row * 4 + col] %= 101;
				}
			return result;
		}
	};

	/** @brief Minimum or maximum as a monoid operation, for the general engines. */
	template<bool IsMaximum>
	struct extremum {
		std::uint64_t operator()(std::uint64_t left, std::uint64_t right) const {
			if constexpr (IsMaximum)
				return left > right ? left : right;
			else
				return left < right ? left : right;
		}
	};
	/** @brief One pre-generated endpoint mutation. */
	enum class action { push_front,
						push_back,
						pop_front,
						pop_back };

	/** @brief One update of a trace, with its value already built, so generation stays outside the timing. */
	template<std::copyable Value>
	struct event {
		action kind;
		Value value;
	};

	/** @brief The payload itself, or a matrix derived from it. */
	template<typename Value>
		requires std::same_as<Value, std::uint64_t> || std::same_as<Value, matrix>
	Value materialize(std::uint64_t payload) {
		if constexpr (std::same_as<Value, std::uint64_t>)
			return payload;
		else {
			matrix result{};
			for (auto &element : result) {
				payload = std::rotl(payload, 7) ^ 0x9e3779b97f4a7c15ULL;
				element = payload % 101;
			}
			return result;
		}
	}

	/** @brief A legal trace of the given shape and value shape that never exceeds @p width live values and ends empty. */
	template<typename Value>
		requires std::same_as<Value, std::uint64_t> || std::same_as<Value, matrix>
	std::vector<event<Value>> make_trace(unsigned shape, std::size_t width, unsigned value_shape) {
		atl_assert(width > 0 && width <= 131072 && shape < 4 && value_shape < 4);
		std::vector<event<Value>> trace;
		std::mt19937_64 rng(0x48424153);
		std::size_t live = 0, arrivals = 0;
		auto const append = [&](action kind) {
			std::uint64_t payload = 0;
			if (kind == action::push_front || kind == action::push_back) {
				payload = value_shape == 0 ? rng() % 1'000'000 : value_shape == 1 ? arrivals
														   : value_shape == 2	  ? 4'000'000 - arrivals
																				  : 7;
				++arrivals;
				++live;
			} else {
				atl_assert(live != 0);
				--live;
			}
			trace.push_back({kind, materialize<Value>(payload)});
		};
		std::size_t const rounds = width < 4096 ? 4096 / width : 1;
		if (shape == 0) {
			for (std::size_t i = 0; i < width; ++i) append(action::push_back);
			for (std::size_t i = 0; i < 4 * width * rounds; ++i) {
				append(action::pop_front);
				append(action::push_back);
			}
		} else if (shape == 1) {
			for (std::size_t round = 0; round < 4 * rounds; ++round) {
				for (std::size_t i = 0; i < width; ++i) append(action::push_back);
				while (live) append(action::pop_front);
			}
		} else if (shape == 2) {
			for (std::size_t round = 0; round < 2 * rounds; ++round) {
				for (bool is_mirrored : {false, true}) {
					for (std::size_t i = 0; i < width; ++i) append(is_mirrored ? action::push_front : action::push_back);
					while (live) append((live % 2 != 0) == is_mirrored ? action::pop_front : action::pop_back);
				}
			}
		} else {
			for (std::size_t i = 0; i < 8 * width * rounds; ++i) {
				bool const is_filling = i % (4 * width) < 2 * width;
				bool const should_drive_boundary = i % (2 * width) >= width;
				bool const prefers_push = should_drive_boundary ? is_filling : (is_filling ? rng() % 4 != 0 : rng() % 4 == 0);
				bool const should_push = live == 0 || (live < width && prefers_push);
				append(should_push ? (rng() % 2 ? action::push_front : action::push_back) : (rng() % 2 ? action::pop_front : action::pop_back));
			}
		}
		while (live) append(action::pop_front);
		return trace;
	}

	/** @brief Apply one update; FIFO-only engines are never given a front push or back pop. */
	template<typename Container>
		requires requires(Container &c, typename Container::value_type v) { c.push_back(v); c.pop_front(); }
	void apply(Container &container, event<typename Container::value_type> const &entry) {
		switch (entry.kind) {
		case action::push_front:
			if constexpr (requires { container.push_front(entry.value); })
				container.push_front(entry.value);
			else
				std::abort();
			break;
		case action::push_back:
			container.push_back(entry.value);
			break;
		case action::pop_front:
			container.pop_front();
			break;
		case action::pop_back:
			if constexpr (requires { container.pop_back(); })
				container.pop_back();
			else
				std::abort();
			break;
		}
	}

	/** @brief A checksum word that depends on the whole fold. */
	template<typename Value>
		requires std::same_as<Value, std::uint64_t> || std::same_as<Value, matrix>
	std::uint64_t digest(Value const &value) {
		if constexpr (std::same_as<Value, std::uint64_t>)
			return value;
		else {
			std::uint64_t result = 0;
			for (auto element : value) result = std::rotl(result, 7) ^ element;
			return result;
		}
	}

	/** @brief Replay a trace, reading the fold after every update. */
	template<typename Container>
		requires requires(Container &c) { c.fold(); c.empty(); }
	std::uint64_t replay(Container &container, std::vector<event<typename Container::value_type>> const &trace) {
		std::uint64_t checksum = 0;
		for (auto const &entry : trace) {
			apply(container, entry);
			checksum = std::rotl(checksum, 1) ^ digest(container.fold());
		}
		atl_assert(container.empty());
		return checksum;
	}

	/** @brief The operation, counting its calls. */
	template<std::copy_constructible Operation>
	struct counted {
		Operation operation;
		std::size_t *calls;

		template<typename Value>
			requires std::invocable<Operation const &, Value const &, Value const &>
		auto operator()(Value const &left, Value const &right) const {
			++*calls;
			return std::invoke(operation, left, right);
		}
	};

	/** @brief Print one untimed metric line, which record.sh keeps beside the timing rows. */
	template<std::integral Number>
	void metric(std::string const &row, char const *name, Number value) { std::cout << "ATL_PROFILE\t" << row << '\t' << name << '\t' << value << '\n'; }

	/** @brief Check one engine against the library deque on the trace, take the untimed metrics, then time it. */
	template<template<typename, typename> typename Engine, std::copyable Value, typename ConstructionOperation, atl::binary_operation<Value> FoldOperation>
		requires std::constructible_from<Engine<Value, ConstructionOperation>, Value, ConstructionOperation>
	void race(std::string const &row, std::vector<event<Value>> const &trace, std::size_t width, Value identity, ConstructionOperation construction_operation, FoldOperation fold_operation, bool counts_comparisons = false) {
		{
			Engine<Value, ConstructionOperation> actual(identity, construction_operation);
			atl::aggregate_deque reference(identity, fold_operation);
			for (auto const &entry : trace) {
				apply(actual, entry);
				apply(reference, entry);
				REQUIRE(actual.fold() == reference.fold());
				REQUIRE(actual.size() == reference.size());
				if (!actual.empty()) {
					REQUIRE(actual.front() == reference.front());
					REQUIRE(actual.back() == reference.back());
				}
			}
		}
		for (bool is_reserved : {false, true}) {
			std::string const tag = row + (is_reserved ? " reserved" : " cold");
			std::size_t calls = 0, max_push = 0, max_pop = 0, update_calls = 0, fold_calls = 0;
			allocation_probe::state memory;
			allocation_probe::active = &memory;
			try {
				{
					Engine<Value, counted<ConstructionOperation>> actual(identity, counted<ConstructionOperation>{construction_operation, &calls});
					if (is_reserved) actual.reserve(width);
					for (auto const &entry : trace) {
						calls = 0;
						apply(actual, entry);
						bool const is_push = entry.kind == action::push_front || entry.kind == action::push_back;
						if (is_push)
							max_push = calls > max_push ? calls : max_push;
						else
							max_pop = calls > max_pop ? calls : max_pop;
						update_calls += calls;
						calls = 0;
						(void)actual.fold();
						fold_calls += calls;
					}
				}
			} catch (...) {
				allocation_probe::active = nullptr;
				throw;
			}
			allocation_probe::active = nullptr;
			REQUIRE(memory.live == 0);
			metric(tag, "updates", trace.size());
			metric(tag, counts_comparisons ? "update_comparisons" : "update_aggregate_calls", update_calls);
			metric(tag, "fold_aggregate_calls", fold_calls);
			metric(tag, counts_comparisons ? "max_push_comparisons" : "max_push_aggregate_calls", max_push);
			metric(tag, "max_pop_aggregate_calls", max_pop);
			metric(tag, "peak_requested_heap_bytes", memory.peak);
			metric(tag, "allocation_calls", memory.allocations);
			std::vector<std::chrono::nanoseconds::rep> latency(trace.size());
			Engine<Value, ConstructionOperation> actual(identity, construction_operation);
			if (is_reserved) actual.reserve(width);
			std::uint64_t checksum = 0;
			for (std::size_t i = 0; i < trace.size(); ++i) {
				auto const start = std::chrono::steady_clock::now();
				apply(actual, trace[i]);
				auto const stop = std::chrono::steady_clock::now();
				latency[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
				checksum = std::rotl(checksum, 1) ^ digest(actual.fold());
			}
			Catch::Benchmark::deoptimize_value(checksum);
			std::ranges::sort(latency);
			metric(tag, "p99_update_ns", latency[latency.size() - 1 - latency.size() / 100]);
			metric(tag, "p999_update_ns", latency[latency.size() - 1 - latency.size() / 1000]);
			metric(tag, "max_update_ns", latency.back());
		}
		BENCHMARK(row + " end_to_end") {
			Engine<Value, ConstructionOperation> actual(identity, construction_operation);
			return replay(actual, trace);
		};
		BENCHMARK(row + " end_to_end_with_reserve") {
			Engine<Value, ConstructionOperation> actual(identity, construction_operation);
			actual.reserve(width);
			return replay(actual, trace);
		};
		BENCHMARK_ADVANCED(row + " reserved_replay")(Catch::Benchmark::Chronometer meter) {
			Engine<Value, ConstructionOperation> actual(identity, construction_operation);
			actual.reserve(width);
			meter.measure([&] { return replay(actual, trace); });
		};
	}

	/** @brief The library deque on every shape; the queue engines on the FIFO shapes (0 and 1). */
	template<std::copyable Value, atl::binary_operation<Value> Operation>
	void general_race(std::string const &tag, std::vector<event<Value>> const &trace, unsigned shape, std::size_t width, Value identity, Operation operation) {
		race<atl::aggregate_deque>(tag + " library aggregate_deque", trace, width, identity, operation, operation);
		if (shape < 2) {
			race<atl::aggregate_queue>(tag + " library aggregate_queue", trace, width, identity, operation, operation);
			race<atl::experimental::aggregate_queue_by_daba_lite_ring>(tag + " by daba_lite_ring", trace, width, identity, operation, operation);
			race<atl::experimental::aggregate_queue_by_daba_lite_chunks>(tag + " by daba_lite_chunks", trace, width, identity, operation, operation);
		}
	}
}

TEST_CASE("aggregate queue/deque: library engines and contenders", "[aggregate_queue_deque]") {
	std::array<std::string, 4> const shape_names{"fixed_fifo", "sawtooth_fifo", "alternating_deque", "random_deque"};
	std::array<std::string, 4> const value_names{"random", "increasing", "decreasing", "plateau"};
	for (std::size_t width : {std::size_t(16), std::size_t(4096), std::size_t(131072)}) {
		for (unsigned shape = 0; shape < 4; ++shape) {
			std::string const tag = shape_names[shape] + " width=" + std::to_string(width);
			{
				auto const trace = make_trace<std::uint64_t>(shape, width, 0);
				general_race(tag + " xor", trace, shape, width, std::uint64_t(0), std::bit_xor<>{});
			}
			{
				auto const trace = make_trace<matrix>(shape, width, 0);
				matrix identity{};
				for (std::size_t i = 0; i < 4; ++i) identity[4 * i + i] = 1;
				general_race(tag + " matrix4", trace, shape, width, identity, multiply{});
			}
			for (unsigned value_shape = 0; value_shape < 4; ++value_shape) {
				auto const trace = make_trace<std::uint64_t>(shape, width, value_shape);
				std::string const row = tag + " " + value_names[value_shape];
				general_race(row + " min", trace, shape, width, (std::numeric_limits<std::uint64_t>::max)(), extremum<false>{});
				general_race(row + " max", trace, shape, width, std::uint64_t(0), extremum<true>{});
				if (shape < 2) {
					race<atl::experimental::aggregate_queue_by_monotonic_candidates>(row + " min by monotonic_candidates", trace, width, (std::numeric_limits<std::uint64_t>::max)(), std::less<>{}, extremum<false>{}, true);
					race<atl::experimental::aggregate_queue_by_monotonic_candidates>(row + " max by monotonic_candidates", trace, width, std::uint64_t(0), std::greater<>{}, extremum<true>{}, true);
				}
			}
		}
	}
}
