#include <atl/ds/aggregate_deque.hpp>
#include <atl/ds/aggregate_queue.hpp>
#include <atl/experimental/ds/aggregate_queue_variants.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace {
	/** @brief Allocations left before the next one throws; the maximum disarms the probe. */
	std::size_t allocations_before_failure = (std::numeric_limits<std::size_t>::max)();

	/** @brief malloc, or std::bad_alloc once the armed count is exhausted; not constexpr (malloc). */
	[[nodiscard]] void *allocate_or_fail(std::size_t bytes) {
		if (allocations_before_failure != (std::numeric_limits<std::size_t>::max)()) {
			if (allocations_before_failure == 0) throw std::bad_alloc();
			--allocations_before_failure;
		}
		if (auto *result = std::malloc(bytes ? bytes : 1)) return result;
		throw std::bad_alloc();
	}
}

void *operator new(std::size_t bytes) { return allocate_or_fail(bytes); }

void *operator new[](std::size_t bytes) { return allocate_or_fail(bytes); }

void operator delete(void *pointer) noexcept { std::free(pointer); }

void operator delete[](void *pointer) noexcept { std::free(pointer); }

void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }

void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

namespace {
	/** @brief Run @p body with the probe armed; whether it threw std::bad_alloc. The probe is disarmed before any Catch assertion runs. */
	template<std::invocable Body>
	bool fails_allocation(std::size_t successful_allocations, Body body) {
		allocations_before_failure = successful_allocations;
		try {
			body();
			allocations_before_failure = (std::numeric_limits<std::size_t>::max)();
			return false;
		} catch (std::bad_alloc const &) {
			allocations_before_failure = (std::numeric_limits<std::size_t>::max)();
			return true;
		} catch (...) {
			allocations_before_failure = (std::numeric_limits<std::size_t>::max)();
			throw;
		}
	}

	/** @brief A reserve that fails at any of its allocations leaves the contents unchanged and usable. */
	template<typename Container, bool IsMinimum = false>
		requires std::constructible_from<Container, int>
	void reserve_failures() {
		bool has_succeeded = false;
		for (std::size_t permitted = 0; permitted < 32; ++permitted) {
			Container values(IsMinimum ? 1000 : 0);
			values.push_back(2);
			values.push_back(3);
			bool const has_failed = fails_allocation(permitted, [&] { values.reserve(193); });
			REQUIRE(values.size() == 2);
			REQUIRE(values.front() == 2);
			REQUIRE(values.back() == 3);
			REQUIRE(values.fold() == (IsMinimum ? 2 : 5));
			values.pop_front();
			REQUIRE(values.fold() == 3);
			values.push_back(7);
			REQUIRE(values.fold() == (IsMinimum ? 3 : 10));
			if (!has_failed) {
				has_succeeded = true;
				break;
			}
		}
		REQUIRE(has_succeeded);
	}
}

namespace {
	template<bool IsFront, typename Container>
	void push_value(Container &values, int value) {
		if constexpr (IsFront)
			values.push_front(value);
		else
			values.push_back(value);
	}

	/** @brief The values from front to back, read by popping a copy from the front. */
	template<typename Container>
	std::vector<int> drained_from_front(Container values) {
		std::vector<int> result;
		while (!values.empty()) {
			result.push_back(values.front());
			values.pop_front();
		}
		return result;
	}

	/** @brief The values from front to back, read by popping a copy from the back. */
	template<typename Container>
		requires requires(Container &values) { values.pop_back(); }
	std::vector<int> drained_from_back(Container values) {
		std::vector<int> result;
		while (!values.empty()) {
			result.push_back(values.back());
			values.pop_back();
		}
		std::ranges::reverse(result);
		return result;
	}

	/** @brief The container holds exactly @p expected, front to back, through every ending it offers, with the matching fold. */
	template<bool IsMinimum, typename Container>
	void require_contents(Container const &values, std::vector<int> const &expected) {
		REQUIRE(values.size() == expected.size());
		REQUIRE(drained_from_front(values) == expected);
		if constexpr (requires(Container &copy) { copy.pop_back(); }) REQUIRE(drained_from_back(values) == expected);
		int fold = IsMinimum ? 1000 : 0;
		for (int const value : expected) fold = IsMinimum ? (std::min)(fold, value) : fold + value;
		REQUIRE(values.fold() == fold);
	}

	/** @brief The sequence 1..n in push order, as read from front to back. */
	template<bool IsFront>
	std::vector<int> pushed_sequence(int n) {
		std::vector<int> result;
		for (int j = 1; j <= n; ++j) result.push_back(IsFront ? n - j + 1 : j);
		return result;
	}

	/**
	 * @brief A push that fails at its k-th allocation, for each k until the push succeeds, leaves the contents
	 * unchanged element by element; the same push then succeeds.
	 */
	template<typename Container, bool IsMinimum = false, bool IsFront = false>
		requires std::constructible_from<Container, int>
	void growth_failures() {
		int failures = 0;
		for (int i = 1; i <= 100; ++i) {
			for (std::size_t permitted = 0;; ++permitted) {
				Container values(IsMinimum ? 1000 : 0);
				for (int j = 1; j < i; ++j) push_value<IsFront>(values, j);
				bool const has_failed = fails_allocation(permitted, [&] { push_value<IsFront>(values, i); });
				if (!has_failed) {
					require_contents<IsMinimum>(values, pushed_sequence<IsFront>(i));
					break;
				}
				++failures;
				REQUIRE(permitted < 16);
				require_contents<IsMinimum>(values, pushed_sequence<IsFront>(i - 1));
				push_value<IsFront>(values, i);
				require_contents<IsMinimum>(values, pushed_sequence<IsFront>(i));
			}
		}
		REQUIRE(failures > 0);
	}

	/**
	 * @brief A copy assignment that fails at its k-th allocation, for each k until it succeeds, leaves the target and
	 * the source unchanged element by element and the target usable; both stacks of the source are occupied.
	 */
	template<typename Container, bool IsMinimum = false>
		requires std::constructible_from<Container, int>
	void copy_assignment_failures() {
		std::vector<int> source_values, target_values;
		for (int i = 2; i <= 60; ++i) source_values.push_back(i);
		for (int i = 100; i < 105; ++i) target_values.push_back(i);
		bool has_failed_once = false;
		for (std::size_t permitted = 0;; ++permitted) {
			Container source(IsMinimum ? 1000 : 0), target(IsMinimum ? 1000 : 0);
			for (int i = 1; i <= 30; ++i) source.push_back(i);
			source.pop_front();
			for (int i = 31; i <= 60; ++i) source.push_back(i);
			for (int const value : target_values) target.push_back(value);
			bool const has_failed = fails_allocation(permitted, [&] { target = source; });
			require_contents<IsMinimum>(source, source_values);
			if (!has_failed) {
				require_contents<IsMinimum>(target, source_values);
				break;
			}
			has_failed_once = true;
			REQUIRE(permitted < 200);
			require_contents<IsMinimum>(target, target_values);
			target.pop_front();
			target.push_back(7);
			std::vector<int> expected(target_values.begin() + 1, target_values.end());
			expected.push_back(7);
			require_contents<IsMinimum>(target, expected);
		}
		REQUIRE(has_failed_once);
	}
}

TEST_CASE("aggregate containers: failed growth preserves contents", "[ds][aggregate_queue]") {
	growth_failures<atl::aggregate_queue<int, std::plus<>>>();
	growth_failures<atl::aggregate_deque<int, std::plus<>>>();
	growth_failures<atl::aggregate_deque<int, std::plus<>>, false, true>();
	growth_failures<atl::experimental::aggregate_queue_by_daba_lite_ring<int, std::plus<>>>();
	growth_failures<atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>>>();
	growth_failures<atl::experimental::aggregate_queue_by_monotonic_candidates<int>, true>();
}

TEST_CASE("aggregate containers: allocation failures preserve contents", "[ds][aggregate_queue]") {
	reserve_failures<atl::aggregate_queue<int, std::plus<>>>();
	reserve_failures<atl::aggregate_deque<int, std::plus<>>>();
	reserve_failures<atl::experimental::aggregate_queue_by_daba_lite_ring<int, std::plus<>>>();
	reserve_failures<atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>>>();
	reserve_failures<atl::experimental::aggregate_queue_by_monotonic_candidates<int>, true>();

	for (std::size_t permitted = 0; permitted < 4; ++permitted) {
		atl::aggregate_deque<int, std::plus<>> values(0);
		for (int i = 1; i <= 33; ++i) values.push_back(i);
		bool const has_failed = fails_allocation(permitted, [&] { values.pop_front(); });
		REQUIRE(values.size() == (has_failed ? 33 : 32));
		REQUIRE(values.front() == (has_failed ? 1 : 2));
		REQUIRE(values.back() == 33);
		REQUIRE(values.fold() == (has_failed ? 561 : 560));
	}
	{
		atl::aggregate_queue<int, std::plus<>> values(0);
		for (int i = 1; i <= 33; ++i) values.push_back(i);
		REQUIRE(fails_allocation(0, [&] { values.pop_front(); }));
		REQUIRE(values.front() == 1);
		REQUIRE(values.back() == 33);
		REQUIRE(values.fold() == 561);
		values.pop_front();
		REQUIRE(values.fold() == 560);
	}
	{
		atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>> values(0);
		for (int i = 0; i < 31; ++i) values.push_back(1);
		REQUIRE(fails_allocation(0, [&] { values.push_back(2); }));
		REQUIRE(values.size() == 31);
		REQUIRE(values.fold() == 31);
		values.push_back(2);
		REQUIRE(values.fold() == 33);
	}
}

TEST_CASE("aggregate containers: failed copy assignment preserves the target", "[ds][aggregate_queue]") {
	copy_assignment_failures<atl::aggregate_queue<int, std::plus<>>>();
	copy_assignment_failures<atl::aggregate_deque<int, std::plus<>>>();
	copy_assignment_failures<atl::experimental::aggregate_queue_by_daba_lite_ring<int, std::plus<>>>();
	copy_assignment_failures<atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>>>();
	copy_assignment_failures<atl::experimental::aggregate_queue_by_monotonic_candidates<int>, true>();
}

TEST_CASE("aggregate reserve counts occupancy across arbitrarily many operations", "[ds][aggregate_queue]") {
	auto const exercise = []<typename Container>()
		requires std::constructible_from<Container, int>
	{
		Container values(0);
		values.reserve(65);
		bool const has_failed = fails_allocation(0, [&] {
			for (unsigned round = 0; round < 2000; ++round) {
				for (unsigned i = 0; i < 65; ++i) values.push_back(1);
				while (!values.empty()) values.pop_front();
			}
		});
		REQUIRE(!has_failed);
		REQUIRE(values.empty());
	};
	exercise.template operator()<atl::aggregate_queue<int, std::plus<>>>();
	exercise.template operator()<atl::aggregate_deque<int, std::plus<>>>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_ring<int, std::plus<>>>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_daba_lite_chunks<int, std::plus<>>>();
	exercise.template operator()<atl::experimental::aggregate_queue_by_monotonic_candidates<int, std::greater<>>>();
}
