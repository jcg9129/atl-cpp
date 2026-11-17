#include <atl/ds/persistent_adapters.hpp>
#include <atl/ds/persistent_array.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <deque>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>

using atl::uint64;

// (There is no persistent_stack: a persistent stack is persistent_list.)

// Random push/pop cross-checked against std::deque. The queue is unbounded, so the
// only restriction is that we pop from a non-empty queue.
TEST_CASE("persistent_queue matches a std::deque model", "[persistent_adapters]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	atl::persistent_queue<int> q;
	std::deque<int> ref;
	for (int step = 0; step < 5000; ++step) {
		if (ref.empty() || rng() % 2) {
			int v = int(rng());
			q = q.push(v);
			ref.push_back(v);
		} else {
			q = q.pop();
			ref.pop_front();
		}
		REQUIRE(q.size() == ref.size());
		REQUIRE(q.empty() == ref.empty());
		if (!ref.empty()) {
			REQUIRE(q.front() == ref.front());
			REQUIRE(q.back() == ref.back());
		}
	}
}

TEST_CASE("persistent_queue: full persistence branches off any version", "[persistent_adapters]") {
	atl::persistent_queue<int> q;
	auto a = q.push(1).push(2).push(3); // front..back: 1, 2, 3
	auto b = a.pop();					// 2, 3      (branch off a)
	auto c = a.push(4);					// 1, 2, 3, 4 (another branch off a)
	REQUIRE(a.size() == 3);
	REQUIRE(a.front() == 1); // a is untouched by b and c
	REQUIRE(a.back() == 3);
	REQUIRE(b.size() == 2);
	REQUIRE(b.front() == 2);
	REQUIRE(b.back() == 3);
	REQUIRE(c.size() == 4);
	REQUIRE(c.front() == 1);
	REQUIRE(c.back() == 4);
}

namespace {
	// complete but for operator[] returning by value: front() would dangle on it
	struct by_value_backing {
		explicit by_value_backing(std::size_t) {}

		int operator[](std::size_t) const { return 0; }

		std::size_t size() const { return 0; }

		by_value_backing set(std::size_t, int) const { return *this; }
	};
}

TEST_CASE("persistent_deque matches a std::deque model at both ends (wraps around)", "[persistent_adapters]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const cap = 8;

	atl::persistent_deque<int> d(cap);
	std::deque<int> ref;
	for (int step = 0; step < 5000; ++step) {
		int op = rng() % 4;
		if (op == 0 && ref.size() < cap) {
			int v = int(rng());
			d = d.push_back(v);
			ref.push_back(v);
		} else if (op == 1 && ref.size() < cap) {
			int v = int(rng());
			d = d.push_front(v);
			ref.push_front(v);
		} else if (op == 2 && !ref.empty()) {
			d = d.pop_back();
			ref.pop_back();
		} else if (!ref.empty()) {
			d = d.pop_front();
			ref.pop_front();
		}
		REQUIRE(d.size() == ref.size());
		if (!ref.empty()) {
			REQUIRE(d.front() == ref.front());
			REQUIRE(d.back() == ref.back());
			std::size_t const i = rng() % ref.size();
			REQUIRE(d[i] == ref[i]);
		}
	}
}

TEST_CASE("persistent_deque: default constructor, reserve and backing concept", "[persistent_adapters]") {
	atl::persistent_deque<int> placeholder;
	REQUIRE(placeholder.capacity() == 0);
	REQUIRE(placeholder.empty());
	placeholder = atl::persistent_deque<int>(3);
	placeholder.reserve(100);
	auto d = placeholder.push_back(1).push_back(2).push_front(0);
	for (int i = 0; i < 3; ++i) REQUIRE(d[std::size_t(i)] == i);

	atl::persistent_deque<int, atl::semi_persistent_array<int>> semi(2);
	semi.reserve(4);
	REQUIRE(semi.push_back(5)[0] == 5);

	static_assert(atl::detail::persistent_ring_backing<atl::persistent_array<int>, int>);
	static_assert(atl::detail::persistent_ring_backing<atl::semi_persistent_array<int>, int>);
	static_assert(!atl::detail::persistent_ring_backing<by_value_backing, int>);
}

TEST_CASE("persistent_deque: full persistence branches off any version", "[persistent_adapters]") {
	atl::persistent_deque<int> d(8);
	auto a = d.push_back(1).push_front(0); // [0, 1]
	auto b = a.push_back(2);			   // [0, 1, 2]
	auto c = a.push_front(9);			   // [9, 0, 1]  (branch off a)
	REQUIRE(a.size() == 2);
	REQUIRE(a.front() == 0);
	REQUIRE(a.back() == 1);
	REQUIRE(b.back() == 2);
	REQUIRE(c.front() == 9);
	REQUIRE(c.back() == 1);
}

TEST_CASE("adapters: emplace constructs in place", "[persistent_adapters]") {
	atl::persistent_queue<int> q;
	auto q2 = q.emplace(7).emplace(8);
	REQUIRE(q2.size() == 2);
	REQUIRE(q2.front() == 7);
	REQUIRE(q2.back() == 8);

	atl::persistent_deque<int> d(4);
	auto d2 = d.emplace_back(2).emplace_front(1); // [1, 2]
	REQUIRE(d2.size() == 2);
	REQUIRE(d2.front() == 1);
	REQUIRE(d2.back() == 2);
}

TEST_CASE("persistent_queue: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_adapters]") {
	using queue = atl::persistent_queue<int>;
	static_assert(requires { static_cast<queue (queue::*)(int) const &>(&queue::push); });
	static_assert(requires { static_cast<queue (queue::*)(int) &&>(&queue::push); });
	static_assert(requires { static_cast<queue (queue::*)() const &>(&queue::pop); });
	static_assert(requires { static_cast<queue (queue::*)() &&>(&queue::pop); });
	static_assert(requires(queue q) { { std::move(q).emplace(1) } -> std::same_as<queue>; });

	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto same = [](queue const &q, std::deque<int> const &ref) {
		if (q.size() != ref.size()) return false;
		return ref.empty() || (q.front() == ref.front() && q.back() == ref.back());
	};
	queue moved, copied;
	std::deque<int> ref;
	std::vector<std::pair<queue, std::deque<int>>> kept;
	for (int step = 0; step < 1500; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		int v = int(rng());
		if (ref.empty() || rng() % 3 != 0) {
			if (rng() % 2) {
				moved = std::move(moved).push(v);
				copied = copied.push(v);
			} else {
				moved = std::move(moved).emplace(v);
				copied = copied.emplace(v);
			}
			ref.push_back(v);
		} else {
			moved = std::move(moved).pop();
			copied = copied.pop();
			ref.pop_front();
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[q, r] : kept) REQUIRE(same(q, r));
	queue a = queue().push(1);
	auto b = std::move(a).push(2);
	a = b; // assign into a moved-from queue
	REQUIRE(a.size() == 2);
	auto c = std::move(a).pop();
	REQUIRE(c.front() == 2);
}

TEST_CASE("persistent_deque: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_adapters]") {
	using deque = atl::persistent_deque<int>;
	static_assert(requires { static_cast<deque (deque::*)(int) const &>(&deque::push_back); });
	static_assert(requires { static_cast<deque (deque::*)(int) &&>(&deque::push_back); });
	static_assert(requires { static_cast<deque (deque::*)(int) const &>(&deque::push_front); });
	static_assert(requires { static_cast<deque (deque::*)(int) &&>(&deque::push_front); });
	static_assert(requires { static_cast<deque (deque::*)() const &>(&deque::pop_back); });
	static_assert(requires { static_cast<deque (deque::*)() &&>(&deque::pop_back); });
	static_assert(requires { static_cast<deque (deque::*)() const &>(&deque::pop_front); });
	static_assert(requires { static_cast<deque (deque::*)() &&>(&deque::pop_front); });
	static_assert(requires(deque d) { { std::move(d).emplace_back(1) } -> std::same_as<deque>; });
	static_assert(requires(deque d) { { std::move(d).emplace_front(1) } -> std::same_as<deque>; });

	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t cap = 37;
	auto same = [](deque const &d, std::deque<int> const &ref) {
		if (d.size() != ref.size()) return false;
		for (std::size_t i = 0; i < ref.size(); ++i)
			if (d[i] != ref[i]) return false;
		return true;
	};
	deque moved(cap), copied(cap);
	std::deque<int> ref;
	std::vector<std::pair<deque, std::deque<int>>> kept;
	for (int step = 0; step < 2000; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		int v = int(rng());
		switch (rng() % 6) {
		case 0:
			if (ref.size() < cap) {
				moved = std::move(moved).push_back(v);
				copied = copied.push_back(v);
				ref.push_back(v);
			}
			break;
		case 1:
			if (ref.size() < cap) {
				moved = std::move(moved).push_front(v);
				copied = copied.push_front(v);
				ref.push_front(v);
			}
			break;
		case 2:
			if (ref.size() < cap) {
				moved = std::move(moved).emplace_back(v);
				copied = copied.emplace_back(v);
				ref.push_back(v);
			}
			break;
		case 3:
			if (ref.size() < cap) {
				moved = std::move(moved).emplace_front(v);
				copied = copied.emplace_front(v);
				ref.push_front(v);
			}
			break;
		case 4:
			if (!ref.empty()) {
				moved = std::move(moved).pop_back();
				copied = copied.pop_back();
				ref.pop_back();
			}
			break;
		default:
			if (!ref.empty()) {
				moved = std::move(moved).pop_front();
				copied = copied.pop_front();
				ref.pop_front();
			}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[d, r] : kept) REQUIRE(same(d, r));
	deque a = deque(4).push_back(1);
	auto b = std::move(a).push_back(2);
	a = b; // assign into a moved-from deque
	REQUIRE(a.size() == 2);
	auto c = std::move(a).pop_front();
	REQUIRE(c.size() == 1);
}

namespace {
	/** @brief Sparse backing certifying circular indices near the size_t limit without a huge allocation. */
	class sparse_ring_backing {
		std::size_t capacity_;
		std::map<std::size_t, int> values_;

	public:
		explicit sparse_ring_backing(std::size_t capacity) : capacity_(capacity) {}

		std::size_t size() const { return capacity_; }

		int const &operator[](std::size_t i) const { return values_.at(i); }

		sparse_ring_backing set(std::size_t i, int value) const {
			auto copy = *this;
			copy.values_[i] = value;
			return copy;
		}
	};
}

TEST_CASE("persistent deque wraps indices without overflowing size_t", "[persistent_adapters]") {
	atl::persistent_deque<int, sparse_ring_backing> empty((std::numeric_limits<std::size_t>::max)());
	auto const deque = empty.push_front(1).push_back(2).push_back(3);
	REQUIRE(deque.front() == 1);
	REQUIRE(deque[1] == 2);
	REQUIRE(deque.back() == 3);
}
