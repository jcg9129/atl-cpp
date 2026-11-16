#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/ds/time_divide_and_conquer_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using atl::uint8;
namespace ex = atl::experimental;

namespace {
	/**
	 * @brief A rollback structure that records the inserted items, to compare
	 *        against the live set. The oracle of this family is @ref alive_at (and
	 *        @ref brute_lifetimes for lifetimes_of): every engine is checked against it.
	 */
	struct item_stack {
		std::vector<uint32> items;

		/** @brief The token: how many items are inserted. */
		std::size_t time() const { return items.size(); }

		/** @brief Drop the items inserted after token `t`. */
		void rollback(std::size_t t) { items.resize(t); }
	};

	/** @brief The same, but with an overloaded `time`, so no default rollback mode applies. */
	struct overloaded_stack {
		std::vector<uint32> items;

		/** @brief The token, as item_stack's. */
		std::size_t time() const { return items.size(); }

		/** @brief An overload, which makes `&overloaded_stack::time` ambiguous. */
		std::size_t time(int) const { return items.size(); }

		/** @brief As item_stack's. */
		void rollback(std::size_t t) { items.resize(t); }
	};

	/** @brief Neither copyable nor rollback-capable: no restore mode fits. */
	struct move_only {
		std::unique_ptr<int> p;
	};

	/** @brief A free-function save, for by_rollback. */
	std::size_t free_save(item_stack &s) { return s.items.size(); }

	/** @brief A free-function restore, for by_rollback. */
	void free_restore(item_stack &s, std::size_t t) { s.items.resize(t); }

	static_assert(std::same_as<atl::default_restore_t<atl::rollback_dsu>,
							   atl::by_rollback<decltype(&atl::rollback_dsu::time), decltype(&atl::rollback_dsu::rollback)>>);
	static_assert(std::same_as<atl::default_restore_t<atl::persistent_dsu<>>, atl::by_copy_t>);
	static_assert(std::same_as<atl::default_restore_t<overloaded_stack>, atl::by_copy_t>);
	static_assert(atl::restore_mode<atl::by_copy_t, item_stack>);
	static_assert(!atl::restore_mode<atl::by_copy_t, move_only>);
	static_assert(!atl::restore_mode<atl::default_restore_t<move_only>, move_only>);
	static_assert(atl::lifetime_range<std::vector<atl::item_lifetime>>);
	static_assert(atl::lifetime_range<std::vector<std::pair<int, int>>>);
	static_assert(!atl::lifetime_range<std::vector<int>>);
	static_assert(atl::item_event_range<std::vector<atl::item_event<std::string>>>);
	static_assert(!atl::item_event_range<std::vector<std::pair<int, int>>>);

	/** @brief The engines behind one call shape. */
	enum class engine { library,
						recursive,
						unpruned,
						unstretched,
						skipping,
						dense_csr,
						rank_compressed_skipping };

	/** @brief Every engine, for the loops. */
	constexpr engine all_engines[]{engine::library, engine::recursive, engine::unpruned,
								   engine::unstretched, engine::skipping, engine::dense_csr,
								   engine::rank_compressed_skipping};

	template<typename... Args>
		requires requires(Args &&...args) { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
	constexpr void run(engine e, Args &&...args) {
		switch (e) {
		case engine::library:
			atl::time_divide_and_conquer(std::forward<Args>(args)...);
			break;
		case engine::recursive:
			ex::time_divide_and_conquer_by_recursive_vectors(std::forward<Args>(args)...);
			break;
		case engine::unpruned:
			ex::time_divide_and_conquer_by_unpruned_csr(std::forward<Args>(args)...);
			break;
		case engine::unstretched:
			ex::time_divide_and_conquer_by_configurable_csr<true, false>(std::forward<Args>(args)...);
			break;
		case engine::skipping:
			ex::time_divide_and_conquer_by_frame_skipping(std::forward<Args>(args)...);
			break;
		case engine::dense_csr:
			ex::time_divide_and_conquer_by_dense_csr(std::forward<Args>(args)...);
			break;
		case engine::rank_compressed_skipping:
			ex::time_divide_and_conquer_by_ranked_frame_skipping(std::forward<Args>(args)...);
			break;
		}
	}

	/** @brief Random lifetimes in `[0, T)`, empty ones included. */
	std::vector<std::pair<int, int>> random_lifetimes(std::size_t m, uint32 time_count, std::mt19937_64 &rng) {
		std::vector<std::pair<int, int>> lifetimes(m);
		for (auto &[b, e] : lifetimes) {
			b = int(rng() % (time_count + 1)), e = int(rng() % (time_count + 1));
			if (b > e) std::swap(b, e);
		}
		return lifetimes;
	}

	/** @brief The items alive at `t`, sorted: the oracle. */
	std::vector<uint32> alive_at(std::vector<std::pair<int, int>> const &lifetimes, uint32 t) {
		std::vector<uint32> alive;
		for (uint32 i = 0; i < lifetimes.size(); ++i)
			if (lifetimes[i].first <= int(t) && int(t) < lifetimes[i].second) alive.push_back(i);
		return alive;
	}

	/** @brief Runs one engine and mode on one instance and checks every visit against the oracle. */
	template<atl::restore_mode<item_stack> Mode>
	void check_instance(engine e, Mode mode, std::vector<std::pair<int, int>> const &lifetimes, uint32 time_count,
						std::vector<uint8> const &is_query) {
		item_stack s;
		std::vector<uint32> visited;
		uint64 inserts = 0;
		uint32 predicate_calls = 0;
		run(
			e, s, time_count, lifetimes,
			[&](item_stack &st, uint32 i) {
				st.items.push_back(i);
				++inserts;
			},
			[&](item_stack &st, uint32 t) {
				auto got = st.items;
				std::ranges::sort(got);
				CHECK(got == alive_at(lifetimes, t));
				visited.push_back(t);
			},
			[&](uint32 t) {
				++predicate_calls;
				return is_query[t] != 0;
			},
			mode);
		CHECK(predicate_calls == time_count);
		std::vector<uint32> expected;
		for (uint32 t = 0; t < time_count; ++t)
			if (is_query[t]) expected.push_back(t);
		CHECK(visited == expected);
		CHECK(s.items.empty());																		  // restored (rollback) or never touched (copy)
		if (expected.empty() && e != engine::recursive && e != engine::unpruned) CHECK(inserts == 0); // pruned entirely
	}
}

TEST_CASE("time_divide_and_conquer: random instances, every engine and mode", "[time_divide_and_conquer]") {
	std::mt19937_64 rng(2026'10'24);
	for (int iter = 0; iter < 400; ++iter) {
		uint32 const time_count = uint32(1 + rng() % 70);
		auto const lifetimes = random_lifetimes(rng() % 40, time_count, rng);
		int const density = int(rng() % 4); // none, sparse, half, all
		std::vector<uint8> is_query(time_count);
		for (auto &q : is_query) q = density == 3 || (density > 0 && rng() % (density == 1 ? 10 : 2) == 0);
		for (engine e : all_engines) {
			check_instance(e, atl::default_restore<item_stack>(), lifetimes, time_count, is_query);
			check_instance(e, atl::by_copy, lifetimes, time_count, is_query);
			check_instance(e, atl::by_rollback{free_save, free_restore}, lifetimes, time_count, is_query);
			check_instance(
				e,
				atl::by_rollback{[](item_stack &s) { return std::pair{s.items.size(), std::string("token")}; },
								 [](item_stack &s, std::pair<std::size_t, std::string> token) {
									 CHECK(token.second == "token");
									 s.items.resize(token.first);
								 }},
				lifetimes, time_count, is_query);
		}
	}
}

namespace {
	/** @brief An item_stack that counts its copies, to measure the frames of copy mode. */
	struct counted_stack {
		std::vector<uint32> items;
		uint64 *copies;

		/** @brief Starts empty, counting into `*c`. */
		explicit counted_stack(uint64 *c) : copies(c) {}

		/** @brief Counted. */
		counted_stack(counted_stack const &o) : items(o.items), copies(o.copies) { ++*copies; }

		/** @brief Counted. */
		counted_stack &operator=(counted_stack const &o) {
			items = o.items, copies = o.copies;
			++*copies;
			return *this;
		}
	};

	/** @brief The frames one engine opens on one instance: copies in copy mode, saves in rollback mode (checked against the oracle). */
	struct frame_counts {
		uint64 copies = 0, saves = 0;
	};

	/** @brief Runs one engine by copy and by rollback on one instance, oracle-checking every visit, and counts its frames. */
	frame_counts count_frames(engine e, std::vector<std::pair<int, int>> const &lifetimes, uint32 time_count,
							  std::vector<uint8> const &is_query) {
		frame_counts out;
		auto const insert = []<typename Stack>(Stack &st, uint32 i)
			requires std::same_as<Stack, item_stack> || std::same_as<Stack, counted_stack>
		{ st.items.push_back(i); };
		auto const visit = [&]<typename Stack>(Stack &st, uint32 t)
			requires std::same_as<Stack, item_stack> || std::same_as<Stack, counted_stack>
		{
			auto got = st.items;
			std::ranges::sort(got);
			CHECK(got == alive_at(lifetimes, t));
		};
		auto const should_visit = [&](uint32 t) { return is_query[t] != 0; };
		counted_stack c(&out.copies);
		run(e, c, time_count, lifetimes, insert, visit, should_visit, atl::by_copy);
		uint64 restores = 0;
		item_stack r;
		run(e, r, time_count, lifetimes, insert, visit, should_visit,
			atl::by_rollback{[&](item_stack &s) {
								 ++out.saves;
								 return s.items.size();
							 },
							 [&](item_stack &s, std::size_t t) {
								 ++restores;
								 s.items.resize(t);
							 }});
		CHECK(restores == out.saves);
		CHECK(r.items.empty());
		return out;
	}
}

TEST_CASE("time_divide_and_conquer: skipping opens fewer frames", "[time_divide_and_conquer]") {
	// T = 4, all visited: [2, 4) sits on node 3, [3, 4) on leaf 7, a last child below node 3's frame
	std::vector<std::pair<int, int>> const nested{{2, 4}, {3, 4}};
	std::vector<uint8> const all(4, 1);
	CHECK(count_frames(engine::dense_csr, nested, 4, all).copies == 2);
	CHECK(count_frames(engine::skipping, nested, 4, all).copies == 1);
	CHECK(count_frames(engine::skipping, nested, 4, all).saves == 1);
	// [0, 1) on leaf 4, a left child whose live sibling needs node 2's state: no skip; [1, 2) on leaf 5 is
	// last, but nothing above it has items: no open frame to absorb into
	std::vector<std::pair<int, int>> const leaves{{0, 1}, {1, 2}};
	CHECK(count_frames(engine::skipping, leaves, 4, all).copies == 2);
	// a pruned right sibling makes the left child last: queries only at 0, [0, 4) on the root absorbs [0, 1)
	std::vector<std::pair<int, int>> const pruned{{0, 4}, {0, 1}};
	std::vector<uint8> const first{1, 0, 0, 0};
	CHECK(count_frames(engine::dense_csr, pruned, 4, first).copies == 2);
	CHECK(count_frames(engine::skipping, pruned, 4, first).copies == 1);

	std::mt19937_64 rng(691);
	uint64 previous_total = 0, skipping_total = 0;
	for (int iter = 0; iter < 300; ++iter) {
		uint32 const time_count = uint32(1 + rng() % 90);
		auto const lifetimes = random_lifetimes(rng() % 50, time_count, rng);
		std::vector<uint8> is_query(time_count);
		int const density = int(rng() % 3);
		for (auto &q : is_query) q = density == 2 || rng() % (density == 0 ? 10 : 2) == 0;
		auto const dense_csr = count_frames(engine::dense_csr, lifetimes, time_count, is_query);
		auto const skipping = count_frames(engine::skipping, lifetimes, time_count, is_query);
		CHECK(skipping.copies <= dense_csr.copies);
		CHECK(skipping.saves <= dense_csr.saves);
		CHECK(skipping.copies == skipping.saves); // one frame is one copy or one save
		CHECK(dense_csr.copies == dense_csr.saves);
		if (std::cmp_less_equal(std::ranges::count(is_query, uint8(1)), time_count / 10))
			CHECK(count_frames(engine::rank_compressed_skipping, lifetimes, time_count, is_query).copies
				  <= count_frames(engine::library, lifetimes, time_count, is_query).copies);
		previous_total += dense_csr.copies, skipping_total += skipping.copies;
	}
	CHECK(skipping_total < previous_total);
}

TEST_CASE("time_divide_and_conquer: dispatch at one tenth", "[time_divide_and_conquer]") {
	std::vector<std::pair<int, int>> const lifetimes{{0, 99}};
	for (uint32 queries : {0u, 1u, 9u, 10u, 11u, 50u, 100u}) {
		CAPTURE(queries);
		std::vector<uint8> is_query(100, 0);
		for (uint32 k = 0; k < queries; ++k) is_query[k * 100 / queries] = 1;
		check_instance(engine::library, atl::by_copy, lifetimes, 100, is_query);
		check_instance(engine::library, atl::default_restore<item_stack>(), lifetimes, 100, is_query);
		auto const library = count_frames(engine::library, lifetimes, 100, is_query);
		auto const dense_csr = count_frames(engine::dense_csr, lifetimes, 100, is_query);
		if (queries == 9 || queries == 10) {
			CHECK(library.copies == 1);
			CHECK(library.saves == 1);
			CHECK(dense_csr.copies > library.copies);
		} else if (queries > 10) {
			CHECK(library.copies == dense_csr.copies);
			CHECK(library.saves == dense_csr.saves);
		}
	}
}

TEST_CASE("time_divide_and_conquer: sparse forward lifetimes preserve original times", "[time_divide_and_conquer]") {
	// Endpoints fall before, on and after the two queries, including padding and empty ranked intervals.
	std::list<atl::item_lifetime> const lifetimes{{0, 33}, {6, 7}, {7, 8}, {8, 31}, {31, 32}, {32, 33}, {7, 7}, {33, 33}};
	for (bool const is_copying : {false, true}) {
		item_stack s;
		uint32 predicate_calls = 0;
		std::vector<uint32> visited;
		auto const insert = [](item_stack &st, uint32 i) { st.items.push_back(i); };
		auto const visit = [&](item_stack &st, uint32 t) {
			auto got = st.items;
			std::ranges::sort(got);
			CHECK(got == (t == 7 ? std::vector<uint32>{0, 2} : std::vector<uint32>{0, 4}));
			visited.push_back(t);
		};
		auto const should_visit = [&](uint32 t) {
			CHECK(t == predicate_calls++);
			return t == 7 || t == 31;
		};
		if (is_copying)
			atl::time_divide_and_conquer(s, 33, lifetimes, insert, visit, should_visit, atl::by_copy);
		else
			atl::time_divide_and_conquer(s, 33, lifetimes, insert, visit, should_visit);
		CHECK(predicate_calls == 33);
		CHECK(visited == std::vector<uint32>{7, 31});
		CHECK(s.items.empty());
	}
}

TEST_CASE("time_divide_and_conquer: retained engines on edge instances", "[time_divide_and_conquer]") {
	// items alive across the whole range, zero-length lifetimes at every position, queries at the ends only
	for (uint32 time_count : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 19u, 20u, 31u, 32u, 33u, 63u, 64u, 65u, 99u, 100u, 101u, 127u, 128u, 129u}) {
		std::vector<std::pair<int, int>> lifetimes{{0, int(time_count)}, {0, int(time_count)}, {0, 1}, {int(time_count) - 1, int(time_count)}};
		for (int t = 0; t <= int(time_count); ++t) lifetimes.push_back({t, t});
		std::vector<uint8> ends(time_count, 0), all(time_count, 1), last(time_count, 0), none(time_count, 0);
		ends.front() = ends.back() = 1;
		last.back() = 1;
		for (auto const &is_query : {ends, all, last, none})
			for (engine e : {engine::library, engine::skipping, engine::dense_csr, engine::rank_compressed_skipping}) {
				check_instance(e, atl::by_copy, lifetimes, time_count, is_query);
				check_instance(e, atl::default_restore<item_stack>(), lifetimes, time_count, is_query);
			}
	}
	CHECK(atl_test::aborts([] {
		item_stack s;
		std::vector<std::pair<int, int>> past_end{{3, 6}};
		ex::time_divide_and_conquer_by_ranked_frame_skipping(
			s, 5, past_end, [](item_stack &st, uint32 i) { st.items.push_back(i); }, [](item_stack &, uint32) {});
	}));
}

TEST_CASE("time_divide_and_conquer: rollback_dsu by rollback and persistent_dsu by copy", "[time_divide_and_conquer]") {
	std::mt19937_64 rng(813);
	for (int iter = 0; iter < 200; ++iter) {
		std::size_t const n = 1 + rng() % 12;
		uint32 const time_count = uint32(1 + rng() % 50);
		auto const lifetimes = random_lifetimes(rng() % 30, time_count, rng);
		std::vector<std::pair<uint32, uint32>> edge(lifetimes.size());
		for (auto &[u, v] : edge) u = uint32(rng() % n), v = uint32(rng() % n);
		std::vector<std::size_t> expected(time_count);
		for (uint32 t = 0; t < time_count; ++t) {
			atl::rollback_dsu fresh(n);
			for (uint32 i : alive_at(lifetimes, t)) fresh.unite(edge[i].first, edge[i].second);
			expected[t] = fresh.count();
		}
		std::vector<std::size_t> got(time_count);
		atl::rollback_dsu d(n);
		atl::time_divide_and_conquer(
			d, time_count, lifetimes, [&](atl::rollback_dsu &s, uint32 i) { s.unite(edge[i].first, edge[i].second); },
			[&](atl::rollback_dsu &s, uint32 t) { got[t] = s.count(); });
		CHECK(got == expected);
		CHECK(d.count() == n);
		std::ranges::fill(got, 0);
		atl::persistent_dsu<> p(n);
		atl::time_divide_and_conquer(
			p, time_count, lifetimes,
			[&](atl::persistent_dsu<> &s, uint32 i) { s = std::move(s).unite(edge[i].first, edge[i].second); },
			[&](atl::persistent_dsu<> &s, uint32 t) { got[t] = s.count(); });
		CHECK(got == expected);
		CHECK(p.count() == n);
		for (engine e : {engine::skipping, engine::rank_compressed_skipping}) {
			std::ranges::fill(got, 0);
			run(
				e, p, time_count, lifetimes,
				[&](atl::persistent_dsu<> &s, uint32 i) { s = std::move(s).unite(edge[i].first, edge[i].second); },
				[&](atl::persistent_dsu<> &s, uint32 t) { got[t] = s.count(); });
			CHECK(got == expected);
			std::ranges::fill(got, 0);
			run(
				e, d, time_count, lifetimes, [&](atl::rollback_dsu &s, uint32 i) { s.unite(edge[i].first, edge[i].second); },
				[&](atl::rollback_dsu &s, uint32 t) { got[t] = s.count(); });
			CHECK(got == expected);
			CHECK(d.count() == n);
		}
	}
}

TEST_CASE("time_divide_and_conquer: edge cases", "[time_divide_and_conquer]") {
	item_stack s;
	int visits = 0;
	auto const insert = [](item_stack &st, uint32 i) { st.items.push_back(i); };
	auto const count_visit = [&](item_stack &, uint32) { ++visits; };
	std::vector<atl::item_lifetime> none;
	atl::time_divide_and_conquer(s, 0, none, insert, count_visit);
	CHECK(visits == 0);
	atl::time_divide_and_conquer(s, 1, none, insert, count_visit);
	CHECK(visits == 1);
	std::vector<atl::item_lifetime> const whole{{0, 5}, {2, 2}, {5, 5}};
	atl::time_divide_and_conquer(
		s, 5, whole, insert, [&](item_stack &st, uint32) { CHECK(st.items == std::vector<uint32>{0}); }, atl::by_copy);
	CHECK(s.items.empty());
	// an overloaded time() falls back to by_copy; by_rollback still takes lambdas
	overloaded_stack o;
	atl::time_divide_and_conquer(
		o, 5, whole, [](overloaded_stack &st, uint32 i) { st.items.push_back(i); },
		[&](overloaded_stack &st, uint32) { CHECK(st.items.size() == 1); });
	atl::time_divide_and_conquer(
		o, 5, whole, [](overloaded_stack &st, uint32 i) { st.items.push_back(i); },
		[&](overloaded_stack &st, uint32) { CHECK(st.items.size() == 1); }, atl::every_time{},
		atl::by_rollback{[](overloaded_stack &st) { return st.items.size(); },
						 [](overloaded_stack &st, std::size_t t) { st.rollback(t); }});
	CHECK(o.items.empty());
}

TEST_CASE("time_divide_and_conquer: preconditions", "[time_divide_and_conquer]") {
	item_stack s;
	auto const insert = [](item_stack &st, uint32 i) { st.items.push_back(i); };
	auto const visit = [](item_stack &, uint32) {};
	CHECK(atl_test::aborts([&] {
		std::vector<std::pair<int, int>> reversed{{3, 2}};
		atl::time_divide_and_conquer(s, 5, reversed, insert, visit);
	}));
	CHECK(atl_test::aborts([&] {
		std::vector<std::pair<int, int>> past_end{{3, 6}};
		atl::time_divide_and_conquer(s, 5, past_end, insert, visit);
	}));
	for (auto const &invalid : {std::pair{10, 9}, std::pair{30, 34}, std::pair{-1, 10}})
		CHECK(atl_test::aborts([&] {
			std::vector<std::pair<int, int>> const lifetimes{invalid};
			atl::time_divide_and_conquer(s, 33, lifetimes, insert, visit, [](uint32 t) { return t == 7; });
		}));
}

namespace {
	/** @brief The lifetimes oracle: a per-key FIFO of open adds in an ordered map. */
	template<std::totally_ordered Key>
		requires std::copyable<Key>
	std::vector<atl::item_lifetime> brute_lifetimes(std::vector<atl::item_event<Key>> const &events) {
		std::vector<atl::item_lifetime> out;
		std::map<Key, std::deque<std::size_t>> open;
		for (uint32 t = 0; t < events.size(); ++t) {
			if (events[t].kind == atl::event_kind::add) {
				open[events[t].key].push_back(out.size());
				out.push_back({t, uint32(events.size())});
			} else if (events[t].kind == atl::event_kind::remove) {
				auto &q = open[events[t].key];
				out[q.front()].end = t;
				q.pop_front();
			}
		}
		return out;
	}

	/** @brief A random valid log over `keys` distinct keys made by `make_key`. */
	template<std::copyable Key, std::invocable<uint32> MakeKey>
		requires std::convertible_to<std::invoke_result_t<MakeKey &, uint32>, Key>
	std::vector<atl::item_event<Key>> random_log(std::size_t length, uint32 keys, MakeKey make_key, std::mt19937_64 &rng) {
		std::vector<atl::item_event<Key>> events;
		std::vector<uint32> live; // a multiset of open keys
		for (std::size_t i = 0; i < length; ++i) {
			auto const r = rng() % 3;
			if (r == 0 && !live.empty()) {
				std::size_t const j = rng() % live.size();
				events.push_back({atl::event_kind::remove, make_key(live[j])});
				live[j] = live.back(), live.pop_back();
			} else if (r == 1) {
				events.push_back({atl::event_kind::query, make_key(uint32(rng() % keys))}); // the key is ignored
			} else {
				uint32 const k = uint32(rng() % keys);
				live.push_back(k);
				events.push_back({atl::event_kind::add, make_key(k)});
			}
		}
		return events;
	}
}

TEST_CASE("lifetimes_of: random logs against the FIFO oracle", "[time_divide_and_conquer][lifetimes_of]") {
	std::mt19937_64 rng(100551);
	for (int iter = 0; iter < 300; ++iter) {
		std::size_t const length = rng() % 60;
		uint32 const keys = uint32(1 + rng() % 6);
		auto const ints = random_log<int>(length, keys, [](uint32 k) { return int(k) - 3; }, rng);
		CHECK(atl::lifetimes_of(ints) == brute_lifetimes(ints));
		auto const strings = random_log<std::string>(length, keys, [](uint32 k) { return std::string(k + 1, 'x'); }, rng);
		CHECK(atl::lifetimes_of(strings) == brute_lifetimes(strings));
		using pair = std::pair<uint32, uint32>;
		auto const pairs = random_log<pair>(length, keys, [](uint32 k) { return pair{k % 2, k / 2}; }, rng);
		auto const lifetimes = atl::lifetimes_of(pairs);
		CHECK(lifetimes == brute_lifetimes(pairs));
		for (auto const &l : lifetimes) CHECK(pairs[l.begin].kind == atl::event_kind::add);
	}
	CHECK(atl::lifetimes_of(std::vector<atl::item_event<int>>{}).empty());
}

TEST_CASE("lifetimes_of: preconditions", "[time_divide_and_conquer][lifetimes_of]") {
	using event = atl::item_event<int>;
	CHECK(atl_test::aborts([] { (void)atl::lifetimes_of(std::vector<event>{{atl::event_kind::remove, 1}}); }));
	CHECK(atl_test::aborts([] {
		(void)atl::lifetimes_of(std::vector<event>{{atl::event_kind::add, 1}, {atl::event_kind::remove, 1}, {atl::event_kind::remove, 1}});
	}));
}

TEST_CASE("time_divide_and_conquer: a move-only lifetime view on both paths", "[time_divide_and_conquer]") {
	for (std::size_t const visited : {std::size_t(1), std::size_t(10)}) {
		atl::rollback_dsu d(2);
		std::size_t visits = 0;
		atl::time_divide_and_conquer(
			d, 10, std::views::all(std::vector<atl::item_lifetime>{{0, 1}}),
			[](atl::rollback_dsu &s, atl::uint32) { s.unite(0, 1); },
			[&](atl::rollback_dsu &s, atl::uint32 t) {
				++visits;
				REQUIRE(s.same(0, 1) == (t == 0));
			},
			[&](atl::uint32 t) { return t < visited; });
		REQUIRE(visits == visited);
	}
}

namespace {
	static_assert([] {
		std::array<std::pair<int, int>, 4> const lifetimes{{{0, 4}, {1, 3}, {2, 2}, {3, 4}}};
		std::array<std::pair<int, int>, 4> const edges{{{0, 1}, {1, 2}, {2, 3}, {1, 3}}};
		auto const counts = [&]<atl::restore_mode<atl::rollback_dsu> Mode>(engine e, Mode mode) {
			atl::rollback_dsu d(4);
			std::array<std::size_t, 4> out{};
			run(e, d, 4, lifetimes, [&](atl::rollback_dsu &state, atl::uint32 i) { state.unite(edges[i].first, edges[i].second); }, [&](atl::rollback_dsu &state, atl::uint32 t) { out[t] = state.count(); }, [](atl::uint32 t) { return t != 2; }, mode);
			return out;
		};
		std::array<std::size_t, 4> const expected{3, 2, 0, 2};
		for (engine e : all_engines) {
			if (counts(e, atl::default_restore<atl::rollback_dsu>()) != expected || counts(e, atl::by_copy) != expected) return false;
		}
		return true;
	}());
}
