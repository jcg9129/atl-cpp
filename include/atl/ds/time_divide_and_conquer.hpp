#pragma once
#include <atl/ds/flat_hash_map.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief @ref atl::time_divide_and_conquer, the offline-deletion engine: items
 *        alive on time intervals are inserted into a structure that only
 *        supports insertion (plus undo, or cheap copies), and a callback sees
 *        the structure holding exactly the live items at each chosen time.
 *        @ref atl::lifetimes_of turns an add / remove / query event log into
 *        those intervals.
 */

namespace atl {
	/**
	 * @brief The half-open time interval `[begin, end)` on which one item is
	 *        alive, as @ref lifetimes_of returns it. Empty when `begin == end`.
	 */
	struct item_lifetime {
		uint32 begin; ///< @brief The first time the item is alive.
		uint32 end;	  ///< @brief The first time after `begin` the item is gone.

		/** @brief Componentwise equality. */
		constexpr bool operator==(item_lifetime const &) const = default;
	};

	/** @brief What one @ref item_event does. */
	enum class event_kind : uint8 {
		add,	///< a new copy of the key comes alive
		remove, ///< the oldest live copy of the key dies
		query	///< nothing changes: a time point to look at
	};

	/**
	 * @brief One entry of an event log for @ref lifetimes_of: add a copy of
	 *        `key`, remove the oldest live copy of `key`, or a query.
	 *
	 * A query's key is ignored by @ref lifetimes_of; it can carry the query's
	 * own data (the vertex pair a connectivity query asks about, say), read back
	 * as `events[t].key` in the visit callback.
	 *
	 * @tparam Key Anything hashable by @ref atl::hash and equality comparable:
	 *             an integer, a vertex pair `std::pair<uint32, uint32>`,
	 *             a string, a tuple; specialize @ref atl::hash for an own type.
	 *             Left unconstrained here (a plain record); @ref item_event_range,
	 *             what @ref lifetimes_of accepts, enforces it.
	 */
	template<typename Key>
	struct item_event {
		event_kind kind; ///< @brief The operation.
		Key key;		 ///< @brief The key it applies to (ignored for a query).
	};

	/**
	 * @brief Restore mode for @ref time_divide_and_conquer: mutate the structure in
	 *        place and undo on the way back up. `save(s)` returns a token of any
	 *        type, `restore(s, token)` returns `s` to the state `save` saw. Both are
	 *        called through `std::invoke`, so member pointers work:
	 *        `atl::by_rollback{&atl::rollback_dsu::time, &atl::rollback_dsu::rollback}`.
	 *
	 * @tparam Save    Invocable on `Structure &`, returning the token.
	 * @tparam Restore Invocable on `Structure &` and the token.
	 */
	template<std::move_constructible Save, std::move_constructible Restore>
	struct by_rollback {
		[[no_unique_address]] Save save;	   ///< @brief The state token getter.
		[[no_unique_address]] Restore restore; ///< @brief The undo to a token.
	};

	/** @brief Tag type of @ref by_copy. */
	struct by_copy_t {
		/** @brief Construct the explicit copy-mode tag. */
		explicit by_copy_t() = default;
	};

	/**
	 * @brief Restore mode for @ref time_divide_and_conquer: copy the structure at
	 *        every tree node that has items, insert into the copy, drop it on the
	 *        way back up. For copyable structures without undo (an xor basis, a
	 *        knapsack array) and for persistent ones, whose copies are O(1).
	 */
	inline constexpr by_copy_t by_copy{};

	/** @brief The default visit predicate of @ref time_divide_and_conquer: every time. */
	struct every_time {
		/** @brief Always true. */
		constexpr bool operator()(uint32) const noexcept { return true; }
	};

	namespace detail {
		/** @brief An element of a lifetime range with `begin` / `end` members, as @ref item_lifetime. */
		template<typename Element>
		concept member_lifetime = requires(Element const &e) {
			{ e.begin } -> std::convertible_to<std::size_t>;
			{ e.end } -> std::convertible_to<std::size_t>;
		};

		/** @brief A tuple-like two-element lifetime, as `std::pair<int, int>`. */
		template<typename Element>
		concept pair_lifetime = requires(Element const &e) {
			// tuple_size of a non-tuple is incomplete: the typename requirement
			// guards the arity check before tuple_size_v is instantiated
			typename std::tuple_size<Element>::type;
			requires std::tuple_size_v<Element> == 2;
			{ std::get<0>(e) } -> std::convertible_to<std::size_t>;
			{ std::get<1>(e) } -> std::convertible_to<std::size_t>;
		};

		/** @brief The `[begin, end)` of one lifetime element, either shape. */
		template<typename Element>
			requires member_lifetime<Element> || pair_lifetime<Element>
		constexpr std::pair<std::size_t, std::size_t> lifetime_bounds(Element const &e) {
			if constexpr (member_lifetime<Element>)
				return {std::size_t(e.begin), std::size_t(e.end)};
			else
				return {std::size_t(std::get<0>(e)), std::size_t(std::get<1>(e))};
		}

		/**
		 * @brief Whether `Structure` has unambiguous `time()` and `rollback(token)`
		 *        members (@ref rollback_dsu, @ref rollback_weighted_dsu): taking their
		 *        addresses fails on an overload set, which keeps such a type out of
		 *        the default rollback mode instead of making it ambiguous.
		 */
		template<typename Structure>
		concept time_rollback_members = requires {
			&Structure::time;
			&Structure::rollback;
		} && std::invocable<decltype(&Structure::time), Structure &> && std::invocable<decltype(&Structure::rollback), Structure &, std::invoke_result_t<decltype(&Structure::time), Structure &>>;

		template<typename Mode>
		inline constexpr bool is_by_rollback = false;

		template<std::move_constructible Save, std::move_constructible Restore>
		inline constexpr bool is_by_rollback<by_rollback<Save, Restore>> = true;

		template<typename Save, typename Structure>
		using rollback_token_t = std::remove_cvref_t<std::invoke_result_t<Save &, Structure &>>;

		template<typename Structure>
		struct default_restore_mode {
			using type = by_copy_t;
		};

		template<time_rollback_members Structure>
		struct default_restore_mode<Structure> {
			using type = by_rollback<decltype(&Structure::time), decltype(&Structure::rollback)>;
		};
	}

	/**
	 * @brief A range of item lifetimes for @ref time_divide_and_conquer: a sized
	 *        forward range (it is read twice) whose elements are `[begin, end)`
	 *        intervals, as @ref item_lifetime (members `begin` / `end`) or any
	 *        tuple-like pair (`std::pair<int, int>`). Item `i` is the `i`-th element.
	 * @tparam R Const-iterable sized forward range; lifetime extraction must work on its const-bound references.
	 */
	template<typename R>
	concept lifetime_range =
		std::ranges::forward_range<R const> && std::ranges::sized_range<R const>
		&& (detail::member_lifetime<std::ranges::range_value_t<R>> || detail::pair_lifetime<std::ranges::range_value_t<R>>)
		&& requires(std::remove_reference_t<std::ranges::range_reference_t<R const>> const &e) { detail::lifetime_bounds(e); };

	/** @brief `insert(s, i)` applies item `i` (a `uint32`) to the structure.
	 * @tparam Insert Insertion callback. @tparam Structure Mutable represented state. */
	template<typename Insert, typename Structure>
	concept item_insertion = std::invocable<Insert &, Structure &, uint32>;

	/** @brief `visit(s, t)` reads the structure at time `t` (a `uint32`).
	 * @tparam Visit Visit callback. @tparam Structure Mutable access to the represented state. */
	template<typename Visit, typename Structure>
	concept time_visit = std::invocable<Visit &, Structure &, uint32>;

	/** @brief `should_visit(t)` selects the times `t` (a `uint32`) to visit.
	 * @tparam Predicate Boolean time predicate. */
	template<typename Predicate>
	concept time_predicate = std::predicate<Predicate &, uint32>;

	/**
	 * @brief A restore mode of @ref time_divide_and_conquer for `Structure`:
	 *        @ref by_copy_t when `Structure` is copyable, or a @ref by_rollback whose
	 *        `save` is invocable on `Structure &` and whose `restore` is invocable on
	 *        `Structure &` and `save`'s result.
	 * @tparam Mode Restoration policy. @tparam Structure State restored by the policy.
	 */
	template<typename Mode, typename Structure>
	concept restore_mode = (std::same_as<Mode, by_copy_t> && std::copyable<Structure>) || (detail::is_by_rollback<Mode> && requires(Mode &m, Structure &s) {
							   { std::invoke(m.save, s) } -> std::movable;
							   std::invoke(m.restore, s, std::invoke(m.save, s));
						   });

	/**
	 * @brief The restore mode used when none is passed:
	 *        `by_rollback{&Structure::time, &Structure::rollback}` when both members
	 *        exist unambiguously, otherwise @ref by_copy (for a copyable structure).
	 * @tparam Structure State whose restore mode is selected.
	 */
	template<typename Structure>
	using default_restore_t = typename detail::default_restore_mode<Structure>::type;

	/** @brief The value of @ref default_restore_t.
	 * @tparam Structure State whose restore mode is selected.
	 * @return Deduced rollback member pointers, or the copy-mode tag. */
	template<typename Structure>
	[[nodiscard]] constexpr default_restore_t<Structure> default_restore() {
		if constexpr (std::same_as<default_restore_t<Structure>, by_copy_t>)
			return by_copy;
		else
			return {&Structure::time, &Structure::rollback};
	}

	namespace detail {
		/**
		 * @brief The undo stack of the rollback mode: one token per entered tree
		 *        node that has items; the structure itself is mutated in place.
		 */
		template<typename Structure, typename Save, typename Restore>
			requires std::invocable<Save &, Structure &> && std::move_constructible<Save> && std::move_constructible<Restore>
				  && std::movable<rollback_token_t<Save, Structure>>
				  && std::invocable<Restore &, Structure &, rollback_token_t<Save, Structure>>
		class rollback_frames {
			using token = rollback_token_t<Save, Structure>;
			Structure &structure_;
			by_rollback<Save, Restore> &mode_;
			std::vector<token> tokens_;

		public:
			/** @brief Frames over `structure`; `depth` bounds the nesting (a reserve hint). */
			constexpr rollback_frames(Structure &structure, by_rollback<Save, Restore> &mode, std::size_t depth)
				: structure_(structure), mode_(mode) {
				tokens_.reserve(depth);
			}

			/** @brief The structure the current node inserts into and visits. */
			constexpr Structure &top() noexcept { return structure_; }

			/** @brief Open a node: remember the state to come back to. */
			constexpr void push() { tokens_.push_back(std::invoke(mode_.save, structure_)); }

			/** @brief Close the innermost open node: undo its insertions (and everything below it). */
			constexpr void pop() {
				std::invoke(mode_.restore, structure_, std::move(tokens_.back()));
				tokens_.pop_back();
			}
		};

		/**
		 * @brief The copy stack of the copy mode: one copy per entered tree node
		 *        that has items. Slots are kept after a pop and copy-assigned on the
		 *        next push at that depth, so a structure owning buffers (a vector)
		 *        reuses their capacity instead of reallocating per node.
		 */
		template<std::copyable Structure>
		class copy_frames {
			Structure &root_;
			std::vector<Structure> copies_;
			std::size_t depth_ = 0;

		public:
			/**
			 * @brief Frames over `root`, which is only read (the callbacks see it
			 *        directly before the first push); at most `depth` nested pushes.
			 */
			constexpr copy_frames(Structure &root, std::size_t depth) : root_(root) {
				copies_.reserve(depth); // push_back(top()) below must not reallocate under its own argument
			}

			/** @brief The current state: the innermost copy, or the root before any push. */
			constexpr Structure &top() noexcept { return depth_ ? copies_[depth_ - 1] : root_; }

			/** @brief Open a node: a fresh copy of the current state, to insert into. */
			constexpr void push() {
				if (depth_ < copies_.size())
					copies_[depth_] = top();
				else
					copies_.push_back(top());
				++depth_;
			}

			/** @brief Close the innermost open node, dropping its copy. */
			constexpr void pop() noexcept { --depth_; }
		};

		/**
		 * @brief Call `node(v)` for each node `v` of the canonical cover of
		 *        `[begin, end)` in a perfect segment tree with `leaves` leaves,
		 *        heap-numbered (root 1, leaf `t` at `leaves + t`): the O(log)
		 *        maximal nodes whose leaf ranges tile the interval.
		 */
		template<typename NodeVisit>
			requires std::invocable<NodeVisit &, uint32>
		constexpr void for_each_cover_node(uint32 begin, uint32 end, uint32 leaves, NodeVisit &&node) {
			for (uint32 l = begin + leaves, r = end + leaves; l < r; l >>= 1, r >>= 1) {
				if (l & 1) node(l++);
				if (r & 1) node(--r);
			}
		}

		/**
		 * @brief A frame stack of a restore mode (@ref rollback_frames,
		 *        @ref copy_frames): `push` opens a node, `pop` closes the innermost,
		 *        `top` is the structure to insert into and visit.
		 */
		template<typename Frames>
		concept restore_frames = requires(Frames &f) {
			f.push();
			f.pop();
			f.top();
			requires std::is_lvalue_reference_v<decltype(f.top())>;
		};

		template<restore_frames Frames>
		using frames_structure_t = std::remove_reference_t<decltype(std::declval<Frames &>().top())>;

		/**
		 * @brief The iterative depth-first walk shared by both restore modes: enter
		 *        node 1, insert its items, descend into live children, visit live
		 *        leaves, close nodes on the way up.
		 * @param offsets the CSR row starts, `2 * leaves + 1` entries
		 * @param items the items of each node, `items[offsets[v] .. offsets[v + 1])`
		 * @param is_live per node, whether its subtree holds a visited time; the root is live
		 */
		template<restore_frames Frames, item_insertion<frames_structure_t<Frames>> Insert,
				 time_visit<frames_structure_t<Frames>> Visit>
		constexpr void walk_time_tree(Frames &frames, uint32 leaves, std::vector<uint32> const &offsets,
									  std::vector<uint32> const &items, std::vector<uint8> const &is_live, Insert &insert,
									  Visit &visit) {
			uint32 v = 1;
			while (true) {
				if (offsets[v] != offsets[v + 1]) {
					frames.push();
					auto &s = frames.top();
					for (uint32 k = offsets[v]; k < offsets[v + 1]; ++k) std::invoke(insert, s, items[k]);
				}
				if (v < leaves) {
					v = is_live[2 * std::size_t(v)] ? 2 * v : 2 * v + 1; // a live node has a live child
					continue;
				}
				std::invoke(visit, frames.top(), v - leaves);
				while (true) {
					if (offsets[v] != offsets[v + 1]) frames.pop();
					if (v == 1) return;
					if (!(v & 1) && is_live[v + 1]) {
						++v;
						break;
					}
					v >>= 1;
				}
			}
		}

		/**
		 * @brief Build the pruned CSR over `time_count` times and walk it; `is_live`
		 *        has `2 * leaves` entries with the selected leaves already marked.
		 * @pre `0 < time_count <= leaves == bit_ceil(time_count) <= 2^30`;
		 *      lifetime bounds and item counts satisfy the public engine's contract.
		 */
		template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
				 time_visit<Structure> Visit, restore_mode<Structure> Mode>
		constexpr void run_time_tree(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
									 uint32 leaves, std::vector<uint8> &is_live, Insert &insert, Visit &visit, Mode &mode) {
			atl_assert(time_count > 0 && time_count <= (std::size_t(1) << 30));
			atl_assert(leaves == std::bit_ceil(uint32(time_count)));
			atl_assert(is_live.size() == 2 * std::size_t(leaves));
			uint32 const height = uint32(std::countr_zero(leaves));
			for (std::size_t v = leaves; v-- > 1;) is_live[v] = uint8(is_live[2 * v] | is_live[2 * v + 1]);
			atl_assert(is_live[1]);
			// A lifetime reaching T is stretched over the padding leaves [T, leaves), which are
			// never visited: a suffix of the tree needs at most one node per level instead of two.
			auto const cover_end = [&](std::size_t end) { return end == time_count ? leaves : uint32(end); };
			std::vector<uint32> offsets(2 * std::size_t(leaves) + 1, 0);
			uint64 entries = 0;
			for (auto const &lifetime : lifetimes) {
				auto const [begin, end] = lifetime_bounds(lifetime);
				atl_assert(begin <= end && end <= time_count);
				if (begin == end) continue;
				for_each_cover_node(uint32(begin), cover_end(end), leaves, [&](uint32 v) {
					if (is_live[v]) ++offsets[v], ++entries;
				});
			}
			atl_assert(entries < uint64((std::numeric_limits<uint32>::max)()));
			uint32 running = 0; // offsets[v] becomes the end of row v, then the fill decrements it to the start
			for (auto &o : offsets) o = running += o;
			std::vector<uint32> items(static_cast<std::size_t>(entries));
			{
				uint32 i = 0;
				for (auto const &lifetime : lifetimes) {
					auto const [begin, end] = lifetime_bounds(lifetime);
					if (begin != end) {
						for_each_cover_node(uint32(begin), cover_end(end), leaves, [&](uint32 v) {
							if (is_live[v]) items[--offsets[v]] = i;
						});
					}
					++i;
				}
			}
			// filled backwards, so each row lists its items in decreasing index order; insertion order is free

			if constexpr (std::same_as<Mode, by_copy_t>) {
				copy_frames<Structure> frames(structure, height + 1);
				walk_time_tree(frames, leaves, offsets, items, is_live, insert, visit);
			} else {
				rollback_frames frames(structure, mode, height + 1);
				walk_time_tree(frames, leaves, offsets, items, is_live, insert, visit);
			}
		}

	}

	/**
	 * @brief The offline-deletion engine (segment tree over time): item `i` is
	 *        alive on `lifetimes[i] = [begin, end)` of the times `[0, time_count)`;
	 *        for every time `t` with `should_visit(t)`, in increasing order,
	 *        `visit(s, t)` sees `s` holding exactly the items alive at `t`.
	 *
	 * Each lifetime is cut into the O(log T) nodes of a segment tree over time
	 * that tile it; a depth-first walk inserts a node's items on the way down and
	 * restores the state on the way up, so the structure only ever needs
	 * insertion plus one of the two restore modes:
	 * - @ref by_rollback `{save, restore}`: mutate in place, `restore(s, save(s))`
	 *   undoes everything inserted since `save`. Each insertion must cost its
	 *   bound in the **worst case**: the walk undoes and redoes, which defeats
	 *   amortized bounds (path compression, say).
	 * - @ref by_copy: copy `s` at every node with items and insert into the copy.
	 *   O(T) copies in total, O(log T) alive at once; `structure` is left unchanged.
	 *
	 * Without `mode`, a structure with `time()` / `rollback(token)` members uses
	 * `by_rollback{&Structure::time, &Structure::rollback}`, and a copyable one
	 * without them uses @ref by_copy. Subtrees holding no visited time are pruned:
	 * neither walked nor inserted into. With Q <= T / 10 selected times, rank
	 * compression builds the tree over Q leaves and maps visits back to original
	 * times. Denser workloads only count selected times during the leaf-flag pass:
	 * the extra O(T) rank pass and buffer are reserved for the measured sparse regime.
	 *
	 *     atl::rollback_dsu d(n);
	 *     atl::time_divide_and_conquer(d, T, lifetimes,
	 *         [&](atl::rollback_dsu &d, uint32 i) { d.unite(edge[i].u, edge[i].v); },
	 *         [&](atl::rollback_dsu &d, uint32 t) { answer[t] = d.count(); },
	 *         [&](uint32 t) { return is_query[t]; });
	 *
	 * @tparam Structure Insertable state with rollback or copy restoration.
	 * @tparam Lifetimes Const sized forward range of half-open item intervals.
	 * @tparam Insert Item-index callback.
	 * @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Predicate selecting visited times.
	 * @tparam Mode State restoration policy.
	 * @param structure the empty-time state; in rollback mode it ends as it started
	 * @param time_count   `T`, the number of time points
	 * @param lifetimes one `[begin, end)` per item (see @ref lifetime_range)
	 * @param insert       `insert(s, i)` applies item `i` to `s`
	 * @param visit        `visit(s, t)` reads the state at `t`; it must leave the
	 *                     represented state unchanged (internal rebalancing such as
	 *                     path compression is fine in copy mode)
	 * @param should_visit which times to visit (default: all)
	 * @param mode         @ref by_rollback or @ref by_copy (default: see above)
	 * @pre `begin <= end <= time_count` for every lifetime, `time_count <= 2^30`,
	 *      fewer than `2^32 - 1` items and cover entries. Counts are asserted;
	 *      lifetime bounds are asserted when at least one time is selected.
	 * @pre Insertions represent an order-independent collection of active items;
	 *      insertion order is unspecified. Restore must undo all represented changes,
	 *      and copies must isolate represented mutations. These laws are documented-only.
	 * @note With no selected times (including T == 0), no insertion, visit, save or
	 *       restore occurs. The visit predicate is evaluated once per time, in order.
	 * @note Callbacks and allocation may throw; rollback restoration on exceptional
	 *       exit is not guaranteed.
	 * @complexity O(T + m log(T + 1)) setup, O(m log(T + 1)) insertions and Q visits for `m` items in
	 *             rollback mode, with the restore cost bounded by the inserts it
	 *             undoes; copy mode adds O(T · copy); memory O(T + m log(T + 1)),
	 *             excluding saved tokens and O(log(T + 1)) live structure copies.
	 *             When 0 < Q <= T / 10, O(T + m log(Q + 1)) setup and memory,
	 *             O(m log(Q + 1)) inserts, Q visits, O(Q) copies in copy mode.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
										   Insert insert, Visit visit, ShouldVisit should_visit = {},
										   Mode mode = default_restore<Structure>()) {
		atl_assert(time_count <= (std::size_t(1) << 30)); // keeps end + leaves <= 2^31 in the cover walk
		atl_assert(uint64(std::ranges::size(lifetimes)) < uint64((std::numeric_limits<uint32>::max)()));
		if (time_count == 0) return;
		uint32 const leaves = std::bit_ceil(uint32(time_count));

		// is_live[v]: v's subtree holds a visited time (padding leaves >= T never do)
		std::vector<uint8> is_live(2 * std::size_t(leaves), 0);
		uint32 query_count = 0;
		for (uint32 t = 0; t < uint32(time_count); ++t) {
			auto const is_visited = uint8(bool(std::invoke(should_visit, t)));
			is_live[leaves + t] = is_visited;
			query_count += is_visited;
		}
		if (query_count == 0) return;

		// Ranks cost an extra O(T) pass and buffer: only the measured sparse regime pays for them.
		if (query_count <= time_count / 10) {
			std::vector<uint32> rank(time_count + 1, 0), times;
			times.reserve(query_count);
			for (uint32 t = 0; t < uint32(time_count); ++t) {
				if (is_live[leaves + t]) times.push_back(t);
				rank[t + 1] = rank[t] + is_live[leaves + t];
			}
			auto const ranked = std::views::transform(std::ranges::ref_view(lifetimes), [&]<typename Lifetime>(Lifetime const &lifetime)
														  requires(detail::member_lifetime<Lifetime> || detail::pair_lifetime<Lifetime>)
													  {
														  auto const [begin, end] = detail::lifetime_bounds(lifetime);
														  atl_assert(begin <= end && end <= time_count);
														  return item_lifetime{rank[begin], rank[end]};
													  });
			uint32 const ranked_leaves = std::bit_ceil(query_count);
			is_live.assign(2 * std::size_t(ranked_leaves), 0);
			for (uint32 k = 0; k < query_count; ++k) is_live[ranked_leaves + k] = 1;
			auto original_visit = [&](Structure &s, uint32 k) { std::invoke(visit, s, times[k]); };
			detail::run_time_tree(structure, query_count, ranked, ranked_leaves, is_live, insert, original_visit, mode);
		} else {
			detail::run_time_tree(structure, time_count, lifetimes, leaves, is_live, insert, visit, mode);
		}
	}

	/**
	 * @brief @ref time_divide_and_conquer visiting every time with an explicit
	 *        restore mode, without spelling out @ref every_time.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes One half-open interval per item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback reading one time's represented state.
	 * @param mode State restoration policy.
	 * @pre The primary overload's bounds and callback laws apply.
	 * @complexity As the primary overload with Q == time_count.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, restore_mode<Structure> Mode>
	constexpr void time_divide_and_conquer(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
										   Insert insert, Visit visit, Mode mode) {
		time_divide_and_conquer(structure, time_count, lifetimes, std::move(insert), std::move(visit), every_time{},
								std::move(mode));
	}

	namespace detail {

		template<typename Event>
		inline constexpr bool is_item_event = false;

		template<typename Key>
		inline constexpr bool is_item_event<item_event<Key>> = true;
	}

	/**
	 * @brief A range of @ref item_event over a key @ref lifetimes_of can match:
	 *        hashable by @ref atl::hash, equality comparable, default constructible
	 *        and copyable (the hash-map key requirements), with no-throw hashing.
	 *        Const-bound event references must support kind comparisons and key lookup/insertion.
	 * @tparam R Const input range of item_event records.
	 */
	template<typename R>
	concept item_event_range =
		std::ranges::input_range<R const> && detail::is_item_event<std::ranges::range_value_t<R>>
		&& std::default_initializable<decltype(std::ranges::range_value_t<R>::key)>
		&& std::copyable<decltype(std::ranges::range_value_t<R>::key)>
		&& std::equality_comparable<decltype(std::ranges::range_value_t<R>::key)>
		&& detail::nothrow_hasher_for<hash<decltype(std::ranges::range_value_t<R>::key)>, decltype(std::ranges::range_value_t<R>::key)>
		&& requires(std::remove_reference_t<std::ranges::range_reference_t<R const>> const &event,
					flat_hash_map<decltype(std::ranges::range_value_t<R>::key), uint32> &map) {
			   bool(event.kind == event_kind::add);
			   bool(event.kind == event_kind::remove);
			   { map[event.key] } -> std::same_as<uint32 &>;
			   { map.find(event.key) } -> std::same_as<typename flat_hash_map<decltype(std::ranges::range_value_t<R>::key), uint32>::iterator>;
		   };

	/**
	 * @brief The item lifetimes behind an event log, ready for
	 *        @ref time_divide_and_conquer over `T = events.size()` times: one item
	 *        per add event, in log order. Time `t` is the state **after** event
	 *        `t`, so the item added by event `a` and removed by event `r` lives on
	 *        `[a, r)`, and one never removed on `[a, T)`. A remove closes the oldest
	 *        live copy of its key (the keys form a multiset). Since `begin` is the
	 *        add event's index, the item's key is `events[lifetime.begin].key`.
	 *
	 * Queries change nothing; they are the time points to pass to `should_visit`
	 * (`events[t].kind == event_kind::query`). A log with no query events works
	 * too: visiting every `t` then sees the state after each change.
	 *
	 * @tparam Events Const input range of item_event records.
	 * @param events Input log, iterated as const; query keys are ignored.
	 * @return One lifetime per add event, in add order.
	 * @pre every remove finds a live copy of its key; fewer than `2^32 - 1` events (asserted)
	 * @complexity O(T) expected (one hash lookup per add or remove); memory O(T)
	 * @note Not constexpr (seeded flat_hash_map).
	 */
	template<item_event_range Events>
	[[nodiscard]] std::vector<item_lifetime> lifetimes_of(Events const &events) {
		using key_type = decltype(std::ranges::range_value_t<Events>::key);
		constexpr uint32 none = (std::numeric_limits<uint32>::max)();
		std::vector<item_lifetime> lifetimes;
		// per key a FIFO of its live copies, linked through next_live; the map
		// stores slot + 1 so a fresh entry's 0 reads as "new key"
		flat_hash_map<key_type, uint32> slot_of;
		std::vector<uint32> oldest, newest, next_live;
		if constexpr (std::ranges::sized_range<Events const>) {
			atl_assert(uint64(std::ranges::size(events)) < uint64(none));
			lifetimes.reserve(std::ranges::size(events));
			next_live.reserve(std::ranges::size(events));
		}
		uint32 t = 0;
		for (auto const &event : events) {
			atl_assert(t < none - 1);
			if (event.kind == event_kind::add) {
				auto const item = uint32(lifetimes.size());
				lifetimes.push_back({t, none});
				next_live.push_back(none);
				uint32 &slot = slot_of[event.key];
				if (slot == 0) {
					oldest.push_back(item);
					newest.push_back(item);
					slot = uint32(oldest.size());
				} else {
					uint32 const k = slot - 1;
					if (oldest[k] == none)
						oldest[k] = item;
					else
						next_live[newest[k]] = item;
					newest[k] = item;
				}
			} else if (event.kind == event_kind::remove) {
				auto const it = slot_of.find(event.key);
				atl_assert(it != slot_of.end());
				uint32 const k = it->second - 1;
				uint32 const item = oldest[k];
				atl_assert(item != none); // removing a key with no live copy
				lifetimes[item].end = t;
				oldest[k] = next_live[item];
			}
			++t;
		}
		for (auto &lifetime : lifetimes)
			if (lifetime.end == none) lifetime.end = t;
		return lifetimes;
	}
}
