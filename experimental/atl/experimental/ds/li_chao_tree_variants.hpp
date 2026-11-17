#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::li_chao_tree (the array Li Chao tree).
 *
 * - @ref atl::experimental::li_chao_tree_by_inline_flags, each slot carries its own `is_used` flag next
 *   to the line (`2 * sizeof(Value) + 1` bytes, padded to 24 for `int64` on the measured x86-64 ABI),
 *   so a query reads every slot on its leaf-to-root path, empty or not. The
 *   library tree keeps the used flags in a separate bitset (one bit per
 *   slot, small enough to stay cached) and touches a slot only when it holds
 *   a line.
 *
 * - @ref atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search, a copy of the library tree with the insertion
 *   descent carrying the range's end coordinates and values down the
 *   levels, and with the coordinate-list search replaced by a branchless
 *   binary search or an Eytzinger-layout one.
 *   The benchmark verdicts are on the class below.
 *
 * Race verdict (benchmarks/ds/li_chao_tree.cpp, measurements of
 * 2026-10-23, Xeon Platinum 8280, ms, plain -O2 / -march=native; 2e5
 * inserts + 2e5 mixed operations, int64 lines; both trees with padding
 * extension and endpoint pruning):
 *
 * | shape | bitset (library) | flag in slot |
 * |-----------------------------------------------------|------------------|--------------|
 * | full-range segments over query coordinates, 1e9 (*) | 9.0 / 9.9 | 8.7 / 9.6 |
 * | segments over query coordinates, 1e9 | 203 / 203 | 212 / 221 |
 * | full-range segments over the range [0, 1e6) (*) | 29.9 / 29.8 | 40.3 / 38.8 |
 * | segments over the range [0, 1e6) | 206 / 207 | 246 / 250 |
 *
 * (*) every query at the domain's first point. Lost: within 1.1x on the
 * coordinate shapes, 1.2-1.35x slower over the dense range, and 24-byte
 * instead of 16-byte slots for int64 lines.
 */

#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief @ref atl::li_chao_tree with the used flag stored in each slot
	 *        next to the function instead of in a separate bitset; same
	 *        interface and semantics.
	 *
	 * Lines `y = slope * x + intercept` -- or members of any function
	 *        family in which two members cross at most once -- and segments
	 *        (a function on `[l, r)` only), over a domain of query points
	 *        fixed up front: every integer of `[lo, hi)`, or a sorted list of
	 *        coordinates. Keeps the best value at each point (the Li Chao
	 *        tree).
	 *
	 * A perfect binary tree over the domain's positions; each node holds at
	 * most one function (with its used flag beside it), the
	 * best one at the node's midpoint among those that reached it, and a query
	 * at a point takes the best of the functions on its leaf-to-root path.
	 * Inserting a function descends one root-to-leaf path, pushing the loser
	 * at each midpoint into the one half where it can still win -- which is
	 * where the crossing-at-most-once property is used; a function that wins
	 * (or loses) at both ends of a node's range wins (or loses) on all of it,
	 * so the loser is dropped there (the endpoint pruning). A segment is split
	 * over the O(log n) canonical nodes covering its positions (a segment
	 * reaching the last point also covers the padding positions past it),
	 * each insertion then descending from there.
	 *
	 * @tparam Coordinate The domain's point type (`int64`; any totally
	 *         ordered type for the coordinate-list domain, `double` say).
	 * @tparam Compare `comp(a, b)` reads "a is better than b" on the values:
	 *         std::less<> for the minimum (lower envelope), std::greater<>
	 *         for the maximum.
	 * @tparam Function The family's member type (@ref atl::envelope_function),
	 *         `linear_function<Coordinate>` by default, which also gets the
	 *         `insert(slope, intercept)` shorthands and an exact overflow
	 *         check. The value type is what `f(x)` returns.
	 * @pre any two inserted functions cross at most once over the domain:
	 *      `f(x)` better than `g(x)` on a prefix or a suffix of the points
	 *      (not checkable; documented, as for every non-line family). Every
	 *      evaluation `f(x)` at a covered domain point is valid, including all intermediates (for a line
	 *      over integers this is asserted at insertion).
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
		requires std::totally_ordered<Coordinate> && std::copyable<Coordinate>
			  && envelope_function<Function, Coordinate, Compare>
	class li_chao_tree_by_inline_flags {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr li_chao_tree_by_inline_flags() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions yet.
		 * @pre `lo <= hi` and `hi - lo` representable in Coordinate and less than `2^31` (asserted); memory is about
		 *      `2 * bit_ceil(hi - lo)` slots of `sizeof(Function) + 1` bytes,
		 *      padded.
		 * @complexity O(hi - lo)
		 */
		constexpr li_chao_tree_by_inline_flags(Coordinate lo, Coordinate hi, Compare comp = {})
			requires integral<Coordinate>
			: comp_(std::move(comp)), lo_(lo) {
			allocate(atl::detail::li_chao_width(lo, hi));
		}

		/**
		 * @brief The points of `coordinates`, the only ones a query may ask
		 *        and a segment's ends are matched against; no functions yet.
		 * @pre `coordinates` strictly increasing (asserted) and fewer than
		 *      `2^31`; the padded slot count must fit size_t (asserted).
		 * @complexity O(n)
		 */
		constexpr explicit li_chao_tree_by_inline_flags(std::vector<Coordinate> coordinates, Compare comp = {})
			: comp_(std::move(comp)), coordinates_(std::move(coordinates)), has_coordinates_(true) {
			atl_assert(coordinates_.size() <= std::size_t((std::numeric_limits<int32>::max)()));
			atl_assert(std::ranges::adjacent_find(coordinates_, std::ranges::greater_equal{}) == coordinates_.end());
			allocate(coordinates_.size());
		}

		/** @brief The number of query points. */
		[[nodiscard]] constexpr std::size_t size() const { return size_; }

		/** @brief Whether the domain has no points. */
		[[nodiscard]] constexpr bool empty() const { return size_ == 0; }

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log n)
		 */
		constexpr void insert(Function f) {
			if (size_ == 0) return;
			atl_assert(atl::detail::evaluates_on(f, coordinate(0), coordinate(size_ - 1)));
			descend(1, 0, leaves_, std::move(f));
		}

		/**
		 * @brief Add `f` on the domain points `x` with `l <= x < r` only (a
		 *        segment); an empty range adds nothing.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Function const &f, Coordinate const &l, Coordinate const &r) {
			uint32 const begin = position_at_least(l), end = position_at_least(r);
			if (begin >= end) return;
			atl_assert(atl::detail::evaluates_on(f, coordinate(begin), coordinate(end - 1)));
			// a range reaching the last point also takes the padding positions,
			// as in atl::li_chao_tree
			uint64 first = uint64(begin) + leaves_, last = uint64(end == size_ ? leaves_ : end) + leaves_;
			// at the level of `width`-leaf subtrees a node covers the positions
			// from (node - leaves / width) * width
			uint64 width = 1;
			auto const settle = [&](uint64 node) {
				descend(uint32(node), uint32((node - leaves_ / width) * width), uint32(width), f);
			};
			for (; first < last; first >>= 1, last >>= 1, width <<= 1) {
				if (first & 1) settle(first++);
				if (last & 1) settle(--last);
			}
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its product and sum at every domain point fit (asserted at the two
		 *      ends for an integral coordinate)
		 * @complexity O(log n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate const &l, Coordinate const &r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, l, r);
		}

		/**
		 * @brief The best value at `x` over the functions and segments
		 *        covering it, or none when nothing covers it.
		 * @pre `x` is a domain point (asserted)
		 * @complexity O(log n) (plus a binary search over the coordinates)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			uint32 const position = position_of(x);
			std::optional<value_type> best;
			for (uint32 node = position + leaves_; node > 0; node >>= 1) {
				if (!slots_[node].is_used) continue;
				value_type value = slots_[node].held(x);
				if (!best || comp_(value, *best)) best = std::move(value);
			}
			return best;
		}

	private:
		/** @brief A tree node: a function, unless `is_used` is false. */
		struct slot {
			Function held{};
			bool is_used = false;
		};

		/**
		 * @brief Size the tree for `points` positions, all slots empty.
		 * @pre `points < 2^31` (asserted by both constructors), so the
		 *      positions and the `2 * bit_ceil(points)` slot ids fit `uint32`.
		 */
		constexpr void allocate(std::size_t points) {
			atl_assert(points <= std::size_t((std::numeric_limits<int32>::max)()));
			size_ = uint32(points);
			leaves_ = std::bit_ceil((std::max<uint32>)(size_, 1));
			atl_assert(std::size_t(leaves_) <= (std::numeric_limits<std::size_t>::max)() / 2);
			slots_.assign(2 * std::size_t(leaves_), slot{});
		}

		/** @brief The coordinate of position `i`; padding positions repeat the last point. */
		[[nodiscard]] constexpr Coordinate coordinate(uint32 i) const {
			i = (std::min)(i, size_ - 1);
			if constexpr (integral<Coordinate>)
				return has_coordinates_ ? coordinates_[i] : Coordinate(lo_ + Coordinate(i));
			else
				return coordinates_[i];
		}

		/** @brief The position of the domain point `x` (asserted to be one). */
		[[nodiscard]] constexpr uint32 position_of(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					atl_assert(!(x < lo_) && x < Coordinate(lo_ + Coordinate(size_)));
					return uint32(x - lo_);
				}
			}
			auto const it = std::ranges::lower_bound(coordinates_, x);
			atl_assert(it != coordinates_.end() && !(x < *it));
			return uint32(it - coordinates_.begin());
		}

		/** @brief The first position whose coordinate is at least `x` (`size()` if none). */
		[[nodiscard]] constexpr uint32 position_at_least(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					if (!(lo_ < x)) return 0;
					// lo + size <= hi fits, x - lo may not
					if (!(x < Coordinate(lo_ + Coordinate(size_)))) return size_;
					return uint32(x - lo_);
				}
			}
			return uint32(std::ranges::lower_bound(coordinates_, x) - coordinates_.begin());
		}

		/**
		 * @brief Settle `added` into the subtree of `node`, which covers the
		 *        `width` positions from `first`: keep the better function at
		 *        each midpoint and send the other into the half where it can
		 *        still win.
		 */
		constexpr void descend(uint32 node, uint32 first, uint32 width, Function added) {
			while (true) {
				slot &cell = slots_[node];
				Function &here = cell.held;
				if (!cell.is_used) {
					here = std::move(added);
					cell.is_used = true;
					return;
				}
				uint32 const middle = first + width / 2;
				Coordinate const x_middle = coordinate(middle), x_first = coordinate(first);
				bool const wins_middle = comp_(added(x_middle), here(x_middle));
				bool const wins_first = comp_(added(x_first), here(x_first));
				// winning (or losing) at both ends: on all of the range, as in atl::li_chao_tree
				if (wins_first == wins_middle && width > 1) {
					Coordinate const x_last = coordinate(first + width - 1);
					if (comp_(added(x_last), here(x_last)) == wins_middle) {
						if (wins_middle) here = std::move(added);
						return;
					}
				}
				if (wins_middle) std::swap(here, added);
				if (width == 1) return;
				width /= 2;
				// the function now in `added` loses at the midpoint; crossing at
				// most once, it can win only left of it if it wins at the first
				// point, else only right of it
				if (wins_first != wins_middle) {
					node = 2 * node;
				} else {
					node = 2 * node + 1;
					first = middle;
				}
			}
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<Coordinate> coordinates_;
		std::vector<slot> slots_;
		Coordinate lo_{};
		uint32 size_ = 0, leaves_ = 0;
		bool has_coordinates_ = false;
	};

	/** @brief How @ref atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search finds the position of a coordinate in its sorted list. */
	enum class coordinate_search {
		/** @brief `std::ranges::lower_bound` (the library search). */
		standard,
		/** @brief A branchless binary search: the step is a conditional move, no mispredicted branch per level. */
		branchless,
		/** @brief Binary search over a BFS-order (Eytzinger) copy: the first levels share cache lines and prefetch, and the walk is branchless. */
		eytzinger
	};

	/**
	 * @brief An experimental copy of @ref atl::li_chao_tree with two tuning
	 *        knobs. Neither has a consistent advantage over the library.
	 *
	 * Record ds-li_chao_tree, 2026-11-01, Xeon Platinum 8280, plain -O2 /
	 * -march=native, ms, int64, 2e5 inserts then 2e5 mixed operations.
	 * Carrying endpoints has no win: random segments 200 / 206 against
	 * 196 / 197 (+2% / +5%). Branchless search has no win on lines:
	 * 27.5 / 27.7 against 27.6 / 27.3; full-range segments 44.1 / 43.9
	 * against 42.0 / 42.0 (+5% / +5%). Eytzinger loses full-range segments:
	 * 61.4 / 60.8 against 42.0 / 42.0 (+46% / +45%). Kept as losers.
	 *
	 * - `CarriesEndpoints`: the insertion descent carries the range's end
	 *   coordinates and the inserted function's values there down the levels
	 *   (a descent to the left needs one fresh coordinate and one fresh
	 *   value, to the right none: the new first point is the old midpoint)
	 *   instead of reloading `coordinate(first)`, `coordinate(last)` and
	 *   re-evaluating the inserted function at both ends at every level. Off:
	 *   the library descent.
	 * - `Search`: the strategy @ref atl::experimental::coordinate_search for `position_of` and
	 *   `position_at_least` over a coordinate list (the dense `[lo, hi)`
	 *   domain does no search). Every query and every segment end pays one
	 *   search, which is where the coordinate-mode shapes spend their time
	 *   beyond the dense ones.
	 *
	 * `CarriesEndpoints = false` with `coordinate_search::standard` is the
	 * library tree; the interface is identical.
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>,
			 bool CarriesEndpoints = true, coordinate_search Search = coordinate_search::eytzinger>
		requires std::totally_ordered<Coordinate> && std::copyable<Coordinate>
			  && envelope_function<Function, Coordinate, Compare>
	class li_chao_tree_by_endpoint_carry_and_coordinate_search {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr li_chao_tree_by_endpoint_carry_and_coordinate_search() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions yet.
		 * @pre `lo <= hi` and `hi - lo` representable in Coordinate and less than `2^31` (asserted); memory is about
		 *      `2 * bit_ceil(hi - lo)` slots of `sizeof(Function)` bytes and
		 *      one bit.
		 * @complexity O(hi - lo)
		 */
		constexpr li_chao_tree_by_endpoint_carry_and_coordinate_search(Coordinate lo, Coordinate hi, Compare comp = {})
			requires integral<Coordinate>
			: comp_(std::move(comp)), lo_(lo) {
			allocate(atl::detail::li_chao_width(lo, hi));
		}

		/**
		 * @brief The points of `coordinates`, the only ones a query may ask
		 *        and a segment's ends are matched against; no functions yet.
		 * @pre `coordinates` strictly increasing (asserted) and fewer than
		 *      `2^31`; the padded slot count must fit size_t (asserted).
		 * @complexity O(n)
		 */
		constexpr explicit li_chao_tree_by_endpoint_carry_and_coordinate_search(std::vector<Coordinate> coordinates, Compare comp = {})
			: comp_(std::move(comp)), coordinates_(std::move(coordinates)), has_coordinates_(true) {
			atl_assert(coordinates_.size() <= std::size_t((std::numeric_limits<int32>::max)()));
			atl_assert(std::ranges::adjacent_find(coordinates_, std::ranges::greater_equal{}) == coordinates_.end());
			allocate(coordinates_.size());
			if constexpr (Search == coordinate_search::eytzinger) build_eytzinger();
		}

		/** @brief The number of query points. */
		constexpr std::size_t size() const { return size_; }

		/** @brief Whether the domain has no points. */
		constexpr bool empty() const { return size_ == 0; }

		/**
		 * @brief Add `f` over the whole domain.
		 * @complexity O(log n)
		 */
		constexpr void insert(Function f) {
			if (size_ == 0) return;
			atl_assert(atl::detail::evaluates_on(f, coordinate(0), coordinate(size_ - 1)));
			descend(1, 0, leaves_, std::move(f));
		}

		/**
		 * @brief Add `f` on the domain points `x` with `l <= x < r` only (a
		 *        segment); an empty range adds nothing.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Function const &f, Coordinate const &l, Coordinate const &r) {
			uint32 const begin = position_at_least(l), end = position_at_least(r);
			if (begin >= end) return;
			atl_assert(atl::detail::evaluates_on(f, coordinate(begin), coordinate(end - 1)));
			// a range reaching the last point also takes the padding positions:
			// they repeat the last point and no query asks there, and the range
			// then splits into one node per level on its left side only (a
			// whole-domain range into the root alone)
			uint64 first = uint64(begin) + leaves_, last = uint64(end == size_ ? leaves_ : end) + leaves_;
			// at the level of `width`-leaf subtrees a node covers the positions
			// from (node - leaves / width) * width
			uint64 width = 1;
			auto const settle = [&](uint64 node) {
				descend(uint32(node), uint32((node - leaves_ / width) * width), uint32(width), f);
			};
			for (; first < last; first >>= 1, last >>= 1, width <<= 1) {
				if (first & 1) settle(first++);
				if (last & 1) settle(--last);
			}
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its product and sum at every domain point fit (asserted at the two
		 *      ends for an integral coordinate)
		 * @complexity O(log n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate const &l, Coordinate const &r)
			requires atl::detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, l, r);
		}

		/**
		 * @brief The best value at `x` over the functions and segments
		 *        covering it, or none when nothing covers it.
		 * @pre `x` is a domain point (asserted)
		 * @complexity O(log n) (plus a binary search over the coordinates)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			uint32 const position = position_of(x);
			std::optional<value_type> best;
			for (uint32 node = position + leaves_; node > 0; node >>= 1) {
				if (!is_used(node)) continue;
				value_type value = functions_[node](x);
				if (!best || comp_(value, *best)) best = std::move(value);
			}
			return best;
		}

	private:
		[[nodiscard]] constexpr bool is_used(uint32 node) const { return used_[node >> 6] >> (node & 63) & 1; }

		/**
		 * @brief Size the tree for `points` positions, all slots empty.
		 * @pre `points < 2^31` (asserted by both constructors), so the
		 *      positions and the `2 * bit_ceil(points)` slot ids fit `uint32`.
		 */
		constexpr void allocate(std::size_t points) {
			atl_assert(points <= std::size_t((std::numeric_limits<int32>::max)()));
			size_ = uint32(points);
			leaves_ = std::bit_ceil((std::max<uint32>)(size_, 1));
			atl_assert(std::size_t(leaves_) <= (std::numeric_limits<std::size_t>::max)() / 2);
			functions_.assign(2 * std::size_t(leaves_), Function{});
			used_.assign((2 * std::size_t(leaves_)) / 64 + ((2 * std::size_t(leaves_)) % 64 != 0), 0);
		}

		/** @brief The coordinate of position `i`; padding positions repeat the last point. */
		[[nodiscard]] constexpr Coordinate coordinate(uint32 i) const {
			i = (std::min)(i, size_ - 1);
			if constexpr (integral<Coordinate>)
				return has_coordinates_ ? coordinates_[i] : Coordinate(lo_ + Coordinate(i));
			else
				return coordinates_[i];
		}

		/** @brief The position of the domain point `x` (asserted to be one). */
		[[nodiscard]] constexpr uint32 position_of(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					atl_assert(!(x < lo_) && x < Coordinate(lo_ + Coordinate(size_)));
					return uint32(x - lo_);
				}
			}
			uint32 const position = lower_position(x);
			atl_assert(position < size_ && !(x < coordinates_[position]));
			return position;
		}

		/** @brief The first position whose coordinate is at least `x` (`size()` if none). */
		[[nodiscard]] constexpr uint32 position_at_least(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					if (!(lo_ < x)) return 0;
					// lo + size <= hi fits, x - lo may not
					if (!(x < Coordinate(lo_ + Coordinate(size_)))) return size_;
					return uint32(x - lo_);
				}
			}
			return lower_position(x);
		}

		/** @brief The first position whose coordinate is at least `x` in the coordinate list (`size()` if none), by the `Search` strategy. */
		[[nodiscard]] constexpr uint32 lower_position(Coordinate const &x) const {
			if constexpr (Search == coordinate_search::standard) {
				return uint32(std::ranges::lower_bound(coordinates_, x) - coordinates_.begin());
			} else if constexpr (Search == coordinate_search::branchless) {
				if (size_ == 0) return 0;
				Coordinate const *base = coordinates_.data();
				for (uint32 count = size_; count > 1;) {
					uint32 const half = count / 2;
					base = (base[half - 1] < x) ? base + half : base;
					count -= half;
				}
				return uint32(base - coordinates_.data()) + uint32(*base < x);
			} else {
				// the BFS-order search: the tree index k descends to the child 2k + (a[k] < x); the answer
				// is the last node where the walk went left, recovered by stripping the trailing right turns
				uint32 k = 1;
				while (k <= size_) k = 2 * k + uint32(eytzinger_[k] < x);
				k >>= std::countr_one(k) + 1;
				return k == 0 ? size_ : eytzinger_rank_[k];
			}
		}

		/** @brief Fill the BFS-order copy of the coordinates and each node's sorted position. */
		constexpr void build_eytzinger() {
			eytzinger_.assign(std::size_t(size_) + 1, Coordinate{});
			eytzinger_rank_.assign(std::size_t(size_) + 1, 0);
			uint32 next = 0;
			// in-order walk of the implicit tree numbers the nodes by sorted position
			auto const fill = [&](auto const &self, uint32 k) -> void {
				if (k > size_) return;
				self(self, 2 * k);
				eytzinger_[k] = coordinates_[next];
				eytzinger_rank_[k] = next++;
				self(self, 2 * k + 1);
			};
			fill(fill, 1);
		}

		/**
		 * @brief Settle `added` into the subtree of `node`, which covers the
		 *        `width` positions from `first`: keep the better function at
		 *        each midpoint and send the other into the half where it can
		 *        still win.
		 */
		constexpr void descend(uint32 node, uint32 first, uint32 width, Function added) {
			if constexpr (!CarriesEndpoints) {
				descend_reloading(node, first, width, std::move(added));
				return;
			}
			// the range's end points, `added`'s values there: carried down, not reloaded per level
			Coordinate x_first{}, x_last{};
			value_type added_first{}, added_last{};
			bool has_ends = false;
			while (true) {
				Function &here = functions_[node];
				if (!is_used(node)) {
					here = std::move(added);
					used_[node >> 6] |= uint64(1) << (node & 63);
					return;
				}
				if (!has_ends) {
					x_first = coordinate(first);
					x_last = coordinate(first + width - 1);
					added_first = added(x_first);
					added_last = added(x_last);
					has_ends = true;
				}
				uint32 const middle = first + width / 2;
				Coordinate const x_middle = coordinate(middle);
				value_type here_middle = here(x_middle), added_middle = added(x_middle);
				value_type here_first = here(x_first);
				bool const wins_middle = comp_(added_middle, here_middle);
				bool const wins_first = comp_(added_first, here_first);
				value_type here_last{};
				if (wins_first == wins_middle && width > 1) {
					here_last = here(x_last);
					if (comp_(added_last, here_last) == wins_middle) {
						if (wins_middle) here = std::move(added);
						return;
					}
				}
				if (wins_middle) {
					std::swap(here, added);
					std::swap(added_middle, here_middle);
					std::swap(added_first, here_first);
					std::swap(added_last, here_last);
				}
				if (width == 1) return;
				width /= 2;
				if (wins_first != wins_middle) {
					node = 2 * node;
					x_last = coordinate(middle - 1);
					added_last = added(x_last);
				} else {
					node = 2 * node + 1;
					first = middle;
					x_first = x_middle;
					added_first = std::move(added_middle);
				}
			}
		}

		/** @brief The library descent, reloading the coordinates and the loser's end values at every level. */
		constexpr void descend_reloading(uint32 node, uint32 first, uint32 width, Function added) {
			while (true) {
				Function &here = functions_[node];
				if (!is_used(node)) {
					here = std::move(added);
					used_[node >> 6] |= uint64(1) << (node & 63);
					return;
				}
				uint32 const middle = first + width / 2;
				Coordinate const x_middle = coordinate(middle), x_first = coordinate(first);
				bool const wins_middle = comp_(added(x_middle), here(x_middle));
				bool const wins_first = comp_(added(x_first), here(x_first));
				if (wins_first == wins_middle && width > 1) {
					Coordinate const x_last = coordinate(first + width - 1);
					if (comp_(added(x_last), here(x_last)) == wins_middle) {
						if (wins_middle) here = std::move(added);
						return;
					}
				}
				if (wins_middle) std::swap(here, added);
				if (width == 1) return;
				width /= 2;
				if (wins_first != wins_middle) {
					node = 2 * node;
				} else {
					node = 2 * node + 1;
					first = middle;
				}
			}
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<Coordinate> coordinates_;
		// the BFS-order copy (1-based) and each node's position in `coordinates_` (the Eytzinger search only)
		std::vector<Coordinate> eytzinger_;
		std::vector<uint32> eytzinger_rank_;
		std::vector<Function> functions_;
		// one bit per slot, kept apart from the functions so a query skips the
		// empty slots on its path without touching their (uncached) memory
		std::vector<uint64> used_;
		Coordinate lo_{};
		uint32 size_ = 0, leaves_ = 0;
		bool has_coordinates_ = false;
	};
}
