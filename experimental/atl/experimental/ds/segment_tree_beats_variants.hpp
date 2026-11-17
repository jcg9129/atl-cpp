#pragma once

/**
 * @file
 * @brief Race losers for @ref atl::beats_segment_tree (the chmin / chmax / add /
 *        assign + sum / min / max class).
 *
 * The contenders, all driven through one template by the benchmark and the oracle
 * tests:
 *
 * - generic engine, @ref atl::segment_tree_beats over @ref atl::beats::extremes and
 *   @ref atl::beats::clamp_add: the generic engine (boundary pushes and the canonical
 *   decomposition of the shared iterative base, a recursive descent below failing
 *   canonical nodes, identity-tag push skip, collapse fast path in the mapping). It
 *   stays public for the generic members and the other library sets.
 * - library, @ref atl::beats_segment_tree: the library implementation. Only an add is pending; a
 *   push clamps a child to the parent's maximum and minimum (or flattens it under a
 *   single-valued parent).
 * - by compact recursion @ref beats_segment_tree_by_recursive_compact_nodes: classic top-down recursion over `[0, n)`, `2n - 1`
 *   nodes (no power-of-two padding), one recursion locating and beating.
 * - by explicit stack @ref beats_segment_tree_by_explicit_stack: generic engine with an explicit-stack descent instead of recursion.
 * - by fixed clamp-add @ref beats_segment_tree_by_fixed_clamp_add: hand-specialized class, clamp-add tag, one
 *   descent per operation with fused compares, no `bool` protocol.
 * - by three adds @ref beats_segment_tree_by_three_adds: class-specific three adds (to the maximum's class, the
 *   minimum's class, the rest) as the pending tag.
 * - by tagless layout: @ref beats_segment_tree_by_tagless_layout, the library's algorithm in the other two layouts of
 *   @ref beats_layout (AoS aligned to a 64 B line, SoA arrays).
 * - by optional mapping @ref beats_segment_tree_by_optional_mapping: the engine with a mapping that returns
 *   `std::optional<Value>` instead of modifying in place.
 * - by push or collapse policy @ref beats_segment_tree_by_unconditional_push (no identity-tag skip on push) and
 *   @ref beats_segment_tree_by_no_collapse (flattens only single-valued nodes, so a clamp past a
 *   whole node descends). The engine keeps both the skip and the collapse.
 *
 * Verdict (x86-64 measurements using benchmarks/ds/segment_tree_beats.cpp,
 * 2026-11-01, Xeon
 * 8280, `-O2` and `-march=native`; five shapes at n = q = 2e5 and 5e5): library beat generic engine
 * beyond the spread on all ten rows under both flags, by 18% to 46%. Percentages below
 * are a contender's minimum against the named reference on the same row and flags;
 * the shapes are balanced mix, chmin-only (no add), add-recharge, collapse and query-heavy.
 * The guide uses method names for all contenders.
 */

#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/ds/segment_tree_beats.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/** @brief How @ref segment_tree_beats_by_policy descends below a node whose mapping failed. */
	enum class beats_descent {
		recursive,		///< a recursive member function (generic engine, the engine)
		explicit_stack, ///< a loop over a fixed-size stack of pending visits (by explicit stack)
	};

	/**
	 * @brief A fallible mapping in the copying protocol of by optional mapping: `map(f, s)` returns
	 *        the new summary, or `std::nullopt` to descend; `s` is left untouched.
	 */
	template<typename Mapping, typename Tag, typename Value>
	concept copying_fallible_action = std::regular_invocable<Mapping const &, Tag const &, Value const &>
								   && std::same_as<std::invoke_result_t<Mapping const &, Tag const &, Value const &>, std::optional<Value>>;

	/** @brief The library clamp-add mapping in the copying protocol (by optional mapping). */
	struct copying_clamp_add_mapping {
		/** @brief The mapped copy of @p s, or `std::nullopt` where the library mapping fails. */
		template<signed_integral Element, signed_integral Sum>
		[[nodiscard]] constexpr std::optional<beats::extremes<Element, Sum>> operator()(beats::clamp_add<Element> const &f,
																						beats::extremes<Element, Sum> const &s) const noexcept {
			beats::extremes<Element, Sum> t = s;
			if (beats::mapping{}(f, t)) return t;
			return std::nullopt;
		}
	};

	/**
	 * @brief The clamp-add mapping without the collapse fast path (push policy): a clamp that
	 *        would flatten a node holding several values fails instead, so it descends
	 *        until the nodes are single-valued. Satisfies the push law: a node only
	 *        absorbs one-class moves or is single-valued, and so are its children.
	 */
	struct uncollapsed_clamp_add_mapping {
		/** @brief Absorb a clamp only when it moves one extreme class or the node is single-valued. */
		template<signed_integral Element, signed_integral Sum>
		constexpr bool operator()(beats::clamp_add<Element> const &f, beats::extremes<Element, Sum> &s) const noexcept {
			atl_assert(f.lower <= f.upper);
			if (s.width == 0) return true;
			if (s.max == s.min) {
				Element const v = (std::min)((std::max)(atl::detail::beats_shift_value(s.max, f.add), f.lower), f.upper);
				s.max = s.min = v;
				s.sum = atl::detail::beats_mul(Sum(v), Sum(s.width));
				return true;
			}
			if (f.add != 0) {
				s.max = atl::detail::beats_shift_value(s.max, f.add);
				s.min = atl::detail::beats_shift_value(s.min, f.add);
				if (s.second_max != (std::numeric_limits<Element>::lowest)()) s.second_max = atl::detail::beats_shift_value(s.second_max, f.add);
				if (s.second_min != (std::numeric_limits<Element>::max)()) s.second_min = atl::detail::beats_shift_value(s.second_min, f.add);
				s.sum = atl::detail::beats_add(s.sum, atl::detail::beats_mul(Sum(f.add), Sum(s.width)));
			}
			if (f.upper <= s.min || f.lower >= s.max || f.lower == f.upper) return false;
			if (f.lower > s.min) {
				if (f.lower >= s.second_min) return false;
				s.sum = atl::detail::beats_add(s.sum, atl::detail::beats_mul(atl::detail::beats_sub(Sum(f.lower), Sum(s.min)), Sum(s.min_count)));
				if (s.second_max == s.min) s.second_max = f.lower;
				s.min = f.lower;
			}
			if (f.upper < s.max) {
				if (f.upper <= s.second_max) return false;
				s.sum = atl::detail::beats_sub(s.sum, atl::detail::beats_mul(atl::detail::beats_sub(Sum(s.max), Sum(f.upper)), Sum(s.max_count)));
				if (s.second_min == s.max) s.second_min = f.upper;
				s.max = f.upper;
			}
			return true;
		}
	};

	/**
	 * @brief The engine's algorithm specialized to the clamp-add algebra, with
	 *        the race knobs: the descent (by explicit stack), the mapping protocol (by optional mapping: a
	 *        @ref copying_fallible_action), the mapping itself (collapse) and the
	 *        identity-tag push skip (push policy).
	 *
	 * @tparam Element The element type.
	 * @tparam Sum The sum type.
	 * @tparam Mapping An in-place @ref atl::fallible_action or a
	 *                            @ref copying_fallible_action on the preset algebra.
	 * @tparam Descent The descent below failing nodes.
	 * @tparam IsSkippingIdentity Whether a push returns early on an identity tag.
	 */
	template<signed_integral Element, signed_integral Sum, typename Mapping, beats_descent Descent, bool IsSkippingIdentity>
		requires fallible_action<Mapping, beats::clamp_add<Element>, beats::extremes<Element, Sum>>
			  || copying_fallible_action<Mapping, beats::clamp_add<Element>, beats::extremes<Element, Sum>>
	class segment_tree_beats_by_policy : atl::detail::segment_tree_base<beats::extremes<Element, Sum>> {
		using value = beats::extremes<Element, Sum>;
		using tag = beats::clamp_add<Element>;
		using base = atl::detail::segment_tree_base<value>;
		using base::ascend_boundaries;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::size_;

		[[no_unique_address]] Mapping mapping_;
		std::vector<tag> pending_tags_;

		/** @brief Map @p f onto @p s in either protocol; false when it must descend. */
		constexpr bool map(tag const &f, value &s) const {
			if constexpr (fallible_action<Mapping, tag, value>) {
				return mapping_(f, s);
			} else {
				std::optional<value> mapped = mapping_(f, s);
				if (!mapped) return false;
				s = *mapped;
				return true;
			}
		}

		/** @brief Recompute node @p k. */
		constexpr void pull(std::size_t k) { nodes_[k] = beats::merge{}(nodes_[2 * k], nodes_[2 * k + 1]); }

		/** @brief Hand a pushed tag to child @p c. */
		constexpr void settle(std::size_t c, tag const &f) {
			bool const absorbed = map(f, nodes_[c]);
			atl_assert(absorbed);
			if (c < size_) pending_tags_[c] = beats::compose{}(f, pending_tags_[c]);
		}

		/** @brief Settle node @p k's pending tag onto its children. */
		constexpr void push_down(std::size_t k) {
			if constexpr (IsSkippingIdentity)
				if (pending_tags_[k] == tag{}) return;
			settle(2 * k, pending_tags_[k]);
			settle(2 * k + 1, pending_tags_[k]);
			pending_tags_[k] = tag{};
		}

		/** @brief Settle every pending tag above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief Apply @p f below node @p k, descending where the mapping fails. */
		constexpr void beat(std::size_t k, tag const &f) {
			if constexpr (Descent == beats_descent::recursive) {
				if (map(f, nodes_[k])) {
					if (k < size_) pending_tags_[k] = beats::compose{}(f, pending_tags_[k]);
					return;
				}
				atl_assert(k < size_);
				push_down(k);
				beat(2 * k, f);
				beat(2 * k + 1, f);
				pull(k);
			} else {
				// entries are 2k (visit node k) or 2k + 1 (pull node k after its children);
				// each level leaves at most two entries behind, so 2 * 64 + 2 slots suffice
				std::size_t stack[2 * 64 + 2];
				std::size_t top = 0;
				stack[top++] = 2 * k;
				while (top > 0) {
					std::size_t const entry = stack[--top], node = entry >> 1;
					if (entry & 1) {
						pull(node);
						continue;
					}
					if (map(f, nodes_[node])) {
						if (node < size_) pending_tags_[node] = beats::compose{}(f, pending_tags_[node]);
						continue;
					}
					atl_assert(node < size_);
					push_down(node);
					stack[top++] = 2 * node + 1;
					stack[top++] = 2 * (2 * node + 1);
					stack[top++] = 2 * (2 * node);
				}
			}
		}

		/** @brief Push the boundary paths of `[l, r)`, then visit its canonical nodes. */
		template<std::invocable<std::size_t> Visit>
		constexpr void visit_canonical(std::size_t l, std::size_t r, Visit visit) {
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, visit, visit);
		}

		/** @brief Apply @p f to `[l, r)`. */
		constexpr void apply(std::size_t l, std::size_t r, tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, [&](std::size_t k) { beat(k, f); }, [&](std::size_t k) { beat(k, f); });
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

	public:
		/** @brief From initial elements. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
		constexpr explicit segment_tree_beats_by_policy(R &&elements) {
			atl_assert(std::ranges::size(elements) < (atl::uint64(1) << 32) - 1);
			base::reset(std::ranges::size(elements), value{});
			pending_tags_.assign(size_, tag{});
			std::size_t i = 0;
			for (auto &&x : elements) nodes_[leaf(i++)] = value::of(static_cast<Element>(atl::detail::forward_element<R>(x)));
			this->build(beats::merge{});
		}

		/** @brief `a[i] = (min)(a[i], x)` on `[l, r)`. */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, (std::numeric_limits<Element>::lowest)(), x}); }

		/** @brief `a[i] = (max)(a[i], x)` on `[l, r)`. */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, x, (std::numeric_limits<Element>::max)()}); }

		/** @brief `a[i] += x` on `[l, r)`. */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			apply(l, r, tag{x, (std::numeric_limits<Element>::lowest)(), (std::numeric_limits<Element>::max)()});
		}

		/** @brief `a[i] = x` on `[l, r)`. */
		constexpr void assign(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, x, x}); }

		/** @brief `a[i] = x`. */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = value::of(x);
			this->recompute_up(k, beats::merge{});
		}

		/** @brief `a[i]`. */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)].max;
		}

		/** @brief Sum of `[l, r)`. */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total = 0;
			if (l < r) visit_canonical(l, r, [&](std::size_t k) { total += nodes_[k].sum; });
			return total;
		}

		/** @brief Minimum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = (std::numeric_limits<Element>::max)();
			visit_canonical(l, r, [&](std::size_t k) { best = (std::min)(best, nodes_[k].min); });
			return best;
		}

		/** @brief Maximum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = (std::numeric_limits<Element>::lowest)();
			visit_canonical(l, r, [&](std::size_t k) { best = (std::max)(best, nodes_[k].max); });
			return best;
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return n_; }
	};

	/**
	 * @brief by explicit stack: the engine's algorithm with an explicit-stack descent.
	 *
	 * Lost: +22% to +92% against library on every row. Against generic engine it wins only collapse (-14%
	 * / -13% at `-O2`, -9% / -7% native, 2e5 / 5e5), where descents dominate, and is level
	 * to +6% elsewhere.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	using beats_segment_tree_by_explicit_stack = segment_tree_beats_by_policy<Element, Sum, beats::mapping, beats_descent::explicit_stack, true>;

	/**
	 * @brief by optional mapping: the engine's algorithm with the copying `std::optional<Value>` mapping protocol.
	 *
	 * Lost: +10% to +52% against generic engine on every row (balanced mix 5e5: 1395 vs 1162 ms at `-O2`,
	 * 1457 vs 1179 native), so the engine's mapping stays in place. It copies a whole
	 * summary per mapping.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	using beats_segment_tree_by_optional_mapping = segment_tree_beats_by_policy<Element, Sum, copying_clamp_add_mapping, beats_descent::recursive, true>;

	/**
	 * @brief by push or collapse policy: the engine's algorithm without the identity-tag push skip.
	 *
	 * Lost: +3% to +35% against generic engine on every row (collapse 5e5: 328 vs 296 ms at `-O2`, 407
	 * vs 302 native), so the engine keeps the skip.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	using beats_segment_tree_by_unconditional_push = segment_tree_beats_by_policy<Element, Sum, beats::mapping, beats_descent::recursive, false>;

	/**
	 * @brief by push or collapse policy: the engine's algorithm without the collapse fast path.
	 *
	 * Lost: +25% to +27% against generic engine on balanced mix (5e5: 1462 vs 1162 ms at `-O2`, 1468 vs 1179
	 * native) and +3% to +11% on chmin-only, collapse and query-heavy; only add-recharge
	 * leans its way, -2% to -4%. The engine keeps the collapse.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	using beats_segment_tree_by_no_collapse = segment_tree_beats_by_policy<Element, Sum, uncollapsed_clamp_add_mapping, beats_descent::recursive, true>;

	/**
	 * @brief by compact recursion: classic recursive beats over `[0, n)` in `2n - 1` nodes: node `k`
	 *        covering `[lo, hi)` has children `k + 1` (`[lo, mid)`) and
	 *        `k + 2 (mid - lo)` (`[mid, hi)`). The library algebra; one recursion
	 *        locates the range and beats below it.
	 *
	 * Lost: +9% to +126% against library on every row (balanced mix 5e5: 1353 vs 720 ms at `-O2`,
	 * 1420 vs 693 native). Without padding it beats generic engine on add-recharge and collapse (-1%
	 * to -24%) but loses to it by +13% to +23% on balanced mix, chmin-only and query-heavy.
	 *
	 * @tparam Element The element type.
	 * @tparam Sum The sum type.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	class beats_segment_tree_by_recursive_compact_nodes {
		using value = beats::extremes<Element, Sum>;
		using tag = beats::clamp_add<Element>;

		std::size_t n_ = 0;
		std::vector<value> nodes_;
		std::vector<tag> tags_;

		/** @brief Build node @p k over `[lo, hi)` from @p a. */
		constexpr void build(std::size_t k, std::size_t lo, std::size_t hi, std::vector<Element> const &a) {
			if (hi - lo == 1) {
				nodes_[k] = value::of(Element(a[lo]));
				return;
			}
			std::size_t const mid = (lo + hi) / 2, right = k + 2 * (mid - lo);
			build(k + 1, lo, mid, a);
			build(right, mid, hi, a);
			nodes_[k] = beats::merge{}(nodes_[k + 1], nodes_[right]);
		}

		/** @brief Settle node @p k's tag onto its children (`[lo, hi)` internal). */
		constexpr void push(std::size_t k, std::size_t lo, std::size_t hi) {
			if (tags_[k] == tag{}) return;
			std::size_t const mid = (lo + hi) / 2;
			for (std::size_t const c : {k + 1, k + 2 * (mid - lo)}) {
				bool const absorbed = beats::mapping{}(tags_[k], nodes_[c]);
				atl_assert(absorbed);
				tags_[c] = beats::compose{}(tags_[k], tags_[c]);
			}
			tags_[k] = tag{};
		}

		/** @brief Apply @p f to `[l, r)` within node @p k over `[lo, hi)`. */
		constexpr void update(std::size_t k, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, tag const &f) {
			if (r <= lo || hi <= l) return;
			if (l <= lo && hi <= r && beats::mapping{}(f, nodes_[k])) {
				tags_[k] = beats::compose{}(f, tags_[k]);
				return;
			}
			atl_assert(hi - lo > 1);
			push(k, lo, hi);
			std::size_t const mid = (lo + hi) / 2, right = k + 2 * (mid - lo);
			update(k + 1, lo, mid, l, r, f);
			update(right, mid, hi, l, r, f);
			nodes_[k] = beats::merge{}(nodes_[k + 1], nodes_[right]);
		}

		/** @brief Visit the nodes tiling `[l, r)` within node @p k over `[lo, hi)`. */
		template<std::invocable<beats::extremes<Element, Sum> const &> Visit>
		constexpr void query(std::size_t k, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Visit &visit) {
			if (r <= lo || hi <= l) return;
			if (l <= lo && hi <= r) {
				visit(nodes_[k]);
				return;
			}
			push(k, lo, hi);
			std::size_t const mid = (lo + hi) / 2;
			query(k + 1, lo, mid, l, r, visit);
			query(k + 2 * (mid - lo), mid, hi, l, r, visit);
		}

		/** @brief Assign leaf @p i the summary @p x within node @p k over `[lo, hi)`. */
		constexpr void assign_leaf(std::size_t k, std::size_t lo, std::size_t hi, std::size_t i, value const &x) {
			if (hi - lo == 1) {
				nodes_[k] = x;
				return;
			}
			push(k, lo, hi);
			std::size_t const mid = (lo + hi) / 2, right = k + 2 * (mid - lo);
			if (i < mid)
				assign_leaf(k + 1, lo, mid, i, x);
			else
				assign_leaf(right, mid, hi, i, x);
			nodes_[k] = beats::merge{}(nodes_[k + 1], nodes_[right]);
		}

		/** @brief Apply @p f to `[l, r)`. */
		constexpr void apply(std::size_t l, std::size_t r, tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l < r) update(0, 0, n_, l, r, f);
		}

	public:
		/** @brief From initial elements. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
		constexpr explicit beats_segment_tree_by_recursive_compact_nodes(R &&elements) : n_(std::ranges::size(elements)) {
			atl_assert(n_ < (atl::uint64(1) << 32) - 1);
			atl_assert(n_ <= (std::numeric_limits<std::size_t>::max)() / 2);
			std::vector<Element> a;
			a.reserve(n_);
			for (auto &&x : elements) a.push_back(static_cast<Element>(atl::detail::forward_element<R>(x)));
			nodes_.assign(n_ ? 2 * n_ - 1 : 0, value{});
			tags_.assign(nodes_.size(), tag{});
			if (n_ > 0) build(0, 0, n_, a);
		}

		/** @brief `a[i] = (min)(a[i], x)` on `[l, r)`. */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, (std::numeric_limits<Element>::lowest)(), x}); }

		/** @brief `a[i] = (max)(a[i], x)` on `[l, r)`. */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, x, (std::numeric_limits<Element>::max)()}); }

		/** @brief `a[i] += x` on `[l, r)`. */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			apply(l, r, tag{x, (std::numeric_limits<Element>::lowest)(), (std::numeric_limits<Element>::max)()});
		}

		/** @brief `a[i] = x` on `[l, r)`. */
		constexpr void assign(std::size_t l, std::size_t r, Element x) { apply(l, r, tag{0, x, x}); }

		/** @brief `a[i] = x`. */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			assign_leaf(0, 0, n_, i, value::of(x));
		}

		/** @brief `a[i]`. */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			Element v = 0;
			auto read = [&](value const &s) { v = s.max; };
			query(0, 0, n_, i, i + 1, read);
			return v;
		}

		/** @brief Sum of `[l, r)`. */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total = 0;
			auto add_sum = [&](value const &s) { total += s.sum; };
			if (l < r) query(0, 0, n_, l, r, add_sum);
			return total;
		}

		/** @brief Minimum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = (std::numeric_limits<Element>::max)();
			auto take = [&](value const &s) { best = (std::min)(best, s.min); };
			query(0, 0, n_, l, r, take);
			return best;
		}

		/** @brief Maximum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = (std::numeric_limits<Element>::lowest)();
			auto take = [&](value const &s) { best = (std::max)(best, s.max); };
			query(0, 0, n_, l, r, take);
			return best;
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return n_; }
	};

	/**
	 * @brief by fixed clamp-add: a hand-specialized preset with the clamp-add tag: one descent per
	 *        operation that compares only what that operation needs (chmin reads the
	 *        maximum and second maximum, never the clamp protocol), and a push that
	 *        applies the composite without a success flag (its success is the push
	 *        law, asserted).
	 *
	 * Lost: it ties generic engine on the add mix (balanced mix 5e5: -1% at `-O2` and native, 1152 vs 1162
	 * and 1173 vs 1179 ms), which is the library's largest lead (+56% to +70% against library), and on
	 * query-heavy. It wins without add: chmin-only is -48% to -49% against generic engine, and even
	 * beats library there (-14% / -11% at `-O2`, -4% / -4% native, 2e5 / 5e5), as on collapse
	 * at `-O2` (-25% / -22%; native it is +13% / +10%). The library exposes one preset class: there
	 * is no dispatch on whether a caller's stream will use add.
	 *
	 * @tparam Element The element type.
	 * @tparam Sum The sum type.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	class beats_segment_tree_by_fixed_clamp_add : atl::detail::segment_tree_base<beats::extremes<Element, Sum>> {
		using value = beats::extremes<Element, Sum>;
		using tag = beats::clamp_add<Element>;
		using base = atl::detail::segment_tree_base<value>;
		using base::ascend_boundaries;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::size_;

		static constexpr Element lowest = (std::numeric_limits<Element>::lowest)(), highest = (std::numeric_limits<Element>::max)();

		std::vector<tag> tags_;

		/** @brief Every element of node @p k becomes @p x. */
		constexpr void flatten(std::size_t k, Element x) {
			value &s = nodes_[k];
			s.max = s.min = x;
			s.second_max = lowest, s.second_min = highest;
			s.max_count = s.min_count = s.width;
			s.sum = Sum(x) * Sum(s.width);
		}

		/** @brief Lower node @p k's maximum class to @p x. @pre `second_max < x < max` */
		constexpr void lower_max(std::size_t k, Element x) {
			value &s = nodes_[k];
			s.sum -= (Sum(s.max) - Sum(x)) * Sum(s.max_count);
			if (s.second_min == s.max) s.second_min = x;
			s.max = x;
		}

		/** @brief Raise node @p k's minimum class to @p x. @pre `min < x < second_min` */
		constexpr void raise_min(std::size_t k, Element x) {
			value &s = nodes_[k];
			s.sum += (Sum(x) - Sum(s.min)) * Sum(s.min_count);
			if (s.second_max == s.min) s.second_max = x;
			s.min = x;
		}

		/** @brief Add @p x to every element of node @p k (its tag too). */
		constexpr void shift(std::size_t k, Element x) {
			value &s = nodes_[k];
			s.max += x, s.min += x;
			if (s.second_max != lowest) s.second_max += x;
			if (s.second_min != highest) s.second_min += x;
			s.sum += Sum(x) * Sum(s.width);
		}

		/** @brief Apply a pushed composite @p t to child @p c, which by the push law absorbs it. */
		constexpr void settle(std::size_t c, tag const &t) {
			if (nodes_[c].width == 0) return;
			if (t.add != 0) shift(c, t.add);
			value const &s = nodes_[c];
			if (t.upper <= s.min) {
				flatten(c, t.upper);
			} else if (t.lower >= s.max || t.lower == t.upper) {
				flatten(c, t.lower);
			} else {
				if (t.lower > s.min) {
					atl_assert(t.lower < s.second_min);
					raise_min(c, t.lower);
				}
				if (t.upper < s.max) {
					atl_assert(t.upper > s.second_max);
					lower_max(c, t.upper);
				}
			}
			if (c < size_) tags_[c] = beats::compose{}(t, tags_[c]);
		}

		/** @brief Settle node @p k's tag onto its children. */
		constexpr void push_down(std::size_t k) {
			if (tags_[k] == tag{}) return;
			settle(2 * k, tags_[k]);
			settle(2 * k + 1, tags_[k]);
			tags_[k] = tag{};
		}

		/** @brief Settle every tag above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief Recompute node @p k. */
		constexpr void pull(std::size_t k) { nodes_[k] = beats::merge{}(nodes_[2 * k], nodes_[2 * k + 1]); }

		/** @brief Chmin below node @p k. */
		constexpr void chmin_node(std::size_t k, Element x) {
			value const &s = nodes_[k];
			if (x >= s.max) return;
			if (x <= s.min) {
				flatten(k, x);
				if (k < size_) tags_[k] = tag{0, x, x};
			} else if (x > s.second_max) {
				lower_max(k, x);
				if (k < size_) {
					tag &t = tags_[k];
					t.lower = (std::min)(t.lower, x), t.upper = (std::min)(t.upper, x);
					if (t.lower == t.upper) t.add = 0;
				}
			} else {
				push_down(k);
				chmin_node(2 * k, x);
				chmin_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Chmax below node @p k. */
		constexpr void chmax_node(std::size_t k, Element x) {
			value const &s = nodes_[k];
			if (x <= s.min) return;
			if (x >= s.max) {
				flatten(k, x);
				if (k < size_) tags_[k] = tag{0, x, x};
			} else if (x < s.second_min) {
				raise_min(k, x);
				if (k < size_) {
					tag &t = tags_[k];
					t.lower = (std::max)(t.lower, x), t.upper = (std::max)(t.upper, x);
					if (t.lower == t.upper) t.add = 0;
				}
			} else {
				push_down(k);
				chmax_node(2 * k, x);
				chmax_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Push the boundaries of `[l, r)`, run @p on_node on its canonical nodes, pull back. */
		template<std::invocable<std::size_t> OnNode>
		constexpr void update(std::size_t l, std::size_t r, OnNode on_node) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, on_node, on_node);
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

		/** @brief Push the boundaries of `[l, r)`, then visit its canonical nodes. */
		template<std::invocable<std::size_t> Visit>
		constexpr void visit_canonical(std::size_t l, std::size_t r, Visit visit) {
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, visit, visit);
		}

	public:
		/** @brief From initial elements. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
		constexpr explicit beats_segment_tree_by_fixed_clamp_add(R &&elements) {
			atl_assert(std::ranges::size(elements) < (atl::uint64(1) << 32) - 1);
			base::reset(std::ranges::size(elements), value{});
			tags_.assign(size_, tag{});
			std::size_t i = 0;
			for (auto &&x : elements) nodes_[leaf(i++)] = value::of(static_cast<Element>(atl::detail::forward_element<R>(x)));
			this->build(beats::merge{});
		}

		/** @brief `a[i] = (min)(a[i], x)` on `[l, r)`. */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { chmin_node(k, x); });
		}

		/** @brief `a[i] = (max)(a[i], x)` on `[l, r)`. */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { chmax_node(k, x); });
		}

		/** @brief `a[i] += x` on `[l, r)`. */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) {
				shift(k, x);
				if (k < size_) {
					tag &t = tags_[k];
					if (t.lower != t.upper) t.add += x;
					t.lower = atl::detail::beats_shift_bound(t.lower, x), t.upper = atl::detail::beats_shift_bound(t.upper, x);
				}
			});
		}

		/** @brief `a[i] = x` on `[l, r)`. */
		constexpr void assign(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) {
				flatten(k, x);
				if (k < size_) tags_[k] = tag{0, x, x};
			});
		}

		/** @brief `a[i] = x`. */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = value::of(x);
			this->recompute_up(k, beats::merge{});
		}

		/** @brief `a[i]`. */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)].max;
		}

		/** @brief Sum of `[l, r)`. */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total = 0;
			if (l < r) visit_canonical(l, r, [&](std::size_t k) { total += nodes_[k].sum; });
			return total;
		}

		/** @brief Minimum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = highest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::min)(best, nodes_[k].min); });
			return best;
		}

		/** @brief Maximum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = lowest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::max)(best, nodes_[k].max); });
			return best;
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return n_; }
	};

	/** @brief Node layout of @ref beats_segment_tree_by_tagless_layout (by tagless layout); the third, plain AoS of 56 B for int64, is @ref atl::beats_segment_tree. */
	enum class beats_layout {
		aos_aligned, ///< one struct per node, aligned to a 64 B cache line
		soa,		 ///< one array per field
	};

	namespace detail {
		/** @brief The per-node fields of the fixed contenders, in one struct. */
		template<signed_integral Element, signed_integral Sum>
		struct beats_node {
			Element max = (std::numeric_limits<Element>::lowest)();		   ///< largest element
			Element second_max = (std::numeric_limits<Element>::lowest)(); ///< largest below `max`
			Element min = (std::numeric_limits<Element>::max)();		   ///< smallest element
			Element second_min = (std::numeric_limits<Element>::max)();	   ///< smallest above `min`
			Sum sum = 0;												   ///< sum
			uint32 max_count = 0;										   ///< elements equal to `max`
			uint32 min_count = 0;										   ///< elements equal to `min`
			uint32 width = 0;											   ///< elements
		};

		/** @brief @ref beats_node padded to a 64 B cache line. */
		template<signed_integral Element, signed_integral Sum>
		struct alignas(64) beats_node_aligned : beats_node<Element, Sum> {};

		/** @brief Per-node field storage in one of the @ref beats_layout layouts, accessed by node index. */
		template<signed_integral Element, signed_integral Sum, beats_layout Layout>
		class beats_node_store {
			using node = beats_node_aligned<Element, Sum>;
			std::vector<node> nodes_;

		public:
			/** @brief @p count identity nodes. */
			constexpr void assign(std::size_t count) { nodes_.assign(count, node{}); }

			/** @brief Node @p k's `max` field. */
			constexpr Element &(max)(std::size_t k) { return nodes_[k].max; }

			/** @brief Node @p k's `second_max` field. */
			constexpr Element &second_max(std::size_t k) { return nodes_[k].second_max; }

			/** @brief Node @p k's `min` field. */
			constexpr Element &(min)(std::size_t k) { return nodes_[k].min; }

			/** @brief Node @p k's `second_min` field. */
			constexpr Element &second_min(std::size_t k) { return nodes_[k].second_min; }

			/** @brief Node @p k's `sum` field. */
			constexpr Sum &sum(std::size_t k) { return nodes_[k].sum; }

			/** @brief Node @p k's `max_count` field. */
			constexpr uint32 &max_count(std::size_t k) { return nodes_[k].max_count; }

			/** @brief Node @p k's `min_count` field. */
			constexpr uint32 &min_count(std::size_t k) { return nodes_[k].min_count; }

			/** @brief Node @p k's `width` field. */
			constexpr uint32 &width(std::size_t k) { return nodes_[k].width; }
		};

		/** @brief The SoA layout: one array per field. */
		template<signed_integral Element, signed_integral Sum>
		class beats_node_store<Element, Sum, beats_layout::soa> {
			std::vector<Element> max_, second_max_, min_, second_min_;
			std::vector<Sum> sum_;
			std::vector<uint32> max_count_, min_count_, width_;

		public:
			/** @brief @p count identity nodes. */
			constexpr void assign(std::size_t count) {
				max_.assign(count, (std::numeric_limits<Element>::lowest)());
				second_max_.assign(count, (std::numeric_limits<Element>::lowest)());
				min_.assign(count, (std::numeric_limits<Element>::max)());
				second_min_.assign(count, (std::numeric_limits<Element>::max)());
				sum_.assign(count, 0);
				max_count_.assign(count, 0);
				min_count_.assign(count, 0);
				width_.assign(count, 0);
			}

			/** @brief Node @p k's `max` field. */
			constexpr Element &(max)(std::size_t k) { return max_[k]; }

			/** @brief Node @p k's `second_max` field. */
			constexpr Element &second_max(std::size_t k) { return second_max_[k]; }

			/** @brief Node @p k's `min` field. */
			constexpr Element &(min)(std::size_t k) { return min_[k]; }

			/** @brief Node @p k's `second_min` field. */
			constexpr Element &second_min(std::size_t k) { return second_min_[k]; }

			/** @brief Node @p k's `sum` field. */
			constexpr Sum &sum(std::size_t k) { return sum_[k]; }

			/** @brief Node @p k's `max_count` field. */
			constexpr uint32 &max_count(std::size_t k) { return max_count_[k]; }

			/** @brief Node @p k's `min_count` field. */
			constexpr uint32 &min_count(std::size_t k) { return min_count_[k]; }

			/** @brief Node @p k's `width` field. */
			constexpr uint32 &width(std::size_t k) { return width_[k]; }
		};
	}

	/**
	 * @brief by tagless layout: the library's algorithm (@ref atl::beats_segment_tree) in the other two
	 *        layouts: only an add is pending per internal node. A push adds it to the
	 *        children, then flattens them under a single-valued parent, or else clamps a
	 *        child's maximum down to the parent's (only its maximum class can lie above: a
	 *        parent's maximum moves only within its own class) and its minimum up to the
	 *        parent's.
	 *
	 * Both lost to the library's plain 56 B AoS. The 64 B line: +0% to +5% at `-O2`, +4% to +10%
	 * native, with 8 B more per node (balanced mix 5e5: 731 vs 720 ms, 719 vs 693). SoA: +5% to
	 * +79%, worst on balanced mix (5e5: 1208 vs 720 ms at `-O2`, 1239 vs 693 native) and
	 * query-heavy.
	 *
	 * This is the code as raced: unchecked arithmetic, values well inside `Element`.
	 * Under a single-valued parent it still adds to the children before flattening them,
	 * which the library class avoids, as that can overflow next to the sentinels.
	 *
	 * @tparam Element The element type.
	 * @tparam Sum The sum type.
	 * @tparam Layout The node layout (by tagless layout).
	 */
	template<signed_integral Element, signed_integral Sum = Element, beats_layout Layout = beats_layout::aos_aligned>
	class beats_segment_tree_by_tagless_layout {
		static constexpr Element lowest = (std::numeric_limits<Element>::lowest)(), highest = (std::numeric_limits<Element>::max)();

		std::size_t n_ = 0, size_ = 1, log_ = 0;
		detail::beats_node_store<Element, Sum, Layout> s_;
		std::vector<Element> adds_; ///< pending add per internal node

		/** @brief Recompute node @p k from its children. */
		constexpr void pull(std::size_t k) {
			std::size_t const a = 2 * k, b = 2 * k + 1;
			if ((s_.max)(a) > (s_.max)(b)) {
				(s_.max)(k) = (s_.max)(a), s_.max_count(k) = s_.max_count(a), s_.second_max(k) = (std::max)(s_.second_max(a), (s_.max)(b));
			} else if ((s_.max)(a) < (s_.max)(b)) {
				(s_.max)(k) = (s_.max)(b), s_.max_count(k) = s_.max_count(b), s_.second_max(k) = (std::max)((s_.max)(a), s_.second_max(b));
			} else {
				(s_.max)(k) = (s_.max)(a), s_.max_count(k) = s_.max_count(a) + s_.max_count(b),
				s_.second_max(k) = (std::max)(s_.second_max(a), s_.second_max(b));
			}
			if ((s_.min)(a) < (s_.min)(b)) {
				(s_.min)(k) = (s_.min)(a), s_.min_count(k) = s_.min_count(a), s_.second_min(k) = (std::min)(s_.second_min(a), (s_.min)(b));
			} else if ((s_.min)(a) > (s_.min)(b)) {
				(s_.min)(k) = (s_.min)(b), s_.min_count(k) = s_.min_count(b), s_.second_min(k) = (std::min)((s_.min)(a), s_.second_min(b));
			} else {
				(s_.min)(k) = (s_.min)(a), s_.min_count(k) = s_.min_count(a) + s_.min_count(b),
				s_.second_min(k) = (std::min)(s_.second_min(a), s_.second_min(b));
			}
			s_.sum(k) = s_.sum(a) + s_.sum(b);
			s_.width(k) = s_.width(a) + s_.width(b);
		}

		/** @brief Every element of node @p k becomes @p x; its pending add is dropped. */
		constexpr void flatten(std::size_t k, Element x) {
			(s_.max)(k) = (s_.min)(k) = x;
			s_.second_max(k) = lowest, s_.second_min(k) = highest;
			s_.max_count(k) = s_.min_count(k) = s_.width(k);
			s_.sum(k) = Sum(x) * Sum(s_.width(k));
			if (k < size_) adds_[k] = 0;
		}

		/** @brief Lower node @p k's maximum class to @p x. @pre `second_max < x < max` */
		constexpr void lower_max(std::size_t k, Element x) {
			s_.sum(k) -= (Sum((s_.max)(k)) - Sum(x)) * Sum(s_.max_count(k));
			if ((s_.min)(k) == (s_.max)(k))
				(s_.min)(k) = x;
			else if (s_.second_min(k) == (s_.max)(k))
				s_.second_min(k) = x;
			(s_.max)(k) = x;
		}

		/** @brief Raise node @p k's minimum class to @p x. @pre `min < x < second_min` */
		constexpr void raise_min(std::size_t k, Element x) {
			s_.sum(k) += (Sum(x) - Sum((s_.min)(k))) * Sum(s_.min_count(k));
			if ((s_.max)(k) == (s_.min)(k))
				(s_.max)(k) = x;
			else if (s_.second_max(k) == (s_.min)(k))
				s_.second_max(k) = x;
			(s_.min)(k) = x;
		}

		/** @brief Add @p x to every element of node @p k, owing it to the subtree. */
		constexpr void shift(std::size_t k, Element x) {
			if (s_.width(k) == 0) return;
			(s_.max)(k) += x, (s_.min)(k) += x;
			if (s_.second_max(k) != lowest) s_.second_max(k) += x;
			if (s_.second_min(k) != highest) s_.second_min(k) += x;
			s_.sum(k) += Sum(x) * Sum(s_.width(k));
			if (k < size_) adds_[k] += x;
		}

		/** @brief Settle node @p k onto its children: the add, then the parent's extremes. */
		constexpr void push_down(std::size_t k) {
			std::size_t const a = 2 * k, b = 2 * k + 1;
			if (adds_[k] != 0) {
				shift(a, adds_[k]);
				shift(b, adds_[k]);
				adds_[k] = 0;
			}
			Element const top = (s_.max)(k), bottom = (s_.min)(k);
			for (std::size_t const c : {a, b}) {
				if (s_.width(c) == 0) continue;
				if (top == bottom) {
					if ((s_.max)(c) != top || (s_.min)(c) != top) flatten(c, top);
					continue;
				}
				if ((s_.max)(c) > top) lower_max(c, top);
				if ((s_.min)(c) < bottom) raise_min(c, bottom);
			}
		}

		/** @brief Settle every pending tag above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief Push the partially covered ancestors of `[l, r)`, root first. */
		constexpr void push_boundaries(std::size_t l, std::size_t r) {
			l += size_, r += size_;
			for (std::size_t i = log_; i >= 1; --i) {
				if ((l >> i) << i != l) push_down(l >> i);
				if ((r >> i) << i != r) push_down((r - 1) >> i);
			}
		}

		/** @brief Recompute the partially covered ancestors of `[l, r)`, leaf first. */
		constexpr void pull_boundaries(std::size_t l, std::size_t r) {
			l += size_, r += size_;
			for (std::size_t i = 1; i <= log_; ++i) {
				if ((l >> i) << i != l) pull(l >> i);
				if ((r >> i) << i != r) pull((r - 1) >> i);
			}
		}

		/** @brief Visit the canonical nodes of `[l, r)`. */
		template<std::invocable<std::size_t> Visit>
		constexpr void decompose(std::size_t l, std::size_t r, Visit visit) {
			for (l += size_, r += size_; l < r; l >>= 1, r >>= 1) {
				if (l & 1) visit(l++);
				if (r & 1) visit(--r);
			}
		}

		/** @brief An update of `[l, r)` running @p on_node on the canonical nodes. */
		template<std::invocable<std::size_t> OnNode>
		constexpr void update(std::size_t l, std::size_t r, OnNode on_node) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			push_boundaries(l, r);
			decompose(l, r, on_node);
			pull_boundaries(l, r);
		}

		/** @brief Chmin below node @p k. */
		constexpr void chmin_node(std::size_t k, Element x) {
			if (x >= (s_.max)(k)) return;
			if (x <= (s_.min)(k)) {
				flatten(k, x);
			} else if (x > s_.second_max(k)) {
				lower_max(k, x);
			} else {
				push_down(k);
				chmin_node(2 * k, x);
				chmin_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Chmax below node @p k. */
		constexpr void chmax_node(std::size_t k, Element x) {
			if (x <= (s_.min)(k)) return;
			if (x >= (s_.max)(k)) {
				flatten(k, x);
			} else if (x < s_.second_min(k)) {
				raise_min(k, x);
			} else {
				push_down(k);
				chmax_node(2 * k, x);
				chmax_node(2 * k + 1, x);
				pull(k);
			}
		}

	public:
		/** @brief From initial elements. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
		constexpr explicit beats_segment_tree_by_tagless_layout(R &&elements) : n_(std::ranges::size(elements)) {
			atl_assert(n_ < (atl::uint64(1) << 32) - 1);
			atl_assert(n_ <= (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2)));
			size_ = std::bit_ceil(n_ ? n_ : std::size_t(1));
			log_ = std::size_t(std::countr_zero(size_));
			s_.assign(2 * size_);
			adds_.assign(size_, 0);
			std::size_t i = 0;
			for (auto &&x : elements) {
				std::size_t const k = size_ + i++;
				s_.width(k) = 1;
				flatten(k, static_cast<Element>(atl::detail::forward_element<R>(x)));
			}
			for (std::size_t k = size_; k-- > 1;) pull(k);
		}

		/** @brief `a[i] = (min)(a[i], x)` on `[l, r)`. */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { chmin_node(k, x); });
		}

		/** @brief `a[i] = (max)(a[i], x)` on `[l, r)`. */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { chmax_node(k, x); });
		}

		/** @brief `a[i] += x` on `[l, r)`. */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { shift(k, x); });
		}

		/** @brief `a[i] = x` on `[l, r)`. */
		constexpr void assign(std::size_t l, std::size_t r, Element x) {
			update(l, r, [&](std::size_t k) { flatten(k, x); });
		}

		/** @brief `a[i] = x`. */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			std::size_t k = size_ + i;
			push_to_leaf(k);
			flatten(k, x);
			while (k >>= 1) pull(k);
		}

		/** @brief `a[i]`. */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(size_ + i);
			return (s_.max)(size_ + i);
		}

		/** @brief Sum of `[l, r)`. */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total = 0;
			if (l == r) return total;
			push_boundaries(l, r);
			decompose(l, r, [&](std::size_t k) { total += s_.sum(k); });
			return total;
		}

		/** @brief Minimum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = highest;
			push_boundaries(l, r);
			decompose(l, r, [&](std::size_t k) { best = (std::min)(best, (s_.min)(k)); });
			return best;
		}

		/** @brief Maximum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = lowest;
			push_boundaries(l, r);
			decompose(l, r, [&](std::size_t k) { best = (std::max)(best, (s_.max)(k)); });
			return best;
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return n_; }
	};

	/**
	 * @brief by three adds: class-specific three-add tag: pending adds for the maximum's class, the
	 *        minimum's class and every other element. A push routes a child's maximum
	 *        (minimum) class the max (min) add when the child holds the parent's
	 *        maximum (minimum), the other add otherwise. The tag cannot flatten a node
	 *        holding three or more values, so an assign or a clamp past a whole node
	 *        descends until each node holds at most two.
	 *
	 * Lost: +13% to +62% against library on every row (balanced mix 5e5: 1154 vs 720 ms at `-O2`, 1121
	 * vs 693 native). Against generic engine it wins without add (chmin-only -32% to -36%) and is -1% to
	 * -6% on balanced mix.
	 *
	 * @tparam Element The element type.
	 * @tparam Sum The sum type.
	 */
	template<signed_integral Element, signed_integral Sum = Element>
	class beats_segment_tree_by_three_adds : atl::detail::segment_tree_base<beats::extremes<Element, Sum>> {
		using value = beats::extremes<Element, Sum>;
		using base = atl::detail::segment_tree_base<value>;
		using base::ascend_boundaries;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::size_;

		static constexpr Element lowest = (std::numeric_limits<Element>::lowest)(), highest = (std::numeric_limits<Element>::max)();

		/** @brief The three pending adds of an internal node. */
		struct three_adds {
			Element to_max = 0;	  ///< for the maximum's class
			Element to_min = 0;	  ///< for the minimum's class
			Element to_other = 0; ///< for every other element

			constexpr bool operator==(three_adds const &) const = default;
		};

		std::vector<three_adds> tags_;

		/**
		 * @brief Add @p to_max to node @p k's maximum class, @p to_min to its minimum
		 *        class and @p to_other to the rest; a single-valued node takes the one
		 *        of the first two that differs from @p to_other (they agree when both
		 *        apply).
		 */
		constexpr void update(std::size_t k, Element to_max, Element to_min, Element to_other) {
			value &s = nodes_[k];
			if (s.width == 0) return;
			if (s.max == s.min) {
				if (to_max == to_other)
					to_max = to_min;
				else
					to_min = to_max;
				s.sum += Sum(to_max) * Sum(s.width);
			} else {
				s.sum += Sum(to_max) * Sum(s.max_count) + Sum(to_min) * Sum(s.min_count) + Sum(to_other) * Sum(s.width - s.max_count - s.min_count);
				if (s.second_max == s.min)
					s.second_max += to_min;
				else
					s.second_max += to_other;
				if (s.second_min == s.max)
					s.second_min += to_max;
				else
					s.second_min += to_other;
			}
			s.max += to_max, s.min += to_min;
			if (s.max == s.min) { // two values merged
				s.second_max = lowest, s.second_min = highest;
				s.max_count = s.min_count = s.width;
			}
			if (k < size_) {
				three_adds &t = tags_[k];
				t.to_max += to_max, t.to_min += to_min, t.to_other += to_other;
			}
		}

		/** @brief Settle node @p k's three adds onto its children. */
		constexpr void push_down(std::size_t k) {
			three_adds const t = tags_[k];
			if (t == three_adds{}) return;
			std::size_t const a = 2 * k, b = 2 * k + 1;
			Element const top = (std::max)(nodes_[a].max, nodes_[b].max), bottom = (std::min)(nodes_[a].min, nodes_[b].min);
			for (std::size_t const c : {a, b})
				update(c, nodes_[c].max == top ? t.to_max : t.to_other, nodes_[c].min == bottom ? t.to_min : t.to_other, t.to_other);
			tags_[k] = three_adds{};
		}

		/** @brief Settle every tag above leaf node @p k. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

		/** @brief Recompute node @p k. */
		constexpr void pull(std::size_t k) { nodes_[k] = beats::merge{}(nodes_[2 * k], nodes_[2 * k + 1]); }

		/** @brief Chmin below node @p k. */
		constexpr void chmin_node(std::size_t k, Element x) {
			value const &s = nodes_[k];
			if (x >= s.max) return;
			if (x > s.second_max) {
				update(k, Element(x - s.max), 0, 0);
			} else if (x <= s.min && s.second_max == s.min) {
				update(k, Element(x - s.max), Element(x - s.min), 0);
			} else {
				push_down(k);
				chmin_node(2 * k, x);
				chmin_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Chmax below node @p k. */
		constexpr void chmax_node(std::size_t k, Element x) {
			value const &s = nodes_[k];
			if (x <= s.min) return;
			if (x < s.second_min) {
				update(k, 0, Element(x - s.min), 0);
			} else if (x >= s.max && s.second_min == s.max) {
				update(k, Element(x - s.max), Element(x - s.min), 0);
			} else {
				push_down(k);
				chmax_node(2 * k, x);
				chmax_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief Assign below node @p k: direct on at most two values, else descend. */
		constexpr void assign_node(std::size_t k, Element x) {
			value const &s = nodes_[k];
			if (s.max == s.min || s.second_max == s.min) {
				update(k, Element(x - s.max), Element(x - s.min), 0);
			} else {
				push_down(k);
				assign_node(2 * k, x);
				assign_node(2 * k + 1, x);
				pull(k);
			}
		}

		/** @brief An update of `[l, r)` running @p on_node on the canonical nodes. */
		template<std::invocable<std::size_t> OnNode>
		constexpr void range_update(std::size_t l, std::size_t r, OnNode on_node) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, on_node, on_node);
			ascend_boundaries(l, r, [&](std::size_t k) { pull(k); });
		}

		/** @brief Push the boundaries of `[l, r)`, then visit its canonical nodes. */
		template<std::invocable<std::size_t> Visit>
		constexpr void visit_canonical(std::size_t l, std::size_t r, Visit visit) {
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(l, r, visit, visit);
		}

	public:
		/** @brief From initial elements. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
		constexpr explicit beats_segment_tree_by_three_adds(R &&elements) {
			atl_assert(std::ranges::size(elements) < (atl::uint64(1) << 32) - 1);
			base::reset(std::ranges::size(elements), value{});
			tags_.assign(size_, three_adds{});
			std::size_t i = 0;
			for (auto &&x : elements) nodes_[leaf(i++)] = value::of(static_cast<Element>(atl::detail::forward_element<R>(x)));
			this->build(beats::merge{});
		}

		/** @brief `a[i] = (min)(a[i], x)` on `[l, r)`. */
		constexpr void chmin(std::size_t l, std::size_t r, Element x) {
			range_update(l, r, [&](std::size_t k) { chmin_node(k, x); });
		}

		/** @brief `a[i] = (max)(a[i], x)` on `[l, r)`. */
		constexpr void chmax(std::size_t l, std::size_t r, Element x) {
			range_update(l, r, [&](std::size_t k) { chmax_node(k, x); });
		}

		/** @brief `a[i] += x` on `[l, r)`. */
		constexpr void add(std::size_t l, std::size_t r, Element x) {
			range_update(l, r, [&](std::size_t k) { update(k, x, x, x); });
		}

		/** @brief `a[i] = x` on `[l, r)`. */
		constexpr void assign(std::size_t l, std::size_t r, Element x) {
			range_update(l, r, [&](std::size_t k) { assign_node(k, x); });
		}

		/** @brief `a[i] = x`. */
		constexpr void set(std::size_t i, Element x) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = value::of(x);
			this->recompute_up(k, beats::merge{});
		}

		/** @brief `a[i]`. */
		[[nodiscard]] constexpr Element element(std::size_t i) {
			atl_assert(i < n_);
			push_to_leaf(leaf(i));
			return nodes_[leaf(i)].max;
		}

		/** @brief Sum of `[l, r)`. */
		[[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			Sum total = 0;
			if (l < r) visit_canonical(l, r, [&](std::size_t k) { total += nodes_[k].sum; });
			return total;
		}

		/** @brief Minimum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(min)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = highest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::min)(best, nodes_[k].min); });
			return best;
		}

		/** @brief Maximum of `[l, r)`. @pre `l < r` */
		[[nodiscard]] constexpr Element(max)(std::size_t l, std::size_t r) {
			atl_assert(l < r && r <= n_);
			Element best = lowest;
			visit_canonical(l, r, [&](std::size_t k) { best = (std::max)(best, nodes_[k].max); });
			return best;
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return n_; }
	};
}
