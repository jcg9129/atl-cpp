#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/splay_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief The template-argument contract shared with `treap_sequence:`
		 *        tiers switched on by non-`void` arguments, each tier's algebra
		 *        well-formed.
		 */
		template<typename T, typename Op, typename F, typename Mapping, typename Compose>
		concept splay_sequence_algebra = (std::is_void_v<Op> || binary_operation<Op, T>)
									  && (std::is_void_v<F> == std::is_void_v<Mapping>)
									  && (std::is_void_v<F> == std::is_void_v<Compose>)
									  && (std::is_void_v<F> || (!std::is_void_v<Op> && std::semiregular<F>))
									  && (std::is_void_v<Mapping> || action<Mapping, F, T>)
									  && (std::is_void_v<Compose> || binary_operation<Compose, F>);

		/**
		 * @brief Node pool of a splay sequence: storage in the chosen layout,
		 *        recycling, the algebra, and the `sized_splay_store` interface
		 *        (with `push` and `pull`) the engine drives.
		 *
		 * Node 0 is the null sentinel: size 0 and, on the fold tiers, the
		 * identity in both aggregates, so `pull` folds through empty sides
		 * without branching.
		 */
		template<splay_sequence_layout Layout, std::semiregular T, typename Op, typename F, typename Mapping, typename Compose,
				 bool IsReversible>
			requires splay_sequence_algebra<T, Op, F, Mapping, Compose>
		struct splay_sequence_pool {
			static constexpr bool has_op = !std::is_void_v<Op>;
			static constexpr bool has_tags = !std::is_void_v<F>;
			static constexpr bool is_interleaved = std::same_as<Layout, splay_layout_interleaved>;
			static constexpr bool is_columnar = std::same_as<Layout, splay_layout_columnar>;

			struct no_slot {};

			struct no_aggregate_slot {};

			struct no_reversed_aggregate_slot {};

			struct no_tag_slot {};

			struct no_reversed_slot {};

			struct aggregate_slot {
				T aggregate{};
			};

			/** @brief The subtree fold in reversed order (reversible fold tier). */
			struct reversed_aggregate_slot {
				T reversed_aggregate{}; ///< fold in reversed order, for correct folds under pending flips
			};

			struct tag_slot {
				F tag{};
			};

			struct reversed_slot {
				bool is_reversed = false;
			};

			template<typename X>
			using slot = std::conditional_t<std::is_void_v<X>, no_slot, X>;

			struct payload : std::conditional_t<has_op, aggregate_slot, no_aggregate_slot>,
							 std::conditional_t<IsReversible && has_op, reversed_aggregate_slot, no_reversed_aggregate_slot>,
							 std::conditional_t<has_tags, tag_slot, no_tag_slot>,
							 std::conditional_t<IsReversible, reversed_slot, no_reversed_slot> {
				T value{};
			};

			/** @brief A node's links and subtree size (the split layout's 16-byte record). */
			struct link {
				uint32 child[2]{};
				uint32 parent = 0, size = 0;
			};

			/** @brief The interleaved layout's one record per node. */
			struct node : payload, link {};

			/** @brief Columnar storage: one array per field; unused fields stay empty. */
			struct columns {
				std::array<std::vector<uint32>, 2> children;
				std::vector<uint32> parents, sizes;
				std::vector<T> values, aggregates, reversed_aggregates;
				std::vector<slot<F>> tags;
				std::vector<uint8> reversed_flags;
			};

			struct interleaved_storage {
				std::vector<node> nodes;
			};

			struct split_storage {
				std::vector<link> links;
				std::vector<payload> payloads;
			};

			using storage = std::conditional_t<is_interleaved, interleaved_storage,
											   std::conditional_t<is_columnar, columns, split_storage>>;

			storage store;
			std::vector<uint32> recycled;
			[[no_unique_address]] slot<Op> op;
			[[no_unique_address]] slot<Mapping> mapping;
			[[no_unique_address]] slot<Compose> compose;
			[[no_unique_address]] std::conditional_t<has_op, T, no_slot> element_identity;
			[[no_unique_address]] slot<F> tag_identity;

			explicit splay_sequence_pool(slot<Op> fold = {}, slot<Mapping> map = {}, slot<Compose> composition = {},
										 std::conditional_t<has_op, T, no_slot> identity = {}, slot<F> tag_unit = {})
				: op(detail::move_or_copy(fold)), mapping(detail::move_or_copy(map)), compose(detail::move_or_copy(composition)),
				  element_identity(std::move(identity)), tag_identity(std::move(tag_unit)) {
				append_node();
				seal_identities();
			}

			decltype(auto) fold_values(auto &&a, auto &&b) const
				requires has_op && requires { detail::invoke_algebra<T>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<T>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			decltype(auto) map_value(auto &&tag_value, auto &&value) const
				requires has_tags && requires { detail::invoke_algebra<T>(mapping, std::forward<decltype(tag_value)>(tag_value), std::forward<decltype(value)>(value)); }
			{ return detail::invoke_algebra<T>(mapping, std::forward<decltype(tag_value)>(tag_value), std::forward<decltype(value)>(value)); }

			decltype(auto) compose_tags(auto &&a, auto &&b) const
				requires has_tags && requires { detail::invoke_algebra<slot<F>>(compose, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<slot<F>>(compose, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			/** @brief Install the algebra's identities into the null sentinel. */
			void seal_identities() {
				if constexpr (has_op) aggregate(0) = element_identity;
				if constexpr (IsReversible && has_op) reversed_aggregate(0) = element_identity;
			}

			[[nodiscard]] uint32 child(uint32 x, bool side) const {
				if constexpr (is_interleaved)
					return store.nodes[x].child[side];
				else if constexpr (is_columnar)
					return store.children[side][x];
				else
					return store.links[x].child[side];
			}

			[[nodiscard]] uint32 parent(uint32 x) const {
				if constexpr (is_interleaved)
					return store.nodes[x].parent;
				else if constexpr (is_columnar)
					return store.parents[x];
				else
					return store.links[x].parent;
			}

			[[nodiscard]] uint32 size(uint32 x) const {
				if constexpr (is_interleaved)
					return store.nodes[x].size;
				else if constexpr (is_columnar)
					return store.sizes[x];
				else
					return store.links[x].size;
			}

			void set_child(uint32 x, bool side, uint32 c) {
				if constexpr (is_interleaved)
					store.nodes[x].child[side] = c;
				else if constexpr (is_columnar)
					store.children[side][x] = c;
				else
					store.links[x].child[side] = c;
			}

			void set_parent(uint32 x, uint32 p) {
				if constexpr (is_interleaved)
					store.nodes[x].parent = p;
				else if constexpr (is_columnar)
					store.parents[x] = p;
				else
					store.links[x].parent = p;
			}

			/** @brief Set x's subtree size. */
			void set_size(uint32 x, uint32 n) {
				if constexpr (is_interleaved)
					store.nodes[x].size = n;
				else if constexpr (is_columnar)
					store.sizes[x] = n;
				else
					store.links[x].size = n;
			}

			[[nodiscard]] T &value(uint32 x) {
				if constexpr (is_interleaved)
					return store.nodes[x].value;
				else if constexpr (is_columnar)
					return store.values[x];
				else
					return store.payloads[x].value;
			}

			[[nodiscard]] T &aggregate(uint32 x)
				requires has_op
			{
				if constexpr (is_interleaved)
					return store.nodes[x].aggregate;
				else if constexpr (is_columnar)
					return store.aggregates[x];
				else
					return store.payloads[x].aggregate;
			}

			[[nodiscard]] T &reversed_aggregate(uint32 x)
				requires(IsReversible && has_op)
			{
				if constexpr (is_interleaved)
					return store.nodes[x].reversed_aggregate;
				else if constexpr (is_columnar)
					return store.reversed_aggregates[x];
				else
					return store.payloads[x].reversed_aggregate;
			}

			[[nodiscard]] slot<F> &tag(uint32 x)
				requires has_tags
			{
				if constexpr (is_interleaved)
					return store.nodes[x].tag;
				else if constexpr (is_columnar)
					return store.tags[x];
				else
					return store.payloads[x].tag;
			}

			[[nodiscard]] bool is_reversed(uint32 x)
				requires IsReversible
			{
				if constexpr (is_interleaved)
					return store.nodes[x].is_reversed;
				else if constexpr (is_columnar)
					return store.reversed_flags[x] != 0;
				else
					return store.payloads[x].is_reversed;
			}

			void set_reversed(uint32 x, bool is_set)
				requires IsReversible
			{
				if constexpr (is_interleaved)
					store.nodes[x].is_reversed = is_set;
				else if constexpr (is_columnar)
					store.reversed_flags[x] = uint8(is_set);
				else
					store.payloads[x].is_reversed = is_set;
			}

			/** @brief Size and aggregates from the children; branchless through null sides. */
			void pull(uint32 x) {
				uint32 const l = child(x, 0), r = child(x, 1);
				set_size(x, size(l) + 1 + size(r));
				if constexpr (has_op) aggregate(x) = fold_values(fold_values(aggregate(l), value(x)), aggregate(r));
				if constexpr (IsReversible && has_op)
					reversed_aggregate(x) = fold_values(fold_values(reversed_aggregate(r), value(x)), reversed_aggregate(l));
			}

			/**
			 * @brief Fold tag @p f into node @p x: value and aggregates now, the tag
			 *        composed for the subtree below (the lazy tree's convention).
			 */
			void apply_tag(uint32 x, slot<F> const &f)
				requires has_tags
			{
				value(x) = map_value(f, value(x));
				aggregate(x) = map_value(f, aggregate(x));
				if constexpr (IsReversible) reversed_aggregate(x) = map_value(f, reversed_aggregate(x));
				tag(x) = compose_tags(f, tag(x));
			}

			/**
			 * @brief Mark subtree @p x reversed: its aggregates swap now, the structural
			 *        swap stays pending (settled by `push`).
			 */
			void flip(uint32 x)
				requires IsReversible
			{
				if (x == 0) return;
				set_reversed(x, !is_reversed(x));
				if constexpr (has_op) std::ranges::swap(aggregate(x), reversed_aggregate(x));
			}

			/**
			 * @brief Drop x's pending-reversal mark without touching anything else (a
			 *        hook for push strategies that settle reversal parities in bulk).
			 */
			void clear_reversed(uint32 x)
				requires(IsReversible && !has_tags)
			{
				set_reversed(x, false);
			}

			/** @brief Engine hook: settle x's pending reversal and tag one level down. */
			void push(uint32 x) {
				if constexpr (IsReversible) {
					if (is_reversed(x)) {
						uint32 const l = child(x, 0), r = child(x, 1);
						set_child(x, 0, r);
						set_child(x, 1, l);
						flip(l);
						flip(r);
						set_reversed(x, false);
					}
				}
				if constexpr (has_tags) {
					slot<F> &pending = tag(x);
					// an identity tag has nothing to settle: skip the child mappings
					if constexpr (std::equality_comparable<F>)
						if (pending == tag_identity) return;
					if (uint32 const l = child(x, 0); l != 0) apply_tag(l, pending);
					if (uint32 const r = child(x, 1); r != 0) apply_tag(r, pending);
					pending = tag_identity;
				}
			}

			uint32 append_node() {
				if constexpr (is_interleaved) {
					store.nodes.emplace_back();
					return uint32(store.nodes.size() - 1);
				} else if constexpr (is_columnar) {
					store.children[0].push_back(0);
					store.children[1].push_back(0);
					store.parents.push_back(0);
					store.sizes.push_back(0);
					store.values.emplace_back();
					if constexpr (has_op) store.aggregates.emplace_back();
					if constexpr (IsReversible && has_op) store.reversed_aggregates.emplace_back();
					if constexpr (has_tags) store.tags.emplace_back();
					if constexpr (IsReversible) store.reversed_flags.push_back(0);
					return uint32(store.sizes.size() - 1);
				} else {
					store.links.emplace_back();
					store.payloads.emplace_back();
					return uint32(store.links.size() - 1);
				}
			}

			[[nodiscard]] std::size_t slot_count() const {
				if constexpr (is_interleaved)
					return store.nodes.size();
				else if constexpr (is_columnar)
					return store.sizes.size();
				else
					return store.links.size();
			}

			/**
			 * @brief A fresh detached node holding `T(args...)`: one construction
			 *        plus one move into the slot.
			 */
			template<typename... Args>
				requires std::constructible_from<T, Args...>
			uint32 make(Args &&...args) {
				// built before the arrays may grow: an argument can alias an
				// element of this very pool (a copy, `emplace_back(s.at(0))`)
				T fresh(std::forward<Args>(args)...);
				uint32 id;
				if (!recycled.empty()) {
					id = recycled.back();
					recycled.pop_back();
				} else {
					// ids are 32-bit; id 0 is the null sentinel
					atl_assert(slot_count() < (std::numeric_limits<uint32>::max)());
					id = append_node();
				}
				set_child(id, 0, 0);
				set_child(id, 1, 0);
				set_parent(id, 0);
				set_size(id, 1);
				value(id) = std::move(fresh);
				if constexpr (has_op) aggregate(id) = value(id);
				if constexpr (IsReversible && has_op) reversed_aggregate(id) = value(id);
				if constexpr (IsReversible) set_reversed(id, false);
				if constexpr (has_tags) tag(id) = tag_identity;
				return id;
			}

			/**
			 * @brief Return every node of subtree @p root to the free list: a stackless
			 *        preorder walk by parent pointers, links left intact until reuse.
			 */
			void recycle(uint32 root) {
				for (uint32 x = root; x != 0;) {
					// a heap-owning element is released now, not when the id is reused
					if constexpr (!std::is_trivially_destructible_v<T>) {
						value(x) = T{};
						if constexpr (has_op) aggregate(x) = T{};
						if constexpr (IsReversible && has_op) reversed_aggregate(x) = T{};
					}
					recycled.push_back(x);
					if (uint32 const l = child(x, 0); l != 0) {
						x = l;
						continue;
					}
					if (uint32 const r = child(x, 1); r != 0) {
						x = r;
						continue;
					}
					for (;;) { // climb to the next unvisited right subtree
						if (x == root) return;
						uint32 const p = parent(x), r = child(p, 1);
						if (r != 0 && r != x) {
							x = r;
							break;
						}
						x = p;
					}
				}
			}
		};

		/**
		 * @brief The engine operations a splay sequence calls on its pool:
		 *        rank splays, joins, splits, inserts, handle splays, the
		 *        in-order visit and the balanced build.
		 */
		template<typename Engine, typename Store>
		concept splay_sequence_engine = sized_splay_store<Store> && requires(Store &s, uint32 x) {
			{ Engine::splay_at(s, x, x) } -> std::same_as<uint32>;
			{ Engine::join(s, x, x) } -> std::same_as<uint32>;
			{ Engine::split_at(s, x, x) } -> std::same_as<std::pair<uint32, uint32>>;
			{ Engine::insert_at(s, x, x, x) } -> std::same_as<uint32>;
			Engine::splay(s, x);
			Engine::for_each(s, x, [](uint32) {});
			{ Engine::build_balanced(s, std::vector<uint32>{}) } -> std::same_as<uint32>;
		};

		struct splay_sequence_adopt_t {
			explicit splay_sequence_adopt_t() = default;
		};

		/**
		 * @brief The whole `atl::splay_sequence`, over any engine and node
		 *        layout (CRTP: @p Derived is the class handed back by `split` and
		 *        `sibling`), so the experimental contenders share every line.
		 */
		template<typename Derived, typename Engine, splay_sequence_layout Layout, std::semiregular T, typename Op, typename F,
				 typename Mapping, typename Compose, bool IsReversible>
			requires splay_sequence_algebra<T, Op, F, Mapping, Compose>
				  && splay_sequence_engine<Engine, splay_sequence_pool<Layout, T, Op, F, Mapping, Compose, IsReversible>>
		class splay_sequence_impl {
			using pool = splay_sequence_pool<Layout, T, Op, F, Mapping, Compose, IsReversible>;
			static constexpr bool has_op = pool::has_op;
			static constexpr bool has_tags = pool::has_tags;

			/** @brief The pool's algebra-member slot. */
			template<typename X>
			using slot = typename pool::template slot<X>;

			std::shared_ptr<pool> pool_;
			uint32 root_ = 0;

			/**
			 * @brief `[l, r)` gathered into one subtree: `middle`; `lower` holds `l - 1`
			 *        at the root and `upper` holds `r` (0 when the side is absent).
			 */
			struct range_handle {
				uint32 lower, upper, middle;
			};

			/**
			 * @brief Gather the non-empty `[l, r)` into one subtree with two
			 *        splays: rank `l - 1` to the root, then rank `r - l` to the top
			 *        of its (temporarily detached) right subtree, whose left
			 *        subtree is then exactly the range. After changing the middle,
			 *        re-pull with `settle`.
			 */
			range_handle isolate(uint32 l, uint32 r) {
				pool &p = *pool_;
				uint32 lower = 0, rest = root_;
				if (l > 0) {
					lower = Engine::splay_at(p, root_, l - 1);
					rest = p.child(lower, 1);
					p.set_parent(rest, 0); // the splay below must stop at rest's root
				}
				uint32 upper = 0, middle = rest;
				if (r - l < p.size(rest)) {
					upper = Engine::splay_at(p, rest, r - l);
					middle = p.child(upper, 0);
					rest = upper;
				}
				if (lower != 0) {
					p.set_child(lower, 1, rest);
					p.set_parent(rest, lower);
					root_ = lower;
				} else {
					root_ = rest;
				}
				return {lower, upper, middle};
			}

			/** @brief Re-pull the two nodes `isolate` splayed, after the range below them changed. */
			void settle(range_handle const &h) {
				if (h.upper != 0) pool_->pull(h.upper);
				if (h.lower != 0) pool_->pull(h.lower);
			}

			/**
			 * @brief In-order visit of subtree values without settling anything:
			 *        pending reversal parity and tags ride along an explicit stack.
			 */
			template<typename Visit>
				requires std::invocable<Visit &, T const &>
			void visit_values(uint32 root, Visit &visit) const {
				pool &p = *pool_;

				/** @brief A visited node with the reversal parity and tag composite above it. */
				struct frame {
					uint32 x;
					bool is_flipped;					 ///< x's children are to be read mirrored
					[[no_unique_address]] slot<F> above; ///< composite of the ancestors' tags
				};

				std::vector<frame> stack;
				auto const descend_left = [&](uint32 x, bool is_flipped, slot<F> above) {
					while (x != 0) {
						if constexpr (IsReversible) is_flipped = is_flipped != p.is_reversed(x);
						slot<F> below = above;
						if constexpr (has_tags) below = p.compose_tags(above, p.tag(x));
						stack.push_back({x, is_flipped, std::move(above)});
						x = p.child(x, !is_flipped ? 0 : 1);
						above = std::move(below);
					}
				};
				descend_left(root, false, p.tag_identity);
				while (!stack.empty()) {
					frame f = std::move(stack.back());
					stack.pop_back();
					slot<F> below = f.above;
					if constexpr (has_tags) {
						below = p.compose_tags(f.above, p.tag(f.x));
						T const shown = p.map_value(f.above, p.value(f.x));
						std::invoke(visit, shown);
					} else {
						std::invoke(visit, std::as_const(p.value(f.x)));
					}
					descend_left(p.child(f.x, !f.is_flipped ? 1 : 0), f.is_flipped, std::move(below));
				}
			}

			/** @brief Nodes for `[it, last)`, in order, linked balanced. @complexity O(n) */
			template<std::ranges::range Source, std::input_iterator It, std::sentinel_for<It> Se>
				requires requires(It it) { static_cast<T>(detail::forward_element<Source>(*it)); }
			uint32 build_tree(It it, Se last) {
				std::vector<uint32> ids;
				if constexpr (std::sized_sentinel_for<Se, It>) ids.reserve(std::size_t(last - it));
				for (; it != last; ++it) ids.push_back(pool_->make(static_cast<T>(forward_element<Source>(*it))));
				return Engine::build_balanced(*pool_, ids);
			}

			/** @brief A balanced copy of @p o's tree in this pool, tags settled. @complexity O(n) */
			uint32 clone_from(splay_sequence_impl const &o) {
				std::vector<uint32> ids;
				ids.reserve(o.size());
				auto copy = [&](T const &v) { ids.push_back(pool_->make(v)); };
				o.visit_values(o.root_, copy);
				return Engine::build_balanced(*pool_, ids);
			}

		public:
			/** @brief Stored element type. */
			using value_type = T;
			/** @brief Type used for positions and element counts. */
			using size_type = std::size_t;

			/** @brief Adopt subtree @p root of @p shared (the `split`/`sibling` path). @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl(splay_sequence_adopt_t, std::shared_ptr<pool> shared, uint32 root)
				: pool_(std::move(shared)), root_(root) {}

			/** @brief An empty sequence (plain tier; algebra tiers start from values). @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl()
				requires(std::is_void_v<Op>)
				: pool_(std::make_shared<pool>()) {}

			/** @brief Construct @p n copies of @p fill (plain tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl(std::size_t n, T fill)
				requires(std::is_void_v<Op>)
				: pool_(std::make_shared<pool>()) {
				auto const same = std::views::repeat(fill, n);
				root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
			}

			/** @brief From initial values (plain tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
			template<std::ranges::sized_range R>
				requires(std::is_void_v<Op> && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
			explicit splay_sequence_impl(R &&values) : pool_(std::make_shared<pool>()) {
				root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
			}

			/** @brief Construct @p n elements, all equal to @p s_identity (fold tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
			template<typename OpArg = Op>
				requires(!std::is_void_v<Op> && std::is_void_v<F> && std::same_as<OpArg, Op>)
			splay_sequence_impl(std::size_t n, T s_identity, std::type_identity_t<OpArg> op) : pool_(std::make_shared<pool>(detail::move_or_copy(op), slot<Mapping>{}, slot<Compose>{}, std::move(s_identity))) {
				auto const same = std::views::repeat(pool_->element_identity, n);
				root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
			}

			/**
			 * @brief From initial values with a fold algebra; `T` is non-deduced so
			 *        class deduction goes through the guides, which reject narrowing.
			 *        @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<std::ranges::sized_range R, typename OpArg = Op>
				requires(!std::is_void_v<Op> && std::is_void_v<F> && std::same_as<OpArg, Op> && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
			splay_sequence_impl(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<OpArg> op) : pool_(std::make_shared<pool>(detail::move_or_copy(op), slot<Mapping>{}, slot<Compose>{}, std::move(s_identity))) {
				root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
			}

			/** @brief Construct @p n elements equal to @p s_identity, with the full tag algebra.
			 * @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename OpArg = Op, typename FArg = F, typename MapArg = Mapping, typename CompArg = Compose>
				requires(!std::is_void_v<F> && std::same_as<OpArg, Op> && std::same_as<FArg, F> && std::same_as<MapArg, Mapping>
						 && std::same_as<CompArg, Compose>)
			splay_sequence_impl(std::size_t n, T s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)
				: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), std::move(s_identity), std::move(f_identity))) {
				auto const same = std::views::repeat(pool_->element_identity, n);
				root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
			}

			/**
			 * @brief From initial values with the full tag algebra; narrowing-guarded
			 *        as above. @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<std::ranges::sized_range R, typename OpArg = Op, typename FArg = F, typename MapArg = Mapping,
					 typename CompArg = Compose>
				requires(!std::is_void_v<F> && std::same_as<OpArg, Op> && std::same_as<FArg, F> && std::same_as<MapArg, Mapping>
						 && std::same_as<CompArg, Compose> && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
			splay_sequence_impl(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping,
								std::type_identity_t<CompArg> compose)
				: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), std::move(s_identity), std::move(f_identity))) {
				root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
			}

			/** @brief Clone the source nodes into its shared pool. @param o Source handle. @complexity O(source size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl(splay_sequence_impl const &o) : pool_(o.pool_), root_(o.pool_ ? clone_from(o) : 0) {}

			/** @brief Transfer the pool and tree root. @param o Source, empty on return. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl(splay_sequence_impl &&o) noexcept : pool_(std::move(o.pool_)), root_(o.root_) { o.root_ = 0; }

			/** @brief Recycle this tree, then clone the source into its pool. @param o Source tree. @return `*this`. @complexity O(destination size + source size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl &operator=(splay_sequence_impl const &o) {
				if (this != &o) {
					if (pool_) pool_->recycle(root_);
					pool_ = o.pool_;
					root_ = pool_ ? clone_from(o) : 0;
				}
				return *this;
			}

			/** @brief Recycle this tree, then transfer the source handle. @param o Source, empty on return. @return `*this`. @complexity O(destination size) @note Not constexpr (uses a shared_ptr node pool). */
			splay_sequence_impl &operator=(splay_sequence_impl &&o) noexcept {
				if (this != &o) {
					if (pool_) pool_->recycle(root_);
					pool_ = std::move(o.pool_);
					root_ = o.root_;
					o.root_ = 0;
				}
				return *this;
			}

			/** @brief Recycle this handle's nodes into the shared pool. @note Not constexpr (uses a shared_ptr node pool). */
			~splay_sequence_impl() {
				if (pool_) pool_->recycle(root_);
			}

			/** @brief Number of elements in this handle; zero for a moved-from handle. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			std::size_t size() const noexcept { return pool_ ? pool_->size(root_) : 0; }

			/** @brief Whether this handle contains no elements. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			bool empty() const noexcept { return size() == 0; }

			/**
			 * @brief Insert @p x before position @p i. @pre `i <= size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void insert(std::size_t i, T x) {
				atl_assert(pool_ && i <= size());
				root_ = Engine::insert_at(*pool_, root_, uint32(i), pool_->make(std::move(x)));
			}

			/**
			 * @brief Remove the element at @p i. @pre `i < size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void erase(std::size_t i) { erase(i, i + 1); }

			/**
			 * @brief Remove `[l, r)`. @pre `l <= r <= size()`
			 *        @complexity amortized O(log n) + O(r - l)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void erase(std::size_t l, std::size_t r) {
				atl_assert(pool_ && l <= r && r <= size());
				if (l == r) return;
				range_handle const h = isolate(uint32(l), uint32(r));
				if (h.upper != 0)
					pool_->set_child(h.upper, 0, 0);
				else if (h.lower != 0)
					pool_->set_child(h.lower, 1, 0);
				else
					root_ = 0;
				settle(h);
				pool_->recycle(h.middle);
			}

			/** @brief Append @p x. @complexity amortized O(log n), O(1) for a run of appends @note Not constexpr (uses a shared_ptr node pool). */
			void push_back(T x) {
				atl_assert(pool_);
				root_ = Engine::join(*pool_, root_, pool_->make(std::move(x)));
			}

			/** @brief Prepend @p x. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			void push_front(T x) {
				atl_assert(pool_);
				root_ = Engine::join(*pool_, pool_->make(std::move(x)), root_);
			}

			/**
			 * @brief Insert `T(args...)` before position @p i: built once from
			 *        @p args and moved into the pool's node (no copy, one move).
			 * @pre `i <= size()` @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename... Args>
				requires std::constructible_from<T, Args...>
			void emplace(std::size_t i, Args &&...args) {
				atl_assert(pool_ && i <= size());
				root_ = Engine::insert_at(*pool_, root_, uint32(i), pool_->make(std::forward<Args>(args)...));
			}

			/** @brief Append `T(args...)`. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			template<typename... Args>
				requires std::constructible_from<T, Args...>
			void emplace_back(Args &&...args) {
				atl_assert(pool_);
				root_ = Engine::join(*pool_, root_, pool_->make(std::forward<Args>(args)...));
			}

			/** @brief Prepend `T(args...)`. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			template<typename... Args>
				requires std::constructible_from<T, Args...>
			void emplace_front(Args &&...args) {
				atl_assert(pool_);
				root_ = Engine::join(*pool_, pool_->make(std::forward<Args>(args)...), root_);
			}

			/** @brief Remove the last element. @pre `!empty()`. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			void pop_back() { erase(size() - 1); }

			/** @brief Remove the first element. @pre `!empty()`. @complexity amortized O(log n) @note Not constexpr (uses a shared_ptr node pool). */
			void pop_front() { erase(0); }

			/**
			 * @brief The value at @p i; splays it to the root (hence non-const).
			 * @pre `i < size()` @complexity amortized O(log n)
			 * @note The reference is invalidated by any node allocation on a
			 *       handle sharing this pool (see the class note).
			 */
			[[nodiscard]] T const &at(std::size_t i)
				requires(!has_tags)
			{
				atl_assert(pool_ && i < size());
				return pool_->value(access(uint32(i)));
			}

			/**
			 * @brief The value at @p i, by value as on the treap's tagged tier; splays
			 *        it to the root. @pre `i < size()` @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			[[nodiscard]] T at(std::size_t i)
				requires(has_tags)
			{
				atl_assert(pool_ && i < size());
				return pool_->value(access(uint32(i)));
			}

			/**
			 * @brief Assign @p x to position @p i. @pre `i < size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void set(std::size_t i, T x) {
				atl_assert(pool_ && i < size());
				root_ = Engine::splay_at(*pool_, root_, uint32(i));
				pool_->value(root_) = std::move(x);
				pool_->pull(root_);
			}

			/**
			 * @brief Fold of `[l, r)`, or `s_identity` if empty; two splays, hence
			 *        non-const. @pre `l <= r <= size()` @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			[[nodiscard]] T fold(std::size_t l, std::size_t r)
				requires(has_op)
			{
				atl_assert(pool_ && l <= r && r <= size());
				if (l == r) return pool_->element_identity;
				return pool_->aggregate(isolate(uint32(l), uint32(r)).middle);
			}

			/** @brief Fold of the whole sequence. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
			[[nodiscard]] T fold() const
				requires(has_op)
			{
				atl_assert(pool_);
				return pool_->aggregate(root_); // the null sentinel holds the identity
			}

			/**
			 * @brief Apply tag @p f to every element of `[l, r)`. @pre `l <= r <= size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			void apply(std::size_t l, std::size_t r, FArg const &f) {
				atl_assert(pool_ && l <= r && r <= size());
				if (l == r) return;
				range_handle const h = isolate(uint32(l), uint32(r));
				pool_->apply_tag(h.middle, f);
				settle(h);
			}

			/**
			 * @brief Reverse the order of `[l, r)`. @pre `l <= r <= size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void reverse(std::size_t l, std::size_t r)
				requires(IsReversible)
			{
				atl_assert(pool_ && l <= r && r <= size());
				if (r - l < 2) return;
				range_handle const h = isolate(uint32(l), uint32(r));
				pool_->flip(h.middle);
				settle(h);
			}

			/**
			 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires
			 *        `pred` monotone and `pred(s_identity)`. @pre `l <= size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename Pred>
				requires std::predicate<Pred &, T const &>
			[[nodiscard]] std::size_t max_right(std::size_t l, Pred pred)
				requires(has_op)
			{
				atl_assert(pool_ && l <= size() && std::invoke(pred, std::as_const(pool_->element_identity)));
				if (l == size()) return l;
				pool &p = *pool_;
				uint32 lower = 0, rest = root_;
				if (l > 0) {
					root_ = lower = Engine::splay_at(p, root_, uint32(l - 1));
					rest = p.child(lower, 1);
					p.set_parent(rest, 0);
				}
				auto const [top, length] = Engine::max_right(p, rest, p.element_identity, std::as_const(p.op), pred);
				if (lower != 0) {
					p.set_child(lower, 1, top);
					p.set_parent(top, lower);
				} else {
					root_ = top;
				}
				return l + length;
			}

			/**
			 * @brief Detach and return `[i, size())`; this sequence keeps `[0, i)`. The
			 *        two share the pool. @pre `i <= size()`
			 *        @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			[[nodiscard]] Derived split(std::size_t i) {
				atl_assert(pool_ && i <= size());
				auto const [l, r] = Engine::split_at(*pool_, root_, uint32(i));
				root_ = l;
				return Derived(splay_sequence_adopt_t{}, pool_, r);
			}

			/**
			 * @brief An empty sequence sharing this one's pool: a valid `concat`
			 *        partner, in O(1).
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			[[nodiscard]] Derived sibling() const {
				atl_assert(pool_ != nullptr);
				return Derived(splay_sequence_adopt_t{}, pool_, 0);
			}

			/**
			 * @brief Append @p other (emptied by the call). @pre Distinct source on the same pool (a sequence
			 *        derived from this one via split/copy). @complexity amortized O(log n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			void concat(Derived &&other) {
				splay_sequence_impl &o = other;
				atl_assert(this != &o && pool_ && o.pool_ == pool_);
				root_ = Engine::join(*pool_, root_, o.root_);
				o.root_ = 0;
			}

			/**
			 * @brief In-order visit of every element's current value, settling nothing
			 *        (pending tags are folded in on the fly). @complexity O(n)
			 * @note Not constexpr (uses a shared_ptr node pool).
			 */
			template<typename Visit>
				requires std::invocable<Visit &, T const &>
			void for_each(Visit visit) const {
				if (pool_ && root_ != 0) visit_values(root_, visit);
			}

			/** @brief Empty the sequence, recycling its nodes. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
			void clear() {
				if (pool_) pool_->recycle(root_);
				root_ = 0;
			}

		private:
			/**
			 * @brief Splay rank @p i up (or, for an engine with a read-only access,
			 *        as far as it likes) and return its node.
			 */
			uint32 access(uint32 i) {
				if constexpr (requires { Engine::read_at(*pool_, root_, i); }) {
					auto const [root, x] = Engine::read_at(*pool_, root_, i);
					root_ = root;
					return x;
				} else {
					root_ = Engine::splay_at(*pool_, root_, i);
					return root_;
				}
			}
		};

		/** @brief The library node layout of `atl::splay_sequence`. */
		template<typename T, typename Op, typename F, bool IsReversible>
		using splay_sequence_default_layout = splay_layout_interleaved;
	}

	/**
	 * @brief Implicit splay tree: a sequence with amortized O(log n) surgery
	 *        anywhere, the drop-in counterpart of `treap_sequence`.
	 *
	 * The same template, tiers, members and deduction guides as
	 * `treap_sequence`, so switching is a type change. A splay tree moves every
	 * accessed position to the root, which wins where accesses are local
	 * (cursor editing, sequential scans, a hot range: the working-set and
	 * sequential-access bounds) and loses where they are uniformly random.
	 * It is deterministic: no seed.
	 *
	 * Differences from `treap_sequence`, stated up front: `at`, `fold(l, r)`
	 * and `max_right` restructure the tree and are therefore non-const;
	 * `fold()`, `size`, `empty` and `for_each` stay const. A copy is rebuilt
	 * perfectly balanced with every pending tag settled.
	 *
	 * Sequences derived from one origin (via `split` / copies) share a node
	 * pool; each handle owns its nodes (destruction recycles them). `concat`
	 * requires the shared pool. A moved-from handle is empty and only good for
	 * destruction or assignment.
	 *
	 * @note References returned by `at` point into the shared pool's node
	 *       storage: any call that allocates a node on this handle **or any
	 *       handle sharing its pool** may dangle them. Copy the value out
	 *       before mutating.
	 *
	 * @note Constructor algebra arguments are non-deduced; const-qualified callable
	 *       types accept ordinary callable arguments. Defaulted template parameters
	 *       keep disabled tiers from declaring void parameters.
	 *
	 * @tparam T            The element type.
	 * @tparam Op           Optional element fold: any `op(T, T)` convertible to `T`.
	 * @tparam F            Optional tag type applied to ranges (requires `Op`).
	 * @tparam Mapping      Tag application `mapping(F, T)` (with `F`).
	 * @tparam Compose      Tag composition `compose(F, F)`, newer first (with `F`).
	 * @tparam IsReversible Whether `reverse(l, r)` exists.
	 * @tparam Layout       The node layout: `splay_layout_interleaved`
	 *                      (default; links and payload in one record) or
	 *                      `splay_layout_split` (links and payload in two arrays;
	 *                      measurements in docs/splay-sequence.md).
	 */
	template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void,
			 bool IsReversible = false, typename Layout = detail::splay_sequence_default_layout<T, Op, F, IsReversible>>
		requires detail::splay_sequence_algebra<T, Op, F, Mapping, Compose> && detail::splay_selectable_layout<Layout>
	class splay_sequence : public detail::splay_sequence_impl<splay_sequence<T, Op, F, Mapping, Compose, IsReversible, Layout>,
															  splay_engine, Layout, T, Op, F, Mapping, Compose, IsReversible> {
		using base = detail::splay_sequence_impl<splay_sequence, splay_engine, Layout, T, Op, F, Mapping, Compose, IsReversible>;

	public:
		/** @brief Inherit the tier-specific constructors. */
		using base::base;
	};

	/** @brief The reversible tier: same class, `reverse(l, r)` enabled. */
	template<typename T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void,
			 detail::splay_selectable_layout Layout = detail::splay_sequence_default_layout<T, Op, F, true>>
	using reversible_splay_sequence = splay_sequence<T, Op, F, Mapping, Compose, true, Layout>;

	/** @brief Deduce the plain element type from a sized range. */
	// Deduction guides, as treap_sequence's (always the non-reversible tier).
	template<std::ranges::sized_range R>
	splay_sequence(R &&) -> splay_sequence<std::ranges::range_value_t<R>>;
	/** @brief Deduce the plain element type from the fill value. */
	template<typename T>
	splay_sequence(std::size_t, T) -> splay_sequence<T>;
	/** @brief Deduce the element and fold types from a count and identity. */
	template<typename T, binary_operation<T> Op>
	splay_sequence(std::size_t, T, Op) -> splay_sequence<T, Op>;
	/** @brief Deduces `T` from the identity; rejected if that would narrow the range's values. */
	template<std::ranges::sized_range R, typename T, binary_operation<T> Op>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	splay_sequence(R &&, T, Op) -> splay_sequence<T, Op>;
	/** @brief Deduce a tagged sequence from a count and both identities. */
	template<typename T, typename F, binary_operation<T> Op, action<F, T> Mapping, binary_operation<F> Compose>
	splay_sequence(std::size_t, T, F, Op, Mapping, Compose) -> splay_sequence<T, Op, F, Mapping, Compose>;
	/** @brief Deduce a tagged sequence from a range, rejecting narrowing to the identity type. */
	template<std::ranges::sized_range R, typename T, typename F, binary_operation<T> Op, action<F, T> Mapping,
			 binary_operation<F> Compose>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	splay_sequence(R &&, T, F, Op, Mapping, Compose) -> splay_sequence<T, Op, F, Mapping, Compose>;
}
