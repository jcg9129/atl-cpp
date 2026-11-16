#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/detail/radix.hpp>
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
		 * @brief Shared node pool underlying both persistent array variants.
		 *
		 * Holds two node arenas (values live in @ref leaves, child ids in
		 * @ref branches — see @ref persistent_arena), the shared radix index math,
		 * and the O(n) bottom-up build.
		 * The two persistence strategies differ only in the node payload and the
		 * update logic, so those are left to the derived pool.
		 */
		template<std::semiregular T, std::size_t Radix, std::movable Leaf, std::movable Branch>
			requires valid_radix<Radix>
		struct trie_node_pool {
			using radix = radix_index<Radix>;

			using id = uint32;

			/** @brief Leaf nodes holding the values; no sentinel, so every id is a live node. */
			persistent_arena<Leaf> leaves;
			/** @brief Internal nodes holding child ids into the level below. */
			persistent_arena<Branch> branches;
			/** @brief Number of internal levels above the leaves (0: the root is a leaf). */
			std::size_t height = 0;
			/** @brief Number of elements, shared by every version built from this pool. */
			std::size_t size = 0;

			/**
			 * @brief Build the trie bottom-up from a range of values.
			 *
			 * Packs the values into leaves, then repeatedly groups a level's ids into
			 * parent branch nodes. `make_leaf`/`make_branch` wrap a filled cell/child
			 * array into the derived pool's node type.
			 * @complexity O(n + Radix) for variable Radix; O(n + 1) for fixed Radix.
			 */
			template<typename MakeLeaf, typename MakeBranch, std::ranges::sized_range Range>
				requires std::ranges::input_range<Range>
					  && std::convertible_to<detail::forwarded_element_t<Range>, T>
					  && std::invocable<MakeLeaf &, std::array<T, Radix> &&>
					  && std::invocable<MakeBranch &, std::array<id, Radix> &&>
					  && std::convertible_to<std::invoke_result_t<MakeLeaf &, std::array<T, Radix> &&>, Leaf>
					  && std::convertible_to<std::invoke_result_t<MakeBranch &, std::array<id, Radix> &&>, Branch>
			id build(Range &&values, MakeLeaf make_leaf, MakeBranch make_branch) {
				auto const element_count = std::ranges::size(values);
				if constexpr (sizeof(element_count) > sizeof(std::size_t))
					atl_assert(element_count <= (std::numeric_limits<std::size_t>::max)());
				size = static_cast<std::size_t>(element_count);
				height = 0;
				if (size == 0) return 0;
				height = radix::height_for(size);
				// every level's node count is known up front: reserve it all, so no
				// leaf is relocated by arena growth during the build
				std::size_t const leaf_count = size / Radix + std::size_t(size % Radix != 0);
				leaves.reserve_extra(leaf_count);
				branches.reserve_extra(leaf_count / (Radix - 1) + height);
				std::vector<id> below;
				below.reserve(leaf_count);
				auto it = std::ranges::begin(values);
				auto last = std::ranges::end(values);
				while (it != last) {
					std::array<T, Radix> cell{};
					std::size_t count = 0;
					for (; count < Radix && it != last; ++count, ++it)
						cell[count] = static_cast<T>(detail::forward_element<Range>(*it));
					below.push_back(leaves.make(std::invoke(make_leaf, std::move(cell))));
				}
				for (std::size_t level = 1; level <= height; ++level) {
					std::vector<id> parents;
					parents.reserve(below.size() / Radix + std::size_t(below.size() % Radix != 0));
					for (std::size_t i = 0; i < below.size(); i += Radix) {
						std::array<id, Radix> child{};
						std::size_t count = 0;
						for (; count < Radix && i + count < below.size(); ++count) child[count] = below[i + count];
						parents.push_back(branches.make(std::invoke(make_branch, std::move(child))));
					}
					below = std::move(parents);
				}
				return below[0];
			}

			/**
			 * @brief Reserve room for `updates` further point-updates, so the pools
			 *        do not reallocate during that many updates.
			 */
			void reserve(std::size_t updates) {
				std::size_t const levels = height ? height : 1;
				atl_assert(updates <= (std::numeric_limits<std::size_t>::max)() / levels);
				leaves.reserve_extra(updates);
				branches.reserve_extra(updates * levels);
			}
		};
	}

	/**
	 * @brief Fully persistent immutable array (a bit-partitioned radix trie).
	 *
	 * An immutable random-access sequence: `set` returns a new version sharing all
	 * untouched structure with the old one (path copying), and *every* version
	 * stays queryable and updatable — you may branch a new version off any past
	 * one. Versions derived from one origin share a single arena (a pool of nodes
	 * addressed by uint32 ids); a version is a cheap `shared_ptr` + id, and the
	 * whole arena is freed once the last version referencing it dies. Individual
	 * unreachable nodes are not reclaimed — a good fit for keep-all workloads.
	 *
	 * @tparam T     The element type.
	 * @tparam Radix The trie branching factor (power of two). The default 16 makes
	 *               an internal node exactly one 64-byte cache line with 32-bit child ids. Larger radices copy more slots per update.
	 *
	 * @note Not constexpr (shared ownership uses runtime-only std::shared_ptr).
	 * @pre All node ids must fit uint32 (asserted on append). Reserve arithmetic
	 *      must fit size_t and the arenas' max_size() (asserted).
	 * @note Concurrent updates to any shared pool require external synchronization.
	 * @note `operator[]` returns a reference into the arena; like std::vector, it
	 *       may be invalidated by `set` or `reserve` on any version sharing the arena.
	 *
	 * @ref semi_persistent_array supports a linear update history with lower
	 * amortized allocation, while keeping older versions readable.
	 *
	 */
	template<std::semiregular T, std::size_t Radix = 16>
		requires detail::valid_radix<Radix>
	class persistent_array {
		struct leaf {
			std::array<T, Radix> value;
		};

		struct branch {
			std::array<uint32, Radix> child;
		};

		struct pool : detail::trie_node_pool<T, Radix, leaf, branch> {
			using base = detail::trie_node_pool<T, Radix, leaf, branch>;
			using typename base::id;
			using typename base::radix;

			template<std::ranges::sized_range Range>
				requires std::ranges::input_range<Range> && std::convertible_to<detail::forwarded_element_t<Range>, T>
			id build(Range &&values) {
				return base::build(
					std::forward<decltype(values)>(values), [](std::array<T, Radix> &&cell) { return leaf{std::move(cell)}; },
					[](std::array<id, Radix> &&child) { return branch{child}; });
			}

			leaf const &leaf_at(id root, std::size_t index) const {
				id node = root;
				for (std::size_t level = this->height; level >= 1; --level)
					node = this->branches[node].child[radix::digit(index, level)];
				return this->leaves[node];
			}

			T const &at(id root, std::size_t index) const { return leaf_at(root, index).value[index & radix::mask]; }

			// `value` is threaded by rvalue to the target leaf, where it is moved
			// into the slot; the rebuilt node is moved into the pool. Only the path
			// down to the target is visited, so the value is consumed exactly once.
			id set(id node, std::size_t level, std::size_t index, T &&value) {
				if (level == 0) {
					leaf copy = this->leaves[node];
					copy.value[index & radix::mask] = std::move(value);
					return this->leaves.make(std::move(copy));
				}
				std::size_t digit = radix::digit(index, level);
				id child = set(this->branches[node].child[digit], level - 1, index, std::move(value));
				branch copy = this->branches[node];
				copy.child[digit] = child;
				return this->branches.make(std::move(copy));
			}
		};

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		persistent_array(std::shared_ptr<pool> shared, uint32 root) : pool_(std::move(shared)), root_(root) {}

		persistent_array set_in(std::shared_ptr<pool> shared, std::size_t index, T value) const {
			atl_assert(index < shared->size);
			uint32 root = shared->set(root_, shared->height, index, std::move(value));
			return persistent_array(std::move(shared), root);
		}

	public:
		/** @brief An empty array.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		persistent_array() : pool_(std::make_shared<pool>()) {}

		/**
		 * @brief Build from `values`.
		 * @param values Sized readable range; owning rvalues donate elements, views do not.
		 * @tparam Range Sized input range with elements convertible to T after forwarding.
		 * @pre The range size fits size_t (asserted before conversion).
		 * @complexity O(n + Radix) for variable Radix; O(n + 1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		template<std::ranges::sized_range Range>
			requires std::ranges::input_range<Range> && std::convertible_to<detail::forwarded_element_t<Range>, T>
		explicit persistent_array(Range &&values) : pool_(std::make_shared<pool>()) {
			root_ = pool_->build(std::forward<Range>(values));
		}

		/**
		 * @brief An array of `n` copies of `value`.
		 * @param n Number of elements.
		 * @param value Initial value of every element.
		 * @complexity O(n + Radix) for variable Radix; O(n + 1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		explicit persistent_array(std::size_t n, T value = T{}) : persistent_array(std::vector<T>(n, std::move(value))) {}

		/** @brief Number of elements in every version of this family.
		 * @note Not constexpr (shared_ptr access). */
		std::size_t size() const noexcept { return pool_->size; }

		/** @brief Whether this version has no elements.
		 * @note Not constexpr (shared_ptr access). */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The element at `index`.
		 * @param index Element index.
		 * @pre `index < size()` (asserted).
		 * @return Reference into the shared arena; may be invalidated by updates or reservation.
		 * @complexity O(1 + log_Radix n)
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] T const &operator[](std::size_t index) const {
			atl_assert(index < size());
			return pool_->at(root_, index);
		}

		/**
		 * @brief Leaf-caching forward iterator: one root-to-leaf descent per
		 *        `Radix` elements. A full scan costs O(n + ceil(n / Radix) * log_Radix n).
		 * @note Points into the arena like `operator[]`, so it may be invalidated by
		 *       `set` or `reserve` on any version sharing the arena.
		 * @pre Compare only iterators of the same version, or two value-initialized iterators.
		 * @note Iterators do not own the pool; at least one family version must outlive them.
		 * @note Not constexpr (the pool lookup is runtime-only).
		 */
		class iterator {
			pool const *pool_ = nullptr;
			uint32 root_ = 0;
			std::size_t index_ = 0;
			T const *leaf_ = nullptr;

			friend class persistent_array;

			iterator(pool const *p, uint32 root, std::size_t index) : pool_(p), root_(root), index_(index) {
				if (index_ < pool_->size) leaf_ = pool_->leaf_at(root_, index_).value.data();
			}

		public:
			/** @brief Standard forward iterator category. */
			using iterator_category = std::forward_iterator_tag;
			/** @brief Element value type. */
			using value_type = T;
			/** @brief Signed distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Read-only element reference. */
			using reference = T const &;
			/** @brief Read-only element pointer. */
			using pointer = T const *;

			/** @brief Construct a singular iterator. */
			iterator() = default;

			/** @brief Read the current element. @pre This iterator is dereferenceable.
			 * @note Not constexpr (iterator operations use runtime pool access). */
			[[nodiscard]] reference operator*() const { return leaf_[index_ & (Radix - 1)]; }

			/** @brief Point to the current element. @pre This iterator is dereferenceable.
			 * @note Not constexpr (iterator operations use runtime pool access). */
			[[nodiscard]] pointer operator->() const { return &**this; }

			/** @brief Advance one position. @pre This iterator precedes end().
			 * @note Not constexpr (iterator operations use runtime pool access). */
			iterator &operator++() {
				++index_;
				if ((index_ & (Radix - 1)) == 0 && index_ < pool_->size)
					leaf_ = pool_->leaf_at(root_, index_).value.data();
				return *this;
			}

			/** @brief Return the old position and advance. @pre This iterator precedes end().
			 * @note Not constexpr (iterator operations use runtime pool access). */
			iterator operator++(int) {
				iterator old = *this;
				++*this;
				return old;
			}

			/** @brief Iterators of one version compare by position.
			 * @param o Iterator from the same version, or singular when this one is singular.
			 * @pre Both iterators belong to one version, or both are value-initialized.
			 * @note Not constexpr (the iterator API follows runtime-only pool access).
			 */
			[[nodiscard]] bool operator==(iterator const &o) const { return index_ == o.index_; }
		};

		/** @brief The first element.
		 * @return Iterator at index zero, or end() when empty.
		 * @complexity O(1 + log_Radix n)
		 * @note Not constexpr (shared_ptr access and pool lookup).
		 */
		[[nodiscard]] iterator begin() const { return iterator(pool_.get(), root_, 0); }

		/** @brief Past the last element.
		 * @return Iterator at index size().
		 * @complexity O(1)
		 * @note Not constexpr (shared_ptr access).
		 */
		[[nodiscard]] iterator end() const { return iterator(pool_.get(), root_, size()); }

		/**
		 * @brief The version with element `index` replaced by `value`.
		 * @param index Element index.
		 * @param value Replacement value.
		 * @pre `index < size()` (asserted).
		 * @return New version sharing the family pool.
		 * @complexity O(Radix * (1 + log_Radix(n)))
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] persistent_array set(std::size_t index, T value) const & {
			return set_in(pool_, index, std::move(value));
		}

		/**
		 * @brief `set` consuming this version: the pool is moved, not copied (no atomic
		 *        refcount traffic), so `a = std::move(a).set(i, x)` is the cheap linear idiom. The moved-from array may only be destroyed or assigned to.
		 * @param index Element index.
		 * @param value Replacement value.
		 * @pre `index < size()` (asserted).
		 * @return New version sharing the family pool.
		 * @complexity O(Radix * (1 + log_Radix(n)))
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] persistent_array set(std::size_t index, T value) && {
			return set_in(std::move(pool_), index, std::move(value));
		}

		/** @brief Reserve arena capacity for `updates` further set()s.
		 * @param updates Number of additional point updates, not elements or nodes.
		 * @pre Capacity arithmetic fits size_t and arena max_size() (asserted).
		 * @note May invalidate references and iterators from any family version.
		 * @complexity Linear in existing node storage if relocation occurs.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		void reserve(std::size_t updates) { pool_->reserve(updates); }
	};

	/** @brief Deduce the element type from a sized input range. @tparam Range Source range. */
	template<std::ranges::sized_range Range>
		requires std::ranges::input_range<Range>
	persistent_array(Range &&) -> persistent_array<std::ranges::range_value_t<Range>>;

	/**
	 * @brief Semi-persistent immutable array (DSST node-copying, one mod box).
	 *
	 * Same immutable random-access interface as @ref persistent_array, but every
	 * node carries one timestamped modification box, so the first change to a node
	 * writes the box with no allocation and only a second change copies it. Over a
	 * linear history this is O(1) amortised space per `set` instead of the trie's
	 * O(log n) (for fixed Radix).
	 *
	 * The price is a narrower contract: this is *partial* persistence. You may read
	 * any past version, but you may only `set` on the newest one — branching a new
	 * version off an older one is a precondition violation. Chain updates on the
	 * value returned by the previous `set`.
	 *
	 * @tparam T     The element type.
	 * @tparam Radix The trie branching factor (power of two), default 16.
	 *
	 * @note Not constexpr (shared ownership uses runtime-only std::shared_ptr).
	 * @note References may be invalidated by `set` or `reserve` on any shared version.
	 * @pre At most INT32_MAX successful updates per family (asserted). Node ids
	 *      must fit uint32; reserve arithmetic must fit size_t and max_size() (asserted).
	 * @note Nodes remain allocated until the last family handle dies. Concurrent
	 *       updates require external synchronization.
	 *
	 * The algorithm uses timestamped modification boxes (DSST node copying).
	 */
	template<std::semiregular T, std::size_t Radix = 16>
		requires detail::valid_radix<Radix>
	class semi_persistent_array {
		struct leaf {
			std::array<T, Radix> value;
			int32 box_version;
			uint16 box_field;
			T box_value;
		};

		struct branch {
			std::array<uint32, Radix> child;
			int32 box_version;
			uint16 box_field;
			uint32 box_value;
		};

		struct pool : detail::trie_node_pool<T, Radix, leaf, branch> {
			using base = detail::trie_node_pool<T, Radix, leaf, branch>;
			using typename base::id;
			using typename base::radix;

			int32 latest_version = 0;

			template<std::ranges::sized_range Range>
				requires std::ranges::input_range<Range> && std::convertible_to<detail::forwarded_element_t<Range>, T>
			id build(Range &&values) {
				latest_version = 0;
				return base::build(
					std::forward<decltype(values)>(values),
					[](std::array<T, Radix> &&cell) { return leaf{std::move(cell), -1, 0, T{}}; },
					[](std::array<id, Radix> &&child) { return branch{child, -1, 0, 0}; });
			}

			// The box overrides the stored field, but only for versions at or after
			// the box's timestamp; older versions see the original field.
			id child_at(id node, std::size_t digit, int32 version) const {
				branch const &b = this->branches[node];
				return (b.box_version >= 0 && b.box_field == digit && version >= b.box_version) ? b.box_value
																								: b.child[digit];
			}

			T const &value_at(id node, std::size_t digit, int32 version) const {
				leaf const &l = this->leaves[node];
				return (l.box_version >= 0 && l.box_field == digit && version >= l.box_version) ? l.box_value
																								: l.value[digit];
			}

			T const &at(id root, int32 version, std::size_t index) const {
				id node = root;
				for (std::size_t level = this->height; level >= 1; --level)
					node = child_at(node, radix::digit(index, level), version);
				return value_at(node, index & radix::mask, version);
			}

			id update_leaf(id node, std::size_t index, T &&value, int32 write_version) {
				std::size_t digit = index & radix::mask;
				if (this->leaves[node].box_version < 0) {
					this->leaves[node].box_version = write_version;
					this->leaves[node].box_field = uint16(digit);
					this->leaves[node].box_value = std::move(value);
					return node;
				}
				leaf copy = this->leaves[node];
				copy.value[copy.box_field] = copy.box_value;
				copy.value[digit] = std::move(value);
				copy.box_version = -1;
				return this->leaves.make(std::move(copy));
			}

			id update_branch(id node, std::size_t level, std::size_t index, T &&value, int32 write_version,
							 int32 read_version) {
				std::size_t digit = radix::digit(index, level);
				id old_child = child_at(node, digit, read_version);
				id new_child = level == 1
								 ? update_leaf(old_child, index, std::move(value), write_version)
								 : update_branch(old_child, level - 1, index, std::move(value), write_version, read_version);
				if (new_child == old_child) return node; // child boxed in place, no redirect
				if (this->branches[node].box_version < 0) {
					this->branches[node].box_version = write_version;
					this->branches[node].box_field = uint16(digit);
					this->branches[node].box_value = new_child;
					return node;
				}
				branch copy = this->branches[node];
				copy.child[copy.box_field] = copy.box_value;
				copy.child[digit] = new_child;
				copy.box_version = -1;
				return this->branches.make(std::move(copy));
			}

			std::pair<id, int32> set(id root, int32 from_version, std::size_t index, T &&value) {
				atl_assert(latest_version < (std::numeric_limits<int32>::max)());
				int32 write_version = latest_version + 1;
				id new_root = this->height
								? update_branch(root, this->height, index, std::move(value), write_version, from_version)
								: update_leaf(root, index, std::move(value), write_version);
				latest_version = write_version;
				return {new_root, write_version};
			}
		};

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;
		int32 version_ = 0;

		semi_persistent_array(std::shared_ptr<pool> shared, uint32 root, int32 version)
			: pool_(std::move(shared)), root_(root), version_(version) {}

		semi_persistent_array set_in(std::shared_ptr<pool> shared, std::size_t index, T value) const {
			atl_assert(index < shared->size);
			atl_assert(version_ == shared->latest_version);
			auto [root, version] = shared->set(root_, version_, index, std::move(value));
			return semi_persistent_array(std::move(shared), root, version);
		}

	public:
		/** @brief An empty array.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		semi_persistent_array() : pool_(std::make_shared<pool>()) {}

		/**
		 * @brief Build from `values`.
		 * @param values Sized readable range; owning rvalues donate elements, views do not.
		 * @tparam Range Sized input range with elements convertible to T after forwarding.
		 * @pre The range size fits size_t (asserted before conversion).
		 * @complexity O(n + Radix) for variable Radix; O(n + 1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		template<std::ranges::sized_range Range>
			requires std::ranges::input_range<Range> && std::convertible_to<detail::forwarded_element_t<Range>, T>
		explicit semi_persistent_array(Range &&values) : pool_(std::make_shared<pool>()) {
			root_ = pool_->build(std::forward<Range>(values));
		}

		/**
		 * @brief An array of `n` copies of `value`.
		 * @param n Number of elements.
		 * @param value Initial value of every element.
		 * @complexity O(n + Radix) for variable Radix; O(n + 1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		explicit semi_persistent_array(std::size_t n, T value = T{})
			: semi_persistent_array(std::vector<T>(n, std::move(value))) {}

		/** @brief Number of elements in every version of this family.
		 * @note Not constexpr (shared_ptr access). */
		std::size_t size() const noexcept { return pool_->size; }

		/** @brief Whether this version has no elements.
		 * @note Not constexpr (shared_ptr access). */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The element at `index`, as of this version.
		 * @param index Element index.
		 * @pre `index < size()` (asserted).
		 * @return Reference into the shared arena; may be invalidated by updates or reservation.
		 * @complexity O(1 + log_Radix n)
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] T const &operator[](std::size_t index) const {
			atl_assert(index < size());
			return pool_->at(root_, version_, index);
		}

		/**
		 * @brief The version with element `index` replaced by `value`.
		 * @pre This must be the newest version (partial persistence).
		 * @param index Element index.
		 * @param value Replacement value.
		 * @pre `index < size()` (asserted).
		 * @return New version sharing the family pool.
		 * @complexity O(Radix * (1 + log_Radix(n))) worst case, O(Radix) amortised space; O(1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] semi_persistent_array set(std::size_t index, T value) const & {
			return set_in(pool_, index, std::move(value));
		}

		/**
		 * @brief `set` consuming this version: the pool is moved, not copied. The moved-from array may only be destroyed or assigned to.
		 * @pre This must be the newest version (partial persistence).
		 * @param index Element index.
		 * @param value Replacement value.
		 * @pre `index < size()` (asserted).
		 * @return New version sharing the family pool.
		 * @complexity O(Radix * (1 + log_Radix(n))) worst case, O(Radix) amortised space; O(1) for fixed Radix.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		[[nodiscard]] semi_persistent_array set(std::size_t index, T value) && {
			return set_in(std::move(pool_), index, std::move(value));
		}

		/** @brief Reserve arena capacity for `updates` further set()s.
		 * @param updates Number of additional point updates, not elements or nodes.
		 * @pre Capacity arithmetic fits size_t and arena max_size() (asserted).
		 * @note May invalidate references and iterators from any family version.
		 * @complexity Linear in existing node storage if relocation occurs.
		 * @note Not constexpr (shared pool access and ownership are runtime-only).
		 */
		void reserve(std::size_t updates) { pool_->reserve(updates); }
	};

	/** @brief Deduce the element type from a sized input range. @tparam Range Source range. */
	template<std::ranges::sized_range Range>
		requires std::ranges::input_range<Range>
	semi_persistent_array(Range &&) -> semi_persistent_array<std::ranges::range_value_t<Range>>;
}
