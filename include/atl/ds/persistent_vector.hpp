#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/detail/radix.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief A version handle into a growable trie.
		 *
		 * Holds the trie @c root, the @c tail leaf carrying the last <= Radix
		 * elements, the logical @c size, and the trie @c height. @c version is used
		 * only by the DSST pool. @c is_strict marks a canonical radix trie (no relaxed
		 * nodes), letting push_back take a fast strict path.
		 */
		struct version_handle {
			uint32 root = 0;
			uint32 tail = 0;
			uint32 size = 0;
			std::size_t height = 0;
			int32 version = 0;
			bool is_strict = true;
		};

		/**
		 * @brief Elements held in the trie (the tail holds the rest): the largest
		 *        multiple of Radix that is < @p size. Used by the strict DSST pool.
		 */
		template<std::size_t Radix>
			requires valid_radix<Radix>
		constexpr uint32 trie_count(uint32 size) {
			return size == 0 ? 0 : uint32(((size - 1) / Radix) * Radix);
		}

		/**
		 * @brief Fully persistent growable vector: a hybrid RRB trie + tail.
		 *
		 * Nodes stay strict (radix-addressed, fast) until concat/slice relaxes them
		 * (size-table). Every op reduces to tailless-trie primitives
		 * (concat/take/drop) plus the tail.
		 */
		template<std::semiregular T, std::size_t Radix>
			requires valid_radix<Radix>
		struct rrb_tree_pool {
			using radix = radix_index<Radix>;
			using handle = version_handle;

			/** @brief A leaf: up to Radix values, the first @c count of them live. */
			struct leaf {
				std::array<T, Radix> value;
				uint32 count;
			};

			/**
			 * @brief An internal node: @c count children of total @c size elements.
			 *        A relaxed node is navigated by its cumulative @c size_table, a
			 *        strict one by radix shifts.
			 */
			struct branch {
				std::array<uint32, Radix> child;
				std::array<uint32, Radix> size_table; // cumulative sizes; used only when relaxed
				uint32 count;
				uint32 size;
				bool is_relaxed;
			};

			/** @brief A tailless subtree: root id, height and element count. */
			struct trie {
				uint32 root = 0;
				std::size_t height = 0;
				uint32 size = 0;
			};

			persistent_arena<leaf> leaves{leaf{}}; // id 0 = null sentinel
			persistent_arena<branch> branches{branch{}};

			/** @brief A handle with the given fields and @c version left at 0 (unused by this pool). */
			static handle make_handle(uint32 root, uint32 tail, uint32 size, std::size_t height, bool is_strict = true) {
				handle v;
				v.root = root;
				v.tail = tail;
				v.size = size;
				v.height = height;
				v.is_strict = is_strict;
				return v;
			}

			/** @brief Element count of the subtree rooted at @p id at level @p height. */
			uint32 subtree_size(uint32 id, std::size_t height) const {
				return height == 0 ? leaves[id].count : branches[id].size;
			}

			/** @brief A fresh leaf holding only @p x. */
			uint32 new_singleton_leaf(T &&x) {
				leaf nd{};
				nd.count = 1;
				nd.value[0] = std::move(x);
				return leaves.make(std::move(nd));
			}

			/** @brief A fresh leaf holding copies of the @p n values at @p src. */
			uint32 new_leaf(T const *src, uint32 n) {
				leaf nd{};
				nd.count = n;
				for (uint32 j = 0; j < n; ++j) nd.value[j] = src[j];
				return leaves.make(std::move(nd));
			}

			/** @brief As `new_leaf`, but moves from @p src, for buffers the caller owns. */
			uint32 new_leaf_move(T *src, uint32 n) {
				leaf nd{};
				nd.count = n;
				for (uint32 j = 0; j < n; ++j) nd.value[j] = std::move(src[j]);
				return leaves.make(std::move(nd));
			}

			/**
			 * @brief Create an internal node; auto-detects strictness (all-but-last
			 *        children full => radix-navigable) so untouched subtrees stay
			 *        fast.
			 */
			uint32 new_branch(uint32 const *ch, uint32 n, std::size_t height) {
				// uint32 sizes bound any reachable tree far below this; the shift
				// arithmetic below is 64-bit-safe only under it
				atl_assert(radix::bits * height < 64);
				branch nd{};
				nd.count = n;
				uint32 acc = 0;
				uint64 const full = uint64(1) << (radix::bits * height);
				bool is_strict = true;
				for (uint32 j = 0; j < n; ++j) {
					nd.child[j] = ch[j];
					uint32 s = subtree_size(ch[j], height - 1);
					acc += s;
					nd.size_table[j] = acc;
					if (j + 1 < n && s != full) is_strict = false;
				}
				nd.size = acc;
				nd.is_relaxed = !is_strict;
				return branches.make(std::move(nd));
			}

			/**
			 * @brief Index into a tailless trie: radix shift on strict nodes,
			 *        size-table search on relaxed ones.
			 * @complexity O(log n)
			 */
			T const &trie_get(uint32 id, std::size_t height, uint32 i) const {
				while (height > 0) {
					branch const &nd = branches[id];
					uint32 k;
					if (nd.is_relaxed) {
						k = 0;
						while (nd.size_table[k] <= i) ++k;
						if (k) i -= nd.size_table[k - 1];
					} else {
						k = uint32(uint64(i) >> (radix::bits * height));
						i -= uint32(uint64(k) << (radix::bits * height));
					}
					id = nd.child[k];
					--height;
				}
				return leaves[id].value[i];
			}

			/**
			 * @brief Path-copying point update of a tailless trie.
			 * @complexity O(log n)
			 */
			uint32 trie_set(uint32 id, std::size_t height, uint32 i, T &&x) {
				if (height == 0) {
					leaf nd = leaves[id];
					nd.value[i] = std::move(x);
					return leaves.make(std::move(nd));
				}
				uint32 k, sub;
				{
					branch const &cur = branches[id];
					if (cur.is_relaxed) {
						k = 0;
						while (cur.size_table[k] <= i) ++k;
						sub = k ? i - cur.size_table[k - 1] : i;
					} else {
						k = uint32(uint64(i) >> (radix::bits * height));
						sub = i - uint32(uint64(k) << (radix::bits * height));
					}
				}
				uint32 nc = trie_set(branches[id].child[k], height - 1, sub, std::move(x));
				branch nd = branches[id];
				nd.child[k] = nc;
				return branches.make(std::move(nd));
			}

			/**
			 * @brief Packs @p vals into full leaves (the last may be partial), moving
			 *        the elements out; every caller owns the buffer it passes.
			 */
			std::vector<uint32> chunk_leaves(std::vector<T> &vals) {
				std::vector<uint32> out;
				for (std::size_t i = 0; i < vals.size(); i += Radix)
					out.push_back(new_leaf_move(vals.data() + i, uint32(std::min<std::size_t>(Radix, vals.size() - i))));
				return out;
			}

			/** @brief Groups the level-(@p height - 1) ids @p ch into branches at level @p height. */
			std::vector<uint32> chunk_branches(std::vector<uint32> const &ch, std::size_t height) {
				std::vector<uint32> out;
				for (std::size_t i = 0; i < ch.size(); i += Radix)
					out.push_back(new_branch(ch.data() + i, uint32(std::min<std::size_t>(Radix, ch.size() - i)), height));
				return out;
			}

			/**
			 * @brief Concatenate two nodes at a level, returning compacted nodes
			 *        (the recursive core of concat).
			 */
			std::vector<uint32> concat_nodes(uint32 a, uint32 b, std::size_t height) {
				if (height == 0) {
					std::vector<T> vals;
					for (uint32 j = 0; j < leaves[a].count; ++j) vals.push_back(leaves[a].value[j]);
					for (uint32 j = 0; j < leaves[b].count; ++j) vals.push_back(leaves[b].value[j]);
					return chunk_leaves(vals);
				}
				std::vector<uint32> mid =
					concat_nodes(branches[a].child[branches[a].count - 1], branches[b].child[0], height - 1);
				std::vector<uint32> children;
				for (uint32 k = 0; k + 1 < branches[a].count; ++k) children.push_back(branches[a].child[k]);
				for (uint32 m : mid) children.push_back(m);
				for (uint32 k = 1; k < branches[b].count; ++k) children.push_back(branches[b].child[k]);
				return chunk_branches(children, height);
			}

			/** @brief A one-child branch above @p id, raising a subtree one level. */
			uint32 wrap_in_branch(uint32 id, std::size_t height) {
				uint32 one = id;
				return new_branch(&one, 1, height + 1);
			}

			/** @brief Strips single-child roots, so a trie's height is the least that holds it. */
			void canonicalize(uint32 &root, std::size_t &height) {
				while (height > 0 && branches[root].count == 1) root = branches[root].child[0], --height;
			}

			/** @brief Concatenate two tailless tries, O(log n). */
			trie trie_concat(trie a, trie b) {
				if (a.size == 0) return b;
				if (b.size == 0) return a;
				// sizes are 32-bit: v.concat(v) doubles in O(log n) nodes, so 32
				// doublings would wrap the size with no memory pressure at all
				atl_assert(uint64(a.size) + b.size <= (std::numeric_limits<uint32>::max)());
				uint32 ar = a.root, br = b.root;
				std::size_t ah = a.height, bh = b.height;
				while (ah < bh) ar = wrap_in_branch(ar, ah), ++ah;
				while (bh < ah) br = wrap_in_branch(br, bh), ++bh;
				std::vector<uint32> list = concat_nodes(ar, br, ah);
				trie r;
				if (list.size() == 1)
					r = {list[0], ah, a.size + b.size};
				else
					r = {new_branch(list.data(), uint32(list.size()), ah + 1), ah + 1, a.size + b.size};
				canonicalize(r.root, r.height);
				return r;
			}

			/**
			 * @brief Keep the first @p n elements of a subtree (the recursive core
			 *        of slice).
			 */
			uint32 take_node(uint32 id, std::size_t height, uint32 n) {
				if (height == 0) return new_leaf(leaves[id].value.data(), n);
				uint32 k, pre;
				{
					branch const &nd = branches[id];
					if (nd.is_relaxed) {
						k = 0;
						while (nd.size_table[k] < n) ++k;
						pre = k ? nd.size_table[k - 1] : 0;
					} else {
						k = uint32(uint64(n - 1) >> (radix::bits * height));
						pre = uint32(uint64(k) << (radix::bits * height));
					}
				}
				uint32 rem = n - pre;
				std::vector<uint32> kept;
				for (uint32 j = 0; j < k; ++j) kept.push_back(branches[id].child[j]);
				if (rem == subtree_size(branches[id].child[k], height - 1))
					kept.push_back(branches[id].child[k]);
				else
					kept.push_back(take_node(branches[id].child[k], height - 1, rem));
				return new_branch(kept.data(), uint32(kept.size()), height);
			}

			/**
			 * @brief Drop the first @p n elements of a subtree (the recursive core
			 *        of slice).
			 */
			uint32 drop_node(uint32 id, std::size_t height, uint32 n) {
				if (height == 0) {
					T tmp[Radix];
					for (uint32 j = n; j < leaves[id].count; ++j) tmp[j - n] = leaves[id].value[j];
					return new_leaf_move(tmp, leaves[id].count - n);
				}
				uint32 k, pre;
				{
					branch const &nd = branches[id];
					if (nd.is_relaxed) {
						k = 0;
						while (nd.size_table[k] <= n) ++k;
						pre = k ? nd.size_table[k - 1] : 0;
					} else {
						k = uint32(uint64(n) >> (radix::bits * height));
						pre = uint32(uint64(k) << (radix::bits * height));
					}
				}
				uint32 off = n - pre;
				std::vector<uint32> kept;
				kept.push_back(off == 0 ? branches[id].child[k] : drop_node(branches[id].child[k], height - 1, off));
				for (uint32 j = k + 1; j < branches[id].count; ++j) kept.push_back(branches[id].child[j]);
				return new_branch(kept.data(), uint32(kept.size()), height);
			}

			/**
			 * @brief The first @p n elements as a tailless trie.
			 * @complexity O(log n)
			 */
			trie take(trie t, uint32 n) {
				if (n == 0) return {};
				if (n >= t.size) return t;
				trie r{take_node(t.root, t.height, n), t.height, n};
				canonicalize(r.root, r.height);
				return r;
			}

			/**
			 * @brief All-but-first @p n elements as a tailless trie.
			 * @complexity O(log n)
			 */
			trie drop(trie t, uint32 n) {
				if (n == 0) return t;
				if (n >= t.size) return {};
				trie r{drop_node(t.root, t.height, n), t.height, t.size - n};
				canonicalize(r.root, r.height);
				return r;
			}

			/** @brief Bottom-up O(n) build of a tailless trie from a range; elements
			 *         are moved out of an rvalue range. */
			template<std::ranges::sized_range Range>
			trie trie_build(Range &&values) {
				atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
				std::vector<T> buf;
				buf.reserve(std::ranges::size(values));
				for (auto &&v : values) buf.push_back(static_cast<T>(detail::forward_element<Range>(std::forward<decltype(v)>(v))));
				if (buf.empty()) return {};
				std::vector<uint32> level = chunk_leaves(buf);
				std::size_t height = 0;
				while (level.size() > 1) level = chunk_branches(level, ++height);
				return {level[0], height, uint32(buf.size())};
			}

			/** @brief The trie part of @p v, without its tail. */
			trie trie_of(handle v) const { return {v.root, v.height, v.size - leaves[v.tail].count}; }

			/** @brief The tail leaf of @p v viewed as a height-0 trie. */
			trie tail_trie(handle v) const { return {v.tail, 0, leaves[v.tail].count}; }

			/** @brief All of @p v (trie followed by tail) as one tailless trie. */
			trie full_trie(handle v) { return trie_concat(trie_of(v), tail_trie(v)); }

			/**
			 * @brief Normalise a tailless trie into a (trie, tail) vector by pulling
			 *        its rightmost leaf into the tail.
			 */
			handle extract_tail(trie t, bool is_strict = true) {
				if (t.size == 0) return make_handle(0, 0, 0, 0, is_strict);
				uint32 id = t.root;
				for (uint32 h = t.height; h > 0; --h) id = branches[id].child[branches[id].count - 1];
				uint32 c = leaves[id].count;
				if (t.size == c) return make_handle(0, id, t.size, 0, is_strict);
				trie tt = take(t, t.size - c);
				return make_handle(tt.root, id, t.size, tt.height, is_strict);
			}

			/**
			 * @brief The leaf of @p v holding element @p i, with @p i's offset in it
			 *        written to @p offset (the iterator's leaf reload).
			 * @complexity O(log_Radix n)
			 */
			leaf const &leaf_at(handle v, uint32 i, uint32 &offset) const {
				uint32 const tc = v.size - leaves[v.tail].count;
				if (i >= tc) {
					offset = i - tc;
					return leaves[v.tail];
				}
				uint32 id = v.root;
				for (std::size_t height = v.height; height > 0; --height) {
					branch const &nd = branches[id];
					uint32 k;
					if (nd.is_relaxed) {
						k = 0;
						while (nd.size_table[k] <= i) ++k;
						if (k) i -= nd.size_table[k - 1];
					} else {
						k = uint32(uint64(i) >> (radix::bits * height));
						i -= uint32(uint64(k) << (radix::bits * height));
					}
					id = nd.child[k];
				}
				offset = i;
				return leaves[id];
			}

			/**
			 * @brief Room for @p pushes further `push_back`s onto a vector of
			 *        @p size: one tail leaf per push, and one spill path of at most
			 *        height + 2 branches per `Radix` pushes.
			 */
			void reserve(std::size_t size, std::size_t pushes) {
				atl_assert(pushes <= (std::numeric_limits<uint32>::max)() - size);
				leaves.reserve_extra(pushes);
				std::size_t const paths = pushes / Radix + 1;
				std::size_t const per_path = radix::height_for(size + pushes) + 2;
				atl_assert(paths <= (std::numeric_limits<std::size_t>::max)() / per_path);
				branches.reserve_extra(paths * per_path);
			}

			/**
			 * @brief Element at index @p i (tail fast-path, else trie).
			 * @complexity O(log_Radix n)
			 */
			T const &at(handle v, uint32 i) const {
				uint32 tc = v.size - leaves[v.tail].count;
				if (i >= tc) return leaves[v.tail].value[i - tc];
				return trie_get(v.root, v.height, i);
			}

			/**
			 * @brief Point update returning a new version.
			 * @complexity O(log_Radix n)
			 */
			handle set(handle v, uint32 i, T &&x) {
				uint32 tc = v.size - leaves[v.tail].count;
				if (i >= tc) {
					leaf nd = leaves[v.tail];
					nd.value[i - tc] = std::move(x);
					return make_handle(v.root, leaves.make(std::move(nd)), v.size, v.height, v.is_strict);
				}
				return make_handle(trie_set(v.root, v.height, i, std::move(x)), v.tail, v.size, v.height, v.is_strict);
			}

			/** @brief Build a single-child spine down to @p leafid (push_leaf helper). */
			uint32 build_spine(std::size_t level, uint32 leafid) {
				if (level == 0) return leafid;
				uint32 c = build_spine(level - 1, leafid);
				return new_branch(&c, 1, level);
			}

			/** @brief Path-copy @p leafid into position @p leaf_no (push_leaf helper). */
			uint32 append_leaf(uint32 node, std::size_t level, uint32 leaf_no, uint32 leafid) {
				uint32 d = uint32((uint64(leaf_no) >> (radix::bits * (level - 1))) & radix::mask);
				uint32 cnt = branches[node].count;
				uint32 kids[Radix];
				for (uint32 j = 0; j < cnt; ++j) kids[j] = branches[node].child[j];
				if (level == 1) {
					kids[d] = leafid;
					if (d == cnt) ++cnt;
				} else if (d < cnt) {
					kids[d] = append_leaf(kids[d], level - 1, leaf_no, leafid);
				} else {
					kids[d] = build_spine(level - 1, leafid);
					++cnt;
				}
				return new_branch(kids, cnt, level);
			}

			/**
			 * @brief Fast strict append of a full tail leaf into a canonical radix
			 *        trie (no relaxed nodes, no heap-vector churn) -- the common
			 *        push_back spill.
			 */
			std::pair<uint32, uint32> push_leaf(uint32 root, std::size_t height, uint32 tc, uint32 leafid) {
				if (tc == 0) return {leafid, 0};
				uint32 leaf_no = tc / Radix;
				if (uint64(leaf_no) == (uint64(1) << (radix::bits * height))) {
					uint32 kids[2] = {root, build_spine(height, leafid)};
					return {new_branch(kids, 2, height + 1), height + 1};
				}
				return {append_leaf(root, height, leaf_no, leafid), height};
			}

			/**
			 * @brief Append; O(Radix) amortised. Uses a fast strict append when the
			 *        vector is still strict, else the general concat spill.
			 */
			handle push_back(handle v, T &&x) {
				if (v.size == 0) return make_handle(0, new_singleton_leaf(std::move(x)), 1, 0);
				atl_assert(v.size < (std::numeric_limits<uint32>::max)());
				uint32 tc = leaves[v.tail].count;
				if (tc < Radix) {
					leaf nd = leaves[v.tail];
					nd.value[tc] = std::move(x);
					nd.count = tc + 1;
					return make_handle(v.root, leaves.make(std::move(nd)), v.size + 1, v.height, v.is_strict);
				}
				if (v.is_strict) {
					auto [nr, nh] = push_leaf(v.root, v.height, v.size - Radix, v.tail);
					return make_handle(nr, new_singleton_leaf(std::move(x)), v.size + 1, nh, true);
				}
				trie t = trie_concat(trie_of(v), tail_trie(v));
				return make_handle(t.root, new_singleton_leaf(std::move(x)), v.size + 1, t.height, false);
			}

			/**
			 * @brief Remove the last element (tail shrink, or pull the last trie
			 *        leaf back into the tail).
			 */
			handle pop_back(handle v) {
				uint32 tc = leaves[v.tail].count;
				if (tc > 1) {
					leaf nd = leaves[v.tail];
					nd.count = tc - 1;
					return make_handle(v.root, leaves.make(std::move(nd)), v.size - 1, v.height, v.is_strict);
				}
				if (v.size == 1) return make_handle(0, 0, 0, 0, v.is_strict);
				return extract_tail(trie_of(v), v.is_strict);
			}

			/** @brief O(n) build from a range. */
			template<std::ranges::sized_range Range>
			handle build(Range &&values) {
				return extract_tail(trie_build(std::forward<Range>(values)));
			}

			/** @brief The subrange [from, to), O(log n). */
			handle slice(handle v, uint32 from, uint32 to) {
				if (from >= to) return make_handle(0, 0, 0, 0);
				return extract_tail(drop(take(full_trie(v), to), from), false);
			}

			/** @brief Insert before index @p i (take ++ [x] ++ drop), O(log n). */
			handle insert(handle v, uint32 i, T &&x) {
				trie f = full_trie(v);
				trie mid{new_singleton_leaf(std::move(x)), 0, 1};
				return extract_tail(trie_concat(trie_concat(take(f, i), mid), drop(f, i)), false);
			}

			/** @brief Concatenate two vectors sharing this pool. */
			handle concat(handle a, handle b) {
				if (a.size == 0) return b;
				if (b.size == 0) return a;
				return extract_tail(trie_concat(full_trie(a), full_trie(b)), false);
			}

			/**
			 * @brief Deep-copy the subtree @p id of @p src at level @p height into
			 *        this pool, returning its new id (the recursive core of
			 *        `copy_from`), O(subtree size).
			 */
			uint32 copy_node_from(rrb_tree_pool const &src, uint32 id, std::size_t height) {
				if (height == 0) return new_leaf(src.leaves[id].value.data(), src.leaves[id].count);
				uint32 kids[Radix];
				uint32 n = src.branches[id].count;
				for (uint32 j = 0; j < n; ++j) kids[j] = copy_node_from(src, src.branches[id].child[j], height - 1);
				return new_branch(kids, n, height);
			}

			/**
			 * @brief Deep-copy another pool's vector into this one (for concat
			 *        across arenas), O(size).
			 */
			handle copy_from(rrb_tree_pool const &src, handle b) {
				uint32 tc = b.size - src.leaves[b.tail].count;
				uint32 root = tc == 0 ? 0 : copy_node_from(src, b.root, b.height);
				uint32 tail = new_leaf(src.leaves[b.tail].value.data(), src.leaves[b.tail].count);
				return make_handle(root, tail, b.size, b.height, b.is_strict);
			}
		};

		/**
		 * @brief Semi-persistent growable trie + tail via DSST node-copying.
		 *
		 * Appends into fresh territory (past a high-water mark) mutate in place with
		 * no allocation; a slot revisited after pop_back and every set are
		 * version-stamped in a mod box so past versions keep their value. Partial
		 * persistence (extend only the newest version).
		 */
		template<std::semiregular T, std::size_t Radix>
			requires valid_radix<Radix>
		struct semi_rrb_tree_pool {
			using radix = radix_index<Radix>;
			using handle = version_handle;

			/** @brief A full-width leaf plus one mod box overriding slot @c box_field from @c box_version on. */
			struct leaf {
				std::array<T, Radix> value;
				int32 box_version;
				uint16 box_field;
				T box_value;
			};

			/** @brief An internal node plus one mod box overriding child @c box_field from @c box_version on. */
			struct branch {
				std::array<uint32, Radix> child;
				int32 box_version;
				uint16 box_field;
				uint32 box_value;
			};

			/** @brief A fresh node: value-initialized, with its mod box marked empty. */
			template<typename Node>
				requires std::same_as<Node, leaf> || std::same_as<Node, branch>
			static Node make_blank() {
				Node nd{};
				nd.box_version = -1;
				return nd;
			}

			persistent_arena<leaf> leaves{make_blank<leaf>()}; // id 0 = null sentinel
			persistent_arena<branch> branches{make_blank<branch>()};
			int32 latest_version = 0;
			uint32 high_water = 0; // number of element slots ever written

			uint32 new_leaf() { return leaves.make(make_blank<leaf>()); }

			uint32 new_branch() { return branches.make(make_blank<branch>()); }

			/**
			 * @brief Box-aware read: the mod box overrides the stored field only for
			 *        versions >= its timestamp.
			 */
			T const &leaf_read(uint32 id, uint32 d, int32 version) const {
				leaf const &l = leaves[id];
				return (l.box_version >= 0 && l.box_field == d && version >= l.box_version) ? l.box_value : l.value[d];
			}

			/**
			 * @brief Box-aware read: the mod box overrides the stored field only for
			 *        versions >= its timestamp.
			 */
			uint32 branch_read(uint32 id, uint32 d, int32 version) const {
				branch const &b = branches[id];
				return (b.box_version >= 0 && b.box_field == d && version >= b.box_version) ? b.box_value : b.child[d];
			}

			/**
			 * @brief Element at index @p i as of this version (tail fast-path, else
			 *        trie).
			 * @complexity O(log_Radix n)
			 */
			T const &at(handle v, uint32 i) const {
				uint32 t = trie_count<Radix>(v.size);
				if (i >= t) return leaf_read(v.tail, uint32(i - t), v.version);
				uint32 node = v.root;
				for (std::size_t level = v.height; level >= 1; --level)
					node = branch_read(node, uint32(radix::digit(i, level)), v.version);
				return leaf_read(node, i & radix::mask, v.version);
			}

			/**
			 * @brief Store a child: fresh => write in place; else fill the one mod
			 *        box, copying on a second differing write.
			 */
			uint32 branch_set(uint32 id, uint32 d, uint32 val, int32 t, bool is_fresh) {
				if (is_fresh) {
					branches[id].child[d] = val;
					return id;
				}
				if (branches[id].box_version < 0) {
					branches[id].box_version = t;
					branches[id].box_field = uint16(d);
					branches[id].box_value = val;
					return id;
				}
				if (branches[id].box_field == d && branches[id].box_version == t) {
					branches[id].box_value = val;
					return id;
				}
				uint32 n = new_branch();
				branches[n].child = branches[id].child;
				branches[n].child[branches[id].box_field] = branches[id].box_value;
				branches[n].child[d] = val;
				return n;
			}

			/**
			 * @brief Store an element: fresh => write in place; else fill the one mod
			 *        box, copying on a second differing write.
			 */
			uint32 leaf_set(uint32 id, uint32 d, T &&val, int32 t, bool is_fresh) {
				if (is_fresh) {
					leaves[id].value[d] = std::move(val);
					return id;
				}
				if (leaves[id].box_version < 0) {
					leaves[id].box_version = t;
					leaves[id].box_field = uint16(d);
					leaves[id].box_value = std::move(val);
					return id;
				}
				if (leaves[id].box_field == d && leaves[id].box_version == t) {
					leaves[id].box_value = std::move(val);
					return id;
				}
				uint32 n = new_leaf();
				leaves[n].value = leaves[id].value;
				leaves[n].value[leaves[id].box_field] = leaves[id].box_value;
				leaves[n].value[d] = std::move(val);
				return n;
			}

			/** @brief A fresh single-child path of @p level branches down to @p leafid. */
			uint32 build_path(std::size_t level, uint32 leafid) {
				if (level == 0) return leafid;
				uint32 n = new_branch();
				branches[n].child[0] = build_path(level - 1, leafid);
				return n;
			}

			/**
			 * @brief Hangs @p leafid at leaf position @p leaf_no under @p node, stamping
			 *        each touched branch with version @p t; returns the node's
			 *        (possibly copied) id.
			 */
			uint32 insert_leaf(uint32 node, std::size_t level, uint32 leaf_no, uint32 leafid, int32 t) {
				uint32 d = uint32((uint64(leaf_no) >> (radix::bits * (level - 1))) & radix::mask);
				if (level == 1) return branch_set(node, d, leafid, t, false);
				uint32 child = branch_read(node, d, t);
				uint32 nc = child ? insert_leaf(child, level - 1, leaf_no, leafid, t) : build_path(level - 1, leafid);
				return branch_set(node, d, nc, t, false);
			}

			/**
			 * @brief Spills the full tail @p leafid into the trie of @p t elements,
			 *        growing a new root when the trie is full; returns {root, height}.
			 */
			std::pair<uint32, uint32> push_leaf(uint32 root, std::size_t height, uint32 t, uint32 leafid, int32 ver) {
				if (t == 0) return {leafid, 0};
				uint32 leaf_no = t / Radix;
				if (uint64(leaf_no) == (uint64(1) << (radix::bits * height))) {
					uint32 n = new_branch();
					branches[n].child[0] = root;
					branches[n].child[1] = build_path(height, leafid);
					return {n, height + 1};
				}
				return {insert_leaf(root, height, leaf_no, leafid, ver), height};
			}

			/**
			 * @brief Append; fresh territory mutates in place, else version-stamps.
			 * @complexity O(Radix) amortised
			 */
			handle push_back(handle v, T &&x) {
				atl_assert(v.size < (std::numeric_limits<uint32>::max)());
				atl_assert(latest_version < (std::numeric_limits<int32>::max)());
				int32 t = ++latest_version;
				bool is_fresh = v.size >= high_water;
				if (v.size >= high_water) high_water = v.size + 1;
				if (v.size == 0) {
					uint32 id = new_leaf();
					leaves[id].value[0] = std::move(x);
					return {0, id, 1, 0, t};
				}
				uint32 tc = trie_count<Radix>(v.size), ts = v.size - tc;
				if (ts < Radix) return {v.root, leaf_set(v.tail, ts, std::move(x), t, is_fresh), v.size + 1, v.height, t};
				auto [nr, nh] = push_leaf(v.root, v.height, tc, v.tail, v.version);
				uint32 id = new_leaf();
				leaves[id].value[0] = std::move(x);
				return {nr, id, v.size + 1, nh, t};
			}

			/** @brief The id of leaf number @p leaf_no as of @p version. */
			uint32 leaf_at(uint32 node, std::size_t height, uint32 leaf_no, int32 version) const {
				for (std::size_t level = height; level >= 1; --level)
					node = branch_read(node, uint32((uint64(leaf_no) >> (radix::bits * (level - 1))) & radix::mask),
									   version);
				return node;
			}

			/**
			 * @brief Unhooks leaf number @p last (as of @p version), stamping each
			 *        touched branch with @p t; returns the node's (possibly copied) id.
			 */
			uint32 remove_leaf(uint32 node, std::size_t level, uint32 last, int32 t, int32 version) {
				uint32 d = uint32((uint64(last) >> (radix::bits * (level - 1))) & radix::mask);
				if (level == 1) return branch_set(node, d, 0, t, false);
				uint32 nc = remove_leaf(branch_read(node, d, version), level - 1, last, t, version);
				return branch_set(node, d, nc, t, false);
			}

			/**
			 * @brief Remove the last element (tail shrink, or pull the last trie
			 *        leaf back into the tail).
			 */
			handle pop_back(handle v) {
				atl_assert(latest_version < (std::numeric_limits<int32>::max)());
				int32 t = ++latest_version;
				uint32 tc = trie_count<Radix>(v.size), ts = v.size - tc;
				if (ts > 1) return {v.root, v.tail, v.size - 1, v.height, t};
				if (tc == 0) return {0, 0, 0, 0, t};
				uint32 num_leaves = tc / Radix;
				uint32 newtail = leaf_at(v.root, v.height, num_leaves - 1, v.version);
				if (num_leaves == 1) return {0, newtail, v.size - 1, 0, t};
				uint32 nr = remove_leaf(v.root, v.height, num_leaves - 1, t, v.version);
				std::size_t nh = v.height, target = 0;
				for (uint32 cap = 1; cap < num_leaves - 1; cap *= Radix) ++target;
				while (nh > target) nr = branch_read(nr, 0, t), --nh;
				return {nr, newtail, v.size - 1, nh, t};
			}

			/**
			 * @brief Writes @p x at element @p i of the trie under @p node (read as of
			 *        @p version, stamped @p t); returns the node's (possibly copied) id.
			 */
			uint32 update_trie(uint32 node, std::size_t level, uint32 i, T &&x, int32 t, int32 version) {
				if (level == 0) return leaf_set(node, i & radix::mask, std::move(x), t, false);
				uint32 d = uint32(radix::digit(i, level));
				uint32 nc = update_trie(branch_read(node, d, version), level - 1, i, std::move(x), t, version);
				return branch_set(node, d, nc, t, false);
			}

			/**
			 * @brief Point update returning a new version.
			 * @complexity O(log_Radix n)
			 */
			handle set(handle v, uint32 i, T &&x) {
				atl_assert(latest_version < (std::numeric_limits<int32>::max)());
				int32 t = ++latest_version;
				uint32 tc = trie_count<Radix>(v.size);
				if (i >= tc)
					return {v.root, leaf_set(v.tail, uint32(i - tc), std::move(x), t, false), v.size, v.height, t};
				return {update_trie(v.root, v.height, i, std::move(x), t, v.version), v.tail, v.size, v.height, t};
			}

			/** @brief O(n) build from a range. */
			template<std::ranges::sized_range Range>
			handle build(Range &&values) {
				atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
				uint32 n = uint32(std::ranges::size(values));
				high_water = n;
				if (n == 0) return {};
				uint32 ts = ((n - 1) % Radix) + 1, trie_n = n - ts;
				auto it = std::ranges::begin(values);
				std::vector<uint32> below;
				for (uint32 done = 0; done < trie_n;) {
					uint32 id = new_leaf();
					for (std::size_t j = 0; j < Radix; ++j, ++it, ++done)
						leaves[id].value[j] = static_cast<T>(detail::forward_element<Range>(*it));
					below.push_back(id);
				}
				uint32 tail = new_leaf();
				for (uint32 j = 0; j < ts; ++j, ++it) leaves[tail].value[j] = static_cast<T>(detail::forward_element<Range>(*it));
				std::size_t height = 0;
				while (below.size() > 1) {
					++height;
					std::vector<uint32> parents;
					for (std::size_t i = 0; i < below.size(); i += Radix) {
						uint32 id = new_branch();
						for (std::size_t j = 0; j < Radix && i + j < below.size(); ++j) branches[id].child[j] = below[i + j];
						parents.push_back(id);
					}
					below = std::move(parents);
				}
				return {below.empty() ? 0 : below[0], tail, n, height, 0};
			}
		};
	}

	/**
	 * @brief Fully persistent growable vector (hybrid RRB trie + tail).
	 *
	 * The Clojure/immer vector representation extended to a relaxed radix balanced
	 * (RRB) tree: elements live in Radix-way leaves with the last partial block
	 * kept as a directly-accessible "tail", so `push_back`/`pop_back` usually touch
	 * only the tail. Nodes stay strict (radix-addressed, fast index) until
	 * `concat`/`slice`/`insert` touches them, when they become relaxed (size-table)
	 * — so an un-spliced vector keeps full radix speed. Fully persistent — every
	 * version stays queryable and updatable, branch a new one off any version.
	 * Versions share one arena (uint32-id nodes); a value is a cheap `shared_ptr` +
	 * a compact handle, and the arena is freed once the last version dies.
	 *
	 * @tparam T     The element type.
	 * @tparam Radix Branching factor, a power of two (default 16 = one cache-line
	 *               internal node and the measured balance of push/index/memory for
	 *               persistent use; 32 favours indexing, smaller favours push).
	 *
	 * @note Not constexpr (versions own a std::shared_ptr pool); this applies to all members. `operator[]`/`back` return a reference
	 *       into the arena, invalidated by a later mutation on any shared version.
	 * @note Constructing from a range builds in O(n) without per-element copies.
	 *
	 * @complexity push_back/pop_back O(Radix) amortised; operator[]/set O(log n);
	 *             concat/slice/insert O(log n) when the operands share an arena (as
	 *             they do after slicing one vector), else concat imports the other
	 *             in O(its size).
	 *
	 */
	template<std::semiregular T, std::size_t Radix = 16>
		requires detail::valid_radix<Radix>
	class persistent_vector {
		using pool = detail::rrb_tree_pool<T, Radix>;
		std::shared_ptr<pool> pool_;
		detail::version_handle handle_;

		persistent_vector(std::shared_ptr<pool> shared, detail::version_handle handle)
			: pool_(std::move(shared)), handle_(handle) {}

		/** @brief `concat` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector concat_in(std::shared_ptr<pool> shared, persistent_vector const &other) const {
			if (shared == other.pool_) {
				auto next = shared->concat(handle_, other.handle_);
				return {std::move(shared), next};
			}
			detail::version_handle imported = shared->copy_from(*other.pool_, other.handle_);
			auto next = shared->concat(handle_, imported);
			return {std::move(shared), next};
		}

		/** @brief `push_back` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector push_back_in(std::shared_ptr<pool> shared, T value) const {
			auto next = shared->push_back(handle_, std::move(value));
			return {std::move(shared), next};
		}

		/** @brief `pop_back` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector pop_back_in(std::shared_ptr<pool> shared) const {
			atl_assert(handle_.size > 0);
			auto next = shared->pop_back(handle_);
			return {std::move(shared), next};
		}

		/** @brief `set` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector set_in(std::shared_ptr<pool> shared, std::size_t i, T value) const {
			atl_assert(i < handle_.size);
			auto next = shared->set(handle_, uint32(i), std::move(value));
			return {std::move(shared), next};
		}

		/** @brief `slice` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector slice_in(std::shared_ptr<pool> shared, std::size_t from, std::size_t to) const {
			atl_assert(from <= to && to <= handle_.size);
			auto next = shared->slice(handle_, uint32(from), uint32(to));
			return {std::move(shared), next};
		}

		/** @brief `insert` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_vector insert_in(std::shared_ptr<pool> shared, std::size_t i, T value) const {
			atl_assert(i <= handle_.size);
			auto next = shared->insert(handle_, uint32(i), std::move(value));
			return {std::move(shared), next};
		}

	public:
		/** @brief An empty vector. */
		persistent_vector() : pool_(std::make_shared<pool>()) {}

		/**
		 * @brief Build from `values` in O(n), copying lvalues/views and moving owning rvalues or yielded prvalues.
		 */
		template<std::ranges::sized_range Range>
			requires std::convertible_to<detail::forwarded_element_t<Range>, T>
		explicit persistent_vector(Range &&values) : pool_(std::make_shared<pool>()) {
			handle_ = pool_->build(std::forward<Range>(values));
		}

		/**
		 * @brief A vector of `n` copies of `value`.
		 * @complexity O(n)
		 */
		explicit persistent_vector(std::size_t n, T value = T{}) : persistent_vector(std::vector<T>(n, std::move(value))) {}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return handle_.size; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return handle_.size == 0; }

		/**
		 * @brief The element at `i`.
		 * @complexity O(log_Radix n)
		 */
		[[nodiscard]] T const &operator[](std::size_t i) const {
			atl_assert(i < handle_.size);
			return pool_->at(handle_, uint32(i));
		}

		/** @brief The first element. @pre Non-empty. */
		[[nodiscard]] T const &front() const { return (*this)[0]; }

		/** @brief The last element. @pre Non-empty. */
		[[nodiscard]] T const &back() const { return (*this)[handle_.size - 1]; }

		/**
		 * @brief Leaf-caching forward iterator: one descent per leaf (relaxed
		 *        leaves may be partial), so a full scan is O(n) instead of the
		 *        O(n log n) of an index loop.
		 * @note Points into the arena like `operator[]`, so it is invalidated by
		 *       a later mutation on any version sharing the arena.
		 */
		class iterator {
			pool const *pool_ = nullptr;
			detail::version_handle handle_;
			uint32 index_ = 0;
			uint32 left_in_leaf_ = 0;
			T const *current_ = nullptr;

			friend class persistent_vector;

			iterator(pool const *p, detail::version_handle handle, uint32 index)
				: pool_(p), handle_(handle), index_(index) {
				load();
			}

			void load() {
				if (index_ >= handle_.size) return;
				uint32 offset;
				auto const &lf = pool_->leaf_at(handle_, index_, offset);
				current_ = lf.value.data() + offset;
				left_in_leaf_ = lf.count - offset;
			}

		public:
			/** @brief Forward traversal category. */
			using iterator_category = std::forward_iterator_tag;
			/** @brief Element type. */
			using value_type = T;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Read-only element reference. */
			using reference = T const &;
			/** @brief Read-only element pointer. */
			using pointer = T const *;

			/** @brief Singular iterator; assign a version iterator before dereferencing. */
			iterator() = default;

			/** @brief Refer to the current element. @pre Not past the end. */
			[[nodiscard]] reference operator*() const { return *current_; }

			/** @brief Point to the current element. @pre Not past the end. */
			[[nodiscard]] pointer operator->() const { return current_; }

			/** @brief Advance to the next element. @pre Not past the end. @return This iterator. */
			iterator &operator++() {
				++index_;
				++current_;
				if (--left_in_leaf_ == 0) load();
				return *this;
			}

			/** @brief Advance and return the unadvanced iterator. @pre Not past the end. */
			iterator operator++(int) {
				iterator old = *this;
				++*this;
				return old;
			}

			/** @brief Iterators of one version compare by position. */
			[[nodiscard]] bool operator==(iterator const &o) const { return index_ == o.index_; }
		};

		/** @brief The first element. @complexity O(log_Radix n) */
		[[nodiscard]] iterator begin() const { return iterator(pool_.get(), handle_, 0); }

		/** @brief Past the last element. @complexity O(1) */
		[[nodiscard]] iterator end() const { return iterator(pool_.get(), handle_, handle_.size); }

		/**
		 * @brief Element-wise equality; O(1) for two handles of one version.
		 * @complexity O(n + ceil(n/Radix) log_Radix n)
		 */
		[[nodiscard]] friend bool operator==(persistent_vector const &a, persistent_vector const &b)
			requires std::equality_comparable<T>
		{
			if (a.size() != b.size()) return false;
			if (a.pool_ == b.pool_ && a.handle_.root == b.handle_.root && a.handle_.tail == b.handle_.tail) return true;
			return std::ranges::equal(a, b);
		}

		/**
		 * @brief Reserve arena room for @p pushes further `push_back`s onto this
		 *        version, so a growth loop does not reallocate mid-run.
		 */
		void reserve(std::size_t pushes) { pool_->reserve(handle_.size, pushes); }

		/**
		 * @brief The vector with `value` appended.
		 * @complexity O(Radix) amortised
		 */
		[[nodiscard]] persistent_vector push_back(T value) const & { return push_back_in(pool_, std::move(value)); }

		/**
		 * @brief `push_back` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).push_back(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] persistent_vector push_back(T value) && { return push_back_in(std::move(pool_), std::move(value)); }

		/**
		 * @brief The vector with its last element removed.
		 * @pre Non-empty.
		 * @complexity O(Radix) amortised
		 */
		[[nodiscard]] persistent_vector pop_back() const & { return pop_back_in(pool_); }

		/**
		 * @brief `pop_back` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).pop_back(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] persistent_vector pop_back() && { return pop_back_in(std::move(pool_)); }

		/**
		 * @brief The vector with element `i` replaced by `value`.
		 * @complexity O(log_Radix n)
		 */
		[[nodiscard]] persistent_vector set(std::size_t i, T value) const & { return set_in(pool_, i, std::move(value)); }

		/**
		 * @brief `set` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).set(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] persistent_vector set(std::size_t i, T value) && { return set_in(std::move(pool_), i, std::move(value)); }

		/**
		 * @brief The concatenation `*this ++ other`.
		 * @complexity O(log n) if the two share an arena, else O(other.size()).
		 */
		[[nodiscard]] persistent_vector concat(persistent_vector const &other) const & { return concat_in(pool_, other); }

		/**
		 * @brief `concat` consuming this version: the pool is moved, not copied. The moved-from
		 *        vector may only be destroyed or assigned to. `std::move(v).concat(v)` is
		 *        handled (it falls back to the copying overload).
		 * @complexity O(log n) if the two share an arena, else O(other.size()).
		 */
		[[nodiscard]] persistent_vector concat(persistent_vector const &other) && {
			if (&other == this) return concat_in(pool_, other);
			return concat_in(std::move(pool_), other);
		}

		/**
		 * @brief The subrange `[from, to)`.
		 * @complexity O(log n)
		 */
		[[nodiscard]] persistent_vector slice(std::size_t from, std::size_t to) const & { return slice_in(pool_, from, to); }

		/**
		 * @brief `slice` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).slice(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] persistent_vector slice(std::size_t from, std::size_t to) && { return slice_in(std::move(pool_), from, to); }

		/**
		 * @brief A copy with `value` inserted before index `i`.
		 * @complexity O(log n)
		 */
		[[nodiscard]] persistent_vector insert(std::size_t i, T value) const & { return insert_in(pool_, i, std::move(value)); }

		/**
		 * @brief `insert` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).insert(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] persistent_vector insert(std::size_t i, T value) && { return insert_in(std::move(pool_), i, std::move(value)); }

		/**
		 * @brief `push_back` of an element constructed in place from `args`.
		 * @complexity O(Radix) amortised
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_vector emplace_back(Args &&...args) const & {
			return push_back(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace_back` consuming this version (see the rvalue `push_back`).
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_vector emplace_back(Args &&...args) && {
			return std::move(*this).push_back(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `insert` before index `i` of an element constructed from `args`.
		 * @complexity O(log n)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_vector emplace(std::size_t i, Args &&...args) const & {
			return insert(i, T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace` consuming this version (see the rvalue `insert`).
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_vector emplace(std::size_t i, Args &&...args) && {
			return std::move(*this).insert(i, T(std::forward<Args>(args)...));
		}
	};

	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::ranges::sized_range Range>
	persistent_vector(Range &&) -> persistent_vector<std::ranges::range_value_t<Range>>;

	/**
	 * @brief Semi-persistent growable vector (trie + tail, DSST node-copying).
	 *
	 * Same interface as @ref atl::persistent_vector, but tuned for a linear history:
	 * appends into fresh territory mutate in place (no allocation), and only
	 * revisits (a `set`, or a `push_back` after `pop_back`) are version-stamped.
	 * This makes incremental `push_back` close to an ephemeral vector's cost while
	 * keeping every past version readable.
	 *
	 * @tparam T     The element type.
	 * @tparam Radix Branching factor, a power of two (default 16).
	 *
	 * @pre Partial persistence: you may read any past version but may only mutate
	 *      the newest one (`push_back`/`pop_back`/`set` on an older version is a
	 *      precondition violation). Chain mutations on the value each returns.
	 * @note Not constexpr (versions own a std::shared_ptr pool); this applies to all members; `operator[]`/`back` references are invalidated by a
	 *       later mutation.
	 */
	template<std::semiregular T, std::size_t Radix = 16>
		requires detail::valid_radix<Radix>
	class semi_persistent_vector {
		using pool = detail::semi_rrb_tree_pool<T, Radix>;
		std::shared_ptr<pool> pool_;
		detail::version_handle handle_;

		semi_persistent_vector(std::shared_ptr<pool> shared, detail::version_handle handle)
			: pool_(std::move(shared)), handle_(handle) {}

		/** @brief `push_back` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		semi_persistent_vector push_back_in(std::shared_ptr<pool> shared, T value) const {
			atl_assert(handle_.version == shared->latest_version);
			auto next = shared->push_back(handle_, std::move(value));
			return {std::move(shared), next};
		}

		/** @brief `pop_back` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		semi_persistent_vector pop_back_in(std::shared_ptr<pool> shared) const {
			atl_assert(handle_.size > 0);
			atl_assert(handle_.version == shared->latest_version);
			auto next = shared->pop_back(handle_);
			return {std::move(shared), next};
		}

		/** @brief `set` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		semi_persistent_vector set_in(std::shared_ptr<pool> shared, std::size_t i, T value) const {
			atl_assert(i < handle_.size);
			atl_assert(handle_.version == shared->latest_version);
			auto next = shared->set(handle_, uint32(i), std::move(value));
			return {std::move(shared), next};
		}

	public:
		/** @brief An empty vector. */
		semi_persistent_vector() : pool_(std::make_shared<pool>()) {}

		/**
		 * @brief Build from `values` in O(n), copying lvalues/views and moving owning rvalues or yielded prvalues.
		 */
		template<std::ranges::sized_range Range>
			requires std::convertible_to<detail::forwarded_element_t<Range>, T>
		explicit semi_persistent_vector(Range &&values) : pool_(std::make_shared<pool>()) {
			handle_ = pool_->build(std::forward<Range>(values));
		}

		/**
		 * @brief A vector of `n` copies of `value`.
		 * @complexity O(n)
		 */
		explicit semi_persistent_vector(std::size_t n, T value = T{})
			: semi_persistent_vector(std::vector<T>(n, std::move(value))) {}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return handle_.size; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return handle_.size == 0; }

		/**
		 * @brief The element at `i`, as of this version.
		 * @complexity O(log_Radix n)
		 */
		[[nodiscard]] T const &operator[](std::size_t i) const {
			atl_assert(i < handle_.size);
			return pool_->at(handle_, uint32(i));
		}

		/** @brief The first element. @pre Non-empty. */
		[[nodiscard]] T const &front() const { return (*this)[0]; }

		/** @brief The last element. @pre Non-empty. */
		[[nodiscard]] T const &back() const { return (*this)[handle_.size - 1]; }

		/**
		 * @brief The vector with `value` appended.
		 * @pre This must be the newest version (partial persistence).
		 * @complexity O(Radix) amortised, O(1) amortised space
		 */
		[[nodiscard]] semi_persistent_vector push_back(T value) const & { return push_back_in(pool_, std::move(value)); }

		/**
		 * @brief `push_back` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).push_back(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] semi_persistent_vector push_back(T value) && { return push_back_in(std::move(pool_), std::move(value)); }

		/**
		 * @brief The vector with its last element removed.
		 * @pre Non-empty and the newest version (partial persistence).
		 * @complexity O(Radix) amortised
		 */
		[[nodiscard]] semi_persistent_vector pop_back() const & { return pop_back_in(pool_); }

		/**
		 * @brief `pop_back` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).pop_back(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] semi_persistent_vector pop_back() && { return pop_back_in(std::move(pool_)); }

		/**
		 * @brief The vector with element `i` replaced by `value`.
		 * @pre This must be the newest version (partial persistence).
		 * @complexity O(log_Radix n)
		 */
		[[nodiscard]] semi_persistent_vector set(std::size_t i, T value) const & { return set_in(pool_, i, std::move(value)); }

		/**
		 * @brief `set` consuming this version: the pool is moved, not copied (no atomic refcount
		 *        traffic), so `v = std::move(v).set(...)` is the cheap linear idiom. The moved-from
		 *        vector may only be destroyed or assigned to. Otherwise as the lvalue overload.
		 */
		[[nodiscard]] semi_persistent_vector set(std::size_t i, T value) && { return set_in(std::move(pool_), i, std::move(value)); }

		/**
		 * @brief `push_back` of an element constructed in place from `args`.
		 * @pre This must be the newest version (partial persistence).
		 * @complexity O(Radix) amortised
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] semi_persistent_vector emplace_back(Args &&...args) const & {
			return push_back(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace_back` consuming this version (see the rvalue `push_back`).
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] semi_persistent_vector emplace_back(Args &&...args) && {
			return std::move(*this).push_back(T(std::forward<Args>(args)...));
		}
	};

	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::ranges::sized_range Range>
	semi_persistent_vector(Range &&) -> semi_persistent_vector<std::ranges::range_value_t<Range>>;
}
