#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <vector>

namespace atl::detail {
	/**
	 * @brief A node usable in a @ref hashed_child_pool.
	 *
	 * `symbol` is the key the node is filed under in its parent's sibling BST —
	 * the label of the edge leading to it (a trie character, an automaton
	 * transition symbol); `left`/`right` are that BST's links. Any further
	 * fields — an explicit target state, payload counts — belong to the owner
	 * and are untouched by the pool. Value initialization starts payload from its
	 * member initializers or zero. Vector growth move-constructs nodes; they are never
	 * assigned, so nodes with const payload fields are fine.
	 */
	template<typename Node>
	concept sibling_node = std::default_initializable<Node> && std::move_constructible<Node> && requires(Node &n) {
		{ n.symbol } -> std::same_as<uint32 &>;
		{ n.left } -> std::same_as<uint32 &>;
		{ n.right } -> std::same_as<uint32 &>;
	};

	/**
	 * @brief A shared arena of nodes whose children are held as small binary
	 *        search trees ordered by a *seeded hash* of the child symbol.
	 *
	 * Each owner keeps a uint32 sibling-BST root. The pool touches only symbol,
	 * left and right; payload and target-state fields belong to the owner.
	 * Storage is O(nodes). A probe takes O(height), expected O(log siblings)
	 * for well-distributed hash order; a worst-case tree can be a chain.
	 * Node references are invalidated when the arena reallocates; indices remain
	 * valid until clear(). Roots and links must name nodes in this pool or nil.
	 * Construct from main or later, after process seed initialization.
	 *
	 * splitmix64 is a bijection, so distinct symbols get distinct order keys and
	 * the BST needs no tie-breaking; the seed is drawn per process (pin it with
	 * @ref atl::set_hash_seed) and captured at construction for the pool's lifetime. Later process seed changes
	 * do not change existing sibling order.
	 *
	 * @tparam Node A @ref sibling_node; emplaced child nodes are value-initialised,
	 *              then stamped with symbol and nil links. add() leaves Node{} intact.
	 */
	template<sibling_node Node>
	class hashed_child_pool {
		std::vector<Node> nodes_;
		uint64 seed_;

		/**
		 * @brief Append a node filed under @p sym with no children; the caller
		 *        links it into its parent's sibling BST. Returns its index.
		 * @pre fewer than 2^32 - 1 nodes exist (asserted: `nil` is reserved).
		 */
		uint32 create(uint32 sym) {
			atl_assert(nodes_.size() < nil);
			nodes_.push_back(Node{});
			uint32 const i = uint32(nodes_.size() - 1);
			nodes_[i].symbol = sym;
			nodes_[i].left = nil;
			nodes_[i].right = nil;
			return i;
		}

	public:
		/** @brief The null index: an empty sibling BST, or a missing child link. */
		static constexpr uint32 nil = uint32(-1);

		/** @brief An empty pool, ordering siblings by the process hash seed. */
		hashed_child_pool() : seed_(hash_seed()) {}

		/**
		 * @brief The sibling-ordering key of @p sym: `splitmix64(sym + seed)`, a
		 *        bijection, so distinct symbols never tie.
		 */
		uint64 order(uint32 sym) const { return splitmix64(uint64(sym) + seed_); }

		/** @brief The seed @ref order mixes in, fixed for the pool's lifetime. */
		uint64 seed() const { return seed_; }

		Node &operator[](uint32 i) { return nodes_[i]; }

		Node const &operator[](uint32 i) const { return nodes_[i]; }

		uint32 size() const { return uint32(nodes_.size()); }

		void reserve(std::size_t n) { nodes_.reserve(n); }

		void clear() { nodes_.clear(); }

		/**
		 * @brief Append a detached Node{} and return its index.
		 * The owner must initialize its symbol and sibling links before using it
		 * in a sibling tree; owner-provided member initializers are preserved.
		 * @pre Fewer than nil nodes exist (asserted before allocation).
		 */
		uint32 add() {
			atl_assert(nodes_.size() < nil);
			nodes_.push_back(Node{});
			return uint32(nodes_.size() - 1);
		}

		/** @brief The node in the sibling BST @p root carrying @p sym, or @ref nil. */
		uint32 find(uint32 root, uint32 sym) const {
			uint32 cur = root;
			if (cur == nil) return nil;
			uint64 const k = order(sym);
			for (;;) {
				uint32 const c = nodes_[cur].symbol;
				if (c == sym) return cur;
				cur = k < order(c) ? nodes_[cur].left : nodes_[cur].right;
				if (cur == nil) return nil;
			}
		}

		/** @brief The outcome of @ref emplace. */
		struct emplace_result {
			uint32 node;	   /**< @brief The node carrying the symbol. */
			uint32 root;	   /**< @brief The sibling BST's root after the call. */
			bool was_inserted; /**< @brief Whether @ref node was created by this call. */
		};

		/**
		 * @brief Ensure a node carrying @p sym in the sibling BST @p root, creating
		 *        it if absent.
		 * @return `{node, root, was_inserted}`. `root` differs from @p root only when
		 *         the tree was empty. Pass @p root **by value** and write the
		 *         returned `root` back to the owner's slot: a reference to that
		 *         slot would dangle across the internal reallocation.
		 */
		emplace_result emplace(uint32 root, uint32 sym) {
			if (root == nil) {
				uint32 const n = create(sym);
				return {n, n, true};
			}
			uint64 const k = order(sym);
			uint32 cur = root;
			for (;;) {
				uint32 const c = nodes_[cur].symbol;
				if (c == sym) return {cur, root, false};
				if (k < order(c)) {
					if (nodes_[cur].left == nil) {
						uint32 const n = create(sym);
						nodes_[cur].left = n;
						return {n, root, true};
					}
					cur = nodes_[cur].left;
				} else {
					if (nodes_[cur].right == nil) {
						uint32 const n = create(sym);
						nodes_[cur].right = n;
						return {n, root, true};
					}
					cur = nodes_[cur].right;
				}
			}
		}

		/**
		 * @brief Visit every node index in the sibling BST @p root. Order is unspecified
		 * (the tree is keyed by hashed symbol, not by symbol). Allocates an
		 * explicit stack per call — recursion would risk overflow on a large
		 * sibling set; keep off hot paths or iterate the arena directly. The visitor must not clear
		 * the pool or change sibling links during traversal.
		 */
		template<typename Visit>
			requires std::invocable<Visit &, uint32>
		void for_each(uint32 root, Visit visit) const {
			if (root == nil) return;
			std::vector<uint32> stack{root};
			while (!stack.empty()) {
				uint32 const n = stack.back();
				stack.pop_back();
				std::invoke(visit, n);
				if (nodes_[n].left != nil) stack.push_back(nodes_[n].left);
				if (nodes_[n].right != nil) stack.push_back(nodes_[n].right);
			}
		}
	};
}
