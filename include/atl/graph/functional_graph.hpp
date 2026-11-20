#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/partition.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::graph {
	/** @brief Sized input range of integral successor ids, excluding bool and character types.
	 * @tparam Successors Input range whose references convert to its integral value type.
	 */
	template<typename Successors>
	concept successor_range = vertex_range<Successors> && std::ranges::sized_range<Successors>
						   && detail::vertex_integral<std::ranges::range_value_t<Successors>>
						   && std::convertible_to<std::ranges::range_reference_t<Successors>, std::ranges::range_value_t<Successors>>;

	/** @brief Canonical cycle/tree decomposition; cycles are numbered by increasing minimum vertex.
	 * Each cycle starts at its minimum and follows successor order. Empty graphs have offsets {0}.
	 */
	struct functional_graph_decomposition {
		std::vector<uint32> cycle_id;		///< Eventual cycle id, including tails.
		std::vector<uint32> cycle_position; ///< Position on the cycle; none on tails.
		std::vector<uint32> depth_to_cycle; ///< Steps to the first cycle vertex.
		std::vector<vertex_id> cycle_root;	///< First cycle vertex reached, itself on cycles.
		vertex_groups cycles;				///< Only cycle vertices, in canonical successor order.
	};

	/** @brief Inclusive supported step bound for functional-graph queries. */
	inline constexpr uint64 maximum_functional_graph_steps = 1'000'000'000'000'000'000ULL;

	/** @brief One successor query; start must be a vertex and steps at most 10^18 (asserted by consumers). */
	struct functional_graph_successor_query {
		vertex_id start; ///< Starting vertex.
		uint64 steps;	 ///< Number of outgoing edges to traverse.
	};

	/** @brief Endpoint and ordered outgoing-edge fold of a walk; zero steps returns the supplied identity.
	 * @tparam Value Copyable outgoing-edge aggregate.
	 */
	template<std::copyable Value>
	struct functional_graph_walk_result {
		vertex_id vertex; ///< Endpoint after all requested edges.
		Value fold;		  ///< Fold in traversal order, identity for zero edges.
	};

	/** @brief Sized, single-pass query range; batch results preserve order and duplicates.
	 * @tparam Queries Input range whose values and converted references are functional_graph_successor_query.
	 */
	template<typename Queries>
	concept functional_graph_query_range = std::ranges::input_range<Queries> && std::ranges::sized_range<Queries>
										&& std::same_as<std::ranges::range_value_t<Queries>, functional_graph_successor_query>
										&& std::convertible_to<std::ranges::range_reference_t<Queries>, functional_graph_successor_query>;

	/** @brief Sized per-vertex value range with exact Value type, constructible under the owning-range forwarding rule.
	 * @tparam Values Sized input range of outgoing-edge values.
	 * @tparam Value Copyable value type constructed from each forwarded range reference.
	 */
	template<typename Values, typename Value>
	concept functional_graph_value_range = std::ranges::input_range<Values> && std::ranges::sized_range<Values>
										&& std::copyable<Value> && std::same_as<std::ranges::range_value_t<Values>, Value>
										&& std::constructible_from<Value, atl::detail::forwarded_element_t<Values>>;

	namespace detail {
		/** @brief Tag for successors whose size and ids have been validated. */
		struct validated_successors_t {
			explicit validated_successors_t() = default;
		};

		inline constexpr validated_successors_t validated_successors{};

		/** @brief Checked stored-index narrowing; none remains reserved. */
		[[nodiscard]] constexpr uint32 functional_graph_id(std::size_t value) {
			atl_assert(value < none);
			return uint32(value);
		}

		/** @brief Copy and validate one input pass before preprocessing.
		 * @pre n < 2^31; each raw id is nonnegative, representable and less than n (asserted before conversion).
		 * @complexity O(n) time and storage.
		 */
		template<successor_range Successors>
		[[nodiscard]] constexpr std::vector<vertex_id> functional_graph_input(Successors &&successors) {
			auto const raw_size = std::ranges::size(successors);
			atl_assert(raw_size < (uint64(1) << 31));
			atl_assert(raw_size <= (std::numeric_limits<std::size_t>::max)());
			auto const n = std::size_t(raw_size);
			std::vector<vertex_id> result;
			result.reserve(n);
			for (std::ranges::range_value_t<Successors> const raw : successors) {
				atl_assert(fits_vertex_id(raw));
				atl_assert(std::cmp_less(raw, n));
				atl_assert(result.size() < n);
				result.push_back(target(raw));
			}
			atl_assert(result.size() == n);
			return result;
		}

		/** @brief Recover canonical cycles and propagate tails in successor-before-child order.
		 * @pre Input is validated; is_cycle identifies exactly the cycle vertices and order lists every tail once,
		 * with its successor earlier when off-cycle. @complexity O(n) time/space.
		 */
		[[nodiscard]] constexpr functional_graph_decomposition functional_graph_metadata(
			std::vector<vertex_id> const &successors, std::vector<uint32> const &is_cycle, std::vector<vertex_id> const &order) {
			auto const n = successors.size();
			functional_graph_decomposition result{std::vector<uint32>(n, none), std::vector<uint32>(n, none), std::vector<uint32>(n, 0), std::vector<vertex_id>(n, none), {}};
			result.cycles.members.reserve(n);
			result.cycles.offsets.reserve(n + 1);
			for (vertex_id start = 0; start < n; ++start) {
				if (!is_cycle[start] || result.cycle_id[start] != none) continue;
				auto const id = functional_graph_id(result.cycles.count());
				vertex_id v = start;
				uint32 position = 0;
				do {
					result.cycle_id[v] = id;
					result.cycle_position[v] = position++;
					result.cycle_root[v] = v;
					result.cycles.members.push_back(v);
					v = successors[v];
				} while (v != start);
				result.cycles.offsets.push_back(functional_graph_id(result.cycles.members.size()));
			}
			for (vertex_id v : order) {
				vertex_id const parent = successors[v];
				result.cycle_id[v] = result.cycle_id[parent];
				result.cycle_root[v] = result.cycle_root[parent];
				result.depth_to_cycle[v] = functional_graph_id(std::size_t(result.depth_to_cycle[parent]) + 1);
			}
			return result;
		}

		/** @brief Indegree peeling followed by cycle walks and reverse propagation; O(n) time/space. */
		[[nodiscard]] constexpr functional_graph_decomposition functional_graph_peeling(std::vector<vertex_id> const &successors) {
			auto const n = successors.size();
			std::vector<uint32> indegree(n, 0);
			for (vertex_id v : successors) ++indegree[v];
			std::vector<vertex_id> order;
			order.reserve(n);
			for (vertex_id v = 0; v < n; ++v)
				if (indegree[v] == 0) order.push_back(v);
			for (std::size_t at = 0; at < order.size(); ++at)
				if (--indegree[successors[order[at]]] == 0) order.push_back(successors[order[at]]);
			std::ranges::reverse(order);
			return functional_graph_metadata(successors, indegree, order);
		}

		/** @brief Validate a query before any indexing or step arithmetic. */
		constexpr void functional_graph_check(std::size_t n, vertex_id start, uint64 steps) {
			atl_assert(start < n);
			atl_assert(steps <= maximum_functional_graph_steps);
		}

		/** @brief Endpoint after steps from a cycle vertex; arithmetic stays wide until the modulo bound. */
		[[nodiscard]] constexpr vertex_id functional_graph_cycle_vertex(functional_graph_decomposition const &decomposition,
																		vertex_id root, uint64 steps) {
			auto const cycle = decomposition.cycles[decomposition.cycle_id[root]];
			uint64 const position = (uint64(decomposition.cycle_position[root]) + steps % cycle.size()) % cycle.size();
			atl_assert(position < cycle.size());
			return cycle[std::size_t(position)];
		}

		/** @brief Owning decomposition and cut forest for O(log n) endpoints, O(1) after reaching a cycle.
		 * Empty input bypasses lca, whose parent-array constructor requires a nonempty forest.
		 */
		struct functional_graph_topology {
			functional_graph_decomposition decomposition; ///< Owned canonical metadata.
			std::optional<lca> forest;					  ///< Cut forest, absent for an empty graph.

			/** @brief Build from validated successors, in O(n) time/space. */
			constexpr explicit functional_graph_topology(std::vector<vertex_id> const &successors)
				: decomposition(functional_graph_peeling(successors)) {
				if (successors.empty()) return;
				// The cut goes into a local array: writing it into a by-value vector parameter gives a
				// wrong forest under MSVC's constant evaluation.
				std::vector<vertex_id> cut(successors.size());
				for (vertex_id v = 0; v < successors.size(); ++v) cut[v] = decomposition.depth_to_cycle[v] == 0 ? none : successors[v];
				forest.emplace(cut);
			}

			/** @brief Validated k-step endpoint without consulting any values. */
			[[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const {
				functional_graph_check(decomposition.cycle_id.size(), start, steps);
				uint32 const depth = decomposition.depth_to_cycle[start];
				// steps < depth implies a tail vertex, so the graph is nonempty and forest is engaged; the
				// explicit test only spells that out for static analysis
				if (steps < depth && forest) return forest->ancestor(start, functional_graph_id(std::size_t(steps)));
				return functional_graph_cycle_vertex(decomposition, decomposition.cycle_root[start], steps - depth);
			}
		};

		inline constexpr std::size_t functional_graph_maximum_fold_vertices = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);

		/** @brief Enforce the segment-tree doubled-layout bound without overflowing size_t. */
		constexpr void functional_graph_check_fold_size(std::size_t n) {
			atl_assert(n <= functional_graph_maximum_fold_vertices);
		}

		/** @brief Consume values once, moving only owning rvalue elements; assert exact graph size before allocation. */
		template<std::copyable Value, typename Values>
			requires functional_graph_value_range<Values, Value>
		[[nodiscard]] constexpr std::vector<Value> functional_graph_values(Values &&values, std::size_t n) {
			atl_assert(std::ranges::size(values) == n);
			std::vector<Value> result;
			result.reserve(n);
			for (auto &&value : values) {
				atl_assert(result.size() < n);
				result.emplace_back(atl::detail::forward_element<Values>(std::forward<decltype(value)>(value)));
			}
			atl_assert(result.size() == n);
			return result;
		}

		/** @brief Ordered monoid exponentiation with no unused final squaring.
		 * @pre Associative, pure operation and two-sided identity; all intermediate values representable (documented).
		 * @complexity O(log(exponent+1)) operations, O(1) auxiliary values.
		 */
		template<std::copyable Value, binary_operation<Value> Operation>
		[[nodiscard]] constexpr Value functional_graph_power(Value value, uint64 exponent, Value identity, Operation const &operation) {
			while (exponent != 0) {
				if (exponent & 1) identity = std::invoke(std::as_const(operation), identity, value);
				exponent >>= 1;
				if (exponent != 0) value = std::invoke(std::as_const(operation), value, value);
			}
			return identity;
		}

		/** @brief Preserve the const-invocable monoid contract, including member pointers, when storing it in a segment tree. */
		template<std::copyable Value, binary_operation<Value> Operation>
			requires std::copyable<Operation>
		struct functional_graph_operation {
			Operation operation; ///< Owned operation; semantic state is stable/pure.

			/** @brief Invoke ordered composition; monoid laws and representability are documented preconditions. */
			[[nodiscard]] constexpr Value operator()(Value const &left, Value const &right) const {
				return std::invoke(std::as_const(operation), left, right);
			}
		};

		/** @brief Linear-space ordered cycle folds: prefixes/suffixes for rotated laps, segment tree for remainders.
		 * @pre Monoid laws, purity and representable intermediate folds (documented).
		 */
		template<std::copyable Value, binary_operation<Value> Operation>
			requires std::copyable<Operation>
		class functional_graph_cycle_folds {
			Value identity_;
			Operation operation_;
			std::vector<Value> prefix_, suffix_;														 ///< Per-cycle folds before/at each flat position.
			std::optional<segment_tree<Value, functional_graph_operation<Value, Operation>>> intervals_; ///< Successor-order cycle intervals.

		public:
			/** @brief Build O(n) cycle storage and folds from validated metadata and values. */
			constexpr functional_graph_cycle_folds(functional_graph_decomposition const &decomposition,
												   std::vector<Value> const &values, Value identity, Operation operation)
				: identity_(std::move(identity)), operation_(std::move(operation)) {
				auto const n = decomposition.cycles.members.size();
				functional_graph_check_fold_size(n);
				prefix_.assign(n, identity_);
				suffix_.assign(n, identity_);
				std::vector<Value> cycle_values;
				cycle_values.reserve(n);
				for (vertex_id v : decomposition.cycles.members) cycle_values.push_back(values[v]);
				for (std::size_t id = 0; id < decomposition.cycles.count(); ++id) {
					std::size_t const begin = decomposition.cycles.offsets[id], end = decomposition.cycles.offsets[id + 1];
					Value fold = identity_;
					for (std::size_t p = begin; p < end; ++p) {
						prefix_[p] = fold;
						fold = std::invoke(std::as_const(operation_), fold, cycle_values[p]);
					}
					fold = identity_;
					for (std::size_t p = end; p-- > begin;) {
						fold = std::invoke(std::as_const(operation_), cycle_values[p], fold);
						suffix_[p] = fold;
					}
				}
				intervals_.emplace(std::move(cycle_values), identity_, functional_graph_operation<Value, Operation>{operation_});
			}

			/** @brief Fold steps cycle edges from root, in O(log n + log(steps+1)) operations. */
			[[nodiscard]] constexpr Value fold(functional_graph_decomposition const &decomposition, vertex_id root, uint64 steps) const {
				auto const id = decomposition.cycle_id[root];
				std::size_t const begin = decomposition.cycles.offsets[id], end = decomposition.cycles.offsets[id + 1];
				std::size_t const position = begin + decomposition.cycle_position[root];
				auto const length = end - begin;
				Value result = identity_;
				if (steps >= length) {
					Value lap = std::invoke(std::as_const(operation_), suffix_[position], prefix_[position]);
					result = functional_graph_power(std::move(lap), steps / length, identity_, operation_);
				}
				uint64 const residue = steps % length;
				atl_assert(residue < length);
				auto const remaining = std::size_t(residue);
				if (remaining <= end - position)
					return std::invoke(std::as_const(operation_), result, intervals_->fold(position, position + remaining));
				result = std::invoke(std::as_const(operation_), result, suffix_[position]);
				return std::invoke(std::as_const(operation_), result, intervals_->fold(begin, begin + remaining - (end - position)));
			}
		};

		/** @brief Shared lca-based walk engine; IsPrefix caches upward folds to heavy-chain heads.
		 * O(n) build/space; tail folds cost O(log² n), or O(log n) with IsPrefix.
		 * Cycle powering adds O(log(steps+1)). Monoid laws/purity/representability are documented preconditions.
		 */
		template<std::copyable Value, binary_operation<Value> Operation, bool IsPrefix>
			requires std::copyable<Operation>
		class functional_graph_walk_engine {
			functional_graph_topology topology_;
			Value identity_;
			Operation operation_;
			functional_graph_cycle_folds<Value, Operation> cycles_;
			std::optional<segment_tree<Value, functional_graph_operation<Value, Operation>>> upward_; ///< Mirrored heavy-first preorder.
			std::vector<vertex_id> head_;															  ///< Derived heavy-chain heads in prefix mode.
			std::vector<Value> prefix_;																  ///< Upward fold including the chain head in prefix mode.

			/** @brief Build from validated successors and values; O(n) time/space.
			 * Both arrive by const reference: MSVC's constant evaluation misreads by-value vector
			 * parameters of a delegated-to constructor.
			 */
			constexpr functional_graph_walk_engine(std::vector<vertex_id> const &successors, std::vector<Value> const &values,
												   Value identity, Operation operation, validated_successors_t)
				: topology_(successors), identity_(std::move(identity)), operation_(std::move(operation)),
				  cycles_(topology_.decomposition, values, identity_, operation_) {
				auto const n = values.size();
				std::vector<Value> upward_values(n, identity_);
				if (n) {
					auto const &forest = *topology_.forest;
					if constexpr (IsPrefix) {
						head_.resize(n);
						prefix_.assign(n, identity_);
					}
					for (std::size_t p = 0; p < n; ++p) {
						vertex_id const v = forest.order()[p];
						upward_values[n - 1 - p] = values[v];
						if constexpr (IsPrefix) {
							vertex_id const parent = forest.parent(v);
							// The first preorder child is heavy; every other child starts a chain.
							bool const is_heavy = parent != none && forest.entry_time(parent) + 1 == p;
							head_[v] = is_heavy ? head_[parent] : v;
							prefix_[v] = is_heavy ? Value(std::invoke(std::as_const(operation_), values[v], prefix_[parent])) : values[v];
						}
					}
				}
				upward_.emplace(std::move(upward_values), identity_, functional_graph_operation<Value, Operation>{operation_});
			}

			/** @brief Read successors and values in sequence, each exactly once, before delegating to the owned build. */
			template<successor_range Successors, typename Values>
				requires functional_graph_value_range<Values, Value>
			[[nodiscard]] static constexpr auto inputs(Successors &&successors, Values &&values) {
				atl_assert(std::ranges::size(successors) <= functional_graph_maximum_fold_vertices);
				auto owned_successors = functional_graph_input(std::forward<Successors>(successors));
				auto owned_values = functional_graph_values<Value>(std::forward<Values>(values), owned_successors.size());
				return std::pair(std::move(owned_successors), std::move(owned_values));
			}

			constexpr functional_graph_walk_engine(std::pair<std::vector<vertex_id>, std::vector<Value>> const &owned,
												   Value identity, Operation operation)
				: functional_graph_walk_engine(owned.first, owned.second, std::move(identity), std::move(operation), validated_successors) {}

		public:
			/** @brief Own and preprocess inputs; asserted graph/value-count bounds match the public walk index. */
			template<successor_range Successors, typename Values>
				requires functional_graph_value_range<Values, Value>
			constexpr functional_graph_walk_engine(Successors &&successors, Values &&values, Value identity, Operation operation)
				: functional_graph_walk_engine(inputs(std::forward<Successors>(successors), std::forward<Values>(values)),
											   std::move(identity), std::move(operation)) {}

			/** @brief Number of vertices. */
			constexpr std::size_t size() const noexcept { return topology_.decomposition.cycle_id.size(); }

			/** @brief Owned metadata; references survive until destruction, move or assignment. */
			[[nodiscard]] constexpr functional_graph_decomposition const &decomposition() const noexcept { return topology_.decomposition; }

			/** @brief Value-independent endpoint query; asserted start/step bounds, O(log n) time. */
			[[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const { return topology_.successor(start, steps); }

			/** @brief Ordered edge fold and endpoint; asserted start/step bounds. No query scratch allocation by the engine. */
			[[nodiscard]] constexpr functional_graph_walk_result<Value> walk(vertex_id start, uint64 steps) const {
				vertex_id const endpoint = successor(start, steps);
				auto const &decomp = topology_.decomposition;
				uint64 const tail_steps = (std::ranges::min)(steps, uint64(decomp.depth_to_cycle[start]));
				Value fold = identity_;
				if (tail_steps) {
					auto const &forest = *topology_.forest;
					vertex_id const top = forest.ancestor(start, functional_graph_id(std::size_t(tail_steps)));
					if constexpr (IsPrefix) {
						vertex_id v = start;
						while (head_[v] != head_[top]) {
							fold = std::invoke(std::as_const(operation_), fold, prefix_[v]);
							v = forest.parent(head_[v]);
						}
						fold = std::invoke(std::as_const(operation_), fold, upward_->fold(size() - 1 - forest.entry_time(v), size() - 1 - forest.entry_time(top)));
					} else {
						auto const append_interval = [&](uint32 l, uint32 r, [[maybe_unused]] bool is_reversed) {
							atl_assert(is_reversed);
							fold = std::invoke(std::as_const(operation_), fold, upward_->fold(size() - r, size() - l));
						};
						forest.path(start, top, append_interval, false);
					}
				}
				if (steps > tail_steps)
					fold = std::invoke(std::as_const(operation_), fold, cycles_.fold(decomp, decomp.cycle_root[start], steps - tail_steps));
				return {endpoint, std::move(fold)};
			}
		};
	}

	/** @brief Canonical functional-graph decomposition using indegree peeling.
	 * @pre n < 2^31 and every raw successor is nonnegative, representable as vertex_id and < n (asserted).
	 * Empty input is valid. Consumes the input once; owns all output arrays.
	 * @param successors Sized input range with one target for each vertex.
	 * @return Owned canonical metadata, computed iteratively with heap buffers.
	 * @complexity O(n) time and space.
	 */
	template<successor_range Successors>
	[[nodiscard]] constexpr functional_graph_decomposition functional_graph_decomposition_of(Successors &&successors) {
		return detail::functional_graph_peeling(detail::functional_graph_input(std::forward<Successors>(successors)));
	}

	/** @brief Static successor index using peeling, lca::ancestor and cycle arithmetic.
	 * Owns its state; copies are independent. Query members are const and allocate no storage.
	 * Preprocessing uses heap buffers, with no recursion per vertex; allocation failures propagate.
	 * @complexity O(n) build/space; O(log n) endpoint, O(1) if steps reaches the cycle.
	 */
	class functional_graph_successor_index {
		detail::functional_graph_topology topology_;

	public:
		/** @brief Consume successors once and own preprocessing.
		 * @pre n < 2^31 (asserted prior to allocation); raw successors are nonnegative, representable and < n
		 * (asserted during input consumption, prior to conversion).
		 * Empty input is valid. @complexity O(n) time/space.
		 */
		template<successor_range Successors>
		constexpr explicit functional_graph_successor_index(Successors &&successors)
			: topology_(detail::functional_graph_input(std::forward<Successors>(successors))) {}

		/** @brief Number of vertices. */
		constexpr std::size_t size() const noexcept { return topology_.decomposition.cycle_id.size(); }

		/** @brief Owned canonical metadata; reference lasts until destruction, move or assignment. */
		[[nodiscard]] constexpr functional_graph_decomposition const &decomposition() const noexcept { return topology_.decomposition; }

		/** @brief k-step endpoint, including start at zero steps.
		 * @param start Vertex whose outgoing edges are followed. @param steps Number of edges to traverse.
		 * @return Vertex reached, with start returned for zero steps.
		 * @pre start < size() and steps <= 10^18 (asserted). @complexity O(log n), O(1) upon reaching a cycle.
		 */
		[[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const { return topology_.successor(start, steps); }
	};

	/** @brief Static ordered outgoing-edge folds using the cut lca forest and cycle arithmetic.
	 * values[v] is the value of the step v -> successors[v]; walk(v,k) folds the k values of the
	 * vertices it leaves (start included, endpoint excluded), in traversal order.
	 * No inverse, commutativity or idempotence is required. Copies own independent storage.
	 * @tparam Value Copyable outgoing-edge value.
	 * @tparam Operation Copyable const-invocable binary operation returning a value convertible to Value.
	 * @pre Operation is associative, stable/pure, identity is two-sided, and every intermediate fold
	 * (including precomputed blocks) is representable in Value (documented, not asserted).
	 * @complexity O(n) build/space; walk O(log² n + log(k+1)); successor O(log n), O(1) on reaching a cycle.
	 * Preprocessing is iterative; allocation failures propagate.
	 * Queries are const; only Value/Operation may allocate. Supply explicit template arguments; no deduction guides.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
		requires std::copyable<Operation>
	class functional_graph_walk_index {
		detail::functional_graph_walk_engine<Value, Operation, false> engine_;

	public:
		/** @brief Consume each input range once; move values only from owning rvalues.
		 * @pre n < 2^31; raw successors nonnegative, representable and < n; values.size() == n
		 * (asserted). Also `n <= 2^(numeric_limits<size_t>::digits - 2)` for segment-tree storage (asserted).
		 * Empty input is valid.
		 * @param successors One outgoing target per vertex, copied in a single pass.
		 * @param values Outgoing-edge values in vertex order; owning rvalues may be moved.
		 * @param identity Two-sided neutral element. @param operation Pure associative ordered composition.
		 * @complexity O(n) time/space.
		 */
		template<successor_range Successors, typename Values>
			requires functional_graph_value_range<Values, Value>
		constexpr functional_graph_walk_index(Successors &&successors, Values &&values, Value identity, Operation operation)
			: engine_(std::forward<Successors>(successors), std::forward<Values>(values), std::move(identity), std::move(operation)) {}

		/** @brief Number of vertices. */
		constexpr std::size_t size() const noexcept { return engine_.size(); }

		/** @brief Canonical metadata; reference lasts until destruction, move or assignment. */
		[[nodiscard]] constexpr functional_graph_decomposition const &decomposition() const noexcept { return engine_.decomposition(); }

		/** @brief Value-independent endpoint. @pre start < size(), steps <= 10^18 (asserted). @complexity O(log n). */
		[[nodiscard]] constexpr vertex_id successor(vertex_id start, uint64 steps) const { return engine_.successor(start, steps); }

		/** @brief Endpoint and edge fold, {start, identity} at zero steps.
		 * @param start Vertex whose outgoing edges are followed. @param steps Number of edges to traverse.
		 * @return Endpoint and values of the departed vertices folded in traversal order.
		 * @pre start < size(), steps <= 10^18 (asserted). @complexity O(log² n + log(steps+1)).
		 */
		[[nodiscard]] constexpr functional_graph_walk_result<Value> walk(vertex_id start, uint64 steps) const { return engine_.walk(start, steps); }
	};
}
