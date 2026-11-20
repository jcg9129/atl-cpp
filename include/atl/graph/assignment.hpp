#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief The type @ref atl::graph::assignment sums costs and potentials in: a signed
	 *        integer type (character types excluded; `int128` admitted).
	 *        Signed, because potentials go negative; integer, because the
	 *        path search's plateau test (`step == 0`) is exact-arithmetic
	 *        logic.
	 * @tparam Sum Type used for exact signed arithmetic.
	 */
	template<typename Sum>
	concept assignment_sum = std::numeric_limits<Sum>::is_integer && std::numeric_limits<Sum>::is_signed
						  && !detail::character_type<Sum> && std::semiregular<Sum>
						  && std::totally_ordered<Sum> && std::convertible_to<int, Sum>
						  && std::constructible_from<Sum, std::size_t>
						  && requires(Sum a, Sum b) {
								 { a + b } -> std::convertible_to<Sum>;
								 { a - b } -> std::convertible_to<Sum>;
								 { -a } -> std::convertible_to<Sum>;
								 { a / 4 } -> std::convertible_to<Sum>;
								 a += b;
								 a -= b;
							 };

	/**
	 * @brief Minimum-cost dense matching with an exact dual certificate: the minimum total cost, the
	 *        matching in both directions (`none` = unmatched, only on the
	 *        larger side), and the dual potentials: `row_potential[i] +
	 *        col_potential[j] <= cost[i][j]` everywhere, with equality on
	 *        matched pairs. When `rows != cols`, the larger side's potentials
	 *        are additionally `<= 0`, and exactly `0` on its unmatched
	 *        vertices — the remaining dual conditions of the rectangular LP,
	 *        so the potentials certify optimality for every shape.
	 * @tparam Sum Exact signed integer arithmetic type.
	 */
	template<assignment_sum Sum>
	struct assignment_result {
		Sum cost{};						///< Minimum sum over matched entries.
		std::vector<uint32> row_to_col; ///< Column for each row, or none.
		std::vector<uint32> col_to_row; ///< Row for each column, or none.
		std::vector<Sum> row_potential; ///< Row dual variables.
		std::vector<Sum> col_potential; ///< Column dual variables.
	};

	namespace detail {
		/**
		 * @brief A matrix element type @ref atl::graph::assignment accepts for sum type
		 *        @p Sum: arithmetic (a `numeric_limits` type, so `int128`
		 *        counts) and converting to it.
		 */
		template<typename C, typename Sum>
		concept assignment_cost = std::numeric_limits<C>::is_specialized && std::convertible_to<C, Sum>;

		/**
		 * @brief Dense LAP solver on a flat row-major matrix, rows <= cols.
		 *        The matrix keeps the caller's cost type C (scans are
		 *        memory-bound: a 32-bit matrix is half the traffic); all
		 *        arithmetic runs in Sum.
		 *
		 * Successive shortest paths in Hungarian/Dijkstra form: only column
		 * potentials are stored — every matched row is tight on its column
		 * (`u[i] = cost[i][j_i] - v[j_i]` is its implicit potential), and
		 * each free row augments along the cheapest alternating path.
		 * Without preprocessing, the matching is cheapest over the processed
		 * rows; it need not be cheapest among all matchings of that cardinality.
		 *
		 * The path search is the Jonker-Volgenant scan structure rather
		 * than a naive Dijkstra: `order` is partitioned into
		 * [scanned)[labeled)[todo), the next minimum is collected in one
		 * batch pass over todo, and a zero-slack edge promotes its column
		 * immediately — on tie-heavy matrices the plateau never gets
		 * rescanned.
		 *
		 * With @p should_reduce, the Jonker-Volgenant preprocessing runs first:
		 * column reduction (v[j] = column minimum, greedily assigning the
		 * argmin rows), reduction transfer, and two augmenting-row-reduction
		 * passes — on typical instances it resolves most rows for the price
		 * of a few matrix scans, leaving the path search a small tail.
		 * The initialization maintains dual feasibility and tight matched edges;
		 * the path searches extend these properties until all rows are matched.
		 * This certifies optimality on square matrices, where all columns are
		 * matched too. Rectangular instances must pass should_reduce = false
		 * so unmatched columns retain the zero potentials required by the dual.
		 * @pre `rows <= cols`, `cost.size() == rows * cols` (asserted);
		 *      @p should_reduce only when `rows == cols`
		 * @complexity O(rows² · cols) worst case
		 */
		template<assignment_sum Sum, assignment_cost<Sum> C>
		[[nodiscard]] constexpr assignment_result<Sum> dense_assignment(std::size_t rows, std::size_t cols,
																		std::vector<C> const &cost, bool should_reduce) {
			atl_assert(rows <= cols && (rows == 0 || cols <= cost.max_size() / rows));
			atl_assert(cost.size() == rows * cols && (!should_reduce || rows == cols));
			Sum constexpr unreached = (std::numeric_limits<Sum>::max)() / 4;
			assignment_result<Sum> r;
			r.row_to_col.assign(rows, none);
			r.col_to_row.assign(cols, none);
			std::vector<Sum> v(cols, 0);
			auto const at = [&](std::size_t i, std::size_t j) { return Sum(cost[i * cols + j]); };

			if (should_reduce && rows > 0) {
				// column reduction: v = column minima, argmin rows grab their
				// first column — reduced costs are >= 0 from here on. The
				// minima are collected row by row (sequential reads of the
				// row-major matrix; strict `<` keeps the first minimal row):
				// 6.6x faster than a column-major scan at n = 2000 (record
				// 2026-10-15, Xeon Platinum 8280, plain -O2)
				std::vector<uint8> once(rows, 0);
				std::vector<uint32> argmin(cols, 0);
				for (std::size_t j = 0; j < cols; ++j) v[j] = at(0, j);
				for (std::size_t i = 1; i < rows; ++i)
					for (std::size_t j = 0; j < cols; ++j)
						if (at(i, j) < v[j]) {
							v[j] = at(i, j);
							argmin[j] = uint32(i);
						}
				for (std::size_t j = cols; j-- > 0;) {
					std::size_t const best = argmin[j];
					if (r.row_to_col[best] == none) {
						r.row_to_col[best] = uint32(j);
						r.col_to_row[j] = uint32(best);
						once[best] = 1;
					} else {
						once[best] = 0;
					}
				}
				// reduction transfer: a singly-assigned row donates its slack
				// to its column, staying tight but freeing dual room
				for (std::size_t i = 0; i < rows; ++i)
					if (once[i] && cols > 1) {
						uint32 const j1 = r.row_to_col[i];
						Sum slack = unreached;
						for (std::size_t j = 0; j < cols; ++j)
							if (j != j1) slack = (std::min)(slack, Sum(at(i, j) - v[j]));
						v[j1] = Sum(at(i, j1) - slack);
					}
				// augmenting row reduction, two plain passes: each free row
				// bids for its best column at the second-best price,
				// displacing the owner; ties prefer a free column. A
				// displaced row is simply retried by the next pass (or left
				// to the path search) — no queue, no termination worries
				for (int pass = 0; pass < 2; ++pass)
					for (uint32 i = 0; i < rows; ++i) {
						if (r.row_to_col[i] != none) continue;
						std::size_t j1 = 0;
						Sum best = Sum(at(i, 0) - v[0]), second = unreached;
						for (std::size_t j = 1; j < cols; ++j) {
							Sum const cur = Sum(at(i, j) - v[j]);
							if (cur < best || (cur == best && r.col_to_row[j1] != none)) {
								second = best;
								best = cur;
								j1 = j;
							} else {
								second = (std::min)(second, cur);
							}
						}
						if (best < second) v[j1] = Sum(v[j1] - (second - best));
						uint32 const loser = r.col_to_row[j1];
						if (loser != none) r.row_to_col[loser] = none;
						r.row_to_col[i] = uint32(j1);
						r.col_to_row[j1] = i;
					}
			}

			std::vector<Sum> dist(cols);
			std::vector<uint32> from_row(cols), order(cols);
			std::iota(order.begin(), order.end(), uint32(0)); // stays a permutation across rows
			for (uint32 s = 0; s < uint32(rows); ++s) {
				if (r.row_to_col[s] != none) continue;
				for (std::size_t j = 0; j < cols; ++j) {
					dist[j] = Sum(at(s, j) - v[j]);
					from_row[j] = s;
				}
				std::size_t scanned = 0, labeled = 0, settled = 0;
				uint32 open = none;
				while (open == none) {
					if (scanned == labeled) {
						settled = scanned;
						Sum best = dist[order[scanned]];
						for (std::size_t k = scanned; k < cols; ++k) {
							uint32 const j = order[k];
							if (dist[j] > best) continue;
							if (dist[j] < best) {
								best = dist[j];
								labeled = scanned;
							}
							std::swap(order[k], order[labeled]);
							++labeled;
						}
						for (std::size_t k = scanned; k < labeled; ++k)
							if (r.col_to_row[order[k]] == none) {
								open = order[k];
								break;
							}
						if (open != none) break;
					}
					uint32 const j1 = order[scanned++];
					uint32 const i2 = r.col_to_row[j1];
					Sum const tight = Sum(at(i2, j1) - v[j1]);
					for (std::size_t k = labeled; k < cols; ++k) {
						uint32 const j2 = order[k];
						Sum const step = Sum(at(i2, j2) - v[j2] - tight);
						if (dist[j1] + step >= dist[j2]) continue;
						dist[j2] = Sum(dist[j1] + step);
						from_row[j2] = i2;
						if (step == 0) { // same plateau: promote on the spot
							if (r.col_to_row[j2] == none) {
								open = j2;
								break;
							}
							std::swap(order[k], order[labeled]);
							++labeled;
						}
					}
				}
				// columns scanned from `settled` onward sit on the final plateau
				// (dist == dist[open]), so their update is a no-op
				for (std::size_t k = 0; k < settled; ++k)
					v[order[k]] = Sum(v[order[k]] + (dist[order[k]] - dist[open]));
				for (uint32 j = open; j != none;) {
					uint32 const mover = from_row[j];
					r.col_to_row[j] = mover;
					std::swap(r.row_to_col[mover], j);
				}
			}

			r.row_potential.resize(rows);
			for (std::size_t i = 0; i < rows; ++i) {
				uint32 const j = r.row_to_col[i];
				r.row_potential[i] = Sum(at(i, j) - v[j]);
				r.cost = Sum(r.cost + at(i, j));
			}
			r.col_potential = std::move(v);
			return r;
		}

		/**
		 * @brief Adjacent-2x2 Monge test (adjacent inequalities imply the full
		 *        property): `c[i][j] + c[i+1][j+1] <= c[i][j+1] + c[i+1][j]`
		 *        everywhere means the identity permutation is optimal — the
		 *        classic exchange argument, uncrossing never helps. `+1` =
		 *        Monge, `-1` = anti-Monge (reversed permutation optimal), `0` =
		 *        neither.
		 * @pre @p cost is a square `n x n` row-major matrix
		 * @complexity O(n²)
		 */
		template<assignment_sum Sum, assignment_cost<Sum> C>
		[[nodiscard]] constexpr int monge_kind(std::size_t n, std::vector<C> const &cost) {
			bool is_monge = true, is_anti_monge = true;
			for (std::size_t i = 0; i + 1 < n && (is_monge || is_anti_monge); ++i)
				for (std::size_t j = 0; j + 1 < n; ++j) {
					Sum const straight = Sum(Sum(cost[i * n + j]) + Sum(cost[(i + 1) * n + j + 1]));
					Sum const crossed = Sum(Sum(cost[i * n + j + 1]) + Sum(cost[(i + 1) * n + j]));
					is_monge = is_monge && straight <= crossed;
					is_anti_monge = is_anti_monge && straight >= crossed;
					if (!is_monge && !is_anti_monge) return 0;
				}
			return is_monge ? 1 : is_anti_monge ? -1
												: 0;
		}

		/**
		 * @brief The diagonal matching of a Monge matrix with its dual, in
		 *        O(n) beyond the detection scan: `v` telescopes the diagonal's
		 *        column steps, and both feasibility directions follow from the
		 *        Monge row-difference monotonicity. @p should_flip mirrors
		 *        columns for the anti-Monge case (mirroring makes it Monge).
		 * @pre @p cost is a square `n x n` row-major matrix that `monge_kind`
		 *      classified as Monge (@p should_flip false) or anti-Monge (true)
		 * @complexity O(n)
		 */
		template<assignment_sum Sum, assignment_cost<Sum> C>
		[[nodiscard]] constexpr assignment_result<Sum> monge_diagonal(std::size_t n, std::vector<C> const &cost,
																	  bool should_flip) {
			assignment_result<Sum> r;
			r.row_to_col.resize(n);
			r.col_to_row.resize(n);
			r.row_potential.resize(n);
			r.col_potential.resize(n);
			auto const at = [&](std::size_t i, std::size_t j) {
				return Sum(cost[i * n + (should_flip ? n - 1 - j : j)]);
			};
			Sum acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				if (i > 0) acc = Sum(acc + (at(i, i) - at(i, i - 1)));
				std::size_t const j = should_flip ? n - 1 - i : i;
				r.row_to_col[i] = uint32(j);
				r.col_to_row[j] = uint32(i);
				r.col_potential[j] = acc;
				r.row_potential[i] = Sum(at(i, i) - acc);
				r.cost = Sum(r.cost + at(i, i));
			}
			return r;
		}
	}

	/**
	 * @brief The dense cost matrices @ref atl::graph::assignment accepts: a sized
	 *        random-access range of equally sized, sized random-access rows
	 *        of a common arithmetic cost type (iterated as `Matrix const`).
	 * @tparam Matrix Const-indexable dense numeric matrix.
	 */
	template<typename Matrix>
	concept cost_matrix =
		std::ranges::random_access_range<Matrix const> && std::ranges::sized_range<Matrix const>
		&& std::ranges::random_access_range<std::ranges::range_reference_t<Matrix const>>
		&& std::ranges::sized_range<std::ranges::range_reference_t<Matrix const>>
		&& std::numeric_limits<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>>::is_specialized
		&& std::semiregular<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>>
		&& requires(Matrix const &matrix) {
			   matrix[std::size_t(0)];
			   { matrix[std::size_t(0)][std::size_t(0)] } -> std::convertible_to<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>>;
		   };

	namespace detail {
		template<typename Sum, typename Matrix>
		using sum_t = std::conditional_t<
			std::is_void_v<Sum>,
			std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>, Sum>;
	}

	/**
	 * @brief The assignment problem: match every row of a dense cost matrix
	 *        to a distinct column (every column, when there are more rows
	 *        than columns — the matrix is transposed internally) minimizing
	 *        the total cost. Maximization is negation at the call site.
	 *
	 * Jonker-Volgenant: column reduction, reduction transfer and augmenting
	 * row reduction resolve most rows in a few matrix scans, and shortest
	 * augmenting paths (the JV scan partition over column potentials) finish
	 * the tail. Square Monge and anti-Monge matrices short-circuit to the
	 * diagonal in the O(n²) detection scan, dual certificate included.
	 * Costs may be negative; ties are fine. The preprocessing does not pay
	 * on every shape: plain shortest augmenting paths are ahead on tie-heavy
	 * and banded matrices (500x500 costs 0..9: 0.74 vs 0.92 ms; banded 0.46
	 * vs 0.66 ms; record 2026-10-15, Xeon Platinum 8280, plain -O2; see
	 * docs/assignment.md).
	 *
	 * Sums run in `Sum` (default: the matrix's cost type) — pass it
	 * explicitly to keep the matrix narrow but the arithmetic wide, e.g.
	 * `assignment<int64>(int_matrix)`: the solver scans the matrix many
	 * times, and a 32-bit matrix is half the memory traffic. `Sum` must be a
	 * signed integral type (`assignment_sum`: an unsigned matrix needs an
	 * explicit signed `Sum`; floating costs are not supported) and exactly represent every converted cost and intermediate.
	 * A sufficient arithmetic bound is `8 (k + 1) max(1, max|cost|) < max(Sum)`,
	 * where k is the smaller dimension. This covers sums, potential differences,
	 * Monge pair sums and the path search sentinel `max(Sum)/4`.
	 *
	 * The result is self-certifying by LP duality: `row_potential[i] +
	 * col_potential[j] <= cost[i][j]` with equality on matched pairs, and,
	 * when `rows != cols`, the larger side's potentials `<= 0` with `0` on
	 * its unmatched vertices — together a full optimality certificate for
	 * every shape (checked in tests on square and rectangular instances).
	 *
	 * @tparam Sum Signed integer arithmetic type, or void for the element type.
	 * @tparam Matrix Const sized random-access matrix with numeric elements.
	 * @param costs Equally sized rows; input storage is not retained or modified.
	 * @pre Both dimensions are below none, and their product fits vector max_size
	 * (asserted). All rows have equal size (asserted). Arithmetic obeys the bound above.
	 * @return Owned matching and exact dual certificate; ties have no fixed order.
	 * @complexity O(k² l + rows cols) time and O(rows cols + rows + cols) heap storage,
	 * where k=min(rows,cols), l=max(rows,cols); O(n²) for square Monge input.
	 * O(1) call stack; allocation failures propagate.
	 */
	template<typename Sum = void, cost_matrix Matrix>
		requires(assignment_sum<detail::sum_t<Sum, Matrix>>
				 && detail::assignment_cost<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>,
											detail::sum_t<Sum, Matrix>>)
	[[nodiscard]] constexpr auto assignment(Matrix const &costs) {
		using C = std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>;
		using S = detail::sum_t<Sum, Matrix>;
		std::size_t const rows = std::ranges::size(costs);
		std::size_t const cols = rows == 0 ? 0 : std::ranges::size(costs[0]);
		std::vector<C> flat;
		atl_assert(rows < none && cols < none);
		atl_assert(cols == 0 || rows <= flat.max_size() / cols);
		flat.reserve(rows * cols);
		bool const should_flip = rows > cols;
		if (!should_flip) {
			for (auto const &row : costs) {
				atl_assert(std::size_t(std::ranges::size(row)) == cols);
				for (auto const &x : row) flat.push_back(x);
			}
			if (rows == cols && rows > 0)
				if (int const kind = detail::monge_kind<S>(rows, flat); kind != 0)
					return detail::monge_diagonal<S>(rows, flat, kind < 0);
			return detail::dense_assignment<S>(rows, cols, flat, rows == cols);
		}
		flat.resize(rows * cols);
		for (std::size_t i = 0; i < rows; ++i) atl_assert(std::size_t(std::ranges::size(costs[i])) == cols);
		// transpose in 32x32 tiles: a row-order scatter keeps `cols` write
		// streams open at stride `rows` (1.9x slower at 2000x1000 int32,
		// record 2026-10-15, Xeon Platinum 8280, plain -O2)
		std::size_t constexpr tile = 32;
		for (std::size_t i0 = 0; i0 < rows; i0 += (std::min)(tile, rows - i0))
			for (std::size_t j0 = 0; j0 < cols; j0 += (std::min)(tile, cols - j0))
				for (std::size_t i = i0; i < i0 + (std::min)(tile, rows - i0); ++i)
					for (std::size_t j = j0; j < j0 + (std::min)(tile, cols - j0); ++j) flat[j * rows + i] = costs[i][j];
		auto r = detail::dense_assignment<S>(cols, rows, flat, false);
		std::swap(r.row_to_col, r.col_to_row);
		std::swap(r.row_potential, r.col_potential);
		return r;
	}
}
