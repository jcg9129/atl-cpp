#pragma once

/**
 * @file
 * @brief `atl_assert(condition)`, the library's precondition check.
 *
 * With `ATL_ASSERT` defined, a false `condition` prints itself with file and
 * line to `stderr` and calls `std::abort()`; reached during constant
 * evaluation, the failing branch is not a constant expression, so the
 * evaluation is a compile error. Without `ATL_ASSERT` the condition is never
 * evaluated (it is still type-checked, and the names it uses count as used),
 * so it must not carry side effects the program relies on. Use it as a
 * statement: both forms are `do ... while (0)` statements.
 */

// Kept at file scope (not inside the #ifdef) so amalgamation tools hoist them
// to the top of the generated file rather than leaving them inside namespace atl.
#include <cstdio>
#include <cstdlib>

#ifdef ATL_ASSERT
/**
 * @brief Check a precondition when `ATL_ASSERT` is defined; otherwise only type-check it.
 * @param condition Boolean-testable expression without required side effects.
 * A false enabled check prints the expression and source location and aborts;
 * during constant evaluation it prevents a constant-expression result.
 * Use as a statement. Disabled checks never evaluate the expression.
 * @complexity The cost of evaluating condition when enabled; O(1) otherwise.
 */
#define atl_assert(condition)                                                                             \
	do {                                                                                                  \
		if (!(condition)) {                                                                               \
			std::fprintf(stderr, "atl_assert: (%s), file %s, line %d\n", #condition, __FILE__, __LINE__); \
			std::abort();                                                                                 \
		}                                                                                                 \
	} while (0)
#else
// Unevaluated, so free at runtime and in constant evaluation.
/**
 * @brief Check a precondition when `ATL_ASSERT` is defined; otherwise only type-check it.
 * @param condition Boolean-testable expression without required side effects.
 * A false enabled check prints the expression and source location and aborts;
 * during constant evaluation it prevents a constant-expression result.
 * Use as a statement. Disabled checks never evaluate the expression.
 * @complexity The cost of evaluating condition when enabled; O(1) otherwise.
 */
#define atl_assert(condition)          \
	do {                               \
		(void)sizeof(bool(condition)); \
	} while (0)
#endif
