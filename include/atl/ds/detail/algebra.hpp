#pragma once
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace atl::detail {
	template<typename Result, typename Callable, typename... Args>
	concept algebra_invocation = std::invocable<Callable const &, Args...> && std::convertible_to<std::invoke_result_t<Callable const &, Args...>, Result>;

	/** @brief Preserve temporaries and reference results; const operands remain available
	 * when an algebra deliberately deletes its mutable-operand overload. */
	template<typename Result, typename Callable, typename... Args>
		requires algebra_invocation<Result, Callable, Args &&...> || algebra_invocation<Result, Callable, std::remove_reference_t<Args> const &...>
	constexpr decltype(auto) invoke_algebra(Callable const &callable, Args &&...args) {
		if constexpr (algebra_invocation<Result, Callable, Args &&...>)
			return std::invoke(callable, std::forward<Args>(args)...);
		else
			return std::invoke(callable, std::as_const(args)...);
	}

	/** @brief A deleted move constructor must not hide a usable copy constructor. */
	template<typename Callable>
		requires std::is_move_constructible_v<Callable> || std::is_copy_constructible_v<Callable>
	constexpr decltype(auto) move_or_copy(Callable &callable) noexcept {
		if constexpr (std::is_move_constructible_v<Callable>)
			return std::move(callable);
		else
			return (callable);
	}
}
