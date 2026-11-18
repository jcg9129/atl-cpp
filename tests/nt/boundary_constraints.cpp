#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/crt.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/static_modulus.hpp>
#include <concepts>
#include <ostream>
#include <string>
#include <vector>

namespace {
	template<typename Base, typename Exponent>
	concept default_power_call = requires(Base base, Exponent exp) { atl::power(base, exp); };

	template<typename Base, typename Exponent>
	concept identity_power_call = requires(Base base, Exponent exp) { atl::power(base, exp, base); };

	struct no_assignment {
		explicit no_assignment(int);
		no_assignment(no_assignment const &) = default;
		no_assignment &operator=(no_assignment const &) = delete;
		friend no_assignment operator*(no_assignment const &, no_assignment const &);
	};

	struct no_product {};

	template<typename Value>
	concept stream_insertable = requires(std::ostream &stream, Value const &value) { stream << value; };

	static_assert(stream_insertable<atl::mod_int32<7>>);
	static_assert(stream_insertable<atl::mod_int64<18446744073709551557ull>>);
	static_assert(stream_insertable<atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>>);
	// The concrete software type remains admissible when uint128 aliases the native type.
	using Software = atl::detail::software_uint128;
	static_assert(std::same_as<Software, atl::uint128> || !stream_insertable<atl::mod_int<Software, Software(7)>>);

	static_assert(!default_power_call<no_assignment, unsigned>);
	static_assert(!identity_power_call<no_assignment, unsigned>);
	static_assert(!default_power_call<no_product, unsigned>);
	static_assert(!identity_power_call<no_product, unsigned>);
	static_assert(!default_power_call<int, int>);
	static_assert(!identity_power_call<int, int>);
	static_assert(!default_power_call<std::string, unsigned>);

	template<auto Modulus>
	concept static_modulus_type = requires { typename atl::static_modulus<Modulus>; };

	template<auto Modulus>
	concept montgomery_type = requires { typename atl::montgomery<Modulus>; };

	template<auto Modulus>
	concept modulus_engine_type = requires { typename atl::modulus_engine<Modulus>; };

	template<auto Modulus>
	concept lazy_montgomery = requires(decltype(Modulus) a) {
		atl::montgomery<Modulus>::reduce_lazy(atl::uint128(0));
		atl::montgomery<Modulus>::mul_lazy(a, a);
		atl::montgomery<Modulus>::mul_lazy_fixed(a, a, a);
	};

	static_assert(!static_modulus_type<0u> && !static_modulus_type<7>);
	static_assert(!montgomery_type<0u> && !montgomery_type<2u> && !montgomery_type<7>);
	static_assert(!montgomery_type<(atl::uint128(1) << 89) - 1>);
	static_assert(!modulus_engine_type<0u> && !modulus_engine_type<7>);
	static_assert(lazy_montgomery<(atl::uint64(1) << 62) - 1>);
	static_assert(!lazy_montgomery<(atl::uint64(1) << 62) + 1>);

	template<typename Integer>
	concept dynamic_modulus_type = requires { typename atl::dynamic_mod_int<Integer>; };

	template<typename Integer>
	concept integral_modular_calls = requires(Integer value) {
		atl::extended_gcd(value, value);
		atl::reduce_mod(value, 7u);
		atl::mod_int32<7>(value);
		atl::mod_int32<7>::unchecked(value);
		atl::dynamic_mod_int32<>(value);
		atl::dynamic_mod_int32<>::unchecked(value);
	};

	template<typename Integer>
	concept unsigned_modular_calls = requires(Integer value) {
		atl::add_mod_unchecked(value, value, value);
		atl::sub_mod_unchecked(value, value, value);
		atl::mul_mod_unchecked(value, value, value);
		atl::mul_mod_binary(value, value, value);
		atl::add_mod(value, value, value);
		atl::sub_mod(value, value, value);
		atl::mul_mod(value, value, value);
		atl::pow_mod(value, 0, value);
		atl::neg_mod(value, value);
		atl::incr_mod(value, value);
		atl::decr_mod(value, value);
		atl::inverse_mod(value, value);
	};

	static_assert(dynamic_modulus_type<atl::uint32> && dynamic_modulus_type<atl::uint64>);
	static_assert(!dynamic_modulus_type<int> && !dynamic_modulus_type<atl::uint128>);
	static_assert(integral_modular_calls<int> && integral_modular_calls<atl::uint128>);
	static_assert(!integral_modular_calls<double> && !integral_modular_calls<bool>);
	static_assert(unsigned_modular_calls<unsigned> && unsigned_modular_calls<atl::uint128>);
	static_assert(!unsigned_modular_calls<int> && !unsigned_modular_calls<double>);
}

namespace x2_boundaries {
	template<typename Value, typename Rule>
	concept multiplicative_call = requires(Rule rule) { atl::multiplicative_fn_range<Value>(13, rule); };
	template<typename Value>
	concept divisor_power_call = requires { atl::divisor_power_sum_range<Value>(13, 2); };
	template<typename Value, typename Operation>
	concept staircase_call = requires(Value value, Operation operation) {
		atl::floor_staircase_product(1, 1, 1, 1, value, value, value, operation);
	};

	struct no_product {};

	struct not_copyable {
		not_copyable() = default;
		not_copyable(not_copyable const &) = delete;
	};

	struct bad_result {
		void operator()(unsigned, unsigned) const;
	};

	struct move_rule {
		not_copyable operator()(not_copyable const &, not_copyable const &) const;
	};

	struct multiply_only {
		multiply_only() = default;
		explicit multiply_only(int);
		friend multiply_only operator*(multiply_only const &, multiply_only const &);
	};

	struct unassignable_value {
		unassignable_value() = default;
		unassignable_value(unassignable_value const &) = default;
		unassignable_value &operator=(unassignable_value const &) = delete;
	};

	struct unassignable_operation {
		unassignable_value operator()(unassignable_value const &, unassignable_value const &) const;
	};
	struct invalid_step;

	struct invalid_step_value {
		invalid_step_value() = default;
		explicit invalid_step_value(atl::uint64);
		friend invalid_step_value operator*(invalid_step_value const &, invalid_step_value const &);
		friend invalid_step_value operator+(invalid_step_value const &, int);
		friend invalid_step operator-(invalid_step_value const &, int);
		friend void operator*(invalid_step_value const &, invalid_step &);
	};

	struct invalid_step {
		operator invalid_step_value() const;
	};

	template<typename Rule>
	concept fused_multiplicative_call = requires(Rule rule) {
		atl::experimental::multiplicative_fn_range_by_fused_sieve<atl::uint64>(13, rule);
	};

	struct mismatched_engine_results {
		unsigned operator()(atl::detail::barrett_multiplier<atl::barrett> const &) const;
		atl::uint64 operator()(atl::detail::barrett_multiplier<atl::barrett64> const &) const;
		atl::uint64 operator()(atl::detail::montgomery64 const &) const;
	};

	template<typename Body>
	concept engine_dispatch_call = requires(Body const &body) { atl::detail::with_engine(7, body); };
	static_assert(!engine_dispatch_call<mismatched_engine_results>);

	static_assert(atl::binary_operation<unassignable_operation, unassignable_value>);
	static_assert(!staircase_call<unassignable_value, unassignable_operation>);
	static_assert(atl::detail::multiplicative_value<invalid_step_value>);
	static_assert(!divisor_power_call<invalid_step_value>);
	static_assert(!fused_multiplicative_call<int>);

	static_assert(!multiplicative_call<unsigned, int>);
	static_assert(!multiplicative_call<no_product, int>);
	static_assert(!divisor_power_call<multiply_only>);
	static_assert(!staircase_call<unsigned, bad_result>);
	static_assert(!staircase_call<not_copyable, move_rule>);
	static_assert(!atl::congruence_range<std::vector<unsigned>>);
	static_assert(!atl::congruence_range<unsigned>);
	static_assert(atl::congruence_range<std::vector<atl::congruence<atl::uint128>>>);
}

namespace {
	// Constraint-only arithmetic needs a definition to stay warning-free across compilers.
	[[maybe_unused]] no_assignment operator*(no_assignment const &value, no_assignment const &) { return value; }
}
