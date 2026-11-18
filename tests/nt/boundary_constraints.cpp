#include <atl/experimental/nt/bitwise_convolution_variants.hpp>
#include <atl/experimental/nt/convolution_variants.hpp>
#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/crt.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/int128.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <ostream>
#include <span>
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

TEST_CASE("nt constraints reject unsupported calls", "[constraints]") {
	STATIC_REQUIRE(!atl::congruence_range<std::vector<unsigned>>);
	STATIC_REQUIRE(atl::congruence_range<std::vector<atl::congruence<atl::uint128>>>);
}

namespace {
	template<typename Element>
	concept raw_scalar_transform = requires(Element *p) {
		atl::scalar_ntt::forward<atl::mod_int<Element, Element(3)>>(p, 1);
	};
	template<typename R>
	concept writable_ntt_range = requires(R &&r) { atl::ntt(r); atl::inverse_ntt(r); };
	template<typename R>
	concept writable_fft_range = requires(R &&r) { atl::fft(r); atl::inverse_fft(r); };
	template<typename R>
	concept inverse_walsh_range = requires(R &&r) { atl::inverse_walsh_hadamard(r); };
	template<typename A, typename B>
	concept modular_convolution_pair = requires(A const &a, B const &b) { atl::convolution(a, b); };
	using TransformElement = atl::mod_int<atl::uint32, 998244353>;
	static_assert(raw_scalar_transform<atl::uint32>);
	static_assert(writable_ntt_range<std::span<TransformElement>>);
	static_assert(!writable_ntt_range<std::span<TransformElement const>>);
	static_assert(!writable_ntt_range<std::vector<atl::mod_int<atl::uint32, 129>>>);
	static_assert(!writable_ntt_range<std::vector<atl::mod_int<atl::uint32, (atl::uint32(1) << 30) + 1>>>);
	static_assert(!writable_ntt_range<std::vector<atl::dynamic_mod_int<>>>);
	static_assert(!atl::scalar_ntt::supports<atl::mod_int<atl::uint32, 1>>());
	static_assert(!atl::scalar_ntt::supports<atl::mod_int<atl::uint32, 2>>());
	static_assert(atl::scalar_ntt::supports<atl::mod_int<atl::uint32, 3>>());
	static_assert(!atl::scalar_ntt::supports<atl::mod_int<atl::uint32, (atl::uint32(1) << 30) - 1>>());
	static_assert(!atl::scalar_ntt::supports<atl::mod_int<atl::uint32, atl::uint32(1) << 30>>());
	static_assert(writable_fft_range<std::vector<atl::cplx>>);
	static_assert(!writable_fft_range<std::vector<atl::cplx> const>);
	static_assert(!writable_fft_range<std::vector<double>>);
	static_assert(inverse_walsh_range<std::vector<atl::int128>>);
	static_assert(!inverse_walsh_range<std::vector<atl::uint128>>);
	static_assert(modular_convolution_pair<std::span<TransformElement const>, std::vector<TransformElement>>);
	static_assert(!modular_convolution_pair<std::vector<TransformElement>, std::vector<atl::mod_int<atl::uint32, 3>>>);
	static_assert(!atl::ntt_engine<int>);
	static_assert(!atl::set_function_range<std::vector<bool>>);
	static_assert(!atl::ring_range<std::vector<int>, long>);
}

namespace {
	template<typename Element>
	concept radix2_contender_for = requires(std::vector<Element> &v) {
		atl::experimental::ntt_by_radix2(v, false);
		atl::experimental::convolution_by_radix2(v, v);
	};
	template<typename Range>
	concept ring_product_range = requires(Range const &v) {
		atl::or_convolution(v, v);
		atl::and_convolution(v, v);
		atl::xor_convolution(v, v);
		atl::subset_convolution(v, v);
		atl::experimental::subset_convolution_by_rank_major(v, v);
	};
	static_assert(radix2_contender_for<atl::mod_int<atl::uint32, 3>>);
	static_assert(!radix2_contender_for<atl::mod_int<atl::uint32, 129>>);
	static_assert(!radix2_contender_for<atl::mod_int<atl::uint32, 4294967291u>>);
	static_assert(!ring_product_range<std::vector<std::string>>);
	static_assert(ring_product_range<std::span<int const>>);
	static_assert(!atl::detail::butterfly_of<int, int>);
	static_assert(!atl::detail::residue_accessor<int, TransformElement>);
}

#include <atl/experimental/nt/matrix_variants.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/matrix.hpp>

namespace {
	struct incomplete_semiring {
		using value_type = int;
	};

	struct invalid_sum_semiring {
		using value_type = int;
		static int zero();
		static int one();
		static void add(int, int);
		static int multiply(int, int);
	};

	struct invalid_identity_semiring {
		using value_type = int;
		static int zero();
		static void one();
		static int add(int, int);
		static int multiply(int, int);
	};

	struct invalid_product_semiring {
		using value_type = int;
		static int zero();
		static int one();
		static int add(int, int);
		static void multiply(int, int);
	};

	struct invalid_zero_semiring {
		using value_type = int;
		static void zero();
		static int one();
		static int add(int, int);
		static int multiply(int, int);
	};

	template<typename Element, typename Policy>
	concept matrix_policy_call = requires { typename atl::matrix<Element, Policy>; };
	template<typename Range>
	concept recurrence_range_call = requires(Range const &r) {
		atl::berlekamp_massey(r);
		atl::linear_recurrence_term(r, r, 1);
		atl::rational_series_coefficient(r, r, 1);
	};

	struct mutable_field_range {
		atl::mod_int32<998244353> *begin() { return nullptr; }

		atl::mod_int32<998244353> *end() { return nullptr; }
	};

	template<typename Element>
	concept field_elimination_call = requires(atl::matrix<Element> a) {
		atl::rank(a);
		atl::determinant(a);
		atl::inverse(a);
		atl::solve_linear_system(a, std::vector<Element>{});
	};
	template<typename Element, atl::uint32 Tile = 256>
	concept lazy_tile_call = requires(std::span<Element const> a) {
		atl::experimental::matrix_product_by_lazy_tiles<Element, Tile>(a, a, 0);
	};
	static_assert(!atl::semiring<incomplete_semiring>);
	static_assert(!atl::semiring<invalid_sum_semiring>);
	static_assert(!atl::semiring<invalid_identity_semiring>);
	static_assert(!atl::semiring<invalid_product_semiring>);
	static_assert(!atl::semiring<invalid_zero_semiring>);
	static_assert(!matrix_policy_call<long, atl::plus_times_semiring<int>>);
	static_assert(!recurrence_range_call<std::vector<int>>);
	static_assert(!recurrence_range_call<std::vector<double>>);
	static_assert(!recurrence_range_call<mutable_field_range>);
	static_assert(!field_elimination_call<int>);
	static_assert(!field_elimination_call<double>);
	static_assert(field_elimination_call<atl::mod_int32<7>>);
	static_assert(!lazy_tile_call<atl::mod_int64<18446744073709551557ull>>);
	static_assert(!lazy_tile_call<atl::mod_int32<7>, 0>);
}

namespace {
	struct incomparable_value {
		int value = 0;
	};

	struct incomparable_policy {
		using value_type = incomparable_value;
		static value_type zero();
		static value_type one();
		static value_type add(value_type, value_type);
		static value_type multiply(value_type, value_type);
	};

	template<typename Matrix>
	concept matrix_equality_call = requires(Matrix const &m) { m == m; };
	template<typename Element>
	concept lazy_transpose_call = requires(std::span<Element const> a) {
		atl::experimental::matrix_product_by_lazy_transpose<Element>(a, a, 0);
	};
	template<typename Element>
	concept lazy_subtract_call = requires(std::span<Element const> a) {
		atl::experimental::matrix_product_by_lazy_subtract<Element>(a, a, 0);
	};
	template<typename Element>
	concept lazy_kernel_call = requires(std::span<Element const> a) {
		atl::detail::lazy_product<Element>(a, a, 0, 0, 0);
	};
	using FullWidthElement = atl::mod_int64<18446744073709551557ull>;
	static_assert(atl::semiring<incomparable_policy>);
	static_assert(!matrix_equality_call<atl::matrix<incomparable_value, incomparable_policy>>);
	static_assert(matrix_equality_call<atl::matrix<int>>);
	static_assert(!lazy_transpose_call<FullWidthElement>);
	static_assert(!lazy_subtract_call<FullWidthElement>);
	static_assert(!lazy_kernel_call<FullWidthElement>);
	static_assert(lazy_kernel_call<atl::mod_int32<2>>);
	static_assert(!lazy_kernel_call<atl::mod_int32<1>>);
	static_assert(!atl::detail::pivot_rule<atl::detail::first_nonzero_pivot, int>);
}

namespace {
	template<typename Element>
	concept min_plus_policy_call = requires { typename atl::min_plus_semiring<Element>; };
	template<typename Element>
	concept max_plus_policy_call = requires { typename atl::max_plus_semiring<Element>; };
	template<typename Element>
	concept max_min_policy_call = requires { typename atl::max_min_semiring<Element>; };

	static_assert(!matrix_policy_call<bool, atl::plus_times_semiring<bool>>);
	static_assert(!min_plus_policy_call<bool>);
	static_assert(!max_plus_policy_call<bool>);
	static_assert(!max_min_policy_call<bool>);
	static_assert(!max_plus_policy_call<unsigned>);
	static_assert(!max_plus_policy_call<atl::uint128>);
	static_assert(!min_plus_policy_call<std::string>);
	static_assert(!max_plus_policy_call<std::string>);
	static_assert(!max_min_policy_call<std::string>);
}
