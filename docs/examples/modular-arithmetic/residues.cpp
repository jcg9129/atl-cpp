#include <atl/nt/constants.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <iostream>

int main() {
	using Z = atl::mod_int32<998244353>;
	std::cout << (Z(5) * Z(7) - 1).value() << '\n';
	using D = atl::dynamic_mod_int64<>;
	D::set_mod(atl::largest_prime_64);
	std::cout << (D(-1) * D(-1)).value() << '\n';
	using engine = atl::montgomery<atl::largest_prime_64>;
	std::cout << engine::from_form(engine::power(engine::to_form(2), 64)) << '\n';
	using wide = atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>;
	std::cout << atl::uint64((wide(atl::uint128(1) << 88) * wide(4)).value()) << '\n';
	std::cout << atl::static_modulus<atl::largest_prime_32>::mul(4294967290u, 4294967290u) << '\n';
	std::cout << atl::barrett(7).mul(3, 5) << ' ' << atl::reduce_mod(-1, 7u) << '\n';
}
