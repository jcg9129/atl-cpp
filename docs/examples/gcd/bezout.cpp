#include <atl/nt/gcd.hpp>
#include <iostream>

int main() {
	auto const [gcd, x, y] = atl::extended_gcd(240u, 46u);
	std::cout << gcd << ' ' << x << ' ' << y << '\n';
	auto const result = atl::extended_gcd(-12, 18);
	std::cout << result.gcd << ' ' << (-12 * result.x + 18 * result.y) << '\n';
}
