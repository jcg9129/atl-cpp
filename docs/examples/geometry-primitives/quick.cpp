#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	// int64 coordinates: the turn is decided on int128 products, exactly
	geo::point<atl::int64> const o{0, 0}, a{4'000'000'000, 1}, b{8'000'000'000, 2};
	std::cout << geo::orientation(o, a, b) << ' ' << geo::orientation(o, a, geo::point<atl::int64>{8'000'000'000, 3}) << '\n';

	// doubles are exact rationals, and the default policy signs them exactly
	geo::point<double> const p{0.1, 0.1}, q{0.3, 0.3}, r{0.5, 0.5000000000000001};
	std::cout << geo::orientation(p, q, r) << ' ' << geo::orientation(p, q, geo::point<double>{0.5, 0.5}) << '\n';

	// constructed values get an explicit tolerance instead
	geo::approx const tolerance{1e-9};
	std::cout << geo::orientation(p, q, r, tolerance) << ' ' << tolerance.equal(r, geo::point<double>{0.5, 0.5}) << '\n';
}
