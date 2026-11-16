#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <iostream>

int main() {
	atl::weighted_dsu<> weights(4);
	weights.unite(0, 1, 9);
	weights.unite(1, 2, 90);
	std::cout << *weights.difference(0, 2) << '\n';
	std::cout << (weights.unite(2, 0, 123) == atl::unite_outcome::contradicts) << '\n';

	atl::rollback_weighted_dsu<> history(3);
	history.unite(0, 1, 7);
	auto const token = history.time();
	history.unite(1, 2, 5);
	std::cout << *history.difference(0, 2) << '\n';
	history.rollback(token);
	std::cout << history.difference(0, 2).has_value() << '\n';
}
