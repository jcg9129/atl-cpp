#include <atl/ds/dsu.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <iostream>

int main() {
	atl::dsu plain(3);
	plain.unite(0, 1);
	std::cout << plain.count() << '\n';

	atl::rollback_dsu undoable(3);
	auto const token = undoable.time();
	undoable.unite(0, 1);
	undoable.unite(0, 1);
	undoable.rollback(token);
	std::cout << undoable.count() << '\n';

	atl::persistent_dsu<> base(3);
	auto const left = base.unite(0, 1);
	auto const right = base.unite(1, 2);
	std::cout << base.count() << ' ' << left.same(0, 1) << ' ' << right.same(0, 1) << '\n';
}
