#pragma once

// Test helper: an element that counts copies/moves of a distinguished "sentinel"
// value, so the sibling copies done by a persistent structure's path copying do
// not pollute the count. Used to verify that set/push move the passed value into
// place rather than copying it. A use-after-move also shows up: a moved-from
// probe holds v == -1, so reading one back yields a wrong value.
namespace atl_test {
	struct move_probe {
		int v = 0;
		static constexpr int sentinel = 1'000'000;
		static inline int sentinel_copies = 0, sentinel_moves = 0;

		static void reset() {
			sentinel_copies = 0;
			sentinel_moves = 0;
		}

		move_probe() = default;

		move_probe(int x) : v(x) {}

		move_probe(move_probe const &o) : v(o.v) {
			if (o.v == sentinel) ++sentinel_copies;
		}

		move_probe(move_probe &&o) noexcept : v(o.v) {
			if (o.v == sentinel) ++sentinel_moves;
			o.v = -1;
		}

		move_probe &operator=(move_probe const &o) {
			v = o.v;
			if (o.v == sentinel) ++sentinel_copies;
			return *this;
		}

		move_probe &operator=(move_probe &&o) noexcept {
			v = o.v;
			if (o.v == sentinel) ++sentinel_moves;
			o.v = -1;
			return *this;
		}
	};
}
