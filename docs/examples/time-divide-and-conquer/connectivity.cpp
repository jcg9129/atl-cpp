#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <iostream>
#include <utility>
#include <vector>

int main() {
	using edge = std::pair<atl::uint32, atl::uint32>;
	std::vector<atl::item_event<edge>> const events{
		{atl::event_kind::add, {0, 1}},
		{atl::event_kind::add, {1, 2}},
		{atl::event_kind::query, {}},
		{atl::event_kind::remove, {0, 1}},
		{atl::event_kind::query, {}}};
	auto const lifetimes = atl::lifetimes_of(events);
	atl::rollback_dsu state(3);
	atl::time_divide_and_conquer(
		state, events.size(), lifetimes,
		[&](atl::rollback_dsu &s, atl::uint32 item) {
			auto const [u, v] = events[lifetimes[item].begin].key;
			s.unite(u, v);
		},
		[](atl::rollback_dsu &s, atl::uint32 time) {
			std::cout << time << ':' << s.count() << '\n';
		},
		[&](atl::uint32 time) { return events[time].kind == atl::event_kind::query; });
	std::cout << "restored:" << state.count() << '\n';
}
