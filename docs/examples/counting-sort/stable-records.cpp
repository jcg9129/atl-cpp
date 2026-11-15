#include <atl/utils/counting_sort.hpp>
#include <array>
#include <iostream>

struct record {
	int priority;
	char label;
};

int main() {
	std::array<record, 4> records{{{2, 'a'}, {-1, 'b'}, {2, 'c'}, {0, 'd'}}};
	atl::counting_sort(records, &record::priority);
	for (auto const &item : records) std::cout << item.priority << ':' << item.label << '\n';
	atl::counting_sort_clear();
}
