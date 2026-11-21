#include <atl/strings/suffix_tree.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::suffix_tree<26> tree;
	tree.insert(std::string_view("banana"), [](char value) { return value - 'a'; });
	auto const first = tree.walk(tree.root(), 0);
	if (!first) return 1;
	auto const edge = tree.edge(*first);
	std::cout << tree.word(edge.word).size() << ' ' << edge.offset << ' '
			  << edge.length << '\n';
	std::cout << tree.occurrence_counts()[*first] << '\n';
}
