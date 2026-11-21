#include <atl/strings/aho_corasick.hpp>
#include <array>
#include <iostream>
#include <string_view>

int main() {
	using std::string_view;
	auto const symbol = [](char c) { return c - 'a'; };
	atl::aho_corasick<26> const patterns(std::array{string_view("he"), string_view("she"), string_view("hers")}, symbol);
	auto const table = patterns.transition_table();
	auto state = patterns.root();
	std::size_t matches = 0;
	for (char const c : string_view("ushers")) {
		state = table(state, static_cast<atl::uint32>(symbol(c)));
		matches += patterns.match_count(state);
	}
	std::cout << matches << '\n';
}
