#include <atl/strings/suffix_automaton.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::suffix_automaton<26> automaton;
	auto const letter = [](char value) { return value - 'a'; };
	automaton.insert(std::string_view("banana"), letter);
	std::cout << automaton.contains(std::string_view("ana"), letter) << ' '
			  << automaton.distinct_substring_count() << '\n';
	auto state = automaton.root();
	for (char value : std::string_view("ana")) {
		auto const next = automaton.walk(state, letter(value));
		if (!next) return 1;
		state = *next;
	}
	std::cout << automaton.occurrence_counts()[state] << '\n';
}
