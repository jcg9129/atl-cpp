#include <atl/io/fast_io.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <cstdio>

int main() {
	using residue = atl::mod_int32<7>;
	std::FILE *file = std::tmpfile();
	if (!file) return 1;
	{
		atl::io::writer output(file);
		output.writeln(3, 10, -20, 30);
		output.writeln((atl::uint128(1) << 100) + 7);
		output.writeln(residue(-1));
	}
	std::rewind(file);
	{
		atl::io::reader input(file);
		auto values = input.read_n<int>(input.read<unsigned>());
		int sum = 0;
		for (int value : values) sum += value;
		atl::io::writeln(values);
		atl::io::writeln(sum, input.read<atl::uint128>());
		atl::io::writeln(input.read<residue>() + residue(3));
		if (!input.eof()) return 2;
	}
	std::fclose(file);
}
