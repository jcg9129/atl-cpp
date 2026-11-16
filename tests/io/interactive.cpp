// A separate executable keeps the opt-in refill policy consistent across its TUs.
#define ATL_IO_INTERACTIVE
#include <atl/io/fast_io.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <cerrno>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using atl::int64;

namespace {
	bool serve_peer(std::FILE *source, std::FILE *sink) {
		atl::io::reader r(source);
		atl::io::writer w(sink);
		for (int64 q = 1; q <= 3; ++q) {
			int64 const value = r.read<int64>();
			if (value != q * 1'000'003) return false;
			w.writeln(2 * value);
			w.flush();
		}
		return r.eof();
	}

#ifdef _WIN32
	using pipe_handle = HANDLE;
	constexpr pipe_handle invalid_pipe = nullptr;

	void close_pipe(pipe_handle handle) { CloseHandle(handle); }

	bool write_bytes(pipe_handle handle, char const *data, std::size_t size) {
		DWORD written = 0;
		return WriteFile(handle, data, static_cast<DWORD>(size), &written, nullptr) && written == size;
	}

	bool read_byte(pipe_handle handle, char &c) {
		auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
		while (std::chrono::steady_clock::now() < deadline) {
			DWORD available = 0;
			if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) return false;
			if (available != 0) {
				DWORD got = 0;
				return ReadFile(handle, &c, 1, &got, nullptr) && got == 1;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		return false;
	}

	struct peer_process {
		HANDLE process = nullptr;

		~peer_process() {
			if (process) {
				// Also cleans up after a failed REQUIRE in the parent.
				TerminateProcess(process, 1);
				WaitForSingleObject(process, 20000);
				CloseHandle(process);
			}
		}

		bool start(pipe_handle query, pipe_handle reply, pipe_handle, pipe_handle) {
			std::wstring executable(32768, L'\0');
			DWORD const length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
			if (length == 0 || length >= executable.size()) return false;
			executable.resize(length);
			// An executable path cannot contain quotes or end in a backslash.
			// Keep Catch reporter output out of the reply pipe.
			std::wstring command = L"\"" + executable + L"\" --out NUL";
			wchar_t *inherited = GetEnvironmentStringsW();
			if (!inherited) return false;
			std::vector<std::wstring> variables;
			for (wchar_t const *entry = inherited; *entry;) {
				std::wstring const variable(entry);
				entry += variable.size() + 1;
				if (_wcsnicmp(variable.c_str(), L"ATL_IO_TEST_CHILD=", 18) != 0) variables.push_back(variable);
			}
			FreeEnvironmentStringsW(inherited);
			variables.emplace_back(L"ATL_IO_TEST_CHILD=1");
			std::ranges::sort(variables, [](std::wstring const &left, std::wstring const &right) { return _wcsicmp(left.c_str(), right.c_str()) < 0; });
			std::vector<wchar_t> environment;
			for (auto const &variable : variables) {
				environment.insert(environment.end(), variable.begin(), variable.end());
				environment.push_back(L'\0');
			}
			environment.push_back(L'\0');
			STARTUPINFOW startup{};
			startup.cb = static_cast<DWORD>(sizeof startup);
			startup.dwFlags = STARTF_USESTDHANDLES;
			startup.hStdInput = query;
			startup.hStdOutput = reply;
			startup.hStdError = reply;
			PROCESS_INFORMATION info{};
			if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
								CREATE_UNICODE_ENVIRONMENT, environment.data(), nullptr, &startup, &info)) return false;
			process = info.hProcess;
			CloseHandle(info.hThread);
			return true;
		}

		bool succeeded() {
			DWORD code = 1;
			return WaitForSingleObject(process, 20000) == WAIT_OBJECT_0 && GetExitCodeProcess(process, &code) && code == 0;
		}
	};
#else
	using pipe_handle = int;
	constexpr pipe_handle invalid_pipe = -1;

	void close_pipe(pipe_handle handle) { ::close(handle); }

	bool write_bytes(pipe_handle handle, char const *data, std::size_t size) {
		while (size != 0) {
			::ssize_t const written = ::write(handle, data, size);
			if (written < 0 && errno == EINTR) continue;
			if (written <= 0) return false;
			data += written;
			size -= std::size_t(written);
		}
		return true;
	}

	bool read_byte(pipe_handle handle, char &c) {
		pollfd descriptor{handle, POLLIN, 0};
		int ready;
		do ready = ::poll(&descriptor, 1, 20000);
		while (ready < 0 && errno == EINTR);
		return ready == 1 && ::read(handle, &c, 1) == 1;
	}

	struct peer_process {
		pid_t process = -1;

		~peer_process() {
			if (process > 0) {
				::kill(process, SIGKILL);
				::waitpid(process, nullptr, 0);
			}
		}

		bool start(pipe_handle query, pipe_handle reply, pipe_handle parent_query, pipe_handle parent_reply) {
			std::fflush(nullptr);
			process = ::fork();
			if (process == 0) {
				::close(parent_query);
				::close(parent_reply);
				::alarm(20);
				std::FILE *source = ::fdopen(query, "r");
				std::FILE *sink = ::fdopen(reply, "w");
				std::_Exit(source && sink && serve_peer(source, sink) ? 0 : 1);
			}
			return process > 0;
		}

		bool succeeded() {
			int status = 0;
			pid_t const waited = ::waitpid(process, &status, 0);
			if (waited != process) return false;
			process = -1;
			return WIFEXITED(status) && WEXITSTATUS(status) == 0;
		}
	};
#endif

	struct pipe_pair {
		pipe_handle read = invalid_pipe, write = invalid_pipe;

		~pipe_pair() {
			if (read != invalid_pipe) close_pipe(read);
			if (write != invalid_pipe) close_pipe(write);
		}

		bool open() {
#ifdef _WIN32
			SECURITY_ATTRIBUTES attributes{static_cast<DWORD>(sizeof(SECURITY_ATTRIBUTES)), nullptr, TRUE};
			return CreatePipe(&read, &write, &attributes, 0) != 0;
#else
			int descriptors[2];
			if (::pipe(descriptors) != 0) return false;
			read = descriptors[0];
			write = descriptors[1];
			return true;
#endif
		}
	};

	std::string read_line(pipe_handle handle) {
		std::string line;
		char c;
		while (line.size() < 64 && read_byte(handle, c)) {
			if (c == '\n') return line;
			line += c;
		}
		return {};
	}
}

TEST_CASE("io: an interactive reader answers a peer that waits for each reply", "[io]") {
#ifdef _WIN32
	char marker[2]{};
	if (GetEnvironmentVariableA("ATL_IO_TEST_CHILD", marker, static_cast<DWORD>(sizeof marker)) != 0) {
		// stdin stays in text mode: the reader must switch it to binary itself (CR-terminated queries below).
		if (::_setmode(::_fileno(stdout), _O_BINARY) == -1) std::_Exit(1);
		std::_Exit(serve_peer(stdin, stdout) ? 0 : 1);
	}
#endif
	pipe_pair queries, replies;
	REQUIRE(queries.open());
	REQUIRE(replies.open());
#ifdef _WIN32
	REQUIRE(SetHandleInformation(queries.write, HANDLE_FLAG_INHERIT, 0));
	REQUIRE(SetHandleInformation(replies.read, HANDLE_FLAG_INHERIT, 0));
#endif
	peer_process peer;
	REQUIRE(peer.start(queries.read, replies.write, queries.write, replies.read));
	close_pipe(queries.read);
	queries.read = invalid_pipe;
	close_pipe(replies.write);
	replies.write = invalid_pipe;
	for (int64 q = 1; q <= 3; ++q) {
		// Split tokens force a second short read while the query pipe stays open.
		std::string const text = std::to_string(q * 1'000'003) + (q % 2 ? "\r" : "\n"); // a lone CR must not stall the reader
		REQUIRE(write_bytes(queries.write, text.data(), 3));
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		REQUIRE(write_bytes(queries.write, text.data() + 3, text.size() - 3));
		REQUIRE(read_line(replies.read) == std::to_string(2 * q * 1'000'003));
	}
	close_pipe(queries.write);
	queries.write = invalid_pipe;
	REQUIRE(peer.succeeded());
}

TEST_CASE("io: the interactive reader still parses ordinary input", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("  10\t-20\r\n hello 123456789012345678 x", fp);
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		auto const [a, b] = r.read<int, int>();
		REQUIRE(a == 10);
		REQUIRE(b == -20);
		REQUIRE(r.read<std::string>() == "hello");
		REQUIRE(r.read<int64>() == 123456789012345678);
		REQUIRE(r.read<char>() == 'x');
		REQUIRE(r.eof());
	}
	std::fclose(fp);
}
