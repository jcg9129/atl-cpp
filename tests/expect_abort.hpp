#pragma once
#include <concepts>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <algorithm>
#include <catch2/catch.hpp>
#include <charconv>
#include <cwchar>
#include <string>
#include <string_view>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <crtdbg.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace atl_test {
#ifdef _WIN32
	namespace detail {
		inline constexpr wchar_t abort_index_variable[] = L"ATL_TEST_ABORT_INDEX";

		inline std::size_t next_abort_index(std::string const &test_name) {
			// Shared across Body specializations; Catch repeats sections within one test case.
			static std::string previous_test;
			static std::size_t index = 0;
			if (previous_test != test_name) {
				previous_test = test_name;
				index = 0;
			}
			return index++;
		}

		inline std::wstring quote_argument(std::wstring_view argument) {
			// The CRT halves backslashes before quotes, including the closing quote.
			std::wstring quoted = L"\"";
			std::size_t backslashes = 0;
			for (wchar_t const c : argument) {
				if (c == L'\\') {
					++backslashes;
					continue;
				}
				quoted.append(backslashes * (c == L'"' ? 2 : 1), L'\\');
				if (c == L'"') quoted += L'\\';
				quoted += c;
				backslashes = 0;
			}
			quoted.append(backslashes * 2, L'\\');
			return quoted + L'"';
		}

		inline bool reexec_aborts(std::string const &test_name, std::size_t index) {
			std::wstring executable(32768, L'\0');
			DWORD const length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
			if (length == 0 || length >= executable.size()) return false;
			executable.resize(length);

			std::string spec = "\"";
			for (char const c : test_name) {
				if (std::string_view("\\\"[],*~").find(c) != std::string_view::npos) spec += '\\';
				spec += c;
			}
			spec += '"';
			// -1 includes the terminator and avoids narrowing a size_t to an int.
			int const wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, spec.c_str(), -1, nullptr, 0);
			if (wide_length == 0) return false;
			std::wstring wide_spec(static_cast<std::size_t>(wide_length), L'\0');
			if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, spec.c_str(), -1, wide_spec.data(), wide_length)) return false;
			wide_spec.pop_back();
			std::wstring command = quote_argument(executable) + L" " + quote_argument(wide_spec);

			// A private environment block avoids changing the parent's environment.
			wchar_t *const inherited = GetEnvironmentStringsW();
			if (!inherited) return false;
			std::vector<std::wstring> variables;
			std::wstring const prefix = std::wstring(abort_index_variable) + L'=';
			for (wchar_t const *entry = inherited; *entry; entry += std::wcslen(entry) + 1) {
				if (_wcsnicmp(entry, prefix.c_str(), prefix.size()) == 0) continue;
				variables.emplace_back(entry);
			}
			FreeEnvironmentStringsW(inherited);
			variables.push_back(prefix + std::to_wstring(index));
			std::ranges::sort(variables, [](auto const &a, auto const &b) { return _wcsicmp(a.c_str(), b.c_str()) < 0; });
			std::vector<wchar_t> environment;
			for (auto const &variable : variables) {
				environment.insert(environment.end(), variable.begin(), variable.end());
				environment.push_back(L'\0');
			}
			environment.push_back(L'\0');

			STARTUPINFOW startup{};
			startup.cb = static_cast<DWORD>(sizeof(startup));
			PROCESS_INFORMATION process{};
			if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE,
								CREATE_UNICODE_ENVIRONMENT, environment.data(), nullptr, &startup, &process)) return false;
			CloseHandle(process.hThread);
			DWORD const wait = WaitForSingleObject(process.hProcess, 60000);
			DWORD exit_code = 0;
			bool const aborted = wait == WAIT_OBJECT_0 && GetExitCodeProcess(process.hProcess, &exit_code) && exit_code == 3;
			if (wait != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 87);
			CloseHandle(process.hProcess);
			return aborted;
		}
	}
#endif

	/**
	 * @brief Runs @p body in a child and reports whether it aborted,
	 *        which is what a failing `atl_assert` does.
	 *        The way to pin a precondition that is enforced by assertion:
	 *        `REQUIRE(atl_test::aborts([] { ... }));`
	 *
	 * POSIX forks and checks for SIGABRT. Windows re-executes the current Catch
	 * test case with an escaped name spec and ATL_TEST_ABORT_INDEX selecting a
	 * per-test-case invocation index (shared by all Body types). Earlier calls
	 * return true without running their bodies; the selected call runs directly.
	 * Thus Windows test setup and the order of aborts calls must be reproducible,
	 * including section/generator traversal. A returning body exits with 86;
	 * MSVC's abort exit code 3 alone counts as aborting. CRT dialogs are disabled.
	 * The child's stderr is silenced and SIGABRT reset so Catch's crash handler
	 * does not print a failure report. A 60-second timeout kills the child and
	 * counts as not aborting (SIGALRM on POSIX, TerminateProcess on Windows).
	 */
	template<typename Body>
		requires std::invocable<Body &>
	bool aborts(Body &&body) {
#ifdef _WIN32
		std::string const test_name = Catch::getResultCapture().getCurrentTestName();
		std::size_t const index = detail::next_abort_index(test_name);
		char selection[32]{};
		DWORD const length = GetEnvironmentVariableA("ATL_TEST_ABORT_INDEX", selection, static_cast<DWORD>(sizeof(selection)));
		if (length != 0) {
			if (length >= sizeof(selection)) std::_Exit(86);
			std::size_t target = 0;
			auto const parsed = std::from_chars(selection, selection + length, target);
			if (parsed.ec != std::errc{} || parsed.ptr != selection + length) std::_Exit(86);
			if (index < target) return true;
			if (index != target) std::_Exit(86);
			FILE *silenced = nullptr;
			if (freopen_s(&silenced, "NUL", "w", stderr) != 0) std::_Exit(86);
			std::signal(SIGABRT, SIG_DFL);
			_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
			static_cast<void>(_CrtSetReportMode(_CRT_WARN, 0));
			static_cast<void>(_CrtSetReportMode(_CRT_ERROR, 0));
			static_cast<void>(_CrtSetReportMode(_CRT_ASSERT, 0));
			SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
			body();
			std::_Exit(86);
		}
		return detail::reexec_aborts(test_name, index);
#else
		std::fflush(nullptr); // or buffered output would be written twice
		pid_t const child = fork();
		if (child == 0) {
			if (!std::freopen("/dev/null", "w", stderr)) std::_Exit(2);
			std::signal(SIGABRT, SIG_DFL);
			alarm(60);
			body();
			std::_Exit(0);
		}
		int status = 0;
		if (child < 0 || waitpid(child, &status, 0) != child) return false;
		return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
#endif
	}
}
