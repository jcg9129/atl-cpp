#pragma once
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

namespace atl_test {
	/**
	 * @brief Runs @p body in a forked child and reports whether the child
	 *        died from SIGABRT — which is what a failing `atl_assert` does.
	 *        The way to pin a precondition that is enforced by assertion:
	 *        `REQUIRE(atl_test::aborts([] { ... }));`
	 *
	 * The child's stderr is silenced, so the assert's message does not land
	 * in the test log, and SIGABRT is reset to its default action so Catch's
	 * crash handler does not print a failure report from the child. A body
	 * that runs past 60 seconds is killed by SIGALRM and counts as not
	 * aborting, so a hang fails the test instead of stalling the suite.
	 * POSIX only.
	 */
	template<typename Body>
	bool aborts(Body &&body) {
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
	}
}
