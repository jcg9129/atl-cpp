# Library conventions

The rules the library holds itself to, extending README's Code style section.
Document new conventions here alongside their technical rationale.

## Interface shape

- Algorithms are free functions returning result structs
  (`vertex_partition`, `block_decomposition`, `simplicity`, `circle_cover`);
  classes only for genuinely stateful structures.
- Names spelled out in full unless the acronym is universal on its own:
  `scc`, `dsu`, `csr` yes; `cc`, `geo` no. Library namespaces spelled out
  (`atl::geometry`, never `atl::geo`) — short forms are the USER's
  solution-side alias, not the library's.
- Template parameters named by role (`Base`, `Exponent`, `Mod`, `Compare`,
  `Flow`, `Cost`, `Policy`, `Graph`, `EdgeList`); `T`/`U` only for
  genuinely generic types. One exemption: `R` for a range whose role is
  already spelled by the named concept constraining it
  (`template<vertex_range R>`).
- A boolean's name reads as a true-or-false statement: `is_`, `has_`,
  `can_`, `should_` plus the property (`is_mersenne`, `has_tags`), or a
  verb phrase (`owns_marks`, `wraps`). A bare adjective or noun (`mersenne`,
  `strict`, `pending`, `front`) is not a boolean name. This holds for
  every boolean — public members, parameters, private state, locals — and
  for `bool` template parameters in their own casing (`IsStrict`,
  `IsConst`).
- Enforce constraints with concepts/traits at EVERY template boundary with a
  contract — public, detail-namespace, and private member templates alike;
  only pure type-computation aliases are exempt. Callbacks get
  invocable-based named concepts; prefer requires-clauses over in-body
  static_assert for signature checks.
- Undirected graph interfaces are edge-identity-free via the
  skip-one-parent-arc boolean; symmetric-digraph preconditions documented,
  not asserted.
- Persistent containers (a shared node pool behind every version) give each
  version-returning member a pair: `... const &` copies the pool handle into
  the result, `... &&` moves it, so the linear idiom `v = std::move(v).push_front(x)`
  skips the atomic refcount increment and decrement per operation. Both share
  one private helper that takes the pool by value. The moved-from object may
  only be destroyed or assigned to. Getters and queries are not overloaded.
- A constructor callable with one argument (including one whose other
  parameters all have defaults) is `explicit`. The exception is a value
  conversion the type models on purpose, such as an integer converting to
  `mod_int` or `software_int128` as it would to a built-in integer, or an
  edge length converting to a top-tree cluster value; it carries
  `// NOLINT(google-explicit-constructor): <reason>`. clang-tidy enforces this
  on every public header.
- Argument tags are `struct X_t { explicit X_t() = default; };` plus a
  documented `inline constexpr X_t X{};` (`inline constexpr sorted_unique_t
  sorted_unique{};`). Template-only selectors (never passed as a value, e.g.
  `unit_weight`) are a bare documented struct with no value.
- Counts: a bare `count()` means only the number of classes of a
  partition-like structure (`vertex_partition::count()`); every other count
  is `<noun>_count()` (`state_count()`, `edge_count()`).
- `reserve` takes the unit the user thinks in, the parameter is named by it
  (`elements`, `operations`, `nodes`) and the docstring states the unit;
  persistent containers count operations (`persistent_heap::reserve(operations)`).
- Id handles are aliases normally (`using heap_id = uint32;`). A strong id
  type is only for an id that could silently mix with a same-typed value
  (`convex_piecewise_linear_forest`'s `function_id`).
- `[[nodiscard]]` on persistent version-returning members and on every
  function whose only effect is its return value, queries included. Not on
  trivial accessors (`size`, `empty`) and not on mutators that return extras
  (`unite`).
- Undoable structures expose `time()`, `undo()` and `rollback(t)`, where `t`
  is what `time()` returned; `time_divide_and_conquer` auto-detects these
  names.
- Don't justify designs by another library's
  convention — reason from these conventions and first principles; external
  libs are correctness data points only.

## Performance

- Efficiency-first: the fastest correct implementation, constant factors
  included. A perf claim is a hypothesis until benchmarked on x86-64
  (minimum over repeated runs; aarch64 and x86 can disagree, so measure
  on the target architecture).
- `std::ranges::` algorithm spellings always, including partial ranges
  (`std::ranges::sort(first, last)`); tidy only catches whole-container
  calls, so sweep partial-range `std::` calls by hand. Ranges *views* in
  hot paths only when benchmarked zero-cost against a hand-written loop.
- Avoid avoidable copies: sink params by value + `std::move`, move locals
  into containers, perfect-forward pass-through layers, provide
  emplace-style members wherever the matching std container does,
  `auto const &` in range-for over non-trivial elements.
- `constexpr` wherever constant evaluation is possible (exceptions: I/O,
  runtime-only state); every constexpr API gets a static_assert proof in
  the owning folder's `tests/<folder>/constexpr_checks.cpp`.
- Every test translation unit has at least one `TEST_CASE`, including files
  of static_assert proofs: CTest reports a unit without one as "No tests
  ran" and fails it.
- Reuse existing library facilities before writing new code (helpers,
  concepts, `ds/` structures, modular helpers); hand-rolled replacements
  only with a measured, stated reason.

## Benchmarks

- The top-level-only `ATL_BUILD_IO_HARNESS` CMake option defaults to OFF and
  builds `experimental/io_bench` independently of `BUILD_BENCHMARKS`. The
  `benchmarks` preset enables it and selects the `io-harness` build target.
  Harness executables are compiled and linked in CI, never run, registered
  with CTest, or installed. Compile-and-link probes omit the optional println
  and unlocked-stdio backends when the toolchain lacks them.
- Benchmarks measure release behavior: they are built without `ATL_ASSERT`
  by default; tests keep it on. Configure with `-DATL_BENCHMARK_ASSERTS=ON`
  to measure assertion overhead.
- Run `ATL_BENCH_RESULTS=<dir> benchmarks/record.sh <area>-<name>` (e.g.
  `ds-segment_tree`) to collect repeated measurements with the commit,
  machine, compiler, build type, assertion state, minimum and spread per row.
  Records go to `<dir>`, which must exist outside the repository: they are
  local artifacts, while the benchmark sources are part of the library.
- Quoted numbers must come from reproducible benchmark implementations. Cite the source,
  measurement date, CPU and flags; distinguish measurements from hypotheses.
- A benchmark file explains itself. It starts with an `@file` comment stating
  what is raced, the one thing each contender does differently, what the loop
  measures (per-step latency or throughput, and the shapes), and the result
  with its CPU, compiler, flags and date. The full table goes on the guide page.
  Name each contender, in benchmarks and in `experimental/`, as
  `<function>_by_<method>`: what it computes (the library function it races)
  and the one thing it does differently, e.g. `next_bits_combination_by_run_ctz`,
  `xor_basis_by_echelon_scan`. Never name one for its history (`before`,
  `shipped`, `legacy`), its timing in an algorithm (`after_add`) or an internal
  variable. Each gets a doc comment saying what it does differently; benchmark
  labels use the same words.
- Pass contenders to a shared benchmark loop as callables (a lambda per
  contender, constrained with `std::invocable` / `std::is_invocable_r_v`), not
  as function-pointer template parameters: each lambda has its own type, so the
  call is direct and inlinable, and the loop accepts any callable.
- Compare every contender on every shape in the same run so the conclusion
  can be reproduced.
- Measure on x86-64 before choosing an x86 optimization. Results from aarch64
  support local iteration but cannot establish an x86 speedup.
- Compare plain `-O2` and `-march=native`. An optimization that wins only with
  native instructions must cost nothing with plain `-O2`. State any required
  extension (e.g. AVX-512), along with the flags and machine used.
- Measurements before 2026-10-13 used assertions; replace those figures as
  components are next benchmarked.

## Safety and portability

- 32-bit stored indices are deliberate; every `uint32(size)` narrowing gets
  an `atl_assert` at the entry point (central chokepoints preferred). Never
  rely on unsigned wraparound; shift uint64 when the count can reach 32.
  Fast-path dispatch conditions must themselves imply the narrowing is safe.
  Beware discarded-but-UB arithmetic (e.g. a final unused squaring that
  overflows, as power.hpp guards against).
- Portable C++ first; architecture-specific code targets x86-64.
  ARM-specific code must not cost performance on x86 and needs measured
  justification. The NEON `mulhi8` in `nt/detail/ntt_simd.hpp` supports
  fast ARM runs.
- C++23, no transitive-include reliance, standard headers/functions. Headers
  must survive single-translation-unit amalgamation.
- `atl::forward_like`, never `std::forward_like`: older standard libraries
  fail at parse time. Toolchain compatibility gaps get a shim rather than
  a NOLINT or a reversion.
- Constexpr code that MSVC must evaluate never passes a caller's local by
  reference down a recursion: MSVC's constant evaluation reports it as
  uninitialized (valid on GCC and Clang). Take such accumulators by value.
  The same holds for a class-type result (a `std::pair`) returned through a
  recursion: recurse on a scalar (two 32-bit ids packed into one `uint64`)
  and build the class once at the top.
- Never rely on a braced list's left-to-right order when one element copies
  an object and a later one moves from it (`{T(pool_, a), T(std::move(pool_),
  b)}`): MSVC may move first. Copy in a separate statement.
- Loops over a 64-bit counter shift, then test `!= 0` (`e >>= 1; if (e != 0)
  x *= x;`). Testing `e > 1` after the body was miscompiled by MSVC x86 once
  heavily inlined software `uint128` code surrounded it.
- Never use a name that `<windows.h>` defines as a macro, tests included:
  `pascal`, `cdecl`, `far`, `near` (and `small` without
  `WIN32_LEAN_AND_MEAN`). On MSVC, `tests/expect_abort.hpp`,
  `tests/io/interactive.cpp` and Catch2 include it, and users may include it
  before any library header.
- Test-file globals never share a name with a local or parameter in an
  included header: MSVC warns that the local hides the global (C4459), which
  `/W4 /WX` turns into an error. Keep fixtures function-local or give them
  distinctive names.
- Other MSVC `/W4` warnings that GCC and Clang do not give, all fatal under
  `/WX`: a parameter read in only one `if constexpr` branch is unused in the
  other (C4100; mark it `[[maybe_unused]]`); `1u << k` widened to
  `std::size_t` (C4334; shift in the target type); a 64-bit index or count
  stored into `std::size_t` on x86 (C4244); a local, parameter or lambda
  parameter that shadows an enclosing one (C4456, C4457).
- Arithmetic on a value type narrower than `int` (`int16_t` flows or costs)
  promotes to `int`: convert the result back (`F(a - b)`, `F(-delta)`) before
  storing it or passing it to `(std::min)`/`(std::max)`, which otherwise fail
  to deduce. Code whose constraints admit such types gets an accepted-call
  test that instantiates it with them.
- MSVC rejects the constant evaluation of a defaulted private constructor
  reached through a static factory (C2248), and of a named local built from
  one: give the constructor a user-provided body and return a prvalue
  (`return csr(std::move(offsets), std::move(edges));`).
- Constexpr constructors that MSVC must evaluate take vectors and pairs by
  `const &` when they are the target of a delegating constructor, and never
  modify a by-value vector parameter to build from it: MSVC's constant
  evaluation misreads such parameters ("unevaluable pointer value", a read of
  an uninitialized symbol, or silently wrong contents). Build a local copy.
  Likewise never `push_back` an element read from the same vector
  (`v.push_back(v[i])`): size the vector first and assign by index.
- Where an invariant guarantees that a `std::optional` is engaged, test it
  anyway (`if (steps < depth && forest)`) with a comment: clang-tidy's
  `bugprone-unchecked-optional-access` cannot see the invariant.
- Benchmarks that hook allocation cannot use `std::aligned_alloc` (MSVC lacks
  it): over-allocate with `malloc`, align with `std::align` and keep the
  original pointer for the release.
- `std::numeric_limits<T>::min()` is the smallest positive value for a
  floating `T`; a "minus infinity" starting value is `lowest()`.
- Count edges or other range elements in a checked `std::size_t`
  accumulator, not with `std::ranges::distance`, whose signed difference type
  can overflow on 32-bit targets before the bound check runs.
- clang-tidy's `modernize-use-ranges` flags `std::iota`, and libc++ has no
  `std::ranges::iota`: fill index arrays with a plain loop.
- Death tests include `"../expect_abort.hpp"` (quoted, relative). Never
  negate `aborts()`: in the Windows re-exec child every earlier `aborts()`
  call reports true, so a negated check fails there and breaks the death
  tests after it. Check the non-aborting case with a plain `REQUIRE` on its
  result.
- No code copied from attribution-licensed sources (MIT/BSD/Apache/CC-BY
  included): cite the reference in a comment, implement independently.

## Documentation

Guide prose lives in `docs/*.md`; Doxygen discovers those pages and all public
headers recursively, using the root README as its main page. API contracts
belong next to declarations in `include/atl`; implementation namespaces stay
out of the public reference. The generated Guide coverage table provides the
header-to-guide map and a guide index without a maintained page list.

Put complete runnable examples in `docs/examples/<page>/*.cpp`, with exact
stdout in the matching `.out` file (including its final newline). Include only
public library headers. Use Doxygen `\include page/example.cpp` or `\snippet`
in the guide so the displayed code is the tested file. The normal test build
globs these sources. Each `docs/example/...` CTest test builds its executable,
runs it, and compares stdout with `.out` exactly except for line endings (Windows
writes CRLF); a nonzero exit or
a mismatch fails. Run `ctest --preset gcc -R '^docs/'` after configuring GCC.

Every public header outside a `detail/` directory must be referenced by at
least one guide in `docs/*.md`, usually with an explicit public-header
include. Spell out each header; directory references do not provide coverage. All local
header references and links to headers or Markdown pages must exist, including
links in the root README. The `docs/coverage` test and docs target fail with
the complete gap list. Add the guide and examples in the same PR as a new
component. Generated files go in the build tree, never in `docs/`.

- Doxygen `/** @brief ... */` contract on every public API entity,
  including trivial accessors, with `@tparam`/`@param`/`@pre`/
  `@return`/`@complexity` where meaningful. A `namespace detail` entity
  gets a brief only when it states a contract, an invariant or non-obvious
  reasoning; a brief that restates the name ("helper that adds two words")
  is deleted, not written.
- Put template arguments in Doxygen text in backticks, as in
  `congruence<U>`: written bare, Doxygen reads `<U>` as an HTML tag.
- API docs use `/** ... */`; trailing member docs may use `///<`. No `///`
  line docs.
- Every public namespace has a Doxygen block (the list lives in
  `cmake/Documentation.cmake`); without one, Doxygen drops the namespace's
  free functions from the reference and `FAIL_ON_WARNINGS` fails on every
  link to them. A new public namespace adds its brief there.
- A doc comment goes before the whole declaration, `template<...>` and
  `template<>` lines included. Deduction guides need their own brief.
  `@ref` targets are bare names (no trailing `:`).
- Guide pages put no backslash before a letter in prose: Doxygen reads it as
  a command and `FAIL_ON_WARNINGS` fails the build.
  Write such notation in words ("L minus Z").
- Every runtime-only (non-constexpr) public API states why in its docs:
  `@note Not constexpr (uses std::sort on a heap buffer)`.
- `//` comments only for in-body non-obvious notes: the why, a subtlety, an
  invariant, an overflow argument — never narration of what the code plainly
  does. Keep the conclusions of history (why a variant lost, with its
  numbers), benchmark-pending status, licence/attribution notes and tooling
  directives (`NOLINT`); drop chronology (what was tried first, dates of
  changes), commented-out code and banner/separator comments.
- Every shipped component: a `docs/<topic>.md` section with semantics and
  measured numbers and an entry in the generated coverage map. Record technical
  decisions, rejected alternatives, bugs caught and benchmark conclusions in its docs.

## Docs pages

Every algorithm and data structure page in `docs/` follows this template, in
this order. A page that covers several components (as `dp-optimization.md`
does) repeats the sections 3-7 per component, under that component's `##`
heading, with the shared interface described once at the top.

1. **Title and summary.** the heading `atl::name` in code font, then one paragraph: what it is
   and when to reach for it (and when not to).
2. **Include line and quick example.** The `#include`, then a short example
   that compiles as written.
3. **API.** Every public entity with its signature exactly as in the header,
   its parameters, its return value and a one-line description; the template
   parameters and the concepts constraining them. Tables for operations,
   code blocks for result structs.
4. **Complexity.** Time and memory per operation (the API table's last column
   may carry it); amortized or expected bounds are labelled as such.
5. **Preconditions.** Split into asserted (`atl_assert`, active only under
   `-DATL_ASSERT`) and documented-only (the algebraic or structural laws no
   check can verify); value ranges and overflow bounds stated.
6. **Measured performance.** Numbers only from reproducible benchmarks,
   each table naming the benchmark source, measurement date, CPU and flags
   (plain `-O2` / `-march=native`). A page without measurements says so.
7. **Pitfalls and notes.** Common misuse, and how to choose between related
   components (links to their pages go in See also).
8. **`## How it works`**, always the last section before See also: the idea;
   the invariants; why the complexity holds (a short argument, not a proof
   dump); the implementation tricks THIS code uses (read the header and
   describe what it does, not the textbook variant); references by name
   (paper or book chapter), never copied code.

Sections may be merged when short (3-5 often read as one), but the order
holds. Keep the page proportionate: a page is not longer than it needs to be.

## Geometry exactness doctrine

- Predicates EXACT (widened integer types / int256 for ints; filtered
  adaptive expansion arithmetic for doubles); constructions rounded
  `float64`. Document which tier each API sits in.
- Coordinate bounds asserted and documented: general geometry `|x|,|y| <
  2^62`; sweep/arrangement/point-location `< 2^32`; tighter bounds stated
  per header.

## Experimental code (`experimental/atl/experimental/`)

Experimental code follows ALL conventions above — it is held to the same
bar as promoted code, not a scratchpad. Additionally:

- Namespace `atl::experimental`; include paths need
  `-I experimental -I experimental/atl`.
- A race loser's docstring records WHY it lost, with the benchmark numbers
  and shapes from the race verdict.
- Each experimental family designates one readable/certifying oracle
  implementation in its tests.
- Interfaces mirror the promoted counterpart's shape (same result-struct
  style, naming, concepts), so promotion or re-racing is a move, not a
  rewrite.
- A surveyed contender is never dropped silently: it is raced, or its
  exclusion is documented with its technical rationale.
