# Guide to atl {#atl_guides}

`atl` is a header-only C++23 library. Include a public component with
`#include <atl/…>`; the generated API reference describes its contracts and
constraints. The handwritten guides explain semantics, tradeoffs, examples
and measurements.

The site's **Related Pages** tree discovers all guide pages in this checkout.
The generated **Guide coverage** page lists every public header and the guides
that discuss it, with a complete guide index. Both grow automatically when
components and their pages arrive.

See [Library conventions](conventions.md) for contribution rules, including
where to put guide prose, executable examples and API comments.

- [128-bit integers](int128.md): native aliases, portable software integers and formatting.
- [Bit utilities](bit-utilities.md): lowest set bit and fixed-popcount enumeration.
- [XOR bases](xor-basis.md): reduced spans and stamp-filtered queries.
- [Fast I/O](fast-io.md): buffered readers and writers.
