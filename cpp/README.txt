C++ triangle surface field
==========================

Standalone evaluation of mu0 H = sigma/(4 pi) integral_T (p-s)/|p-s|^3 dS
for one finite triangle, plus a native client of the Fortran tan_mod C ABI.

The standalone library implements the kernel and local Taylor algebra itself.
It does not call Python, Fortran, or another evaluator. The native client is a
separately named interface; it loads the Fortran physical kernel and shares
that kernel's process-global cache. That cache is not thread-safe.
Each native_eval call loads, evaluates, clears its cache, and closes its library
handle. Serialize it with other callers of the same native library.

Requirements
------------
C++17 compiler (developed with g++ 14.3.1), Boost.Multiprecision headers
(cpp_dec_float_100 and cpp_rational). Header-only; no Boost libraries.
dlopen for native_cli. Optional native tests also need gfortran, gcc, and GMP.

Build
-----
  make
  make test
  make test-native   # builds a local libtan.so into build/

  cmake -S . -B build && cmake --build build

Do not enable fast-math. Generated binaries stay under build/ (gitignored).

Standalone CLI
--------------
  ./build/triangle_cli ORDER RADIUS SIGMA AX AY AZ BX BY BZ CX CY CZ PX PY PZ

Input tokens parse to binary64 first. Exact plane/edge/vertex membership uses
cpp_rational of those binary64 values (IEEE bit conversion, round-trip checked).
Query and Taylor arithmetic use cpp_dec_float_100, then binary64 output.

  ./build/triangle_cli --batch < lines.txt

Each stdin line is one argument record; one JSON object is written per line.
A per-record error continues; the process exit status is nonzero if any failed.

JSON: status, exponents [[ex,ey,ez],...], coefficients shape (3, nmono).
Success is status 0 with finite numbers. Errors return empty arrays and a
nonzero status/exit. Status 5 invalid order, 6 nonfinite observer, 7 bad
radius, 8 charged edge/vertex, 9 DA centre on the charged sheet, 10 nonfinite
result, 11 invalid geometry/charge/malformed input.

Geometry and charge are validated, including nonzero area, before a zero-charge
short-circuit. Zero sigma then returns zero even on the support. Invalid
order, radius, or observer still reject.

Native CLI
----------
  ./build/native_cli LIBRARY CACHE X Y Z MODE ORDER RADIUS

Calls tan_load_c / tan_eval_c. Coefficient layout is Fortran coeff(3,84),
component fastest. Output is consumed only after status 0. Native statuses
and modes follow ../docs/native-api.md.

Library API
-----------
  triangle_pm::evaluate(...)
  triangle_pm::native_eval(...)

Same content as the CLIs, without a subprocess.

Maps
----
Order q = 0..6, never truncated. Monomials are graded, then x ascending, then
y ascending, z = degree-x-y (84 terms at q=6). Coefficients include alpha!
normalization of mu0 H(p + radius u). Radius rescales the variables; it is not
a convergence-ball certificate.

Limits
------
Fixed-precision claims are corpus-bounded. cpp_dec_float_100 is about 100
decimal digits for the local series, not a universal error bound. Degree-6
truncation must be checked before using a map away from its centre. Timings
on this host do not imply a speedup elsewhere. No GPU path.
