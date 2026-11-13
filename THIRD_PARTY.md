# External dependencies

This repository contains the project field routines, adapters, and validation
sources. It does not contain the COSY INFINITY runtime, licensed COSY sources,
COSY.bin macro library, compiler runtimes, or dependency source trees.

| Dependency | Use | Upstream terms |
|---|---|---|
| GNU Fortran/GCC runtimes | Native library and executables | [GCC licensing](https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html) and the applicable runtime exceptions |
| GMP | Exact rational boundary predicates in the native engine | [GMP licensing](https://gmplib.org/manual/Copying) |
| Boost.Multiprecision | Standalone C++ arithmetic and exact predicates | [Boost Software License](https://www.boost.org/LICENSE_1_0.txt) |
| Python, NumPy, mpmath | Preparation, Python reference, independent validation | Each project's installed license |
| Julia | Standalone and native Julia consumers | [Julia license](https://julialang.org/license/) |
| JAX and jaxlib | Optional named adapter | [JAX license](https://github.com/jax-ml/jax/blob/main/LICENSE) |
| COSY INFINITY | Optional native bridge execution and DA composition | Obtain a suitable licensed runtime/source distribution separately |

Dependencies are installed separately and retain their own terms.
Project source is covered by the MIT license in LICENSE.

`adapters/jax_repair_20261103` is a project-specific implementation. It is not
an upstream magpylib_jax export. Its documented comparison baseline is commit
`6290916635693fb04a099205b48f845a396dffc0`; that package is not vendored here.
