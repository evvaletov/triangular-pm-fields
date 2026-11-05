! Compile with the build directory's module files and libtan, or use CMake.
! Prepare triangle.tan using examples/triangle.py first.
program example_triangle
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use tan_mod
  implicit none
  real(dp) :: coefficients(3,84)
  integer :: status
  call tan_load('triangle.tan', status)
  if (status /= 0) stop 1
  call tan_eval([.25_dp,.25_dp,.75_dp], 1, 6, .125_dp, coefficients, status)
  if (status /= 0) stop 2
  print *, 'mu0 H [T]:', coefficients(:,1)
end program
