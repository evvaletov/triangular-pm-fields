! tan_jet.f90 — truncated 3-variable jets through degree 6, real128.
! Monomials are graded, then x ascending, then y ascending, z = degree-x-y.
! Multiplication uses the 924 ordered convolution pairs at q=6.
module tan_jet_mod
   use, intrinsic :: iso_fortran_env, only: wp => real128, dp => real64
   use, intrinsic :: ieee_arithmetic
   implicit none
   private

   integer, parameter, public :: TAN_QMAX = 6
   integer, parameter, public :: TAN_NM = 84
   integer, parameter, public :: TAN_NPAIR = 924

   integer, parameter, public :: TAN_OK = 0
   integer, parameter, public :: TAN_ERR_NO_CACHE = 1
   integer, parameter, public :: TAN_ERR_IO = 2
   integer, parameter, public :: TAN_ERR_FORMAT = 3
   integer, parameter, public :: TAN_ERR_MODE = 4
   integer, parameter, public :: TAN_ERR_ORDER = 5
   integer, parameter, public :: TAN_ERR_POINT = 6
   integer, parameter, public :: TAN_ERR_RADIUS = 7
   integer, parameter, public :: TAN_ERR_EDGE = 8
   integer, parameter, public :: TAN_ERR_WINDING = 9
   integer, parameter, public :: TAN_ERR_NONFINITE = 10
   integer, parameter, public :: TAN_ERR_DOMAIN = 11

   type, public :: tan_jet
      integer :: q = 0
      real(wp) :: c(TAN_NM) = 0.0_wp
   end type tan_jet

   type, public :: tan_jet3
      type(tan_jet) :: x, y, z
   end type tan_jet3

   integer, public, protected :: nmono_of(0:TAN_QMAX)
   integer, public, protected :: expo_x(TAN_NM), expo_y(TAN_NM), expo_z(TAN_NM), expo_deg(TAN_NM)
   integer, public, protected :: idx_of(0:TAN_QMAX, 0:TAN_QMAX, 0:TAN_QMAX)
   integer, public, protected :: pair_i(TAN_NPAIR), pair_j(TAN_NPAIR), pair_k(TAN_NPAIR)
   integer, public, protected :: npair_of(0:TAN_QMAX)
   logical, save :: tables_ready = .false.

   public :: tan_jet_init
   public :: tan_nmono
   public :: jet_zero, jet_const, jet_coord
   public :: jet_add, jet_sub, jet_neg, jet_scale, jet_axpy
   public :: jet_mul, jet_prod, jet_inv, jet_sqrt, jet_log, jet_pow
   public :: jet_atan2
   public :: jet3_const, jet3_point, jet3_sub, jet3_scale
   public :: jet3_dot, jet3_cross, jet3_norm, jet3_norm2
   public :: jet_is_finite, jet3_is_finite
   public :: tan_finite
   public :: four_pi, tan_pi

   real(wp), parameter :: tan_pi = &
      3.14159265358979323846264338327950288419716939937510_wp
   real(wp), parameter :: four_pi = 4.0_wp * tan_pi

contains

   subroutine tan_jet_init()
      integer :: q, a, b, z, k, i, j, dsum, n
      if (tables_ready) return
      idx_of = 0
      k = 0
      do q = 0, TAN_QMAX
         do a = 0, q
            do b = 0, q - a
               z = q - a - b
               k = k + 1
               expo_x(k) = a
               expo_y(k) = b
               expo_z(k) = z
               expo_deg(k) = q
               idx_of(a, b, z) = k
            end do
         end do
         nmono_of(q) = k
      end do
      if (k /= TAN_NM) error stop 'tan_jet: monomial count is not 84'
      n = 0
      do dsum = 0, TAN_QMAX
         do i = 1, TAN_NM
            if (expo_deg(i) > dsum) cycle
            do j = 1, TAN_NM
               if (expo_deg(j) /= dsum - expo_deg(i)) cycle
               n = n + 1
               pair_i(n) = i
               pair_j(n) = j
               pair_k(n) = idx_of(expo_x(i) + expo_x(j), &
                                  expo_y(i) + expo_y(j), &
                                  expo_z(i) + expo_z(j))
            end do
         end do
         npair_of(dsum) = n
      end do
      if (n /= TAN_NPAIR) error stop 'tan_jet: pair count is not 924'
      tables_ready = .true.
   end subroutine tan_jet_init

   integer function tan_nmono(order)
      integer, intent(in) :: order
      if (order < 0 .or. order > TAN_QMAX) then
         tan_nmono = 0
      else
         tan_nmono = (order+1)*(order+2)*(order+3)/6
      end if
   end function tan_nmono

   elemental logical function tan_finite(x)
      real(wp), intent(in) :: x
      tan_finite = ieee_is_finite(x)
   end function tan_finite

   type(tan_jet) function jet_zero(order) result(j)
      integer, intent(in) :: order
      j%q = order
      j%c = 0.0_wp
   end function jet_zero

   type(tan_jet) function jet_const(val, order) result(j)
      real(wp), intent(in) :: val
      integer, intent(in) :: order
      j%q = order
      j%c = 0.0_wp
      j%c(1) = val
   end function jet_const

   ! Physical coordinate = center + radius * u_axis, already radius-scaled.
   type(tan_jet) function jet_coord(center, axis, radius, order) result(j)
      real(wp), intent(in) :: center, radius
      integer, intent(in) :: axis, order
      integer :: ix, iy, iz, k
      j = jet_const(center, order)
      if (order <= 0) return
      ix = 0; iy = 0; iz = 0
      if (axis == 1) ix = 1
      if (axis == 2) iy = 1
      if (axis == 3) iz = 1
      k = idx_of(ix, iy, iz)
      j%c(k) = radius
   end function jet_coord

   type(tan_jet) function jet_add(a, b) result(c)
      type(tan_jet), intent(in) :: a, b
      integer :: n
      c%q = a%q
      n = nmono_of(a%q)
      c%c = 0.0_wp
      c%c(1:n) = a%c(1:n) + b%c(1:n)
   end function jet_add

   type(tan_jet) function jet_sub(a, b) result(c)
      type(tan_jet), intent(in) :: a, b
      integer :: n
      c%q = a%q
      n = nmono_of(a%q)
      c%c = 0.0_wp
      c%c(1:n) = a%c(1:n) - b%c(1:n)
   end function jet_sub

   type(tan_jet) function jet_neg(a) result(c)
      type(tan_jet), intent(in) :: a
      integer :: n
      c%q = a%q
      n = nmono_of(a%q)
      c%c = 0.0_wp
      c%c(1:n) = -a%c(1:n)
   end function jet_neg

   type(tan_jet) function jet_scale(a, s) result(c)
      type(tan_jet), intent(in) :: a
      real(wp), intent(in) :: s
      integer :: n
      c%q = a%q
      n = nmono_of(a%q)
      c%c = 0.0_wp
      c%c(1:n) = s * a%c(1:n)
   end function jet_scale

   subroutine jet_axpy(s, a, b)
      real(wp), intent(in) :: s
      type(tan_jet), intent(in) :: a
      type(tan_jet), intent(inout) :: b
      integer :: n
      n = nmono_of(a%q)
      b%c(1:n) = b%c(1:n) + s * a%c(1:n)
   end subroutine jet_axpy

   type(tan_jet) function jet_mul(a, b) result(c)
      type(tan_jet), intent(in) :: a, b
      integer :: p, n, q, ia, ib, ic
      real(wp) :: ai, bj
      q = a%q
      n = nmono_of(q)
      c%q = q
      c%c = 0.0_wp
      ! Skip exact-zero factors; keep the original pair order for nonzero terms.
      do p = 1, npair_of(q)
         ia = pair_i(p)
         ib = pair_j(p)
         ai = a%c(ia)
         bj = b%c(ib)
         if (ai == 0.0_wp .or. bj == 0.0_wp) cycle
         ic = pair_k(p)
         c%c(ic) = c%c(ic) + ai * bj
      end do
      if (n < TAN_NM) c%c(n+1:TAN_NM) = 0.0_wp
   end function jet_mul

   type(tan_jet) function jet_prod(a, b, c) result(d)
      type(tan_jet), intent(in) :: a, b, c
      d = jet_mul(jet_mul(a, b), c)
   end function jet_prod

   type(tan_jet) function jet_pow(a, exponent, status) result(c)
      type(tan_jet), intent(in) :: a
      real(wp), intent(in) :: exponent
      integer, intent(out) :: status
      type(tan_jet) :: u, term, total
      real(wp) :: a0, coeff
      integer :: k, q
      status = TAN_OK
      q = a%q
      a0 = a%c(1)
      if (a0 == 0.0_wp) then
         status = TAN_ERR_DOMAIN
         c = jet_zero(q)
         return
      end if
      if (exponent == 0.0_wp) then
         c = jet_const(1.0_wp, q)
         return
      end if
      if (exponent == 1.0_wp) then
         c = a
         return
      end if
      u = jet_scale(jet_sub(a, jet_const(a0, q)), 1.0_wp / a0)
      total = jet_const(1.0_wp, q)
      term = jet_const(1.0_wp, q)
      coeff = 1.0_wp
      do k = 1, q
         term = jet_mul(term, u)
         coeff = coeff * (exponent - real(k, wp) + 1.0_wp) / real(k, wp)
         call jet_axpy(coeff, term, total)
      end do
      c = jet_scale(total, a0 ** exponent)
   end function jet_pow

   type(tan_jet) function jet_inv(a, status) result(c)
      type(tan_jet), intent(in) :: a
      integer, intent(out) :: status
      c = jet_pow(a, -1.0_wp, status)
   end function jet_inv

   type(tan_jet) function jet_sqrt(a, status) result(c)
      type(tan_jet), intent(in) :: a
      integer, intent(out) :: status
      if (a%c(1) < 0.0_wp) then
         status = TAN_ERR_DOMAIN
         c = jet_zero(a%q)
         return
      end if
      c = jet_pow(a, 0.5_wp, status)
   end function jet_sqrt

   type(tan_jet) function jet_log(a, status) result(c)
      type(tan_jet), intent(in) :: a
      integer, intent(out) :: status
      type(tan_jet) :: u, term, total
      real(wp) :: a0, s
      integer :: k, q
      status = TAN_OK
      q = a%q
      a0 = a%c(1)
      if (a0 <= 0.0_wp) then
         status = TAN_ERR_DOMAIN
         c = jet_zero(q)
         return
      end if
      u = jet_scale(jet_sub(a, jet_const(a0, q)), 1.0_wp / a0)
      total = jet_const(log(a0), q)
      term = jet_const(1.0_wp, q)
      s = 1.0_wp
      do k = 1, q
         term = jet_mul(term, u)
         s = -s
         call jet_axpy(-s / real(k, wp), term, total)
      end do
      c = total
   end function jet_log

   ! atan2(y,x) = arg(x+iy). Constant uses atan2; higher terms use Im log(1+u).
   subroutine jet_atan2(y, x, out, status)
      type(tan_jet), intent(in) :: y, x
      type(tan_jet), intent(out) :: out
      integer, intent(out) :: status
      type(tan_jet) :: dx, dy, ure, uim, tre, tim, nre, nim, lim
      real(wp) :: x0, y0, r2, invx, invy, coef
      integer :: k, q
      status = TAN_OK
      q = x%q
      x0 = x%c(1)
      y0 = y%c(1)
      r2 = x0 * x0 + y0 * y0
      if (r2 == 0.0_wp) then
         status = TAN_ERR_DOMAIN
         out = jet_zero(q)
         return
      end if
      out = jet_const(atan2(y0, x0), q)
      if (q == 0) return
      dx = x
      dy = y
      dx%c(1) = 0.0_wp
      dy%c(1) = 0.0_wp
      invx = x0 / r2
      invy = -y0 / r2
      ure = jet_sub(jet_scale(dx, invx), jet_scale(dy, invy))
      uim = jet_add(jet_scale(dx, invy), jet_scale(dy, invx))
      tre = jet_const(1.0_wp, q)
      tim = jet_zero(q)
      lim = jet_zero(q)
      do k = 1, q
         nre = jet_sub(jet_mul(tre, ure), jet_mul(tim, uim))
         nim = jet_add(jet_mul(tre, uim), jet_mul(tim, ure))
         tre = nre
         tim = nim
         coef = 1.0_wp / real(k, wp)
         if (mod(k, 2) == 0) coef = -coef
         call jet_axpy(coef, tim, lim)
      end do
      out = jet_add(out, lim)
   end subroutine jet_atan2

   type(tan_jet3) function jet3_const(v, order) result(p)
      real(wp), intent(in) :: v(3)
      integer, intent(in) :: order
      p%x = jet_const(v(1), order)
      p%y = jet_const(v(2), order)
      p%z = jet_const(v(3), order)
   end function jet3_const

   type(tan_jet3) function jet3_point(center, radius, order) result(p)
      real(wp), intent(in) :: center(3), radius
      integer, intent(in) :: order
      p%x = jet_coord(center(1), 1, radius, order)
      p%y = jet_coord(center(2), 2, radius, order)
      p%z = jet_coord(center(3), 3, radius, order)
   end function jet3_point

   type(tan_jet3) function jet3_sub(a, b) result(c)
      type(tan_jet3), intent(in) :: a, b
      c%x = jet_sub(a%x, b%x)
      c%y = jet_sub(a%y, b%y)
      c%z = jet_sub(a%z, b%z)
   end function jet3_sub

   type(tan_jet3) function jet3_scale(a, s) result(c)
      type(tan_jet3), intent(in) :: a
      real(wp), intent(in) :: s
      c%x = jet_scale(a%x, s)
      c%y = jet_scale(a%y, s)
      c%z = jet_scale(a%z, s)
   end function jet3_scale

   type(tan_jet) function jet3_dot(a, b) result(s)
      type(tan_jet3), intent(in) :: a, b
      s = jet_add(jet_add(jet_mul(a%x, b%x), jet_mul(a%y, b%y)), jet_mul(a%z, b%z))
   end function jet3_dot

   type(tan_jet3) function jet3_cross(a, b) result(c)
      type(tan_jet3), intent(in) :: a, b
      c%x = jet_sub(jet_mul(a%y, b%z), jet_mul(a%z, b%y))
      c%y = jet_sub(jet_mul(a%z, b%x), jet_mul(a%x, b%z))
      c%z = jet_sub(jet_mul(a%x, b%y), jet_mul(a%y, b%x))
   end function jet3_cross

   type(tan_jet) function jet3_norm2(a) result(s)
      type(tan_jet3), intent(in) :: a
      s = jet3_dot(a, a)
   end function jet3_norm2

   type(tan_jet) function jet3_norm(a, status) result(s)
      type(tan_jet3), intent(in) :: a
      integer, intent(out) :: status
      s = jet_sqrt(jet3_norm2(a), status)
   end function jet3_norm

   logical function jet_is_finite(a) result(ok)
      type(tan_jet), intent(in) :: a
      integer :: i, n
      n = nmono_of(a%q)
      ok = .true.
      do i = 1, n
         if (.not. tan_finite(a%c(i))) then
            ok = .false.
            return
         end if
      end do
   end function jet_is_finite

   logical function jet3_is_finite(a) result(ok)
      type(tan_jet3), intent(in) :: a
      ok = jet_is_finite(a%x) .and. jet_is_finite(a%y) .and. jet_is_finite(a%z)
   end function jet3_is_finite

end module tan_jet_mod
