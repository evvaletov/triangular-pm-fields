! test_tan.f90 — local acceptance checks for tan_mod.
program test_tan
   use, intrinsic :: iso_fortran_env, only: wp => real128, dp => real64, &
        output_unit, error_unit
   use, intrinsic :: ieee_arithmetic
   use tan_mod
   use tan_jet_mod, only: tan_jet, tan_jet_init, jet_coord, jet_mul, jet_const, &
        jet_log, jet_sqrt, jet_atan2, four_pi, nmono_of
   implicit none

   integer :: npass, nfail, status, i
   integer :: ex(TAN_NM), ey(TAN_NM), ez(TAN_NM)
   real(dp) :: point(3), coeff(3, TAN_NM), coeff2(3, TAN_NM), bref(3)
   real(dp) :: sigma, bounds(4)
   character(len=256) :: cache, cache2
   type(tan_jet) :: x, y, z, outj

   npass = 0
   nfail = 0
   call expect(tan_nmono(6) == 84, 'monomial count before any initialization')
   call expect(tan_nmono(0) == 1, 'scalar monomial count before initialization')
   call expect(tan_nmono(-1) == 0 .and. tan_nmono(7) == 0, 'invalid monomial count')
   call tan_jet_init()
   cache = 'results/triangle.cache'
   cache2 = 'results/rectangle.cache'

   call check_exponents()
   call check_jet_ops()
   call check_invalid_inputs()
   call check_triangle()
   call check_rectangle()
   call check_exterior_continuation()
   call check_rejections()
   call check_canceled()
   call check_reload()
   call check_batch_and_c()
   call check_modes_and_auto()
   call check_oblique_sheet()
   call check_holes_and_tiny()
   call check_stale_load()

   write(output_unit, '(a,1x,i0,1x,a,1x,i0)') 'PASS', npass, 'FAIL', nfail
   if (nfail /= 0) stop 1
   stop 0

contains

   subroutine expect(cond, msg)
      logical, intent(in) :: cond
      character(len=*), intent(in) :: msg
      if (cond) then
         npass = npass + 1
         write(output_unit, '(a,1x,a)') 'ok', trim(msg)
      else
         nfail = nfail + 1
         write(error_unit, '(a,1x,a)') 'FAIL', trim(msg)
      end if
   end subroutine expect

   subroutine expect_close(a, b, atol, rtol, msg)
      real(dp), intent(in) :: a(:), b(:), atol, rtol
      character(len=*), intent(in) :: msg
      real(dp) :: den, num
      den = max(sqrt(sum(b * b)), 1.0e-30_dp)
      num = sqrt(sum((a - b) * (a - b)))
      call expect(num <= atol + rtol * den, msg)
      if (num > atol + rtol * den) then
         write(error_unit, '(a,3es20.10e3)') '  got ', a
         write(error_unit, '(a,3es20.10e3)') '  ref ', b
         write(error_unit, '(a,es12.4e3,a,es12.4e3)') '  |d|=', num, ' budget=', &
              atol + rtol * den
      end if
   end subroutine expect_close

   subroutine check_exponents()
      integer :: k, q, a, b, z, n
      call tan_exponents(6, ex, ey, ez, n)
      call expect(n == 84, 'nmono(6)=84')
      call expect(nmono_of(0) == 1, 'nmono(0)=1')
      call expect(nmono_of(1) == 4, 'nmono(1)=4')
      call expect(nmono_of(2) == 10, 'nmono(2)=10')
      k = 0
      do q = 0, 6
         do a = 0, q
            do b = 0, q - a
               z = q - a - b
               k = k + 1
               if (ex(k) /= a .or. ey(k) /= b .or. ez(k) /= z) then
                  call expect(.false., 'exponent table mismatch')
                  return
               end if
            end do
         end do
      end do
      call expect(k == 84, 'graded x-then-y exponent walk')
      call expect(ex(1)==0 .and. ey(1)==0 .and. ez(1)==0, '(0,0,0) first')
      call expect(ex(2)==0 .and. ey(2)==0 .and. ez(2)==1, 'deg1 z')
      call expect(ex(3)==0 .and. ey(3)==1 .and. ez(3)==0, 'deg1 y')
      call expect(ex(4)==1 .and. ey(4)==0 .and. ez(4)==0, 'deg1 x')
   end subroutine check_exponents

   subroutine check_jet_ops()
      integer :: st
      real(wp) :: c, r
      c = 2.0_wp
      r = 0.1_wp
      x = jet_coord(c, 1, r, 2)
      y = jet_mul(x, x)
      call expect(abs(y%c(1) - c*c) < 1.0e-30_wp, 'jet square constant')
      ! linear x coefficient of (c + r u_x)^2 is 2 c r, index 4 = (1,0,0)
      call expect(abs(y%c(4) - 2.0_wp*c*r) < 1.0e-30_wp, 'jet square linear')
      ! xx coefficient index 10 = (2,0,0) is r^2
      call expect(abs(y%c(10) - r*r) < 1.0e-30_wp, 'jet square xx')
      x = jet_coord(1.0_wp, 1, 0.2_wp, 3)
      y = jet_log(x, st)
      call expect(st == TAN_OK, 'jet log status')
      call expect(abs(y%c(1) - 0.0_wp) < 1.0e-30_wp, 'log(1) constant')
      x = jet_const(4.0_wp, 2)
      y = jet_sqrt(x, st)
      call expect(st == TAN_OK .and. abs(y%c(1) - 2.0_wp) < 1.0e-30_wp, 'sqrt(4)')
      x = jet_const(1.0_wp, 2)
      z = jet_coord(0.0_wp, 2, 0.3_wp, 2)
      call jet_atan2(z, x, outj, st)
      call expect(st == TAN_OK, 'atan2 status')
      call expect(abs(outj%c(1) - 0.0_wp) < 1.0e-30_wp, 'atan2(0,1) const')
      ! d/du_y atan2(r u_y, 1) at 0 is r, index 3 = (0,1,0)
      call expect(abs(outj%c(3) - 0.3_wp) < 1.0e-28_wp, 'atan2 linear y')
   end subroutine check_jet_ops

   subroutine check_invalid_inputs()
      point = [0.1_dp, 0.2_dp, 0.3_dp]
      call tan_eval(point, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_NO_CACHE, 'eval without cache')
      call tan_load('results/no_such_cache_file.tan', status)
      call expect(status == TAN_ERR_IO, 'missing cache file')
      open(newunit=i, file='results/bad.cache', status='replace')
      write(i, '(a)') 'NOT_A_CACHE'
      close(i)
      call tan_load('results/bad.cache', status)
      call expect(status == TAN_ERR_FORMAT, 'bad header')
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load triangle cache')
      call tan_eval(point, 99, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_MODE, 'invalid mode')
      call tan_eval(point, 1, 7, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_ORDER, 'order 7 rejected')
      call tan_eval(point, 0, 3, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_MODE, 'mode 0 with order>0 rejected')
      call tan_eval(point, 2, 3, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_MODE, 'mode 2 with order>0 rejected')
      call tan_eval(point, 2, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'mode 2 order 0 accepted')
      call tan_eval(point, 1, 0, 0.0_dp, coeff, status)
      call expect(status == TAN_ERR_RADIUS, 'zero radius')
      call tan_eval(point, 1, 0, -1.0_dp, coeff, status)
      call expect(status == TAN_ERR_RADIUS, 'negative radius')
      point(1) = ieee_value(1.0_dp, ieee_quiet_nan)
      call tan_eval(point, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_POINT, 'NaN point')
   end subroutine check_invalid_inputs

   subroutine check_triangle()
      real(dp) :: verts(3, 3), p(3), kref(3)
      sigma = 0.8_dp
      verts(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp]
      verts(:, 2) = [1.2_dp, 0.0_dp, 0.0_dp]
      verts(:, 3) = [0.2_dp, 0.9_dp, 0.0_dp]
      call write_triangle_cache(trim(cache), sigma)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'reload triangle')
      call expect(tan_nplanes() == 1 .and. tan_nedges() == 3, 'triangle np,ne')

      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call vos_triangle(verts, p, kref)
      bref = (sigma / (4.0_dp * acos(-1.0_dp))) * kref
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'triangle scalar +z')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'triangle vs VOS at (0.4,0.3,0.5)')
      call tan_eval(p, 1, 0, 1.0_dp, coeff2, status)
      call expect(status == TAN_OK, 'triangle mode1 order0')
      call expect_close(coeff(:, 1), coeff2(:, 1), 1.0e-15_dp, 0.0_dp, &
           'mode0 vs mode1 order0 constants')

      p = [0.4_dp, 0.3_dp, -0.5_dp]
      call vos_triangle(verts, p, kref)
      bref = (sigma / (4.0_dp * acos(-1.0_dp))) * kref
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'triangle scalar -z')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'triangle vs VOS at (0.4,0.3,-0.5)')

      p = [2.0_dp, 2.0_dp, 1.0_dp]
      call vos_triangle(verts, p, kref)
      bref = (sigma / (4.0_dp * acos(-1.0_dp))) * kref
      call tan_eval(p, 1, 6, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'triangle jet far')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'triangle jet constant vs VOS far')

      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call tan_eval(p, 1, 6, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'triangle q6')
      call vos_triangle(verts, p, kref)
      bref = (sigma / (4.0_dp * acos(-1.0_dp))) * kref
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'q6 constant matches VOS')
      call fd_linear_check(p, 1.0e-4_dp, 1.0e-8_dp, 5.0e-5_dp, &
           'triangle FD linear q6')
   end subroutine check_triangle

   subroutine check_rectangle()
      real(dp) :: p(3), corners(3, 4)
      sigma = 0.8_dp
      bounds = [-1.0_dp, 1.0_dp, -0.75_dp, 0.75_dp]
      call write_rectangle_cache(trim(cache2), bounds, 0.0_dp, sigma)
      call tan_load(trim(cache2), status)
      call expect(status == TAN_OK, 'load rectangle')
      call expect(tan_nplanes() == 1 .and. tan_nedges() == 4, 'rectangle np,ne')
      corners(:, 1) = [-1.0_dp, -0.75_dp, 0.0_dp]
      corners(:, 2) = [ 1.0_dp, -0.75_dp, 0.0_dp]
      corners(:, 3) = [ 1.0_dp,  0.75_dp, 0.0_dp]
      corners(:, 4) = [-1.0_dp,  0.75_dp, 0.0_dp]

      p = [0.125_dp, 0.09375_dp, 1.0_dp]
      call rectangle_field(corners, sigma, p, bref)
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'rectangle ordinary')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'rectangle vs corner at z=1')

      p = [0.125_dp, 0.09375_dp, 0.0_dp]
      call rectangle_field(corners, sigma, p, bref)
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'rectangle on-plane interior scalar')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'on-plane interior average trace')
      call expect(abs(coeff(3, 1)) <= 1.0e-14_dp, 'Bz=0 on plane interior')

      p = [1.4_dp, 0.2_dp, 0.1_dp]
      call rectangle_field(corners, sigma, p, bref)
      call tan_eval(p, 1, 6, 0.05_dp, coeff, status)
      call expect(status == TAN_OK, 'rectangle exterior offset q6')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'rectangle jet constant at (1.4,0.2,0.1)')

      p = [0.0_dp, 0.0_dp, 2.0_dp]
      call rectangle_field(corners, sigma, p, bref)
      call tan_eval(p, 1, 2, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'rectangle axis point')
      call expect_close(coeff(:, 1), bref, 1.0e-12_dp, 1.0e-10_dp, &
           'rectangle at (0,0,2)')
      call expect(abs(coeff(1, 1)) + abs(coeff(2, 1)) <= 1.0e-14_dp, &
           'Bx=By=0 on z-axis of centered rectangle')
   end subroutine check_rectangle

   subroutine check_exterior_continuation()
      real(dp) :: p(3), pp(3), pm(3), cplus(3, TAN_NM), cminus(3, TAN_NM)
      real(dp) :: fd, rad, h, recon
      integer :: nmono, k
      bounds = [-1.0_dp, 1.0_dp, -0.75_dp, 0.75_dp]
      call write_rectangle_cache(trim(cache2), bounds, 0.0_dp, 0.8_dp)
      call tan_load(trim(cache2), status)
      p = [2.0_dp, 2.0_dp, 0.0_dp]
      rad = 0.02_dp
      call tan_eval(p, 1, 4, rad, coeff, status)
      call expect(status == TAN_OK, 'exterior-plane DA accepted')
      call expect(abs(coeff(3, 1)) <= 1.0e-14_dp, 'continued Bz const is 0')
      ! monomial 2 is (0,0,1): dB/du_z, physical dB/dz = coeff / radius
      call expect(abs(coeff(3, 2)) > 1.0e-8_dp, &
           'continued dBz/du_z is nonzero')
      h = 0.01_dp
      pp = p; pp(3) = rad * h
      pm = p; pm(3) = -rad * h
      call tan_eval(pp, 1, 0, 1.0_dp, cplus, status)
      call tan_eval(pm, 1, 0, 1.0_dp, cminus, status)
      fd = (cplus(3, 1) - cminus(3, 1)) / (2.0_dp * h)
      call expect(abs(fd - coeff(3, 2)) <= 1.0e-10_dp + 1.0e-6_dp * abs(fd), &
           'exterior dBz/du_z matches centered FD')
      if (abs(fd - coeff(3, 2)) > 1.0e-10_dp + 1.0e-6_dp * abs(fd)) then
         write(error_unit, '(a,3es16.8e3)') '  jet dBz/duz, FD, const', &
              coeff(3, 2), fd, coeff(3, 1)
      end if
      ! Reconstruct Bz at u=(0,0,1) from the continued jet.
      recon = 0.0_dp
      call tan_exponents(4, ex, ey, ez, nmono)
      do k = 1, nmono
         if (ex(k) == 0 .and. ey(k) == 0) then
            recon = recon + coeff(3, k)
         end if
      end do
      pp = p; pp(3) = rad
      call tan_eval(pp, 1, 0, 1.0_dp, cplus, status)
      call expect(abs(recon - cplus(3, 1)) <= 1.0e-10_dp + 1.0e-6_dp * abs(cplus(3, 1)), &
           'exterior q4 reconstructs Bz at u_z=1')
      ! interior on-plane DA must reject
      p = [0.125_dp, 0.09375_dp, 0.0_dp]
      call tan_eval(p, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'interior on-plane DA rejected')
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'interior on-plane scalar still ok')
   end subroutine check_exterior_continuation

   subroutine check_rejections()
      real(dp) :: p(3)
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      p = [0.0_dp, 0.0_dp, 0.0_dp]
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_EDGE, 'vertex rejected')
      p = [0.6_dp, 0.0_dp, 0.0_dp]
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_EDGE, 'edge midpoint rejected')
      p = [0.6_dp, 0.0_dp, 0.0_dp]
      call tan_eval(p, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_ERR_EDGE, 'DA on edge rejected')
      p = [0.4_dp, 0.3_dp, 0.0_dp]
      call tan_eval(p, 1, 1, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'triangle interior DA rejected')
   end subroutine check_rejections

   subroutine check_canceled()
      real(dp) :: p(3)
      call write_canceled_triangle(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load canceled triangle')
      call expect(tan_nedges() == 6, 'canceled cache keeps 6 opposite edges')
      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call tan_eval(p, 1, 2, 0.05_dp, coeff, status)
      call expect(status == TAN_OK, 'canceled off-plane eval')
      call expect(sqrt(sum(coeff(:, 1)**2)) <= 1.0e-12_dp, &
           'canceled field is zero off plane')
      p = [0.4_dp, 0.3_dp, 0.0_dp]
      call tan_eval(p, 1, 3, 0.02_dp, coeff, status)
      call expect(status == TAN_OK, 'canceled interior DA continuation allowed')
      call expect(sqrt(sum(coeff(:, 1)**2)) <= 1.0e-12_dp, &
           'canceled on-plane field is zero')
   end subroutine check_canceled

   subroutine check_reload()
      real(dp) :: p(3), btri(3), brect(3)
      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      btri = coeff(:, 1)
      call write_rectangle_cache(trim(cache2), [-1.0_dp, 1.0_dp, -0.75_dp, 0.75_dp], &
           0.0_dp, 0.8_dp)
      call tan_load(trim(cache2), status)
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      brect = coeff(:, 1)
      call expect(sqrt(sum((btri - brect)**2)) > 1.0e-6_dp, &
           'reload changed the active cache')
      call tan_load(trim(cache), status)
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect_close(coeff(:, 1), btri, 1.0e-15_dp, 0.0_dp, &
           'reload original cache restores field')
   end subroutine check_reload

   subroutine check_batch_and_c()
      use, intrinsic :: iso_c_binding
      real(c_double) :: pts(3, 3), fields(3, 3)
      integer(c_int) :: sts(3), cst, cmode
      character(kind=c_char) :: cpath(256)
      integer :: n, k
      call write_triangle_cache(trim(cache), 0.8_dp)
      n = len_trim(cache)
      cpath = c_null_char
      do k = 1, n
         cpath(k) = cache(k:k)
      end do
      cpath(n + 1) = c_null_char
      call tan_load_c(cpath, cst)
      call expect(int(cst) == TAN_OK, 'tan_load_c')
      pts(:, 1) = [0.4_dp, 0.3_dp, 0.5_dp]
      pts(:, 2) = [2.0_dp, 2.0_dp, 1.0_dp]
      pts(:, 3) = [0.0_dp, 0.0_dp, 0.0_dp]
      cmode = 1
      call tan_batch_c(3_c_int, pts, cmode, fields, sts)
      call expect(int(sts(1)) == TAN_OK, 'batch point 1 ok')
      call expect(int(sts(2)) == TAN_OK, 'batch point 2 ok')
      call expect(int(sts(3)) == TAN_ERR_EDGE, 'batch vertex rejected')
      call tan_eval(pts(:, 1), 1, 0, 1.0_dp, coeff, status)
      call expect_close(fields(:, 1), coeff(:, 1), 0.0_dp, 0.0_dp, &
           'batch matches tan_eval')
   end subroutine check_batch_and_c

   subroutine check_modes_and_auto()
      real(dp) :: p(3), b0(3), b1(3), b2(3)
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK .and. tan_last_mode() == 0, &
           'mode0 uses binary64 sourcepath')
      b0 = coeff(:, 1)
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK .and. tan_last_mode() == 1, &
           'mode1 uses real128 sourcepath')
      b1 = coeff(:, 1)
      call expect_close(b0, b1, 1.0e-12_dp, 1.0e-10_dp, &
           'mode0 and mode1 agree on regular triangle point')
      call tan_eval(p, 2, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'mode2 regular accepted')
      call expect(tan_last_mode() == 0, 'mode2 regular routes to binary64')
      b2 = coeff(:, 1)
      call expect_close(b2, b1, 1.0e-12_dp, 1.0e-10_dp, &
           'mode2 regular matches mode1')
      p = [0.4_dp, 0.3_dp, 1.0e-6_dp]
      call tan_eval(p, 2, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'mode2 near-plane accepted')
      call expect(tan_last_mode() == 1, 'mode2 near-plane routes to real128')
      p = [0.4_dp, 0.3_dp, 100.0_dp]
      call tan_eval(p, 2, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'mode2 far accepted')
      call expect(tan_last_mode() == 1, 'mode2 far routes to real128')
   end subroutine check_modes_and_auto

   subroutine check_oblique_sheet()
      real(dp) :: p(3), pp(3), pm(3)
      integer :: u
      open(newunit=u, file=trim(cache), status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 3'
      write(u, '(a)') '0 0 0 1 2 3'
      write(u, '(a)') '1 0 0 0 2 -1 0 1'
      write(u, '(a)') '1 2 -1 0 3 0 -1 1'
      write(u, '(a)') '1 3 0 -1 0 0 0 1'
      close(u)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load oblique n=(1,2,3) triangle')
      p = [1.25_dp, -0.25_dp, -0.25_dp]
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique exact-sheet scalar average')
      call tan_eval(p, 1, 0, 1.0_dp, coeff2, status)
      call expect(status == TAN_OK, 'oblique exact-sheet mode1 scalar')
      call expect_close(coeff(:, 1), coeff2(:, 1), 1.0e-12_dp, 1.0e-10_dp, &
           'oblique sheet mode0 vs mode1')
      call tan_eval(p, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'oblique exact-sheet DA rejected')
      pp = p
      pp(1) = nearest(p(1), 1.0_dp)
      call tan_eval(pp, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique one-ULP plus DA accepted')
      pm = p
      pm(1) = nearest(p(1), -1.0_dp)
      call tan_eval(pm, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique one-ULP minus DA accepted')
      call tan_eval(pp, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique one-ULP plus scalar')
      call tan_eval(pm, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique one-ULP minus scalar')
      p = [1.0_dp, 1.0_dp, -1.0_dp]
      call tan_eval(p, 1, 2, 0.01_dp, coeff, status)
      call expect(status == TAN_OK, 'oblique on-plane exterior DA continuation')
   end subroutine check_oblique_sheet

   subroutine check_holes_and_tiny()
      real(dp) :: p(3)
      integer :: u
      ! Full hole: outer +1, inner -1. Inner centre has zero signed density.
      open(newunit=u, file=trim(cache), status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 8'
      write(u, '(a)') '0 0 0 0 0 1'
      write(u, '(a)') '1 0 0 0 4 0 0 1'
      write(u, '(a)') '1 4 0 0 4 4 0 1'
      write(u, '(a)') '1 4 4 0 0 4 0 1'
      write(u, '(a)') '1 0 4 0 0 0 0 1'
      write(u, '(a)') '1 1 1 0 3 1 0 -1'
      write(u, '(a)') '1 3 1 0 3 3 0 -1'
      write(u, '(a)') '1 3 3 0 1 3 0 -1'
      write(u, '(a)') '1 1 3 0 1 1 0 -1'
      close(u)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load full hole')
      p = [2.0_dp, 2.0_dp, 0.0_dp]
      call tan_eval(p, 1, 2, 0.05_dp, coeff, status)
      call expect(status == TAN_OK, 'full hole interior DA continuation')
      p = [0.5_dp, 0.5_dp, 0.0_dp]
      call tan_eval(p, 1, 2, 0.05_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'full hole frame DA rejected')

      ! Partial hole: inner weight -0.5 leaves nonzero signed density.
      open(newunit=u, file=trim(cache), status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 8'
      write(u, '(a)') '0 0 0 0 0 1'
      write(u, '(a)') '1 0 0 0 4 0 0 1'
      write(u, '(a)') '1 4 0 0 4 4 0 1'
      write(u, '(a)') '1 4 4 0 0 4 0 1'
      write(u, '(a)') '1 0 4 0 0 0 0 1'
      write(u, '(a)') '1 1 1 0 3 1 0 -0.5'
      write(u, '(a)') '1 3 1 0 3 3 0 -0.5'
      write(u, '(a)') '1 3 3 0 1 3 0 -0.5'
      write(u, '(a)') '1 1 3 0 1 1 0 -0.5'
      close(u)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load partial hole')
      p = [2.0_dp, 2.0_dp, 0.0_dp]
      call tan_eval(p, 1, 2, 0.05_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'partial hole DA rejected')
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'partial hole scalar still ok')

      ! Tiny but nonzero charge remains a sheet.
      open(newunit=u, file=trim(cache), status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 4'
      write(u, '(a)') '0 0 0 0 0 1'
      write(u, '(a)') '1 0 0 0 1 0 0 5.551115123125783e-17'
      write(u, '(a)') '1 1 0 0 1 1 0 5.551115123125783e-17'
      write(u, '(a)') '1 1 1 0 0 1 0 5.551115123125783e-17'
      write(u, '(a)') '1 0 1 0 0 0 0 5.551115123125783e-17'
      close(u)
      call tan_load(trim(cache), status)
      call expect(status == TAN_OK, 'load tiny-charge sheet')
      p = [0.25_dp, 0.25_dp, 0.0_dp]
      call tan_eval(p, 1, 1, 0.01_dp, coeff, status)
      call expect(status == TAN_ERR_WINDING, 'tiny-charge sheet DA rejected')
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_OK, 'tiny-charge sheet scalar ok')
   end subroutine check_holes_and_tiny

   subroutine check_stale_load()
      real(dp) :: p(3), btri(3)
      p = [0.4_dp, 0.3_dp, 0.5_dp]
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      btri = coeff(:, 1)
      call tan_load('results/no_such_cache_file.tan', status)
      call expect(status == TAN_ERR_IO, 'failed load after success')
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_NO_CACHE, 'failed load cannot eval stale field')
      call expect(all(coeff == 0.0_dp), 'stale-load eval zeros coefficients')
      open(newunit=i, file='results/bad.cache', status='replace')
      write(i, '(a)') 'NOT_A_CACHE'
      close(i)
      call write_triangle_cache(trim(cache), 0.8_dp)
      call tan_load(trim(cache), status)
      call tan_load('results/bad.cache', status)
      call expect(status == TAN_ERR_FORMAT, 'malformed load after success')
      call tan_eval(p, 0, 0, 1.0_dp, coeff, status)
      call expect(status == TAN_ERR_NO_CACHE, 'malformed load cannot eval stale field')
      call tan_load(trim(cache), status)
      call tan_eval(p, 1, 0, 1.0_dp, coeff, status)
      call expect_close(coeff(:, 1), btri, 0.0_dp, 0.0_dp, &
           'reload after failed loads restores field')
   end subroutine check_stale_load

   subroutine fd_linear_check(p, rad, atol, rtol, msg)
      real(dp), intent(in) :: p(3), rad, atol, rtol
      character(len=*), intent(in) :: msg
      real(dp) :: c0(3, TAN_NM), cp(3, TAN_NM), cm(3, TAN_NM)
      real(dp) :: q(3), fd(3), jetv(3)
      integer :: axis, st, idx
      call tan_eval(p, 1, 2, rad, c0, st)
      if (st /= TAN_OK) then
         call expect(.false., msg // ' (center eval)')
         return
      end if
      do axis = 1, 3
         q = p
         q(axis) = p(axis) + rad
         call tan_eval(q, 1, 0, 1.0_dp, cp, st)
         q(axis) = p(axis) - rad
         call tan_eval(q, 1, 0, 1.0_dp, cm, st)
         fd = 0.5_dp * (cp(:, 1) - cm(:, 1))
         if (axis == 1) idx = 4
         if (axis == 2) idx = 3
         if (axis == 3) idx = 2
         jetv = c0(:, idx)
         call expect_close(jetv, fd, atol, rtol, msg)
      end do
   end subroutine fd_linear_check

   subroutine write_triangle_cache(path, sig)
      character(len=*), intent(in) :: path
      real(dp), intent(in) :: sig
      integer :: u
      real(wp) :: v1(3), v2(3), v3(3), n(3)
      v1 = [0.0_wp, 0.0_wp, 0.0_wp]
      v2 = [1.2_wp, 0.0_wp, 0.0_wp]
      v3 = [0.2_wp, 0.9_wp, 0.0_wp]
      n = [0.0_wp, 0.0_wp, 1.0_wp]
      open(newunit=u, file=path, status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 3'
      write(u, '(6(1x,es50.40e4))') v1, n
      write(u, '(i0,7(1x,es50.40e4))') 1, v1, v2, real(sig, wp)
      write(u, '(i0,7(1x,es50.40e4))') 1, v2, v3, real(sig, wp)
      write(u, '(i0,7(1x,es50.40e4))') 1, v3, v1, real(sig, wp)
      close(u)
   end subroutine write_triangle_cache

   subroutine write_canceled_triangle(path, sig)
      character(len=*), intent(in) :: path
      real(dp), intent(in) :: sig
      integer :: u
      real(wp) :: v1(3), v2(3), v3(3), n(3), w
      v1 = [0.0_wp, 0.0_wp, 0.0_wp]
      v2 = [1.2_wp, 0.0_wp, 0.0_wp]
      v3 = [0.2_wp, 0.9_wp, 0.0_wp]
      n = [0.0_wp, 0.0_wp, 1.0_wp]
      w = real(sig, wp)
      open(newunit=u, file=path, status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 6'
      write(u, '(6(1x,es50.40e4))') v1, n
      write(u, '(i0,7(1x,es50.40e4))') 1, v1, v2, w
      write(u, '(i0,7(1x,es50.40e4))') 1, v2, v3, w
      write(u, '(i0,7(1x,es50.40e4))') 1, v3, v1, w
      write(u, '(i0,7(1x,es50.40e4))') 1, v2, v1, w
      write(u, '(i0,7(1x,es50.40e4))') 1, v3, v2, w
      write(u, '(i0,7(1x,es50.40e4))') 1, v1, v3, w
      close(u)
   end subroutine write_canceled_triangle

   subroutine write_rectangle_cache(path, bnds, z0, sig)
      character(len=*), intent(in) :: path
      real(dp), intent(in) :: bnds(4), z0, sig
      integer :: u
      real(wp) :: c1(3), c2(3), c3(3), c4(3), n(3), w, z
      z = real(z0, wp)
      w = real(sig, wp)
      c1 = [real(bnds(1), wp), real(bnds(3), wp), z]
      c2 = [real(bnds(2), wp), real(bnds(3), wp), z]
      c3 = [real(bnds(2), wp), real(bnds(4), wp), z]
      c4 = [real(bnds(1), wp), real(bnds(4), wp), z]
      n = [0.0_wp, 0.0_wp, 1.0_wp]
      open(newunit=u, file=path, status='replace')
      write(u, '(a)') 'TAN_CACHE_V1'
      write(u, '(a)') '1 4'
      write(u, '(6(1x,es50.40e4))') c1, n
      write(u, '(i0,7(1x,es50.40e4))') 1, c1, c2, w
      write(u, '(i0,7(1x,es50.40e4))') 1, c2, c3, w
      write(u, '(i0,7(1x,es50.40e4))') 1, c3, c4, w
      write(u, '(i0,7(1x,es50.40e4))') 1, c4, c1, w
      close(u)
   end subroutine write_rectangle_cache

   subroutine vos_triangle(verts, p, k)
      real(dp), intent(in) :: verts(3, 3), p(3)
      real(dp), intent(out) :: k(3)
      real(wp) :: v(3, 3), q(3), n(3), e1(3), e2(3), r(3, 3), rho(3)
      real(wp) :: area2, mag, h, num, den, omega, t(3), length, outward(3)
      real(wp) :: u, w, offset(3), cr(3), rho_perp, baseline, corr, gap, ll
      integer :: i, j
      do i = 1, 3
         v(:, i) = real(verts(:, i), wp)
      end do
      q = real(p, wp)
      e1 = v(:, 2) - v(:, 1)
      e2 = v(:, 3) - v(:, 1)
      n = cross_wp(e1, e2)
      mag = sqrt(dot_wp(n, n))
      area2 = mag
      n = n / mag
      do i = 1, 3
         r(:, i) = v(:, i) - q
         rho(i) = sqrt(dot_wp(r(:, i), r(:, i)))
      end do
      h = dot_wp(q - v(:, 1), n)
      num = -h * area2
      den = rho(1)*rho(2)*rho(3) + dot_wp(r(:,1), r(:,2))*rho(3) &
           + dot_wp(r(:,2), r(:,3))*rho(1) + dot_wp(r(:,3), r(:,1))*rho(2)
      if (h == 0.0_wp) then
         omega = 0.0_wp
      else
         omega = 2.0_wp * atan2(num, den)
      end if
      k = real(-omega * n, dp)
      do i = 1, 3
         j = i + 1
         if (j > 3) j = 1
         t = v(:, j) - v(:, i)
         length = sqrt(dot_wp(t, t))
         t = t / length
         outward = cross_wp(t, n)
         u = dot_wp(q - v(:, i), t)
         w = dot_wp(v(:, j) - q, t)
         if (abs(u) <= abs(w)) then
            offset = q - v(:, i)
         else
            offset = q - v(:, j)
         end if
         cr = cross_wp(offset, t)
         rho_perp = sqrt(dot_wp(cr, cr))
         baseline = 2.0_wp * (max(-u, 0.0_wp) + max(-w, 0.0_wp))
         corr = rho_perp * (rho_perp / (rho(i) + abs(u)) + rho_perp / (rho(j) + abs(w)))
         gap = baseline + corr
         ll = log(1.0_wp + 2.0_wp * length / gap)
         k = k + real(outward * ll, dp)
      end do
   end subroutine vos_triangle

   subroutine rectangle_field(corners, sig, p, b)
      real(dp), intent(in) :: corners(3, 4), sig, p(3)
      real(dp), intent(out) :: b(3)
      real(wp) :: origin(3), uu(3), vv(3), nn(3), uh(3), vh(3)
      real(wp) :: Ulen, Vlen, rel(3), X, Y, Z, acc(3), xi, yj, Xx, Yy, R, sgn
      integer :: i, j
      origin = real(corners(:, 1), wp)
      uu = real(corners(:, 2), wp) - origin
      vv = real(corners(:, 4), wp) - origin
      nn = cross_wp(uu, vv)
      nn = nn / sqrt(dot_wp(nn, nn))
      Ulen = sqrt(dot_wp(uu, uu))
      uh = uu / Ulen
      vh = cross_wp(nn, uh)
      Vlen = dot_wp(vv, vh)
      rel = real(p, wp) - origin
      X = dot_wp(rel, uh)
      Y = dot_wp(rel, vh)
      Z = dot_wp(rel, nn)
      acc = 0.0_wp
      do i = 0, 1
         if (i == 0) then
            xi = 0.0_wp
         else
            xi = Ulen
         end if
         do j = 0, 1
            if (j == 0) then
               yj = 0.0_wp
            else
               yj = Vlen
            end if
            if (mod(i + j, 2) == 0) then
               sgn = 1.0_wp
            else
               sgn = -1.0_wp
            end if
            Xx = X - xi
            Yy = Y - yj
            R = sqrt(Xx*Xx + Yy*Yy + Z*Z)
            acc(1) = acc(1) + sgn * log_R_minus(R, Yy, Xx, Z)
            acc(2) = acc(2) + sgn * log_R_minus(R, Xx, Yy, Z)
            if (Z == 0.0_wp) then
               acc(3) = acc(3)
            else
               acc(3) = acc(3) + sgn * atan(Xx * Yy / (Z * R))
            end if
         end do
      end do
      acc = acc * real(sig, wp) / four_pi
      b(1) = real(acc(1)*uh(1) + acc(2)*vh(1) + acc(3)*nn(1), dp)
      b(2) = real(acc(1)*uh(2) + acc(2)*vh(2) + acc(3)*nn(2), dp)
      b(3) = real(acc(1)*uh(3) + acc(2)*vh(3) + acc(3)*nn(3), dp)
   end subroutine rectangle_field

   real(wp) function log_R_minus(R, coord, other, Z)
      real(wp), intent(in) :: R, coord, other, Z
      if (coord <= 0.0_wp) then
         log_R_minus = log(R - coord)
      else
         log_R_minus = log(other*other + Z*Z) - log(R + coord)
      end if
   end function log_R_minus

   real(wp) function dot_wp(a, b)
      real(wp), intent(in) :: a(3), b(3)
      dot_wp = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
   end function dot_wp

   function cross_wp(a, b) result(c)
      real(wp), intent(in) :: a(3), b(3)
      real(wp) :: c(3)
      c(1) = a(2)*b(3) - a(3)*b(2)
      c(2) = a(3)*b(1) - a(1)*b(3)
      c(3) = a(1)*b(2) - a(2)*b(1)
   end function cross_wp

end program test_tan
