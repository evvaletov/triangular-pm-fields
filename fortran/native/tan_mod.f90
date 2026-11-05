! tan_mod.f90 — native cached plane-boundary assembly, scalar and degree-6 jets.
! Geometry is consumed from a TAN_CACHE_V1 text file. Mode 0 is a cached
! binary64 scalar kernel. Mode 1 is real128 scalar/DA. Mode 2 dispatches
! scalar queries to binary64 only in guarded regular cases, else real128.
! Ambiguous plane/edge/support decisions use fortran/triangle_exact_geometry.c.
module tan_mod
   use, intrinsic :: iso_fortran_env, only: wp => real128, dp => real64, int64
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic
   use tan_jet_mod
   implicit none
   private

   public :: tan_load, tan_eval, tan_nmono, tan_exponents, tan_last_mode, tan_clear
   public :: tan_load_c, tan_eval_c, tan_batch_c, tan_last_mode_c
   public :: tan_loaded, tan_nplanes, tan_nedges
   public :: TAN_OK, TAN_ERR_NO_CACHE, TAN_ERR_IO, TAN_ERR_FORMAT
   public :: TAN_ERR_MODE, TAN_ERR_ORDER, TAN_ERR_POINT, TAN_ERR_RADIUS
   public :: TAN_ERR_EDGE, TAN_ERR_WINDING, TAN_ERR_NONFINITE, TAN_ERR_DOMAIN
   public :: TAN_QMAX, TAN_NM, TAN_NPAIR
   public :: expo_x, expo_y, expo_z, expo_deg, nmono_of

   integer, parameter :: LINE_LEN = 262144
   integer, parameter :: RAW_CAP = 128
   real(dp), parameter :: PI_DP = 3.14159265358979323846264338327950288_dp
   real(dp), parameter :: FOUR_PI_DP = 4.0_dp * PI_DP
   real(dp), parameter :: CANCEL_LIMIT = 5000.0_dp
   real(dp), parameter :: FAR_RATIO = 10.0_dp
   real(dp), parameter :: CLOSE_RATIO = 1.0e-4_dp
   real(dp), parameter :: THIN_RATIO = 0.01_dp
   real(dp), parameter :: EXTENT_HUGE = 1.0e8_dp
   real(dp), parameter :: EXTENT_TINY = 1.0e-8_dp
   real(dp), parameter :: SCALE_RATIO = 1.0e6_dp

   type :: tan_plane
      real(wp) :: a(3) = 0.0_wp
      real(wp) :: nraw(3) = 0.0_wp
      real(wp) :: nhat(3) = 0.0_wp
      real(wp) :: nnorm = 0.0_wp
      real(dp) :: a_dp(3) = 0.0_dp
      real(dp) :: nraw_dp(3) = 0.0_dp
      real(dp) :: nhat_dp(3) = 0.0_dp
      real(dp) :: nnorm_dp = 0.0_dp
      real(dp) :: ell = 0.0_dp
      real(dp) :: extent = 0.0_dp
      real(dp) :: scale = 0.0_dp
      real(dp) :: area = 0.0_dp
      real(dp) :: max_abs_w = 0.0_dp
      logical :: force_quad = .false.
      logical :: dp_filter_ok = .false.
   end type tan_plane

   type :: tan_edge
      integer :: iplane = 0
      integer :: src0 = 0
      real(wp) :: a(3) = 0.0_wp
      real(wp) :: b(3) = 0.0_wp
      real(wp) :: chord(3) = 0.0_wp
      real(wp) :: w = 0.0_wp
      real(wp) :: length = 0.0_wp
      real(wp) :: t(3) = 0.0_wp
      real(wp) :: nxt(3) = 0.0_wp
      real(dp) :: a_dp(3) = 0.0_dp
      real(dp) :: b_dp(3) = 0.0_dp
      real(dp) :: chord_dp(3) = 0.0_dp
      real(dp) :: w_dp = 0.0_dp
      real(dp) :: length_dp = 0.0_dp
      real(dp) :: t_dp(3) = 0.0_dp
      real(dp) :: nxt_dp(3) = 0.0_dp
   end type tan_edge

   logical, save :: have_cache = .false.
   integer, save :: npl = 0, ned = 0
   integer, save :: last_mode_used = -1
   logical, save :: cache_force_quad = .false.
   real(dp), save :: cache_ell = 0.0_dp
   type(tan_plane), allocatable, save :: planes(:)
   type(tan_edge), allocatable, save :: edges(:)
   integer, allocatable, save :: e0(:), e1(:)

   interface
      function tan_exact_load_c_impl(path) bind(C, name='tan_exact_load_c')
         import :: c_char, c_int
         integer(c_int) :: tan_exact_load_c_impl
         character(kind=c_char), intent(in) :: path(*)
      end function tan_exact_load_c_impl

      subroutine tan_exact_clear_c_impl() bind(C, name='tan_exact_clear_c')
      end subroutine tan_exact_clear_c_impl

      function tan_exact_plane_c_impl(plane0, p, raw_height, capacity) &
           bind(C, name='tan_exact_plane_c')
         import :: c_int, c_double, c_char
         integer(c_int) :: tan_exact_plane_c_impl
         integer(c_int), value :: plane0, capacity
         real(c_double), intent(in) :: p(3)
         character(kind=c_char), intent(out) :: raw_height(*)
      end function tan_exact_plane_c_impl

      function tan_exact_edge_c_impl(edge0, p) bind(C, name='tan_exact_edge_c')
         import :: c_int, c_double
         integer(c_int) :: tan_exact_edge_c_impl
         integer(c_int), value :: edge0
         real(c_double), intent(in) :: p(3)
      end function tan_exact_edge_c_impl

      function tan_exact_support_c_impl(plane0, p) bind(C, name='tan_exact_support_c')
         import :: c_int, c_double
         integer(c_int) :: tan_exact_support_c_impl
         integer(c_int), value :: plane0
         real(c_double), intent(in) :: p(3)
      end function tan_exact_support_c_impl
   end interface

contains

   logical function tan_loaded()
      tan_loaded = have_cache
   end function tan_loaded

   integer function tan_nplanes()
      tan_nplanes = npl
   end function tan_nplanes

   integer function tan_nedges()
      tan_nedges = ned
   end function tan_nedges

   integer function tan_last_mode()
      tan_last_mode = last_mode_used
   end function tan_last_mode

   integer(c_int) function tan_last_mode_c() bind(C, name='tan_last_mode_c')
      tan_last_mode_c = int(last_mode_used, c_int)
   end function tan_last_mode_c

   subroutine tan_exponents(order, ex, ey, ez, n)
      integer, intent(in) :: order
      integer, intent(out) :: ex(:), ey(:), ez(:), n
      integer :: i
      call tan_jet_init()
      if (order < 0 .or. order > TAN_QMAX) then
         n = 0
         return
      end if
      n = nmono_of(order)
      do i = 1, n
         ex(i) = expo_x(i)
         ey(i) = expo_y(i)
         ez(i) = expo_z(i)
      end do
   end subroutine tan_exponents

   subroutine tan_clear()
      have_cache = .false.
      npl = 0
      ned = 0
      cache_force_quad = .false.
      cache_ell = 0.0_dp
      last_mode_used = -1
      if (allocated(planes)) deallocate(planes)
      if (allocated(edges)) deallocate(edges)
      if (allocated(e0)) deallocate(e0)
      if (allocated(e1)) deallocate(e1)
      call tan_exact_clear_c_impl()
   end subroutine tan_clear

   subroutine tan_load(path, status)
      character(len=*), intent(in) :: path
      integer, intent(out) :: status
      integer :: unit, ios, i, np_in, ne_in, ip, k
      integer(int64) :: file_bytes, minimum_bytes
      character(len=:), allocatable :: line, cpath
      type(tan_plane), allocatable :: pl(:)
      type(tan_edge), allocatable :: ed(:), grouped(:)
      integer, allocatable :: count(:), start(:), cursor(:)
      real(wp) :: mag, ax, ay, az, bx, by, bz, w, nx, ny, nz
      real(wp) :: chord(3), length, t(3), nxt(3)
      integer(c_int) :: cst
      real(dp) :: xmin(3), xmax(3), area_raw, cr(3), maxw, ell, extent, scale
      real(dp) :: thin, nrm

      call tan_jet_init()
      call tan_clear()
      status = TAN_OK
      np_in = 0
      ne_in = 0
      allocate(character(len=LINE_LEN) :: line, stat=ios)
      if (ios /= 0) then
         status = TAN_ERR_IO
         return
      end if

      open(newunit=unit, file=trim(path), status='old', action='read', &
           iostat=ios)
      if (ios /= 0) then
         status = TAN_ERR_IO
         return
      end if

      call read_nonempty(unit, line, ios)
      if (ios /= 0) then
         close(unit)
         status = TAN_ERR_FORMAT
         return
      end if
      if (trim(adjustl(line)) /= 'TAN_CACHE_V1') then
         close(unit)
         status = TAN_ERR_FORMAT
         return
      end if

      call read_nonempty(unit, line, ios)
      if (ios /= 0) then
         close(unit)
         status = TAN_ERR_FORMAT
         return
      end if
      read(line, *, iostat=ios) np_in, ne_in
      if (ios /= 0 .or. np_in < 0 .or. ne_in < 0) then
         close(unit)
         status = TAN_ERR_FORMAT
         return
      end if

      ! A plane/edge needs at least 11/15 text bytes (six/eight one-character
      ! tokens plus separators). The entire file size is a conservative bound;
      ! the C exact parser later checks the remaining bytes and full grammar.
      ! Convert BEFORE arithmetic, and reserve room for one-past-end indices.
      file_bytes = -1_int64
      inquire(unit=unit, size=file_bytes, iostat=ios)
      if (ios /= 0 .or. file_bytes < 0_int64) then
         close(unit)
         status = TAN_ERR_IO
         return
      end if
      minimum_bytes = 11_int64 * int(np_in, int64) + 15_int64 * int(ne_in, int64)
      if (minimum_bytes > file_bytes .or. np_in == huge(np_in) .or. ne_in == huge(ne_in)) then
         close(unit)
         status = TAN_ERR_FORMAT
         return
      end if
      allocate(pl(max(np_in, 1)), ed(max(ne_in, 1)), stat=ios)
      if (ios /= 0) then
         close(unit)
         status = TAN_ERR_IO
         return
      end if
      do i = 1, np_in
         call read_nonempty(unit, line, ios)
         if (ios /= 0) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         read(line, *, iostat=ios) ax, ay, az, nx, ny, nz
         if (ios /= 0) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         if (.not. all_finite6(ax, ay, az, nx, ny, nz)) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         mag = hypot3(nx, ny, nz)
         if (mag == 0.0_wp) then
            close(unit)
            status = TAN_ERR_DOMAIN
            return
         end if
         pl(i)%a = [ax, ay, az]
         pl(i)%nraw = [nx, ny, nz]
         pl(i)%nnorm = mag
         pl(i)%nhat = [nx, ny, nz] / mag
         pl(i)%a_dp = real(pl(i)%a, dp)
         pl(i)%nraw_dp = real(pl(i)%nraw, dp)
         pl(i)%nnorm_dp = real(mag, dp)
         if (pl(i)%nnorm_dp == 0.0_dp) then
            close(unit)
            status = TAN_ERR_DOMAIN
            return
         end if
         pl(i)%nhat_dp = real(pl(i)%nhat, dp)
         nrm = hypot3_dp(pl(i)%nhat_dp(1), pl(i)%nhat_dp(2), pl(i)%nhat_dp(3))
         if (nrm == 0.0_dp) then
            close(unit)
            status = TAN_ERR_DOMAIN
            return
         end if
         pl(i)%nhat_dp = pl(i)%nhat_dp / nrm
         pl(i)%dp_filter_ok = all(real(pl(i)%a_dp,wp) == pl(i)%a) .and. &
              all(abs(pl(i)%nhat) >= real(tiny(1.0_dp),wp) .or. pl(i)%nhat == 0.0_wp)
      end do

      do i = 1, ne_in
         call read_nonempty(unit, line, ios)
         if (ios /= 0) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         read(line, *, iostat=ios) ip, ax, ay, az, bx, by, bz, w
         if (ios /= 0) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         if (.not. all_finite7(ax, ay, az, bx, by, bz, w)) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         if (ip < 1 .or. ip > np_in) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         if (w == 0.0_wp) then
            close(unit)
            status = TAN_ERR_FORMAT
            return
         end if
         chord = [bx - ax, by - ay, bz - az]
         length = hypot3(chord(1), chord(2), chord(3))
         if (length == 0.0_wp) then
            close(unit)
            status = TAN_ERR_DOMAIN
            return
         end if
         t = chord / length
         nxt = cross3(pl(ip)%nhat, t)
         ed(i)%iplane = ip
         ed(i)%src0 = i - 1
         ed(i)%a = [ax, ay, az]
         ed(i)%b = [bx, by, bz]
         ed(i)%chord = chord
         ed(i)%w = w
         ed(i)%length = length
         ed(i)%t = t
         ed(i)%nxt = nxt
         ed(i)%a_dp = real(ed(i)%a, dp)
         ed(i)%b_dp = real(ed(i)%b, dp)
         ed(i)%chord_dp = real(chord, dp)
         ed(i)%w_dp = real(w, dp)
         ed(i)%length_dp = real(length, dp)
         if (ed(i)%length_dp == 0.0_dp) then
            close(unit)
            status = TAN_ERR_DOMAIN
            return
         end if
         ed(i)%t_dp = ed(i)%chord_dp / ed(i)%length_dp
         ed(i)%nxt_dp = cross3_dp(pl(ip)%nhat_dp, ed(i)%t_dp)
      end do
      close(unit)

      allocate(count(max(np_in, 1)), start(max(np_in, 1)), cursor(max(np_in, 1)), stat=ios)
      if (ios /= 0) then
         status = TAN_ERR_IO
         return
      end if
      count = 0
      do i = 1, ne_in
         count(ed(i)%iplane) = count(ed(i)%iplane) + 1
      end do
      start(1) = 1
      do i = 1, np_in - 1
         start(i + 1) = start(i) + count(i)
      end do
      cursor = start
      allocate(grouped(max(ne_in, 1)), stat=ios)
      if (ios /= 0) then
         status = TAN_ERR_IO
         return
      end if
      do i = 1, ne_in
         ip = ed(i)%iplane
         k = cursor(ip)
         grouped(k) = ed(i)
         cursor(ip) = k + 1
      end do

      allocate(e0(max(np_in, 1)), e1(max(np_in, 1)), stat=ios)
      if (ios /= 0) then
         call tan_clear()
         status = TAN_ERR_IO
         return
      end if
      do i = 1, np_in
         if (count(i) == 0) then
            e0(i) = 1
            e1(i) = 0
         else
            e0(i) = start(i)
            e1(i) = start(i) + count(i) - 1
         end if
      end do

      cache_force_quad = .false.
      cache_ell = 0.0_dp
      do i = 1, np_in
         ell = 0.0_dp
         maxw = 0.0_dp
         area_raw = 0.0_dp
         scale = max(abs(pl(i)%a_dp(1)), abs(pl(i)%a_dp(2)), abs(pl(i)%a_dp(3)))
         xmin = pl(i)%a_dp
         xmax = pl(i)%a_dp
         do k = e0(i), e1(i)
            ell = max(ell, grouped(k)%length_dp)
            maxw = max(maxw, abs(grouped(k)%w_dp))
            cr = cross3_dp(grouped(k)%a_dp, grouped(k)%b_dp)
            area_raw = area_raw + grouped(k)%w_dp * dot3_dp(pl(i)%nraw_dp, cr)
            scale = max(scale, abs(grouped(k)%a_dp(1)), abs(grouped(k)%a_dp(2)), &
                 abs(grouped(k)%a_dp(3)), abs(grouped(k)%b_dp(1)), &
                 abs(grouped(k)%b_dp(2)), abs(grouped(k)%b_dp(3)))
            xmin = min(xmin, grouped(k)%a_dp)
            xmin = min(xmin, grouped(k)%b_dp)
            xmax = max(xmax, grouped(k)%a_dp)
            xmax = max(xmax, grouped(k)%b_dp)
         end do
         extent = hypot3_dp(xmax(1) - xmin(1), xmax(2) - xmin(2), xmax(3) - xmin(3))
         pl(i)%ell = ell
         pl(i)%extent = extent
         pl(i)%scale = scale
         pl(i)%max_abs_w = maxw
         if (pl(i)%nnorm_dp > 0.0_dp) then
            pl(i)%area = 0.5_dp * area_raw / pl(i)%nnorm_dp
         else
            pl(i)%area = 0.0_dp
         end if
         pl(i)%force_quad = .false.
         if (count(i) > 0) then
            if (ell <= 0.0_dp .or. .not. ieee_is_finite(ell) .or. &
                 .not. ieee_is_finite(extent) .or. .not. ieee_is_finite(scale) .or. &
                 .not. ieee_is_finite(pl(i)%area)) then
               pl(i)%force_quad = .true.
            elseif (ell > EXTENT_HUGE .or. ell < EXTENT_TINY .or. &
                 extent > EXTENT_HUGE .or. (extent > 0.0_dp .and. extent < EXTENT_TINY) &
                 .or. scale > EXTENT_HUGE) then
               pl(i)%force_quad = .true.
            elseif (extent > 0.0_dp .and. scale / extent > SCALE_RATIO) then
               pl(i)%force_quad = .true.
            else
               thin = abs(pl(i)%area) / max(maxw, tiny(1.0_dp)) / (ell * ell)
               if (thin < THIN_RATIO) pl(i)%force_quad = .true.
            end if
         end if
         if (pl(i)%force_quad) cache_force_quad = .true.
         cache_ell = max(cache_ell, ell)
      end do

      npl = np_in
      ned = ne_in
      call move_alloc(pl, planes)
      call move_alloc(grouped, edges)

      allocate(character(len=len_trim(path)+1) :: cpath, stat=ios)
      if (ios /= 0) then
         call tan_clear()
         status = TAN_ERR_IO
         return
      end if
      cpath(:) = trim(path) // c_null_char
      cst = tan_exact_load_c_impl(cpath)
      if (cst /= 0_c_int) then
         call tan_clear()
         if (cst == 1_c_int) then
            status = TAN_ERR_IO
         elseif (cst == 3_c_int) then
            status = TAN_ERR_FORMAT
         else
            status = TAN_ERR_FORMAT
         end if
         return
      end if

      have_cache = .true.
      status = TAN_OK
   end subroutine tan_load

   subroutine tan_eval(point, mode, order, radius, coeff, status)
      real(dp), intent(in) :: point(3)
      integer, intent(in) :: mode, order
      real(dp), intent(in) :: radius
      real(dp), intent(out) :: coeff(3, TAN_NM)
      integer, intent(out) :: status
      real(wp) :: p(3), rad
      integer :: n, i
      logical :: cancel, need_quad
      real(dp) :: sumabs

      call tan_jet_init()
      coeff = 0.0_dp
      status = TAN_OK
      last_mode_used = -1

      if (.not. have_cache) then
         status = TAN_ERR_NO_CACHE
         return
      end if
      if (mode < 0 .or. mode > 2) then
         status = TAN_ERR_MODE
         return
      end if
      if (order < 0 .or. order > TAN_QMAX) then
         status = TAN_ERR_ORDER
         return
      end if
      if ((mode == 0 .or. mode == 2) .and. order /= 0) then
         status = TAN_ERR_MODE
         return
      end if
      if (.not. (ieee_is_finite(point(1)) .and. ieee_is_finite(point(2)) &
           .and. ieee_is_finite(point(3)))) then
         status = TAN_ERR_POINT
         return
      end if
      if (.not. ieee_is_finite(radius) .or. radius <= 0.0_dp) then
         status = TAN_ERR_RADIUS
         return
      end if

      p = [real(point(1), wp), real(point(2), wp), real(point(3), wp)]
      rad = real(radius, wp)

      if (order == 0) then
         if (mode == 0) then
            call eval_scalar_dp(point, coeff(:, 1), status, cancel, sumabs)
            last_mode_used = 0
         elseif (mode == 1) then
            call eval_scalar_wp(point, p, coeff(:, 1), status)
            last_mode_used = 1
         else
            need_quad = cache_force_quad
            if (.not. need_quad) call query_needs_quad(point, need_quad)
            if (need_quad) then
               call eval_scalar_wp(point, p, coeff(:, 1), status)
               last_mode_used = 1
            else
               call eval_scalar_dp(point, coeff(:, 1), status, cancel, sumabs)
               last_mode_used = 0
               if (status == TAN_ERR_NONFINITE .or. status == TAN_ERR_DOMAIN &
                    .or. cancel) then
                  coeff(:, 1) = 0.0_dp
                  call eval_scalar_wp(point, p, coeff(:, 1), status)
                  last_mode_used = 1
               end if
            end if
         end if
      else
         call eval_jet(point, p, order, rad, coeff, status)
         last_mode_used = 1
      end if

      if (status /= TAN_OK) then
         coeff = 0.0_dp
         return
      end if
      n = nmono_of(order)
      do i = 1, n
         if (.not. (ieee_is_finite(coeff(1, i)) .and. ieee_is_finite(coeff(2, i)) &
              .and. ieee_is_finite(coeff(3, i)))) then
            coeff = 0.0_dp
            status = TAN_ERR_NONFINITE
            return
         end if
      end do
   end subroutine tan_eval

   subroutine query_needs_quad(p, need)
      real(dp), intent(in) :: p(3)
      logical, intent(out) :: need
      integer :: ipl, st, ie
      real(dp) :: h, s, rmin, d1, d2, ell
      logical :: on_plane

      need = .false.
      if (cache_ell <= 0.0_dp) then
         need = .true.
         return
      end if
      rmin = huge(rmin)
      do ipl = 1, npl
         call classify_plane_dp(p, ipl, h, s, on_plane, st)
         if (st /= TAN_OK) then
            need = .true.
            return
         end if
         ell = planes(ipl)%ell
         if (ell <= 0.0_dp) cycle
         if (abs(h) / ell < CLOSE_RATIO) then
            need = .true.
            return
         end if
         do ie = e0(ipl), e1(ipl)
            d1 = hypot3_dp(edges(ie)%a_dp(1) - p(1), edges(ie)%a_dp(2) - p(2), &
                 edges(ie)%a_dp(3) - p(3))
            d2 = hypot3_dp(edges(ie)%b_dp(1) - p(1), edges(ie)%b_dp(2) - p(2), &
                 edges(ie)%b_dp(3) - p(3))
            rmin = min(rmin, d1, d2)
         end do
      end do
      if (rmin > FAR_RATIO * cache_ell) need = .true.
   end subroutine query_needs_quad

   ! Ordinary binary64 signs use a forward-error filter. Only ambiguous
   ! heights enter the real128/exact predicate path. This changes the cost of
   ! classification, never the rule that a tolerance cannot decide incidence.
   subroutine classify_plane_dp(p, ipl, h, s, on_plane, status)
      real(dp), intent(in) :: p(3)
      integer, intent(in) :: ipl
      real(dp), intent(out) :: h, s
      logical, intent(out) :: on_plane
      integer, intent(out) :: status
      real(dp) :: dx(3), products(3), sumabs, bound
      real(wp) :: hq, sq

      dx = p - planes(ipl)%a_dp
      products = dx * planes(ipl)%nhat_dp
      h = sum(products)
      sumabs = sum(abs(products))
      bound = 32.0_dp * (epsilon(h) * sumabs + tiny(1.0_dp))
      if (planes(ipl)%dp_filter_ok .and. ieee_is_finite(h) .and. &
           ieee_is_finite(sumabs) .and. abs(h) > bound) then
         on_plane = .false.
         s = sign(1.0_dp,h)
         status = TAN_OK
      else
         hq = 0.0_wp
         sq = 1.0_wp
         call classify_plane(p,ipl,hq,sq,on_plane,status)
         h = real(hq,dp)
         s = real(sq,dp)
      end if
   end subroutine classify_plane_dp

   subroutine classify_plane(p_dp, ipl, h, s, on_plane, status)
      real(dp), intent(in) :: p_dp(3)
      integer, intent(in) :: ipl
      real(wp), intent(out) :: h, s
      logical, intent(out) :: on_plane
      integer, intent(out) :: status
      real(wp) :: dx(3), raw, sumabs, lim, raw_exact
      integer :: sgn

      status = TAN_OK
      on_plane = .false.
      dx = [real(p_dp(1), wp), real(p_dp(2), wp), real(p_dp(3), wp)] - planes(ipl)%a
      raw = planes(ipl)%nraw(1) * dx(1) + planes(ipl)%nraw(2) * dx(2) + &
           planes(ipl)%nraw(3) * dx(3)
      sumabs = abs(planes(ipl)%nraw(1)) * abs(dx(1)) + &
           abs(planes(ipl)%nraw(2)) * abs(dx(2)) + &
           abs(planes(ipl)%nraw(3)) * abs(dx(3))
      lim = 32.0_wp * epsilon(raw) * sumabs
      if ((.not. tan_finite(raw)) .or. abs(raw) <= lim) then
         call exact_plane(p_dp, ipl, sgn, raw_exact, status)
         if (status /= TAN_OK) return
         if (sgn == 0) then
            on_plane = .true.
            h = 0.0_wp
            s = 1.0_wp
         else
            h = raw_exact / planes(ipl)%nnorm
            s = real(sgn, wp)
         end if
      else
         h = raw / planes(ipl)%nnorm
         s = sign(1.0_wp, raw)
         if (s == 0.0_wp) s = 1.0_wp
      end if
      if (.not. tan_finite(h)) status = TAN_ERR_DOMAIN
   end subroutine classify_plane

   subroutine exact_plane(p_dp, ipl, sgn, raw, status)
      real(dp), intent(in) :: p_dp(3)
      integer, intent(in) :: ipl
      integer, intent(out) :: sgn, status
      real(wp), intent(out) :: raw
      character(kind=c_char) :: buf(RAW_CAP)
      character(len=RAW_CAP) :: txt
      real(c_double) :: pq(3)
      integer(c_int) :: res
      integer :: i, n, ios

      status = TAN_OK
      raw = 0.0_wp
      pq = real(p_dp, c_double)
      buf = c_null_char
      res = tan_exact_plane_c_impl(int(ipl - 1, c_int), pq, buf, int(RAW_CAP, c_int))
      if (res < -1_c_int .or. res > 1_c_int) then
         status = TAN_ERR_DOMAIN
         return
      end if
      sgn = int(res)
      txt = ' '
      n = 0
      do i = 1, RAW_CAP
         if (buf(i) == c_null_char) exit
         n = n + 1
         txt(n:n) = buf(i)
      end do
      if (n <= 0) then
         status = TAN_ERR_DOMAIN
         return
      end if
      read(txt(1:n), *, iostat=ios) raw
      if (ios /= 0 .or. .not. tan_finite(raw)) then
         status = TAN_ERR_DOMAIN
         return
      end if
      if (sgn == 0) raw = 0.0_wp
   end subroutine exact_plane

   subroutine exact_edge(p_dp, ie, hit, status)
      real(dp), intent(in) :: p_dp(3)
      integer, intent(in) :: ie
      logical, intent(out) :: hit
      integer, intent(out) :: status
      real(c_double) :: pq(3)
      integer(c_int) :: res

      status = TAN_OK
      hit = .false.
      pq = real(p_dp, c_double)
      res = tan_exact_edge_c_impl(int(edges(ie)%src0, c_int), pq)
      if (res == 1_c_int) then
         hit = .true.
      elseif (res == 0_c_int) then
         hit = .false.
      else
         status = TAN_ERR_DOMAIN
      end if
   end subroutine exact_edge

   subroutine exact_support(p_dp, ipl, code, status)
      real(dp), intent(in) :: p_dp(3)
      integer, intent(in) :: ipl
      integer, intent(out) :: code, status
      real(c_double) :: pq(3)
      integer(c_int) :: res

      status = TAN_OK
      pq = real(p_dp, c_double)
      res = tan_exact_support_c_impl(int(ipl - 1, c_int), pq)
      if (res < 0_c_int .or. res > 2_c_int) then
         status = TAN_ERR_DOMAIN
         code = -1
         return
      end if
      code = int(res)
   end subroutine exact_support

   subroutine eval_scalar_dp(p, b, status, cancel, termabs)
      real(dp), intent(in) :: p(3)
      real(dp), intent(out) :: b(3)
      integer, intent(out) :: status
      logical, intent(out) :: cancel
      real(dp), intent(out) :: termabs
      real(dp) :: acc(3), comp(3), contrib(3), hdp, sdp, outn
      integer :: ipl, ie
      logical :: hit, on_plane

      status = TAN_OK
      cancel = .false.
      termabs = 0.0_dp
      acc = 0.0_dp
      comp = 0.0_dp
      do ipl = 1, npl
         call classify_plane_dp(p, ipl, hdp, sdp, on_plane, status)
         if (status /= TAN_OK) return
         do ie = e0(ipl), e1(ipl)
            call edge_scalar_dp(p, ipl, ie, sdp, hdp, on_plane, contrib, hit, status)
            if (status /= TAN_OK) return
            if (hit) then
               status = TAN_ERR_EDGE
               b = 0.0_dp
               return
            end if
            termabs = termabs + abs(contrib(1)) + abs(contrib(2)) + abs(contrib(3))
            call kahan3(acc, comp, contrib)
         end do
      end do
      if (.not. (ieee_is_finite(acc(1)) .and. ieee_is_finite(acc(2)) &
           .and. ieee_is_finite(acc(3)))) then
         status = TAN_ERR_NONFINITE
         b = 0.0_dp
         return
      end if
      outn = abs(acc(1)) + abs(acc(2)) + abs(acc(3))
      if (termabs > CANCEL_LIMIT * max(outn, tiny(1.0_dp))) cancel = .true.
      b = acc
   end subroutine eval_scalar_dp

   subroutine edge_scalar_dp(p, ipl, ie, s, h, on_plane, contrib, hit, status)
      real(dp), intent(in) :: p(3), s, h
      integer, intent(in) :: ipl, ie
      logical, intent(in) :: on_plane
      real(dp), intent(out) :: contrib(3)
      logical, intent(out) :: hit
      integer, intent(out) :: status
      real(dp) :: ra(3), rb(3), rhoa, rhob, u, ww, rho, rho2, gap
      real(dp) :: baseline, correction, L, omega, den_dot, den
      real(dp) :: cr(3), offset(3), fac, nh(3), t(3), nxt(3), length, w
      real(dp) :: crn, scale, lim, chord(3)
      logical :: uncertain

      status = TAN_OK
      hit = .false.
      contrib = 0.0_dp
      nh = planes(ipl)%nhat_dp
      t = edges(ie)%t_dp
      nxt = edges(ie)%nxt_dp
      length = edges(ie)%length_dp
      w = edges(ie)%w_dp
      chord = edges(ie)%chord_dp
      ra = edges(ie)%a_dp - p
      rb = edges(ie)%b_dp - p
      rhoa = hypot3_dp(ra(1), ra(2), ra(3))
      rhob = hypot3_dp(rb(1), rb(2), rb(3))
      ! The established 1e-3 gap chart avoids unnecessary near-edge work.
      ! A true segment/vertex cannot pass this well-separated-gap filter.
      gap = rhoa + rhob - length
      if (gap > 1.0e-3_dp*(rhoa+rhob) .and. ieee_is_finite(gap) .and. &
           rhoa > 0.0_dp .and. rhob > 0.0_dp) then
         continue
      else
         u = -dot3_dp(ra, t)
         ww = dot3_dp(rb, t)
         if (abs(u) <= abs(ww)) then
            offset = -ra
         else
            offset = -rb
         end if
         cr = cross3_dp(offset, t)
         rho2 = dot3_dp(cr, cr)
         rho = sqrt(max(rho2, 0.0_dp))
         scale = abs(offset(1)) * (abs(t(2)) + abs(t(3))) + &
              abs(offset(2)) * (abs(t(3)) + abs(t(1))) + &
              abs(offset(3)) * (abs(t(1)) + abs(t(2)))
         lim = 32.0_dp * epsilon(rho) * scale
         uncertain = (rho <= lim) .or. (rhoa == 0.0_dp) .or. (rhob == 0.0_dp) &
              .or. (.not. ieee_is_finite(rho))
         if (uncertain) then
            call exact_edge(p, ie, hit, status)
            if (status /= TAN_OK) return
            if (hit) return
            if (rhoa == 0.0_dp .or. rhob == 0.0_dp .or. .not. ieee_is_finite(rhoa) &
                 .or. .not. ieee_is_finite(rhob)) then
               status = TAN_ERR_DOMAIN
               return
            end if
         end if
         baseline = 2.0_dp * (max(-u, 0.0_dp) + max(-ww, 0.0_dp))
         correction = rho * (rho / (rhoa + abs(u)) + rho / (rhob + abs(ww)))
         gap = baseline + correction
         if (gap <= 0.0_dp) then
            if (rho == 0.0_dp) then
               call exact_edge(p, ie, hit, status)
               if (status /= TAN_OK) return
               if (hit) return
               status = TAN_ERR_DOMAIN
               return
            end if
            gap = rho2 * (1.0_dp / (rhoa + abs(u)) + 1.0_dp / (rhob + abs(ww)))
         end if
         if (gap <= 0.0_dp .or. .not. ieee_is_finite(gap)) then
            status = TAN_ERR_DOMAIN
            return
         end if
      end if
      L = log(1.0_dp + 2.0_dp * length / gap)

      cr = cross3_dp(ra, chord)
      crn = dot3_dp(nh, cr)
      den_dot = rhoa * rhob + dot3_dp(ra, rb)
      if (dot3_dp(ra, rb) < 0.0_dp) then
         den_dot = dot3_dp(cr, cr) / (rhoa * rhob - dot3_dp(ra, rb))
      end if
      den = den_dot + s * h * (rhoa + rhob)
      if (on_plane) then
         omega = 0.0_dp
      else
         if (crn == 0.0_dp .and. den == 0.0_dp) then
            status = TAN_ERR_DOMAIN
            return
         end if
         omega = -2.0_dp * s * atan2(crn, den)
      end if
      fac = -w / FOUR_PI_DP
      contrib = fac * (L * nxt + omega * nh)
      if (.not. (ieee_is_finite(contrib(1)) .and. ieee_is_finite(contrib(2)) &
           .and. ieee_is_finite(contrib(3)))) then
         status = TAN_ERR_NONFINITE
      end if
   end subroutine edge_scalar_dp

   subroutine eval_scalar_wp(p_dp, p, b, status)
      real(dp), intent(in) :: p_dp(3)
      real(wp), intent(in) :: p(3)
      real(dp), intent(out) :: b(3)
      integer, intent(out) :: status
      real(wp) :: acc(3), h, s, contrib(3)
      integer :: ipl, ie
      logical :: hit, on_plane

      status = TAN_OK
      acc = 0.0_wp
      do ipl = 1, npl
         call classify_plane(p_dp, ipl, h, s, on_plane, status)
         if (status /= TAN_OK) return
         do ie = e0(ipl), e1(ipl)
            call edge_scalar_wp(p_dp, p, ipl, ie, s, h, on_plane, contrib, hit, status)
            if (status /= TAN_OK) return
            if (hit) then
               status = TAN_ERR_EDGE
               b = 0.0_dp
               return
            end if
            acc = acc + contrib
         end do
      end do
      if (.not. (tan_finite(acc(1)) .and. tan_finite(acc(2)) .and. tan_finite(acc(3)))) then
         status = TAN_ERR_NONFINITE
         b = 0.0_dp
         return
      end if
      b = real(acc, dp)
   end subroutine eval_scalar_wp

   subroutine edge_scalar_wp(p_dp, p, ipl, ie, s, h, on_plane, contrib, hit, status)
      real(dp), intent(in) :: p_dp(3)
      real(wp), intent(in) :: p(3), s, h
      integer, intent(in) :: ipl, ie
      logical, intent(in) :: on_plane
      real(wp), intent(out) :: contrib(3)
      logical, intent(out) :: hit
      integer, intent(out) :: status
      real(wp) :: ra(3), rb(3), rhoa, rhob, u, ww, rho, rho2, gap
      real(wp) :: baseline, correction, L, omega, den_dot, den
      real(wp) :: cr(3), offset(3), fac, nh(3), t(3), nxt(3), length, w
      real(wp) :: crn, scale, lim, chord(3)
      logical :: uncertain

      status = TAN_OK
      hit = .false.
      contrib = 0.0_wp
      nh = planes(ipl)%nhat
      t = edges(ie)%t
      nxt = edges(ie)%nxt
      length = edges(ie)%length
      w = edges(ie)%w
      chord = edges(ie)%chord
      ra = edges(ie)%a - p
      rb = edges(ie)%b - p
      rhoa = hypot3(ra(1), ra(2), ra(3))
      rhob = hypot3(rb(1), rb(2), rb(3))
      u = -dot3(ra, t)
      ww = dot3(rb, t)
      if (abs(u) <= abs(ww)) then
         offset = -ra
      else
         offset = -rb
      end if
      cr = cross3(offset, t)
      rho2 = dot3(cr, cr)
      rho = sqrt(max(rho2, 0.0_wp))
      scale = abs(offset(1)) * (abs(t(2)) + abs(t(3))) + &
           abs(offset(2)) * (abs(t(3)) + abs(t(1))) + &
           abs(offset(3)) * (abs(t(1)) + abs(t(2)))
      lim = 32.0_wp * epsilon(rho) * scale
      uncertain = (rho <= lim) .or. (rhoa == 0.0_wp) .or. (rhob == 0.0_wp) &
           .or. (.not. tan_finite(rho))
      if (uncertain) then
         call exact_edge(p_dp, ie, hit, status)
         if (status /= TAN_OK) return
         if (hit) return
         if (rhoa == 0.0_wp .or. rhob == 0.0_wp .or. .not. tan_finite(rhoa) &
              .or. .not. tan_finite(rhob)) then
            status = TAN_ERR_DOMAIN
            return
         end if
      end if
      baseline = 2.0_wp * (max(-u, 0.0_wp) + max(-ww, 0.0_wp))
      correction = rho * (rho / (rhoa + abs(u)) + rho / (rhob + abs(ww)))
      gap = baseline + correction
      if (gap <= 0.0_wp) then
         if (rho == 0.0_wp) then
            call exact_edge(p_dp, ie, hit, status)
            if (status /= TAN_OK) return
            if (hit) return
            status = TAN_ERR_DOMAIN
            return
         end if
         gap = rho2 * (1.0_wp / (rhoa + abs(u)) + 1.0_wp / (rhob + abs(ww)))
      end if
      if (gap <= 0.0_wp .or. .not. tan_finite(gap)) then
         status = TAN_ERR_DOMAIN
         return
      end if
      L = log(1.0_wp + 2.0_wp * length / gap)

      cr = cross3(ra, chord)
      crn = dot3(nh, cr)
      den_dot = rhoa * rhob + dot3(ra, rb)
      if (dot3(ra, rb) < 0.0_wp) then
         den_dot = dot3(cr, cr) / (rhoa * rhob - dot3(ra, rb))
      end if
      den = den_dot + s * h * (rhoa + rhob)
      if (on_plane) then
         omega = 0.0_wp
      else
         if (crn == 0.0_wp .and. den == 0.0_wp) then
            status = TAN_ERR_DOMAIN
            return
         end if
         omega = -2.0_wp * s * atan2(crn, den)
      end if
      fac = -w / four_pi
      contrib = fac * (L * nxt + omega * nh)
      if (.not. (tan_finite(contrib(1)) .and. tan_finite(contrib(2)) &
           .and. tan_finite(contrib(3)))) then
         status = TAN_ERR_NONFINITE
      end if
   end subroutine edge_scalar_wp

   subroutine eval_jet(p_dp, p0, order, radius, coeff, status)
      real(dp), intent(in) :: p_dp(3)
      real(wp), intent(in) :: p0(3), radius
      integer, intent(in) :: order
      real(dp), intent(out) :: coeff(3, TAN_NM)
      integer, intent(out) :: status
      type(tan_jet3) :: p, ra, rb, offset, cr, nhj, nxtj, chordj
      type(tan_jet) :: h, rhoa, rhob, u, ww, rho2, gap, L, omega
      type(tan_jet) :: num, den_dot, den, den_raw, cr2, invgap, tmp1, tmp2
      type(tan_jet) :: bx, by, bz
      type(tan_jet3) :: aconst, bconst
      real(wp) :: h0, s, w, length, fac
      integer :: ipl, ie, k, n, st, su, sw, support
      logical :: on_plane, use_rat, hit
      real(wp) :: u0, w0, rhoa0, rhob0, rho0, scale, lim

      status = TAN_OK
      n = nmono_of(order)
      p = jet3_point(p0, radius, order)
      bx = jet_zero(order)
      by = jet_zero(order)
      bz = jet_zero(order)

      do ipl = 1, npl
         nhj = jet3_const(planes(ipl)%nhat, order)
         h = jet3_dot(jet3_sub(p, jet3_const(planes(ipl)%a, order)), nhj)
         if (.not. jet_is_finite(h)) then
            status = TAN_ERR_NONFINITE
            return
         end if
         call classify_plane(p_dp, ipl, h0, s, on_plane, status)
         if (status /= TAN_OK) return
         h%c(1) = h0
         if (on_plane) then
            call exact_support(p_dp, ipl, support, status)
            if (status /= TAN_OK) return
            if (support == 2) then
               status = TAN_ERR_EDGE
               return
            elseif (support == 1) then
               status = TAN_ERR_WINDING
               return
            elseif (support /= 0) then
               status = TAN_ERR_DOMAIN
               return
            end if
            s = 1.0_wp
         end if

         do ie = e0(ipl), e1(ipl)
            w = edges(ie)%w
            length = edges(ie)%length
            aconst = jet3_const(edges(ie)%a, order)
            bconst = jet3_const(edges(ie)%b, order)
            chordj = jet3_const(edges(ie)%chord, order)
            ra = jet3_sub(aconst, p)
            rb = jet3_sub(bconst, p)
            rhoa = jet3_norm(ra, st)
            if (st /= TAN_OK) then
               status = TAN_ERR_EDGE
               return
            end if
            rhob = jet3_norm(rb, st)
            if (st /= TAN_OK) then
               status = TAN_ERR_EDGE
               return
            end if
            rhoa0 = rhoa%c(1)
            rhob0 = rhob%c(1)
            u = jet_neg(jet3_dot(ra, jet3_const(edges(ie)%t, order)))
            ww = jet3_dot(rb, jet3_const(edges(ie)%t, order))
            u0 = u%c(1)
            w0 = ww%c(1)
            if (abs(u0) <= abs(w0)) then
               offset = jet3_scale(ra, -1.0_wp)
            else
               offset = jet3_scale(rb, -1.0_wp)
            end if
            cr = jet3_cross(offset, jet3_const(edges(ie)%t, order))
            rho2 = jet3_norm2(cr)
            rho0 = sqrt(max(rho2%c(1), 0.0_wp))
            scale = abs(offset%x%c(1)) * (abs(edges(ie)%t(2)) + abs(edges(ie)%t(3))) + &
                 abs(offset%y%c(1)) * (abs(edges(ie)%t(3)) + abs(edges(ie)%t(1))) + &
                 abs(offset%z%c(1)) * (abs(edges(ie)%t(1)) + abs(edges(ie)%t(2)))
            lim = 32.0_wp * epsilon(rho0) * scale
            if ((rho0 <= lim) .or. (rhoa0 == 0.0_wp) .or. (rhob0 == 0.0_wp) &
                 .or. (.not. tan_finite(rho0))) then
               call exact_edge(p_dp, ie, hit, status)
               if (status /= TAN_OK) return
               if (hit) then
                  status = TAN_ERR_EDGE
                  return
               end if
               if (rhoa0 == 0.0_wp .or. rhob0 == 0.0_wp) then
                  status = TAN_ERR_DOMAIN
                  return
               end if
            end if

            su = 1
            sw = 1
            if (u0 < 0.0_wp) su = -1
            if (w0 < 0.0_wp) sw = -1
            tmp1 = jet_add(rhoa, jet_scale(u, real(su, wp)))
            tmp2 = jet_add(rhob, jet_scale(ww, real(sw, wp)))
            if (tmp1%c(1) == 0.0_wp .or. tmp2%c(1) == 0.0_wp) then
               status = TAN_ERR_EDGE
               return
            end if
            tmp1 = jet_inv(tmp1, st)
            if (st /= TAN_OK) then
               status = st
               return
            end if
            tmp2 = jet_inv(tmp2, st)
            if (st /= TAN_OK) then
               status = st
               return
            end if
            gap = jet_add(jet_scale(u, real(su - 1, wp)), &
                          jet_scale(ww, real(sw - 1, wp)))
            gap = jet_add(gap, jet_mul(rho2, jet_add(tmp1, tmp2)))
            if (gap%c(1) <= 0.0_wp) then
               status = TAN_ERR_DOMAIN
               return
            end if
            invgap = jet_inv(gap, st)
            if (st /= TAN_OK) then
               status = st
               return
            end if
            L = jet_log(jet_add(jet_const(1.0_wp, order), &
                                jet_scale(invgap, 2.0_wp * length)), st)
            if (st /= TAN_OK) then
               status = st
               return
            end if

            cr = jet3_cross(ra, chordj)
            num = jet3_dot(nhj, cr)
            tmp1 = jet3_dot(ra, rb)
            den_raw = jet_add(jet_mul(rhoa, rhob), tmp1)
            use_rat = (tmp1%c(1) < 0.0_wp)
            if (use_rat) then
               cr2 = jet3_norm2(cr)
               den_dot = jet_sub(jet_mul(rhoa, rhob), tmp1)
               if (den_dot%c(1) == 0.0_wp) then
                  status = TAN_ERR_DOMAIN
                  return
               end if
               den_dot = jet_mul(cr2, jet_inv(den_dot, st))
               if (st /= TAN_OK) then
                  status = st
                  return
               end if
            else
               den_dot = den_raw
            end if
            den = jet_add(den_dot, jet_scale(jet_mul(h, jet_add(rhoa, rhob)), s))
            if (num%c(1) == 0.0_wp .and. den%c(1) == 0.0_wp) then
               status = TAN_ERR_DOMAIN
               return
            end if
            call jet_atan2(num, den, omega, st)
            if (st /= TAN_OK) then
               status = st
               return
            end if
            omega = jet_scale(omega, -2.0_wp * s)

            fac = -w / four_pi
            nxtj = jet3_const(edges(ie)%nxt, order)
            call jet_axpy(fac, jet_add(jet_mul(L, nxtj%x), jet_mul(omega, nhj%x)), bx)
            call jet_axpy(fac, jet_add(jet_mul(L, nxtj%y), jet_mul(omega, nhj%y)), by)
            call jet_axpy(fac, jet_add(jet_mul(L, nxtj%z), jet_mul(omega, nhj%z)), bz)
         end do
      end do

      if (.not. (jet_is_finite(bx) .and. jet_is_finite(by) .and. jet_is_finite(bz))) then
         status = TAN_ERR_NONFINITE
         return
      end if
      coeff = 0.0_dp
      do k = 1, n
         coeff(1, k) = real(bx%c(k), dp)
         coeff(2, k) = real(by%c(k), dp)
         coeff(3, k) = real(bz%c(k), dp)
      end do
   end subroutine eval_jet

   subroutine tan_load_c(path, status) bind(C, name='tan_load_c')
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), intent(out) :: status
      character(len=4096) :: fpath
      integer :: i, st
      fpath = ' '
      do i = 1, len(fpath)
         if (path(i) == c_null_char) exit
         fpath(i:i) = path(i)
      end do
      if (i > len(fpath)) then
         call tan_clear()
         status = TAN_ERR_IO
         return
      end if
      call tan_load(trim(fpath), st)
      status = int(st, c_int)
   end subroutine tan_load_c

   subroutine tan_eval_c(point, mode, order, radius, coeff, status) &
        bind(C, name='tan_eval_c')
      real(c_double), intent(in) :: point(3)
      integer(c_int), value :: mode, order
      real(c_double), value :: radius
      real(c_double), intent(out) :: coeff(3, TAN_NM)
      integer(c_int), intent(out) :: status
      integer :: st
      call tan_eval(point, int(mode), int(order), radius, coeff, st)
      status = int(st, c_int)
   end subroutine tan_eval_c

   subroutine tan_batch_c(count, points, mode, fields, statuses) &
        bind(C, name='tan_batch_c')
      integer(c_int), value :: count, mode
      real(c_double), intent(in) :: points(3, *)
      real(c_double), intent(out) :: fields(3, *)
      integer(c_int), intent(out) :: statuses(*)
      real(dp) :: coeff(3, TAN_NM), pt(3)
      integer :: i, st
      do i = 1, int(count)
         pt = points(:, i)
         call tan_eval(pt, int(mode), 0, 1.0_dp, coeff, st)
         fields(:, i) = coeff(:, 1)
         statuses(i) = int(st, c_int)
      end do
   end subroutine tan_batch_c

   subroutine read_nonempty(unit, line, ios)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      integer, intent(out) :: ios
      integer :: n
      do
         read(unit, '(a)', iostat=ios) line
         if (ios /= 0) return
         n = len_trim(line)
         if (n > 0) then
            if (line(n:n) == achar(13)) then
               line(n:n) = ' '
               n = len_trim(line)
            end if
         end if
         if (n == 0) cycle
         line = adjustl(line)
         if (line(1:1) == '#') cycle
         return
      end do
   end subroutine read_nonempty

   logical function all_finite6(a, b, c, d, e, f)
      real(wp), intent(in) :: a, b, c, d, e, f
      all_finite6 = tan_finite(a) .and. tan_finite(b) .and. tan_finite(c) &
           .and. tan_finite(d) .and. tan_finite(e) .and. tan_finite(f)
   end function all_finite6

   logical function all_finite7(a, b, c, d, e, f, g)
      real(wp), intent(in) :: a, b, c, d, e, f, g
      all_finite7 = all_finite6(a, b, c, d, e, f) .and. tan_finite(g)
   end function all_finite7

   subroutine kahan3(acc, comp, v)
      real(dp), intent(inout) :: acc(3), comp(3)
      real(dp), intent(in) :: v(3)
      real(dp) :: y, t
      integer :: k
      do k = 1, 3
         y = v(k) - comp(k)
         t = acc(k) + y
         comp(k) = (t - acc(k)) - y
         acc(k) = t
      end do
   end subroutine kahan3

   real(wp) function hypot3(x, y, z)
      real(wp), intent(in) :: x, y, z
      real(wp) :: ax, ay, az, m, rx, ry, rz
      ax = abs(x); ay = abs(y); az = abs(z)
      m = ax
      if (ay > m) m = ay
      if (az > m) m = az
      if (m == 0.0_wp) then
         hypot3 = 0.0_wp
         return
      end if
      rx = x / m; ry = y / m; rz = z / m
      hypot3 = m * sqrt(rx * rx + ry * ry + rz * rz)
   end function hypot3

   real(dp) function hypot3_dp(x, y, z)
      real(dp), intent(in) :: x, y, z
      real(dp) :: ax, ay, az, m, rx, ry, rz
      ax = abs(x); ay = abs(y); az = abs(z)
      m = ax
      if (ay > m) m = ay
      if (az > m) m = az
      if (m == 0.0_dp) then
         hypot3_dp = 0.0_dp
         return
      end if
      rx = x / m; ry = y / m; rz = z / m
      hypot3_dp = m * sqrt(rx * rx + ry * ry + rz * rz)
   end function hypot3_dp

   real(wp) function dot3(a, b)
      real(wp), intent(in) :: a(3), b(3)
      dot3 = a(1) * b(1) + a(2) * b(2) + a(3) * b(3)
   end function dot3

   real(dp) function dot3_dp(a, b)
      real(dp), intent(in) :: a(3), b(3)
      dot3_dp = a(1) * b(1) + a(2) * b(2) + a(3) * b(3)
   end function dot3_dp

   function cross3(a, b) result(c)
      real(wp), intent(in) :: a(3), b(3)
      real(wp) :: c(3)
      c(1) = a(2) * b(3) - a(3) * b(2)
      c(2) = a(3) * b(1) - a(1) * b(3)
      c(3) = a(1) * b(2) - a(2) * b(1)
   end function cross3

   function cross3_dp(a, b) result(c)
      real(dp), intent(in) :: a(3), b(3)
      real(dp) :: c(3)
      c(1) = a(2) * b(3) - a(3) * b(2)
      c(2) = a(3) * b(1) - a(1) * b(3)
      c(3) = a(1) * b(2) - a(2) * b(1)
   end function cross3_dp

end module tan_mod
