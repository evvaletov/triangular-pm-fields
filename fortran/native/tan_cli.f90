! tan_cli.f90 — standalone evaluator for a TAN_CACHE_V1 file.
! Usage:
!   tan_cli CACHE X Y Z [MODE] [ORDER] [RADIUS]
! Prints status, then one line per monomial: i ex ey ez Bx By Bz
program tan_cli
   use, intrinsic :: iso_fortran_env, only: dp => real64, output_unit, error_unit
   use tan_mod
   implicit none
   character(len=4096) :: cache, arg
   integer :: narg, mode, order, status, n, i, ios
   integer :: ex(TAN_NM), ey(TAN_NM), ez(TAN_NM)
   real(dp) :: point(3), radius, coeff(3, TAN_NM)

   narg = command_argument_count()
   if (narg < 4) then
      write(error_unit, '(a)') &
         'usage: tan_cli CACHE X Y Z [MODE] [ORDER] [RADIUS]'
      stop 2
   end if
   call get_command_argument(1, cache)
   call get_command_argument(2, arg); read(arg, *, iostat=ios) point(1)
   if (ios /= 0) stop 2
   call get_command_argument(3, arg); read(arg, *, iostat=ios) point(2)
   if (ios /= 0) stop 2
   call get_command_argument(4, arg); read(arg, *, iostat=ios) point(3)
   if (ios /= 0) stop 2
   mode = 1
   order = 0
   radius = 1.0_dp
   if (narg >= 5) then
      call get_command_argument(5, arg); read(arg, *, iostat=ios) mode
      if (ios /= 0) stop 2
   end if
   if (narg >= 6) then
      call get_command_argument(6, arg); read(arg, *, iostat=ios) order
      if (ios /= 0) stop 2
   end if
   if (narg >= 7) then
      call get_command_argument(7, arg); read(arg, *, iostat=ios) radius
      if (ios /= 0) stop 2
   end if

   call tan_load(trim(cache), status)
   if (status /= TAN_OK) then
      write(output_unit, '(a,1x,i0)') 'status', status
      stop 1
   end if
   call tan_eval(point, mode, order, radius, coeff, status)
   write(output_unit, '(a,1x,i0)') 'status', status
   if (status /= TAN_OK) stop 1
   call tan_exponents(order, ex, ey, ez, n)
   write(output_unit, '(a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,es24.16e3)') &
      'nmono', n, 'mode', mode, 'order', order, 'radius', radius
   write(output_unit, '(a)') '# i ex ey ez Bx By Bz'
   do i = 1, n
      write(output_unit, '(4(1x,i4),3(1x,es24.16e3))') &
         i, ex(i), ey(i), ez(i), coeff(1, i), coeff(2, i), coeff(3, i)
   end do
end program tan_cli
