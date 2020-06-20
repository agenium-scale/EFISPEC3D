
module cifo4

  implicit none

  integer :: IG_NGLL = 5

  integer, allocatable, dimension(:,:,:,:) :: ig_hexa_gll_glonum

  real, allocatable, dimension(:,:) :: rg_gll_displacement
  real, allocatable, dimension(:,:) :: rg_gll_lagrange_deriv

  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dxidx
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dxidy
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dxidz

  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_detdx
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_detdy
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_detdz

  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dzedx
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dzedy
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_dzedz

  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_rhovp2
  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_rhovs2

  real, allocatable, dimension(:,:,:,:) :: rg_hexa_gll_jacobian_det

  real, allocatable, dimension(:) :: rg_gll_weight

  real, allocatable, dimension(:,:) :: rg_gll_acceleration
  real, allocatable, dimension(:,:) :: rg_gll_acceleration_ref

contains

  subroutine compute_internal_forces_order4( elt_start, elt_end )

    integer, intent(in) :: elt_start
    integer, intent(in) :: elt_end

    integer iel,igll, k, l , m
    real duxdxi, duydxi, duzdxi
    real duxdet, duydet, duzdet
    real duxdze, duydze, duzdze

    real duxdx, duxdy, duxdz
    real duydx, duydy, duydz
    real duzdx, duzdy, duzdz

    real dxidx, dxidy, dxidz
    real detdx, detdy, detdz
    real dzedx, dzedy, dzedz

    real trace_tau
    real tauxx, tauxy, tauxz
    real tauyy, tauyz
    real tauzz

    real tmpx1, tmpy1, tmpz1
    real tmpx2, tmpy2, tmpz2
    real tmpx3, tmpy3, tmpz3

    real fac1, fac2, fac3

    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpx1
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpx2
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpx3
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpy1
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpy2
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpy3
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpz1
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpz2
    real, dimension(IG_NGLL,IG_NGLL,IG_NGLL)         :: intpz3

    real, dimension(3, IG_NGLL,IG_NGLL,IG_NGLL) :: rl_displacement_gll
    real, dimension(3,IG_NGLL,IG_NGLL,IG_NGLL) :: rl_acceleration_gll

    do iel = elt_start,elt_end

       do k = 1,IG_NGLL        !zeta
          do l = 1,IG_NGLL     !eta
             do m = 1,IG_NGLL  !xi

                igll                         = ig_hexa_gll_glonum(m,l,k,iel)
                !print *, igll, rg_gll_displacement(1,igll), rg_gll_displacement(2,igll), rg_gll_displacement(3,igll)
                rl_displacement_gll(1,m,l,k) = rg_gll_displacement(1,igll)
                rl_displacement_gll(2,m,l,k) = rg_gll_displacement(2,igll)
                rl_displacement_gll(3,m,l,k) = rg_gll_displacement(3,igll)

             enddo
          enddo
       enddo
       !print *, rl_displacement_gll(:,:,:,:)
       !print *, rl_displacement_gll(1, 1, 1, 1)
       !print *, rl_displacement_gll(2, 1, 1, 1)
       !print *, rl_displacement_gll(3, 1, 1, 1)
       do k = 1,IG_NGLL        !zeta
          do l = 1,IG_NGLL     !eta
             do m = 1,IG_NGLL  !xi

                duxdxi = rl_displacement_gll(1,1,l,k)*rg_gll_lagrange_deriv(1,m) &
                       + rl_displacement_gll(1,2,l,k)*rg_gll_lagrange_deriv(2,m) &
                       + rl_displacement_gll(1,3,l,k)*rg_gll_lagrange_deriv(3,m) &
                       + rl_displacement_gll(1,4,l,k)*rg_gll_lagrange_deriv(4,m) &
                       + rl_displacement_gll(1,5,l,k)*rg_gll_lagrange_deriv(5,m)


                duydxi = rl_displacement_gll(2,1,l,k)*rg_gll_lagrange_deriv(1,m) &
                       + rl_displacement_gll(2,2,l,k)*rg_gll_lagrange_deriv(2,m) &
                       + rl_displacement_gll(2,3,l,k)*rg_gll_lagrange_deriv(3,m) &
                       + rl_displacement_gll(2,4,l,k)*rg_gll_lagrange_deriv(4,m) &
                       + rl_displacement_gll(2,5,l,k)*rg_gll_lagrange_deriv(5,m)

                duzdxi = rl_displacement_gll(3,1,l,k)*rg_gll_lagrange_deriv(1,m) &
                       + rl_displacement_gll(3,2,l,k)*rg_gll_lagrange_deriv(2,m) &
                       + rl_displacement_gll(3,3,l,k)*rg_gll_lagrange_deriv(3,m) &
                       + rl_displacement_gll(3,4,l,k)*rg_gll_lagrange_deriv(4,m) &
                       + rl_displacement_gll(3,5,l,k)*rg_gll_lagrange_deriv(5,m)

                duxdet = rl_displacement_gll(1,m,1,k)*rg_gll_lagrange_deriv(1,l) &
                       + rl_displacement_gll(1,m,2,k)*rg_gll_lagrange_deriv(2,l) &
                       + rl_displacement_gll(1,m,3,k)*rg_gll_lagrange_deriv(3,l) &
                       + rl_displacement_gll(1,m,4,k)*rg_gll_lagrange_deriv(4,l) &
                       + rl_displacement_gll(1,m,5,k)*rg_gll_lagrange_deriv(5,l)

                duydet = rl_displacement_gll(2,m,1,k)*rg_gll_lagrange_deriv(1,l) &
                       + rl_displacement_gll(2,m,2,k)*rg_gll_lagrange_deriv(2,l) &
                       + rl_displacement_gll(2,m,3,k)*rg_gll_lagrange_deriv(3,l) &
                       + rl_displacement_gll(2,m,4,k)*rg_gll_lagrange_deriv(4,l) &
                       + rl_displacement_gll(2,m,5,k)*rg_gll_lagrange_deriv(5,l)

                duzdet = rl_displacement_gll(3,m,1,k)*rg_gll_lagrange_deriv(1,l) &
                       + rl_displacement_gll(3,m,2,k)*rg_gll_lagrange_deriv(2,l) &
                       + rl_displacement_gll(3,m,3,k)*rg_gll_lagrange_deriv(3,l) &
                       + rl_displacement_gll(3,m,4,k)*rg_gll_lagrange_deriv(4,l) &
                       + rl_displacement_gll(3,m,5,k)*rg_gll_lagrange_deriv(5,l)

                duxdze = rl_displacement_gll(1,m,l,1)*rg_gll_lagrange_deriv(1,k) &
                       + rl_displacement_gll(1,m,l,2)*rg_gll_lagrange_deriv(2,k) &
                       + rl_displacement_gll(1,m,l,3)*rg_gll_lagrange_deriv(3,k) &
                       + rl_displacement_gll(1,m,l,4)*rg_gll_lagrange_deriv(4,k) &
                       + rl_displacement_gll(1,m,l,5)*rg_gll_lagrange_deriv(5,k)

                duydze = rl_displacement_gll(2,m,l,1)*rg_gll_lagrange_deriv(1,k) &
                       + rl_displacement_gll(2,m,l,2)*rg_gll_lagrange_deriv(2,k) &
                       + rl_displacement_gll(2,m,l,3)*rg_gll_lagrange_deriv(3,k) &
                       + rl_displacement_gll(2,m,l,4)*rg_gll_lagrange_deriv(4,k) &
                       + rl_displacement_gll(2,m,l,5)*rg_gll_lagrange_deriv(5,k)

                duzdze = rl_displacement_gll(3,m,l,1)*rg_gll_lagrange_deriv(1,k) &
                       + rl_displacement_gll(3,m,l,2)*rg_gll_lagrange_deriv(2,k) &
                       + rl_displacement_gll(3,m,l,3)*rg_gll_lagrange_deriv(3,k) &
                       + rl_displacement_gll(3,m,l,4)*rg_gll_lagrange_deriv(4,k) &
                       + rl_displacement_gll(3,m,l,5)*rg_gll_lagrange_deriv(5,k)

                !print *, duxdxi !, duydxi, duzdxi
                !print *, duxdet, duydet, duzdet

                dxidx = rg_hexa_gll_dxidx(m,l,k,iel)
                dxidy = rg_hexa_gll_dxidy(m,l,k,iel)
                dxidz = rg_hexa_gll_dxidz(m,l,k,iel)
                detdx = rg_hexa_gll_detdx(m,l,k,iel)
                detdy = rg_hexa_gll_detdy(m,l,k,iel)
                detdz = rg_hexa_gll_detdz(m,l,k,iel)
                dzedx = rg_hexa_gll_dzedx(m,l,k,iel)
                dzedy = rg_hexa_gll_dzedy(m,l,k,iel)
                dzedz = rg_hexa_gll_dzedz(m,l,k,iel)

                duxdx = duxdxi*dxidx + duxdet*detdx + duxdze*dzedx
                duxdy = duxdxi*dxidy + duxdet*detdy + duxdze*dzedy
                duxdz = duxdxi*dxidz + duxdet*detdz + duxdze*dzedz
                duydx = duydxi*dxidx + duydet*detdx + duydze*dzedx
                duydy = duydxi*dxidy + duydet*detdy + duydze*dzedy
                duydz = duydxi*dxidz + duydet*detdz + duydze*dzedz
                duzdx = duzdxi*dxidx + duzdet*detdx + duzdze*dzedx
                duzdy = duzdxi*dxidy + duzdet*detdy + duzdze*dzedy
                duzdz = duzdxi*dxidz + duzdet*detdz + duzdze*dzedz

                trace_tau = (rg_hexa_gll_rhovp2(m,l,k,iel) - 2.0*rg_hexa_gll_rhovs2(m,l,k,iel))*(duxdx+duydy+duzdz)
                tauxx     = trace_tau + 2.0*rg_hexa_gll_rhovs2(m,l,k,iel)*duxdx
                tauyy     = trace_tau + 2.0*rg_hexa_gll_rhovs2(m,l,k,iel)*duydy
                tauzz     = trace_tau + 2.0*rg_hexa_gll_rhovs2(m,l,k,iel)*duzdz
                tauxy     =                 rg_hexa_gll_rhovs2(m,l,k,iel)*(duxdy+duydx)
                tauxz     =                 rg_hexa_gll_rhovs2(m,l,k,iel)*(duxdz+duzdx)
                tauyz     =                 rg_hexa_gll_rhovs2(m,l,k,iel)*(duydz+duzdy)

                intpx1(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxx*dxidx+tauxy*dxidy+tauxz*dxidz)
                intpx2(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxx*detdx+tauxy*detdy+tauxz*detdz)
                intpx3(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxx*dzedx+tauxy*dzedy+tauxz*dzedz)

                intpy1(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxy*dxidx+tauyy*dxidy+tauyz*dxidz)
                intpy2(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxy*detdx+tauyy*detdy+tauyz*detdz)
                intpy3(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxy*dzedx+tauyy*dzedy+tauyz*dzedz)

                intpz1(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxz*dxidx+tauyz*dxidy+tauzz*dxidz)
                intpz2(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxz*detdx+tauyz*detdy+tauzz*detdz)
                intpz3(m,l,k) = rg_hexa_gll_jacobian_det(m,l,k,iel)*(tauxz*dzedx+tauyz*dzedy+tauzz*dzedz)

             enddo
          enddo
       enddo

       do k = 1,IG_NGLL
         do l = 1,IG_NGLL
           do m = 1,IG_NGLL

             tmpx1 = intpx1(1,l,k)*rg_gll_lagrange_deriv(m,1)*rg_gll_weight(1) &
                   + intpx1(2,l,k)*rg_gll_lagrange_deriv(m,2)*rg_gll_weight(2) &
                   + intpx1(3,l,k)*rg_gll_lagrange_deriv(m,3)*rg_gll_weight(3) &
                   + intpx1(4,l,k)*rg_gll_lagrange_deriv(m,4)*rg_gll_weight(4) &
                   + intpx1(5,l,k)*rg_gll_lagrange_deriv(m,5)*rg_gll_weight(5)

             tmpy1 = intpy1(1,l,k)*rg_gll_lagrange_deriv(m,1)*rg_gll_weight(1) &
                   + intpy1(2,l,k)*rg_gll_lagrange_deriv(m,2)*rg_gll_weight(2) &
                   + intpy1(3,l,k)*rg_gll_lagrange_deriv(m,3)*rg_gll_weight(3) &
                   + intpy1(4,l,k)*rg_gll_lagrange_deriv(m,4)*rg_gll_weight(4) &
                   + intpy1(5,l,k)*rg_gll_lagrange_deriv(m,5)*rg_gll_weight(5)

             tmpz1 = intpz1(1,l,k)*rg_gll_lagrange_deriv(m,1)*rg_gll_weight(1) &
                   + intpz1(2,l,k)*rg_gll_lagrange_deriv(m,2)*rg_gll_weight(2) &
                   + intpz1(3,l,k)*rg_gll_lagrange_deriv(m,3)*rg_gll_weight(3) &
                   + intpz1(4,l,k)*rg_gll_lagrange_deriv(m,4)*rg_gll_weight(4) &
                   + intpz1(5,l,k)*rg_gll_lagrange_deriv(m,5)*rg_gll_weight(5)

             tmpx2 = intpx2(m,1,k)*rg_gll_lagrange_deriv(l,1)*rg_gll_weight(1) &
                   + intpx2(m,2,k)*rg_gll_lagrange_deriv(l,2)*rg_gll_weight(2) &
                   + intpx2(m,3,k)*rg_gll_lagrange_deriv(l,3)*rg_gll_weight(3) &
                   + intpx2(m,4,k)*rg_gll_lagrange_deriv(l,4)*rg_gll_weight(4) &
                   + intpx2(m,5,k)*rg_gll_lagrange_deriv(l,5)*rg_gll_weight(5)

             tmpy2 = intpy2(m,1,k)*rg_gll_lagrange_deriv(l,1)*rg_gll_weight(1) &
                   + intpy2(m,2,k)*rg_gll_lagrange_deriv(l,2)*rg_gll_weight(2) &
                   + intpy2(m,3,k)*rg_gll_lagrange_deriv(l,3)*rg_gll_weight(3) &
                   + intpy2(m,4,k)*rg_gll_lagrange_deriv(l,4)*rg_gll_weight(4) &
                   + intpy2(m,5,k)*rg_gll_lagrange_deriv(l,5)*rg_gll_weight(5)

             tmpz2 = intpz2(m,1,k)*rg_gll_lagrange_deriv(l,1)*rg_gll_weight(1) &
                   + intpz2(m,2,k)*rg_gll_lagrange_deriv(l,2)*rg_gll_weight(2) &
                   + intpz2(m,3,k)*rg_gll_lagrange_deriv(l,3)*rg_gll_weight(3) &
                   + intpz2(m,4,k)*rg_gll_lagrange_deriv(l,4)*rg_gll_weight(4) &
                   + intpz2(m,5,k)*rg_gll_lagrange_deriv(l,5)*rg_gll_weight(5)

             tmpx3 = intpx3(m,l,1)*rg_gll_lagrange_deriv(k,1)*rg_gll_weight(1) &
                   + intpx3(m,l,2)*rg_gll_lagrange_deriv(k,2)*rg_gll_weight(2) &
                   + intpx3(m,l,3)*rg_gll_lagrange_deriv(k,3)*rg_gll_weight(3) &
                   + intpx3(m,l,4)*rg_gll_lagrange_deriv(k,4)*rg_gll_weight(4) &
                   + intpx3(m,l,5)*rg_gll_lagrange_deriv(k,5)*rg_gll_weight(5)

             tmpy3 = intpy3(m,l,1)*rg_gll_lagrange_deriv(k,1)*rg_gll_weight(1) &
                   + intpy3(m,l,2)*rg_gll_lagrange_deriv(k,2)*rg_gll_weight(2) &
                   + intpy3(m,l,3)*rg_gll_lagrange_deriv(k,3)*rg_gll_weight(3) &
                   + intpy3(m,l,4)*rg_gll_lagrange_deriv(k,4)*rg_gll_weight(4) &
                   + intpy3(m,l,5)*rg_gll_lagrange_deriv(k,5)*rg_gll_weight(5)

             tmpz3 = intpz3(m,l,1)*rg_gll_lagrange_deriv(k,1)*rg_gll_weight(1) &
                   + intpz3(m,l,2)*rg_gll_lagrange_deriv(k,2)*rg_gll_weight(2) &
                   + intpz3(m,l,3)*rg_gll_lagrange_deriv(k,3)*rg_gll_weight(3) &
                   + intpz3(m,l,4)*rg_gll_lagrange_deriv(k,4)*rg_gll_weight(4) &
                   + intpz3(m,l,5)*rg_gll_lagrange_deriv(k,5)*rg_gll_weight(5)

             fac1 = rg_gll_weight(l)*rg_gll_weight(k)
             fac2 = rg_gll_weight(m)*rg_gll_weight(k)
             fac3 = rg_gll_weight(m)*rg_gll_weight(l)

             rl_acceleration_gll(1,m,l,k) = (fac1*tmpx1 + fac2*tmpx2 + fac3*tmpx3)
             rl_acceleration_gll(2,m,l,k) = (fac1*tmpy1 + fac2*tmpy2 + fac3*tmpy3)
             rl_acceleration_gll(3,m,l,k) = (fac1*tmpz1 + fac2*tmpz2 + fac3*tmpz3)

           enddo
         enddo
       enddo

       do k = 1,IG_NGLL        !zeta
         do l = 1,IG_NGLL     !eta
           do m = 1,IG_NGLL  !xi

             igll                        = ig_hexa_gll_glonum(m,l,k,iel)

             rg_gll_acceleration(1,igll) = rg_gll_acceleration(1,igll) - rl_acceleration_gll(1,m,l,k)
             rg_gll_acceleration(2,igll) = rg_gll_acceleration(2,igll) - rl_acceleration_gll(2,m,l,k)
             rg_gll_acceleration(3,igll) = rg_gll_acceleration(3,igll) - rl_acceleration_gll(3,m,l,k)

           enddo
         enddo
       enddo
    enddo

  end subroutine compute_internal_forces_order4

end module cifo4


program main

  use cifo4

  implicit none

  integer :: i, j
  integer :: dims, d1, d2, d3, d4
  integer, parameter :: fileid=1
  integer, allocatable, dimension(:,:,:,:) :: a0

  integer :: beg, end, rate

  print *, "Read file"

  open(unit=fileid,file="../data/ig_hexa_gll_glonum.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(ig_hexa_gll_glonum(d1, d2, d3, d4))
  read(fileid) ig_hexa_gll_glonum(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_gll_displacement.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2
  print *, dims, d1, d2
  allocate(rg_gll_displacement(d1, d2))
  read(fileid) rg_gll_displacement(:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_gll_lagrange_deriv.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2
  print *, dims, d1, d2
  allocate(rg_gll_lagrange_deriv(d1, d2))
  read(fileid) rg_gll_lagrange_deriv(:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dxidx.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dxidx(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dxidx(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dxidy.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dxidy(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dxidy(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dxidz.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dxidz(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dxidz(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_detdx.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_detdx(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_detdx(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_detdy.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_detdy(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_detdy(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_detdz.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_detdz(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_detdz(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dzedx.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dzedx(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dzedx(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dzedy.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dzedy(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dzedy(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_dzedz.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_dzedz(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_dzedz(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_rhovp2.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_rhovp2(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_rhovp2(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_rhovs2.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_rhovs2(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_rhovs2(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_hexa_gll_jacobian_det.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2, d3, d4
  print *, dims, d1, d2, d3, d4
  allocate(rg_hexa_gll_jacobian_det(d1, d2, d3, d4))
  read(fileid) rg_hexa_gll_jacobian_det(:,:,:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_gll_weight.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1
  print *, dims, d1
  allocate(rg_gll_weight(d1))
  read(fileid) rg_gll_weight(:)
  close(fileid)

  open(unit=fileid,file="../data/rg_gll_acceleration_before_loop_iel.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2
  print *, dims, d1, d2
  allocate(rg_gll_acceleration(d1, d2))
  read(fileid) rg_gll_acceleration(:,:)
  close(fileid)

  open(unit=fileid,file="../data/rg_gll_acceleration_after_loop_iel.dat",action="read",access="stream",form="unformatted")
  read(fileid) dims, d1, d2
  print *, dims, d1, d2
  allocate(rg_gll_acceleration_ref(d1, d2))
  read(fileid) rg_gll_acceleration_ref(:,:)
  close(fileid)

  !print *, size(ig_hexa_gll_glonum)/125

  call system_clock( beg, rate )

  call compute_internal_forces_order4(1, size(ig_hexa_gll_glonum)/125)

  call system_clock( end )

  print *, "elapsed time: ", real(end - beg) / real(rate)

  !do i=1, size(rg_gll_acceleration_ref, 2)
  !  do j=1, size(rg_gll_acceleration_ref, 1)
  !    print *, rg_gll_acceleration_ref(j,i), rg_gll_acceleration(j, i)
  !  enddo
  !enddo


  open(unit=fileid, file="main-f90.dat",access="stream",status="replace",action="write",form="unformatted")
  write(fileid) 2, size(rg_gll_acceleration_ref, 1), size(rg_gll_acceleration_ref, 2)
  !print *, 2, size(rg_gll_acceleration_ref, 1), size(rg_gll_acceleration_ref)

  do i=1, size(rg_gll_acceleration_ref, 2)
    do j=1, size(rg_gll_acceleration_ref, 1)
      !print *, rg_gll_acceleration( j , i )
      write(fileid) rg_gll_acceleration( j, i )
    enddo
  enddo
  close(fileid)

  
  deallocate(ig_hexa_gll_glonum)
  deallocate(rg_gll_displacement)
  deallocate(rg_gll_lagrange_deriv)

  deallocate(rg_hexa_gll_dxidx)
  deallocate(rg_hexa_gll_dxidy)
  deallocate(rg_hexa_gll_dxidz)
  deallocate(rg_hexa_gll_detdx)
  deallocate(rg_hexa_gll_detdy)
  deallocate(rg_hexa_gll_detdz)
  deallocate(rg_hexa_gll_dzedx)
  deallocate(rg_hexa_gll_dzedy)
  deallocate(rg_hexa_gll_dzedz)

  deallocate(rg_hexa_gll_rhovp2)
  deallocate(rg_hexa_gll_rhovs2)

  deallocate(rg_hexa_gll_jacobian_det)
  deallocate(rg_gll_weight)
  deallocate(rg_gll_acceleration)
  deallocate(rg_gll_acceleration_ref)

end program main

