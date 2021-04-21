// Copyright (C) 2021  Sylvain Jubertie
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <cstddef>
#include <iostream>

#include <cifo4.hpp>


#define IDX2( m, l ) ( 5 * l + m )
#define IDX3( m, l, k ) ( 25 * k + 5 * l + m )
#define IDX4( m, l, k, iel ) ( 125 * iel + 25 * k + 5 * l + m )


std::vector< uint32_t, allocator< uint32_t> > ig_hexa_gll_glonum;

std::vector< float, allocator< float> > rg_gll_displacement;
std::vector< float, allocator< float> > rg_gll_weight;

std::vector< float, allocator< float> > rg_gll_lagrange_deriv;
std::vector< float, allocator< float> > rg_gll_acceleration;

std::vector< float, allocator< float> > rg_hexa_gll_dxidx;
std::vector< float, allocator< float> > rg_hexa_gll_dxidy;
std::vector< float, allocator< float> > rg_hexa_gll_dxidz;
std::vector< float, allocator< float> > rg_hexa_gll_detdx;
std::vector< float, allocator< float> > rg_hexa_gll_detdy;
std::vector< float, allocator< float> > rg_hexa_gll_detdz;
std::vector< float, allocator< float> > rg_hexa_gll_dzedx;
std::vector< float, allocator< float> > rg_hexa_gll_dzedy;
std::vector< float, allocator< float> > rg_hexa_gll_dzedz;

std::vector< float, allocator< float> > rg_hexa_gll_rhovp2;
std::vector< float, allocator< float> > rg_hexa_gll_rhovs2;
std::vector< float, allocator< float> > rg_hexa_gll_jacobian_det;


void compute_internal_forces_order4( std::size_t elt_start, std::size_t elt_end )
{
  float rl_displacement_gll[5*5*5*3];

  // Allocate all the local arrays at once ( 5 * 5 * 5 * 9 * 4 = 4.5 kB ).
  float local[ 5 * 5 * 5 * 9 ];

  float * intpx1 = &local[    0 ];
  float * intpy1 = &local[  125 ];
  float * intpz1 = &local[  250 ];

  float * intpx2 = &local[  375 ];
  float * intpy2 = &local[  500 ];
  float * intpz2 = &local[  625 ];

  float * intpx3 = &local[  750 ];
  float * intpy3 = &local[  875 ];
  float * intpz3 = &local[ 1000 ];


  for( std::size_t iel = elt_start ; iel < elt_end ; ++iel )
  {
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto igll = ig_hexa_gll_glonum[ IDX4( 0, l, k, iel ) ] - 1; // Fortran!
          auto coeff = rg_gll_lagrange_deriv[ IDX2( 0, m ) ];

          auto duxdxi = rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          auto duydxi = rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          auto duzdxi = rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( 1, l, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 1, m ) ];

          duxdxi += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydxi += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdxi += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( 2, l, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 2, m ) ];

          duxdxi += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydxi += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdxi += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( 3, l, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 3, m ) ];

          duxdxi += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydxi += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdxi += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( 4, l, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 4, m ) ];

          duxdxi += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydxi += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdxi += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          //

          igll = ig_hexa_gll_glonum[ IDX4( m, 0, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 0, l ) ];

          auto duxdet = rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          auto duydet = rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          auto duzdet = rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, 1, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 1, l ) ];

          duxdet += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydet += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdet += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, 2, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 2, l ) ];

          duxdet += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydet += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdet += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, 3, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 3, l ) ];

          duxdet += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydet += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdet += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, 4, k, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 4, l ) ];

          duxdet += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydet += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdet += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          //

          igll = ig_hexa_gll_glonum[ IDX4( m, l, 0, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 0, k ) ];

          auto duxdze = rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          auto duydze = rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          auto duzdze = rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, l, 1, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 1, k ) ];

          duxdze += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydze += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdze += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, l, 2, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 2, k ) ];

          duxdze += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydze += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdze += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, l, 3, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 3, k ) ];

          duxdze += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydze += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdze += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          igll = ig_hexa_gll_glonum[ IDX4( m, l, 4, iel ) ] - 1;
          coeff = rg_gll_lagrange_deriv[ IDX2( 4, k ) ];

          duxdze += rg_gll_displacement[ 0 + 3 * igll ] * coeff;
          duydze += rg_gll_displacement[ 1 + 3 * igll ] * coeff;
          duzdze += rg_gll_displacement[ 2 + 3 * igll ] * coeff;

          //

          auto dxidx = rg_hexa_gll_dxidx[ IDX4( m, l, k, iel ) ];
          auto detdx = rg_hexa_gll_detdx[ IDX4( m, l, k, iel ) ];
          auto dzedx = rg_hexa_gll_dzedx[ IDX4( m, l, k, iel ) ];

          auto duxdx = duxdxi*dxidx + duxdet*detdx + duxdze*dzedx;
          auto duydx = duydxi*dxidx + duydet*detdx + duydze*dzedx;
          auto duzdx = duzdxi*dxidx + duzdet*detdx + duzdze*dzedx;

          auto dxidy = rg_hexa_gll_dxidy[ IDX4( m, l, k, iel ) ];
          auto detdy = rg_hexa_gll_detdy[ IDX4( m, l, k, iel ) ];
          auto dzedy = rg_hexa_gll_dzedy[ IDX4( m, l, k, iel ) ];

          auto duxdy = duxdxi*dxidy + duxdet*detdy + duxdze*dzedy;
          auto duydy = duydxi*dxidy + duydet*detdy + duydze*dzedy;
          auto duzdy = duzdxi*dxidy + duzdet*detdy + duzdze*dzedy;

          auto dxidz = rg_hexa_gll_dxidz[ IDX4( m, l, k, iel ) ];
          auto detdz = rg_hexa_gll_detdz[ IDX4( m, l, k, iel ) ];
          auto dzedz = rg_hexa_gll_dzedz[ IDX4( m, l, k, iel ) ];

          auto duxdz = duxdxi*dxidz + duxdet*detdz + duxdze*dzedz;
          auto duydz = duydxi*dxidz + duydet*detdz + duydze*dzedz;
          auto duzdz = duzdxi*dxidz + duzdet*detdz + duzdze*dzedz;

          // Load 8 values from rg_hexa_gll_rhovp2 and rg_hexa_gll_rhovs2.
          auto rhovp2 = rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel ) ];
          auto rhovs2 = rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel ) ];

          auto trace_tau = ( rhovp2 - 2.0 * rhovs2 )*(duxdx+duydy+duzdz);
          auto tauxx     = trace_tau + 2.0*rhovs2*duxdx;
          auto tauyy     = trace_tau + 2.0*rhovs2*duydy;
          auto tauzz     = trace_tau + 2.0*rhovs2*duzdz;
          auto tauxy     =                 rhovs2*(duxdy+duydx);
          auto tauxz     =                 rhovs2*(duxdz+duzdx);
          auto tauyz     =                 rhovs2*(duydz+duzdy);

          intpx1[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxx*dxidx+tauxy*dxidy+tauxz*dxidz);
          intpx2[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxx*detdx+tauxy*detdy+tauxz*detdz);
          intpx3[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxx*dzedx+tauxy*dzedy+tauxz*dzedz);

          intpy1[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxy*dxidx+tauyy*dxidy+tauyz*dxidz);
          intpy2[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxy*detdx+tauyy*detdy+tauyz*detdz);
          intpy3[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxy*dzedx+tauyy*dzedy+tauyz*dzedz);

          intpz1[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxz*dxidx+tauyz*dxidy+tauzz*dxidz);
          intpz2[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxz*detdx+tauyz*detdy+tauzz*detdz);
          intpz3[ IDX3( m, l, k ) ] = rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel ) ]*(tauxz*dzedx+tauyz*dzedy+tauzz*dzedz);
        }
      }
    }

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx1 = intpx1[ IDX3( 0, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpx1[ IDX3( 1, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpx1[ IDX3( 2, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpx1[ IDX3( 3, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpx1[ IDX3( 4, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpy1 = intpy1[ IDX3( 0, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpy1[ IDX3( 1, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpy1[ IDX3( 2, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpy1[ IDX3( 3, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpy1[ IDX3( 4, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpz1 = intpz1[ IDX3( 0, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpz1[ IDX3( 1, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpz1[ IDX3( 2, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpz1[ IDX3( 3, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpz1[ IDX3( 4, l, k ) ] * rg_gll_lagrange_deriv[ IDX2( m, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpx2 = intpx2[ IDX3( m, 0, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpx2[ IDX3( m, 1, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpx2[ IDX3( m, 2, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpx2[ IDX3( m, 3, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpx2[ IDX3( m, 4, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpy2 = intpy2[ IDX3( m, 0, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpy2[ IDX3( m, 1, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpy2[ IDX3( m, 2, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpy2[ IDX3( m, 3, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpy2[ IDX3( m, 4, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpz2 = intpz2[ IDX3( m, 0, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpz2[ IDX3( m, 1, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpz2[ IDX3( m, 2, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpz2[ IDX3( m, 3, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpz2[ IDX3( m, 4, k ) ] * rg_gll_lagrange_deriv[ IDX2( l, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpx3 = intpx3[ IDX3( m, l, 0 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpx3[ IDX3( m, l, 1 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpx3[ IDX3( m, l, 2 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpx3[ IDX3( m, l, 3 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpx3[ IDX3( m, l, 4 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpy3 = intpy3[ IDX3( m, l, 0 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpy3[ IDX3( m, l, 1 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpy3[ IDX3( m, l, 2 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpy3[ IDX3( m, l, 3 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpy3[ IDX3( m, l, 4 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 4 ) ]*rg_gll_weight[ 4 ];

          auto tmpz3 = intpz3[ IDX3( m, l, 0 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 0 ) ]*rg_gll_weight[ 0 ]
                     + intpz3[ IDX3( m, l, 1 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 1 ) ]*rg_gll_weight[ 1 ]
                     + intpz3[ IDX3( m, l, 2 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 2 ) ]*rg_gll_weight[ 2 ]
                     + intpz3[ IDX3( m, l, 3 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 3 ) ]*rg_gll_weight[ 3 ]
                     + intpz3[ IDX3( m, l, 4 ) ] * rg_gll_lagrange_deriv[ IDX2( k, 4 ) ]*rg_gll_weight[ 4 ];

          auto fac1 = rg_gll_weight[ l ]*rg_gll_weight[ k ];
          auto fac2 = rg_gll_weight[ m ]*rg_gll_weight[ k ];
          auto fac3 = rg_gll_weight[ m ]*rg_gll_weight[ l ];

          auto index = ig_hexa_gll_glonum[ IDX4( m, l, k, iel ) ] - 1;

          rg_gll_acceleration[ 0 + 3 * index ] -= fac1 * tmpx1 + fac2 * tmpx2 + fac3 * tmpx3;
          rg_gll_acceleration[ 1 + 3 * index ] -= fac1 * tmpy1 + fac2 * tmpy2 + fac3 * tmpy3;
          rg_gll_acceleration[ 2 + 3 * index ] -= fac1 * tmpz1 + fac2 * tmpz2 + fac3 * tmpz3;
        }
      }
    }
  }
}
