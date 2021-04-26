// Copyright (C) 2021  Sylvain Jubertie
//
// EFISPEC3D is available at https://gitlab.brgm.fr/brgm/efispec3d or at http://efispec.free.fr
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
#include <iomanip>

#include <arm_neon.h>

#include <cifo4.hpp>


#define IDX2( m, l ) ( 5 * l + m )
#define IDX3( m, l, k ) ( 25 * k + 5 * l + m )
#define IDX4( m, l, k, iel ) ( 125 * (iel) + 25 * (k) + 5 * (l) + (m) )
/*
#define pr256( r ) { float t[ 8 ] alignas( 32 );            \
                     _mm256_store_ps( t, r );               \
                     for( std::size_t i = 0 ; i < 8 ; ++i ) \
                     {                                      \
                       std::cout << t[ i ] << ' ';          \
                     }                                      \
                   }
*/

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


void print( float32x4_t const & v )
{
  for( std::size_t i = 0 ; i < 4 ; ++i )
  {
    std::cout << v[ i ] << ' ';
  }
  std::cout << std::endl;
}


void compute_internal_forces_order4( std::size_t elt_start, std::size_t elt_end )
{
  float32x4_t rl_displacement_gll[5*5*5*3];

  float32x4_t local[ 5 * 5 * 5 * 9 ];

  float32x4_t * intpx1 = &local[    0 ];
  float32x4_t * intpy1 = &local[  125 ];
  float32x4_t * intpz1 = &local[  250 ];

  float32x4_t * intpx2 = &local[  375 ];
  float32x4_t * intpy2 = &local[  500 ];
  float32x4_t * intpz2 = &local[  625 ];

  float32x4_t * intpx3 = &local[  750 ];
  float32x4_t * intpy3 = &local[  875 ];
  float32x4_t * intpz3 = &local[ 1000 ];

  for( std::size_t iel = elt_start ; iel + 4 <= elt_end ; iel += 4 )
  {

    //tt = tic();

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto id = 3 * IDX3( m, l, k );
          auto id0 = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, ( iel + 0 ) ) ] - 1 );
          auto id1 = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, ( iel + 1 ) ) ] - 1 );
          auto id2 = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, ( iel + 2 ) ) ] - 1 );
          auto id3 = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, ( iel + 3 ) ) ] - 1 );

          rl_displacement_gll[ 0 + id ][ 0 ] = rg_gll_displacement[ 0 + id0 ];
          rl_displacement_gll[ 0 + id ][ 1 ] = rg_gll_displacement[ 0 + id1 ];
          rl_displacement_gll[ 0 + id ][ 2 ] = rg_gll_displacement[ 0 + id2 ];
          rl_displacement_gll[ 0 + id ][ 3 ] = rg_gll_displacement[ 0 + id3 ];

          rl_displacement_gll[ 1 + id ][ 0 ] = rg_gll_displacement[ 1 + id0 ];
          rl_displacement_gll[ 1 + id ][ 1 ] = rg_gll_displacement[ 1 + id1 ];
          rl_displacement_gll[ 1 + id ][ 2 ] = rg_gll_displacement[ 1 + id2 ];
          rl_displacement_gll[ 1 + id ][ 3 ] = rg_gll_displacement[ 1 + id3 ];

          rl_displacement_gll[ 2 + id ][ 0 ] = rg_gll_displacement[ 2 + id0 ];
          rl_displacement_gll[ 2 + id ][ 1 ] = rg_gll_displacement[ 2 + id1 ];
          rl_displacement_gll[ 2 + id ][ 2 ] = rg_gll_displacement[ 2 + id2 ];
          rl_displacement_gll[ 2 + id ][ 3 ] = rg_gll_displacement[ 2 + id3 ];
        }
      }
    }

/*
    for( std::size_t i = 0 ; i < 125*3*4 ; ++i )
    {
      std::cout << ((float*)rl_displacement_gll)[ i ] << ' ';
    }
    std::cout << std::endl;
  */

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {

          auto coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 0, m ) ] );

          auto index = 0 + 3 * IDX3( 0, l, k );

          auto duxdxi = rl_displacement_gll[ 0 + index ] * coeff;
          auto duydxi = rl_displacement_gll[ 1 + index ] * coeff;
          auto duzdxi = rl_displacement_gll[ 2 + index ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 1, m ) ] );

          duxdxi += rl_displacement_gll[ 3 + index ] * coeff;
          duydxi += rl_displacement_gll[ 4 + index ] * coeff;
          duzdxi += rl_displacement_gll[ 5 + index ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 2, m ) ] );

          duxdxi += rl_displacement_gll[ 6 + index ] * coeff;
          duydxi += rl_displacement_gll[ 7 + index ] * coeff;
          duzdxi += rl_displacement_gll[ 8 + index ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 3, m ) ] );

          duxdxi += rl_displacement_gll[  9 + index ] * coeff;
          duydxi += rl_displacement_gll[ 10 + index ] * coeff;
          duzdxi += rl_displacement_gll[ 11 + index ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 4, m ) ] );

          duxdxi += rl_displacement_gll[ 12 + index ] * coeff;
          duydxi += rl_displacement_gll[ 13 + index ] * coeff;
          duzdxi += rl_displacement_gll[ 14 + index ] * coeff;

          //

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 0, l ) ] );

          auto duxdet = rl_displacement_gll[ 0 + 3 * IDX3( m, 0, k ) ] * coeff;
          auto duydet = rl_displacement_gll[ 1 + 3 * IDX3( m, 0, k ) ] * coeff;
          auto duzdet = rl_displacement_gll[ 2 + 3 * IDX3( m, 0, k ) ] * coeff;

	  //print( duxdet );

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 1, l ) ] );

          duxdet += rl_displacement_gll[ 0 + 3 * IDX3( m, 1, k ) ] * coeff;
          duydet += rl_displacement_gll[ 1 + 3 * IDX3( m, 1, k ) ] * coeff;
          duzdet += rl_displacement_gll[ 2 + 3 * IDX3( m, 1, k ) ] * coeff;

	  //print( duxdet );

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 2, l ) ] );

          duxdet += rl_displacement_gll[ 0 + 3 * IDX3( m, 2, k ) ] * coeff;
          duydet += rl_displacement_gll[ 1 + 3 * IDX3( m, 2, k ) ] * coeff;
          duzdet += rl_displacement_gll[ 2 + 3 * IDX3( m, 2, k ) ] * coeff;

	  //print( duxdet );

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 3, l ) ] );

          duxdet += rl_displacement_gll[ 0 + 3 * IDX3( m, 3, k ) ] * coeff;
          duydet += rl_displacement_gll[ 1 + 3 * IDX3( m, 3, k ) ] * coeff;
          duzdet += rl_displacement_gll[ 2 + 3 * IDX3( m, 3, k ) ] * coeff;

	  //print( duxdet );

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 4, l ) ] );

          duxdet += rl_displacement_gll[ 0 + 3 * IDX3( m, 4, k ) ] * coeff;
          duydet += rl_displacement_gll[ 1 + 3 * IDX3( m, 4, k ) ] * coeff;
          duzdet += rl_displacement_gll[ 2 + 3 * IDX3( m, 4, k ) ] * coeff;

	  //print( duxdet );

	  //std::cout << std::endl;
          //

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 0, k ) ] );

          auto duxdze = rl_displacement_gll[ 0 + 3 * IDX3( m, l, 0 ) ] * coeff;
          auto duydze = rl_displacement_gll[ 1 + 3 * IDX3( m, l, 0 ) ] * coeff;
          auto duzdze = rl_displacement_gll[ 2 + 3 * IDX3( m, l, 0 ) ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 1, k ) ] );

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 1 ) ] * coeff;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 1 ) ] * coeff;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 1 ) ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 2, k ) ] );

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 2 ) ] * coeff;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 2 ) ] * coeff;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 2 ) ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 3, k ) ] );

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 3 ) ] * coeff;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 3 ) ] * coeff;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 3 ) ] * coeff;

          coeff = vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( 4, k ) ] );

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 4 ) ] * coeff;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 4 ) ] * coeff;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 4 ) ] * coeff;

          //

          auto id0 = IDX4( m, l, k, ( iel + 0 ) );
          auto id1 = IDX4( m, l, k, ( iel + 1 ) );
          auto id2 = IDX4( m, l, k, ( iel + 2 ) );
          auto id3 = IDX4( m, l, k, ( iel + 3 ) );

          float32x4_t dxidx;
          dxidx[ 0 ] = rg_hexa_gll_dxidx[ id0 ];
          dxidx[ 1 ] = rg_hexa_gll_dxidx[ id1 ];
          dxidx[ 2 ] = rg_hexa_gll_dxidx[ id2 ];
          dxidx[ 3 ] = rg_hexa_gll_dxidx[ id3 ];

          float32x4_t detdx;
          detdx[ 0 ] = rg_hexa_gll_detdx[ id0 ];
          detdx[ 1 ] = rg_hexa_gll_detdx[ id1 ];
          detdx[ 2 ] = rg_hexa_gll_detdx[ id2 ];
          detdx[ 3 ] = rg_hexa_gll_detdx[ id3 ];

          float32x4_t dzedx;
          dzedx[ 0 ] = rg_hexa_gll_dzedx[ id0 ];
          dzedx[ 1 ] = rg_hexa_gll_dzedx[ id1 ];
          dzedx[ 2 ] = rg_hexa_gll_dzedx[ id2 ];
          dzedx[ 3 ] = rg_hexa_gll_dzedx[ id3 ];

          auto duxdx = duxdxi * dxidx + duxdet * detdx + duxdze * dzedx;
          auto duydx = duydxi * dxidx + duydet * detdx + duydze * dzedx;
          auto duzdx = duzdxi * dxidx + duzdet * detdx + duzdze * dzedx;

          //print( dxidx );
          //print( duxdx );

          float32x4_t dxidy;
          dxidy[ 0 ] = rg_hexa_gll_dxidy[ id0 ];
          dxidy[ 1 ] = rg_hexa_gll_dxidy[ id1 ];
          dxidy[ 2 ] = rg_hexa_gll_dxidy[ id2 ];
          dxidy[ 3 ] = rg_hexa_gll_dxidy[ id3 ];

          float32x4_t detdy;
          detdy[ 0 ] = rg_hexa_gll_detdy[ id0 ];
          detdy[ 1 ] = rg_hexa_gll_detdy[ id1 ];
          detdy[ 2 ] = rg_hexa_gll_detdy[ id2 ];
          detdy[ 3 ] = rg_hexa_gll_detdy[ id3 ];

          float32x4_t dzedy;
          dzedy[ 0 ] = rg_hexa_gll_dzedy[ id0 ];
          dzedy[ 1 ] = rg_hexa_gll_dzedy[ id1 ];
          dzedy[ 2 ] = rg_hexa_gll_dzedy[ id2 ];
          dzedy[ 3 ] = rg_hexa_gll_dzedy[ id3 ];

          auto duxdy = duxdxi * dxidy + duxdet * detdy + duxdze * dzedy;
          auto duydy = duydxi * dxidy + duydet * detdy + duydze * dzedy;
          auto duzdy = duzdxi * dxidy + duzdet * detdy + duzdze * dzedy;

          float32x4_t dxidz;
          dxidz[ 0 ] = rg_hexa_gll_dxidz[ id0 ];
          dxidz[ 1 ] = rg_hexa_gll_dxidz[ id1 ];
          dxidz[ 2 ] = rg_hexa_gll_dxidz[ id2 ];
          dxidz[ 3 ] = rg_hexa_gll_dxidz[ id3 ];

          float32x4_t detdz;
          detdz[ 0 ] = rg_hexa_gll_detdz[ id0 ];
          detdz[ 1 ] = rg_hexa_gll_detdz[ id1 ];
          detdz[ 2 ] = rg_hexa_gll_detdz[ id2 ];
          detdz[ 3 ] = rg_hexa_gll_detdz[ id3 ];

          float32x4_t dzedz;
          dzedz[ 0 ] = rg_hexa_gll_dzedz[ id0 ];
          dzedz[ 1 ] = rg_hexa_gll_dzedz[ id1 ];
          dzedz[ 2 ] = rg_hexa_gll_dzedz[ id2 ];
          dzedz[ 3 ] = rg_hexa_gll_dzedz[ id3 ];

          auto duxdz = duxdxi * dxidz + duxdet * detdz + duxdze * dzedz;
          auto duydz = duydxi * dxidz + duydet * detdz + duydze * dzedz;
          auto duzdz = duzdxi * dxidz + duzdet * detdz + duzdze * dzedz;

          //std::cout << std::endl;

          float32x4_t rhovp2;
          rhovp2[ 0 ] = rg_hexa_gll_rhovp2[ id0 ];
          rhovp2[ 1 ] = rg_hexa_gll_rhovp2[ id1 ];
          rhovp2[ 2 ] = rg_hexa_gll_rhovp2[ id2 ];
          rhovp2[ 3 ] = rg_hexa_gll_rhovp2[ id3 ];

          float32x4_t rhovs2;
          rhovs2[ 0 ] = rg_hexa_gll_rhovs2[ id0 ];
          rhovs2[ 1 ] = rg_hexa_gll_rhovs2[ id1 ];
          rhovs2[ 2 ] = rg_hexa_gll_rhovs2[ id2 ];
          rhovs2[ 3 ] = rg_hexa_gll_rhovs2[ id3 ];

          //print( rhovp2 );
          //std::cout << std::endl;

          auto trace_tau = ( rhovp2 - vdupq_n_f32( 2.0f ) * rhovs2 )*(duxdx+duydy+duzdz);
          auto tauxx     = trace_tau + vdupq_n_f32( 2.0f )*rhovs2*duxdx;
          auto tauyy     = trace_tau + vdupq_n_f32( 2.0f )*rhovs2*duydy;
          auto tauzz     = trace_tau + vdupq_n_f32( 2.0f )*rhovs2*duzdz;
          auto tauxy     =                 rhovs2*(duxdy+duydx);
          auto tauxz     =                 rhovs2*(duxdz+duzdx);
          auto tauyz     =                 rhovs2*(duydz+duzdy);

          //print( rhovp2 );
          //print( rhovs2 );
          //print( duxdx );
          //print( duydy );
          //print( duzdz );
          //print( trace_tau );
          //std::cout << std::endl;

	        float32x4_t tmp;
      	  tmp[ 0 ] = rg_hexa_gll_jacobian_det[ id0 ];
	        tmp[ 1 ] = rg_hexa_gll_jacobian_det[ id1 ];
	        tmp[ 2 ] = rg_hexa_gll_jacobian_det[ id2 ];
	        tmp[ 3 ] = rg_hexa_gll_jacobian_det[ id3 ];

          intpx1[ IDX3( m, l, k ) ] = tmp * (tauxx*dxidx+tauxy*dxidy+tauxz*dxidz);
          intpx2[ IDX3( m, l, k ) ] = tmp * (tauxx*detdx+tauxy*detdy+tauxz*detdz);
          intpx3[ IDX3( m, l, k ) ] = tmp * (tauxx*dzedx+tauxy*dzedy+tauxz*dzedz);

          intpy1[ IDX3( m, l, k ) ] = tmp * (tauxy*dxidx+tauyy*dxidy+tauyz*dxidz);
          intpy2[ IDX3( m, l, k ) ] = tmp * (tauxy*detdx+tauyy*detdy+tauyz*detdz);
          intpy3[ IDX3( m, l, k ) ] = tmp * (tauxy*dzedx+tauyy*dzedy+tauyz*dzedz);

          intpz1[ IDX3( m, l, k ) ] = tmp * (tauxz*dxidx+tauyz*dxidy+tauzz*dxidz);
          intpz2[ IDX3( m, l, k ) ] = tmp * (tauxz*detdx+tauyz*detdy+tauzz*detdz);
          intpz3[ IDX3( m, l, k ) ] = tmp * (tauxz*dzedx+tauyz*dzedy+tauzz*dzedz);

/*
          for( std::size_t i = 0 ; i < 4 ; ++i )
          {
            std::cout << intpz1[ IDX3( m, l, k ) ][ i ] << ' ';
          }
          std::cout << std::endl;
*/
        }
      }
    }

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx1 = intpx1[ IDX3( 0, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpx1[ IDX3( 1, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpx1[ IDX3( 2, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpx1[ IDX3( 3, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpx1[ IDX3( 4, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );



          auto tmpy1 = intpy1[ IDX3( 0, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpy1[ IDX3( 1, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpy1[ IDX3( 2, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpy1[ IDX3( 3, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpy1[ IDX3( 4, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          auto tmpz1 = intpz1[ IDX3( 0, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpz1[ IDX3( 1, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpz1[ IDX3( 2, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpz1[ IDX3( 3, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpz1[ IDX3( 4, l, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );


          auto tmpx2 = intpx2[ IDX3( m, 0, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpx2[ IDX3( m, 1, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpx2[ IDX3( m, 2, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpx2[ IDX3( m, 3, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpx2[ IDX3( m, 4, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          auto tmpy2 = intpy2[ IDX3( m, 0, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpy2[ IDX3( m, 1, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpy2[ IDX3( m, 2, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpy2[ IDX3( m, 3, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpy2[ IDX3( m, 4, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          auto tmpz2 = intpz2[ IDX3( m, 0, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpz2[ IDX3( m, 1, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpz2[ IDX3( m, 2, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpz2[ IDX3( m, 3, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpz2[ IDX3( m, 4, k ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );


          auto tmpx3 = intpx3[ IDX3( m, l, 0 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpx3[ IDX3( m, l, 1 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpx3[ IDX3( m, l, 2 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpx3[ IDX3( m, l, 3 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpx3[ IDX3( m, l, 4 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          auto tmpy3 = intpy3[ IDX3( m, l, 0 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpy3[ IDX3( m, l, 1 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpy3[ IDX3( m, l, 2 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpy3[ IDX3( m, l, 3 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpy3[ IDX3( m, l, 4 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          auto tmpz3 = intpz3[ IDX3( m, l, 0 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * vdupq_n_f32( rg_gll_weight[ 0 ] )
                     + intpz3[ IDX3( m, l, 1 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * vdupq_n_f32( rg_gll_weight[ 1 ] )
                     + intpz3[ IDX3( m, l, 2 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * vdupq_n_f32( rg_gll_weight[ 2 ] )
                     + intpz3[ IDX3( m, l, 3 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * vdupq_n_f32( rg_gll_weight[ 3 ] )
                     + intpz3[ IDX3( m, l, 4 ) ] * vdupq_n_f32( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * vdupq_n_f32( rg_gll_weight[ 4 ] );

          //print( tmpx3 );
          //print( tmpy3 );
          //print( tmpz3 );
	  //std::cout << std::endl;

          auto fac1 = vdupq_n_f32( rg_gll_weight[ l ] ) * vdupq_n_f32( rg_gll_weight[ k ] );
          auto fac2 = vdupq_n_f32( rg_gll_weight[ m ] ) * vdupq_n_f32( rg_gll_weight[ k ] );
          auto fac3 = vdupq_n_f32( rg_gll_weight[ m ] ) * vdupq_n_f32( rg_gll_weight[ l ] );

          auto rx = fac1 * tmpx1 + fac2 * tmpx2 + fac3 * tmpx3;
          auto ry = fac1 * tmpy1 + fac2 * tmpy2 + fac3 * tmpy3;
          auto rz = fac1 * tmpz1 + fac2 * tmpz2 + fac3 * tmpz3;

          //print( rx );
          //print( ry );
          //print( rz );
	  //std::cout << std::endl;

          float tx[ 4 ] __attribute__((aligned(32)));//alignas(32);
          float ty[ 4 ] __attribute__((aligned(32)));//alignas(32);
          float tz[ 4 ] __attribute__((aligned(32)));//alignas(32);

          vst1q_f32( tx, rx );
          vst1q_f32( ty, ry );
          vst1q_f32( tz, rz );

          for( std::size_t i = 0 ; i < 4 ; ++i )
          {
	    //std::cout << IDX4( m, l, k, iel + i ) << std::endl;

            auto idx = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, iel + i ) ] - 1 );

	    //std::cout << idx << std::endl;

            //std::cout << std::endl;

            rg_gll_acceleration[ 0 + idx ] -= tx[ i ];
            rg_gll_acceleration[ 1 + idx ] -= ty[ i ];
            rg_gll_acceleration[ 2 + idx ] -= tz[ i ];
          }

        }
      }

    }

  }

  //std::cout << "[intrin] 1 = " << avg(tt_firstloop) << "\n";
  ////std::cout << "[intrin] 2.1 = " << avg(tt_midloop1) << "\n";
  //std::cout << "[intrin] 2.2 = " << avg(tt_midloop2) << "\n";
  //std::cout << "[intrin] 3 = " << avg(tt_lastloop) << "\n";

}
