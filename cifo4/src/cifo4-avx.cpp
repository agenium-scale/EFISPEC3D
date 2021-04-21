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

#include <immintrin.h>

#include <cifo4.hpp>


#define IDX2( m, l ) ( 5 * l + m )
#define IDX3( m, l, k ) ( 25 * k + 5 * l + m )
#define IDX4( m, l, k, iel ) ( 125 * (iel) + 25 * (k) + 5 * (l) + (m) )

#define pr256( r ) { float t[ 8 ] alignas( 32 );            \
                     _mm256_store_ps( t, r );               \
                     for( std::size_t i = 0 ; i < 8 ; ++i ) \
                     {                                      \
                       std::cout << t[ i ] << ' ';          \
                     }                                      \
                   }


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
  __m256 rl_displacement_gll[5*5*5*3];

  __m256 local[ 5 * 5 * 5 * 9 ];

  __m256 * intpx1 = &local[    0 ];
  __m256 * intpy1 = &local[  125 ];
  __m256 * intpz1 = &local[  250 ];

  __m256 * intpx2 = &local[  375 ];
  __m256 * intpy2 = &local[  500 ];
  __m256 * intpz2 = &local[  625 ];

  __m256 * intpx3 = &local[  750 ];
  __m256 * intpy3 = &local[  875 ];
  __m256 * intpz3 = &local[ 1000 ];

  for( std::size_t iel = elt_start ; iel < elt_end ; iel+=8 )
  {
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto r0x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r1x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r2x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r3x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r4x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 7 ) ) ] - 1 ) ] );

          auto r0y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r1y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r2y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r3y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r4y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 7 ) ) ] - 1 ) ] );

          auto r0z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 0, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r1z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 1, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r2z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 2, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r3z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 3, l, k, ( iel + 7 ) ) ] - 1 ) ] );
          auto r4z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 0 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 1 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 2 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 3 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 4 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 5 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 6 ) ) ] - 1 ) ]
                                   , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( 4, l, k, ( iel + 7 ) ) ] - 1 ) ] );

          auto coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, m ) ] );

          auto duxdxi = r0x * coeff;
          auto duydxi = r0y * coeff;
          auto duzdxi = r0z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, m ) ] );

          duxdxi += r1x * coeff;
          duydxi += r1y * coeff;
          duzdxi += r1z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, m ) ] );

          duxdxi += r2x * coeff;
          duydxi += r2y * coeff;
          duzdxi += r2z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, m ) ] );

          duxdxi += r3x * coeff;
          duydxi += r3y * coeff;
          duzdxi += r3z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, m ) ] );

          duxdxi += r4x * coeff;
          duydxi += r4y * coeff;
          duzdxi += r4z * coeff;

          //std::cout << k << ' ' << l << ' ' << m << std::endl;
/*
          std::cout << "duxdxi=" << '\n';
          pr256( duxdxi);
          std::cout << "duydxi=" << '\n';
          pr256( duydxi );
          std::cout << "duzdxi=" << '\n';
          pr256( duzdxi );
          std::cout << std::endl;
*/
//
          r0x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 7 ) ) ] - 1 ) ] );
          r1x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 7 ) ) ] - 1 ) ] );
          r2x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 7 ) ) ] - 1 ) ] );
          r3x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 7 ) ) ] - 1 ) ] );
          r4x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 7 ) ) ] - 1 ) ] );

          r0y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 7 ) ) ] - 1 ) ] );
          r1y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 7 ) ) ] - 1 ) ] );
          r2y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 7 ) ) ] - 1 ) ] );
          r3y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 7 ) ) ] - 1 ) ] );
          r4y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 7 ) ) ] - 1 ) ] );

          r0z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 0, k, ( iel + 7 ) ) ] - 1 ) ] );
          r1z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 1, k, ( iel + 7 ) ) ] - 1 ) ] );
          r2z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 2, k, ( iel + 7 ) ) ] - 1 ) ] );
          r3z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 3, k, ( iel + 7 ) ) ] - 1 ) ] );
          r4z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, 4, k, ( iel + 7 ) ) ] - 1 ) ] );

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, l ) ] );

          auto duxdet = r0x * coeff;
          auto duydet = r0y * coeff;
          auto duzdet = r0z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, l ) ] );

          duxdet += r1x * coeff;
          duydet += r1y * coeff;
          duzdet += r1z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, l ) ] );

          duxdet += r2x * coeff;
          duydet += r2y * coeff;
          duzdet += r2z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, l ) ] );

          duxdet += r3x * coeff;
          duydet += r3y * coeff;
          duzdet += r3z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, l ) ] );

          duxdet += r4x * coeff;
          duydet += r4y * coeff;
          duzdet += r4z * coeff;

          r0x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 7 ) ) ] - 1 ) ] );
          r1x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 7 ) ) ] - 1 ) ] );
          r2x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 7 ) ) ] - 1 ) ] );
          r3x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 7 ) ) ] - 1 ) ] );
          r4x = _mm256_setr_ps( rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 0 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 7 ) ) ] - 1 ) ] );

          r0y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 7 ) ) ] - 1 ) ] );
          r1y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 7 ) ) ] - 1 ) ] );
          r2y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 7 ) ) ] - 1 ) ] );
          r3y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 7 ) ) ] - 1 ) ] );
          r4y = _mm256_setr_ps( rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 1 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 7 ) ) ] - 1 ) ] );

          r0z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 0, ( iel + 7 ) ) ] - 1 ) ] );
          r1z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 1, ( iel + 7 ) ) ] - 1 ) ] );
          r2z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 2, ( iel + 7 ) ) ] - 1 ) ] );
          r3z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 3, ( iel + 7 ) ) ] - 1 ) ] );
          r4z = _mm256_setr_ps( rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 0 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 1 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 2 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 3 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 4 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 5 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 6 ) ) ] - 1 ) ]
                              , rg_gll_displacement[ 2 + 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, 4, ( iel + 7 ) ) ] - 1 ) ] );

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, k ) ] );

          auto duxdze = r0x * coeff;
          auto duydze = r0y * coeff;
          auto duzdze = r0z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, k ) ] );

          duxdze += r1x * coeff;
          duydze += r1y * coeff;
          duzdze += r1z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, k ) ] );

          duxdze += r2x * coeff;
          duydze += r2y * coeff;
          duzdze += r2z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, k ) ] );

          duxdze += r3x * coeff;
          duydze += r3y * coeff;
          duzdze += r3z * coeff;

          coeff = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, k ) ] );

          duxdze += r4x * coeff;
          duydze += r4y * coeff;
          duzdze += r4z * coeff;

          //
          auto dxidx = _mm256_setr_ps( rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dxidx[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto detdx = _mm256_setr_ps( rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_detdx[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto dzedx = _mm256_setr_ps( rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dzedx[ IDX4( m, l, k, ( iel + 7 ) ) ] );

          auto duxdx = duxdxi * dxidx + duxdet * detdx + duxdze * dzedx;
          auto duydx = duydxi * dxidx + duydet * detdx + duydze * dzedx;
          auto duzdx = duzdxi * dxidx + duzdet * detdx + duzdze * dzedx;

          auto dxidy = _mm256_setr_ps( rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dxidy[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto detdy = _mm256_setr_ps( rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_detdy[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto dzedy = _mm256_setr_ps( rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dzedy[ IDX4( m, l, k, ( iel + 7 ) ) ] );

          auto duxdy = duxdxi * dxidy + duxdet * detdy + duxdze * dzedy;
          auto duydy = duydxi * dxidy + duydet * detdy + duydze * dzedy;
          auto duzdy = duzdxi * dxidy + duzdet * detdy + duzdze * dzedy;

          auto dxidz = _mm256_setr_ps( rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dxidz[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto detdz = _mm256_setr_ps( rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_detdz[ IDX4( m, l, k, ( iel + 7 ) ) ] );
          auto dzedz = _mm256_setr_ps( rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 0 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 1 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 2 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 3 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 4 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 5 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 6 ) ) ]
                                     , rg_hexa_gll_dzedz[ IDX4( m, l, k, ( iel + 7 ) ) ] );

          auto duxdz = duxdxi * dxidz + duxdet * detdz + duxdze * dzedz;
          auto duydz = duydxi * dxidz + duydet * detdz + duydze * dzedz;
          auto duzdz = duzdxi * dxidz + duzdet * detdz + duzdze * dzedz;

          auto rhovp2 = _mm256_setr_ps( rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 0 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 1 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 2 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 3 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 4 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 5 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 6 ) ]
                                      , rg_hexa_gll_rhovp2[ IDX4( m, l, k, iel + 7 ) ] );

          auto rhovs2 = _mm256_setr_ps( rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 0 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 1 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 2 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 3 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 4 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 5 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 6 ) ]
                                      , rg_hexa_gll_rhovs2[ IDX4( m, l, k, iel + 7 ) ] );

          auto trace_tau = ( rhovp2 - _mm256_set1_ps( 2.0f ) * rhovs2 )*(duxdx+duydy+duzdz);
          auto tauxx     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duxdx;
          auto tauyy     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duydy;
          auto tauzz     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duzdz;
          auto tauxy     =                 rhovs2*(duxdy+duydx);
          auto tauxz     =                 rhovs2*(duxdz+duzdx);
          auto tauyz     =                 rhovs2*(duydz+duzdy);

          auto tmp = _mm256_setr_ps( rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 0 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 1 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 2 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 3 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 4 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 5 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 6 ) ]
                                   , rg_hexa_gll_jacobian_det[ IDX4( m, l, k, iel + 7 ) ] );

          intpx1[ IDX3( m, l, k ) ] = tmp * (tauxx*dxidx+tauxy*dxidy+tauxz*dxidz);
          intpx2[ IDX3( m, l, k ) ] = tmp * (tauxx*detdx+tauxy*detdy+tauxz*detdz);
          intpx3[ IDX3( m, l, k ) ] = tmp * (tauxx*dzedx+tauxy*dzedy+tauxz*dzedz);

          intpy1[ IDX3( m, l, k ) ] = tmp * (tauxy*dxidx+tauyy*dxidy+tauyz*dxidz);
          intpy2[ IDX3( m, l, k ) ] = tmp * (tauxy*detdx+tauyy*detdy+tauyz*detdz);
          intpy3[ IDX3( m, l, k ) ] = tmp * (tauxy*dzedx+tauyy*dzedy+tauyz*dzedz);

          intpz1[ IDX3( m, l, k ) ] = tmp * (tauxz*dxidx+tauyz*dxidy+tauzz*dxidz);
          intpz2[ IDX3( m, l, k ) ] = tmp * (tauxz*detdx+tauyz*detdy+tauzz*detdz);
          intpz3[ IDX3( m, l, k ) ] = tmp * (tauxz*dzedx+tauyz*dzedy+tauzz*dzedz);
        }
      }
    }

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx1 = intpx1[ IDX3( 0, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpx1[ IDX3( 1, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpx1[ IDX3( 2, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpx1[ IDX3( 3, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpx1[ IDX3( 4, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpy1 = intpy1[ IDX3( 0, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpy1[ IDX3( 1, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpy1[ IDX3( 2, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpy1[ IDX3( 3, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpy1[ IDX3( 4, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpz1 = intpz1[ IDX3( 0, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpz1[ IDX3( 1, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpz1[ IDX3( 2, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpz1[ IDX3( 3, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpz1[ IDX3( 4, l, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpx2 = intpx2[ IDX3( m, 0, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpx2[ IDX3( m, 1, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpx2[ IDX3( m, 2, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpx2[ IDX3( m, 3, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpx2[ IDX3( m, 4, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpy2 = intpy2[ IDX3( m, 0, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpy2[ IDX3( m, 1, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpy2[ IDX3( m, 2, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpy2[ IDX3( m, 3, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpy2[ IDX3( m, 4, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpz2 = intpz2[ IDX3( m, 0, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpz2[ IDX3( m, 1, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpz2[ IDX3( m, 2, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpz2[ IDX3( m, 3, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpz2[ IDX3( m, 4, k ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpx3 = intpx3[ IDX3( m, l, 0 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpx3[ IDX3( m, l, 1 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpx3[ IDX3( m, l, 2 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpx3[ IDX3( m, l, 3 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpx3[ IDX3( m, l, 4 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpy3 = intpy3[ IDX3( m, l, 0 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpy3[ IDX3( m, l, 1 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpy3[ IDX3( m, l, 2 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpy3[ IDX3( m, l, 3 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpy3[ IDX3( m, l, 4 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto tmpz3 = intpz3[ IDX3( m, l, 0 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 0 ] )
                     + intpz3[ IDX3( m, l, 1 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 1 ] )
                     + intpz3[ IDX3( m, l, 2 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 2 ] )
                     + intpz3[ IDX3( m, l, 3 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 3 ] )
                     + intpz3[ IDX3( m, l, 4 ) ] * _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] ) * _mm256_set1_ps( rg_gll_weight[ 4 ] );

          auto fac1 = _mm256_set1_ps( rg_gll_weight[ l ] ) * _mm256_set1_ps( rg_gll_weight[ k ] );
          auto fac2 = _mm256_set1_ps( rg_gll_weight[ m ] ) * _mm256_set1_ps( rg_gll_weight[ k ] );
          auto fac3 = _mm256_set1_ps( rg_gll_weight[ m ] ) * _mm256_set1_ps( rg_gll_weight[ l ] );

          auto rx = fac1 * tmpx1 + fac2 * tmpx2 + fac3 * tmpx3;
          auto ry = fac1 * tmpy1 + fac2 * tmpy2 + fac3 * tmpy3;
          auto rz = fac1 * tmpz1 + fac2 * tmpz2 + fac3 * tmpz3;

          float tx[ 8 ] __attribute__((aligned(32)));//alignas(32); // ICC !!!
          float ty[ 8 ] __attribute__((aligned(32)));//alignas(32);
          float tz[ 8 ] __attribute__((aligned(32)));//alignas(32);

          _mm256_store_ps( tx, rx );
          _mm256_store_ps( ty, ry );
          _mm256_store_ps( tz, rz );

          for( std::size_t i = 0 ; i < 8 ; ++i )
          {
            auto idx = ig_hexa_gll_glonum[ IDX4( m, l, k, iel + i ) ] - 1;

            rg_gll_acceleration[ 0 + 3 * idx ] -= tx[ i ];
            rg_gll_acceleration[ 1 + 3 * idx ] -= ty[ i ];
            rg_gll_acceleration[ 2 + 3 * idx ] -= tz[ i ];
          }

        }
      }
    }
  }
}
