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
  __m256 rl_acceleration_gll[5*5*5*3];
  __m256 rl_dxi[5*5*5*3];
  __m256 rl_det[5*5*5*3];
  __m256 rl_dze[5*5*5*3];

  __m256 intpx1[ 125 ];
  __m256 intpy1[ 125 ];
  __m256 intpz1[ 125 ];
  __m256 intpx2[ 125 ];
  __m256 intpy2[ 125 ];
  __m256 intpz2[ 125 ];
  __m256 intpx3[ 125 ];
  __m256 intpy3[ 125 ];
  __m256 intpz3[ 125 ];

  __m256 intpX[ 125 * 3 ];
  __m256 intpY[ 125 * 3 ];
  __m256 intpZ[ 125 * 3 ];

  float ldw[ 5 ][ 5 ];
  float w2[ 5 ][ 5 ];

  uint32_t glonum[ 5 * 5 * 5 ][ 8 ];

  // Precompute some values.
  for( std::size_t i = 0 ; i < 5 ; ++i )
  {
    ldw[ i ][ 0 ] = rg_gll_lagrange_deriv[ IDX2( i, 0 ) ] * rg_gll_weight[ 0 ];
    ldw[ i ][ 1 ] = rg_gll_lagrange_deriv[ IDX2( i, 1 ) ] * rg_gll_weight[ 1 ];
    ldw[ i ][ 2 ] = rg_gll_lagrange_deriv[ IDX2( i, 2 ) ] * rg_gll_weight[ 2 ];
    ldw[ i ][ 3 ] = rg_gll_lagrange_deriv[ IDX2( i, 3 ) ] * rg_gll_weight[ 3 ];
    ldw[ i ][ 4 ] = rg_gll_lagrange_deriv[ IDX2( i, 4 ) ] * rg_gll_weight[ 4 ];
  }

  for( std::size_t i = 0 ; i < 5 ; ++i )
  {
    for( std::size_t j = 0 ; j < 5 ; ++j )
    {
      w2[ i ][ j ] = rg_gll_weight[ i ] * rg_gll_weight[ j ];
    }
  }

  // Main loop.
  for( std::size_t iel = elt_start ; iel < elt_end ; iel+=8 )
  {

    // Reset local acceleration array.
    for( std::size_t j = 0 ; j < 375 ; ++j )
    {
      rl_acceleration_gll[ j ] = _mm256_setzero_ps();
    }


    asm("#Copy global -> local displacement values.");
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto const dst = 3 * IDX3( m, l, k );

          for( std::size_t i = 0 ; i < 8 ; ++i )
          {
            auto const src = 3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, ( iel + i ) ) ] - 1 );
            glonum[ IDX3( m, l , k ) ][ i ] = src;
            ((float*)&rl_displacement_gll[ 0 + dst ])[ i ] = rg_gll_displacement[ 0 + src ];
            ((float*)&rl_displacement_gll[ 1 + dst ])[ i ] = rg_gll_displacement[ 1 + src ];
            ((float*)&rl_displacement_gll[ 2 + dst ])[ i ] = rg_gll_displacement[ 2 + src ];
          }
        }
      }
    }

#define SPLIT_2NDBLOCK

#ifdef SPLIT_2NDBLOCK
    asm("#Second block 0.");
    for( std::size_t m = 0 ; m < 5 ; ++m )
    {
      auto const coeff0 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, m ) ] );
      auto const coeff1 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, m ) ] );
      auto const coeff2 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, m ) ] );
      auto const coeff3 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, m ) ] );
      auto const coeff4 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, m ) ] );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        /*
        for( std::size_t l = 0 ; l < 5 ; ++l )
        {
          auto const index = 0 + 3 * IDX3( 0, l, k );

          auto duxdxi = rl_displacement_gll[ 0 + index ] * coeff0;
          auto duydxi = rl_displacement_gll[ 1 + index ] * coeff0;
          auto duzdxi = rl_displacement_gll[ 2 + index ] * coeff0;

          duxdxi += rl_displacement_gll[ 3 + index ] * coeff1;
          duydxi += rl_displacement_gll[ 4 + index ] * coeff1;
          duzdxi += rl_displacement_gll[ 5 + index ] * coeff1;

          duxdxi += rl_displacement_gll[ 6 + index ] * coeff2;
          duydxi += rl_displacement_gll[ 7 + index ] * coeff2;
          duzdxi += rl_displacement_gll[ 8 + index ] * coeff2;

          duxdxi += rl_displacement_gll[  9 + index ] * coeff3;
          duydxi += rl_displacement_gll[ 10 + index ] * coeff3;
          duzdxi += rl_displacement_gll[ 11 + index ] * coeff3;

          duxdxi += rl_displacement_gll[ 12 + index ] * coeff4;
          duydxi += rl_displacement_gll[ 13 + index ] * coeff4;
          duzdxi += rl_displacement_gll[ 14 + index ] * coeff4;

          rl_dxi[ 0 + 3 * IDX3( m, l, k ) ] = duxdxi;
          rl_dxi[ 1 + 3 * IDX3( m, l, k ) ] = duydxi;
          rl_dxi[ 2 + 3 * IDX3( m, l, k ) ] = duzdxi;
        }
        */

          auto const index0 = 0 + 3 * IDX3( 0, 0, k );

          auto duxdxi0 = rl_displacement_gll[ 0 + index0 ] * coeff0;
          auto duydxi0 = rl_displacement_gll[ 1 + index0 ] * coeff0;
          auto duzdxi0 = rl_displacement_gll[ 2 + index0 ] * coeff0;

          auto const index1 = 0 + 3 * IDX3( 0, 1, k );

          auto duxdxi1 = rl_displacement_gll[ 0 + index1 ] * coeff0;
          auto duydxi1 = rl_displacement_gll[ 1 + index1 ] * coeff0;
          auto duzdxi1 = rl_displacement_gll[ 2 + index1 ] * coeff0;

          auto const index2 = 0 + 3 * IDX3( 0, 2, k );

          auto duxdxi2 = rl_displacement_gll[ 0 + index2 ] * coeff0;
          auto duydxi2 = rl_displacement_gll[ 1 + index2 ] * coeff0;
          auto duzdxi2 = rl_displacement_gll[ 2 + index2 ] * coeff0;

          duxdxi0 += rl_displacement_gll[ 3 + index0 ] * coeff1;
          duydxi0 += rl_displacement_gll[ 4 + index0 ] * coeff1;
          duzdxi0 += rl_displacement_gll[ 5 + index0 ] * coeff1;

          duxdxi1 += rl_displacement_gll[ 3 + index1 ] * coeff1;
          duydxi1 += rl_displacement_gll[ 4 + index1 ] * coeff1;
          duzdxi1 += rl_displacement_gll[ 5 + index1 ] * coeff1;

          duxdxi2 += rl_displacement_gll[ 3 + index2 ] * coeff1;
          duydxi2 += rl_displacement_gll[ 4 + index2 ] * coeff1;
          duzdxi2 += rl_displacement_gll[ 5 + index2 ] * coeff1;

          duxdxi0 += rl_displacement_gll[ 6 + index0 ] * coeff2;
          duydxi0 += rl_displacement_gll[ 7 + index0 ] * coeff2;
          duzdxi0 += rl_displacement_gll[ 8 + index0 ] * coeff2;

          duxdxi1 += rl_displacement_gll[ 6 + index1 ] * coeff2;
          duydxi1 += rl_displacement_gll[ 7 + index1 ] * coeff2;
          duzdxi1 += rl_displacement_gll[ 8 + index1 ] * coeff2;

          duxdxi2 += rl_displacement_gll[ 6 + index2 ] * coeff2;
          duydxi2 += rl_displacement_gll[ 7 + index2 ] * coeff2;
          duzdxi2 += rl_displacement_gll[ 8 + index2 ] * coeff2;

          duxdxi0 += rl_displacement_gll[  9 + index0 ] * coeff3;
          duydxi0 += rl_displacement_gll[ 10 + index0 ] * coeff3;
          duzdxi0 += rl_displacement_gll[ 11 + index0 ] * coeff3;

          duxdxi1 += rl_displacement_gll[  9 + index1 ] * coeff3;
          duydxi1 += rl_displacement_gll[ 10 + index1 ] * coeff3;
          duzdxi1 += rl_displacement_gll[ 11 + index1 ] * coeff3;

          duxdxi2 += rl_displacement_gll[  9 + index2 ] * coeff3;
          duydxi2 += rl_displacement_gll[ 10 + index2 ] * coeff3;
          duzdxi2 += rl_displacement_gll[ 11 + index2 ] * coeff3;

          duxdxi0 += rl_displacement_gll[ 12 + index0 ] * coeff4;
          duydxi0 += rl_displacement_gll[ 13 + index0 ] * coeff4;
          duzdxi0 += rl_displacement_gll[ 14 + index0 ] * coeff4;

          duxdxi1 += rl_displacement_gll[ 12 + index1 ] * coeff4;
          duydxi1 += rl_displacement_gll[ 13 + index1 ] * coeff4;
          duzdxi1 += rl_displacement_gll[ 14 + index1 ] * coeff4;

          duxdxi2 += rl_displacement_gll[ 12 + index2 ] * coeff4;
          duydxi2 += rl_displacement_gll[ 13 + index2 ] * coeff4;
          duzdxi2 += rl_displacement_gll[ 14 + index2 ] * coeff4;

          rl_dxi[ 0 + 3 * IDX3( m, 0, k ) ] = duxdxi0;
          rl_dxi[ 1 + 3 * IDX3( m, 0, k ) ] = duydxi0;
          rl_dxi[ 2 + 3 * IDX3( m, 0, k ) ] = duzdxi0;

          rl_dxi[ 0 + 3 * IDX3( m, 1, k ) ] = duxdxi1;
          rl_dxi[ 1 + 3 * IDX3( m, 1, k ) ] = duydxi1;
          rl_dxi[ 2 + 3 * IDX3( m, 1, k ) ] = duzdxi1;

          rl_dxi[ 0 + 3 * IDX3( m, 2, k ) ] = duxdxi2;
          rl_dxi[ 1 + 3 * IDX3( m, 2, k ) ] = duydxi2;
          rl_dxi[ 2 + 3 * IDX3( m, 2, k ) ] = duzdxi2;


          auto const index3 = 0 + 3 * IDX3( 0, 3, k );

          auto duxdxi3 = rl_displacement_gll[ 0 + index3 ] * coeff0;
          auto duydxi3 = rl_displacement_gll[ 1 + index3 ] * coeff0;
          auto duzdxi3 = rl_displacement_gll[ 2 + index3 ] * coeff0;

          auto const index = 0 + 3 * IDX3( 0, 4, k );

          auto duxdxi4 = rl_displacement_gll[ 0 + index ] * coeff0;
          auto duydxi4 = rl_displacement_gll[ 1 + index ] * coeff0;
          auto duzdxi4 = rl_displacement_gll[ 2 + index ] * coeff0;
//


          duxdxi3 += rl_displacement_gll[ 3 + index3 ] * coeff1;
          duydxi3 += rl_displacement_gll[ 4 + index3 ] * coeff1;
          duzdxi3 += rl_displacement_gll[ 5 + index3 ] * coeff1;

          duxdxi4 += rl_displacement_gll[ 3 + index ] * coeff1;
          duydxi4 += rl_displacement_gll[ 4 + index ] * coeff1;
          duzdxi4 += rl_displacement_gll[ 5 + index ] * coeff1;
//


          duxdxi3 += rl_displacement_gll[ 6 + index3 ] * coeff2;
          duydxi3 += rl_displacement_gll[ 7 + index3 ] * coeff2;
          duzdxi3 += rl_displacement_gll[ 8 + index3 ] * coeff2;

          duxdxi4 += rl_displacement_gll[ 6 + index ] * coeff2;
          duydxi4 += rl_displacement_gll[ 7 + index ] * coeff2;
          duzdxi4 += rl_displacement_gll[ 8 + index ] * coeff2;

//


          duxdxi3 += rl_displacement_gll[  9 + index3 ] * coeff3;
          duydxi3 += rl_displacement_gll[ 10 + index3 ] * coeff3;
          duzdxi3 += rl_displacement_gll[ 11 + index3 ] * coeff3;

          duxdxi4 += rl_displacement_gll[  9 + index ] * coeff3;
          duydxi4 += rl_displacement_gll[ 10 + index ] * coeff3;
          duzdxi4 += rl_displacement_gll[ 11 + index ] * coeff3;
//


          duxdxi3 += rl_displacement_gll[ 12 + index3 ] * coeff4;
          duydxi3 += rl_displacement_gll[ 13 + index3 ] * coeff4;
          duzdxi3 += rl_displacement_gll[ 14 + index3 ] * coeff4;

          duxdxi4 += rl_displacement_gll[ 12 + index ] * coeff4;
          duydxi4 += rl_displacement_gll[ 13 + index ] * coeff4;
          duzdxi4 += rl_displacement_gll[ 14 + index ] * coeff4;

//

          rl_dxi[ 0 + 3 * IDX3( m, 3, k ) ] = duxdxi3;
          rl_dxi[ 1 + 3 * IDX3( m, 3, k ) ] = duydxi3;
          rl_dxi[ 2 + 3 * IDX3( m, 3, k ) ] = duzdxi3;

          rl_dxi[ 0 + 3 * IDX3( m, 4, k ) ] = duxdxi4;
          rl_dxi[ 1 + 3 * IDX3( m, 4, k ) ] = duydxi4;
          rl_dxi[ 2 + 3 * IDX3( m, 4, k ) ] = duzdxi4;
      }
    }

    asm("#Second block 1.");

    for( std::size_t l = 0 ; l < 5 ; ++l )
    {
      auto const coeff0 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, l ) ] );
      auto const coeff1 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, l ) ] );
      auto const coeff2 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, l ) ] );
      auto const coeff3 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, l ) ] );
      auto const coeff4 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, l ) ] );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto const index = 0 + 3 * IDX3( m, 0, k );

          auto duxdet = rl_displacement_gll[ 0 + index ] * coeff0;
          auto duydet = rl_displacement_gll[ 1 + index ] * coeff0;
          auto duzdet = rl_displacement_gll[ 2 + index ] * coeff0;

          duxdet += rl_displacement_gll[ 15 + index ] * coeff1;
          duydet += rl_displacement_gll[ 16 + index ] * coeff1;
          duzdet += rl_displacement_gll[ 17 + index ] * coeff1;

          duxdet += rl_displacement_gll[ 30 + index ] * coeff2;
          duydet += rl_displacement_gll[ 31 + index ] * coeff2;
          duzdet += rl_displacement_gll[ 32 + index ] * coeff2;

          duxdet += rl_displacement_gll[ 45 + index ] * coeff3;
          duydet += rl_displacement_gll[ 46 + index ] * coeff3;
          duzdet += rl_displacement_gll[ 47 + index ] * coeff3;

          duxdet += rl_displacement_gll[ 60 + index ] * coeff4;
          duydet += rl_displacement_gll[ 61 + index ] * coeff4;
          duzdet += rl_displacement_gll[ 62 + index ] * coeff4;

          rl_det[ 0 + 3 * IDX3( m, l, k ) ] = duxdet;
          rl_det[ 1 + 3 * IDX3( m, l, k ) ] = duydet;
          rl_det[ 2 + 3 * IDX3( m, l, k ) ] = duzdet;
        }
      }
    }

    asm("#Second block 2.");
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      auto const coeff0 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, k ) ] );
      auto const coeff1 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, k ) ] );
      auto const coeff2 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, k ) ] );
      auto const coeff3 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, k ) ] );
      auto const coeff4 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, k ) ] );

      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto duxdze = rl_displacement_gll[ 0 + 3 * IDX3( m, l, 0 ) ] * coeff0;
          auto duydze = rl_displacement_gll[ 1 + 3 * IDX3( m, l, 0 ) ] * coeff0;
          auto duzdze = rl_displacement_gll[ 2 + 3 * IDX3( m, l, 0 ) ] * coeff0;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 1 ) ] * coeff1;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 1 ) ] * coeff1;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 1 ) ] * coeff1;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 2 ) ] * coeff2;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 2 ) ] * coeff2;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 2 ) ] * coeff2;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 3 ) ] * coeff3;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 3 ) ] * coeff3;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 3 ) ] * coeff3;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 4 ) ] * coeff4;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 4 ) ] * coeff4;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 4 ) ] * coeff4;

          rl_dze[ 0 + 3 * IDX3( m, l, k ) ] = duxdze;
          rl_dze[ 1 + 3 * IDX3( m, l, k ) ] = duydze;
          rl_dze[ 2 + 3 * IDX3( m, l, k ) ] = duzdze;
        }
      }
    }

#else

    asm("#Second block partial fusion.");
    for( std::size_t i = 0 ; i < 5 ; ++i )
    {
      auto const coeff0 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 0, i ) ] );
      auto const coeff1 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 1, i ) ] );
      auto const coeff2 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 2, i ) ] );
      auto const coeff3 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 3, i ) ] );
      auto const coeff4 = _mm256_set1_ps( rg_gll_lagrange_deriv[ IDX2( 4, i ) ] );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t l = 0 ; l < 5 ; ++l )
        {
          auto const index = 0 + 3 * IDX3( 0, l, k );

          auto duxdxi = rl_displacement_gll[ 0 + index ] * coeff0;
          auto duydxi = rl_displacement_gll[ 1 + index ] * coeff0;
          auto duzdxi = rl_displacement_gll[ 2 + index ] * coeff0;

          duxdxi += rl_displacement_gll[ 3 + index ] * coeff1;
          duydxi += rl_displacement_gll[ 4 + index ] * coeff1;
          duzdxi += rl_displacement_gll[ 5 + index ] * coeff1;

          duxdxi += rl_displacement_gll[ 6 + index ] * coeff2;
          duydxi += rl_displacement_gll[ 7 + index ] * coeff2;
          duzdxi += rl_displacement_gll[ 8 + index ] * coeff2;

          duxdxi += rl_displacement_gll[  9 + index ] * coeff3;
          duydxi += rl_displacement_gll[ 10 + index ] * coeff3;
          duzdxi += rl_displacement_gll[ 11 + index ] * coeff3;

          duxdxi += rl_displacement_gll[ 12 + index ] * coeff4;
          duydxi += rl_displacement_gll[ 13 + index ] * coeff4;
          duzdxi += rl_displacement_gll[ 14 + index ] * coeff4;

          rl_dxi[ 0 + 3 * IDX3( i, l, k ) ] = duxdxi;
          rl_dxi[ 1 + 3 * IDX3( i, l, k ) ] = duydxi;
          rl_dxi[ 2 + 3 * IDX3( i, l, k ) ] = duzdxi;
        }
      }
      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto const index = 0 + 3 * IDX3( m, 0, k );

          auto duxdet = rl_displacement_gll[ 0 + index ] * coeff0;
          auto duydet = rl_displacement_gll[ 1 + index ] * coeff0;
          auto duzdet = rl_displacement_gll[ 2 + index ] * coeff0;

          duxdet += rl_displacement_gll[ 15 + index ] * coeff1;
          duydet += rl_displacement_gll[ 16 + index ] * coeff1;
          duzdet += rl_displacement_gll[ 17 + index ] * coeff1;

          duxdet += rl_displacement_gll[ 30 + index ] * coeff2;
          duydet += rl_displacement_gll[ 31 + index ] * coeff2;
          duzdet += rl_displacement_gll[ 32 + index ] * coeff2;

          duxdet += rl_displacement_gll[ 45 + index ] * coeff3;
          duydet += rl_displacement_gll[ 46 + index ] * coeff3;
          duzdet += rl_displacement_gll[ 47 + index ] * coeff3;

          duxdet += rl_displacement_gll[ 60 + index ] * coeff4;
          duydet += rl_displacement_gll[ 61 + index ] * coeff4;
          duzdet += rl_displacement_gll[ 62 + index ] * coeff4;

          rl_det[ 0 + 3 * IDX3( m, i, k ) ] = duxdet;
          rl_det[ 1 + 3 * IDX3( m, i, k ) ] = duydet;
          rl_det[ 2 + 3 * IDX3( m, i, k ) ] = duzdet;
        }
      }

      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto duxdze = rl_displacement_gll[ 0 + 3 * IDX3( m, l, 0 ) ] * coeff0;
          auto duydze = rl_displacement_gll[ 1 + 3 * IDX3( m, l, 0 ) ] * coeff0;
          auto duzdze = rl_displacement_gll[ 2 + 3 * IDX3( m, l, 0 ) ] * coeff0;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 1 ) ] * coeff1;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 1 ) ] * coeff1;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 1 ) ] * coeff1;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 2 ) ] * coeff2;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 2 ) ] * coeff2;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 2 ) ] * coeff2;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 3 ) ] * coeff3;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 3 ) ] * coeff3;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 3 ) ] * coeff3;

          duxdze += rl_displacement_gll[ 0 + 3 * IDX3( m, l, 4 ) ] * coeff4;
          duydze += rl_displacement_gll[ 1 + 3 * IDX3( m, l, 4 ) ] * coeff4;
          duzdze += rl_displacement_gll[ 2 + 3 * IDX3( m, l, 4 ) ] * coeff4;

          rl_dze[ 0 + 3 * IDX3( m, l, i ) ] = duxdze;
          rl_dze[ 1 + 3 * IDX3( m, l, i ) ] = duydze;
          rl_dze[ 2 + 3 * IDX3( m, l, i ) ] = duzdze;
        }
      }

    }


#endif




    asm("#Second block, part 2.");

    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          /*
          auto const i0 = 3 * IDX3( m, l, k );
          auto const i1 = (ig_hexa_gll_glonum.size()/125) * IDX3( m, l, k ) + iel;

          auto const dxidx = _mm256_load_ps( &( rg_hexa_gll_dxidx[ i1 ] ) );
          auto const detdx = _mm256_load_ps( &( rg_hexa_gll_detdx[ i1 ] ) );
          auto const dzedx = _mm256_load_ps( &( rg_hexa_gll_dzedx[ i1 ] ) );

          auto const duxdx = rl_dxi[ 0 + i0 ] * dxidx + rl_det[ 0 + i0 ] * detdx + rl_dze[ 0 + i0] * dzedx;
          auto const duydx = rl_dxi[ 1 + i0 ] * dxidx + rl_det[ 1 + i0 ] * detdx + rl_dze[ 1 + i0] * dzedx;
          auto const duzdx = rl_dxi[ 2 + i0 ] * dxidx + rl_det[ 2 + i0 ] * detdx + rl_dze[ 2 + i0] * dzedx;

          auto const dxidy = _mm256_load_ps( &( rg_hexa_gll_dxidy[ i1 ] ) );
          auto const detdy = _mm256_load_ps( &( rg_hexa_gll_detdy[ i1 ] ) );
          auto const dzedy = _mm256_load_ps( &( rg_hexa_gll_dzedy[ i1 ] ) );

          auto const duxdy = rl_dxi[ 0 + i0 ] * dxidy + rl_det[ 0 + i0 ] * detdy + rl_dze[ 0 + i0 ] * dzedy;
          auto const duydy = rl_dxi[ 1 + i0 ] * dxidy + rl_det[ 1 + i0 ] * detdy + rl_dze[ 1 + i0 ] * dzedy;
          auto const duzdy = rl_dxi[ 2 + i0 ] * dxidy + rl_det[ 2 + i0 ] * detdy + rl_dze[ 2 + i0 ] * dzedy;

          auto const dxidz = _mm256_load_ps( &( rg_hexa_gll_dxidz[ i1 ] ) );
          auto const detdz = _mm256_load_ps( &( rg_hexa_gll_detdz[ i1 ] ) );
          auto const dzedz = _mm256_load_ps( &( rg_hexa_gll_dzedz[ i1 ] ) );

          auto const duxdz = rl_dxi[ 0 + i0 ] * dxidz + rl_det[ 0 + i0 ] * detdz + rl_dze[ 0 + i0 ] * dzedz;
          auto const duydz = rl_dxi[ 1 + i0 ] * dxidz + rl_det[ 1 + i0 ] * detdz + rl_dze[ 1 + i0 ] * dzedz;
          auto const duzdz = rl_dxi[ 2 + i0 ] * dxidz + rl_det[ 2 + i0 ] * detdz + rl_dze[ 2 + i0 ] * dzedz;

          //

          auto const rhovp2 = _mm256_load_ps( &(rg_hexa_gll_rhovp2[ i1 ]) );
          auto const rhovs2 = _mm256_load_ps( &(rg_hexa_gll_rhovs2[ i1 ]) );

          auto const trace_tau = ( rhovp2 - 2.0 * rhovs2 )*(duxdx+duydy+duzdz);
          auto const tauxx     = trace_tau + 2.0*rhovs2*duxdx;
          auto const tauyy     = trace_tau + 2.0*rhovs2*duydy;
          auto const tauzz     = trace_tau + 2.0*rhovs2*duzdz;
          auto const tauxy     =                 rhovs2*(duxdy+duydx);
          auto const tauxz     =                 rhovs2*(duxdz+duzdx);
          auto const tauyz     =                 rhovs2*(duydz+duzdy);

          auto const tmp = _mm256_load_ps( &( rg_hexa_gll_jacobian_det[ i1 ] ) );

          intpx1[ IDX3( m, l, k ) ] = tmp * (tauxx*dxidx+tauxy*dxidy+tauxz*dxidz);
          intpx2[ IDX3( m, l, k ) ] = tmp * (tauxx*detdx+tauxy*detdy+tauxz*detdz);
          intpx3[ IDX3( m, l, k ) ] = tmp * (tauxx*dzedx+tauxy*dzedy+tauxz*dzedz);

          intpy1[ IDX3( m, l, k ) ] = tmp * (tauxy*dxidx+tauyy*dxidy+tauyz*dxidz);
          intpy2[ IDX3( m, l, k ) ] = tmp * (tauxy*detdx+tauyy*detdy+tauyz*detdz);
          intpy3[ IDX3( m, l, k ) ] = tmp * (tauxy*dzedx+tauyy*dzedy+tauyz*dzedz);

          intpz1[ IDX3( m, l, k ) ] = tmp * (tauxz*dxidx+tauyz*dxidy+tauzz*dxidz);
          intpz2[ IDX3( m, l, k ) ] = tmp * (tauxz*detdx+tauyz*detdy+tauzz*detdz);
          intpz3[ IDX3( m, l, k ) ] = tmp * (tauxz*dzedx+tauyz*dzedy+tauzz*dzedz);
          */

          auto const i0 = 3 * IDX3( m, l, k );
          //auto const i1 = (ig_hexa_gll_glonum.size()/125) * IDX3( m, l, k ) + iel;
          // NEW SOA FORMAT
          auto const i1 = iel * 125 + 8 * IDX3( m, l, k );

          auto const dxidx = _mm256_load_ps( &( rg_hexa_gll_dxidx[ i1 ] ) );

          auto duxdx = rl_dxi[ 0 + i0 ] * dxidx;
          auto duydx = rl_dxi[ 1 + i0 ] * dxidx;
          auto duzdx = rl_dxi[ 2 + i0 ] * dxidx;

          auto const detdx = _mm256_load_ps( &( rg_hexa_gll_detdx[ i1 ] ) );

          duxdx += rl_det[ 0 + i0 ] * detdx;
          duydx += rl_det[ 1 + i0 ] * detdx;
          duzdx += rl_det[ 2 + i0 ] * detdx;

          auto const dzedx = _mm256_load_ps( &( rg_hexa_gll_dzedx[ i1 ] ) );

          duxdx += rl_dze[ 0 + i0] * dzedx;
          duydx += rl_dze[ 1 + i0] * dzedx;
          duzdx += rl_dze[ 2 + i0] * dzedx;

          auto const dxidy = _mm256_load_ps( &( rg_hexa_gll_dxidy[ i1 ] ) );
          auto const detdy = _mm256_load_ps( &( rg_hexa_gll_detdy[ i1 ] ) );
          auto const dzedy = _mm256_load_ps( &( rg_hexa_gll_dzedy[ i1 ] ) );

          auto const duxdy = rl_dxi[ 0 + i0 ] * dxidy + rl_det[ 0 + i0 ] * detdy + rl_dze[ 0 + i0 ] * dzedy;
          auto const duydy = rl_dxi[ 1 + i0 ] * dxidy + rl_det[ 1 + i0 ] * detdy + rl_dze[ 1 + i0 ] * dzedy;
          auto const duzdy = rl_dxi[ 2 + i0 ] * dxidy + rl_det[ 2 + i0 ] * detdy + rl_dze[ 2 + i0 ] * dzedy;

          auto const dxidz = _mm256_load_ps( &( rg_hexa_gll_dxidz[ i1 ] ) );
          auto const detdz = _mm256_load_ps( &( rg_hexa_gll_detdz[ i1 ] ) );
          auto const dzedz = _mm256_load_ps( &( rg_hexa_gll_dzedz[ i1 ] ) );

          auto const duxdz = rl_dxi[ 0 + i0 ] * dxidz + rl_det[ 0 + i0 ] * detdz + rl_dze[ 0 + i0 ] * dzedz;
          auto const duydz = rl_dxi[ 1 + i0 ] * dxidz + rl_det[ 1 + i0 ] * detdz + rl_dze[ 1 + i0 ] * dzedz;
          auto const duzdz = rl_dxi[ 2 + i0 ] * dxidz + rl_det[ 2 + i0 ] * detdz + rl_dze[ 2 + i0 ] * dzedz;

          //

          auto const rhovp2 = _mm256_load_ps( &(rg_hexa_gll_rhovp2[ i1 ]) );
          auto const rhovs2 = _mm256_load_ps( &(rg_hexa_gll_rhovs2[ i1 ]) );

          auto const trace_tau = ( rhovp2 - _mm256_set1_ps( 2.0f ) * rhovs2 )*(duxdx+duydy+duzdz);
          auto const tauxx     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duxdx;
          auto const tauxy     =                 rhovs2*(duxdy+duydx);
          auto const tauxz     =                 rhovs2*(duxdz+duzdx);

          auto const tmp = _mm256_load_ps( &( rg_hexa_gll_jacobian_det[ i1 ] ) );

          intpX[ 0 + 3 * IDX3( m, l, k ) ] = tmp * (tauxx*dxidx+tauxy*dxidy+tauxz*dxidz);
          intpX[ 1 + 3 * IDX3( m, l, k ) ] = tmp * (tauxx*detdx+tauxy*detdy+tauxz*detdz);
          intpX[ 2 + 3 * IDX3( m, l, k ) ] = tmp * (tauxx*dzedx+tauxy*dzedy+tauxz*dzedz);

          auto const tauyy     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duydy;
          auto const tauyz     =                 rhovs2*(duydz+duzdy);

          intpY[ 0 + 3 * IDX3( m, l, k ) ] = tmp * (tauxy*dxidx+tauyy*dxidy+tauyz*dxidz);
          intpY[ 1 + 3 * IDX3( m, l, k ) ] = tmp * (tauxy*detdx+tauyy*detdy+tauyz*detdz);
          intpY[ 2 + 3 * IDX3( m, l, k ) ] = tmp * (tauxy*dzedx+tauyy*dzedy+tauyz*dzedz);

          auto const tauzz     = trace_tau + _mm256_set1_ps( 2.0f )*rhovs2*duzdz;

          intpZ[ 0 + 3 * IDX3( m, l, k ) ] = tmp * (tauxz*dxidx+tauyz*dxidy+tauzz*dxidz);
          intpZ[ 1 + 3 * IDX3( m, l, k ) ] = tmp * (tauxz*detdx+tauyz*detdy+tauzz*detdz);
          intpZ[ 2 + 3 * IDX3( m, l, k ) ] = tmp * (tauxz*dzedx+tauyz*dzedy+tauzz*dzedz);
        }
      }
    }





#define SPLIT

#ifdef SPLIT
    asm("#Third block 0.");
    for( std::size_t m = 0 ; m < 5 ; ++m )
    {
      auto m0 = _mm256_set1_ps( ldw[ m ][ 0 ]/*rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] * rg_gll_weight[ 0 ]*/ );
      auto m1 = _mm256_set1_ps( ldw[ m ][ 1 ] /*rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] * rg_gll_weight[ 1 ]*/ );
      auto m2 = _mm256_set1_ps( ldw[ m ][ 2 ] /*rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] * rg_gll_weight[ 2 ]*/ );
      auto m3 = _mm256_set1_ps( ldw[ m ][ 3 ] /*rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] * rg_gll_weight[ 3 ]*/ );
      auto m4 = _mm256_set1_ps( ldw[ m ][ 4 ] /*rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] * rg_gll_weight[ 4 ]*/ );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t l = 0 ; l < 5 ; ++l )
        {
          auto tmpx1 = intpX[ 0 + 3 * IDX3( 0, l, k ) ] * m0
                     + intpX[ 0 + 3 * IDX3( 1, l, k ) ] * m1
                     + intpX[ 0 + 3 * IDX3( 2, l, k ) ] * m2
                     + intpX[ 0 + 3 * IDX3( 3, l, k ) ] * m3
                     + intpX[ 0 + 3 * IDX3( 4, l, k ) ] * m4;

          auto tmpy1 = intpY[ 0 + 3 * IDX3( 0, l, k ) ] * m0
                     + intpY[ 0 + 3 * IDX3( 1, l, k ) ] * m1
                     + intpY[ 0 + 3 * IDX3( 2, l, k ) ] * m2
                     + intpY[ 0 + 3 * IDX3( 3, l, k ) ] * m3
                     + intpY[ 0 + 3 * IDX3( 4, l, k ) ] * m4;

          auto tmpz1 = intpZ[ 0 + 3 * IDX3( 0, l, k ) ] * m0
                     + intpZ[ 0 + 3 * IDX3( 1, l, k ) ] * m1
                     + intpZ[ 0 + 3 * IDX3( 2, l, k ) ] * m2
                     + intpZ[ 0 + 3 * IDX3( 3, l, k ) ] * m3
                     + intpZ[ 0 + 3 * IDX3( 4, l, k ) ] * m4;

          auto fac1 = _mm256_set1_ps( w2[ k ][ l ] /*rg_gll_weight[ l ] * rg_gll_weight[ k ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( m, l, k ) ] = fac1 * tmpx1;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, l, k ) ] = fac1 * tmpy1;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, l, k ) ] = fac1 * tmpz1;
        }
      }
    }

asm("#Third block 1.");
    for( std::size_t l = 0 ; l < 5 ; ++l )
    {
      auto l0 = _mm256_set1_ps( ldw[ l ][ 0 ]/*rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] * rg_gll_weight[ 0 ]*/ );
      auto l1 = _mm256_set1_ps( ldw[ l ][ 1 ]/*rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] * rg_gll_weight[ 1 ]*/ );
      auto l2 = _mm256_set1_ps( ldw[ l ][ 2 ]/*rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] * rg_gll_weight[ 2 ]*/ );
      auto l3 = _mm256_set1_ps( ldw[ l ][ 3 ]/*rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] * rg_gll_weight[ 3 ]*/ );
      auto l4 = _mm256_set1_ps( ldw[ l ][ 4 ]/*rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] * rg_gll_weight[ 4 ]*/ );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx2 = intpX[ 1 + 3 * IDX3( m, 0, k ) ] * l0
                     + intpX[ 1 + 3 * IDX3( m, 1, k ) ] * l1
                     + intpX[ 1 + 3 * IDX3( m, 2, k ) ] * l2
                     + intpX[ 1 + 3 * IDX3( m, 3, k ) ] * l3
                     + intpX[ 1 + 3 * IDX3( m, 4, k ) ] * l4;

          auto tmpy2 = intpY[ 1 + 3 * IDX3( m, 0, k ) ] * l0
                     + intpY[ 1 + 3 * IDX3( m, 1, k ) ] * l1
                     + intpY[ 1 + 3 * IDX3( m, 2, k ) ] * l2
                     + intpY[ 1 + 3 * IDX3( m, 3, k ) ] * l3
                     + intpY[ 1 + 3 * IDX3( m, 4, k ) ] * l4;

          auto tmpz2 = intpZ[ 1 + 3 * IDX3( m, 0, k ) ] * l0
                     + intpZ[ 1 + 3 * IDX3( m, 1, k ) ] * l1
                     + intpZ[ 1 + 3 * IDX3( m, 2, k ) ] * l2
                     + intpZ[ 1 + 3 * IDX3( m, 3, k ) ] * l3
                     + intpZ[ 1 + 3 * IDX3( m, 4, k ) ] * l4;

          auto fac2 = _mm256_set1_ps( w2[ m ][ k ]/*rg_gll_weight[ m ] * rg_gll_weight[ k ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( m, l, k ) ] += fac2 * tmpx2;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, l, k ) ] += fac2 * tmpy2;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, l, k ) ] += fac2 * tmpz2;
        }
      }
    }

asm("#Third block 2.");
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      auto k0 = _mm256_set1_ps( ldw[ k ][ 0 ]/*rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] * rg_gll_weight[ 0 ]*/ );
      auto k1 = _mm256_set1_ps( ldw[ k ][ 1 ]/*rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] * rg_gll_weight[ 1 ]*/ );
      auto k2 = _mm256_set1_ps( ldw[ k ][ 2 ]/*rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] * rg_gll_weight[ 2 ]*/ );
      auto k3 = _mm256_set1_ps( ldw[ k ][ 3 ]/*rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] * rg_gll_weight[ 3 ]*/ );
      auto k4 = _mm256_set1_ps( ldw[ k ][ 4 ]/*rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] * rg_gll_weight[ 4 ]*/ );

      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx3 = intpX[ 2 + 3 * IDX3( m, l, 0 ) ] * k0
                     + intpX[ 2 + 3 * IDX3( m, l, 1 ) ] * k1
                     + intpX[ 2 + 3 * IDX3( m, l, 2 ) ] * k2
                     + intpX[ 2 + 3 * IDX3( m, l, 3 ) ] * k3
                     + intpX[ 2 + 3 * IDX3( m, l, 4 ) ] * k4;

          auto tmpy3 = intpY[ 2 + 3 * IDX3( m, l, 0 ) ] * k0
                     + intpY[ 2 + 3 * IDX3( m, l, 1 ) ] * k1
                     + intpY[ 2 + 3 * IDX3( m, l, 2 ) ] * k2
                     + intpY[ 2 + 3 * IDX3( m, l, 3 ) ] * k3
                     + intpY[ 2 + 3 * IDX3( m, l, 4 ) ] * k4;

          auto tmpz3 = intpZ[ 2 + 3 * IDX3( m, l, 0 ) ] * k0
                     + intpZ[ 2 + 3 * IDX3( m, l, 1 ) ] * k1
                     + intpZ[ 2 + 3 * IDX3( m, l, 2 ) ] * k2
                     + intpZ[ 2 + 3 * IDX3( m, l, 3 ) ] * k3
                     + intpZ[ 2 + 3 * IDX3( m, l, 4 ) ] * k4;

          auto fac3 = _mm256_set1_ps( w2[ l ][ m ] /*rg_gll_weight[ m ] * rg_gll_weight[ l ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( m, l, k ) ] += fac3 * tmpx3;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, l, k ) ] += fac3 * tmpy3;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, l, k ) ] += fac3 * tmpz3;

/*
          for( std::size_t i = 0 ; i < 8 ; ++i )
          {
            auto idx = ig_hexa_gll_glonum[ IDX4( m, l, k, iel + i ) ] - 1;

            rg_gll_acceleration[ 0 + 3 * idx ] -= rl_acceleration_gll[ 0 + 3 * IDX3( m, l, k ) ][ i ];
            rg_gll_acceleration[ 1 + 3 * idx ] -= rl_acceleration_gll[ 1 + 3 * IDX3( m, l, k ) ][ i ];
            rg_gll_acceleration[ 2 + 3 * idx ] -= rl_acceleration_gll[ 2 + 3 * IDX3( m, l, k ) ][ i ];
          }
*/
        }
      }
    }
#endif

#ifdef FUSION
    asm("#Fused loops.");
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      auto k0 = _mm256_set1_ps( ldw[ k ][ 0 ] );
      auto k1 = _mm256_set1_ps( ldw[ k ][ 1 ] );
      auto k2 = _mm256_set1_ps( ldw[ k ][ 2 ] );
      auto k3 = _mm256_set1_ps( ldw[ k ][ 3 ] );
      auto k4 = _mm256_set1_ps( ldw[ k ][ 4 ] );

      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto const fac = _mm256_set1_ps( rg_gll_weight[ m ] * rg_gll_weight[ l ] );

          auto tmpx1 = intpx1[ IDX3( 0, m, l ) ] * k0
                     + intpx1[ IDX3( 1, m, l ) ] * k1
                     + intpx1[ IDX3( 2, m, l ) ] * k2
                     + intpx1[ IDX3( 3, m, l ) ] * k3
                     + intpx1[ IDX3( 4, m, l ) ] * k4;

          auto tmpy1 = intpy1[ IDX3( 0, m, l ) ] * k0
                     + intpy1[ IDX3( 1, m, l ) ] * k1
                     + intpy1[ IDX3( 2, m, l ) ] * k2
                     + intpy1[ IDX3( 3, m, l ) ] * k3
                     + intpy1[ IDX3( 4, m, l ) ] * k4;

          auto tmpz1 = intpz1[ IDX3( 0, m, l ) ] * k0
                     + intpz1[ IDX3( 1, m, l ) ] * k1
                     + intpz1[ IDX3( 2, m, l ) ] * k2
                     + intpz1[ IDX3( 3, m, l ) ] * k3
                     + intpz1[ IDX3( 4, m, l ) ] * k4;

          rl_acceleration_gll[ 0 + 3 * IDX3( k, m, l ) ] += fac * tmpx1;
          rl_acceleration_gll[ 1 + 3 * IDX3( k, m, l ) ] += fac * tmpy1;
          rl_acceleration_gll[ 2 + 3 * IDX3( k, m, l ) ] += fac * tmpz1;

          auto tmpx2 = intpx2[ IDX3( m, 0, l ) ] * k0
                     + intpx2[ IDX3( m, 1, l ) ] * k1
                     + intpx2[ IDX3( m, 2, l ) ] * k2
                     + intpx2[ IDX3( m, 3, l ) ] * k3
                     + intpx2[ IDX3( m, 4, l ) ] * k4;

          auto tmpy2 = intpy2[ IDX3( m, 0, l ) ] * k0
                     + intpy2[ IDX3( m, 1, l ) ] * k1
                     + intpy2[ IDX3( m, 2, l ) ] * k2
                     + intpy2[ IDX3( m, 3, l ) ] * k3
                     + intpy2[ IDX3( m, 4, l ) ] * k4;

          auto tmpz2 = intpz2[ IDX3( m, 0, l ) ] * k0
                     + intpz2[ IDX3( m, 1, l ) ] * k1
                     + intpz2[ IDX3( m, 2, l ) ] * k2
                     + intpz2[ IDX3( m, 3, l ) ] * k3
                     + intpz2[ IDX3( m, 4, l ) ] * k4;

          rl_acceleration_gll[ 0 + 3 * IDX3( m, k, l ) ] += fac * tmpx2;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, k, l ) ] += fac * tmpy2;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, k, l ) ] += fac * tmpz2;

          auto tmpx3 = intpx3[ IDX3( m, l, 0 ) ] * k0
                     + intpx3[ IDX3( m, l, 1 ) ] * k1
                     + intpx3[ IDX3( m, l, 2 ) ] * k2
                     + intpx3[ IDX3( m, l, 3 ) ] * k3
                     + intpx3[ IDX3( m, l, 4 ) ] * k4;

          auto tmpy3 = intpy3[ IDX3( m, l, 0 ) ] * k0
                     + intpy3[ IDX3( m, l, 1 ) ] * k1
                     + intpy3[ IDX3( m, l, 2 ) ] * k2
                     + intpy3[ IDX3( m, l, 3 ) ] * k3
                     + intpy3[ IDX3( m, l, 4 ) ] * k4;

          auto tmpz3 = intpz3[ IDX3( m, l, 0 ) ] * k0
                     + intpz3[ IDX3( m, l, 1 ) ] * k1
                     + intpz3[ IDX3( m, l, 2 ) ] * k2
                     + intpz3[ IDX3( m, l, 3 ) ] * k3
                     + intpz3[ IDX3( m, l, 4 ) ] * k4;

          rl_acceleration_gll[ 0 + 3 * IDX3( m, l, k ) ] += fac * tmpx3;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, l, k ) ] += fac * tmpy3;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, l, k ) ] += fac * tmpz3;
        }
      }
    }

#endif

#ifdef PLF

    asm("#Partial loop fusion.");
    for( std::size_t i = 0 ; i < 5 ; ++i )
    {
      auto i0 = _mm256_set1_ps( ldw[ i ][ 0 ] );
      auto i1 = _mm256_set1_ps( ldw[ i ][ 1 ] );
      auto i2 = _mm256_set1_ps( ldw[ i ][ 2 ] );
      auto i3 = _mm256_set1_ps( ldw[ i ][ 3 ] );
      auto i4 = _mm256_set1_ps( ldw[ i ][ 4 ] );

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t l = 0 ; l < 5 ; ++l )
        {
          auto tmpx1 = intpx1[ IDX3( 0, l, k ) ] * i0
                     + intpx1[ IDX3( 1, l, k ) ] * i1
                     + intpx1[ IDX3( 2, l, k ) ] * i2
                     + intpx1[ IDX3( 3, l, k ) ] * i3
                     + intpx1[ IDX3( 4, l, k ) ] * i4;

          auto tmpy1 = intpy1[ IDX3( 0, l, k ) ] * i0
                     + intpy1[ IDX3( 1, l, k ) ] * i1
                     + intpy1[ IDX3( 2, l, k ) ] * i2
                     + intpy1[ IDX3( 3, l, k ) ] * i3
                     + intpy1[ IDX3( 4, l, k ) ] * i4;

          auto tmpz1 = intpz1[ IDX3( 0, l, k ) ] * i0
                     + intpz1[ IDX3( 1, l, k ) ] * i1
                     + intpz1[ IDX3( 2, l, k ) ] * i2
                     + intpz1[ IDX3( 3, l, k ) ] * i3
                     + intpz1[ IDX3( 4, l, k ) ] * i4;

          auto fac1 = _mm256_set1_ps( w2[ k ][ l ] /*rg_gll_weight[ l ] * rg_gll_weight[ k ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( i, l, k ) ] += fac1 * tmpx1;
          rl_acceleration_gll[ 1 + 3 * IDX3( i, l, k ) ] += fac1 * tmpy1;
          rl_acceleration_gll[ 2 + 3 * IDX3( i, l, k ) ] += fac1 * tmpz1;
        }
      }

      for( std::size_t k = 0 ; k < 5 ; ++k )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx2 = intpx2[ IDX3( m, 0, k ) ] * i0
                     + intpx2[ IDX3( m, 1, k ) ] * i1
                     + intpx2[ IDX3( m, 2, k ) ] * i2
                     + intpx2[ IDX3( m, 3, k ) ] * i3
                     + intpx2[ IDX3( m, 4, k ) ] * i4;

          auto tmpy2 = intpy2[ IDX3( m, 0, k ) ] * i0
                     + intpy2[ IDX3( m, 1, k ) ] * i1
                     + intpy2[ IDX3( m, 2, k ) ] * i2
                     + intpy2[ IDX3( m, 3, k ) ] * i3
                     + intpy2[ IDX3( m, 4, k ) ] * i4;

          auto tmpz2 = intpz2[ IDX3( m, 0, k ) ] * i0
                     + intpz2[ IDX3( m, 1, k ) ] * i1
                     + intpz2[ IDX3( m, 2, k ) ] * i2
                     + intpz2[ IDX3( m, 3, k ) ] * i3
                     + intpz2[ IDX3( m, 4, k ) ] * i4;

          auto fac2 = _mm256_set1_ps( w2[ m ][ k ]/*rg_gll_weight[ m ] * rg_gll_weight[ k ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( m, i, k ) ] += fac2 * tmpx2;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, i, k ) ] += fac2 * tmpy2;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, i, k ) ] += fac2 * tmpz2;
        }
      }

      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          auto tmpx3 = intpx3[ IDX3( m, l, 0 ) ] * i0
                     + intpx3[ IDX3( m, l, 1 ) ] * i1
                     + intpx3[ IDX3( m, l, 2 ) ] * i2
                     + intpx3[ IDX3( m, l, 3 ) ] * i3
                     + intpx3[ IDX3( m, l, 4 ) ] * i4;

          auto tmpy3 = intpy3[ IDX3( m, l, 0 ) ] * i0
                     + intpy3[ IDX3( m, l, 1 ) ] * i1
                     + intpy3[ IDX3( m, l, 2 ) ] * i2
                     + intpy3[ IDX3( m, l, 3 ) ] * i3
                     + intpy3[ IDX3( m, l, 4 ) ] * i4;

          auto tmpz3 = intpz3[ IDX3( m, l, 0 ) ] * i0
                     + intpz3[ IDX3( m, l, 1 ) ] * i1
                     + intpz3[ IDX3( m, l, 2 ) ] * i2
                     + intpz3[ IDX3( m, l, 3 ) ] * i3
                     + intpz3[ IDX3( m, l, 4 ) ] * i4;

          auto fac3 = _mm256_set1_ps( w2[ l ][ m ] /*rg_gll_weight[ m ] * rg_gll_weight[ l ]*/ );

          rl_acceleration_gll[ 0 + 3 * IDX3( m, l, i ) ] += fac3 * tmpx3;
          rl_acceleration_gll[ 1 + 3 * IDX3( m, l, i ) ] += fac3 * tmpy3;
          rl_acceleration_gll[ 2 + 3 * IDX3( m, l, i ) ] += fac3 * tmpz3;
        }
      }
    }
#endif

    asm("#Store values.");
    for( std::size_t k = 0 ; k < 5 ; ++k )
    {
      for( std::size_t l = 0 ; l < 5 ; ++l )
      {
        for( std::size_t m = 0 ; m < 5 ; ++m )
        {
          for( std::size_t i = 0 ; i < 8 ; ++i )
          {
            auto src = 3 * IDX3( m, l, k );
            auto dst = glonum[ IDX3( m, l, k ) ][ i ];//3 * ( ig_hexa_gll_glonum[ IDX4( m, l, k, iel + i ) ] - 1 );

            rg_gll_acceleration[ 0 + dst ] -= ((float*)&rl_acceleration_gll[ 0 + src ])[ i ];
            rg_gll_acceleration[ 1 + dst ] -= ((float*)&rl_acceleration_gll[ 1 + src ])[ i ];
            rg_gll_acceleration[ 2 + dst ] -= ((float*)&rl_acceleration_gll[ 2 + src ])[ i ];
          }
        }
      }
    }

  }
}
