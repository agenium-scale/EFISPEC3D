#include <cstddef>
#include <iostream>
#include <iomanip>

#include <arm_sve.h>

#include <cifo4.hpp>


#define IDX2( m, l ) ( 5 * l + m )
#define IDX3( m, l, k ) ( 25 * k + 5 * l + m )
#define IDX4( m, l, k, iel ) ( 125 * (iel) + 25 * (k) + 5 * (l) + (m) )


std::vector< uint32_t, boost::alignment::aligned_allocator< uint32_t, 32 > > ig_hexa_gll_glonum;

std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_displacement;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_weight;

std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_lagrange_deriv;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_acceleration;

std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidx;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidy;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidz;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdx;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdy;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdz;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedx;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedy;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedz;

std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_rhovp2;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_rhovs2;
std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_jacobian_det;


void print( svfloat32_t const & v )
{
  float tmp[ 64 ];
  svst1( svptrue_b32(), tmp, v );
  for( std::size_t i = 0 ; i < svcntw() ; ++i )
  {
    std::cout << tmp[ i ] << ' ';
  }
  std::cout << std::endl;
}

void print( svuint32_t const & v )
{
  std::uint32_t tmp[ 64 ];
  svst1( svptrue_b32(), tmp, v );
  for( std::size_t i = 0 ; i < svcntw() ; ++i )
  {
    std::cout << tmp[ i ] << ' ';
  }
  std::cout << std::endl;
}


void compute_internal_forces_order4( std::size_t elt_start, std::size_t elt_end )
{
  float rl_displacement_gll[5*5*5*3*svcntw() ];

  float local[ 5 * 5 * 5 * 9 * svcntw() ];

  float * intpx1 = &local[    0 ];
  float * intpy1 = &local[  125 * svcntw() ];
  float * intpz1 = &local[  250 * svcntw() ];

  float * intpx2 = &local[  375 * svcntw() ];
  float * intpy2 = &local[  500 * svcntw() ];
  float * intpz2 = &local[  625 * svcntw() ];

  float * intpx3 = &local[  750 * svcntw() ];
  float * intpy3 = &local[  875 * svcntw() ];
  float * intpz3 = &local[ 1000 * svcntw() ];

  auto vstrides = svindex_u32( 0u, 125u );

  for( std::size_t iel = elt_start ; iel < elt_end; iel += svcntw() )
    {
      svbool_t mask = svwhilelt_b32_u32( iel, elt_end );

      auto base = svadd_z( mask, vstrides, iel*125u );

      for( std::size_t k = 0 ; k < 5 ; ++k )
	{
	  for( std::size_t l = 0 ; l < 5 ; ++l )
	    {
	      for( std::size_t m = 0 ; m < 5 ; ++m )
		{
		  auto lid = IDX3( m, l, k ); // ++lid;
		  auto gids = svadd_z( mask, base, lid );

	          auto tmp = svld1_gather_index( mask, &ig_hexa_gll_glonum[ 0 ], gids );
		  auto ids = svmul_z( mask, svsub_z( mask, tmp, 1u ), 3u );

		//print(   svld1_gather_index( mask, &rg_gll_displacement[ 0 ], ids ) );

		  svst1( mask, &rl_displacement_gll[ svcntw() * (3 * lid + 0) ], svld1_gather_index( mask, &rg_gll_displacement[ 0 ], ids ) );
		  svst1( mask, &rl_displacement_gll[ svcntw() * (3 * lid + 1) ], svld1_gather_index( mask, &rg_gll_displacement[ 1 ], ids ) );
		  svst1( mask, &rl_displacement_gll[ svcntw() * (3 * lid + 2) ], svld1_gather_index( mask, &rg_gll_displacement[ 2 ], ids ) );
		}
	    }
	}
 /*
      for( std::size_t i = 0 ; i < 125*3*svcntw() ; ++i )
      {
        std::cout << rl_displacement_gll[ i ] << ' ';
      }
      std::cout << std::endl;
   */ 
      for( std::size_t k = 0 ; k < 5 ; ++k )
	{
	  for( std::size_t l = 0 ; l < 5 ; ++l )
	    {
	      for( std::size_t m = 0 ; m < 5 ; ++m )
		{
		  auto coeff = rg_gll_lagrange_deriv[ IDX2( 0, m ) ];

		  auto index = 0 + 3 * IDX3( 0, l, k );

		  auto duxdxi = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + index ) ] ), coeff );
		  auto duydxi = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + index ) ] ), coeff );
		  auto duzdxi = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + index ) ] ), coeff );

                  //print( duzdxi );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 1, m ) ];

		  duxdxi = svmla_z( mask, duxdxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 3 + index ) ] ), coeff );
		  duydxi = svmla_z( mask, duydxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 4 + index ) ] ), coeff );
		  duzdxi = svmla_z( mask, duzdxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 5 + index ) ] ), coeff );

                  //print( duzdxi );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 2, m ) ];

		  duxdxi = svmla_z( mask, duxdxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 6 + index ) ] ), coeff );
		  duydxi = svmla_z( mask, duydxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 7 + index ) ] ), coeff );
		  duzdxi = svmla_z( mask, duzdxi, svld1( mask, &rl_displacement_gll[  svcntw() * ( 8 + index ) ] ), coeff );

                  //print( duzdxi );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 3, m ) ];

		  duxdxi = svmla_z( mask, duxdxi, svld1( mask, &rl_displacement_gll[ svcntw() * (  9 + index ) ] ), coeff );
		  duydxi = svmla_z( mask, duydxi, svld1( mask, &rl_displacement_gll[ svcntw() * ( 10 + index ) ] ), coeff );
		  duzdxi = svmla_z( mask, duzdxi, svld1( mask, &rl_displacement_gll[ svcntw() * ( 11 + index ) ] ), coeff );

                  //print( duzdxi );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 4, m ) ];

		  duxdxi = svmla_z( mask, duxdxi, svld1( mask, &rl_displacement_gll[ svcntw() * ( 12 + index ) ] ), coeff );
		  duydxi = svmla_z( mask, duydxi, svld1( mask, &rl_displacement_gll[ svcntw() * ( 13 + index ) ] ), coeff );
		  duzdxi = svmla_z( mask, duzdxi, svld1( mask, &rl_displacement_gll[ svcntw() * ( 14 + index ) ] ), coeff );

                  //print( duzdxi );

                  //std::cout << std::endl;

		  //

		  coeff = rg_gll_lagrange_deriv[ IDX2( 0, l ) ];

		  auto duxdet = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, 0, k ) ) ] ), coeff );
		  auto duydet = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, 0, k ) ) ] ), coeff );
		  auto duzdet = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, 0, k ) ) ] ), coeff );

		  //print( duxdet );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 1, l ) ];

		  duxdet = svmla_z( mask, duxdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, 1, k ) ) ] ), coeff );
		  duydet = svmla_z( mask, duydet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, 1, k ) ) ] ), coeff );
		  duzdet = svmla_z( mask, duzdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, 1, k ) ) ] ), coeff );
		  
                  //print( duxdet );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 2, l ) ];

		  duxdet = svmla_z( mask, duxdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, 2, k ) ) ] ), coeff );
		  duydet = svmla_z( mask, duydet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, 2, k ) ) ] ), coeff );
		  duzdet = svmla_z( mask, duzdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, 2, k ) ) ] ), coeff );
		  
                  //print( duxdet );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 3, l ) ];

		  duxdet = svmla_z( mask, duxdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, 3, k ) ) ] ), coeff );
		  duydet = svmla_z( mask, duydet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, 3, k ) ) ] ), coeff );
		  duzdet = svmla_z( mask, duzdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, 3, k ) ) ] ), coeff );
		  
                  //print( duxdet );

		  coeff = rg_gll_lagrange_deriv[ IDX2( 4, l ) ];

		  duxdet = svmla_z( mask, duxdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, 4, k ) ) ] ), coeff );
		  duydet = svmla_z( mask, duydet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, 4, k ) ) ] ), coeff );
		  duzdet = svmla_z( mask, duzdet, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, 4, k ) ) ] ), coeff );
		  
                  //print( duxdet );

                  //std::cout << std::endl;
		  //

		  coeff = rg_gll_lagrange_deriv[ IDX2( 0, k ) ];

		  auto duxdze = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, l, 0 ) ) ] ), coeff );
		  auto duydze = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, l, 0 ) ) ] ), coeff );
		  auto duzdze = svmul_z( mask, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, l, 0 ) ) ] ), coeff );
		  //print( duxdet )

		  coeff = rg_gll_lagrange_deriv[ IDX2( 1, k ) ];

		  duxdze = svmla_z( mask, duxdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, l, 1 ) ) ] ), coeff );
		  duydze = svmla_z( mask, duydze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, l, 1 ) ) ] ), coeff );
		  duzdze = svmla_z( mask, duzdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, l, 1 ) ) ] ), coeff );
		  //print( duxdet )

		  coeff = rg_gll_lagrange_deriv[ IDX2( 2, k ) ];

		  duxdze = svmla_z( mask, duxdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, l, 2 ) ) ] ), coeff );
		  duydze = svmla_z( mask, duydze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, l, 2 ) ) ] ), coeff );
		  duzdze = svmla_z( mask, duzdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, l, 2 ) ) ] ), coeff );
		  //print( duxdet )

		  coeff = rg_gll_lagrange_deriv[ IDX2( 3, k ) ];

		  duxdze = svmla_z( mask, duxdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, l, 3 ) ) ] ), coeff );
		  duydze = svmla_z( mask, duydze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, l, 3 ) ) ] ), coeff );
		  duzdze = svmla_z( mask, duzdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, l, 3 ) ) ] ), coeff );
		  //print( duxdet )

		  coeff = rg_gll_lagrange_deriv[ IDX2( 4, k ) ];

		  duxdze = svmla_z( mask, duxdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 0 + 3 * IDX3( m, l, 4 ) ) ] ), coeff );
		  duydze = svmla_z( mask, duydze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 1 + 3 * IDX3( m, l, 4 ) ) ] ), coeff );
		  duzdze = svmla_z( mask, duzdze, svld1( mask, &rl_displacement_gll[ svcntw() * ( 2 + 3 * IDX3( m, l, 4 ) ) ] ), coeff );
		  //print( duxdet )



		  //
         
                  auto lid = IDX3( m, l, k );
		  auto id  = iel * 125 + lid;
	    
		  auto dxidx = svld1_gather_index( mask, &rg_hexa_gll_dxidx[ id ], vstrides );
		  auto detdx = svld1_gather_index( mask, &rg_hexa_gll_detdx[ id ], vstrides );
                  auto dzedx = svld1_gather_index( mask, &rg_hexa_gll_dzedx[ id ], vstrides );
		  
                  //print( dxidx );
	  
		  auto duxdx = svmla_z( mask, svmla_z( mask, svmul_z( mask, duxdxi, dxidx ), duxdet, detdx ), duxdze, dzedx );
		  auto duydx = svmla_z( mask, svmla_z( mask, svmul_z( mask, duydxi, dxidx ), duydet, detdx ), duydze, dzedx );
		  auto duzdx = svmla_z( mask, svmla_z( mask, svmul_z( mask, duzdxi, dxidx ), duzdet, detdx ), duzdze, dzedx );

		  //print( duxdx );
   
	          auto dxidy = svld1_gather_index( mask, &rg_hexa_gll_dxidy[ id ], vstrides );
	          auto detdy = svld1_gather_index( mask, &rg_hexa_gll_detdy[ id ], vstrides );
	          auto dzedy = svld1_gather_index( mask, &rg_hexa_gll_dzedy[ id ], vstrides );
		  //print( duxdet )
	  
		  auto duxdy = svmla_z( mask, svmla_z( mask, svmul_z( mask, duxdxi, dxidy ), duxdet, detdy ), duxdze, dzedy );
		  auto duydy = svmla_z( mask, svmla_z( mask, svmul_z( mask, duydxi, dxidy ), duydet, detdy ), duydze, dzedy );
		  auto duzdy = svmla_z( mask, svmla_z( mask, svmul_z( mask, duzdxi, dxidy ), duzdet, detdy ), duzdze, dzedy );
		  //print( duxdet )

	          auto dxidz = svld1_gather_index( mask, &rg_hexa_gll_dxidz[ id ], vstrides );
	          auto detdz = svld1_gather_index( mask, &rg_hexa_gll_detdz[ id ], vstrides );
	          auto dzedz = svld1_gather_index( mask, &rg_hexa_gll_dzedz[ id ], vstrides );
		  //print( duxdet )

	          auto duxdz = svmla_z( mask, svmla_z( mask, svmul_z( mask, duxdxi, dxidz ), duxdet, detdz ), duxdze, dzedz );
		  auto duydz = svmla_z( mask, svmla_z( mask, svmul_z( mask, duydxi, dxidz ), duydet, detdz ), duydze, dzedz );
		  auto duzdz = svmla_z( mask, svmla_z( mask, svmul_z( mask, duzdxi, dxidz ), duzdet, detdz ), duzdze, dzedz );
		  //print( duxdet )

                  //std::cout << std::endl;

                  //

	          auto rhovp2 = svld1_gather_index( mask, &rg_hexa_gll_rhovp2[ id ], vstrides );
		  auto rhovs2 = svld1_gather_index( mask, &rg_hexa_gll_rhovs2[ id ], vstrides );

                  //print( rhovp2 );
                  //std::cout << std::endl;

	          auto trace_tau = svmul_z( mask, svsub_z( mask, rhovp2, svmul_z( mask, rhovs2, 2.0f ) ), svadd_z( mask, svadd_z( mask, duxdx, duydy ), duzdz) );
		  auto tauxx     = svmla_z( mask, trace_tau ,rhovs2, svmul_z( mask, duxdx, 2.0f ) );
		  auto tauyy     = svmla_z( mask, trace_tau, rhovs2, svmul_z( mask, duydy, 2.0f ) );
		  auto tauzz     = svmla_z( mask, trace_tau, rhovs2, svmul_z( mask, duzdz, 2.0f ) );
		  auto tauxy     = svmul_z( mask, rhovs2, svadd_z( mask, duxdy, duydx ) );
		  auto tauxz     = svmul_z( mask, rhovs2, svadd_z( mask, duxdz, duzdx ) );
		  auto tauyz     = svmul_z( mask, rhovs2, svadd_z( mask, duydz, duzdy ) );

                  //print(  rhovp2 );
                  //print( rhovs2 );
                  //print( duxdx );
                  //print( duydy );
                  //print( duzdz );
	  	  //print( trace_tau );
                  //std::cout << std::endl;

  	          auto tmp = svld1_gather_index( mask, &rg_hexa_gll_jacobian_det[ id ], vstrides );

	          svst1( mask, &intpx1[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxx, dxidx ), tauxy, dxidy ), tauxz, dxidz ) ) );
		  svst1( mask, &intpx2[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxx, detdx ), tauxy, detdy ), tauxz, detdz ) ) );
		  svst1( mask, &intpx3[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxx, dzedx ), tauxy, dzedy ), tauxz, dzedz ) ) );

		  svst1( mask, &intpy1[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxy, dxidx ), tauyy, dxidy ), tauyz, dxidz ) ) );
		  svst1( mask, &intpy2[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxy, detdx ), tauyy, detdy ), tauyz, detdz ) ) );
		  svst1( mask, &intpy3[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxy, dzedx ), tauyy, dzedy ), tauyz, dzedz ) ) );

		  svst1( mask, &intpz1[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxz, dxidx ), tauyz, dxidy ), tauzz, dxidz ) ) );
		  svst1( mask, &intpz2[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxz, detdx ), tauyz, detdy ), tauzz, detdz ) ) );
		  svst1( mask, &intpz3[ svcntw() * lid ], svmul_z( mask, tmp, svmla_z( mask, svmla_z( mask, svmul_z( mask, tauxz, dzedx ), tauyz, dzedy ), tauzz, dzedz ) ) );
	/*	
                  for( std:: size_t i = 0 ; i < svcntw() ; ++i )
                  {
                    std::cout << intpz1[ lid + i ] << ' ';
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
		  auto c0 = rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] * rg_gll_weight[ 0 ];
		  auto c1 = rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] * rg_gll_weight[ 1 ];
		  auto c2 = rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] * rg_gll_weight[ 2 ];
		  auto c3 = rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] * rg_gll_weight[ 3 ];
		  auto c4 = rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] * rg_gll_weight[ 4 ];
		  
		  auto tmpx1 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpx1[ svcntw() * IDX3( 0, l, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpx1[ svcntw() * IDX3( 1, l, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpx1[ svcntw() * IDX3( 2, l, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpx1[ svcntw() * IDX3( 3, l, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpx1[ svcntw() * IDX3( 4, l, k ) ] ),
					c4
					);



		  auto tmpy1 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpy1[ svcntw() * IDX3( 0, l, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpy1[ svcntw() * IDX3( 1, l, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpy1[  svcntw() * IDX3( 2, l, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpy1[  svcntw() * IDX3( 3, l, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpy1[  svcntw() * IDX3( 4, l, k ) ] ),
					c4
					);

		  auto tmpz1 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpz1[  svcntw() * IDX3( 0, l, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpz1[  svcntw() * IDX3( 1, l, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpz1[  svcntw() * IDX3( 2, l, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpz1[  svcntw() * IDX3( 3, l, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpz1[  svcntw() * IDX3( 4, l, k ) ] ),
					c4
					);



		  c0 = rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] * rg_gll_weight[ 0 ];
		  c1 = rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] * rg_gll_weight[ 1 ];
		  c2 = rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] * rg_gll_weight[ 2 ];
		  c3 = rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] * rg_gll_weight[ 3 ];
		  c4 = rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] * rg_gll_weight[ 4 ];

		  auto tmpx2 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpx2[  svcntw() * IDX3( m, 0, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpx2[  svcntw() * IDX3( m, 1, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpx2[  svcntw() * IDX3( m, 2, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpx2[  svcntw() * IDX3( m, 3, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpx2[  svcntw() * IDX3( m, 4, k ) ] ),
					c4
					);

		  auto tmpy2 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpy2[  svcntw() * IDX3( m, 0, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpy2[  svcntw() * IDX3( m, 1, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpy2[  svcntw() * IDX3( m, 2, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpy2[  svcntw() * IDX3( m, 3, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpy2[  svcntw() * IDX3( m, 4, k ) ] ),
					c4
					);

		  auto tmpz2 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpz2[  svcntw() * IDX3( m, 0, k ) ] ),
									    c0
									    ),
								   svld1( mask, &intpz2[  svcntw() * IDX3( m, 1, k ) ] ),
								   c1
								   ),
							  svld1( mask, &intpz2[  svcntw() * IDX3( m, 2, k ) ] ),
							  c2
							  ),
						 svld1( mask, &intpz2[  svcntw() * IDX3( m, 3, k ) ] ),
						 c3
						 ),
					svld1( mask, &intpz2[  svcntw() * IDX3( m, 4, k ) ] ),
					c4
					);


		  c0 = rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] * rg_gll_weight[ 0 ];
		  c1 = rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] * rg_gll_weight[ 1 ];
		  c2 = rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] * rg_gll_weight[ 2 ];
		  c3 = rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] * rg_gll_weight[ 3 ];
		  c4 = rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] * rg_gll_weight[ 4 ];

		  auto tmpx3 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpx3[  svcntw() * IDX3( m, l, 0 ) ] ),
									    c0
									    ),
								   svld1( mask, &intpx3[  svcntw() * IDX3( m, l, 1 ) ] ),
								   c1
								   ),
							  svld1( mask, &intpx3[  svcntw() * IDX3( m, l, 2 ) ] ),
							  c2
							  ),
						 svld1( mask, &intpx3[  svcntw() * IDX3( m, l, 3 ) ] ),
						 c3
						 ),
					svld1( mask, &intpx3[  svcntw() * IDX3( m, l, 4 ) ] ),
					c4
					);

		  auto tmpy3 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpy3[  svcntw() * IDX3( m, l, 0 ) ] ),
									    c0
									    ),
								   svld1( mask, &intpy3[  svcntw() * IDX3( m, l, 1 ) ] ),
								   c1
								   ),
							  svld1( mask, &intpy3[ svcntw() *  IDX3( m, l, 2 ) ] ),
							  c2
							  ),
						 svld1( mask, &intpy3[  svcntw() * IDX3( m, l, 3 ) ] ),
						 c3
						 ),
					svld1( mask, &intpy3[  svcntw() * IDX3( m, l, 4 ) ] ),
					c4
					);

		  auto tmpz3 = svmla_z( mask,
					svmla_z( mask,
						 svmla_z( mask,
							  svmla_z( mask,
								   svmul_z( mask,
									    svld1( mask, &intpz3[  svcntw() * IDX3( m, l, 0 ) ] ),
									    c0
									    ),
								   svld1( mask, &intpz3[  svcntw() * IDX3( m, l, 1 ) ] ),
								   c1
								   ),
							  svld1( mask, &intpz3[  svcntw() * IDX3( m, l, 2 ) ] ),
							  c2
							  ),
						 svld1( mask, &intpz3[  svcntw() * IDX3( m, l, 3 ) ] ),
						 c3
						 ),
					svld1( mask, &intpz3[  svcntw() * IDX3( m, l, 4 ) ] ),
					c4
					);

		  //print( tmpx3 );
		  //print( tmpy3 );
		  //print( tmpz3 );
                  //std::cout << std::endl;

		  auto fac1 = rg_gll_weight[ l ] * rg_gll_weight[ k ];
		  auto fac2 = rg_gll_weight[ m ] * rg_gll_weight[ k ];
		  auto fac3 = rg_gll_weight[ m ] * rg_gll_weight[ l ];
          
		  auto rx = svmla_z( mask,
				     svmla_z( mask,
					      svmul_z( mask, tmpx1, fac1 ),
					      tmpx2,
					      fac2
					      ),
				     tmpx3,
				     fac3
				     );
		  
		  auto ry = svmla_z( mask,
				     svmla_z( mask,
					      svmul_z( mask, tmpy1, fac1 ),
					      tmpy2,
					      fac2
					      ),
				     tmpy3,
				     fac3
				     );

		  auto rz = svmla_z( mask,
				     svmla_z( mask,
					      svmul_z( mask, tmpz1, fac1 ),
					      tmpz2,
					      fac2
					      ),
				     tmpz3,
				     fac3
				     );

		  //print( rx );
		  //print( ry );
		  //print( rz );
                  //std::cout << std::endl;
		  
		  auto lid = IDX3( m, l, k );
		  auto gids = svadd_z( mask, base, lid );

		  //print( gids );
                  auto tmp = svld1_gather_index( mask, ig_hexa_gll_glonum.data(), gids );

		  auto ids = svmul_z( mask, svsub_z( mask, tmp, 1u ), 3u );

                  //print( ids );

                  //std::cout << std::endl;

		  //gather rg_gll_acceleration + sub
		  auto acc0 = svsub_z( mask, svld1_gather_index( mask, &rg_gll_acceleration[ 0 ], ids ), rx );
		  auto acc1 = svsub_z( mask, svld1_gather_index( mask, &rg_gll_acceleration[ 1 ], ids ), ry );
		  auto acc2 = svsub_z( mask, svld1_gather_index( mask, &rg_gll_acceleration[ 2 ], ids ), rz );

		  //scatter
		  svst1_scatter_index( mask, &rg_gll_acceleration[ 0 ], ids, acc0 );
		  svst1_scatter_index( mask, &rg_gll_acceleration[ 1 ], ids, acc1 );
		  svst1_scatter_index( mask, &rg_gll_acceleration[ 2 ], ids, acc2 );
		}
	    }
	}
    
    }

}
