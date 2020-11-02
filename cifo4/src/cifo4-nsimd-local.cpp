#include <cstddef>
#include <iostream>
#include <iomanip>

#include <nsimd/nsimd-all.hpp>

#include <cifo4.hpp>


#define IDX2( m, l ) ( 5 * l + m )
#define IDX3( m, l, k ) ( 25 * k + 5 * l + m )
#define IDX4( m, l, k, iel ) ( 125 * (iel) + 25 * (k) + 5 * (l) + (m) )


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
  auto const len = nsimd::len( f32{} );

  float rl_displacement_gll[5*5*5*3*len ];

  float local[ 5 * 5 * 5 * 9 * len ];

  float * intpx1 = &local[    0 ];
  float * intpy1 = &local[  125 * len ];
  float * intpz1 = &local[  250 * len ];

  float * intpx2 = &local[  375 * len ];
  float * intpy2 = &local[  500 * len ];
  float * intpz2 = &local[  625 * len ];

  float * intpx3 = &local[  750 * len ];
  float * intpy3 = &local[  875 * len ];
  float * intpz3 = &local[ 1000 * len ];

  // Add another iota with a parameter for the increment to simplify and generate only one svindex instruction ?
  // auto vstrides = nsimd::iota( 125u, u32{} );
  auto vstrides = nsimd::set1( 125u, u32{} ) * nsimd::iota( u32{} );

  for( std::size_t iel = elt_start ; iel < elt_end; iel += len )
    {
      auto mask = simd::mask_for_loop_tail( iel, elt_end, u32{} );

      auto base = nsimd::set1( iel * 125u, u32{} ) * vstrides;

      for( std::size_t k = 0 ; k < 5 ; ++k )
	{
	  for( std::size_t l = 0 ; l < 5 ; ++l )
	    {
	      for( std::size_t m = 0 ; m < 5 ; ++m )
		{
		  auto lid = IDX3( m, l, k );
		  auto gids = base + nsimd::set1( lid, u32{} );

		  // Need mask_gather here to avoid segfault
		  auto tmp = nsimd::mask_gather( mask, &ig_hexa_gll_glonum[ 0 ], gids, u32{} );
		  auto ids = nsimd::set1( 3u, u32{} ) * ( tmp - nsimd::set1( 1u, u32{} ) );

		  // no need for mask_store here since no dependencies betwen elements.
		  nsimd::storeu1( &rl_displacement_gll[ len * (3 * lid + 0) ], nsimd::mask_gather( mask, &rg_gll_displacement[ 0 ], ids, f32{} ), f32{} );
		  nsimd::storeu1( &rl_displacement_gll[ len * (3 * lid + 1) ], nsimd::mask_gather( mask, &rg_gll_displacement[ 1 ], ids, f32{} ), f32{} );
		  nsimd::storeu1( &rl_displacement_gll[ len * (3 * lid + 2) ], nsimd::mask_gather( mask, &rg_gll_displacement[ 2 ], ids, f32{} ), f32{} );
		}
	    }
	}

      for( std::size_t k = 0 ; k < 5 ; ++k )
	{
	  for( std::size_t l = 0 ; l < 5 ; ++l )
	    {
	      for( std::size_t m = 0 ; m < 5 ; ++m )
		{
		  auto coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 0, m ) ], f32{} );

		  auto index = 0 + 3 * IDX3( 0, l, k );

		  auto duxdxi = nsimd::loada( &rl_displacement_gll[ len * ( 0 + index ) ], f32{} ) * coeff;
		  auto duydxi = nsimd::loada( &rl_displacement_gll[ len * ( 1 + index ) ], f32{} ) * coeff;
		  auto duzdxi = nsimd::loada( &rl_displacement_gll[ len * ( 2 + index ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 1, m ) ], f32{} );

		  duxdxi += nsimd::loada( &rl_displacement_gll[  len * ( 3 + index ) ], f32{} ) * coeff;
		  duydxi += nsimd::loada( &rl_displacement_gll[  len * ( 4 + index ) ], f32{} ) * coeff;
		  duzdxi += nsimd::loada( &rl_displacement_gll[  len * ( 5 + index ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 2, m ) ], f32{} );

		  duxdxi += nsimd::loada( &rl_displacement_gll[  len * ( 6 + index ) ], f32{} ) * coeff;
		  duydxi += nsimd::loada( &rl_displacement_gll[  len * ( 7 + index ) ], f32{} ) * coeff;
		  duzdxi += nsimd::loada( &rl_displacement_gll[  len * ( 8 + index ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 3, m ) ], f32{} );

		  duxdxi += nsimd::loada( &rl_displacement_gll[ len * (  9 + index ) ], f32{} ) * coeff;
		  duydxi += nsimd::loada( &rl_displacement_gll[ len * ( 10 + index ) ], f32{} ) * coeff;
		  duzdxi += nsimd::loada( &rl_displacement_gll[ len * ( 11 + index ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 4, m ) ], f32{} );

		  duxdxi += nsimd::loada( &rl_displacement_gll[ len * ( 12 + index ) ], f32{} ) * coeff;
		  duydxi += nsimd::loada( &rl_displacement_gll[ len * ( 13 + index ) ], f32{} ) * coeff;
		  duzdxi += nsimd::loada( &rl_displacement_gll[ len * ( 14 + index ) ], f32{} ) * coeff;

		  //

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 0, l ) ], f32{} );

		  auto duxdet = nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, 0, k ) ) ], f32{} ) * coeff;
		  auto duydet = nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, 0, k ) ) ], f32{} ) * coeff;
		  auto duzdet = nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, 0, k ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 1, l ) ], f32{} );

		  duxdet += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, 1, k ) ) ], f32{} ) * coeff;
		  duydet += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, 1, k ) ) ], f32{} ) * coeff;
		  duzdet += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, 1, k ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 2, l ) ], f32{} );

		  duxdet += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, 2, k ) ) ], f32{} ) * coeff;
		  duydet += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, 2, k ) ) ], f32{} ) * coeff;
		  duzdet += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, 2, k ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 3, l ) ], f32{} );

		  duxdet += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, 3, k ) ) ], f32{} ) * coeff;
		  duydet += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, 3, k ) ) ], f32{} ) * coeff;
		  duzdet += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, 3, k ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 4, l ) ], f32{} );

		  duxdet += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, 4, k ) ) ], f32{} ) * coeff;
		  duydet += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, 4, k ) ) ], f32{} ) * coeff;
		  duzdet += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, 4, k ) ) ], f32{} ) * coeff;


		  //

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 0, k ) ], f32{} );

		  auto duxdze = nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, l, 0 ) ) ], f32{} ) * coeff;
		  auto duydze = nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, l, 0 ) ) ], f32{} ) * coeff;
		  auto duzdze = nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, l, 0 ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 1, k ) ], f32{} );

		  duxdze += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, l, 1 ) ) ], f32{} ) * coeff;
		  duydze += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, l, 1 ) ) ], f32{} ) * coeff;
		  duzdze += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, l, 1 ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 2, k ) ], f32{} );

		  duxdze += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, l, 2 ) ) ], f32{} ) * coeff;
		  duydze += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, l, 2 ) ) ], f32{} ) * coeff;
		  duzdze += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, l, 2 ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 3, k ) ], f32{} );

		  duxdze += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, l, 3 ) ) ], f32{} ) * coeff;
		  duydze += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, l, 3 ) ) ], f32{} ) * coeff;
		  duzdze += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, l, 3 ) ) ], f32{} ) * coeff;

		  coeff = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( 4, k ) ], f32{} );

		  duxdze += nsimd::loada( &rl_displacement_gll[ len * ( 0 + 3 * IDX3( m, l, 4 ) ) ], f32{} ) * coeff;
		  duydze += nsimd::loada( &rl_displacement_gll[ len * ( 1 + 3 * IDX3( m, l, 4 ) ) ], f32{} ) * coeff;
		  duzdze += nsimd::loada( &rl_displacement_gll[ len * ( 2 + 3 * IDX3( m, l, 4 ) ) ], f32{} ) * coeff;

		  //

                  auto lid = IDX3( m, l, k );
		  auto id  = iel * 125 + lid;

		  auto dxidx = nsimd::mask_gather( mask, &rg_hexa_gll_dxidx[ id ], vstrides, f32{} );
		  auto detdx = nsimd::mask_gather( mask, &rg_hexa_gll_detdx[ id ], vstrides, f32{} );
                  auto dzedx = nsimd::mask_gather( mask, &rg_hexa_gll_dzedx[ id ], vstrides, f32{} );

		  auto duxdx = duxdxi * dxidx + duxdet * detdx + duxdze * dzedx;
		  auto duydx = duydxi * dxidx + duydet * detdx + duydze * dzedx;
		  auto duzdx = duzdxi * dxidx + duzdet * detdx + duzdze * dzedx;

	          auto dxidy = nsimd::mask_gather( mask, &rg_hexa_gll_dxidy[ id ], vstrides, f32{} );
	          auto detdy = nsimd::mask_gather( mask, &rg_hexa_gll_detdy[ id ], vstrides, f32{} );
	          auto dzedy = nsimd::mask_gather( mask, &rg_hexa_gll_dzedy[ id ], vstrides, f32{} );

		  auto duxdy = duxdxi * dxidy + duxdet * detdy + duxdze * dzedy;
		  auto duydy = duydxi * dxidy + duydet * detdy + duydze * dzedy;
		  auto duzdy = duzdxi * dxidy + duzdet * detdy + duzdze * dzedy;

	          auto dxidz = nsimd::mask_gather( mask, &rg_hexa_gll_dxidz[ id ], vstrides, f32{} );
	          auto detdz = nsimd::mask_gather( mask, &rg_hexa_gll_detdz[ id ], vstrides, f32{} );
	          auto dzedz = nsimd::mask_gather( mask, &rg_hexa_gll_dzedz[ id ], vstrides, f32{} );

		  auto duxdz = duxdxi * dxidz + duxdet * detdz + duxdze * dzedz;
		  auto duydz = duydxi * dxidz + duydet * detdz + duydze * dzedz;
		  auto duzdz = duzdxi * dxidz + duzdet * detdz + duzdze * dzedz;

                  //

	          auto rhovp2 = nsimd::mask_gather( mask, &rg_hexa_gll_rhovp2[ id ], vstrides, f32{} );
		  auto rhovs2 = nsimd::mask_gather( mask, &rg_hexa_gll_rhovs2[ id ], vstrides, f32{} );

		  auto trace_tau = ( rhovp2 - nsimd::set1( 2.0f ) * rhovs2 )*(duxdx+duydy+duzdz);
		  auto tauxx     = trace_tau + nsimd::set1( 2.0f )*rhovs2*duxdx;
		  auto tauyy     = trace_tau + nsimd::set1( 2.0f )*rhovs2*duydy;
		  auto tauzz     = trace_tau + nsimd::set1( 2.0f )*rhovs2*duzdz;
		  auto tauxy     =                 rhovs2*(duxdy+duydx);
		  auto tauxz     =                 rhovs2*(duxdz+duzdx);
		  auto tauyz     =                 rhovs2*(duydz+duzdy);

  	          auto tmp = nsimd::mask_gather( mask, &rg_hexa_gll_jacobian_det[ id ], vstrides, f32{} );

		  nsimd::storea( &intpx1[ IDX3( m, l, k ) ], tmp * (tauxx*dxidx+tauxy*dxidy+tauxz*dxidz), f32{} );
		  nsimd::storea( &intpx2[ IDX3( m, l, k ) ], tmp * (tauxx*detdx+tauxy*detdy+tauxz*detdz), f32{} );
		  nsimd::storea( &intpx3[ IDX3( m, l, k ) ], tmp * (tauxx*dzedx+tauxy*dzedy+tauxz*dzedz), f32{} );

		  nsimd::storea( &intpy1[ IDX3( m, l, k ) ], tmp * (tauxy*dxidx+tauyy*dxidy+tauyz*dxidz), f32{} );
		  nsimd::storea( &intpy2[ IDX3( m, l, k ) ], tmp * (tauxy*detdx+tauyy*detdy+tauyz*detdz), f32{} );
		  nsimd::storea( &intpy3[ IDX3( m, l, k ) ], tmp * (tauxy*dzedx+tauyy*dzedy+tauyz*dzedz), f32{} );

		  nsimd::storea( &intpz1[ IDX3( m, l, k ) ], tmp * (tauxz*dxidx+tauyz*dxidy+tauzz*dxidz), f32{} );
		  nsimd::storea( &intpz2[ IDX3( m, l, k ) ], tmp * (tauxz*detdx+tauyz*detdy+tauzz*detdz), f32{} );
		  nsimd::storea( &intpz3[ IDX3( m, l, k ) ], tmp * (tauxz*dzedx+tauyz*dzedy+tauzz*dzedz), f32{} );

               }
	    }
	}


      for( std::size_t k = 0 ; k < 5 ; ++k )
	{
	  for( std::size_t l = 0 ; l < 5 ; ++l )
	    {
	      for( std::size_t m = 0 ; m < 5 ; ++m )
		{
		  auto c0 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( m, 0 ) ] * rg_gll_weight[ 0 ], f32{} );
		  auto c1 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( m, 1 ) ] * rg_gll_weight[ 1 ], f32{} );
		  auto c2 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( m, 2 ) ] * rg_gll_weight[ 2 ], f32{} );
		  auto c3 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( m, 3 ) ] * rg_gll_weight[ 3 ], f32{} );
		  auto c4 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( m, 4 ) ] * rg_gll_weight[ 4 ], f32{} );

		  auto tmpx1 = nsimd::loada( &intpx1[ IDX3( 0, l, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpx1[ IDX3( 1, l, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpx1[ IDX3( 2, l, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpx1[ IDX3( 3, l, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpx1[ IDX3( 4, l, k ) ], f32{} ) * c4;

		  auto tmpy1 = nsimd::loada( &intpy1[ IDX3( 0, l, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpy1[ IDX3( 1, l, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpy1[ IDX3( 2, l, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpy1[ IDX3( 3, l, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpy1[ IDX3( 4, l, k ) ], f32{} ) * c4;

		  auto tmpz1 = nsimd::loada( &intpz1[ IDX3( 0, l, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpz1[ IDX3( 1, l, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpz1[ IDX3( 2, l, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpz1[ IDX3( 3, l, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpz1[ IDX3( 4, l, k ) ], f32{} ) * c4;

		  c0 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( l, 0 ) ] * rg_gll_weight[ 0 ], f32{} );
		  c1 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( l, 1 ) ] * rg_gll_weight[ 1 ], f32{} );
		  c2 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( l, 2 ) ] * rg_gll_weight[ 2 ], f32{} );
		  c3 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( l, 3 ) ] * rg_gll_weight[ 3 ], f32{} );
		  c4 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( l, 4 ) ] * rg_gll_weight[ 4 ], f32{} );

		  auto tmpx2 = nsimd::loada( &intpx2[ IDX3( m, 0, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpx2[ IDX3( m, 1, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpx2[ IDX3( m, 2, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpx2[ IDX3( m, 3, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpx2[ IDX3( m, 4, k ) ], f32{} ) * c4;

		  auto tmpy2 = nsimd::loada( &intpy2[ IDX3( m, 0, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpy2[ IDX3( m, 1, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpy2[ IDX3( m, 2, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpy2[ IDX3( m, 3, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpy2[ IDX3( m, 4, k ) ], f32{} ) * c4;

		  auto tmpz2 = nsimd::loada( &intpz2[ IDX3( m, 0, k ) ], f32{} ) * c0
		    + nsimd::loada( &intpz2[ IDX3( m, 1, k ) ], f32{} ) * c1
		    + nsimd::loada( &intpz2[ IDX3( m, 2, k ) ], f32{} ) * c2
		    + nsimd::loada( &intpz2[ IDX3( m, 3, k ) ], f32{} ) * c3
		    + nsimd::loada( &intpz2[ IDX3( m, 4, k ) ], f32{} ) * c4;

		  c0 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( k, 0 ) ] * rg_gll_weight[ 0 ], f32{} );
		  c1 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( k, 1 ) ] * rg_gll_weight[ 1 ], f32{} );
		  c2 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( k, 2 ) ] * rg_gll_weight[ 2 ], f32{} );
		  c3 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( k, 3 ) ] * rg_gll_weight[ 3 ], f32{} );
		  c4 = nsimd::set1( rg_gll_lagrange_deriv[ IDX2( k, 4 ) ] * rg_gll_weight[ 4 ], f32{} );

		  auto tmpx3 = nsimd::loada( &intpx3[ IDX3( m, l, 0 ) ], f32{} ) * c0
		    + nsimd::loada( &intpx3[ IDX3( m, l, 1 ) ], f32{} ) * c1
		    + nsimd::loada( &intpx3[ IDX3( m, l, 2 ) ], f32{} ) * c2
		    + nsimd::loada( &intpx3[ IDX3( m, l, 3 ) ], f32{} ) * c3
		    + nsimd::loada( &intpx3[ IDX3( m, l, 4 ) ], f32{} ) * c4;

		  auto tmpy3 = nsimd::loada( &intpy3[ IDX3( m, l, 0 ) ], f32{} ) * c0
		    + nsimd::loada( &intpy3[ IDX3( m, l, 1 ) ], f32{} ) * c1
		    + nsimd::loada( &intpy3[ IDX3( m, l, 2 ) ], f32{} ) * c2
		    + nsimd::loada( &intpy3[ IDX3( m, l, 3 ) ], f32{} ) * c3
		    + nsimd::loada( &intpy3[ IDX3( m, l, 4 ) ], f32{} ) * c4;

		  auto tmpz3 = nsimd::loada( &intpz3[ IDX3( m, l, 0 ) ], f32{} ) * c0
		    + nsimd::loada( &intpz3[ IDX3( m, l, 1 ) ], f32{} ) * c1
		    + nsimd::loada( &intpz3[ IDX3( m, l, 2 ) ], f32{} ) * c2
		    + nsimd::loada( &intpz3[ IDX3( m, l, 3 ) ], f32{} ) * c3
		    + nsimd::loada( &intpz3[ IDX3( m, l, 4 ) ], f32{} ) * c4;

		  //

		  auto fac1 = nsimd::set1( rg_gll_weight[ l ] * rg_gll_weight[ k ], f32{} );
		  auto fac2 = nsimd::set1( rg_gll_weight[ m ] * rg_gll_weight[ k ], f32{} );
		  auto fac3 = nsimd::set1( rg_gll_weight[ m ] * rg_gll_weight[ l ], f32{} );

		  auto rx = fac1 * tmpx1 + fac2 * tmpx2 + fac3 * tmpx3;
		  auto ry = fac1 * tmpy1 + fac2 * tmpy2 + fac3 * tmpy3;
		  auto rz = fac1 * tmpz1 + fac2 * tmpz2 + fac3 * tmpz3;

		  auto lid = IDX3( m, l, k );
		  auto gids = base + nsimd::set1( lid, u32{} );

		  // nsimd
                  auto tmp = nsimd::gather( mask, ig_hexa_gll_glonum.data(), gids, u32{} );

		  auto ids = nsimd::set1( 3u ) * ( tmp - nsimd::set1( 1u ) );

		  auto acc0 = nsimd::gather( mask, &rg_gll_acceleration[ 0 ], ids, f32{} ) - rx;
		  auto acc1 = nsimd::gather( mask, &rg_gll_acceleration[ 1 ], ids, f32{} ) - ry;
		  auto acc2 = nsimd::gather( mask, &rg_gll_acceleration[ 2 ], ids, f32{} ) - rz;

		  nsimd::mask_scatter( mask, &rg_gll_acceleration[ 0 ], ids, acc0, f32{} );
		  nsimd::mask_scatter( mask, &rg_gll_acceleration[ 1 ], ids, acc1, f32{} );
		  nsimd::mask_scatter( mask, &rg_gll_acceleration[ 2 ], ids, acc2, f32{} );
		}
	    }
	}

    }

}
