#include <cstddef>
#include <iomanip>
#include <iostream>

#include <nsimd/nsimd-all.hpp>

#include <cifo4.hpp>

#define IDX2(m, l) (5 * l + m)
#define IDX3(m, l, k) (25 * k + 5 * l + m)
#define IDX4(m, l, k, iel) (125 * (iel) + 25 * (k) + 5 * (l) + (m))

std::vector<uint32_t, boost::alignment::aligned_allocator<uint32_t, 32>>
    ig_hexa_gll_glonum;

std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_gll_displacement;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_gll_weight;

std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_gll_lagrange_deriv;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_gll_acceleration;

std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dxidx;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dxidy;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dxidz;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_detdx;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_detdy;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_detdz;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dzedx;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dzedy;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_dzedz;

std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_rhovp2;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_rhovs2;
std::vector<float, boost::alignment::aligned_allocator<float, 32>>
    rg_hexa_gll_jacobian_det;

using pf32 = nsimd::pack<float>;
using pi32 = nsimd::pack<int>;
using pu32 = nsimd::pack<unsigned int>;

using pli32 = nsimd::packl<int>;
using plu32 = nsimd::packl<unsigned int>;
using plf32 = nsimd::packl<float>;

void compute_internal_forces_order4(std::size_t elt_start,
                                    std::size_t elt_end) {
  using namespace nsimd;

  int len = nsimd::len(f32{});

  float rl_displacement_gll[5 * 5 * 5 * 3 * NSIMD_MAX_LEN(f32)];
  float local[5 * 5 * 5 * 9 * NSIMD_MAX_LEN(f32)];

  float *intpx1 = &local[0];
  float *intpy1 = &local[125 * len];
  float *intpz1 = &local[250 * len];

  float *intpx2 = &local[375 * len];
  float *intpy2 = &local[500 * len];
  float *intpz2 = &local[625 * len];

  float *intpx3 = &local[750 * len];
  float *intpy3 = &local[875 * len];
  float *intpz3 = &local[1000 * len];

  pi32 vstrides = 125 * nsimd::iota<pi32>();

  for (int iel = elt_start; iel + len <= elt_end; iel += len) {

    pi32 base = (iel * 125) + vstrides;

    for (int k = 0; k < 5; ++k) {
      for (int l = 0; l < 5; ++l) {
        for (int m = 0; m < 5; ++m) {

          int lid = IDX3(m, l, k);
          pi32 gids = base + lid;

          // Need mask_gather here to avoid segfault
          auto tmp = gather<pi32>((int *)(&ig_hexa_gll_glonum[0]), gids);
          pi32 ids = 3 * (tmp - 1);

          // no need for mask_store here since no dependencies betwen elements.
          storea(&rl_displacement_gll[len * (3 * lid + 0)],
                 gather<pf32>(&rg_gll_displacement[0], ids));
          storea(&rl_displacement_gll[len * (3 * lid + 1)],
                 gather<pf32>(&rg_gll_displacement[1], ids));
          storea(&rl_displacement_gll[len * (3 * lid + 2)],
                 gather<pf32>(&rg_gll_displacement[2], ids));
        }
      }
    }

    for (int k = 0; k < 5; ++k) {
      for (int l = 0; l < 5; ++l) {

        auto index = 0 + 3 * IDX3(0, l, k);
        pf32 f00 = loada<pf32>(&rl_displacement_gll[len * (0 + index)]);
        pf32 f01 = loada<pf32>(&rl_displacement_gll[len * (1 + index)]);
        pf32 f02 = loada<pf32>(&rl_displacement_gll[len * (2 + index)]);
        pf32 f03 = loada<pf32>(&rl_displacement_gll[len * (3 + index)]);
        pf32 f04 = loada<pf32>(&rl_displacement_gll[len * (4 + index)]);
        pf32 f05 = loada<pf32>(&rl_displacement_gll[len * (5 + index)]);
        pf32 f06 = loada<pf32>(&rl_displacement_gll[len * (6 + index)]);
        pf32 f07 = loada<pf32>(&rl_displacement_gll[len * (7 + index)]);
        pf32 f08 = loada<pf32>(&rl_displacement_gll[len * (8 + index)]);
        pf32 f09 = loada<pf32>(&rl_displacement_gll[len * (9 + index)]);
        pf32 f10 = loada<pf32>(&rl_displacement_gll[len * (10 + index)]);
        pf32 f11 = loada<pf32>(&rl_displacement_gll[len * (11 + index)]);
        pf32 f12 = loada<pf32>(&rl_displacement_gll[len * (12 + index)]);
        pf32 f13 = loada<pf32>(&rl_displacement_gll[len * (13 + index)]);
        pf32 f14 = loada<pf32>(&rl_displacement_gll[len * (14 + index)]);

        pf32 coeff0l = rg_gll_lagrange_deriv[IDX2(0, l)];
        pf32 coeff1l = rg_gll_lagrange_deriv[IDX2(1, l)];
        pf32 coeff2l = rg_gll_lagrange_deriv[IDX2(2, l)];
        pf32 coeff3l = rg_gll_lagrange_deriv[IDX2(3, l)];
        pf32 coeff4l = rg_gll_lagrange_deriv[IDX2(4, l)];

        for (std::size_t m = 0; m < 5; ++m) {

          auto coeff = set1<pf32>(rg_gll_lagrange_deriv[IDX2(0, m)]);

          auto index = 0 + 3 * IDX3(0, l, k);

          // auto duxdxi = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 0 +
          // index ) ] ) * coeff;
          // auto duydxi = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 1 +
          // index ) ] ) * coeff;
          // auto duzdxi = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 2 +
          // index ) ] ) * coeff;

          auto duxdxi = f00 * coeff;
          auto duydxi = f01 * coeff;
          auto duzdxi = f02 * coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(1, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 3 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 4 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 5 + index ) ] ) * coeff;

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 3 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 4 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 5 + index ) ] ) * coeff;

          duxdxi = duxdxi + f03 * coeff;
          duydxi = duydxi + f04 * coeff;
          duzdxi = duzdxi + f05 * coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(2, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 6 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 7 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 8 + index ) ] ) * coeff;

          duxdxi = duxdxi + f06 * coeff;
          duydxi = duydxi + f07 * coeff;
          duzdxi = duzdxi + f08 * coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(3, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 9 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 10 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 11 + index ) ] ) * coeff;

          duxdxi = duxdxi + f09 * coeff;
          duydxi = duydxi + f10 * coeff;
          duzdxi = duzdxi + f11 * coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(4, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 12 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 13 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 14 + index ) ] ) * coeff;

          duxdxi = duxdxi + f12 * coeff;
          duydxi = duydxi + f13 * coeff;
          duzdxi = duzdxi + f14 * coeff;

          //

          // coeff = rg_gll_lagrange_deriv[ IDX2( 0, l ) ];

          // auto duxdet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 0 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;
          // auto duydet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 1 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;
          // auto duzdet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 2 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;

          auto duxdet =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (0 + 3 * IDX3(m, 0, k))]) *
              coeff0l;
          auto duydet =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (1 + 3 * IDX3(m, 0, k))]) *
              coeff0l;
          auto duzdet =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (2 + 3 * IDX3(m, 0, k))]) *
              coeff0l;

          // coeff = rg_gll_lagrange_deriv[ IDX2( 1, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;

          duxdet = duxdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, 1, k))]) *
                       coeff1l;
          duydet = duydet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, 1, k))]) *
                       coeff1l;
          duzdet = duzdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, 1, k))]) *
                       coeff1l;

          // coeff = rg_gll_lagrange_deriv[ IDX2( 2, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;

          duxdet = duxdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, 2, k))]) *
                       coeff2l;
          duydet = duydet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, 2, k))]) *
                       coeff2l;
          duzdet = duzdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, 2, k))]) *
                       coeff2l;

          // coeff = rg_gll_lagrange_deriv[ IDX2( 3, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;

          duxdet = duxdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;
          duydet = duydet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;
          duzdet = duzdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;

          // coeff = rg_gll_lagrange_deriv[ IDX2( 4, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;

          duxdet = duxdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, 4, k))]) *
                       coeff4l;
          duydet = duydet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, 4, k))]) *
                       coeff4l;
          duzdet = duzdet +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, 4, k))]) *
                       coeff4l;

          //

          coeff = rg_gll_lagrange_deriv[IDX2(0, k)];

          auto duxdze =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 0))]) *
              coeff;
          auto duydze =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 0))]) *
              coeff;
          auto duzdze =
              nsimd::loada<pf32>(
                  &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 0))]) *
              coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(1, k)];

          duxdze = duxdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 1))]) *
                       coeff;
          duydze = duydze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 1))]) *
                       coeff;
          duzdze = duzdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 1))]) *
                       coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(2, k)];

          duxdze = duxdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 2))]) *
                       coeff;
          duydze = duydze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 2))]) *
                       coeff;
          duzdze = duzdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 2))]) *
                       coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(3, k)];

          duxdze = duxdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 3))]) *
                       coeff;
          duydze = duydze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 3))]) *
                       coeff;
          duzdze = duzdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 3))]) *
                       coeff;

          coeff = rg_gll_lagrange_deriv[IDX2(4, k)];

          duxdze = duxdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 4))]) *
                       coeff;
          duydze = duydze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 4))]) *
                       coeff;
          duzdze = duzdze +
                   nsimd::loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 4))]) *
                       coeff;

          //

          // tt_midloop1.push_back(tic() - tt);

          // tt = tic();

          auto lid = IDX3(m, l, k);
          auto id = iel * 125 + lid;

          // auto dxidx = nsimd::gather<pf32>( &rg_hexa_gll_dxidx[ id ],
          // vstrides );
          // auto detdx = nsimd::gather<pf32>( &rg_hexa_gll_detdx[ id ],
          // vstrides );
          // auto dzedx = nsimd::gather<pf32>( &rg_hexa_gll_dzedx[ id ],
          // vstrides );

          auto dxidx = gather_linear<pf32>(&rg_hexa_gll_dxidx[id], 125);
          auto detdx = gather_linear<pf32>(&rg_hexa_gll_detdx[id], 125);
          auto dzedx = gather_linear<pf32>(&rg_hexa_gll_dzedx[id], 125);

          auto duxdx = duxdxi * dxidx + duxdet * detdx + duxdze * dzedx;
          auto duydx = duydxi * dxidx + duydet * detdx + duydze * dzedx;
          auto duzdx = duzdxi * dxidx + duzdet * detdx + duzdze * dzedx;

          // auto dxidy = nsimd::gather<pf32>( &rg_hexa_gll_dxidy[ id ],
          // vstrides );
          // auto detdy = nsimd::gather<pf32>( &rg_hexa_gll_detdy[ id ],
          // vstrides );
          // auto dzedy = nsimd::gather<pf32>( &rg_hexa_gll_dzedy[ id ],
          // vstrides );

          auto dxidy = gather_linear<pf32>(&rg_hexa_gll_dxidy[id], 125);
          auto detdy = gather_linear<pf32>(&rg_hexa_gll_detdy[id], 125);
          auto dzedy = gather_linear<pf32>(&rg_hexa_gll_dzedy[id], 125);

          auto duxdy = duxdxi * dxidy + duxdet * detdy + duxdze * dzedy;
          auto duydy = duydxi * dxidy + duydet * detdy + duydze * dzedy;
          auto duzdy = duzdxi * dxidy + duzdet * detdy + duzdze * dzedy;

          // auto dxidz = nsimd::gather<pf32>( &rg_hexa_gll_dxidz[ id ],
          // vstrides );
          // auto detdz = nsimd::gather<pf32>( &rg_hexa_gll_detdz[ id ],
          // vstrides );
          // auto dzedz = nsimd::gather<pf32>( &rg_hexa_gll_dzedz[ id ],
          // vstrides );

          auto dxidz = gather_linear<pf32>(&rg_hexa_gll_dxidz[id], 125);
          auto detdz = gather_linear<pf32>(&rg_hexa_gll_detdz[id], 125);
          auto dzedz = gather_linear<pf32>(&rg_hexa_gll_dzedz[id], 125);

          auto duxdz = duxdxi * dxidz + duxdet * detdz + duxdze * dzedz;
          auto duydz = duydxi * dxidz + duydet * detdz + duydze * dzedz;
          auto duzdz = duzdxi * dxidz + duzdet * detdz + duzdze * dzedz;

          //

          // auto rhovp2 = nsimd::gather<pf32>( &rg_hexa_gll_rhovp2[ id ],
          // vstrides );
          // auto rhovs2 = nsimd::gather<pf32>( &rg_hexa_gll_rhovs2[ id ],
          // vstrides );

          auto rhovp2 = gather_linear<pf32>(&rg_hexa_gll_rhovp2[id], 125);
          auto rhovs2 = gather_linear<pf32>(&rg_hexa_gll_rhovs2[id], 125);

          auto trace_tau = (rhovp2 - 2.0f * rhovs2) * (duxdx + duydy + duzdz);
          auto tauxx = trace_tau + 2.0f * rhovs2 * duxdx;
          auto tauyy = trace_tau + 2.0f * rhovs2 * duydy;
          auto tauzz = trace_tau + 2.0f * rhovs2 * duzdz;
          auto tauxy = rhovs2 * (duxdy + duydx);
          auto tauxz = rhovs2 * (duxdz + duzdx);
          auto tauyz = rhovs2 * (duydz + duzdy);

          // auto tmp = nsimd::gather<pf32>( &rg_hexa_gll_jacobian_det[ id ],
          // vstrides );
          auto tmp = gather_linear<pf32>(&rg_hexa_gll_jacobian_det[id], 125);

          nsimd::storea(&intpx1[IDX3(m, l, k)],
                        tmp * (tauxx * dxidx + tauxy * dxidy + tauxz * dxidz));
          nsimd::storea(&intpx2[IDX3(m, l, k)],
                        tmp * (tauxx * detdx + tauxy * detdy + tauxz * detdz));
          nsimd::storea(&intpx3[IDX3(m, l, k)],
                        tmp * (tauxx * dzedx + tauxy * dzedy + tauxz * dzedz));

          nsimd::storea(&intpy1[IDX3(m, l, k)],
                        tmp * (tauxy * dxidx + tauyy * dxidy + tauyz * dxidz));
          nsimd::storea(&intpy2[IDX3(m, l, k)],
                        tmp * (tauxy * detdx + tauyy * detdy + tauyz * detdz));
          nsimd::storea(&intpy3[IDX3(m, l, k)],
                        tmp * (tauxy * dzedx + tauyy * dzedy + tauyz * dzedz));

          nsimd::storea(&intpz1[IDX3(m, l, k)],
                        tmp * (tauxz * dxidx + tauyz * dxidy + tauzz * dxidz));
          nsimd::storea(&intpz2[IDX3(m, l, k)],
                        tmp * (tauxz * detdx + tauyz * detdy + tauzz * detdz));
          nsimd::storea(&intpz3[IDX3(m, l, k)],
                        tmp * (tauxz * dzedx + tauyz * dzedy + tauzz * dzedz));
        }
      }
    }

    //tt_midloop2.push_back(tic() - tt);

    //tt = tic();

    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t l = 0; l < 5; ++l) {

        pf32 intpx10 = nsimd::loada<pf32>(&intpx1[IDX3(0, l, k)]);
        pf32 intpx11 = nsimd::loada<pf32>(&intpx1[IDX3(1, l, k)]);
        pf32 intpx12 = nsimd::loada<pf32>(&intpx1[IDX3(2, l, k)]);
        pf32 intpx13 = nsimd::loada<pf32>(&intpx1[IDX3(3, l, k)]);
        pf32 intpx14 = nsimd::loada<pf32>(&intpx1[IDX3(4, l, k)]);

        pf32 intpy10 = nsimd::loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy11 = nsimd::loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy12 = nsimd::loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy13 = nsimd::loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy14 = nsimd::loada<pf32>(&intpy1[IDX3(0, l, k)]);

        pf32 intpz10 = nsimd::loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz11 = nsimd::loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz12 = nsimd::loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz13 = nsimd::loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz14 = nsimd::loada<pf32>(&intpz1[IDX3(0, l, k)]);

        for (std::size_t m = 0; m < 5; ++m) {
          auto c0 = nsimd::set1<pf32>(rg_gll_lagrange_deriv[IDX2(m, 0)] *
                                      rg_gll_weight[0]);
          auto c1 = nsimd::set1<pf32>(rg_gll_lagrange_deriv[IDX2(m, 1)] *
                                      rg_gll_weight[1]);
          auto c2 = nsimd::set1<pf32>(rg_gll_lagrange_deriv[IDX2(m, 2)] *
                                      rg_gll_weight[2]);
          auto c3 = nsimd::set1<pf32>(rg_gll_lagrange_deriv[IDX2(m, 3)] *
                                      rg_gll_weight[3]);
          auto c4 = nsimd::set1<pf32>(rg_gll_lagrange_deriv[IDX2(m, 4)] *
                                      rg_gll_weight[4]);

          auto tmpx1 = intpx10 * c0 + intpx11 * c1 + intpx12 * c2 +
                       intpx13 * c3 + intpx14 * c4;

          auto tmpy1 = intpy10 * c0 + intpy11 * c1 + intpy12 * c2 +
                       intpy13 * c3 + intpy14 * c4;

          auto tmpz1 = intpz10 * c0 + intpz11 * c1 + intpz12 * c2 +
                       intpz13 * c3 + intpz14 * c4;

          /*
          auto tmpx1 = nsimd::loada<pf32>( &intpx1[ IDX3( 0, l, k ) ]) * c0
            + nsimd::loada<pf32>( &intpx1[ IDX3( 1, l, k ) ] ) * c1
            + nsimd::loada<pf32>( &intpx1[ IDX3( 2, l, k ) ] ) * c2
            + nsimd::loada<pf32>( &intpx1[ IDX3( 3, l, k ) ] ) * c3
            + nsimd::loada<pf32>( &intpx1[ IDX3( 4, l, k ) ] ) * c4;

          auto tmpy1 = nsimd::loada<pf32>( &intpy1[ IDX3( 0, l, k ) ] ) * c0
            + nsimd::loada<pf32>( &intpy1[ IDX3( 1, l, k ) ] ) * c1
            + nsimd::loada<pf32>( &intpy1[ IDX3( 2, l, k ) ] ) * c2
            + nsimd::loada<pf32>( &intpy1[ IDX3( 3, l, k ) ] ) * c3
            + nsimd::loada<pf32>( &intpy1[ IDX3( 4, l, k ) ] ) * c4;

          auto tmpz1 = nsimd::loada<pf32>( &intpz1[ IDX3( 0, l, k ) ] ) * c0
            + nsimd::loada<pf32>( &intpz1[ IDX3( 1, l, k ) ] ) * c1
            + nsimd::loada<pf32>( &intpz1[ IDX3( 2, l, k ) ] ) * c2
            + nsimd::loada<pf32>( &intpz1[ IDX3( 3, l, k ) ] ) * c3
            + nsimd::loada<pf32>( &intpz1[ IDX3( 4, l, k ) ] ) * c4;
          */

          c0 = rg_gll_lagrange_deriv[IDX2(l, 0)] * rg_gll_weight[0];
          c1 = rg_gll_lagrange_deriv[IDX2(l, 1)] * rg_gll_weight[1];
          c2 = rg_gll_lagrange_deriv[IDX2(l, 2)] * rg_gll_weight[2];
          c3 = rg_gll_lagrange_deriv[IDX2(l, 3)] * rg_gll_weight[3];
          c4 = rg_gll_lagrange_deriv[IDX2(l, 4)] * rg_gll_weight[4];

          auto tmpx2 = nsimd::loada<pf32>(&intpx2[IDX3(m, 0, k)]) * c0 +
                       nsimd::loada<pf32>(&intpx2[IDX3(m, 1, k)]) * c1 +
                       nsimd::loada<pf32>(&intpx2[IDX3(m, 2, k)]) * c2 +
                       nsimd::loada<pf32>(&intpx2[IDX3(m, 3, k)]) * c3 +
                       nsimd::loada<pf32>(&intpx2[IDX3(m, 4, k)]) * c4;

          auto tmpy2 = nsimd::loada<pf32>(&intpy2[IDX3(m, 0, k)]) * c0 +
                       nsimd::loada<pf32>(&intpy2[IDX3(m, 1, k)]) * c1 +
                       nsimd::loada<pf32>(&intpy2[IDX3(m, 2, k)]) * c2 +
                       nsimd::loada<pf32>(&intpy2[IDX3(m, 3, k)]) * c3 +
                       nsimd::loada<pf32>(&intpy2[IDX3(m, 4, k)]) * c4;

          auto tmpz2 = nsimd::loada<pf32>(&intpz2[IDX3(m, 0, k)]) * c0 +
                       nsimd::loada<pf32>(&intpz2[IDX3(m, 1, k)]) * c1 +
                       nsimd::loada<pf32>(&intpz2[IDX3(m, 2, k)]) * c2 +
                       nsimd::loada<pf32>(&intpz2[IDX3(m, 3, k)]) * c3 +
                       nsimd::loada<pf32>(&intpz2[IDX3(m, 4, k)]) * c4;

          c0 = rg_gll_lagrange_deriv[IDX2(k, 0)] * rg_gll_weight[0];
          c1 = rg_gll_lagrange_deriv[IDX2(k, 1)] * rg_gll_weight[1];
          c2 = rg_gll_lagrange_deriv[IDX2(k, 2)] * rg_gll_weight[2];
          c3 = rg_gll_lagrange_deriv[IDX2(k, 3)] * rg_gll_weight[3];
          c4 = rg_gll_lagrange_deriv[IDX2(k, 4)] * rg_gll_weight[4];

          auto tmpx3 = nsimd::loada<pf32>(&intpx3[IDX3(m, l, 0)]) * c0 +
                       nsimd::loada<pf32>(&intpx3[IDX3(m, l, 1)]) * c1 +
                       nsimd::loada<pf32>(&intpx3[IDX3(m, l, 2)]) * c2 +
                       nsimd::loada<pf32>(&intpx3[IDX3(m, l, 3)]) * c3 +
                       nsimd::loada<pf32>(&intpx3[IDX3(m, l, 4)]) * c4;

          auto tmpy3 = nsimd::loada<pf32>(&intpy3[IDX3(m, l, 0)]) * c0 +
                       nsimd::loada<pf32>(&intpy3[IDX3(m, l, 1)]) * c1 +
                       nsimd::loada<pf32>(&intpy3[IDX3(m, l, 2)]) * c2 +
                       nsimd::loada<pf32>(&intpy3[IDX3(m, l, 3)]) * c3 +
                       nsimd::loada<pf32>(&intpy3[IDX3(m, l, 4)]) * c4;

          auto tmpz3 = nsimd::loada<pf32>(&intpz3[IDX3(m, l, 0)]) * c0 +
                       nsimd::loada<pf32>(&intpz3[IDX3(m, l, 1)]) * c1 +
                       nsimd::loada<pf32>(&intpz3[IDX3(m, l, 2)]) * c2 +
                       nsimd::loada<pf32>(&intpz3[IDX3(m, l, 3)]) * c3 +
                       nsimd::loada<pf32>(&intpz3[IDX3(m, l, 4)]) * c4;

          //

          auto fac1 = rg_gll_weight[l] * rg_gll_weight[k];
          auto fac2 = rg_gll_weight[m] * rg_gll_weight[k];
          auto fac3 = rg_gll_weight[m] * rg_gll_weight[l];

          auto rx = fac1 * tmpx1 + fac2 * tmpx2 + fac3 * tmpx3;
          auto ry = fac1 * tmpy1 + fac2 * tmpy2 + fac3 * tmpy3;
          auto rz = fac1 * tmpz1 + fac2 * tmpz2 + fac3 * tmpz3;

          auto lid = IDX3(m, l, k);
          auto gids = base + lid;

          // nsimd
          auto tmp =
              nsimd::gather<pi32>((int *)(ig_hexa_gll_glonum.data()), gids);

          auto ids = 3u * (tmp - 1u);

          auto acc0 = nsimd::gather<pf32>(&rg_gll_acceleration[0], ids) - rx;
          auto acc1 = nsimd::gather<pf32>(&rg_gll_acceleration[1], ids) - ry;
          auto acc2 = nsimd::gather<pf32>(&rg_gll_acceleration[2], ids) - rz;

          nsimd::scatter(&rg_gll_acceleration[0], ids, acc0);
          nsimd::scatter(&rg_gll_acceleration[1], ids, acc1);
          nsimd::scatter(&rg_gll_acceleration[2], ids, acc2);
        }
      }
    }

    //tt_lastloop.push_back(tic() - tt);
  }

  // std::cout << "[nsimd] 1 = " << avg(tt_firstloop) << "\n";
  ////std::cout << "[nsimd] 2.1 = " << avg(tt_midloop1) << "\n";
  // std::cout << "[nsimd] 2.2 = " << avg(tt_midloop2) << "\n";
  // std::cout << "[nsimd] 3 = " << avg(tt_lastloop) << "\n";
}
