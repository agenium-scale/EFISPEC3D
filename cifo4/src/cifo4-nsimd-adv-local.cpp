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

  unsigned int *ig_hexa_gll_glonum2 = ig_hexa_gll_glonum.data();
  float *rg_gll_displacement2 = rg_gll_displacement.data();
  float *rg_gll_weight2 = rg_gll_weight.data();
  float *rg_gll_lagrange_deriv2 = rg_gll_lagrange_deriv.data();
  float *rg_gll_acceleration2 = rg_gll_acceleration.data();
  float *rg_hexa_gll_dxidx2 = rg_hexa_gll_dxidx.data();
  float *rg_hexa_gll_dxidy2 = rg_hexa_gll_dxidy.data();
  float *rg_hexa_gll_dxidz2 = rg_hexa_gll_dxidz.data();
  float *rg_hexa_gll_detdx2 = rg_hexa_gll_detdx.data();
  float *rg_hexa_gll_detdy2 = rg_hexa_gll_detdy.data();
  float *rg_hexa_gll_detdz2 = rg_hexa_gll_detdz.data();
  float *rg_hexa_gll_dzedx2 = rg_hexa_gll_dzedx.data();
  float *rg_hexa_gll_dzedy2 = rg_hexa_gll_dzedy.data();
  float *rg_hexa_gll_dzedz2 = rg_hexa_gll_dzedz.data();
  float *rg_hexa_gll_rhovp22 = rg_hexa_gll_rhovp2.data();
  float *rg_hexa_gll_rhovs22 = rg_hexa_gll_rhovs2.data();
  float *rg_hexa_gll_jacobian_det2 = rg_hexa_gll_jacobian_det.data();

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
          auto tmp = gather<pi32>((int *)(ig_hexa_gll_glonum2), gids);
          pi32 ids = 3 * (tmp - 1);

          // no need for mask_store here since no dependencies betwen elements.
          storea(&rl_displacement_gll[len * (3 * lid + 0)],
                 gather<pf32>(&rg_gll_displacement2[0], ids));
          storea(&rl_displacement_gll[len * (3 * lid + 1)],
                 gather<pf32>(&rg_gll_displacement2[1], ids));
          storea(&rl_displacement_gll[len * (3 * lid + 2)],
                 gather<pf32>(&rg_gll_displacement2[2], ids));
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

        pf32 coeff0l = rg_gll_lagrange_deriv2[IDX2(0, l)];
        pf32 coeff1l = rg_gll_lagrange_deriv2[IDX2(1, l)];
        pf32 coeff2l = rg_gll_lagrange_deriv2[IDX2(2, l)];
        pf32 coeff3l = rg_gll_lagrange_deriv2[IDX2(3, l)];
        pf32 coeff4l = rg_gll_lagrange_deriv2[IDX2(4, l)];

        for (std::size_t m = 0; m < 5; ++m) {

          auto coeff = set1<pf32>(rg_gll_lagrange_deriv2[IDX2(0, m)]);

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

          coeff = rg_gll_lagrange_deriv2[IDX2(1, m)];

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

          //duxdxi = duxdxi + f03 * coeff;
          //duydxi = duydxi + f04 * coeff;
          //duzdxi = duzdxi + f05 * coeff;

          duxdxi = fma(f03, coeff, duxdxi);
          duydxi = fma(f04, coeff, duydxi);
          duzdxi = fma(f05, coeff, duzdxi);

          coeff = rg_gll_lagrange_deriv2[IDX2(2, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 6 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 7 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[  len *
          // ( 8 + index ) ] ) * coeff;

          duxdxi = fma(f06, coeff, duxdxi);
          duydxi = fma(f07, coeff, duydxi);
          duzdxi = fma(f08, coeff, duzdxi);

          coeff = rg_gll_lagrange_deriv2[IDX2(3, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 9 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 10 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 11 + index ) ] ) * coeff;

          duxdxi = fma(f09, coeff, duxdxi);
          duydxi = fma(f10, coeff, duydxi);
          duzdxi = fma(f11, coeff, duzdxi);

          coeff = rg_gll_lagrange_deriv2[IDX2(4, m)];

          // duxdxi = duxdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 12 + index ) ] ) * coeff;
          // duydxi = duydxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 13 + index ) ] ) * coeff;
          // duzdxi = duzdxi + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 14 + index ) ] ) * coeff;

          duxdxi = fma(f12, coeff, duxdxi);
          duydxi = fma(f13, coeff, duydxi);
          duzdxi = fma(f14, coeff, duzdxi);

          //

          // coeff = rg_gll_lagrange_deriv2[ IDX2( 0, l ) ];

          // auto duxdet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 0 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;
          // auto duydet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 1 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;
          // auto duzdet = nsimd::loada<pf32>( &rl_displacement_gll[ len * ( 2 +
          // 3 * IDX3( m, 0, k ) ) ] ) * coeff;

          auto duxdet =
              loada<pf32>(
                  &rl_displacement_gll[len * (0 + 3 * IDX3(m, 0, k))]) *
              coeff0l;
          auto duydet =
              loada<pf32>(
                  &rl_displacement_gll[len * (1 + 3 * IDX3(m, 0, k))]) *
              coeff0l;
          auto duzdet =
              loada<pf32>(
                  &rl_displacement_gll[len * (2 + 3 * IDX3(m, 0, k))]) *
              coeff0l;

          // coeff = rg_gll_lagrange_deriv2[ IDX2( 1, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 1, k ) ) ] ) * coeff;

          duxdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, 1, k))]),
              coeff1l, duxdet);
          duydet = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, 1, k))]),
              coeff1l, duydet);
          duzdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, 1, k))]),
              coeff1l, duzdet);

          // coeff = rg_gll_lagrange_deriv2[ IDX2( 2, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 2, k ) ) ] ) * coeff;

          duxdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, 2, k))]),
              coeff2l, duxdet);
          duydet = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, 2, k))]),
              coeff2l, duydet);
          duzdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, 2, k))]),
              coeff2l, duzdet);

          // coeff = rg_gll_lagrange_deriv2[ IDX2( 3, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 3, k ) ) ] ) * coeff;

          duxdet = duxdet +
                   loada<pf32>(
                       &rl_displacement_gll[len * (0 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;
          duydet = duydet +
                   loada<pf32>(
                       &rl_displacement_gll[len * (1 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;
          duzdet = duzdet +
                   loada<pf32>(
                       &rl_displacement_gll[len * (2 + 3 * IDX3(m, 3, k))]) *
                       coeff3l;

          // coeff = rg_gll_lagrange_deriv2[ IDX2( 4, l ) ];

          // duxdet = duxdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 0 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;
          // duydet = duydet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 1 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;
          // duzdet = duzdet + nsimd::loada<pf32>( &rl_displacement_gll[ len * (
          // 2 + 3 * IDX3( m, 4, k ) ) ] ) * coeff;

          duxdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, 4, k))]),
              coeff4l, duxdet);
          duydet = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, 4, k))]),
              coeff4l, duydet);
          duzdet = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, 4, k))]),
              coeff4l, duzdet);

          //

          coeff = rg_gll_lagrange_deriv2[IDX2(0, k)];

          auto duxdze =
              loada<pf32>(
                  &rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 0))]) *
              coeff;
          auto duydze =
              loada<pf32>(
                  &rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 0))]) *
              coeff;
          auto duzdze =
              loada<pf32>(
                  &rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 0))]) *
              coeff;

          coeff = rg_gll_lagrange_deriv2[IDX2(1, k)];

          duxdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 1))]),
              coeff, duxdze);
          duydze = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 1))]),
              coeff, duydze);
          duzdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 1))]),
              coeff, duzdze);

          coeff = rg_gll_lagrange_deriv2[IDX2(2, k)];

          duxdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 2))]),
              coeff, duxdze);
          duydze = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 2))]),
              coeff, duydze);
          duzdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 2))]),
              coeff, duzdze);

          coeff = rg_gll_lagrange_deriv2[IDX2(3, k)];

          duxdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 3))]),
              coeff, duxdze);
          duydze = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 3))]),
              coeff, duydze);
          duzdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 3))]),
              coeff, duzdze);

          coeff = rg_gll_lagrange_deriv2[IDX2(4, k)];

          duxdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (0 + 3 * IDX3(m, l, 4))]),
              coeff, duxdze);
          duydze = fma(
              loada<pf32>(&rl_displacement_gll[len * (1 + 3 * IDX3(m, l, 4))]),
              coeff, duydze);
          duzdze = fma(
              loada<pf32>(&rl_displacement_gll[len * (2 + 3 * IDX3(m, l, 4))]),
              coeff, duzdze);

          //

          // tt_midloop1.push_back(tic() - tt);

          // tt = tic();

          auto lid = IDX3(m, l, k);
          auto id = iel * 125 + lid;

          // auto dxidx = nsimd::gather<pf32>( &rg_hexa_gll_dxidx2[ id ],
          // vstrides );
          // auto detdx = nsimd::gather<pf32>( &rg_hexa_gll_detdx2[ id ],
          // vstrides );
          // auto dzedx = nsimd::gather<pf32>( &rg_hexa_gll_dzedx2[ id ],
          // vstrides );

          auto dxidx = gather_linear<pf32>(&rg_hexa_gll_dxidx2[id], 125);
          auto detdx = gather_linear<pf32>(&rg_hexa_gll_detdx2[id], 125);
          auto dzedx = gather_linear<pf32>(&rg_hexa_gll_dzedx2[id], 125);

          auto duxdx = fma(duxdxi, dxidx, fma(duxdet, detdx, duxdze * dzedx));
          auto duydx = fma(duydxi, dxidx, fma(duydet, detdx, duydze * dzedx));
          auto duzdx = fma(duzdxi, dxidx, fma(duzdet, detdx, duzdze * dzedx));

          // auto dxidy = nsimd::gather<pf32>( &rg_hexa_gll_dxidy2[ id ],
          // vstrides );
          // auto detdy = nsimd::gather<pf32>( &rg_hexa_gll_detdy2[ id ],
          // vstrides );
          // auto dzedy = nsimd::gather<pf32>( &rg_hexa_gll_dzedy2[ id ],
          // vstrides );

          auto dxidy = gather_linear<pf32>(&rg_hexa_gll_dxidy2[id], 125);
          auto detdy = gather_linear<pf32>(&rg_hexa_gll_detdy2[id], 125);
          auto dzedy = gather_linear<pf32>(&rg_hexa_gll_dzedy2[id], 125);

          auto duxdy = fma(duxdxi, dxidy, fma(duxdet, detdy, duxdze * dzedy));
          auto duydy = fma(duydxi, dxidy, fma(duydet, detdy, duydze * dzedy));
          auto duzdy = fma(duzdxi, dxidy, fma(duzdet, detdy, duzdze * dzedy));

          // auto dxidz = nsimd::gather<pf32>( &rg_hexa_gll_dxidz2[ id ],
          // vstrides );
          // auto detdz = nsimd::gather<pf32>( &rg_hexa_gll_detdz2[ id ],
          // vstrides );
          // auto dzedz = nsimd::gather<pf32>( &rg_hexa_gll_dzedz2[ id ],
          // vstrides );

          auto dxidz = gather_linear<pf32>(&rg_hexa_gll_dxidz2[id], 125);
          auto detdz = gather_linear<pf32>(&rg_hexa_gll_detdz2[id], 125);
          auto dzedz = gather_linear<pf32>(&rg_hexa_gll_dzedz2[id], 125);

          auto duxdz = fma(duxdxi, dxidz, fma(duxdet, detdz, duxdze * dzedz));
          auto duydz = fma(duydxi, dxidz, fma(duydet, detdz, duydze * dzedz));
          auto duzdz = fma(duzdxi, dxidz, fma(duzdet, detdz, duzdze * dzedz));

          //

          // auto rhovp2 = nsimd::gather<pf32>( &rg_hexa_gll_rhovp22[ id ],
          // vstrides );
          // auto rhovs2 = nsimd::gather<pf32>( &rg_hexa_gll_rhovs22[ id ],
          // vstrides );

          auto rhovp2 = gather_linear<pf32>(&rg_hexa_gll_rhovp22[id], 125);
          auto rhovs2 = gather_linear<pf32>(&rg_hexa_gll_rhovs22[id], 125);

          pf32 two = set1<pf32>(2.0f);
          //auto trace_tau = (rhovp2 - 2.0f * rhovs2) * (duxdx + duydy + duzdz);
          auto trace_tau = fnma(two, rhovs2, rhovp2) * (duxdx + duydy + duzdz);

          auto tauxx = fma(two * rhovs2, duxdx, trace_tau);
          auto tauyy = fma(two * rhovs2, duydy, trace_tau);
          auto tauzz = fma(two * rhovs2, duzdz, trace_tau);

          auto tauxy = rhovs2 * (duxdy + duydx);
          auto tauxz = rhovs2 * (duxdz + duzdx);
          auto tauyz = rhovs2 * (duydz + duzdy);

          // auto tmp = nsimd::gather<pf32>( &rg_hexa_gll_jacobian_det2[ id ],
          // vstrides );
          auto tmp = gather_linear<pf32>(&rg_hexa_gll_jacobian_det2[id], 125);

          storea(&intpx1[IDX3(m, l, k)],
                 tmp * fma(tauxx, dxidx, fma(tauxy, dxidy, tauxz * dxidz)));
          storea(&intpx2[IDX3(m, l, k)],
                 tmp * fma(tauxx, detdx, fma(tauxy, detdy, tauxz * detdz)));
          storea(&intpx3[IDX3(m, l, k)],
                 tmp * fma(tauxx, dzedx, fma(tauxy, dzedy, tauxz * dzedz)));

          storea(&intpy1[IDX3(m, l, k)],
                 tmp * fma(tauxy, dxidx, fma(tauyy, dxidy, tauyz * dxidz)));
          storea(&intpy2[IDX3(m, l, k)],
                 tmp * fma(tauxy, detdx, fma(tauyy, detdy, tauyz * detdz)));
          storea(&intpy3[IDX3(m, l, k)],
                 tmp * fma(tauxy, dzedx, fma(tauyy, dzedy, tauyz * dzedz)));

          storea(&intpz1[IDX3(m, l, k)],
                 tmp * fma(tauxz, dxidx, fma(tauyz, dxidy, tauzz * dxidz)));
          storea(&intpz2[IDX3(m, l, k)],
                 tmp * fma(tauxz, detdx, fma(tauyz, detdy, tauzz * detdz)));
          storea(&intpz3[IDX3(m, l, k)],
                 tmp * fma(tauxz, dzedx, fma(tauyz, dzedy, tauzz * dzedz)));
        }
      }
    }

    //tt_midloop2.push_back(tic() - tt);

    //tt = tic();

    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t l = 0; l < 5; ++l) {

        pf32 intpx10 = loada<pf32>(&intpx1[IDX3(0, l, k)]);
        pf32 intpx11 = loada<pf32>(&intpx1[IDX3(1, l, k)]);
        pf32 intpx12 = loada<pf32>(&intpx1[IDX3(2, l, k)]);
        pf32 intpx13 = loada<pf32>(&intpx1[IDX3(3, l, k)]);
        pf32 intpx14 = loada<pf32>(&intpx1[IDX3(4, l, k)]);

        pf32 intpy10 = loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy11 = loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy12 = loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy13 = loada<pf32>(&intpy1[IDX3(0, l, k)]);
        pf32 intpy14 = loada<pf32>(&intpy1[IDX3(0, l, k)]);

        pf32 intpz10 = loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz11 = loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz12 = loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz13 = loada<pf32>(&intpz1[IDX3(0, l, k)]);
        pf32 intpz14 = loada<pf32>(&intpz1[IDX3(0, l, k)]);

        pf32 lc0 = rg_gll_lagrange_deriv2[IDX2(l, 0)] * rg_gll_weight2[0];
        pf32 lc1 = rg_gll_lagrange_deriv2[IDX2(l, 1)] * rg_gll_weight2[1];
        pf32 lc2 = rg_gll_lagrange_deriv2[IDX2(l, 2)] * rg_gll_weight2[2];
        pf32 lc3 = rg_gll_lagrange_deriv2[IDX2(l, 3)] * rg_gll_weight2[3];
        pf32 lc4 = rg_gll_lagrange_deriv2[IDX2(l, 4)] * rg_gll_weight2[4];

        pf32 kc0 = rg_gll_lagrange_deriv2[IDX2(k, 0)] * rg_gll_weight2[0];
        pf32 kc1 = rg_gll_lagrange_deriv2[IDX2(k, 1)] * rg_gll_weight2[1];
        pf32 kc2 = rg_gll_lagrange_deriv2[IDX2(k, 2)] * rg_gll_weight2[2];
        pf32 kc3 = rg_gll_lagrange_deriv2[IDX2(k, 3)] * rg_gll_weight2[3];
        pf32 kc4 = rg_gll_lagrange_deriv2[IDX2(k, 4)] * rg_gll_weight2[4];

        for (std::size_t m = 0; m < 5; ++m) {
          auto c0 =
              set1<pf32>(rg_gll_lagrange_deriv2[IDX2(m, 0)] * rg_gll_weight2[0]);
          auto c1 =
              set1<pf32>(rg_gll_lagrange_deriv2[IDX2(m, 1)] * rg_gll_weight2[1]);
          auto c2 =
              set1<pf32>(rg_gll_lagrange_deriv2[IDX2(m, 2)] * rg_gll_weight2[2]);
          auto c3 =
              set1<pf32>(rg_gll_lagrange_deriv2[IDX2(m, 3)] * rg_gll_weight2[3]);
          auto c4 =
              set1<pf32>(rg_gll_lagrange_deriv2[IDX2(m, 4)] * rg_gll_weight2[4]);

          auto tmpx1 =
              fma(intpx10, c0,
                  fma(intpx11, c1,
                      fma(intpx12, c2, fma(intpx13, c3, intpx14 * c4))));

          auto tmpy1 =
              fma(intpy10, c0,
                  fma(intpy11, c1,
                      fma(intpy12, c2, fma(intpy13, c3, intpy14 * c4))));

          auto tmpz1 =
              fma(intpz10, c0,
                  fma(intpz11, c1,
                      fma(intpz12, c2, fma(intpz13, c3, intpz14 * c4))));

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

          /*
          c0 = rg_gll_lagrange_deriv2[IDX2(l, 0)] * rg_gll_weight2[0];
          c1 = rg_gll_lagrange_deriv2[IDX2(l, 1)] * rg_gll_weight2[1];
          c2 = rg_gll_lagrange_deriv2[IDX2(l, 2)] * rg_gll_weight2[2];
          c3 = rg_gll_lagrange_deriv2[IDX2(l, 3)] * rg_gll_weight2[3];
          c4 = rg_gll_lagrange_deriv2[IDX2(l, 4)] * rg_gll_weight2[4];
          */

          auto tmpx2 =
              fma(loada<pf32>(&intpx2[IDX3(m, 0, k)]), lc0,
                  fma(loada<pf32>(&intpx2[IDX3(m, 1, k)]), lc1,
                      fma(loada<pf32>(&intpx2[IDX3(m, 2, k)]), lc2,
                          fma(loada<pf32>(&intpx2[IDX3(m, 3, k)]), lc3,
                              loada<pf32>(&intpx2[IDX3(m, 4, k)]) * lc4))));

          auto tmpy2 =
              fma(loada<pf32>(&intpy2[IDX3(m, 0, k)]), lc0,
                  fma(loada<pf32>(&intpy2[IDX3(m, 1, k)]), lc1,
                      fma(loada<pf32>(&intpy2[IDX3(m, 2, k)]), lc2,
                          fma(loada<pf32>(&intpy2[IDX3(m, 3, k)]), lc3,
                              loada<pf32>(&intpy2[IDX3(m, 4, k)]) * lc4))));

          auto tmpz2 =
              fma(loada<pf32>(&intpz2[IDX3(m, 0, k)]), lc0,
                  fma(loada<pf32>(&intpz2[IDX3(m, 1, k)]), lc1,
                      fma(loada<pf32>(&intpz2[IDX3(m, 2, k)]), lc2,
                          fma(loada<pf32>(&intpz2[IDX3(m, 3, k)]), lc3,
                              loada<pf32>(&intpz2[IDX3(m, 4, k)]) * lc4))));

          /*
          c0 = rg_gll_lagrange_deriv2[IDX2(k, 0)] * rg_gll_weight2[0];
          c1 = rg_gll_lagrange_deriv2[IDX2(k, 1)] * rg_gll_weight2[1];
          c2 = rg_gll_lagrange_deriv2[IDX2(k, 2)] * rg_gll_weight2[2];
          c3 = rg_gll_lagrange_deriv2[IDX2(k, 3)] * rg_gll_weight2[3];
          c4 = rg_gll_lagrange_deriv2[IDX2(k, 4)] * rg_gll_weight2[4];
          */

          auto tmpx3 =
              fma(loada<pf32>(&intpx3[IDX3(m, l, 0)]), kc0,
                  fma(loada<pf32>(&intpx3[IDX3(m, l, 1)]), kc1,
                      fma(loada<pf32>(&intpx3[IDX3(m, l, 2)]), kc2,
                          fma(loada<pf32>(&intpx3[IDX3(m, l, 3)]), kc3,
                              loada<pf32>(&intpx3[IDX3(m, l, 4)]) * kc4))));

          auto tmpy3 =
              fma(loada<pf32>(&intpy3[IDX3(m, l, 0)]), kc0,
                  fma(loada<pf32>(&intpy3[IDX3(m, l, 1)]), kc1,
                      fma(loada<pf32>(&intpy3[IDX3(m, l, 2)]), kc2,
                          fma(loada<pf32>(&intpy3[IDX3(m, l, 3)]), kc3,
                              loada<pf32>(&intpy3[IDX3(m, l, 4)]) * kc4))));

          auto tmpz3 =
              fma(loada<pf32>(&intpz3[IDX3(m, l, 0)]), kc0,
                  fma(loada<pf32>(&intpz3[IDX3(m, l, 1)]), kc1,
                      fma(loada<pf32>(&intpz3[IDX3(m, l, 2)]), kc2,
                          fma(loada<pf32>(&intpz3[IDX3(m, l, 3)]), kc3,
                              loada<pf32>(&intpz3[IDX3(m, l, 4)]) * kc4))));

          //

          auto fac1 = set1<pf32>(rg_gll_weight2[l] * rg_gll_weight2[k]);
          auto fac2 = set1<pf32>(rg_gll_weight2[m] * rg_gll_weight2[k]);
          auto fac3 = set1<pf32>(rg_gll_weight2[m] * rg_gll_weight2[l]);

          auto rx = fma(fac1, tmpx1, fma(fac2, tmpx2, fac3 * tmpx3));
          auto ry = fma(fac1, tmpy1, fma(fac2, tmpy2, fac3 * tmpy3));
          auto rz = fma(fac1, tmpz1, fma(fac2, tmpz2, fac3 * tmpz3));

          auto lid = IDX3(m, l, k);
          auto gids = base + lid;

          // nsimd
          auto tmp = gather<pi32>((int *)(ig_hexa_gll_glonum.data()), gids);

          auto ids = 3u * (tmp - 1u);
          auto acc0 = gather<pf32>(&rg_gll_acceleration2[0], ids) - rx;
          auto acc1 = gather<pf32>(&rg_gll_acceleration2[1], ids) - ry;
          auto acc2 = gather<pf32>(&rg_gll_acceleration2[2], ids) - rz;

          scatter(&rg_gll_acceleration2[0], ids, acc0);
          scatter(&rg_gll_acceleration2[1], ids, acc1);
          scatter(&rg_gll_acceleration2[2], ids, acc2);
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
