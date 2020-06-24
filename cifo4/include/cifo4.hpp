#include <vector>
#include <cstdint>
#include <ctime>
#include <algorithm>

#include <boost/align/aligned_allocator.hpp>


extern std::vector< uint32_t, boost::alignment::aligned_allocator< uint32_t, 32 > > ig_hexa_gll_glonum;

extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_displacement;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_weight;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_lagrange_deriv;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_gll_acceleration;

extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dxidz;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_detdz;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_dzedz;

extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_rhovp2;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_rhovs2;
extern std::vector< float, boost::alignment::aligned_allocator< float, 32 > > rg_hexa_gll_jacobian_det;


void compute_internal_forces_order4( std::size_t elt_start, std::size_t elt_end );

inline double tic() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_nsec + (double)ts.tv_sec * 10e9;
}

inline double avg(std::vector<double> const &v) {
  double acc = 0.0;
  std::vector<double> tmp(v);
  std::sort(tmp.begin(), tmp.end());
  size_t middle = tmp.size() / 2;
  for (size_t i = middle - 20; i < middle + 20; i++) {
    acc += tmp[i];
  }
  return acc / 40.0;
}
