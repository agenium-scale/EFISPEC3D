// Copyright (C) 2021  Sylvain Jubertie, Agenium Scale
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

// ----------------------------------------------------------------------------

#include <vector>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <algorithm>

// ----------------------------------------------------------------------------
// C++11 allocator

template <typename T> struct allocator {
  using value_type = T;

  allocator() = default;

  template <typename S> allocator(allocator<S> const &) {}

  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T *ptr;
    if (::posix_memalign((void **)&ptr, 32, n * sizeof(T)) == 0) {
      return ptr;
    }
    throw std::bad_alloc();
  }

  void deallocate(T *ptr, std::size_t) { ::free((void *)ptr); }
};

template <class T, class S>
bool operator==(allocator<T> const &, allocator<S> const &) {
  return true;
}

template <class T, class S>
bool operator!=(allocator<T> const &, allocator<S> const &) {
  return false;
}

// ----------------------------------------------------------------------------

extern std::vector< uint32_t, allocator< uint32_t > > ig_hexa_gll_glonum;

extern std::vector< float, allocator< float > > rg_gll_displacement;
extern std::vector< float, allocator< float > > rg_gll_weight;
extern std::vector< float, allocator< float > > rg_gll_lagrange_deriv;
extern std::vector< float, allocator< float > > rg_gll_acceleration;

extern std::vector< float, allocator< float > > rg_hexa_gll_dxidx;
extern std::vector< float, allocator< float > > rg_hexa_gll_dxidy;
extern std::vector< float, allocator< float > > rg_hexa_gll_dxidz;
extern std::vector< float, allocator< float > > rg_hexa_gll_detdx;
extern std::vector< float, allocator< float > > rg_hexa_gll_detdy;
extern std::vector< float, allocator< float > > rg_hexa_gll_detdz;
extern std::vector< float, allocator< float > > rg_hexa_gll_dzedx;
extern std::vector< float, allocator< float > > rg_hexa_gll_dzedy;
extern std::vector< float, allocator< float > > rg_hexa_gll_dzedz;

extern std::vector< float, allocator< float > > rg_hexa_gll_rhovp2;
extern std::vector< float, allocator< float > > rg_hexa_gll_rhovs2;
extern std::vector< float, allocator< float > > rg_hexa_gll_jacobian_det;


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
