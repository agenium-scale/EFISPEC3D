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

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <cifo4.hpp>

// ----------------------------------------------------------------------------

template <typename T>
void vfill(std::string const filename, std::vector<T, allocator<T> > &v) {
  uint32_t dims = 0;
  std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);

  ifs.read(reinterpret_cast<char *>(&dims), sizeof(uint32_t));

  std::size_t size = 1;
  for (uint32_t i = 0; i < dims; ++i) {
    uint32_t dim;
    ifs.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
    size *= dim;
  }

  v.resize(size);

  ifs.read(reinterpret_cast<char *>(v.data()), (long)(size * sizeof(T)));
}

// ----------------------------------------------------------------------------

template <typename T>
void save(std::string const filename, std::vector<T, allocator<T> > &v) {
  uint32_t dims = 2;
  uint32_t dim = 3;
  uint32_t size = (uint32_t)(v.size() / 3);

  std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
  ofs.write(reinterpret_cast<char *>(&dims), sizeof(dims));
  ofs.write(reinterpret_cast<char *>(&dim), sizeof(dim));
  ofs.write(reinterpret_cast<char *>(&size), sizeof(size));
  ofs.write(reinterpret_cast<char *>(v.data()), (long)(v.size() * sizeof(T)));
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  vfill("../data/ig_hexa_gll_glonum.dat", ig_hexa_gll_glonum);

  vfill("../data/rg_gll_displacement.dat", rg_gll_displacement);
  vfill("../data/rg_gll_lagrange_deriv.dat", rg_gll_lagrange_deriv);

  vfill("../data/rg_hexa_gll_dxidx.dat", rg_hexa_gll_dxidx);
  vfill("../data/rg_hexa_gll_dxidy.dat", rg_hexa_gll_dxidy);
  vfill("../data/rg_hexa_gll_dxidz.dat", rg_hexa_gll_dxidz);

  vfill("../data/rg_hexa_gll_detdx.dat", rg_hexa_gll_detdx);
  vfill("../data/rg_hexa_gll_detdy.dat", rg_hexa_gll_detdy);
  vfill("../data/rg_hexa_gll_detdz.dat", rg_hexa_gll_detdz);

  vfill("../data/rg_hexa_gll_dzedx.dat", rg_hexa_gll_dzedx);
  vfill("../data/rg_hexa_gll_dzedy.dat", rg_hexa_gll_dzedy);
  vfill("../data/rg_hexa_gll_dzedz.dat", rg_hexa_gll_dzedz);

  vfill("../data/rg_hexa_gll_rhovp2.dat", rg_hexa_gll_rhovp2);
  vfill("../data/rg_hexa_gll_rhovs2.dat", rg_hexa_gll_rhovs2);
  vfill("../data/rg_hexa_gll_jacobian_det.dat", rg_hexa_gll_jacobian_det);
  vfill("../data/rg_gll_weight.dat", rg_gll_weight);
  vfill("../data/rg_gll_acceleration_before_loop_iel.dat", rg_gll_acceleration);

  std::vector<float, allocator<float> > out_ref;
  vfill("data/rg_gll_acceleration_after_loop_iel.dat", out_ref);
  // vfill( "main-f90.dat", out_ref );

  auto start = std::chrono::system_clock::now();

  compute_internal_forces_order4(0, ig_hexa_gll_glonum.size() / 125);

  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;

  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << std::endl;

  /*
    // Display results.
    for( std::size_t i = 0 ; i < out_ref.size() ; ++i )
    {
      std::cout << i << '\t' << out_ref[ i ] << '\t' << rg_gll_acceleration[ i
    ] << '\t' << out_ref[ i ] - rg_gll_acceleration[ i ] << std::endl;
    }
  */
  if (argc == 2) {
    save(std::string(argv[0]) + ".dat", rg_gll_acceleration);
  }

  return 0;
}
