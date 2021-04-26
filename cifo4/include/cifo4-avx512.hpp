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

#include <vector>
#include <cstdint>

#include <boost/align/aligned_allocator.hpp>


extern std::vector< uint32_t, boost::alignment::aligned_allocator< uint32_t, 64 > > ig_hexa_gll_glonum;

extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_gll_displacement;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_gll_weight;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_gll_lagrange_deriv;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_gll_acceleration;

extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dxidx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dxidy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dxidz;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_detdx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_detdy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_detdz;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dzedx;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dzedy;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_dzedz;

extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_rhovp2;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_rhovs2;
extern std::vector< float, boost::alignment::aligned_allocator< float, 64 > > rg_hexa_gll_jacobian_det;


void compute_internal_forces_order4( std::size_t elt_start, std::size_t elt_end );
