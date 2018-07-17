/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2018, The Simons Foundation
 * Author: H. U.R. Strand
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

#include <triqs/mpi/base.hpp>
#include <triqs/utility/itertools.hpp>

#include "types.hpp"

namespace tprf {

template<class T>
auto mpi_view(const array<T, 1> &arr, triqs::mpi::communicator const & c) {

  auto slice = triqs::mpi::slice_range(0, arr.shape()[0] - 1, c.size(), c.rank()); // NB! needs [first, last] in range

  //std::cout << "mpi_view<array> " << "rank = " << c.rank() << " size = " << arr.shape()[0]
  //	    << " s,e = " << slice.first << ", " << slice.second << "\n";

  return arr(range(slice.first, slice.second + 1));
}

template<class T>
auto mpi_view(const array<T, 1> &arr) {
  triqs::mpi::communicator c;
  return mpi_view(arr, c);
}
  
template<class T>
auto mpi_view(const gf_mesh<T> &mesh, triqs::mpi::communicator const & c) {

  auto slice = triqs::mpi::slice_range(0, mesh.size() - 1, c.size(), c.rank()); // NB! needs [first, last] in range
  int size = slice.second + 1 - slice.first;

  //std::cout << "mpi_view<mesh> " << "rank = " << c.rank() << " size = " << size
  //	    << " s,e = " << slice.first << ", " << slice.second << "\n";
  
  array<mesh_point<gf_mesh<T>>, 1> arr(size);

  auto iter = mesh.begin();
  iter += slice.first;

  for ( auto idx : range(0, size) ) {
    auto w = *iter;
    arr(idx) = w;
    iter++;
  }
  
  return arr;
}

template<class T>
auto mpi_view(const gf_mesh<T> &mesh) {
  triqs::mpi::communicator c;
  return mpi_view(mesh, c);
}
  
} // namespace tprf
