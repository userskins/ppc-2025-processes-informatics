#include "gusev_d_radix_double/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "gusev_d_radix_double/common/include/common.hpp"
#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

namespace gusev_d_radix_double {

GusevDRadixDoubleMPI::GusevDRadixDoubleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GusevDRadixDoubleMPI::ValidationImpl() {
  return true;
}

bool GusevDRadixDoubleMPI::PreProcessingImpl() {
  return true;
}

bool GusevDRadixDoubleMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  int remainder = n % size;
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = (n / size) + (i < remainder ? 1 : 0);
    displs[i] = sum;
    sum += send_counts[i];
  }

  std::vector<double> local_data(send_counts[rank]);

  MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_data.data(), send_counts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GusevDRadixDoubleSEQ::RadixSort(local_data);

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      int source = rank + step;
      if (source < size) {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        size_t current_size = local_data.size();
        local_data.resize(current_size + recv_count);

        MPI_Recv(local_data.data() + current_size, recv_count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        auto mid_iter = local_data.begin() + static_cast<std::ptrdiff_t>(current_size);
        std::ranges::inplace_merge(local_data, mid_iter);
      }
    } else if (rank % (2 * step) == step) {
      int dest = rank - step;
      int count = static_cast<int>(local_data.size());

      MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(local_data.data(), count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

      local_data.clear();
      local_data.shrink_to_fit();
      break;
    }
    step *= 2;
  }

  if (rank == 0) {
    GetOutput() = std::move(local_data);
  }

  return true;
}

bool GusevDRadixDoubleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_radix_double
