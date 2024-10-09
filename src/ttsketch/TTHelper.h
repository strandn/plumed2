#include "BasisFunc.h"
#include "itensor/all.h"

namespace PLMD {
namespace ttsketch {

void ttWrite(const itensor::MPS& tt, unsigned count) {
  auto f = count == 2 ? h5_open("ttsketch.h5", 'w') : h5_open("ttsketch.h5", 'a');
  h5_write(f, "tt_" + to_string(count - 1), tt);
  close(f);
}

itensor::MPS ttRead(unsigned count) {
  auto f = h5_open("ttsketch.h5", 'r');
  auto tt = h5_read<itensor::MPS>(f, "tt_" + to_string(count - 1));
  close(f);
  return tt;
}

double ttEval(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  std::vector<itensor::ITensor> basis_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = itensor::ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis[i - 1](elements[i - 1], j, conv));
    }
  }
  auto result = tt(1) * basis_evals[0];
  for(int i = 2; i <= d; ++i) {
    result *= tt(i) * basis_evals[i - 1];
  }
  return elt(result);
}

std::vector<double> ttGrad(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  std::vector<double> grad(d, 0.0);
  std::vector<itensor::ITensor> basis_evals(d), basisd_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = basisd_evals[i - 1] = itensor::ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis[i - 1](elements[i - 1], j, conv));
      basisd_evals[i - 1].set(s(i) = j, basis[i - 1].grad(elements[i - 1], j, conv));
    }
  }
  for(int k = 1; k <= d; ++k) {
    auto result = tt(1) * (k == 1 ? basisd_evals[0] : basis_evals[0]);
    for(int i = 2; i <= d; ++i) {
      result *= tt(i) * (k == i ? basisd_evals[i - 1] : basis_evals[i - 1]);
    }
    grad[k - 1] = elt(result);
  }
  return grad;
}

}
}
