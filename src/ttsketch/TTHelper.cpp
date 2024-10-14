#include "TTHelper.h"

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

void ttWrite(const string& filename, const MPS& tt, unsigned count) {
  auto f = count == 2 ? h5_open(filename, 'w') : h5_open(filename, 'a');
  h5_write(f, "tt_" + to_string(count - 1), tt);
  close(f);
  file.close();
}

MPS ttRead(const string& filename, unsigned count) {
  auto f = h5_open(filename, 'r');
  auto tt = h5_read<MPS>(f, "tt_" + to_string(count - 1));
  close(f);
  return tt;
}

double ttEval(const MPS& tt, const vector<BasisFunc>& basis, const vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  vector<ITensor> basis_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = ITensor(s(i));
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

vector<double> ttGrad(const MPS& tt, const vector<BasisFunc>& basis, const vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  vector<double> grad(d, 0.0);
  vector<ITensor> basis_evals(d), basisd_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = basisd_evals[i - 1] = ITensor(s(i));
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
