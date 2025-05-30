#include "TTHelper.h"

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

void ttWrite(const string& filename, const MPS& tt, unsigned count) {
  auto f = count == 2 ? h5_open(filename, 'w') : h5_open(filename, 'a');
  h5_write(f, "tt_" + to_string(count - 1), tt);
}

MPS ttRead(const string& filename, unsigned count) {
  auto f = h5_open(filename, 'r');
  auto tt = h5_read<MPS>(f, "tt_" + to_string(count - 1));
  return tt;
}

void ttSumWrite(const string& filename, const MPS& tt, unsigned count) {
  auto f = h5_open(filename, 'a');
  h5_write(f, "vb_" + to_string(count - 1), tt);
}

MPS ttSumRead(const string& filename, unsigned count) {
  auto f = h5_open(filename, 'r');
  auto tt = h5_read<MPS>(f, "vb_" + to_string(count - 1));
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

tuple<Matrix<double>, vector<double>, double> covMat(const MPS& tt, const vector<BasisFunc>& basis) {
  int d = length(tt);
  auto s = siteInds(tt);
  vector<ITensor> basis_int0(d), basis_int1(d), basis_int2(d);
  for(int i = 1; i <= d; ++i) {
    basis_int0[i - 1] = basis_int1[i - 1] = basis_int2[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_int0[i - 1].set(s(i) = j, basis[i - 1].int0(j));
      basis_int1[i - 1].set(s(i) = j, basis[i - 1].int1(j));
      basis_int2[i - 1].set(s(i) = j, basis[i - 1].int2(j));
    }
  }
  auto Z = tt(1) * basis_int0[0];
  for(int i = 2; i <= d; ++i) {
    Z *= tt(i) * basis_int0[i - 1];
  }
  auto rho = tt;
  rho /= elt(Z);
  vector<double> ei(d), eii(d);
  vector<vector<double>> eij(d, vector<double>(d));
  for(int k = 1; k <= d; ++k) {
    auto eival = rho(1) * (k == 1 ? basis_int1[0] : basis_int0[0]);
    auto eiival = rho(1) * (k == 1 ? basis_int2[0] : basis_int0[0]);
    for(int i = 2; i <= d; ++i) {
      eival *= rho(i) * (k == i ? basis_int1[i - 1] : basis_int0[i - 1]);
      eiival *= rho(i) * (k == i ? basis_int2[i - 1] : basis_int0[i - 1]);
    }
    ei[k - 1] = elt(eival);
    eii[k - 1] = elt(eiival);
    for(int l = k + 1; l <= d; ++l) {
      auto eijval = rho(1) * (k == 1 ? basis_int1[0] : basis_int0[0]);
      for(int i = 2; i <= d; ++i) {
        eijval *= rho(i) * (k == i || l == i ? basis_int1[i - 1] : basis_int0[i - 1]);
      }
      eij[k - 1][l - 1] = elt(eijval);
    }
  }
  Matrix<double> sigma(d, d);
  for(int k = 1; k <= d; ++k) {
    for(int l = k; l <= d; ++l) {
      sigma(k - 1, l - 1) = sigma(l - 1, k - 1) = k == l ? eii[k - 1] - pow(ei[k - 1], 2) : eij[k - 1][l - 1] - ei[k - 1] * ei[l - 1];
    }
  }
  return make_tuple(sigma, ei, Z);
}

void marginal2d(const MPS& tt, const vector<BasisFunc>& basis, int pos1, int pos2, vector<vector<double>>& grid, bool conv) {
  int bins = grid.size();
  int d = length(tt);
  auto s = siteInds(tt);
  vector<ITensor> basis_int0(d);
  for(int i = 1; i <= d; ++i) {
    basis_int0[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_int0[i - 1].set(s(i) = j, basis[i - 1].int0(j));
    }
  }
  auto Z = tt(1) * basis_int0[0];
  for(int i = 2; i <= d; ++i) {
    Z *= tt(i) * basis_int0[i - 1];
  }
  auto rho = tt;
  rho /= elt(Z);
  for(int i = 0; i < bins; ++i) {
    for(int j = 0; j < bins; ++j) {
      double x = basis[pos1 - 1].dom().first + i * (basis[pos1 - 1].dom().second - basis[pos1 - 1].dom().first) / bins;
      double y = basis[pos2 - 1].dom().first + j * (basis[pos2 - 1].dom().second - basis[pos2 - 1].dom().first) / bins;
      ITensor xevals(s(pos1)), yevals(s(pos2));
      for(int k = 1; k <= dim(s(pos1)); ++k) {
        xevals.set(s(pos1) = k, basis[pos1 - 1](x, k, conv));
        yevals.set(s(pos2) = k, basis[pos2 - 1](y, k, conv));
      }
      auto val = rho(1) * (pos1 == 1 ? xevals : basis_int0[0]);
      for(int k = 2; k <= d; ++k) {
        if(pos1 == k) {
          val *= rho(k) * xevals;
        } else if(pos2 == k) {
          val *= rho(k) * yevals;
        } else {
          val *= rho(k) * basis_int0[k - 1];
        }
      }
      grid[i][j] = elt(val);
    }
  }
}

}
}
