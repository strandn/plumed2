#ifndef __PLUMED_ttsketch_TTCross_h
#define __PLUMED_ttsketch_TTCross_h
#include "BasisFunc.h"
#include "core/ActionSet.h"
#include "itensor/all.h"

namespace PLMD {
namespace ttsketch {

class TTCross {

private:
  MPS vb_;
  const MPS* G_;
  const std::vector<BasisFunc>& basis_;
  int n_;
  double kbt_;
  double cutoff_;
  int maxrank_;
  int d_;
  int pos_;
  double vshift_;
  std::vector<std::vector<std::vector<double>>> I_;
  std::vector<std::vector<std::vector<double>>> J_;
  std::vector<std::vector<double>> u_;
  std::vector<std::vector<double>> v_;
  std::vector<double> resfirst_;
  Log& log_;
  int aca_n_;
  double aca_epsabs_;
  double aca_epsrel_;
  int aca_limit_;
  int aca_key_;

public:
  TTCross(const std::vector<BasisFunc>& basis, double kbt, double cutoff,
          int maxrank, Log& log, int aca_n, int aca_epsabs, double aca_epsrel,
          int aca_limit, int aca_key);
  double f(const std::vector<double>& x) const;
  void updateIJ(const std::vector<double>& ij);
  std::pair<double, int> diagACA(const std::vector<std::vector<double>>& samples, const std::vector<double>& Rk);
  void continuousACA(const std::vector<std::vector<double>>& samples);
  void updateG(const MPS& G) { this->G_ = &G; }
  void updateVshift(double vshift) { this->vshift_ = vshift; }
  void updateVb(const std::vector<std::vector<double>>& samples);
  double eval(const std::vector<double>& elements, bool isG) const;
  double vtop(const std::vector<std::vector<double>>& samples) const;
  const std::vector<std::vector<std::vector<double>>>& I() const { return this->I_; }
  const std::vector<std::vector<std::vector<double>>>& J() const { return this->J_; }
  int d() const { return this->d_; }
  const BasisFunc& basisi(int i) const { return basis_[i]; }
  void reset();
};

}
}

#endif
