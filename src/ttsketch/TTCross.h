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
  std::vector<double> resfirst_;
  Log& log_;

public:
  TTCross(const std::vector<BasisFunc>& basis, double kbt, double cutoff, int maxrank, Log& log);
  double f(const std::vector<double>& x) const;
  double operator()(const std::vector<double>& elements) const;
  void updateIJ(const std::vector<double>& ij);
  void continuousACA(const std::vector<std::vector<double>>& samples);
  void updateG(const MPS& G) { this->G_ = &G; }
  void updateVshift(double vshift) { this->vshift_ = vshift; }
  void updateVb(const std::vector<std::vector<double>>& samples);
  double eval(const std::vector<double>& elements, bool isG) const;
};

}
}

#endif
