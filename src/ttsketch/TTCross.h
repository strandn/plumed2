#ifndef __PLUMED_ttsketch_TTCross_h
#define __PLUMED_ttsketch_TTCross_h
#include "BasisFunc.h"
#include "core/ActionSet.h"
#include "itensor/all.h"

namespace PLMD {
namespace ttsketch {

class TTCross {

private:
  itensor::MPS vb_;
  const itensor::MPS* G_;
  std::vector<BasisFunc> basis_;
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
  Log* log_;
  bool conv_;
  int nbins_;
  std::vector<std::vector<double>> grid_;
  vector<vector<double>> samples_;
  bool walkers_mpi_;

public:
  TTCross();
  TTCross(const std::vector<BasisFunc>& basis, double kbt, double cutoff, int maxrank, Log& log, bool conv, int nbins, bool walkers_mpi);
  double f(const std::vector<double>& x) const;
  void updateIJ(const std::vector<double>& ij);
  std::pair<double, int> diagACA(const std::vector<double>& Rk);
  void continuousACA();
  void updateG(const itensor::MPS& G) { this->G_ = &G; }
  void updateVshift(double vshift) { this->vshift_ = vshift; }
  void updateVb();
  std::pair<double, std::vector<double>> vtop() const;
  void reset();
  void writeVb(unsigned count) const;
  void readVb(unsigned count);
  void addSample(std::vector<double>& sample);
  bool conv() const { return this->conv_; }
};

}
}

#endif
