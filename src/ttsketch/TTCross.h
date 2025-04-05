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
  std::vector<BasisFunc> basisg_;
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
  Log* log_;
  bool conv_;
  bool convg_;
  int nbins_;
  std::vector<std::vector<double>> grid_;
  std::vector<std::vector<double>> samples_;
  std::vector<std::vector<double>> pivots_;
  bool walkers_mpi_;
  int mpi_rank_;
  bool auto_rank_;
  OFile* pivot_file_;
  std::vector<Value*> args_;

public:
  TTCross();
  TTCross(const std::vector<BasisFunc>& basis,
          const std::vector<BasisFunc>& basisg, double kbt, double cutoff,
          int maxrank, Log& log, bool conv, bool convg, int nbins,
          bool walkers_mpi, int mpi_rank, bool auto_rank, OFile& pivot_file,
          std::vector<Value*>& args);
  double f(const std::vector<double>& x) const;
  void updateIJ(const std::vector<double>& ij);
  std::pair<double, int> diagACA(const std::vector<double>& Rk, std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) const;
  void continuousACA();
  void updateG(const itensor::MPS& G) { this->G_ = &G; }
  void updateVshift(double vshift) { this->vshift_ = vshift; }
  void updateVb();
  std::pair<double, std::vector<double>> vtop();
  void reset();
  void writeVb(unsigned count) const;
  void readVb(unsigned count);
  void addSample(std::vector<double>& sample);
  void trimSamples(int max) { this->samples_.erase(this->samples_.begin(), this->samples_.begin() + (this->samples_.size() - max)); }
  bool conv() const { return this->conv_; }
  const itensor::MPS& vb() const { return this->vb_; }
  const std::vector<std::vector<double>>& aca_samples() { return this->samples_; }
  void approximate(std::vector<double>& approx);
  void addPivot(std::vector<double>& pivot) { this->pivots_.push_back(pivot); }
  void prependPivots() { this->samples_.insert(this->samples_.begin(), this->pivots_.begin(), this->pivots_.end()); }
  const std::vector<std::vector<double>>& aca_pivots() { return this->pivots_; }
};

}
}

#endif
