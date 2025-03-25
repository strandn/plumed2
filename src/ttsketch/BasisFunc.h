#ifndef __PLUMED_ttsketch_BasisFunc_h
#define __PLUMED_ttsketch_BasisFunc_h
#include "tools/Matrix.h"
#include <utility>
#include <vector>

namespace PLMD {
namespace ttsketch {

class BasisFunc {

private:
  std::pair<double, double> dom_;
  int nbasis_;
  double L_;
  double shift_;
  double w_;
  bool kernel_;
  double dx_;
  std::vector<double> centers_;
  Matrix<double> ginv_;
  Matrix<double> gram_;

public:
  BasisFunc();
  BasisFunc(std::pair<double, double> dom, int nbasis, bool conv, double w, bool kernel);
  double fourier(double x, int pos) const;
  double gaussian(double x, int pos) const;
  double fourierd(double x, int pos) const;
  double gaussiand(double x, int pos) const;
  double operator()(double x, int pos, bool conv) const;
  double grad(double x, int pos, bool conv) const;
  int nbasis() const { return this->nbasis_; }
  const std::pair<double, double>& dom() const { return this->dom_; }
  double w() const { return this->w_; }
  double int0(int pos) const;
  double int1(int pos) const;
  double int2(int pos) const;
  bool kernel() const { return this->kernel_; }
  const Matrix<double>& ginv() const { return this->ginv_; }
  const Matrix<double>& gram() const { return this->gram_; }
  void test() const;
  double center(int pos) const { return this->kernel_ ? this->centers_[pos - 1] : 0.0; }
  double dx() const { return this->dx_; }
};

}
}

#endif
