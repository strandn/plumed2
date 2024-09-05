#ifndef __PLUMED_ttsketch_BasisFunc_h
#define __PLUMED_ttsketch_BasisFunc_h
#include <utility>
#include <vector>

namespace PLMD {
namespace ttsketch {

class BasisFunc {

private:
  std::pair<double, double> dom_;
  int nbasis_;
  int nbins_;
  double L_;
  double shift_;
  std::vector<std::vector<double>> grid_;
  std::vector<std::vector<double>> gridd_;
  std::vector<double> xdata_;
  double w_;

public:
  BasisFunc();
  BasisFunc(std::pair<double, double> dom, int nbasis, bool conv, int nbins,
            double w, int conv_n, double conv_epsabs, double conv_epsrel,
            int conv_limit, int conv_key);
  double fourier(double x, int pos) const;
  double operator()(double x, int pos, bool conv) const;
  double grad(double x, int pos, bool conv) const;
  double interpolate(double x, int pos, bool grad) const;
  int nbasis() const { return this->nbasis_; }
  const std::pair<double, double>& dom() const { return this->dom_; }
  int nbins() const { return this->nbins_; }
  double w() const { return this->w_; }
};

}
}

#endif
