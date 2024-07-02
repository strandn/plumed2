#ifndef __PLUMED_ttsketch_BasisFunc_h
#define __PLUMED_ttsketch_BasisFunc_h

namespace PLMD {
namespace ttsketch {

class BasisFunc {

private:
  std::pair<double, double> dom_;
  int nbasis_;
  bool conv_;
  int nbins_;
  double L_;
  double shift_;
  std::vector<std::vector<double>> grid_;
  std::vector<std::vector<double>> gridd_;
  std::vector<double> xdata_;

public:
  BasisFunc();
  BasisFunc(std::pair<double, double> dom, int nbasis = 20, int nbins = 100);
  double fourier(double x, int pos) const;
  double operator()(double x, int pos) const;
  double grad(double x, int pos) const;
  void setConv(bool status) { this->conv_ = status; }
  double interpolate(double x, int pos, bool grad) const;
  int nbasis() const { return this->nbasis_; }
  std::pair<double, double> const& dom() const { return this->dom_; }
  int nbins() const { return this->nbins_; }
};

}
}

#endif
