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
  double w_;
  int gsl_n_;
  double gsl_epsabs_;
  double gsl_epsrel_;
  int gsl_limit_;
  int gsl_key_;

public:
  BasisFunc();
  BasisFunc(std::pair<double, double> dom, int nbasis, bool conv, int nbins,
            double w, int gsl_n, double gsl_epsabs, double gsl_epsrel,
            int gsl_limit, int gsl_key);
  double fourier(double x, int pos) const;
  double operator()(double x, int pos) const;
  double grad(double x, int pos) const;
  double conv() const { return this->conv_; };
  void setConv(bool status) { this->conv_ = status; }
  double interpolate(double x, int pos, bool grad) const;
  int nbasis() const { return this->nbasis_; }
  std::pair<double, double> const& dom() const { return this->dom_; }
  int nbins() const { return this->nbins_; }
  double w() const { return this->w_; }
};

}
}

#endif
