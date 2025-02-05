#include "BasisFunc.h"
#include <gsl/gsl_integration.h>
#include <iostream>

using namespace std;

namespace PLMD {
namespace ttsketch {

struct ConvParams {
  const BasisFunc* instance;
  int j;
  int k;
};

double conv_f(double s, void* params) {
  ConvParams* conv_params = (ConvParams*)params;
  auto& dom = conv_params->instance->dom();
  int nbins = conv_params->instance->nbins();
  int j = conv_params->j;
  int k = conv_params->k;
  double fourier = conv_params->instance->fourier(s, j);
  double sigma = conv_params->instance->w();
  double x = dom.first + (k - 1) * (dom.second - dom.first) / (nbins - 1);
  return fourier * (1 / (sqrt(2 * M_PI) * sigma)) * exp(-pow(x - s, 2) / (2 * pow(sigma, 2)));
}

double conv_df(double s, void* params) {
  ConvParams* conv_params = (ConvParams*)params;
  auto& dom = conv_params->instance->dom();
  int nbins = conv_params->instance->nbins();
  int j = conv_params->j;
  int k = conv_params->k;
  double fourier = conv_params->instance->fourier(s, j);
  double sigma = conv_params->instance->w();
  double x = dom.first + (k - 1) * (dom.second - dom.first) / (nbins - 1);
  return fourier * ((s - x) / (sqrt(2 * M_PI) * pow(sigma, 3))) * exp(-pow(x - s, 2) / (2 * pow(sigma, 2)));
}

BasisFunc::BasisFunc()
  : dom_(make_pair(0.0, 0.0)), nbasis_(0), nbins_(0), L_(0.0), shift_(0.0), w_(0.0), kernel_(false), dx_(0.0) {}

BasisFunc::BasisFunc(pair<double, double> dom, int nbasis, bool conv,
                     int nbins, double w, int conv_n, double conv_epsabs,
                     double conv_epsrel, int conv_limit, int conv_key,
                     bool kernel)
  : dom_(dom), nbasis_(nbasis), nbins_(conv ? nbins : 0),
    L_((dom.second - dom.first) / 2), shift_((dom.second + dom.first) / 2),
    grid_(nbasis, vector<double>(nbins, 0.0)),
    gridd_(nbasis, vector<double>(nbins, 0.0)), xdata_(nbins, 0.0), w_(w),
    kernel_(kernel)
{
  if(kernel) {
    this->dx_ = (dom.second - dom.first) / (nbasis - 1);
    this->centers_ = vector<double>(nbasis - 1);
    for(int i = 0; i < nbasis - 1; ++i) {
      this->centers_[i] = dom.first + i * this->dx_;
    }
    this->gram_ = Matrix<double>(nbasis, nbasis);
    this->gram_(0, 0) = this->dom_.second - this->dom_.first;
    for(int i = 1; i < nbasis; ++i) {
      this->gram_(i, 0) = this->gram_(0, i) = this->dx_ * sqrt(M_PI / 2) *
                                (erf((this->dom_.second -
                                2 * this->dom_.first + this->centers_[i - 1]) /
                                (sqrt(2) * this->dx_)) -
                                erf((this->dom_.first - 2 * this->dom_.second +
                                this->centers_[i - 1]) /
                                (sqrt(2) * this->dx_)));
      for(int j = i; j < nbasis; ++j) {
        double result = 0.0;
        for(int k = -1; k <= 1; ++k) {
          for(int l = -1; l <= 1; ++l) {
            result += this->dx_ / 2 * exp(-pow((this->dom_.first -
                      this->dom_.second) * (k - l) + this->centers_[i - 1] -
                      this->centers_[j - 1], 2) / (4 * pow(this->dx_, 2))) *
                      sqrt(M_PI) * (erf((this->dom_.first * (k + l - 2) -
                      this->dom_.second * (k + l) + this->centers_[i - 1] +
                      this->centers_[j - 1]) / (2 * this->dx_)) -
                      erf((this->dom_.first * (k + l) - this->dom_.second *
                      (k + l + 2) + this->centers_[i - 1] +
                      this->centers_[j - 1]) / (2 * this->dx_)));
          }
        }
        this->gram_(i, j) = this->gram_(j, i) = result;
      }
    }
    Invert(this->gram_, this->ginv_);
  } else if(nbins > 0) {
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(conv_n);
    double result, error;
    for(int j = 0; j < nbasis; ++j) {
      for(int k = 0; k < nbins; ++k) {
        ConvParams conv_params = { this, j + 1, k + 1 };
        gsl_function F;
        F.function = &conv_f;
        F.params = &conv_params;
        gsl_integration_qag(&F, dom.first - this->L_,
                            dom.second + this->L_, conv_epsabs,
                            conv_epsrel, conv_limit, conv_key, workspace,
                            &result, &error);
        this->grid_[j][k] = result;
        gsl_function DF;
        DF.function = &conv_df;
        DF.params = &conv_params;
        gsl_integration_qag(&DF, dom.first - this->L_,
                            dom.second + this->L_, conv_epsabs,
                            conv_epsrel, conv_limit, conv_key, workspace,
                            &result, &error);
        this->gridd_[j][k] = result;
      }
    }
    gsl_integration_workspace_free(workspace);

    for(int i = 0; i < nbins; ++i) {
      this->xdata_[i] = dom.first + i * (dom.second - dom.first) / (nbins - 1);
    }
  }
}

double BasisFunc::fourier(double x, int pos) const {
  if(pos == 1) {
    return 1 / sqrt(2 * this->L_);
  } else if(pos % 2 == 0) {
    return sqrt(1 / this->L_) * cos(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  } else {
    return sqrt(1 / this->L_) * sin(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  }
}

double BasisFunc::gaussian(double x, int pos) const {
  if(pos == 1) {
    return 1.0;
  } else {
    double result = 0.0;
    for(int k = -1; k <= 1; ++k) {
      result += exp(-pow(x - this->centers_[pos - 2] + 2 * k * this->L_, 2) / (2 * pow(this->dx_, 2)));
    }
    return result;
  }
}

double BasisFunc::fourierd(double x, int pos) const {
  if(pos == 1) {
    return 0.0;
  } else if(pos % 2 == 0) {
    return -pow(1 / this->L_, 3 / 2) * M_PI * (pos / 2) * sin(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  } else {
    return pow(1 / this->L_, 3 / 2) * M_PI * (pos / 2) * cos(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  }
}

double BasisFunc::gaussiand(double x, int pos) const {
  if(pos == 1) {
    return 0.0;
  } else {
    double result = 0.0;
    for(int k = -1; k <= 1; ++k) {
      result += (this->centers_[pos - 2] - x - 2 * k * this->L_) /
                pow(this->dx_, 2) * exp(-pow(x - this->centers_[pos - 2] +
                2 * k * this->L_, 2) / (2 * pow(this->dx_, 2)));
    }
    return result;
  }
}

double BasisFunc::operator()(double x, int pos, bool conv) const {
  if(this->kernel_) {
    if(conv) {
      if(pos == 1) {
        return 1.0;
      } else {
        double result = 0.0;
        for(int k = -1; k <= 1; ++k) {
          result += exp(-pow(x - this->centers_[pos - 2] +
                    2 * k * this->L_, 2) / (2 * (pow(this->dx_, 2) +
                    pow(this->w_, 2)))) / (sqrt(1 / pow(this->dx_, 2) + 1 /
                    pow(this->w_, 2)) * this->w_);
        }
        return result;
      }
    } else {
      return gaussian(x, pos);
    }
  } else {
    if(conv && this->nbins_ > 0) {
      return interpolate(x, pos, false);
    } else {
      return fourier(x, pos);
    }
  }
}

double BasisFunc::grad(double x, int pos, bool conv) const {
  if(this->kernel_) {
    if(conv) {
      if(pos == 1) {
        return 0.0;
      }
      else {
        double result = 0.0;
        for(int k = -1; k <= 1; ++k) {
          result += pow(this->dx_, 2) * exp(-pow(x - this->centers_[pos - 2] +
                    2 * k * this->L_, 2) / (2 * (pow(this->dx_, 2) +
                    pow(this->w_, 2)))) * (this->centers_[pos - 2] -
                    2 * k * this->L_ - x) * sqrt(1 / pow(this->dx_, 2) +
                    1 / pow(this->w_, 2)) * this->w_ / pow(pow(this->dx_, 2) +
                    pow(this->w_, 2), 2);
        }
        return result;
      }
    } else {
      return gaussiand(x, pos);
    }
  } else {
    if(conv && this->nbins_ > 0) {
      return interpolate(x, pos, true);
    } else {
      return fourierd(x, pos);
    }
  }
}

double BasisFunc::interpolate(double x, int pos, bool grad) const {
  int i = 0;
  if(x >= this->xdata_[this->nbins_ - 2]) {
    i = this->nbins_ - 2;
  } else {
    while(x > this->xdata_[i + 1]) {
      ++i;
    }
  }
  double xL = this->xdata_[i];
  double yL = grad ? this->gridd_[pos - 1][i] : this->grid_[pos - 1][i];
  double xR = this->xdata_[i + 1];
  double yR = grad ? this->gridd_[pos - 1][i + 1] : this->grid_[pos - 1][i + 1];
  double dydx = (yR - yL) / (xR - xL);
  return yL + dydx * (x - xL);
}

double BasisFunc::int0(int pos) const {
  if(this->kernel_) {
    if(pos == 1) {
      return this->dom_.second - this->dom_.first;
    } else {
      return -this->dx_ * sqrt(M_PI / 2) * (erf((2 * this->dom_.first -
             this->dom_.second - this->centers_[pos - 2]) /
             (sqrt(2) * this->dx_)) + erf((this->dom_.first -
             2 * this->dom_.second + this->centers_[pos - 2]) /
             (sqrt(2) * this->dx_)));
    }
  } else {
    if(pos == 1) {
      return sqrt(2 * this->L_);
    } else if(pos % 2 == 0) {
      return 2 * sqrt(this->L_) / (M_PI * (pos / 2)) * sin(M_PI * (pos / 2));
    } else {
      return 0.0;
    }
  }
}

double BasisFunc::int1(int pos) const {
  if(this->kernel_) {
    if(pos == 1) {
      return (pow(this->dom_.second, 2) - pow(this->dom_.first, 2)) / 2;
    } else {
      double a = this->dom_.first;
      double b = this->dom_.second;
      double c = this->centers_[pos - 2];
      double dx_sq = pow(this->dx_, 2);
      double sqrt2pi = sqrt(2 * M_PI);
      double term1 = exp(-pow(b - 2 * a + c, 2) / (2 * dx_sq)) -
                    exp(-pow(a - 2 * b + c, 2) / (2 * dx_sq));
      double term2 = (b - a + c) * sqrt2pi * erf((a - c) / (sqrt(2) * this->dx_));
      double term3 = (a - b - c) * sqrt2pi * erf((2 * a - b - c) / (sqrt(2) * this->dx_));
      double term4 = sqrt2pi * (c * erf((-a + c) / (sqrt(2) * this->dx_)) -
                      (a - b + c) * erf((a - 2 * b + c) / (sqrt(2) * this->dx_)) +
                      (a - b) * erf((-b + c) / (sqrt(2) * this->dx_)));
      return this->dx_ / 2 * (2 * this->dx_ * term1 + term2 + term3 + term4);
    }
  } else {
    if(pos == 1) {
      return this->shift_ * sqrt(2 * this->L_);
    } else if(pos % 2 == 0) {
      return 2 * this->shift_ * sqrt(this->L_) / (M_PI * (pos / 2)) * sin(M_PI * (pos / 2));
    } else {
      return 2 * pow(this->L_, 1.5) / pow(M_PI * (pos / 2), 2) * (sin(M_PI * (pos / 2)) - M_PI * (pos / 2) * cos(M_PI * (pos / 2)));
    }
  }
}

double BasisFunc::int2(int pos) const {
  if(this->kernel_) {
    if(pos == 1) {
      return (pow(this->dom_.second, 3) - pow(this->dom_.first, 3)) / 3;
    } else {
      double a = this->dom_.first;
      double b = this->dom_.second;
      double c = this->centers_[pos - 2];
      double dx_sq = pow(this->dx_, 2);
      double sqrt2pi = sqrt(2 * M_PI);
      double sqrt2_dx = sqrt(2) * this->dx_;
      double exp1 = exp(-pow(a - c, 2) / (2 * dx_sq));
      double exp2 = exp(-pow(b - c, 2) / (2 * dx_sq));
      double exp3 = exp(-pow(a - 2 * b + c, 2) / (2 * dx_sq));
      double exp4 = exp(-pow(-2 * a + b + c, 2) / (2 * dx_sq));
      double term1 = this->dx_ / 2 * (2 * this->dx_ * (
          a * (2 * exp1 + 2 * exp2 - exp3) +
          b * (-2 * exp1 - 2 * exp2 + exp4) +
          c * (-exp3 + exp4)
      ));
      double diff1_sq = pow(b - a + c, 2) + dx_sq;
      double diff2_sq = pow(a - b + c, 2) + dx_sq;
      double c_sq = pow(c, 2) + dx_sq;
      double erf1 = erf((a - c) / sqrt2_dx);
      double erf2 = erf((2 * a - b - c) / sqrt2_dx);
      double erf3 = erf((-a + c) / sqrt2_dx);
      double erf4 = erf((a - 2 * b + c) / sqrt2_dx);
      double erf5 = erf((-b + c) / sqrt2_dx);
      double term2 = diff1_sq * sqrt2pi * erf1 - diff1_sq * sqrt2pi * erf2;
      double term3 = sqrt2pi * (
          c_sq * erf3 - diff2_sq * erf4 +
          (a - b) * (a - b + 2 * c) * erf5
      );
      return term1 + term2 + term3;
    }
  } else {
    if(pos == 1) {
      return sqrt(2 * this->L_) / 3 * (3 * pow(this->shift_, 2) + pow(this->L_, 2));
    } else if(pos % 2 == 0) {
      return 2 * sqrt(this->L_) / pow(M_PI * (pos / 2), 3) *
            (2 * pow(this->L_, 2) * M_PI * (pos / 2) * cos(M_PI * (pos / 2)) +
            (pow(M_PI * (pos / 2), 2) * (pow(this->shift_, 2) +
            pow(this->L_, 2)) - 2 * pow(this->L_, 2)) * sin(M_PI * (pos / 2)));
    } else {
      return 4 * this->shift_ * pow(this->L_, 1.5) / pow(M_PI * (pos / 2), 2) * (sin(M_PI * (pos / 2)) - M_PI * (pos / 2) * cos(M_PI * (pos / 2)));
    }
  }
}

void BasisFunc::test() const {
  int ntest = 10;
  vector<double> xtest(ntest);
  for(int i = 0; i < ntest; ++i) {
    xtest[i] = this->dom_.first + i * (this->dom_.second - this->dom_.first) / (ntest - 1);
  }
  cout << "Testing gaussian()" << endl << endl;
  cout << this->L_ << " " << this->dx_ << endl;
  for(int i = 1; i < this->nbasis_; ++i) {
    cout << this->centers_[i - 1] << " ";
  }
  cout << endl;
  for(int j = 0; j < ntest; ++j) {
    cout << xtest[j] << " ";
  }
  cout << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    for(int j = 0; j < ntest; ++j) {
      cout << (*this)(xtest[j], i, false) << " ";
    }
    cout << endl;
  }
  cout << endl << "Testing gaussiand()" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    for(int j = 0; j < ntest; ++j) {
      cout << grad(xtest[j], i, false) << " ";
    }
    cout << endl;
  }
  cout << endl << "Testing gaussian() with conv" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    for(int j = 0; j < ntest; ++j) {
      cout << (*this)(xtest[j], i, true) << " ";
    }
    cout << endl;
  }
  cout << endl << "Testing gaussiand() with conv" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    for(int j = 0; j < ntest; ++j) {
      cout << grad(xtest[j], i, true) << " ";
    }
    cout << endl;
  }
  cout << endl << "Testing int0" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    cout << int0(i) << endl;
  }
  cout << endl << "Testing int1" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    cout << int1(i) << endl;
  }
  cout << endl << "Testing int2" << endl << endl;
  for(int i = 1; i <= this->nbasis_; ++i) {
    cout << int2(i) << endl;
  }
  cout << endl << "Testing gram" << endl << endl;
  for(int i = 0; i < this->nbasis_; ++i) {
    for(int j = 0; j < this->nbasis_; ++j) {
      cout << this->gram_(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl << "Testing ginv" << endl << endl;
  for(int i = 0; i < this->nbasis_; ++i) {
    for(int j = 0; j < this->nbasis_; ++j) {
      cout << this->ginv_(i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;
}

}
}
