#include "BasisFunc.h"
#include <gsl/gsl_integration.h>

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
  : dom_(make_pair(0.0, 0.0)), nbasis_(0), nbins_(0), L_(0.0), shift_(0.0), w_(0.0), gaussian_(false), dx_(0.0) {}

BasisFunc::BasisFunc(pair<double, double> dom, int nbasis, bool conv,
                     int nbins, double w, int conv_n, double conv_epsabs,
                     double conv_epsrel, int conv_limit, int conv_key,
                     bool gaussian)
  : dom_(dom), nbasis_(nbasis), nbins_(conv ? nbins : 0),
    L_((dom.second - dom.first) / 2), shift_((dom.second + dom.first) / 2),
    grid_(nbasis, vector<double>(nbins, 0.0)),
    gridd_(nbasis, vector<double>(nbins, 0.0)), xdata_(nbins, 0.0), w_(w),
    gaussian_(gaussian)
{
  if(gaussian) {
    this->dx_ = (dom.second - dom.first) / (nbasis - 2);
    this->centers_ = vector<double>(nbasis - 1);
    for(int i = 0; i < nbasis - 1; ++i) {
      this->centers_[i] = dom.first + i * this->dx_;
    }
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
      result += (this->centers_[pos - 2] - x + 2 * k * this->L_, 2) /
                pow(this->dx_, 2) * exp(-pow(x - this->centers_[pos - 2] +
                2 * k * this->L_, 2) / (2 * pow(this->dx_, 2)));
    }
    return result;
  }
}

double BasisFunc::operator()(double x, int pos, bool conv) const {
  if(this->gaussian_) {
    if(conv && this->nbins_ > 0) {
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
  if(this->gaussian_) {
    if(conv && this->nbins_ > 0) {
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
  if(this->gaussian_) {
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
  if(this->gaussian_) {
    if(pos == 1) {
      return (pow(this->dom_.second, 2) - pow(this->dom_.first, 2)) / 2;
    } else {
      double sqrt2pi = sqrt(2 * M_PI);
      double dx_sq = pow(this->dx_, 2);
      double x1 = this->dom_.first;
      double x2 = this->dom_.second;
      double c = this->centers_[pos - 2];
      double term1 = exp(-pow(x2 - 2 * x1 + c, 2) / (2 * dx_sq)) -
                    exp(-pow(x1 - 2 * x2 + c, 2) / (2 * dx_sq));
      double term2 = (x2 - x1 + c) * sqrt2pi * erf((x1 - c) / (sqrt(2) * this->dx_));
      double term3 = (x1 - x2 - c) * sqrt2pi * erf((2 * x1 - x2 - c) / (sqrt(2) * this->dx_));
      double term4 = sqrt2pi * (c * erf((-x1 + c) / (sqrt(2) * this->dx_)) -
                      (x1 - x2 + c) * erf((x1 - 2 * x2 + c) / (sqrt(2) * this->dx_)) +
                      (x1 - x2) * erf((-x2 + c) / (sqrt(2) * this->dx_)));
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
  if(this->gaussian_) {
    if(pos == 1) {
      return (pow(this->dom_.second, 3) - pow(this->dom_.first, 3)) / 3;
    } else {
      double a = dom.first;
      double b = dom.second;
      double centersi = centers[pos - 2];
      double dx_sq = pow(this->dx_, 2);
      double sqrt2pi = sqrt(2 * M_PI);
      double sqrt2_dx = sqrt(2) * this->dx_;
      double exp1 = exp(-pow(a - centersi, 2) / (2 * dx_sq));
      double exp2 = exp(-pow(b - centersi, 2) / (2 * dx_sq));
      double exp3 = exp(-pow(a - 2 * b + centersi, 2) / (2 * dx_sq));
      double exp4 = exp(-pow(-2 * a + b + centersi, 2) / (2 * dx_sq));
      double term1 = this->dx_ / 2 * (2 * this->dx_ * (
          a * (2 * exp1 + 2 * exp2 - exp3) +
          b * (-2 * exp1 - 2 * exp2 + exp4) +
          centersi * (-exp3 + exp4)
      ));
      double diff1_sq = pow(b - a + centersi, 2) + dx_sq;
      double diff2_sq = pow(a - b + centersi, 2) + dx_sq;
      double centersi_sq = pow(centersi, 2) + dx_sq;
      double erf1 = erf((a - centersi) / sqrt2_dx);
      double erf2 = erf((2 * a - b - centersi) / sqrt2_dx);
      double erf3 = erf((-a + centersi) / sqrt2_dx);
      double erf4 = erf((a - 2 * b + centersi) / sqrt2_dx);
      double erf5 = erf((-b + centersi) / sqrt2_dx);
      double term2 = diff1_sq * sqrt2pi * erf1 - diff1_sq * sqrt2pi * erf2;
      double term3 = sqrt2pi * (
          centersi_sq * erf3 - diff2_sq * erf4 +
          (a - b) * (a - b + 2 * centersi) * erf5
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

}
}
