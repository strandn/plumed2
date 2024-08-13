#include "BasisFunc.h"
#include <gsl/gsl_integration.h>

using namespace std;

namespace PLMD {
namespace ttsketch {

struct GSLParams {
  BasisFunc* instance;
  int j;
  int k;
};

double f(double x, void* params) {
  GSLParams* gsl_params = (GSLParams*)params;
  auto dom = gsl_params->instance->dom();
  int nbins = gsl_params->instance->nbins();
  int j = gsl_params->j;
  int k = gsl_params->k;
  double fourier = gsl_params->instance->fourier(x, j);
  double w = gsl_params->instance->w();
  double sigma = w * (dom.second - dom.first);
  double s = dom.first + (k - 1) * (dom.second - dom.first) / (nbins - 1);
  return fourier * (1 / (sqrt(2 * M_PI) * sigma)) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
}

double df(double x, void* params) {
  GSLParams* gsl_params = (GSLParams*)params;
  auto dom = gsl_params->instance->dom();
  int nbins = gsl_params->instance->nbins();
  int j = gsl_params->j;
  int k = gsl_params->k;
  double fourier = gsl_params->instance->fourier(x, j);
  double w = gsl_params->instance->w();
  double sigma = w * (dom.second - dom.first);
  double s = dom.first + (k - 1) * (dom.second - dom.first) / (nbins - 1);
  return fourier * ((x - s) / (sqrt(2 * M_PI) * pow(sigma, 3))) * exp(-pow(s - x, 2) / (2 * pow(sigma, 2)));
}

BasisFunc::BasisFunc()
  : dom_(make_pair(0.0, 0.0)), nbasis_(0), conv_(false), nbins_(0), L_(0.0),
    shift_(0.0), w_(0.0), isPeriodic_(false) {}

BasisFunc::BasisFunc(pair<double, double> dom, int nbasis, bool conv,
                     int nbins, double w, int gsl_n, double gsl_epsabs,
                     double gsl_epsrel, int gsl_limit, int gsl_key,
                     bool isPeriodic)
  : dom_(dom), nbasis_(nbasis), conv_(true), nbins_(conv ? nbins : 0),
    L_((dom.second - dom.first) / 2), shift_((dom.second + dom.first) / 2),
    grid_(nbasis, vector<double>(nbins, 0.0)),
    gridd_(nbasis, vector<double>(nbins, 0.0)), xdata_(nbins, 0.0), w_(w),
    isPeriodic_(isPeriodic)
{
  if(nbins > 0) {
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(gsl_n);
    double result, error;
    for(int j = 0; j < nbasis; ++j) {
      for(int k = 0; k < nbins; ++k) {
        GSLParams gsl_params = { this, j + 1, k + 1 };
        gsl_function F;
        F.function = &f;
        F.params = &gsl_params;
        if(isPeriodic) {
          gsl_integration_qag(&F, dom.first - L_ / 4, dom.second + L_ / 4, gsl_epsabs, gsl_epsrel, gsl_limit, gsl_key, workspace, &result, &error);
        } else {
          //TODO: fix this
          gsl_integration_qag(&F, dom.first, dom.second, gsl_epsabs, gsl_epsrel, gsl_limit, gsl_key, workspace, &result, &error);
        }
        this->grid_[j][k] = result;
        gsl_function DF;
        DF.function = &df;
        DF.params = &gsl_params;
        if(isPeriodic) {
          gsl_integration_qag(&DF, dom.first - L_ / 4, dom.second + L_ / 4, gsl_epsabs, gsl_epsrel, gsl_limit, gsl_key, workspace, &result, &error);
        } else {
          //TODO: fix this
          gsl_integration_qag(&DF, dom.first, dom.second, gsl_epsabs, gsl_epsrel, gsl_limit, gsl_key, workspace, &result, &error);
        }
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
  if(!isPeriodic_ && (x < this->dom_.first || x > this->dom_.second)) {
    return 0.0;
  }
  if(pos == 1) {
    return 1 / sqrt(2 * this->L_);
  } else if(pos % 2 == 0) {
    return sqrt(1 / this->L_) * cos(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  } else {
    return sqrt(1 / this->L_) * sin(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
  }
}

double BasisFunc::operator()(double x, int pos) const {
  if(this->conv_ && this->nbins_ > 0) {
    return interpolate(x, pos, false);
  } else {
    return fourier(x, pos);
  }
}

double BasisFunc::grad(double x, int pos) const {
  if(this->conv_ && this->nbins_ > 0) {
    return interpolate(x, pos, true);
  } else {
    if(!isPeriodic_ && (x < this->dom_.first || x > this->dom_.second)) {
      return 0.0;
    }
    if(pos == 1) {
      return 0.0;
    } else if(pos % 2 == 0) {
      return -pow(1 / this->L_, 3 / 2) * M_PI * (pos / 2) * sin(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
    } else {
      return pow(1 / this->L_, 3 / 2) * M_PI * (pos / 2) * cos(M_PI * (x - this->shift_) * (pos / 2) / this->L_);
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

}
}
