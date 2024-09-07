#include "TTCross.h"
#include "tools/Matrix.h"
#include "tools/OpenMP.h"
#include <algorithm>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include <iostream>

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

struct ACAParams {
  const TTCross* instance;
  int ii;
  int ss;
  int ll;
  int lr;
};

double aca_f(double x, void* params) {
  ACAParams* aca_params = (ACAParams*)params;
  int ii = aca_params->ii;
  int ss = aca_params->ss;
  int ll = aca_params->ll;
  int lr = aca_params->lr;
  vector<double> elements = { x };
  if(ii != 1) {
    auto& left = aca_params->instance->I()[ii - 1][ll - 1];
    elements.insert(elements.begin(), left.begin(), left.end());
  }
  if(ii != aca_params->instance->d()) {
    auto& right = aca_params->instance->J()[ii][lr - 1];
    elements.insert(elements.end(), right.begin(), right.end());
  }
  return aca_params->instance->f(elements) * aca_params->instance->basisi(ii - 1)(x, ss, false);
}

TTCross::TTCross()
  : G_(nullptr), n_(0), kbt_(0.0), cutoff_(0.0), maxrank_(0), d_(0), pos_(0),
    vshift_(0.0), log_(nullptr), aca_n_(0), aca_epsabs_(0.0), aca_epsrel_(0.0),
    aca_limit_(0), aca_key_(0), conv_(true) { }

TTCross::TTCross(const vector<BasisFunc>& basis, double kbt, double cutoff,
                 int maxrank, Log& log, int aca_n, double aca_epsabs,
                 double aca_epsrel, int aca_limit, int aca_key, bool conv)
  : G_(nullptr), basis_(basis), n_(basis[0].nbasis()), kbt_(kbt), cutoff_(cutoff),
    maxrank_(maxrank), d_(basis.size()), pos_(0), vshift_(0.0),
    I_(vector<vector<vector<double>>>(basis.size())),
    J_(vector<vector<vector<double>>>(basis.size())),
    log_(&log), aca_n_(aca_n), aca_epsabs_(aca_epsabs), aca_epsrel_(aca_epsrel),
    aca_limit_(aca_limit), aca_key_(aca_key), conv_(conv)
{
  I_[0].push_back(vector<double>());
  J_[0].push_back(vector<double>());
}

//TODO: figure out if convolution should be used for vb
double TTCross::f(const vector<double>& x) const {
  double result = 0.0;
  if(this->vb_.length() == 0) {
    result = this->kbt_ * log(max(ttEval(*this->G_, this->basis_, x, this->conv_), 0.1));
  } else {
    result = max(ttEval(this->vb_, this->basis_, x, this->conv_) +
                 this->kbt_ * log(max(ttEval(*this->G_, this->basis_, x, this->conv_), 1.0)) - this->vshift_, -2 * this->kbt_);
  }
  return result;
}

void TTCross::updateIJ(const vector<double>& ij) {
  this->I_[this->pos_].push_back(vector<double>(ij.begin(), ij.begin() + this->pos_));
  this->J_[this->pos_].push_back(vector<double>(ij.begin() + this->pos_, ij.end()));
}

pair<double, int> TTCross::diagACA(const vector<vector<double>>& samples, const vector<double>& Rk) {
  int k = this->I_[this->pos_].size() + 1;
  int ik = 0;
  double dk = 0.0;
  for(unsigned i = 0; i < samples.size(); ++i) {
    if(abs(Rk[i]) > dk) {
      ik = i;
      dk = Rk[i];
    }
  }
  vector<double> Ri(samples.size()), Rj(samples.size());
  for(unsigned i = 0; i < samples.size(); ++i) {
    vector<double> arg(samples[i].begin(), samples[i].begin() + this->pos_);
    arg.insert(arg.end(), samples[ik].begin() + this->pos_, samples[ik].end());
    Ri[i] = f(arg);
    arg = vector<double>(samples[ik].begin(), samples[ik].begin() + this->pos_);
    arg.insert(arg.end(), samples[i].begin() + this->pos_, samples[i].end());
    Rj[i] = f(arg);
  }
  for(int l = 0; l < k - 1; ++l) {
    auto ul = this->u_[l];
    transform(ul.begin(), ul.end(), ul.begin(), bind(multiplies<double>(), placeholders::_1, this->v_[l][ik]));
    transform(Ri.begin(), Ri.end(), ul.begin(), Ri.begin(), minus<double>());
    auto vl = this->v_[l];
    transform(vl.begin(), vl.end(), vl.begin(), bind(multiplies<double>(), placeholders::_1, this->u_[l][ik]));
    transform(Rj.begin(), Rj.end(), vl.begin(), Rj.begin(), minus<double>());
  }
  this->u_.push_back(Ri);
  transform(Rj.begin(), Rj.end(), Rj.begin(), bind(multiplies<double>(), placeholders::_1, 1 / dk));
  this->v_.push_back(Rj);
  return make_pair(abs(dk), ik);
}

void TTCross::continuousACA(const vector<vector<double>>& samples) {
  int order = this->d_;

  this->pos_ = 0;
  for(int i = 0; i < order - 1; ++i) {
    *this->log_ << "pos = " << i + 1 << "\n";
    this->log_->flush();
    ++this->pos_;

    vector<double> Rk(samples.size());
    for(unsigned i = 0; i < samples.size(); ++i) {
      Rk[i] = f(samples[i]);
    }
    this->u_.clear();
    this->v_.clear();
    for(int r = 1; r <= this->maxrank_; ++r) {
      auto [res_new, ik] = diagACA(samples, Rk);
      auto& xy = samples[ik];
      if(this->I_[i + 1].empty()) {
        this->resfirst_.push_back(res_new);
      } else if(res_new > this->resfirst_[i]) {
        this->resfirst_[i] = res_new;
      } else if(res_new / this->resfirst_[i] < this->cutoff_) {
        break;
      }

      updateIJ(xy);
      vector<double> uv(samples.size());
      transform(this->u_[r - 1].begin(), this->u_[r - 1].end(), this->v_[r - 1].begin(), uv.begin(), multiplies<double>());
      transform(Rk.begin(), Rk.end(), uv.begin(), Rk.begin(), minus<double>());
      *this->log_ << "rank = " << r << " res = " << res_new << " xy = ( ";
      for(double elt : xy) {
        *this->log_ << elt << " ";
      }
      *this->log_ << ")\n";
      this->log_->flush();
    }
  }
}

void TTCross::updateVb(const vector<vector<double>>& samples) {
  reset();
  *this->log_ << "\nStarting TT-cross ACA...\n";
  continuousACA(samples);

  auto sites = SiteSet(this->d_, this->n_);
  vector<Index> l(this->d_ - 1);
  MPS psi(this->d_);
  vector<int> ranks(this->d_ - 1);
  for(int i = 1; i < this->d_; ++i) {
    ranks[i - 1] = this->I_[i].size();
  }
  *this->log_ << "Computing Galerkin projection...\n";
  this->log_->flush();
  // unsigned nt = OpenMP::getNumThreads();
  gsl_set_error_handler_off();
  gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(this->aca_n_);
  double result, error;
  for(int ii = 1; ii <= this->d_; ++ii) {
    auto& dom = basisi(ii - 1).dom();
    auto s = sites(ii);
    if(ii != this->d_) {
      l[ii - 1] = Index(ranks[ii - 1], "Link,l=" + to_string(ii));
    }

    if(ii == 1) {
      psi.ref(1) = ITensor(s, prime(l[0]));
      // #pragma omp parallel for num_threads(nt)
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int lr = 1; lr <= dim(l[0]); ++lr) {
          cout << ii << " " << ss << " " << lr << endl;
          // cout << this->aca_epsabs_ << " " << this->aca_epsrel_ << " " << this->aca_limit_ << " " << this->aca_key_ << endl;
          ACAParams aca_params = { this, 1, ss, 0, lr };

          // int nbins = 1000;
          // for(int i = 0; i < nbins; ++i) {
          //   cout << aca_f(dom.first + i * (dom.second - dom.first) / nbins, &aca_params) << ", ";
          // }
          // cout << endl;
          
          gsl_function F;
          F.function = &aca_f;
          F.params = &aca_params;
          gsl_integration_qag(&F, dom.first, dom.second, this->aca_epsabs_,
                              this->aca_epsrel_, this->aca_limit_,
                              this->aca_key_, workspace, &result, &error);
          // cout << result << " " << error << endl;
          // PrintData(psi(1));
          // PrintData(s);
          // cout << ss << endl;
          // PrintData(prime(l[0]));
          // cout << lr << endl;
          // psi.ref(1).set(s = ss, prime(l[0]) = lr, result);
          // cout << "wtf" << endl;
        }
      }
    } else if(ii == this->d_) {
      psi.ref(this->d_) = ITensor(s, l[this->d_ - 2]);
      // #pragma omp parallel for num_threads(nt)
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int ll = 1; ll <= dim(l[this->d_ - 2]); ++ll) {
          cout << ii << " " << ss << " " << ll << endl;
          ACAParams aca_params = { this, this->d_, ss, ll, 0 };
          gsl_function F;
          F.function = &aca_f;
          F.params = &aca_params;
          gsl_integration_qag(&F, dom.first, dom.second, this->aca_epsabs_,
                              this->aca_epsrel_, this->aca_limit_,
                              this->aca_key_, workspace, &result, &error);
          psi.ref(this->d_).set(s = ss, l[this->d_ - 2] = ll, result);
        }
      }
    } else {
      psi.ref(ii) = ITensor(s, l[ii - 2], prime(l[ii - 1]));
      // #pragma omp parallel for num_threads(nt)
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int ll = 1; ll <= dim(l[ii - 2]); ++ll) {
          for(int lr = 1; lr <= dim(l[ii - 1]); ++lr) {
            cout << ii << " " << ss << " " << ll  << " " << lr << endl;
            ACAParams aca_params = { this, ii, ss, ll, lr };
            gsl_function F;
            F.function = &aca_f;
            F.params = &aca_params;
            gsl_integration_qag(&F, dom.first, dom.second, this->aca_epsabs_,
                                this->aca_epsrel_, this->aca_limit_,
                                this->aca_key_, workspace, &result, &error);
            psi.ref(ii).set(s = ss, l[ii - 2] = ll, prime(l[ii - 1]) = lr, result);
          }
        }
      }
    }

    if(ii != this->d_) {
      Matrix<double> Ahat(ranks[ii - 1], ranks[ii - 1]);
      for(int jj = 0; jj < ranks[ii - 1]; ++jj) {
        for(int kk = 0; kk < ranks[ii - 1]; ++kk) {
          vector<double> arg = this->I_[ii][jj];
          arg.insert(arg.end(), this->J_[ii][kk].begin(), this->J_[ii][kk].end());
          Ahat(jj, kk) = f(arg);
        }
      }
      
      Matrix<double> AinvMat;
      Invert(Ahat, AinvMat);
      ITensor Ainv(prime(l[ii - 1]), l[ii - 1]);
      for(int jj = 0; jj < ranks[ii - 1]; ++jj) {
        for(int kk = 0; kk < ranks[ii - 1]; ++kk) {
          Ainv.set(prime(l[ii - 1]) = jj - 1, l[ii - 1] = kk - 1, AinvMat(jj, kk));
        }
      }
      psi.ref(ii) *= Ainv;
    }
    *this->log_ << "Core " << ii << " done!\n";
    this->log_->flush();
  }
  gsl_integration_workspace_free(workspace);

  this->vb_ = psi;
}

double TTCross::vtop(const vector<vector<double>>& samples) const {
  double max = 0.0;
  for(auto& s : samples) {
    if(f(s) > max) {
      max = f(s);
    }
  }
  return max;
}

void TTCross::reset() {
  for(auto& pivots : this->I_) {
    pivots.clear();
  }
  I_[0].push_back(vector<double>());
  for(auto& pivots : this->J_) {
    pivots.clear();
  }
  J_[0].push_back(vector<double>());
  this->resfirst_.clear();
}

double ttEval(const MPS& tt, const vector<BasisFunc>& basis, const vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  vector<ITensor> basis_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis[i - 1](elements[i - 1], j, conv));
    }
  }
  auto result = tt(1) * basis_evals[0];
  for(int i = 2; i <= d; ++i) {
    result *= tt(i) * basis_evals[i - 1];
  }
  return elt(result);
}

vector<double> ttGrad(const MPS& tt, const vector<BasisFunc>& basis, const vector<double>& elements, bool conv) {
  int d = length(tt);
  auto s = siteInds(tt);
  vector<double> grad(d, 0.0);
  vector<ITensor> basis_evals(d), basisd_evals(d);
  for(int i = 1; i <= d; ++i) {
    basis_evals[i - 1] = basisd_evals[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis[i - 1](elements[i - 1], j, conv));
      basisd_evals[i - 1].set(s(i) = j, basis[i - 1].grad(elements[i - 1], j, conv));
    }
  }
  for(int k = 1; k <= d; ++k) {
    auto result = tt(1) * (k == 1 ? basisd_evals[0] : basis_evals[0]);
    for(int i = 2; i <= d; ++i) {
      result *= tt(i) * (k == i ? basisd_evals[i - 1] : basis_evals[i - 1]);
    }
    grad[k - 1] = elt(result);
  }
  return grad;
}

}
}
