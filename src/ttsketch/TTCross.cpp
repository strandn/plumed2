#include "TTCross.h"
#include "TTHelper.h"
#include "tools/Matrix.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

TTCross::TTCross()
  : G_(nullptr), n_(0), kbt_(0.0), cutoff_(0.0), maxrank_(0), d_(0), pos_(0),
    vshift_(0.0), log_(nullptr), conv_(true), convg_(true),
    walkers_mpi_(false), auto_rank_(false) { }

TTCross::TTCross(const vector<BasisFunc>& basis, double kbt, double cutoff,
                 int maxrank, Log& log, bool conv, bool convg, int nbins,
                 bool walkers_mpi, bool auto_rank)
  : G_(nullptr), basis_(basis), n_(basis[0].nbasis()), kbt_(kbt),
    cutoff_(cutoff), maxrank_(maxrank), d_(basis.size()), pos_(0), vshift_(0.0),
    I_(vector<vector<vector<double>>>(basis.size())),
    J_(vector<vector<vector<double>>>(basis.size())), log_(&log), conv_(conv),
    convg_(convg), nbins_(nbins), grid_(basis.size(), vector<double>(nbins)),
    walkers_mpi_(walkers_mpi), auto_rank_(auto_rank)
{
  this->I_[0].push_back(vector<double>());
  this->J_[0].push_back(vector<double>());
  for(unsigned i = 0; i < basis.size(); ++i) {
    auto [min, max] = basis[i].dom();
    for(int j = 0; j < nbins; ++j) {
      this->grid_[i][j] = min + (max - min) * j / nbins;
    }
  }
}

double TTCross::f(const vector<double>& x) const {
  double result = 0.0;
  if(this->vb_.length() == 0) {
    result = this->kbt_ * log(max(ttEval(*this->G_, this->basis_, x, this->convg_), 1.0));
  } else {
    result = max(max(ttEval(this->vb_, this->basis_, x, this->conv_), 0.0) +
                 this->kbt_ * log(max(ttEval(*this->G_, this->basis_, x,
                 this->convg_), 1.0)) - this->vshift_, 0.0);
  }
  return result;
}

void TTCross::updateIJ(const vector<double>& ij) {
  this->I_[this->pos_].push_back(vector<double>(ij.begin(), ij.begin() + this->pos_));
  this->J_[this->pos_].push_back(vector<double>(ij.begin() + this->pos_, ij.end()));
}

pair<double, int> TTCross::diagACA(const vector<double>& Rk, vector<vector<double>>& u, vector<vector<double>>& v) const {
  int k = this->I_[this->pos_].size() + 1;
  int ik = 0;
  double dk = 0.0;
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    if(abs(Rk[i]) > dk) {
      ik = i;
      dk = Rk[i];
    }
  }
  vector<double> Ri(this->samples_.size()), Rj(this->samples_.size());
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    vector<double> arg(this->samples_[i].begin(), this->samples_[i].begin() + this->pos_);
    arg.insert(arg.end(), this->samples_[ik].begin() + this->pos_, this->samples_[ik].end());
    Ri[i] = f(arg);
    arg = vector<double>(this->samples_[ik].begin(), this->samples_[ik].begin() + this->pos_);
    arg.insert(arg.end(), this->samples_[i].begin() + this->pos_, this->samples_[i].end());
    Rj[i] = f(arg);
  }
  for(int l = 0; l < k - 1; ++l) {
    auto ul = u[l];
    transform(ul.begin(), ul.end(), ul.begin(), bind(multiplies<double>(), placeholders::_1, v[l][ik]));
    transform(Ri.begin(), Ri.end(), ul.begin(), Ri.begin(), minus<double>());
    auto vl = v[l];
    transform(vl.begin(), vl.end(), vl.begin(), bind(multiplies<double>(), placeholders::_1, u[l][ik]));
    transform(Rj.begin(), Rj.end(), vl.begin(), Rj.begin(), minus<double>());
  }
  u.push_back(Ri);
  transform(Rj.begin(), Rj.end(), Rj.begin(), bind(multiplies<double>(), placeholders::_1, 1 / dk));
  v.push_back(Rj);
  return make_pair(abs(dk), ik);
}

void TTCross::continuousACA() {
  int order = this->d_;
  vector<double> A(this->samples_.size());
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    A[i] = f(this->samples_[i]);
  }
  if(this->auto_rank_) {
    double error = numeric_limits<double>::max();
    vector<vector<vector<double>>> ulist(order - 1), vlist(order - 1);
    vector<vector<double>> Rklist(order - 1, vector<double>(A));
    int r = 1;
    while(true) {
      this->pos_ = 0;
      for(int i = 0; i < order - 1; ++i) {
        *this->log_ << "pos = " << i + 1 << "\n";
        this->log_->flush();
        ++this->pos_;
        auto [res_new, ik] = diagACA(Rklist[i], ulist[i], vlist[i]);
        auto& xy = this->samples_[ik];
        updateIJ(xy);
        vector<double> uv(this->samples_.size());
        transform(ulist[i][r - 1].begin(), ulist[i][r - 1].end(), vlist[i][r - 1].begin(), uv.begin(), multiplies<double>());
        transform(Rklist[i].begin(), Rklist[i].end(), uv.begin(), Rklist[i].begin(), minus<double>());
        double norm_ratio = sqrt(norm(Rklist[i]) / norm(A));
        *this->log_ << "rank = " << r << " res = " << res_new << " |Rk|/|A| = " << norm_ratio << " xy = ( ";
        for(double elt : xy) {
          *this->log_ << elt << " ";
        }
        *this->log_ << ")\n";
        this->log_->flush();
      }
      vector<double> diff(this->samples_.size());
      approximate(diff);
      transform(diff.begin(), diff.end(), A.begin(), diff.begin(), minus<double>());
      double error_new = sqrt(norm(diff) / norm(A));
      *this->log_ << "Relative l2 error = " << error_new << "\n";
      this->log_->flush();
      if(error_new > error) {
        for(auto& pivots : this->I_) {
          pivots.pop_back();
        }
        for(auto& pivots : this->J_) {
          pivots.pop_back();
        }
        break;
      }
      error = error_new;
      ++r;
    }
  } else {
    this->pos_ = 0;
    for(int i = 0; i < order - 1; ++i) {
      *this->log_ << "pos = " << i + 1 << "\n";
      this->log_->flush();
      ++this->pos_;
      auto Rk = A;
      vector<vector<double>> u, v;
      for(int r = 1; r <= this->maxrank_; ++r) {
        auto [res_new, ik] = diagACA(Rk, u, v);
        auto& xy = this->samples_[ik];
        if(this->I_[i + 1].empty()) {
          this->resfirst_.push_back(res_new);
        } else if(res_new > this->resfirst_[i]) {
          this->resfirst_[i] = res_new;
        } else if(res_new / this->resfirst_[i] < this->cutoff_ || res_new == 0.0) {
          break;
        }
        updateIJ(xy);
        vector<double> uv(this->samples_.size());
        transform(u[r - 1].begin(), u[r - 1].end(), v[r - 1].begin(), uv.begin(), multiplies<double>());
        transform(Rk.begin(), Rk.end(), uv.begin(), Rk.begin(), minus<double>());
        double norm_ratio = sqrt(norm(Rk) / norm(A));
        *this->log_ << "rank = " << r << " res = " << res_new << " |Rk|/|A| = " << norm_ratio << " xy = ( ";
        for(double elt : xy) {
          *this->log_ << elt << " ";
        }
        *this->log_ << ")\n";
        this->log_->flush();
      }
    }
  }
}

void TTCross::approximate(vector<double>& approx) {
  vector<Index> l(this->d_ - 1);
  vector<ITensor> evals(this->d_);
  vector<int> ranks(this->d_ - 1);
  vector<ITensor> Ainv(this->d_ - 1);
  for(int i = 1; i < this->d_; ++i) {
    ranks[i - 1] = this->I_[i].size();
  }
  for(int ii = 1; ii < this->d_; ++ii) {
    l[ii - 1] = Index(ranks[ii - 1], "Link,l=" + to_string(ii));
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
    Ainv[ii] = ITensor(prime(l[ii - 1]), l[ii - 1]);
    for(int jj = 0; jj < ranks[ii - 1]; ++jj) {
      for(int kk = 0; kk < ranks[ii - 1]; ++kk) {
        Ainv[ii].set(prime(l[ii - 1]) = jj + 1, l[ii - 1] = kk + 1, AinvMat(jj, kk));
      }
    }
  }
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    for(unsigned ii = 1; ii <= this->d_; ++ii) {
      if(ii == 1) {
        evals[0] = ITensor(prime(l[0]));
        for(int lr = 1; lr <= dim(l[0]); ++lr) {
          vector<double> elements = { this->samples_[i][0] };
          auto& right = this->J_[1][lr - 1];
          elements.insert(elements.end(), right.begin(), right.end());
          evals[0].set(prime(l[0]) = lr, f(elements));
        }
      } else if(ii == this->d_) {
        evals[this->d_ - 1] = ITensor(l[this->d_ - 2]);
        for(int ll = 1; ll <= dim(l[this->d_ - 2]); ++ll) {
          vector<double> elements = { this->samples_[i][this->d_  - 1] };
          auto& left = this->I_[this->d_ - 1][ll - 1];
          elements.insert(elements.begin(), left.begin(), left.end());
          evals[this->d_ - 1].set(l[this->d_ - 2] = ll, f(elements));
        }
      } else {
        evals[ii - 1] = ITensor(l[ii - 2], prime(l[ii - 1]));
        for(int ll = 1; ll <= dim(l[ii - 2]); ++ll) {
          for(int lr = 1; lr <= dim(l[ii - 1]); ++lr) {
            vector<double> elements = { this->samples_[i][ii - 1] };
            auto& left = this->I_[ii - 1][ll - 1];
            elements.insert(elements.begin(), left.begin(), left.end());
            auto& right = this->J_[ii][lr - 1];
            elements.insert(elements.end(), right.begin(), right.end());
            evals[ii - 1].set(l[ii - 2] = ll, prime(l[ii - 1]) = lr, f(elements));
          }
        }
      }
    }
    ITensor result = evals[0];
    for(int j = 1; j < d; ++j) {
      result *= Ainv[j - 1] * evals[j];
    }
    approx[i] = elt(result);
  }
}

void TTCross::updateVb() {
  reset();
  vector<double> A0(this->samples_.size());
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    A0[i] = f(this->samples_[i]);
  }
  *this->log_ << "Starting TT-cross ACA...\n";
  continuousACA();

  auto sites = SiteSet(this->d_, this->n_);
  vector<Index> l(this->d_ - 1);
  MPS psi(this->d_);
  vector<int> ranks(this->d_ - 1);
  for(int i = 1; i < this->d_; ++i) {
    ranks[i - 1] = this->I_[i].size();
  }
  *this->log_ << "Computing Galerkin projection...\n";
  this->log_->flush();
  vector<double> dx(this->d_);
  for(int ii = 1; ii <= this->d_; ++ii) {
    auto& dom = this->basis_[ii - 1].dom();
    dx[ii - 1] = (dom.second - dom.first) / this->nbins_;
    auto s = sites(ii);
    if(ii != this->d_) {
      l[ii - 1] = Index(ranks[ii - 1], "Link,l=" + to_string(ii));
    }

    if(ii == 1) {
      psi.ref(1) = ITensor(s, prime(l[0]));
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int lr = 1; lr <= dim(l[0]); ++lr) {
          double result = 0.0;
          for(int j = 0; j < this->nbins_; ++j) {
            double x = this->grid_[0][j];
            vector<double> elements = { x };
            auto& right = this->J_[1][lr - 1];
            elements.insert(elements.end(), right.begin(), right.end());
            result += f(elements) * this->basis_[0](x, ss, false) * dx[0];
          }
          psi.ref(1).set(s = ss, prime(l[0]) = lr, result);
        }
      }
    } else if(ii == this->d_) {
      psi.ref(this->d_) = ITensor(s, l[this->d_ - 2]);
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int ll = 1; ll <= dim(l[this->d_ - 2]); ++ll) {
          double result = 0.0;
          for(int j = 0; j < this->nbins_; ++j) {
            double x = this->grid_[this->d_ - 1][j];
            vector<double> elements = { x };
            auto& left = this->I_[this->d_ - 1][ll - 1];
            elements.insert(elements.begin(), left.begin(), left.end());
            result += f(elements) * this->basis_[this->d_ - 1](x, ss, false) * dx[this->d_ - 1];
          }
          psi.ref(this->d_).set(s = ss, l[this->d_ - 2] = ll, result);
        }
      }
    } else {
      psi.ref(ii) = ITensor(s, l[ii - 2], prime(l[ii - 1]));
      for(int ss = 1; ss <= dim(s); ++ss) {
        for(int ll = 1; ll <= dim(l[ii - 2]); ++ll) {
          for(int lr = 1; lr <= dim(l[ii - 1]); ++lr) {
            double result = 0.0;
            for(int j = 0; j < this->nbins_; ++j) {
              double x = this->grid_[ii - 1][j];
              vector<double> elements = { x };
              auto& left = this->I_[ii - 1][ll - 1];
              elements.insert(elements.begin(), left.begin(), left.end());
              auto& right = this->J_[ii][lr - 1];
              elements.insert(elements.end(), right.begin(), right.end());
              result += f(elements) * this->basis_[ii - 1](x, ss, false) * dx[ii - 1];
            }
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
          Ainv.set(prime(l[ii - 1]) = jj + 1, l[ii - 1] = kk + 1, AinvMat(jj, kk));
        }
      }
      psi.ref(ii) *= Ainv;
    }
    *this->log_ << "Core " << ii << " done!\n";
    this->log_->flush();
  }

  this->vb_ = psi;
  vector<double> diff(this->samples_.size());
  for(unsigned i = 0; i < this->samples_.size(); ++i) {
    diff[i] = ttEval(this->vb_, this->basis_, this->samples_[i], this->conv_);
  }
  transform(diff.begin(), diff.end(), A0.begin(), diff.begin(), minus<double>());
  *this->log_ << "Relative l2 error = " << sqrt(norm(diff) / norm(A0)) << "\n";
  this->log_->flush();
}

pair<double, vector<double>> TTCross::vtop() const {
  double max = 0.0;
  vector<double> topsample;
  for(auto& s : this->samples_) {
    if(f(s) > max) {
      max = f(s);
      topsample = s;
    }
  }
  return make_pair(max, topsample);
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

void TTCross::writeVb(unsigned count) const {
  string ttfilename = "ttsketch.h5";
  if(this->walkers_mpi_) {
    ttfilename = "../" + ttfilename;
  }
  auto f = h5_open(ttfilename, 'a');
  h5_write(f, "vb_" + to_string(count - 1), this->vb_);
}

void TTCross::readVb(unsigned count) {
  string ttfilename = "ttsketch.h5";
  if(this->walkers_mpi_) {
    ttfilename = "../" + ttfilename;
  }
  auto f = h5_open(ttfilename, 'r');
  this->vb_ = h5_read<MPS>(f, "vb_" + to_string(count - 1));
}

void TTCross::addSample(vector<double>& sample) {
  this->samples_.push_back(vector<double>(this->d_));
  for(int i = 0; i < this->d_; ++i) {
    this->samples_.back()[i] = *lower_bound(this->grid_[i].begin(), this->grid_[i].end(), sample[i]);
  }
}

}
}
