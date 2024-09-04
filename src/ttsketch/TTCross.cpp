#include "TTCross.h"
#include <algorithm>
#include <cmath>

using namespace std;

namespace PLMD {
namespace ttsketch {

TTCross::TTCross(const vector<BasisFunc>& basis, double kbt, double cutoff, int maxrank, Log& log)
  : G_(nullptr), basis_(basis), n_(basis[0].nbasis()), kbt_(kbt), cutoff_(cutoff),
    maxrank_(maxrank), d_(basis.size()), pos_(0), vshift_(0.0),
    I_(vector<vector<vector<double>>>(vector<vector<double>>(), basis.size())),
    J_(vector<vector<vector<double>>>(vector<vector<double>>(), basis.size())),
    log_(log)
{
  I_[0].push_back(vector<double>());
  J_[0].push_back(vector<double>());
}

//TODO: fix this
double TTCross::f(const vector<double>& x) const {
  double result = this->kbt_ * log(max(eval(x, true), 1.0));
  if(this->vb_.length() != 0) {
    result += max(eval(x, false) - this->vshift_, 0.0);
  }
  return result;
}

double TTCross::operator()(const vector<double>& elements) const {
  vector<double> x(elements.begin(), elements.begin() + this->pos_);
  vector<double> y(elements.begin() + this->pos_, elements.end());
  int k = this->I_[this->pos_].size();
  vector<vector<double>> data, data_prev;
  for(int iter = 0; iter <= k; ++iter) {
    data.resize(k - iter + 1, vector<double>(k - iter + 1, 0.0));
    for(int i = 1; i <= k - iter + 1; ++i) {
      for(int j = 1; j <= k - iter + 1; ++j) {
        if(iter == 0) {
          auto row = i == k + 1 ? x : this->I_[this->pos][i - 1];
          auto col = j == k + 1 ? y : this->J_[this->pos][j - 1];
          row.insert(row.end(), col.begin(), col.end());
          data[i - 1][j - 1] = f(row);
        } else {
          data[i - 1][j - 1] = data_prev[i][j] - data_prev[i][0] * data_prev[0][j] / data_prev[0][0];
        }
      }
    }
    data_prev = data;
  }
  return data[0][0];
}

void TTCross::updateIJ(const vector<double>& ij) {
  this->I_[this->pos_].push_back(ij.begin(), ij.begin() + this->pos_);
  this->J_[this->pos_].push_back(ij.begin() + this->pos_, ij.end());
}

void TTCross::continuousACA(const vector<vector<double>>& samples) {
  int order = this->d_;

  this->pos_ = 0;
  for(int i = 0; i < order - 1; ++i) {
    this->log_ << "pos = " << i + 1 << "\n";
    this->log_.flush();
    ++this->pos_;

    int n_pivots = this->I_[i].size();
    double res_new = 0.0;
    int n_samples = samples.size();
    for(int r = 1; r <= this->maxrank_; ++r) {
      vector<double> results(n_samples);
      for(int k = 0; k < n_samples; ++k) {
        auto& pivot = this->I_[i][k % n_pivots];
        vector<double> arg(samples[k].begin() + this->pos_ - 1, samples[k].end());
        arg.insert(arg.begin(), pivot.begin(), pivot.end());
        results[k] = abs((*this)(arg));
      }

      int top = max_element(results.begin(), results.end()) - results.begin();
      auto& pivot_top = this->I_[i][top % n_pivots];
      vector<double> xy(samples[top].begin() + this->pos_ - 1, samples[top].end());
      xy.insert(xy.begin(), pivot_top.begin(), pivot_top.end());
      double res_new = results[top];
      updateIJ(xy);
      this->log_ << "rank = " << r << " res = " << res_new << " xy = " << xy << "\n";
      this->log_.flush();

      if(this->I_[i + 1].empty()) {
        this->resfirst_.push_back(res_new);
      } else if(res_new > this->resfirst_[i]) {
        this->resfirst_[i] = res_new;
      } else if(res_new / this->resfirst_[i] < this->cutoff_) {
        break;
      }
    }
  }
}

void TTCross::updateVb(const vector<vector<double>>& samples) {
  this->log_ << "\nStarting TT-cross ACA...\n";
  this->log_.flush();
}

}
}
