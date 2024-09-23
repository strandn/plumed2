#include "BasisFunc.h"
#include "TTCross.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/ActionWithArguments.h"
#include "tools/Matrix.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

class TTFreeEnergy : public ActionWithArguments {
  
private:
  MPS vb_;
  vector<BasisFunc> basis_;
  vector<vector<double>> samples_;
  double kbt_;
  unsigned d_;
  int n_;
  double cutoff_;
  int maxrank_;
  vector<vector<vector<double>>> I_;
  vector<vector<vector<double>>> J_;
  vector<vector<double>> u_;
  vector<vector<double>> v_;
  vector<double> resfirst_;
  int aca_n_;
  double aca_epsabs_;
  double aca_epsrel_;
  int aca_limit_;
  int aca_key_;
  // vector<int> whichpos_;
  vector<double> grid_min_;
  vector<double> grid_max_;
  int grid_bin_1d_;
  int grid_bin_2d_;
  string filename_;
  int pos_;

  void doTask();
  void updateIJ(const std::vector<double>& ij);
  std::pair<double, int> diagACA(const std::vector<double>& Rk);
  void continuousACA();

public:
  explicit TTFreeEnergy(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void update() override { }
  void calculate() override { }
  void apply() override { }
  double f(const std::vector<double>& x) const;
  const std::vector<std::vector<std::vector<double>>>& I() const { return this->I_; }
  const std::vector<std::vector<std::vector<double>>>& J() const { return this->J_; }
  int d() const { return this->d_; }
};

PLUMED_REGISTER_ACTION(TTFreeEnergy,"TT_FES")

void TTFreeEnergy::registerKeywords(Keywords& keys) {
  ActionWithArguments::registerKeywords(keys);
  // keys.add("compulsory", "ARG", "Positions of arguments that you would like to make the free energy for");
  keys.use("ARG");
  // keys.add("compulsory", "WHICH", "Arguments that you would like to make the free energy for");
  keys.add("compulsory", "TEMP", "The system temperature");
  keys.add("compulsory", "GRID_MIN", "The minimum to use for the grid");
  keys.add("compulsory", "GRID_MAX", "The maximum to use for the grid");
  keys.add("compulsory", "GRID_BIN_1D", "100", "The number of bins to use for the 1D grid");
  keys.add("compulsory", "GRID_BIN_2D", "100", "The number of bins to use for the 2D grid");
  keys.add("compulsory", "ACA_CUTOFF", "1.0e-6", "Convergence threshold for TT-cross calculations");
  keys.add("compulsory", "ACA_RANK", "50", "Largest possible rank for TT-cross calculations");
  keys.add("compulsory", "ACA_N", "10000000", "Size of integration workspace");
  keys.add("compulsory", "ACA_EPSABS", "1.0e-12", "Absolute error limit for integration");
  keys.add("compulsory", "ACA_EPSREL", "1.0e-8", "Relative error limit for integration");
  keys.add("compulsory", "ACA_LIMIT", "10000000", "Maximum number of subintervals for integration");
  keys.add("compulsory", "ACA_KEY", "6", "Integration rule");
  keys.add("compulsory", "STRIDE", "1", "Frequency of reading samples");
  keys.add("compulsory", "SAMPLEFILE", "COLVAR", "Name of the file where samples are stored");
  keys.add("compulsory", "FILE", "fes", "Name of the file in which to write the free energy");
  keys.add("compulsory", "ITER", "20", "TT bias update number");
}

TTFreeEnergy::TTFreeEnergy(const ActionOptions& ao) :
  Action(ao),
  ActionWithArguments(ao),
  kbt_(0.0),
  pos_(0)
{
  this->kbt_ = getkBT();
  if(this->kbt_ == 0.0) {
    error("You must specify the system temperature using TEMP");
  }
  this->d_ = getNumberOfArguments();
  if(this->d_ < 2) {
    error("Number of arguments must be at least 2");
  }
  parseVector("GRID_MIN", this->grid_min_);
  if(this->grid_min_.size() != this->d_) {
    error("Number of arguments does not match number of GRID_MIN parameters");
  }
  parseVector("GRID_MAX", this->grid_max_);
  if(this->grid_max_.size() != this->d_) {
    error("Number of arguments does not match number of GRID_MAX parameters");
  }
  parse("GRID_BIN_1D", this->grid_bin_1d_);
  if(this->grid_bin_1d_ <= 0) {
    error("GRID_BIN_1D must be positive");
  }
  parse("GRID_BIN_2D", this->grid_bin_2d_);
  if(this->grid_bin_2d_ <= 0) {
    error("GRID_BIN_2D must be positive");
  }

  string filename = "COLVAR";
  parse("SAMPLEFILE", filename);
  IFile ifile;
  if(ifile.FileExist(filename)) {
    ifile.open(filename);
  } else {
    error("The file " + filename + " cannot be found!");
  }
  int every = 1;
  parse("STRIDE", every);
  if(every <= 0) {
    error("STRIDE must be positive");
  }

  vector<double> cv(this->d_);
  vector<Value> tmpvalues;
  int nsamples = 0;
  for(unsigned i = 0; i < this->d_; ++i) {
    tmpvalues.push_back(Value(this, getPntrToArgument(i)->getName(), false));
  }
  while(true) {
    double dummy;
    if(ifile.scanField("time", dummy)) {
      for(unsigned i = 0; i < this->d_; ++i) {
        ifile.scanField(&tmpvalues[i]);
        cv[i] = tmpvalues[i].get();
      }
      if(nsamples % every == 0) {
        this->samples_.push_back(cv);
      }
      ifile.scanField("ttsketch.bias", dummy);
      ifile.scanField();
    } else {
      break;
    }
    ++nsamples;
  }
  ifile.close();
  int count = 0;
  parse("ITER", count);
  if(count <= 0) {
    error("ITER must be positive");
  }
  
  auto f = h5_open("ttsketch.h5", 'r');
  this->vb_ = h5_read<MPS>(f, "vb_" + to_string(count));
  close(f);
  this->n_ = dim(siteIndex(this->vb_, 1));
  log << "  read TT from ttsketch.h5/vb_" << count << "\n";
  log << "  " << this->samples_.size() << " samples retrieved\n";

  for(unsigned i = 0; i < this->d_; ++i) {
    if(this->grid_max_[i] <= this->grid_min_[i]) {
      error("GRID_MAX parameters need to be greater than respective GRID_MIN parameters");
    }
    this->basis_.push_back(BasisFunc(make_pair(this->grid_min_[i], this->grid_max_[i]), this->n_, false, 0, 0.0, 0, 0.0, 0.0, 0, 0));
  }

  parse("ACA_RANK", this->maxrank_);
  if(this->maxrank_ <= 0) {
    error("ACA_RANK must be positive");
  }
  parse("ACA_CUTOFF", this->cutoff_);
  if(this->cutoff_ <= 0.0 || this->cutoff_ > 1.0) {
    error("ACA_CUTOFF must be between 0 and 1");
  }
  parse("ACA_N", this->aca_n_);
  if(this->aca_n_ <= 0) {
    error("ACA_N must be positive");
  }
  parse("ACA_EPSABS", this->aca_epsabs_);
  if(this->aca_epsabs_ < 0.0) {
    error("ACA_EPSABS must be nonnegative");
  }
  parse("ACA_EPSREL", this->aca_epsrel_);
  if(this->aca_epsrel_ <= 0.0) {
    error("ACA_EPSREL must be positive");
  }
  parse("ACA_LIMIT", this->aca_limit_);
  if(this->aca_limit_ <= 0 || this->aca_limit_ > this->aca_n_) {
    error("ACA_LIMIT must be no positive and no greater than ACA_N");
  }
  parse("ACA_KEY", this->aca_key_);
  if(this->aca_key_ < 1 || this->aca_key_ > 6) {
    error("ACA_KEY must be between 1 and 6");
  }

  for(auto& pivots : this->I_) {
    pivots.clear();
  }
  I_[0].push_back(vector<double>());
  for(auto& pivots : this->J_) {
    pivots.clear();
  }
  J_[0].push_back(vector<double>());

  // vector<string> whichnames;
  // parseVector("WHICH", whichnames);
  // if(whichnames.size() > this->d_) {
  //   error("Number of variables cannot be greater than the number of arguments");
  // }
  // for(unsigned i = 0; i < whichnames.size(); ++i) {
  //   bool found = false;
  //   for(unsigned j = 0; j < this->d_; ++j) {
  //     if (whichnames[i] == getPntrToArgument(j)->getName()) {
  //       found = true;
  //       this->whichpos_.push_back(j);
  //     }
  //   }
  //   if(!found) {
  //     error("Variable " + whichnames[i] + " does not match any of the arguments");
  //   }
  // }
  parse("FILE", this->filename_);

  doTask();
}

struct TTFESParams {
  const TTFreeEnergy* instance;
  unsigned ii;
  int ll;
  int lr;
};

double ttfes_f(double x, void* params) {
  TTFESParams* aca_params = (TTFESParams*)params;
  int ii = aca_params->ii;
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
  return aca_params->instance->f(elements);
}

void TTFreeEnergy::doTask() {
  log << "Starting TT-cross ACA...\n";
  continuousACA();

  vector<vector<double>> xlist_1d(this->d_, vector<double>(this->grid_bin_1d_, 0.0));
  vector<vector<double>> xlist_2d(this->d_, vector<double>(this->grid_bin_2d_, 0.0));
  for(unsigned i = 0; i < this->d_; ++i) {
    for(int j = 0; j < this->grid_bin_1d_; ++j) {
      xlist_1d[i][j] = this->grid_min_[i] + j * (this->grid_max_[i] - this->grid_min_[i]) / this->grid_bin_1d_;
    }
    for(int j = 0; j < this->grid_bin_2d_; ++j) {
      xlist_2d[i][j] = this->grid_min_[i] + j * (this->grid_max_[i] - this->grid_min_[i]) / this->grid_bin_2d_;
    }
  }

  auto sites_1d = SiteSet(this->d_, this->grid_bin_1d_);
  auto sites_2d = SiteSet(this->d_, this->grid_bin_2d_);
  vector<Index> l(this->d_ - 1);
  vector<ITensor> intevals(this->d_);
  vector<ITensor> gridevals_1d(this->d_);
  vector<ITensor> gridevals_2d(this->d_);
  vector<int> ranks(this->d_ - 1);
  gsl_set_error_handler_off();
  gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(this->aca_n_);
  double result, error;
  for(unsigned i = 1; i < this->d_; ++i) {
    ranks[i - 1] = this->I_[i].size();
  }
  for(unsigned ii = 1; ii <= this->d_; ++ii) {
    auto s1d = sites_1d(ii);
    auto s2d = sites_2d(ii);
    if(ii != this->d_) {
      l[ii - 1] = Index(ranks[ii - 1], "Link,l=" + to_string(ii));
    }

    if(ii == 1) {
      intevals[0] = ITensor(prime(l[0]));
      gridevals_1d[0] = ITensor(s1d, prime(l[0]));
      gridevals_2d[0] = ITensor(s2d, prime(l[0]));
      for(int lr = 1; lr <= dim(l[0]); ++lr) {
        TTFESParams ttfes_params = { this, 1, 0, lr };
        gsl_function F;
        F.function = &ttfes_f;
        F.params = &ttfes_params;
        gsl_integration_qag(&F, this->grid_min_[0], this->grid_max_[0],
                            this->aca_epsabs_, this->aca_epsrel_,
                            this->aca_limit_, this->aca_key_, workspace,
                            &result, &error);
        intevals[0].set(prime(l[0]) = lr, result);
        for(int ss = 1; ss <= dim(s1d); ++ss) {
          vector<double> elements = { xlist_1d[0][ss - 1] };
          auto& right = this->J_[ii][lr - 1];
          elements.insert(elements.end(), right.begin(), right.end());
          gridevals_1d[0].set(s1d = ss, prime(l[0]) = lr, f(elements));
        }
        for(int ss = 1; ss <= dim(s2d); ++ss) {
          vector<double> elements = { xlist_2d[0][ss - 1] };
          auto& right = this->J_[ii][lr - 1];
          elements.insert(elements.end(), right.begin(), right.end());
          gridevals_2d[0].set(s2d = ss, prime(l[0]) = lr, f(elements));
        }
      }
    } else if(ii == this->d_) {
      intevals[this->d_ - 1] = ITensor(l[this->d_ - 2]);
      gridevals_1d[this->d_ - 1] = ITensor(s1d, l[this->d_ - 2]);
      gridevals_2d[this->d_ - 1] = ITensor(s2d, l[this->d_ - 2]);
      for(int ll = 1; ll <= dim(l[this->d_ - 2]); ++ll) {
        TTFESParams aca_params = { this, this->d_, ll, 0 };
        gsl_function F;
        F.function = &ttfes_f;
        F.params = &aca_params;
        gsl_integration_qag(&F, this->grid_min_[this->d_ - 1],
                            this->grid_max_[this->d_ - 1], this->aca_epsabs_,
                            this->aca_epsrel_, this->aca_limit_,
                            this->aca_key_, workspace, &result, &error);
        intevals[this->d_ - 1].set(l[this->d_ - 2] = ll, result);
        for(int ss = 1; ss <= dim(s1d); ++ss) {
          vector<double> elements = { xlist_1d[this->d_  - 1][ss - 1] };
          auto& left = this->I_[ii - 1][ll - 1];
          elements.insert(elements.begin(), left.begin(), left.end());
          gridevals_1d[this->d_ - 1].set(s1d = ss, l[this->d_ - 2] = ll, f(elements));
        }
        for(int ss = 1; ss <= dim(s2d); ++ss) {
          vector<double> elements = { xlist_2d[this->d_  - 1][ss - 1] };
          auto& left = this->I_[ii - 1][ll - 1];
          elements.insert(elements.begin(), left.begin(), left.end());
          gridevals_2d[this->d_ - 1].set(s2d = ss, l[this->d_ - 2] = ll, f(elements));
        }
      }
    } else {
      intevals[ii - 1] = ITensor(l[ii - 2], prime(l[ii - 1]));
      gridevals_1d[ii - 1] = ITensor(s1d, l[ii - 2], prime(l[ii - 1]));
      gridevals_2d[ii - 1] = ITensor(s2d, l[ii - 2], prime(l[ii - 1]));
      for(int ll = 1; ll <= dim(l[ii - 2]); ++ll) {
        for(int lr = 1; lr <= dim(l[ii - 1]); ++lr) {
          TTFESParams aca_params = { this, ii, ll, lr };
          gsl_function F;
          F.function = &ttfes_f;
          F.params = &aca_params;
          gsl_integration_qag(&F, this->grid_min_[i - 1],
                              this->grid_max_[i - 1], this->aca_epsabs_,
                              this->aca_epsrel_, this->aca_limit_,
                              this->aca_key_, workspace, &result, &error);
          intevals[ii - 1].set(l[ii - 2] = ll, prime(l[ii - 1]) = lr, result);
          for(int ss = 1; ss <= dim(s1d); ++ss) {
            vector<double> elements = { xlist_1d[ii - 1][ss - 1] };
            auto& left = this->I_[ii - 1][ll - 1];
            elements.insert(elements.begin(), left.begin(), left.end());
            auto& right = this->J_[ii][lr - 1];
            elements.insert(elements.end(), right.begin(), right.end());
            gridevals_1d[ii - 1].set(s1d = ss, l[ii - 2] = ll, prime(l[ii - 1]) = lr, f(elements));
          }
          for(int ss = 1; ss <= dim(s2d); ++ss) {
            vector<double> elements = { xlist_2d[ii - 1][ss - 1] };
            auto& left = this->I_[ii - 1][ll - 1];
            elements.insert(elements.begin(), left.begin(), left.end());
            auto& right = this->J_[ii][lr - 1];
            elements.insert(elements.end(), right.begin(), right.end());
            gridevals_2d[ii - 1].set(s2d = ss, l[ii - 2] = ll, prime(l[ii - 1]) = lr, f(elements));
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
      intevals[ii - 1] *= Ainv;
      gridevals_1d[ii - 1] *= Ainv;
      gridevals_2d[ii - 1] *= Ainv;
    }
  }
  gsl_integration_workspace_free(workspace);

  OFile file;
  file.link(*this);
  file.enforceSuffix("");
  file.open(this->filename_);
  file.setHeavyFlush();
  for(unsigned i = 0; i < this->d_; ++i) {
    file.setupPrintValue(getPntrToArgument(i));
  }
  for(unsigned i = 0; i < this->d_; ++i) {
    ITensor grid1d = i == 0 ? gridevals_1d[0] : intevals[0];
    for(unsigned k = 1; k < this->d_; ++k) {
      grid1d *= i == k ? gridevals_1d[k] : intevals[k];
    }
    for(int k = 0; k < this->grid_bin_1d_; ++k) {
      file.printField(getPntrToArgument(i), xlist_1d[i][k]);
      file.printField("fes_" + getPntrToArgument(i)->getName(), grid1d.elt(sites_1d(i + 1) = k + 1));
      file.printField();
    }
  }
  for(unsigned i = 0; i < this->d_; ++i) {
    for(unsigned j = i + 1; j < this->d_; ++j) {
      ITensor grid2d = i == 0 ? gridevals_2d[0] : intevals[0];
      for(unsigned k = 1; k < this->d_; ++k) {
        grid2d *= i == k || j == k ? gridevals_2d[k] : intevals[k];
      }
      for(int k = 0; k < this->grid_bin_2d_; ++k) {
        for(int l = 0; l < this->grid_bin_2d_; ++l) {
          file.printField(getPntrToArgument(i), xlist_2d[i][l]);
          file.printField(getPntrToArgument(j), xlist_2d[i][k]);
          file.printField("fes_" + getPntrToArgument(i)->getName() + "_" + getPntrToArgument(j)->getName(),
                          grid2d.elt(sites_1d(i + 1) = l + 1, sites_1d(j + 1) = k + 1));
          file.printField();
        }
      }
    }
  }
}

double TTFreeEnergy::f(const std::vector<double>& x) const {
  double bias = max(ttEval(this->vb_, this->basis_, x, false), 0.0);
  return exp(bias / this->kbt_);
}

void TTFreeEnergy::updateIJ(const std::vector<double>& ij) {
  this->I_[this->pos_].push_back(vector<double>(ij.begin(), ij.begin() + this->pos_));
  this->J_[this->pos_].push_back(vector<double>(ij.begin() + this->pos_, ij.end()));
}

std::pair<double, int> TTFreeEnergy::diagACA(const std::vector<double>& Rk) {
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

void TTFreeEnergy::continuousACA() {
  int order = this->d_;

  this->pos_ = 0;
  for(int i = 0; i < order - 1; ++i) {
    log << "pos = " << i + 1 << "\n";
    log.flush();
    ++this->pos_;

    vector<double> Rk(this->samples_.size());
    //TODO: parallelize
    for(unsigned i = 0; i < this->samples_.size(); ++i) {
      Rk[i] = f(this->samples_[i]);
    }
    this->u_.clear();
    this->v_.clear();
    for(int r = 1; r <= this->maxrank_; ++r) {
      auto [res_new, ik] = diagACA(Rk);
      auto& xy = this->samples_[ik];
      if(this->I_[i + 1].empty()) {
        this->resfirst_.push_back(res_new);
      } else if(res_new > this->resfirst_[i]) {
        this->resfirst_[i] = res_new;
      } else if(res_new / this->resfirst_[i] < this->cutoff_) {
        break;
      }

      updateIJ(xy);
      vector<double> uv(this->samples_.size());
      transform(this->u_[r - 1].begin(), this->u_[r - 1].end(), this->v_[r - 1].begin(), uv.begin(), multiplies<double>());
      transform(Rk.begin(), Rk.end(), uv.begin(), Rk.begin(), minus<double>());
      log << "rank = " << r << " res = " << res_new << " xy = ( ";
      for(double elt : xy) {
        log << elt << " ";
      }
      log << ")\n";
      log.flush();
    }
  }
}

}
}
