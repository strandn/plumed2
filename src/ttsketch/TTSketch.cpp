#include "BasisFunc.h"
#include "TTCross.h"
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Communicator.h"
#include "tools/Matrix.h"
#include "itensor/all.h"
#include <numeric>

using namespace std;
using namespace itensor;
using namespace PLMD::bias;

namespace PLMD {
namespace ttsketch {

class TTSketch : public Bias {

private:
  int rc_;
  int r_;
  double cutoff_;
  double kbt_;
  int pace_;
  int stride_;
  unsigned d_;
  MPS rho_;
  TTCross aca_;
  vector<BasisFunc> basis_;
  vector<vector<double>> samples_;
  double vmax_;
  double alpha_;
  double lambda_;
  bool isFirstStep_;
  unsigned count_;
  double bf_;
  bool conv_;

  double getBiasAndDerivatives(const vector<double>& cv, vector<double>& der);
  double getBias(const vector<double>& cv);
  void paraSketch();
  MPS createTTCoeff() const;
  pair<vector<ITensor>, IndexSet> intBasisSample(const IndexSet& is) const;
  tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is);

public:
  explicit TTSketch(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate();
  void update();
};

PLUMED_REGISTER_ACTION(TTSketch, "TTSKETCH")

void TTSketch::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.addFlag("NOCONV", false, "Specifies that densities and gradients should not be smoothed via Gaussian kernels whenever evaluated");
  // keys.addFlag("WALKERS_MPI", false, "To be used when gromacs + multiple walkers are used");
  keys.add("optional", "RANK", "Target rank for TTSketch algorithm - compulsory if CUTOFF is not specified");
  keys.add("optional", "CUTOFF", "Truncation error cutoff for singular value decomposition - compulsory if RANK is not specified");
  keys.add("optional", "TEMP", "The system temperature");
  keys.add("optional", "VMAX", "Upper limit of Vbias across all CV space, in units of kT");
  keys.add("optional", "NBINS", "Number of bins per dimension for storing convolution integrals");
  keys.add("optional", "WIDTH", "Width of Gaussian kernels");
  keys.add("optional", "CONV_N", "Size of integration workspace");
  keys.add("optional", "CONV_EPSABS", "Absolute error limit for integration");
  keys.add("optional", "CONV_EPSREL", "Relative error limit for integration");
  keys.add("optional", "CONV_LIMIT", "Maximum number of subintervals for integration");
  keys.add("optional", "CONV_KEY", "Integration rule");
  keys.add("compulsory", "INITRANK", "Initial rank for TTSketch algorithm");
  keys.add("compulsory", "PACE", "1e6", "The frequency for Vbias updates");
  keys.add("compulsory", "SAMPLESTRIDE", "100", "The frequency with which samples are collected for density estimation");
  keys.add("compulsory", "INTERVAL_MIN", "Lower limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "INTERVAL_MAX", "Upper limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "NBASIS", "20", "Number of Fourier basis functions per dimension");
  keys.add("compulsory", "ALPHA", "0.05", "Weight coefficient for random tensor train construction");
  keys.add("compulsory", "LAMBDA", "100.0", "Ratio of largest to smallest allowed density magnitudes");
  keys.add("optional", "BIASFACTOR", "For well-tempering");
  // keys.add("compulsory", "FILE", "name of the file where tensor trains and related data are stored");
  keys.add("compulsory", "ACA_CUTOFF", "0.05", "Convergence threshold for TT-cross calculations");
  keys.add("compulsory", "ACA_RANK", "30", "Largest possible rank for TT-cross calculations");
  keys.add("compulsory", "ACA_N", "10000000", "Size of integration workspace");
  keys.add("compulsory", "ACA_EPSABS", "1.0e-8", "Absolute error limit for integration");
  keys.add("compulsory", "ACA_EPSREL", "1.0e-6", "Relative error limit for integration");
  keys.add("compulsory", "ACA_LIMIT", "10000000", "Maximum number of subintervals for integration");
  keys.add("compulsory", "ACA_KEY", "6", "Integration rule");
}

TTSketch::TTSketch(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  r_(0),
  cutoff_(0.0),
  kbt_(0.0),
  vmax_(numeric_limits<double>::max()),
  isFirstStep_(true),
  count_(1),
  bf_(1.0),
  conv_(true)
{
  bool noconv = false;
  parseFlag("NOCONV", noconv);
  // bool walkers_mpi = false;
  // parseFlag("WALKERS_MPI", walkers_mpi);
  parse("RANK", this->r_);
  parse("CUTOFF", this->cutoff_);
  if(this->r_ <= 0 && (this->cutoff_ <= 0.0 || this->cutoff_ > 1.0)) {
    error("Valid RANK or CUTOFF needs to be specified");
  }
  this->kbt_ = getkBT();
  if(this->kbt_ == 0.0) {
    error("Unless the MD engine passes the temperature to plumed, you must specify it using TEMP");
  }
  parse("VMAX", this->vmax_);
  if(this->vmax_ <= 0.0) {
    error("VMAX must be positive");
  } else if(this->vmax_ != numeric_limits<double>::max()) {
    this->vmax_ *= this->kbt_;
  }
  int nbins = 1000;
  parse("NBINS", nbins);
  if(!noconv && nbins <= 0) {
    error("Gaussian smoothing requires positive NBINS");
  }
  double w = 0.1;
  parse("WIDTH", w);
  if(!noconv && w <= 0.0) {
    error("Gaussian smoothing requires positive WIDTH");
  }
  int conv_n = 10000000;
  parse("CONV_N", conv_n);
  if(!noconv && conv_n <= 0) {
    error("Gaussian smoothing requires positive CONV_N");
  }
  double conv_epsabs = 1.0e-12;
  parse("CONV_EPSABS", conv_epsabs);
  if(!noconv && conv_epsabs < 0.0) {
    error("Gaussian smoothing requires nonnegative CONV_EPSABS");
  }
  double conv_epsrel = 1.0e-8;
  parse("CONV_EPSREL", conv_epsrel);
  if(!noconv && conv_epsrel <= 0.0) {
    error("Gaussian smoothing requires positive CONV_EPSREL");
  }
  int conv_limit = 10000000;
  parse("CONV_LIMIT", conv_limit);
  if(!noconv && (conv_limit <= 0 || conv_limit > conv_n)) {
    error("Gaussian smoothing requires positive CONV_LIMIT no greater than CONV_N");
  }
  int conv_key = 6;
  parse("CONV_KEY", conv_key);
  if(!noconv && (conv_key < 1 || conv_key > 6)) {
    error("Gaussian smoothing requires CONV_KEY between 1 and 6");
  }
  parse("INITRANK", this->rc_);
  if(this->rc_ <= 0) {
    error("INITRANK must be positive");
  }
  parse("PACE", this->pace_);
  if(this->pace_ <= 0) {
    error("PACE must be positive");
  }
  parse("SAMPLESTRIDE", this->stride_);
  if(this->stride_ <= 0 || this->stride_ > this->pace_) {
    error("SAMPLESTRIDE must be positive and no greater than PACE");
  }
  this->d_ = getNumberOfArguments();
  if(this->d_ < 2) {
    error("Number of arguments must be at least 2");
  }
  vector<double> interval_min;
  parseVector("INTERVAL_MIN", interval_min);
  if(interval_min.size() != this->d_) {
    error("Number of arguments does not match number of INTERVAL_MIN parameters");
  }
  vector<double> interval_max;
  parseVector("INTERVAL_MAX", interval_max);
  if(interval_max.size() != this->d_) {
    error("Number of arguments does not match number of INTERVAL_MAX parameters");
  }
  int nbasis = 20;
  parse("NBASIS", nbasis);
  if(nbasis <= 0) {
    error("NBASIS must be positive");
  }
  parse("ALPHA", this->alpha_);
  if(this->alpha_ <= 0.0 || this->alpha_ > 1.0) {
    error("ALPHA must be positive and no greater than 1");
  }
  parse("LAMBDA", this->lambda_);
  if(this->lambda_ <= 1.0) {
    error("LAMBDA must be greater than 1");
  }
  parse("BIASFACTOR", this->bf_);
  if(this->bf_ < 1.0) {
    error("LAMBDA must be greater than 1");
  }
  // string file;
  // parse("FILE", file);
  // if(file.length() == 0) {
  //   error("No TTSketch file name was specified");
  // }
  for(unsigned i = 0; i < this->d_; ++i) {
    if(interval_max[i] <= interval_min[i]) {
      error("INTERVAL_MAX parameters need to be greater than respective INTERVAL_MIN parameters");
    }
    this->basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]),
                                         nbasis, !noconv, nbins, w, conv_n,
                                         conv_epsabs, conv_epsrel, conv_limit,
                                         conv_key));
  }
  this->conv_ = !noconv;

  int aca_rank = 30;
  parse("ACA_RANK", aca_rank);
  if(aca_rank <= 0) {
    error("ACA_RANK must be positive");
  }
  double aca_cutoff = 0.05;
  parse("ACA_CUTOFF", aca_cutoff);
  if(aca_cutoff <= 0.0 || aca_cutoff > 1.0) {
    error("ACA_CUTOFF must be between 0 and 1");
  }
  int aca_n = 10000000;
  parse("ACA_N", aca_n);
  if(aca_n <= 0) {
    error("ACA_N must be positive");
  }
  double aca_epsabs = 1.0e-8;
  parse("ACA_EPSABS", aca_epsabs);
  if(aca_epsabs < 0.0) {
    error("ACA_EPSABS must be nonnegative");
  }
  double aca_epsrel = 1.0e-6;
  parse("ACA_EPSREL", aca_epsrel);
  if(aca_epsrel <= 0.0) {
    error("ACA_EPSREL must be positive");
  }
  int aca_limit = 10000000;
  parse("ACA_LIMIT", aca_limit);
  if(aca_limit <= 0 || aca_limit > aca_n) {
    error("ACA_LIMIT must be no positive and no greater than ACA_N");
  }
  int aca_key = 6;
  parse("ACA_KEY", aca_key);
  if(aca_key < 1 || aca_key > 6) {
    error("ACA_KEY must be between 1 and 6");
  }
  this->aca_ = TTCross(this->basis_, getkBT(), aca_cutoff, aca_rank, log, aca_n, aca_epsabs, aca_epsrel, aca_limit, aca_key, !noconv);
}

void TTSketch::calculate() {
  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; ++i) {
    cv[i] = getArgument(i);
  }

  vector<double> der(this->d_, 0.0);
  double ene = getBiasAndDerivatives(cv, der);
  setBias(ene);
  for(unsigned i = 0; i < this->d_; ++i) {
    setOutputForce(i, -der[i]);
  }
}

void TTSketch::update() {
  bool nowAddATT;
  if(getStep() % this->pace_ == 0 && !this->isFirstStep_) {
    nowAddATT = true;
  } else {
    nowAddATT = false;
    this->isFirstStep_ = false;
  }

  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; ++i) {
    cv[i] = getArgument(i);
  }
  if(getStep() % this->stride_ == 0) {
    this->samples_.push_back(cv);
  }

  if(nowAddATT) {
    int N = this->pace_ / this->stride_;
    log << "Sample limits\n";
    for(unsigned i = 0; i < this->d_; ++i) {
      auto [large, small] = this->basis_[i].dom();
      for(int j = 0; j < N; ++j) {
        int jadj = j + this->samples_.size() - N;
        if(this->samples_[jadj][i] > large) {
          large = this->samples_[jadj][i];
        }
        if(this->samples_[jadj][i] < small) {
          small = this->samples_[jadj][i];
        }
      }
      log << small << " " << large << "\n";
    }

    log << "\nForming TT-sketch density...\n";
    log.flush();
    paraSketch();

    double rhomax = 0.0;
    //TODO: parallelize
    for(auto& s : this->samples_) {
      double rho = ttEval(this->rho_, this->basis_, s, this->conv_);
      if(rho > rhomax) {
        rhomax = rho;
      }
    }
    //TODO: figure out if arithmetic or geometric mean
    double hf = 1.0;
    double vmean = 0.0;
    if(this->bf_ > 1.0) {
      int N = this->pace_ / this->stride_;
      vector<double> vlist(N);
      for(int i = 0; i < N; ++i) {
        vlist[i] = getBias(this->samples_[i + this->samples_.size() - N]);
      }
      vmean = accumulate(vlist.begin(), vlist.end(), 0.0) / N;
      hf = exp(-vmean / (this->kbt_ * (this->bf_ - 1)));
    }
    this->rho_ *= pow(this->lambda_, hf) / rhomax;
    this->aca_.updateG(this->rho_);

    this->aca_.updateVshift(0.0);
    double vpeak = this->aca_.vtop(this->samples_);
    double vshift = max(vpeak - this->vmax_, 0.0);
    this->aca_.updateVshift(vshift);
    log << "\n";
    if(this->bf_ > 1.0) {
      log << "Vmean = " << vmean << " Height = " << this->kbt_ * std::log(pow(this->lambda_, hf)) << "\n";
    }
    log << "Vtop = " << vpeak << " Vshift = " << vshift << "\n\n";

    this->aca_.updateVb(this->samples_);

    vector<double> gradtop(this->d_, 0.0);
    vector<vector<double>> topsamples(this->d_);
    for(auto& s : this->samples_) {
      vector<double> der(this->d_, 0.0);
      getBiasAndDerivatives(s, der);
      for(unsigned i = 0; i < this->d_; ++i) {
        if(abs(der[i]) > gradtop[i]) {
          gradtop[i] = abs(der[i]);
          topsamples[i] = s;
        }
      }
    }
    log << "\ngradtop ";
    for(unsigned i = 0; i < this->d_; ++i) {
      log << gradtop[i] << " ";
    }
    log << "\n";
    
    for(unsigned i = 0; i < this->d_; ++i) {
      for(unsigned j = 0; j < this->d_; ++j) {
        log << topsamples[i][j] << " ";
      }
      log << "\n";
    }
    log << "\n";
    log.flush();
    
    // ofstream file;
    // if(this->count_ == 2) {
    //   file.open("F.txt");
    // } else {
    //   file.open("F.txt", ios_base::app);
    // }
    // for(int i = 0; i < 100; ++i) {
    //   double x = -M_PI + 2 * i * M_PI / 100;
    //   for(int j = 0; j < 100; ++j) {
    //     double y = -M_PI + 2 * j * M_PI / 100;
    //     file << x << " " << y << " " << max(ttEval(this->aca_.vb(), this->basis_, { x, y }, false), 0.0) << endl;
    //   }
    // }
    // file.close();
  }
  if(getStep() % this->pace_ == 1) {
    log << "Vbias update " << this->count_ << "...\n\n";
    log.flush();
  }
}

double TTSketch::getBiasAndDerivatives(const vector<double>& cv, vector<double>& der) {
  double bias = getBias(cv);
  if(bias > 0.0) {
    der = ttGrad(this->aca_.vb(), this->basis_, cv, this->conv_);
  }
  return bias;
}

double TTSketch::getBias(const vector<double>& cv) {
  if(length(this->aca_.vb()) == 0) {
    return 0.0;
  }
  return max(ttEval(this->aca_.vb(), this->basis_, cv, this->conv_), 0.0);
}

void TTSketch::paraSketch() {
  int N = this->pace_ / this->stride_;
  auto coeff = createTTCoeff();
  auto [M, is] = intBasisSample(siteInds(coeff));
  auto& G = this->rho_;
  G = MPS(this->d_);

  auto [Bemp, envi_L, envi_R] = formTensorMoment(M, coeff, is);
  auto links = linkInds(coeff);
  vector<ITensor> V(this->d_);
  G.ref(1) = Bemp(1);
  for(unsigned core_id = 2; core_id <= this->d_; ++core_id) {
    int rank = dim(links(core_id - 1));
    Matrix<double> LMat(N, rank), RMat(N, rank);
    for(int i = 1; i <= N; ++i) {
      for(int j = 1; j <= rank; ++j) {
        LMat(i - 1, j - 1) = envi_L[core_id - 1].elt(is(core_id) = i, links(core_id - 1) = j);
        RMat(i - 1, j - 1) = envi_R[core_id - 2].elt(is(core_id - 1) = i, links(core_id - 1) = j);
      }
    }
    Matrix<double> Lt, AMat, PMat;
    transpose(LMat, Lt);
    mult(Lt, RMat, AMat);
    pseudoInvert(AMat, PMat);

    ITensor A(prime(links(core_id - 1)), links(core_id - 1)), Pinv(prime(links(core_id - 1)), links(core_id - 1));
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= rank; ++j) {
        A.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat(i - 1, j - 1));
        Pinv.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, PMat(i - 1, j - 1));
      }
    }
    G.ref(core_id) = noPrime(Pinv * Bemp(core_id));
    auto original_link_tags = tags(links(core_id - 1));
    ITensor U, S;
    V[core_id - 1] = ITensor(links(core_id - 1));
    if(this->r_ > 0) {
      svd(A, U, S, V[core_id - 1], {"Cutoff=", this->cutoff_, "RightTags=", original_link_tags, "MaxDim=", this->r_});
    } else {
      svd(A, U, S, V[core_id - 1], {"Cutoff=", this->cutoff_, "RightTags=", original_link_tags});
    }
  }
  log << "Initial ranks ";
  for(unsigned i = 1; i < this->d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";
  log.flush();

  G.ref(1) *= V[1];
  for(unsigned core_id = 2; core_id < this->d_; ++core_id) {
    G.ref(core_id) *= V[core_id - 1];
    G.ref(core_id) *= V[core_id];
  }
  G.ref(this->d_) *= V[this->d_ - 1];
  log << "Final ranks ";
  for(unsigned i = 1; i < this->d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";
  log.flush();

  ++this->count_;
}

MPS TTSketch::createTTCoeff() const {
  int n = this->basis_[0].nbasis();
  auto sites = SiteSet(this->d_, n);
  auto coeff = randomMPS(sites, this->rc_);
  for(unsigned i = 1; i <= this->d_; ++i) {
    auto s = sites(i);
    auto sp = prime(s);
    vector<double> Avec(n, this->alpha_);
    Avec[0] = 1.0;
    auto A = diagITensor(Avec, s, sp);
    coeff.ref(i) *= A;
    coeff.ref(i).noPrime();
  }
  return coeff;
}

pair<vector<ITensor>, IndexSet> TTSketch::intBasisSample(const IndexSet& is) const {
  int N = this->pace_ / this->stride_;
  int nb = this->basis_[0].nbasis();
  auto sites_new = SiteSet(this->d_, N);
  vector<ITensor> M;
  vector<Index> is_new;
  for(unsigned i = 1; i <= this->d_; ++i) {
    M.push_back(ITensor(sites_new(i), is(i)));
    is_new.push_back(sites_new(i));
    for(int j = 1; j <= N; ++j) {
      int jadj = j + this->samples_.size() - N;
      for(int k = 1; k <= nb; ++k) {
        M.back().set(sites_new(i) = j, is(i) = k, pow(1.0 / N, 1.0 / this->d_) * this->basis_[i - 1](this->samples_[jadj - 1][i - 1], k, false));
      }
    }
  }
  return make_pair(M, IndexSet(is_new));
}

tuple<MPS, vector<ITensor>, vector<ITensor>> TTSketch::formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is) {
  int N = dim(is(1));
  auto links = linkInds(coeff);
  auto L = coeff;

  for(unsigned i = 1; i <= this->d_; ++i) {
    L.ref(i) *= M[i - 1];
  }

  vector<ITensor> envi_L(this->d_);
  envi_L[1] = L(1) * delta(is(1), is(2));
  for(unsigned i = 2; i < this->d_; ++i) {
    int rankl = dim(links(i - 1));
    int rankr = dim(links(i));
    envi_L[i] = ITensor(is(i + 1), links(i));
    for(int j = 1; j <= N; ++j) {
      for(int k = 1; k <= rankr; ++k) {
        ITensor LHS(links(i - 1)), RHS(links(i - 1));
        for(int ii = 1; ii <= rankl; ++ii) {
          LHS.set(links(i - 1) = ii, envi_L[i - 1].elt(is(i) = j, links(i - 1) = ii));
          RHS.set(links(i - 1) = ii, L(i).elt(links(i - 1) = ii, is(i) = j, links(i) = k));
        }
        envi_L[i].set(is(i + 1) = j, links(i) = k, elt(LHS * RHS));
      }
    }
  }

  vector<ITensor> envi_R(this->d_);
  envi_R[this->d_ - 2] = L(this->d_) * delta(is(this->d_), is(this->d_ - 1));
  for(int i = this->d_ - 3; i >= 0; --i) {
    int rankl = dim(links(i + 1));
    int rankr = dim(links(i + 2));
    envi_R[i] = ITensor(is(i + 1), links(i + 1));
    for(int j = 1; j <= N; ++j) {
      for(int k = 1; k <= rankl; ++k) {
        ITensor LHS(links(i + 2)), RHS(links(i + 2));
        for(int ii = 1; ii <= rankr; ++ii) {
          LHS.set(links(i + 2) = ii, envi_R[i + 1].elt(is(i + 2) = j, links(i + 2) = ii));
          RHS.set(links(i + 2) = ii, L(i + 2).elt(links(i + 2) = ii, is(i + 2) = j, links(i + 1) = k));
        }
        envi_R[i].set(is(i + 1) = j, links(i + 1) = k, elt(LHS * RHS));
      }
    }
  }

  MPS B(this->d_);
  B.ref(1) = envi_R[0] * M[0];
  for(unsigned core_id = 2; core_id < this->d_; ++core_id) {
    int rankl = dim(links(core_id - 1));
    int rankr = dim(links(core_id));
    B.ref(core_id) = ITensor(links(core_id - 1), is(core_id), links(core_id));
    for(int i = 1; i <= rankl; ++i) {
      for(int j = 1; j <= rankr; ++j) {
        for(int k = 1; k <= N; ++k) {
          double Lelt = envi_L[core_id - 1].elt(is(core_id) = k, links(core_id - 1) = i);
          double Relt = envi_R[core_id - 1].elt(is(core_id) = k, links(core_id) = j);
          B.ref(core_id).set(links(core_id - 1) = i, is(core_id) = k, links(core_id) = j, Lelt * Relt);
        }
      }
    }
    B.ref(core_id) *= M[core_id - 1];
  }
  B.ref(this->d_) = envi_L[this->d_ - 1] * M[this->d_ - 1];

  return make_tuple(B, envi_L, envi_R);
}

}
}
