#include "BasisFunc.h"
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Communicator.h"
#include <Eigen/QR>
#include "itensor/all.h"

using namespace std;
using namespace Eigen;
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
  int d_;
  vector<MPS> rholist_;
  vector<double> rhomaxlist_;
  vector<BasisFunc> basis_;
  vector<vector<double>> samples_;
  double vmax_;
  double vshift_;
  double alpha_;
  double lambda_;
  bool isFirstStep_;

  double getBiasAndDerivatives(const vector<double>& cv, vector<double>& der);
  double getBias(const std::vector<double>& cv);
  void paraSketch();
  MPS createTTCoeff() const;
  pair<vector<ITensor>, IndexSet> intBasisSample(const IndexSet& is) const;
  tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is);
  double densEval(int step, const vector<double>& elements) const;
  vector<double> densGrad(int step, const vector<double>& elements) const;
  void setConv(bool status);

public:
  explicit TTSketch(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate();
  void update();
};

PLUMED_REGISTER_ACTION(TTSketch, "TTSKETCH")

TTSketch::TTSketch(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  r_(0),
  cutoff_(0.0),
  kbt_(0.0),
  vmax_(numeric_limits<double>::max()),
  vshift_(0.0),
  isFirstStep_(true)
{
  bool conv = true;
  parseFlag("CONV", conv);
  // bool walkers_mpi = false;
  // parseFlag("WALKERS_MPI", walkers_mpi);
  parse("RANK", r_);
  parse("CUTOFF", cutoff_);
  if(r_ <= 0 && (cutoff_ <= 0.0 || cutoff_ > 1.0)) {
    error("Valid RANK or CUTOFF needs to be specified");
  }
  kbt_ = getkBT();
  if(kbt_ == 0.0) {
    error("Unless the MD engine passes the temperature to plumed, you must specify it using TEMP");
  }
  parse("VMAX", vmax_);
  if(vmax_ <= 0.0) {
    error("VMAX must be positive")
  }
  int nbins = 100;
  arse("NBINS", nbins);
  if(conv && nbins <= 0) {
    error("Gaussian smoothing requires positive NBINS");
  }
  int w = 0.02;
  parse("WIDTH", w);
  if(conv && (width <= 0.0 || width > 1.0)) {
    error("Gaussian smoothing requires positive WIDTH no greater than 1");
  }
  int gsl_n = 1000;
  parse("GSL_N", gsl_n);
  if(conv && gsl_n <= 0) {
    error("Gaussian smoothing requires positive GSL_N");
  }
  int gsl_epsabs = 1.0e-10;
  parse("GSL_EPSABS", gsl_epsabs);
  if(conv && gsl_epsabs < 0.0) {
    error("Gaussian smoothing requires nonnegative GSL_EPSABS");
  }
  int gsl_epsrel = 1.0e-6;
  parse("GSL_EPSREL", gsl_epsrel);
  if(conv && gsl_epsrel < 0.0) {
    error("Gaussian smoothing requires nonnegative GSL_EPSREL");
  }
  int gsl_limit = 1000;
  parse("GSL_LIMIT", gsl_limit);
  if(conv && (gsl_limit <= 0 || gsl_limit > gsl_n)) {
    error("Gaussian smoothing requires positive GSL_LIMIT no greater than GSL_N");
  }
  int gsl_key = 2;
  parse("GSL_KEY", gsl_key);
  if(conv && (gsl_key < 1 || gsl_key > 6)) {
    error("Gaussian smoothing requires GSL_KEY between 1 and 6");
  }
  parse("INITRANK", rc_);
  if(rc_ <= 0) {
    error("INITRANK must be positive");
  }
  parse("PACE", pace_);
  if(pace_ <= 0) {
    error("PACE must be positive");
  }
  parse("STRIDE", stride_);
  if(stride_ <= 0 || stride_ > pace_) {
    error("STRIDE must be positive and no greater than PACE");
  }
  d_ = getNumberOfArguments();
  if(d_ < 2) {
    error("Number of arguments must be at least 2");
  }
  vector<double> interval_min;
  parseVector("INTERVAL_MIN", interval_min);
  if(interval_min.size() != d_) {
    error("Number of arguments does not match number of INTERVAL_MIN parameters")
  }
  vector<double> interval_max;
  parseVector("INTERVAL_MAX", interval_max);
  if(interval_max.size() != d_) {
    error("Number of arguments does not match number of INTERVAL_MAX parameters")
  }
  int nbasis = 20;
  parse("NBASIS", nbasis);
  if(nbasis <= 0) {
    error("NBASIS must be positive");
  }
  parse("ALPHA", alpha_);
  if(alpha_ <= 0.0 || alpha_ > 1.0) {
    error("ALPHA must be positive and no greater than 1");
  }
  parse("LAMBDA", lambda_);
  if(alpha_ <= 1.0) {
    error("LAMBDA must be greater than 1");
  }
  // string file;
  // parse("FILE", file);
  // if(file.length() == 0) {
  //   error("No TTSketch file name was specified");
  // }
  for(int i = 0; i < d_; ++i) {
    if(interval_max[i] <= interval_min[i]) {
      error("INTERVAL_MAX parameters need to be greater than respective INTERVAL_MIN parameters");
    }
    basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]),
                                         nbasis, conv, nbins, w, gsl_n,
                                         gsl_epsabs, gsl_epsrel, gsl_limit,
                                         gsl_key));
  }
}

void TTSketch::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.addFlag("CONV", true, "Specifies that densities and corresponding gradients are to be smoothed via Gaussian kernels whenever evaluated");
  // keys.addFlag("WALKERS_MPI",false,"To be used when gromacs + multiple walkers are used");
  keys.add("optional", "RANK", "Target rank for TTSketch algorithm - compulsory if CUTOFF is not specified");
  keys.add("optional", "CUTOFF", "Truncation error cutoff for singular value decomposition - compulsory if RANK is not specified");
  keys.add("optional", "TEMP", "The system temperature");
  keys.add("optional", "VMAX", "Upper limit of Vbias across all CV space, in units of kT");
  keys.add("optional", "NBINS", "Number of bins per dimension for storing convolution integrals");
  keys.add("optional", "WIDTH", "Width of Gaussian kernels - fraction of domains for all dimensions");
  keys.add("optional", "GSL_N", "Size of integration workspace");
  keys.add("optional", "GSL_EPSABS", "Absolute error limit for integration");
  keys.add("optional", "GSL_EPSREL", "Relative error limit for integration");
  keys.add("optional", "GSL_LIMIT", "Maximum number of subintervals for integration");
  keys.add("optional", "GSL_KEY", "Integration rule");
  keys.add("compulsory", "INITRANK", "Initial rank for TTSketch algorithm");
  keys.add("compulsory", "PACE", "1e6", "The frequency for Vbias updates");
  keys.add("compulsory", "STRIDE", "100", "The frequency with which samples are collected for density estimation");
  keys.add("compulsory", "INTERVAL_MIN", "Lower limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "INTERVAL_MAX", "Upper limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "NBASIS", "20", "Number of Fourier basis functions per dimension");
  keys.add("compulsory", "ALPHA", "0.05", "Weight coefficient for random tensor train construction");
  keys.add("compulsory", "LAMBDA", "100.0", "Ratio of largest to smallest allowed density magnitudes");
  // keys.add("compulsory", "FILE", "name of the file where tensor trains and related data are stored");
}

void TTSketch::calculate() {
  vector<double> cv(d_);
  for(int i = 0; i < d_; ++i) {
    cv[i] = getArgument(i);
  }

  double ene = 0.0;
  vector<double> der(d_, 0.0);
  ene = getBiasAndDerivatives(cv, der);
  setBias(ene);
  for(int i = 0; i < d_; ++i) {
    setOutputForce(i, -der[i]);
  }
}

void TTSketch::update() {
  bool nowAddATT;
  if(getStep() % pace_ && !isFirstStep_) {
    nowAddATT = true;
  } else {
    nowAddATT = false;
    isFirstStep_ = false;
  }

  vector<double> cv(d_);
  for(int i = 0; i < d_; ++i) {
    cv[i] = getArgument(i);
  }
  if(getStep() % stride_) {
    samples_.push_back(cv);
  }

  if(nowAddATT) {
    int N = pace_ / stride_;
    vector<pair<double, double>> domain_small(d_);
    log << "Sample limits\n";
    for(int i = 0; i < d_; ++i) {
      double max = 0.0, min = numeric_limits<double>::max();
      for(int j = 0; j < N; ++j) {
        int jadj = j + samples_.size() - N;
        if(samples_[jadj][i] > max) {
          max = samples_[jadj][i];
        }
        if(samples_[jadj][i] < min) {
          min = samples_[jadj][i];
        }
      }
      log << min << " " << max << "\n";
      domain_small[i] = make_pair(min, max);
    }

    log << "Forming TT...\n";
    setConv(false);
    paraSketch();

    double rhomax = 0.0;
    for(vector<double>& sample : samples_) {
        double rho = densEval(rholist_.size() - 1, sample);
        if(rho > rhomax) {
          rhomax = rho;
        }
    }
    rhomaxlist_.push_back(rhomax);

    double vtop = 0.0;
    vector<double> gradtop(d_, 0.0);
    for(vector<double>& sample : samples_) {
      vector<double> der(d_, 0.0);
      double result = getBiasAndDerivatives(sample, der);
      if(result > vtop) {
        vtop = result;
      }
      for(int i = 0; i < d_; ++i) {
        if(abs(der[i]) > gradtop[i]) {
          gradtop[i] = abs(der[i]);
        }
      }
    }
    vshift = max(vtop - vmax_, 0.0);
    log << "\nVtop = " << vtop << " Vshift = " << vshift << "\n\ngradtop = ";
    for(int i = 0; i < d_; ++i) {
      log << gradtop[i] << " "
    }
    log << "\n\n";

    // samples_.clear();
    setConv(true);
  }
}

double TTSketch::getBiasAndDerivatives(const vector<double>& cv, vector<double>& der) {
  double bias = getBias(cv);
  if(bias == 0.0) {
    return 0.0;
  }
  for(int i = 0; i < rholist_.size(); ++i) {
    double rho = densEval(i, cv);
    if(rho * lambda_ / rhomaxlist_[i] > 1.0) {
      auto deri = densGrad(i, cv);
      transform(der.begin(), der.end(), deri.begin(), der.begin(), plus<double>());
    }
  }
  return bias;
}

double TTSketch::getBias(const std::vector<double>& cv) {
  double bias = 0.0;
  for(int i = 0; i < rholist_.size(); ++i) {
    double rho = densEval(i, cv);
    double rho_adj = max(rho * lambda_ / rhomaxlist_[i], 1.0);
    bias += std::log(rho_adj);
  }
  bias -= vshift_;
  return kbt_ * (bias < 0.0 ? 0.0 : bias);
}

void TTSketch::paraSketch() {
  int N = pace_ / stride_;
  auto coeff = createTTCoeff();
  auto [M, is] = intBasisSample(siteInds(coeff));
  MPS G(d_);

  auto [Bemp, envi_L, envi_R] = formTensorMoment(M, coeff, is);
  auto links = linkInds(coeff);
  vector<ITensor> V(d_);
  G.ref(1) = Bemp(1);
  for(int core_id = 2; core_id <= d_; ++core_id) {
    MatrixXd LMat(N, rc_), RMat(N, rc_);
    for(int i = 1; i <= N; ++i) {
      for(int j = 1; j <= rc_; ++j) {
        LMat(i - 1, j - 1) = envi_L[core_id - 1].elt(is(core_id) = i, links(core_id - 1) = j);
        RMat(i - 1, j - 1) = envi_R[core_id - 2].elt(is(core_id - 1) = i, links(core_id - 1) = j);
      }
    }
    MatrixXd AMat = LMat.transpose() * RMat;
    MatrixXd PMat = AMat.completeOrthogonalDecomposition().pseudoInverse();
    ITensor A(prime(links(core_id - 1)), links(core_id - 1)), Pinv(prime(links(core_id - 1)), links(core_id - 1));
    for(int i = 1; i <= rc_; ++i) {
      for(int j = 1; j <= rc_; ++j) {
        A.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat(i - 1, j - 1));
        Pinv.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, PMat(i - 1, j - 1));
      }
    }
    G.ref(core_id) = noPrime(Pinv * Bemp(core_id));
    auto original_link_tags = tags(links(core_id - 1));
    ITensor U, S;
    V[core_id - 1] = ITensor(links(core_id - 1));
    if(r_ > 0) {
      svd(A, U, S, V[core_id - 1], {"Cutoff=", cutoff_, "RightTags=", original_link_tags, "MaxDim=", r_});
    } else {
      svd(A, U, S, V[core_id - 1], {"Cutoff=", cutoff_, "RightTags=", original_link_tags});
    }
  }
  // PrintData(linkInds(G));
  log << "Initial ranks "
  for(int i = 1; i < d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";

  G.ref(1) *= V[1];
  for(int core_id = 2; core_id < d_; ++core_id) {
    G.ref(core_id) *= V[core_id - 1];
    G.ref(core_id) *= V[core_id];
  }
  G.ref(d_) *= V[d_ - 1];
  // PrintData(linkInds(G));
  log << "Final ranks "
  for(int i = 1; i < d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";

  rholist_.push_back(G);
}

MPS TTSketch::createTTCoeff() {
  int n = basis_[0].nbasis();
  auto sites = SiteSet(d_, n);
  auto coeff = randomMPS(sites, rc_);
  for(int i = 1; i <= d_; ++i) {
    auto s = sites(i);
    auto sp = prime(s);
    vector<double> Avec(n, alpha_);
    Avec[0] = 1.0;
    auto A = diagITensor(Avec, s, sp);
    coeff.ref(i) *= A;
    coeff.ref(i).noPrime();
  }
  return coeff;
}

pair<vector<ITensor>, IndexSet> TTSketch::intBasisSample(const IndexSet& is) const {
  int N = pace_ / stride_;
  int nb = basis_[0].nbasis();
  auto sites_new = SiteSet(d_, N);
  vector<ITensor> M;
  vector<Index> is_new;
  for(int i = 1; i <= d_; ++i) {
    M.push_back(ITensor(sites_new(i), is(i)));
    is_new.push_back(sites_new(i));
    for(int j = 1; j <= N; ++j) {
      int jadj = j + samples_.size() - N;
      for(int k = 1; k <= nb; ++k) {
        // M.back().set(sites_new(i) = j, is(i) = k, pow(1.0 / N, 1.0 / d_) * basis_[i - 1](samples_[j - 1][i - 1], k));
        M.back().set(sites_new(i) = j, is(i) = k, pow(1.0 / N, 1.0 / d_) * basis_[i - 1](samples_[jadj - 1][i - 1], k));
      }
    }
  }
  return make_pair(M, IndexSet(is_new));
}

tuple<MPS, vector<ITensor>, vector<ITensor>> TTSketch::formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is) {
  int N = dim(is(1));
  auto links = linkInds(coeff);
  auto L = coeff;

  for(int i = 1; i <= d_; ++i) {
    L.ref(i) *= M[i - 1];
  }

  vector<ITensor> envi_L(d_);
  envi_L[1] = L(1) * delta(is(1), is(2));
  for(int i = 2; i < d_; ++i) {
    envi_L[i] = ITensor(is(i + 1), links(i));
    for(int j = 1; j <= N; ++j) {
      for(int k = 1; k <= rc_; ++k) {
        ITensor LHS(links(i - 1)), RHS(links(i - 1));
        for(int ii = 1; ii <= rc_; ++ii) {
          LHS.set(links(i - 1) = ii, envi_L[i - 1].elt(is(i) = j, links(i - 1) = ii));
          RHS.set(links(i - 1) = ii, L(i).elt(links(i - 1) = ii, is(i) = j, links(i) = k));
        }
        envi_L[i].set(is(i + 1) = j, links(i) = k, elt(LHS * RHS));
      }
    }
  }

  vector<ITensor> envi_R(d_);
  envi_R[d_ - 2] = L(d_) * delta(is(d_), is(d_ - 1));
  for(int i = d_ - 3; i >= 0; --i) {
    envi_R[i] = ITensor(is(i + 1), links(i + 1));
    for(int j = 1; j <= N; ++j) {
      for(int k = 1; k <= rc_; ++k) {
        ITensor LHS(links(i + 2)), RHS(links(i + 2));
        for(int ii = 1; ii <= rc_; ++ii) {
          LHS.set(links(i + 2) = ii, envi_R[i + 1].elt(is(i + 2) = j, links(i + 2) = ii));
          RHS.set(links(i + 2) = ii, L(i + 2).elt(links(i + 2) = ii, is(i + 2) = j, links(i + 1) = k));
        }
        envi_R[i].set(is(i + 1) = j, links(i + 1) = k, elt(LHS * RHS));
      }
    }
  }

  MPS B(d_);
  B.ref(1) = envi_R[0] * M[0];
  for(int core_id = 2; core_id < d_; ++core_id) {
    B.ref(core_id) = ITensor(links(core_id - 1), is(core_id), links(core_id));
    for(int i = 1; i <= rc_; ++i) {
      for(int j = 1; j <= rc_; ++j) {
        for(int k = 1; k <= N; ++k) {
          double Lelt = envi_L[core_id - 1].elt(is(core_id) = k, links(core_id - 1) = i);
          double Relt = envi_R[core_id - 1].elt(is(core_id) = k, links(core_id) = j);
          B.ref(core_id).set(links(core_id - 1) = i, is(core_id) = k, links(core_id) = j, Lelt * Relt);
        }
      }
    }
    B.ref(core_id) *= M[core_id - 1];
  }
  B.ref(d_) = envi_L[d_ - 1] * M[d_ - 1];

  return make_tuple(B, envi_L, envi_R);
}

double TTSketch::densEval(int step, const vector<double>& elements) const {
  MPS& G = rholist_[step];
  auto s = siteInds(G);
  vector<ITensor> basis_evals(d_);
  for(int i = 1; i <= d_; ++i) {
    basis_evals[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis_[i - 1](elements[i - 1], j));
    }
  }
  auto result = G(1) * basis_evals[0];
  for(int i = 2; i <= d_; ++i) {
    result *= G(i) * basis_evals[i - 1];
  }
  return elt(result);
}

vector<double> TTSketch::densGrad(int step, const vector<double>& elements) const {
  MPS& G = rholist_[step];
  auto s = siteInds(G);
  vector<double> grad(d_, 0.0);
  vector<ITensor> basis_evals(d_), basisd_evals(d_);
  for(int i = 1; i <= d_; ++i) {
    basis_evals[i - 1] = basisd_evals[i - 1] = ITensor(s(i));
    for(int j = 1; j <= dim(s(i)); ++j) {
      basis_evals[i - 1].set(s(i) = j, basis_[i - 1](elements[i - 1], j));
      basisd_evals[i - 1].set(s(i) = j, basis_[i - 1].grad(elements[i - 1], j));
    }
  }
  for(int k = 1; k <= d_; ++k) {
    auto result = G(1) * (k == 1 ? basisd_evals[0] : basis_evals[0]);
    for(int i = 2; i <= d_; ++i) {
      result *= G(i) * (k == i ? basisd_evals[i - 1] : basis_evals[i - 1]);
    }
    grad[k - 1] = elt(result);
  }
  return grad;
}

void TTSketch::setConv(bool status) {
  for(int i = 0; i < d_; ++i) {
    basis_[i].setConv(status);
  }
}

}
}
