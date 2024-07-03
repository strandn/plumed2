#include "BasisFunc.h"
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
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
  int count_;
  double alpha_;
  double lambda_;

public:
  explicit TTSketch(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate();
  void paraSketch();
  MPS createTTCoeff() const;
  pair<vector<ITensor>, IndexSet> intBasisSample(IndexSet const& is) const;
  static tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(vector<ITensor> const& M, MPS const& coeff, IndexSet const& is);
  double densEval(int step, vector<double> const& elements) const;
  vector<double> densGrad(int step, vector<double> const& elements) const;
};

PLUMED_REGISTER_ACTION(TTSketch, "TTSKETCH")

TTSketch::TTSketch(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  r_(0),
  cutoff(0.0),
  kbt_(0.0),
  vmax_(numeric_limits<double>::max()),
  vshift_(0.0),
  count_(0)
{
  bool conv = true;
  parseFlag("CONV", conv);
  bool walkers_mpi = false;
  parseFlag("WALKERS_MPI", walkers_mpi);
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
  keys.addFlag("WALKERS_MPI",false,"To be used when gromacs + multiple walkers are used");
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
  keys.add("compulsory", "INTERVAL_MAX", "Upper limits, outside the limits the system will not feel the biasing force.");
  keys.add("compulsory", "NBASIS", "20", "Number of Fourier basis functions per dimension");
  keys.add("compulsory", "ALPHA", "0.05", "Weight coefficient for random tensor train construction");
  keys.add("compulsory", "LAMBDA", "100.0", "Ratio of largest to smallest allowed density magnitudes");
  // keys.add("compulsory", "FILE", "name of the file where tensor trains and related data are stored");
}

void TTSketch::calculate() {

}

void TTSketch::paraSketch() {

}

MPS TTSketch::createTTCoeff() {
  
}

pair<vector<ITensor>, IndexSet> TTSketch::intBasisSample(IndexSet const& is) const {

}

static tuple<MPS, vector<ITensor>, vector<ITensor>> TTSketch::formTensorMoment(vector<ITensor> const& M,
                                                                               MPS const& coeff,
                                                                               IndexSet const& is) {
  
}

double TTSketch::densEval(int step, vector<double> const& elements) const {

}

vector<double> TTSketch::densGrad(int step, vector<double> const& elements) const {

}

}
}
