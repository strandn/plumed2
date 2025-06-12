#include "TTHelper.h"
#include "bias/Bias.h"
#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "tools/Communicator.h"
#include "tools/File.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/NNLS>

using namespace std;
using namespace itensor;
using namespace PLMD::bias;

namespace PLMD {
namespace ttsketch {

class TTOPES : public Bias {

private:
  bool isFirstStep_;
  double kbt_;
  double biasfactor_;
  double bias_prefactor_;
  unsigned stride_;
  double epsilon_;
  bool no_Zed_;
  double Zed_;
  double threshold2_;
  bool recursive_merge_;
  vector<vector<double>> samples_;
  vector<double> heights_;
  vector<double> traj_;
  unsigned d_;
  int sketch_rc_;
  int sketch_r_;
  double sketch_cutoff_;
  int sketch_stride_;
  double sketch_alpha_;
  vector<BasisFunc> sketch_basis_;
  unsigned sketch_count_;
  MPS tt_;
  bool sketch_conv_;
  double sketch_lambda_;

  double getBiasAndDerivatives(const vector<double>& cv, vector<double>& der);
  double getBias(const vector<double>& cv);
  void paraSketch();
  MPS createTTCoeff() const;
  pair<vector<ITensor>, IndexSet> intBasisSample(const IndexSet& is) const;
  tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is);
  void solveNonNegativeLeastSquares(const Eigen::MatrixXd& Ak, const Eigen::MatrixXd& Bk, Eigen::MatrixXd& Gk);

public:
  explicit TTOPES(const ActionOptions&);
  void calculate() override;
  void update() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(TTOPES, "TTOPES")

void TTOPES::registerKeywords(Keywords& keys)
{
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory", "TEMP", "-1", "temperature. If not set, it is taken from MD engine, but not all MD codes provide it");
  keys.add("compulsory", "PACE", "the frequency for sample retrieval");
  keys.add("compulsory", "BARRIER", "the free energy barrier to be overcome. It is used to set BIASFACTOR and EPSILON to reasonable values");
  keys.add("compulsory", "COMPRESSION_THRESHOLD", "0.0", "merge samples if closer than this threshold");
  keys.add("optional", "BIASFACTOR", "the gamma bias factor used for the well-tempered target distribution.");
  keys.add("optional", "EPSILON", "the value of the regularization constant for the probability");
  keys.addFlag("RECURSIVE_MERGE_OFF", false, "do not recursively attempt sample merging when a new one is added");
  keys.addFlag("NO_ZED", false, "do not normalize over the explored CV space, Z_n=1");
  // keys.add("compulsory", "FILE", "SAMPLES", "a file in which the list of all deposited samples are stored");
  keys.add("optional", "FMT", "specify format for KERNELS file");
  // keys.add("optional", "STATE_RFILE", "read from this file the compressed samples and all the info needed to RESTART the simulation");
  // keys.add("optional", "STATE_WFILE", "write to this file the compressed samples and all the info needed to RESTART the simulation");
  // keys.add("optional", "STATE_WSTRIDE", "number of MD steps between writing the STATE_WFILE. Default is only on CPT events (but not all MD codes set them)");
  // keys.addFlag("STORE_STATES", false, "append to STATE_WFILE instead of ovewriting it each time");
  // keys.addFlag("WALKERS_MPI", false, "switch on MPI version of multiple walkers");
  // keys.addFlag("SERIAL", false, "perform calculations in serial");
  keys.use("RESTART");
  keys.add("optional", "SKETCH_RANK", "target rank for TTSketch algorithm. Compulsory if SKETCH_CUTOFF is not specified");
  keys.add("optional", "SKETCH_CUTOFF", "truncation error cutoff for singular value decomposition. Compulsory if SKETCH_RANK is not specified");
  keys.add("compulsory", "SKETCH_INITRANK", "initial rank for TTSketch algorithm");
  keys.add("compulsory", "SKETCH_PACE", "1e6", "the frequency for TT Vbias updates");
  keys.add("compulsory", "INTERVAL_MIN", "lower limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "INTERVAL_MAX", "upper limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "SKETCH_NBASIS", "20", "number of basis functions per dimension");
  keys.add("compulsory", "SKETCH_ALPHA", "0.05", "weight coefficient for random tensor train construction");
  keys.add("optional", "SKETCH_UNTIL", "after this time, the bias potential freezes");
  keys.add("optional", "SKETCH_WIDTH", "width of Gaussian kernels for smoothing");
  keys.add("optional", "KERNEL_DX", "width of basis function kernels");
  keys.add("compulsory", "SKETCH_LAMBDA", "0.1", "Ridge parameter for ridge regression");

  keys.addOutputComponent("zed", "default", "estimate of Z_n. should become flat once no new CV-space region is explored");
  keys.addOutputComponent("ns", "default", "total number of compressed samples used to represent the bias");
}

TTOPES::TTOPES(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  isFirstStep_(true),
  Zed_(1),
  d_(getNumberOfArguments()),
  sketch_r_(0),
  sketch_cutoff_(0.0),
  sketch_count_(1),
  sketch_conv_(false)
{
  this->kbt_ = getkBT();

  parse("PACE", this->stride_);

  double barrier = 0;
  parse("BARRIER", barrier);
  plumed_massert(barrier >= 0, "the BARRIER should be greater than zero");

  this->biasfactor_ = barrier / this->kbt_;
  string biasfactor_str;
  parse("BIASFACTOR", biasfactor_str);
  if(biasfactor_str == "inf" || biasfactor_str == "INF") {
    this->biasfactor_ = numeric_limits<double>::infinity();
    this->bias_prefactor_ = 1;
  } else {
    plumed_massert(this->biasfactor_ > 1, "BIASFACTOR must be greater than one (use 'inf' for uniform target)");
    this->bias_prefactor_ = 1 - 1.0 / this->biasfactor_;
  }

  this->epsilon_ = exp(-barrier / this->bias_prefactor_ / this->kbt_);
  parse("EPSILON", this->epsilon_);
  plumed_massert(this->epsilon_ > 0, "you must choose a value for EPSILON greater than zero. Is your BARRIER too high?");

  this->threshold2_ = 0.0;
  parse("COMPRESSION_THRESHOLD", this->threshold2_);
  this->threshold2_ *= this->threshold2_;

  this->no_Zed_ = false;
  parseFlag("NO_ZED", this->no_Zed_);
  bool recursive_merge_off = false;
  parseFlag("RECURSIVE_MERGE_OFF", recursive_merge_off);
  this->recursive_merge_ =! recursive_merge_off;

  parse("SKETCH_RANK", this->sketch_r_);
  parse("SKETCH_CUTOFF", this->sketch_cutoff_);
  if(this->sketch_r_ <= 0 && (this->sketch_cutoff_ <= 0.0 || this->sketch_cutoff_ > 1.0)) {
    error("Valid SKETCH_RANK or SKETCH_CUTOFF needs to be specified");
  }
  parse("SKETCH_INITRANK", this->sketch_rc_);
  if(this->sketch_rc_ <= 0) {
    error("SKETCH_INITRANK must be positive");
  }
  parse("SKETCH_PACE", this->sketch_stride_);
  if(this->sketch_stride_ <= 0) {
    error("SKETCH_PACE must be positive");
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
  parse("SKETCH_NBASIS", nbasis);
  if(nbasis <= 1) {
    error("SKETCH_NBASIS must be greater than 1");
  }
  parse("SKETCH_ALPHA", this->sketch_alpha_);
  if(this->sketch_alpha_ <= 0.0 || this->sketch_alpha_ > 1.0) {
    error("SKETCH_ALPHA must be positive and no greater than 1");
  }
  vector<double> w;
  parseVector("SKETCH_WIDTH", w);
  if(w.size() == 0) {
    w.resize(this->d_, 0.0);
  }
  if(w.size() != this->d_) {
    error("Number of arguments does not match number of SKETCH_WIDTH parameters");
  }
  for (double val : w) {
    if (val != 0.0) {
      this->sketch_conv_ = true;
    }
  }
  vector<double> dx;
  parseVector("KERNEL_DX", dx);
  if(dx.size() == 0) {
    dx.resize(this->d_, 0.0);
  }
  if(dx.size() != this->d_) {
    error("Number of arguments does not match number of KERNEL_DX parameters");
  }
  for(unsigned i = 0; i < this->d_; ++i) {
    if(this->sketch_conv_ && w[i] <= 0.0) {
      error("Gaussian smoothing requires positive WIDTH");
    }
    if(dx[i] < 0.0) {
      error("Kernel basis requires positive KERNEL_DX");
    }
    if(interval_max[i] <= interval_min[i]) {
      error("INTERVAL_MAX parameters need to be greater than respective INTERVAL_MIN parameters");
    }
    this->sketch_basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]), nbasis, w[i], true, dx[i]));
  }

  parse("SKETCH_LAMBDA", this->sketch_lambda_);
  if(this->sketch_stride_ <= 0) {
    error("SKETCH_PACE must be positive");
  }

  checkRead();

  if(getRestart()) {
    
  }

  addComponent("zed");
  componentIsNotPeriodic("zed");
  getPntrToComponent("zed")->set(this->Zed_);
  addComponent("ns");
  componentIsNotPeriodic("ns");
  getPntrToComponent("ns")->set(this->samples_.size());
}

void TTOPES::calculate() {
  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; i++) {
    cv[i] = getArgument(i);
  }

  vector<double> der(this->d_, 0.0);

  double ene = getBiasAndDerivatives(cv, der);
  setBias(ene);
  for(unsigned i = 0; i < this->d_; ++i) {
    setOutputForce(i, -der[i]);
  }
}

void TTOPES::update() {
  bool nowTT;
  if(getStep() % this->sketch_stride_== 0 && !this->isFirstStep_) {
    nowTT = true;
    for(unsigned i = 0; i < this->traj_.size(); i += this->d_ + 1) {
      vector<double> step(this->traj_.begin() + i, this->traj_.begin() + i + this->d_);
      double height = this->traj_[i + this->d_];

      if(this->recursive_merge_ && this->threshold2_ != 0.0) {
        while(true) {
          double dmin2 = numeric_limits<double>::infinity();
          unsigned idxmin = 0;
          for(unsigned j = 0; j < this->samples_.size(); ++j) {
            vector<double> di(this->d_);
            for(unsigned k = 0; k < this->d_; ++k) {
              di[k] = abs(difference(k, step[k], this->samples_[j][k]));
            }
            if(norm(di) < dmin2) {
              dmin2 = norm(di);
              idxmin = j;
            }
          }
          for(unsigned k = 0; k < this->d_; ++k) {
            step[k] = (height * step[k] + this->heights_[idxmin] * this->samples_[idxmin][k]) / (height + this->heights_[idxmin]);
          }
          height += this->heights_[idxmin];
          this->samples_.erase(this->samples_.begin() + idxmin);
          this->heights_.erase(this->heights_.begin() + idxmin);
          if(dmin2 >= this->threshold2_) {
            break;
          }
        }
      }

      this->samples_.push_back(step);
      this->heights_.push_back(height);
    }
    this->traj_.clear();
  } else {
    nowTT = false;
    this->isFirstStep_ = false;
  }

  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; ++i) {
    cv[i] = getArgument(i);
  }

  if(nowTT) {
    log << "\nForming TT-sketch density...\n";
    log.flush();
    paraSketch();

    double sum_heights = 0.0;
    for(double height : this->heights_) {
      sum_heights += height;
    }
    log << "\nEmpirical means:\n";
    Matrix<double> sigmahat(this->d_, this->d_);
    vector<double> muhat(this->d_, 0.0);
    for(unsigned k = 0; k < this->d_; ++k) {
      for(unsigned j = 0; j < this->samples_.size(); ++j) {
        muhat[k] += this->heights_[j] * this->samples_[j][k] / sum_heights;
      }
      log << muhat[k] << " ";
    }
    log << "\nEmpirical covariance matrix:\n";
    for(unsigned k = 0; k < this->d_; ++k) {
      for(unsigned l = k; l < this->d_; ++l) {
        sigmahat(k, l) = sigmahat(l, k) = 0.0;
        for(unsigned j = 0; j < this->samples_.size(); ++j) {
          sigmahat(k, l) += this->heights_[j] * (this->samples_[j][k] - muhat[k]) * (this->samples_[j][l] - muhat[l]) / sum_heights;
        }
        sigmahat(l, k) = sigmahat(k, l);
      }
    }
    matrixOut(log, sigmahat);
    auto [sigma, mu, Z] = covMat(this->tt_, this->sketch_basis_);
    log << "Estimated means:\n";
    for(unsigned k = 0; k < this->d_; ++k) {
      log << mu[k] << " ";
    }
    log << "\nEstimated covariance matrix:\n";
    matrixOut(log, sigma);
    auto diff = sigma.getVector();
    auto sigmahatv = sigmahat.getVector();
    transform(diff.begin(), diff.end(), sigmahatv.begin(), diff.begin(), minus<double>());
    log << "Relative l2 error = " << sqrt(norm(diff) / norm(sigmahatv)) << "\n";
    log.flush();

    this->tt_ /= Z;

    getPntrToComponent("ns")->set(this->samples_.size());

    if(!this->no_Zed_) {
      this->Zed_ = 0.0;
      for(auto& sample : this->samples_) {
        this->Zed_ += ttEval(this->tt_, this->sketch_basis_, sample, this->sketch_conv_) / this->samples_.size();
      }
      getPntrToComponent("zed")->set(this->Zed_);
    }

    double vpeak = 0.0;
    vector<double> gradtop(this->d_, 0.0);
    vector<double> topsample;
    vector<vector<double>> topsamples(this->d_);
    for(auto& s : this->samples_) {
      vector<double> der(this->d_, 0.0);
      double bias = getBiasAndDerivatives(s, der);
      if(bias > vpeak) {
        vpeak = bias;
        topsample = s;
      }
      for(unsigned i = 0; i < this->d_; ++i) {
        if(abs(der[i]) > gradtop[i]) {
          gradtop[i] = abs(der[i]);
          topsamples[i] = s;
        }
      }
    }
    log << "Vtop = " << vpeak << "\n";
    for(unsigned j = 0; j < this->d_; ++j) {
      log << topsample[j] << " ";
    }
    log << "\n\n";
    log.flush();

    log << "gradtop ";
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

    if(this->d_ == 2) {
      ofstream file, filep, filex, filey;
      if(this->sketch_count_ == 2) {
        file.open("F.txt");
        filep.open("P.txt");
        filex.open("dFdx.txt");
        filey.open("dFdy.txt");
      } else {
        file.open("F.txt", ios_base::app);
        filep.open("P.txt", ios_base::app);
        filex.open("dFdx.txt", std::ios_base::app);
        filey.open("dFdy.txt", std::ios_base::app);
      }
      for(int i = 0; i < 100; ++i) {
        double x = -M_PI + 2 * i * M_PI / 100;
        for(int j = 0; j < 100; ++j) {
          double y = -M_PI + 2 * j * M_PI / 100;
          vector<double> der(this->d_, 0.0);
          double ene = getBiasAndDerivatives({ x, y }, der);
          double prob = ttEval(this->tt_, this->sketch_basis_, { x, y }, this->sketch_conv_);
          file << x << " " << y << " " << ene << endl;
          filep << x << " " << y << " " << prob << endl;
          filex << x << " " << y << " " << der[0] << endl;
          filey << x << " " << y << " " << der[1] << endl;
        }
      }
      file.close();
      filep.close();
      filex.close();
      filey.close();
    }
  }

  if(getStep() % this->stride_ == 0) {
    this->traj_.insert(this->traj_.end(), cv.begin(), cv.end());
    double log_weight = getOutputQuantity(0) / this->kbt_;
    double height = exp(log_weight);
    this->traj_.insert(this->traj_.end(), height);
  }

  if(getStep() % this->sketch_stride_ == 1) {
    log << "Vbias update " << this->sketch_count_ << "...\n\n";
    log.flush();
  }
}

double TTOPES::getBiasAndDerivatives(const vector<double>& cv, vector<double>& der) {
  if(length(this->tt_) == 0) {
    return 0.0;
  }
  double prob = ttEval(this->tt_, this->sketch_basis_, cv, this->sketch_conv_);
  double bias = this->kbt_ * this->bias_prefactor_ * std::log(prob / this->Zed_ + this->epsilon_);
  vector<double> der_prob = ttGrad(this->tt_, this->sketch_basis_, cv, this->sketch_conv_);
  for(unsigned i = 0; i < this->d_; i++) {
    der[i] = this->kbt_ * this->bias_prefactor_ / (prob / this->Zed_ + this->epsilon_) * der_prob[i] / this->Zed_;
  }
  return bias;
}

double TTOPES::getBias(const vector<double>& cv) {
  if(length(this->tt_) == 0) {
    return 0.0;
  }
  double prob = ttEval(this->tt_, this->sketch_basis_, cv, this->sketch_conv_);
  return this->kbt_ * this->bias_prefactor_ * std::log(prob / this->Zed_ + this->epsilon_);
}

void TTOPES::paraSketch() {
  unsigned N = this->samples_.size();
  auto coeff = createTTCoeff();
  auto [M, is] = intBasisSample(siteInds(coeff));
  auto G = MPS(this->d_);

  auto [Bemp, envi_L, envi_R] = formTensorMoment(M, coeff, is);
  auto links = linkInds(coeff);
  vector<ITensor> U(this->d_), S(this->d_), V(this->d_);
  vector<Index> links_trimmed;
  for(unsigned core_id = 2; core_id <= this->d_; ++core_id) {
    int rank = dim(links(core_id - 1));
    Matrix<double> LMat(N, rank), RMat(N, rank);
    for(unsigned i = 1; i <= N; ++i) {
      for(int j = 1; j <= rank; ++j) {
        LMat(i - 1, j - 1) = envi_L[core_id - 1].elt(is(core_id) = i, links(core_id - 1) = j);
        RMat(i - 1, j - 1) = envi_R[core_id - 2].elt(is(core_id - 1) = i, links(core_id - 1) = j);
      }
    }
    Matrix<double> Lt, AMat, PMat;
    transpose(LMat, Lt);
    mult(Lt, RMat, AMat);

    ITensor A(prime(links(core_id - 1)), links(core_id - 1));
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= rank; ++j) {
        A.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat(i - 1, j - 1));
      }
    }
    auto original_link_tags = tags(links(core_id - 1));
    V[core_id - 1] = ITensor(links(core_id - 1));
    if(this->sketch_r_ > 0) {
      svd(A, U[core_id - 1], S[core_id - 1], V[core_id - 1],
          {"Cutoff=", this->sketch_cutoff_, "RightTags=", original_link_tags, "MaxDim=", this->sketch_r_});
    } else {
      svd(A, U[core_id - 1], S[core_id - 1], V[core_id - 1], {"Cutoff=", this->sketch_cutoff_, "RightTags=", original_link_tags});
    }
    links_trimmed.push_back(commonIndex(S[core_id - 1], V[core_id - 1]));
  }

  for(unsigned i = 1; i <= this->d_; ++i) {
    auto s = siteIndex(Bemp, i);
    ITensor ginv(s, prime(s));
    for(int j = 1; j <= dim(s); ++j) {
      for(int l = 1; l <= dim(s); ++l) {
        ginv.set(s = j, prime(s) = l, this->sketch_basis_[i - 1].ginv()(j - 1, l - 1));
      }
    }
    Bemp.ref(i) *= ginv;
    Bemp.ref(i).noPrime();
  }

  // G.ref(1) = Bemp(1) * V[1];
  Eigen::MatrixXd Ak(dim(links(1)), dim(links_trimmed[0]));
  Eigen::MatrixXd Bk(dim(links(1)), dim(siteIndex(Bemp, 1)));
  Eigen::MatrixXd Gk(dim(links_trimmed[0]), dim(siteIndex(Bemp, 1)));
  for(int i = 1; i <= dim(links(1)); ++i) {
    for(int j = 1; j <= dim(links_trimmed[0]); ++j) {
      Ak(i - 1, j - 1) = V[1].elt(links(1) = i, links_trimmed[0] = j);
    }
  }
  for(int i = 1; i <= dim(links(1)); ++i) {
    for(int j = 1; j <= dim(siteIndex(Bemp, 1)); ++j) {
      Bk(i - 1, j - 1) = Bemp(1).elt(links(1) = i, siteIndex(Bemp, 1) = j);
    }
  }
  solveNonNegativeLeastSquares(Ak, Bk, Gk);
  // cout << "core 1" << endl;
  // cout << "AG" << endl << Ak * Gk << endl;
  // cout << "B" << endl << Bk << endl;
  // Eigen::MatrixXd AGB = Ak * Gk - Bk;
  // cout << "AG-B" << endl << AGB << endl;
  // cout << "|AG-B|/|b| = " << AGB.norm() / Bk.norm() << endl;
  G.ref(1) = ITensor(links_trimmed[0], siteIndex(Bemp, 1));
  for(int i = 1; i <= dim(links_trimmed[0]); ++i) {
    for(int j = 1; j <= dim(siteIndex(Bemp, 1)); ++j) {
      G.ref(1).set(links_trimmed[0] = i, siteIndex(Bemp, 1) = j, Gk(i - 1, j - 1));
    }
  }

  for(unsigned core_id = 2; core_id < this->d_; ++core_id) {
    int rank = dim(links(core_id - 1)), rank_trimmed = dim(links_trimmed[core_id - 2]);
    ITensor A = U[core_id - 1] * S[core_id - 1];
    // ITensor Pinv(links_trimmed[core_id - 2], links(core_id - 1));
    // Matrix<double> AMat(rank, rank_trimmed), PMat;
    // for(int i = 1; i <= rank; ++i) {
    //   for(int j = 1; j <= rank_trimmed; ++j) {
    //     AMat(i - 1, j - 1) = A.elt(prime(links(core_id - 1)) = i, links_trimmed[core_id - 2] = j);
    //   }
    // }
    // pseudoInvert(AMat, PMat);
    // for(int i = 1; i <= rank_trimmed; ++i) {
    //   for(int j = 1; j <= rank; ++j) {
    //     Pinv.set(links_trimmed[core_id - 2] = i, links(core_id - 1) = j, PMat(i - 1, j - 1));
    //   }
    // }
    // G.ref(core_id) = Pinv * Bemp(core_id);
    // if(core_id != this->d_) {
    //   G.ref(core_id) *= V[core_id];
    // }
    auto [C, c] = combiner(links_trimmed[core_id - 1], siteIndex(Bemp, core_id));
    ITensor B = Bemp(core_id) * V[core_id] * C;
    Ak = Eigen::MatrixXd(rank, rank_trimmed);
    Bk = Eigen::MatrixXd(rank, dim(c));
    Gk = Eigen::MatrixXd(rank_trimmed, dim(c));
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= rank_trimmed; ++j) {
        Ak(i - 1, j - 1) = A.elt(prime(links(core_id - 1)) = i, links_trimmed[core_id - 2] = j);
      }
    }
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= dim(c); ++j) {
        Bk(i - 1, j - 1) = B.elt(links(core_id - 1) = i, c = j);
      }
    }
    solveNonNegativeLeastSquares(Ak, Bk, Gk);
    G.ref(core_id) = ITensor(links_trimmed[core_id - 2], c);
    for(int i = 1; i <= rank_trimmed; ++i) {
      for(int j = 1; j <= dim(c); ++j) {
        G.ref(core_id).set(links_trimmed[core_id - 2] = i, c = j, Gk(i - 1, j - 1));
      }
    }
    G.ref(core_id) *= C;
  }

  ITensor A = U[this->d_ - 1] * S[this->d_ - 1];
  Ak = Eigen::MatrixXd(dim(links(this->d_ - 1)), dim(links_trimmed[this->d_ - 2]));
  Bk = Eigen::MatrixXd(dim(links(this->d_ - 1)), dim(siteIndex(Bemp, this->d_)));
  Gk = Eigen::MatrixXd(dim(links_trimmed[this->d_ - 2]), dim(siteIndex(Bemp, this->d_)));
  for(int i = 1; i <= dim(links(this->d_ - 1)); ++i) {
    for(int j = 1; j <= dim(links_trimmed[this->d_ - 2]); ++j) {
      Ak(i - 1, j - 1) = A.elt(prime(links(this->d_ - 1)) = i, links_trimmed[this->d_ - 2] = j);
    }
  }
  for(int i = 1; i <= dim(links(this->d_ - 1)); ++i) {
    for(int j = 1; j <= dim(siteIndex(Bemp, this->d_)); ++j) {
      Bk(i - 1, j - 1) = Bemp(this->d_).elt(links(this->d_ - 1) = i, siteIndex(Bemp, this->d_) = j);
    }
  }
  solveNonNegativeLeastSquares(Ak, Bk, Gk);
  // cout << "core 2" << endl;
  // cout << "AG" << endl << Ak * Gk << endl;
  // cout << "B" << endl << Bk << endl;
  // AGB = Ak * Gk - Bk;
  // cout << "AG-B" << endl << AGB << endl;
  // cout << "|AG-B|/|b| = " << AGB.norm() / Bk.norm() << endl;
  G.ref(this->d_) = ITensor(links_trimmed[this->d_ - 2], siteIndex(Bemp, this->d_));
  for(int i = 1; i <= dim(links_trimmed[this->d_ - 2]); ++i) {
    for(int j = 1; j <= dim(siteIndex(Bemp, this->d_)); ++j) {
      G.ref(this->d_).set(links_trimmed[this->d_ - 2] = i, siteIndex(Bemp, this->d_) = j, Gk(i - 1, j - 1));
    }
  }

  log << "Final ranks ";
  for(unsigned i = 1; i < this->d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";
  log.flush();

  this->tt_ = G;
  ++this->sketch_count_;
}

MPS TTOPES::createTTCoeff() const {
  default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
  normal_distribution<double> distribution(0.0, 1.0);
  int n = this->sketch_basis_[0].nbasis();
  auto sites = SiteSet(this->d_, n);
  auto coeff = MPS(sites, this->sketch_rc_);
  for(int j = 1; j <= n; ++j) {
    for(int k = 1; k <= this->sketch_rc_; ++k) {
      coeff.ref(1).set(sites(1) = j, linkIndex(coeff, 1) = k, distribution(generator));
    }
  }
  for(unsigned i = 2; i <= this->d_ - 1; ++i) {
    for(int j = 1; j <= n; ++j) {
      for(int k = 1; k <= this->sketch_rc_; ++k) {
        for(int l = 1; l <= this->sketch_rc_; ++l) {
          coeff.ref(i).set(sites(i) = j, linkIndex(coeff, i - 1) = k, linkIndex(coeff, i) = l, distribution(generator));
        }
      }
    }
  }
  for(int j = 1; j <= n; ++j) {
    for(int k = 1; k <= this->sketch_rc_; ++k) {
      coeff.ref(this->d_).set(sites(this->d_) = j, linkIndex(coeff, this->d_ - 1) = k, distribution(generator));
    }
  }
  for(unsigned i = 1; i <= this->d_; ++i) {
    auto s = sites(i);
    auto sp = prime(s);
    vector<double> Avec(n, this->sketch_alpha_);
    Avec[0] = 1.0;
    auto A = diagITensor(Avec, s, sp);
    coeff.ref(i) *= A;
    coeff.ref(i).noPrime();
  }
  return coeff;
}

pair<vector<ITensor>, IndexSet> TTOPES::intBasisSample(const IndexSet& is) const {
  unsigned N = this->samples_.size();
  double sum_heights = 0.0;
  for(double height : this->heights_) {
    sum_heights += height;
  }
  int nb = this->sketch_basis_[0].nbasis();
  auto sites_new = SiteSet(this->d_, N);
  vector<ITensor> M;
  vector<Index> is_new;
  for(unsigned i = 1; i <= this->d_; ++i) {
    M.push_back(ITensor(sites_new(i), is(i)));
    is_new.push_back(sites_new(i));
    for(unsigned j = 1; j <= N; ++j) {
      double x = this->samples_[j - 1][i - 1];
      double h = pow(this->heights_[j - 1] / sum_heights, 1.0 / this->d_);
      for(int pos = 1; pos <= nb; ++pos) {
        M.back().set(sites_new(i) = j, is(i) = pos, h * this->sketch_basis_[i - 1](x, pos, false));
      }
    }
  }
  return make_pair(M, IndexSet(is_new));
}

tuple<MPS, vector<ITensor>, vector<ITensor>> TTOPES::formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is) {
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

void TTOPES::solveNonNegativeLeastSquares(const Eigen::MatrixXd& Ak, const Eigen::MatrixXd& Bk, Eigen::MatrixXd& Gk) {
  const unsigned nrows = Ak.rows();  // rows of Ak
  const unsigned ncols = Ak.cols();  // cols of Ak
  const unsigned nrhs  = Bk.cols();  // cols of Bk (number of right-hand sides)

  if(Ak.rows() != Bk.rows()) {
    error("Ak and Bk must have the same number of rows (" + to_string(Ak.rows()) + " vs " + to_string(Bk.rows()) + ")");
  }

  if(Gk.rows() != ncols) {
    error("Gk has the incorrect number of rows (got " + to_string(Gk.rows()) + ", expected " + to_string(ncols) + ")");
  }
  if(Gk.cols() != nrhs) {
    error("Gk has the incorrect number of columns (got " + to_string(Gk.cols()) + ", expected " + to_string(nrhs) + ")");
  }

  cout << "Ak" << endl << Ak << endl;

  // Loop over each column in Bk
  for (unsigned j = 0; j < nrhs; ++j) {
    Eigen::VectorXd b(nrows);
    for (unsigned i = 0; i < nrows; ++i) {
      b(i) = Bk(i, j);
    }

    Eigen::MatrixXd A_aug;
    Eigen::VectorXd b_aug;

    if (this->sketch_lambda_ > 0.0) {
      // Apply Tikhonov regularization: [Ak; sqrt(lambda)*I], [b; 0]
      A_aug.resize(nrows + ncols, ncols);
      A_aug.topRows(nrows) = Ak;
      A_aug.bottomRows(ncols) = std::sqrt(this->sketch_lambda_) * Eigen::MatrixXd::Identity(ncols, ncols);

      b_aug.resize(nrows + ncols);
      b_aug.head(nrows) = b;
      b_aug.tail(ncols).setZero();
    } else {
      A_aug = Ak;
      b_aug = b;
    }

    // Solve NNLS
    Eigen::NNLS<Eigen::MatrixXd> nnls(A_aug);
    cout << "b" << endl << b_aug.transpose() << endl;
    cout << "||A_aug|| = " << A_aug.norm() << ", ||b_aug|| = " << b_aug.norm() << endl;
    const Eigen::VectorXd& g = nnls.solve(b_aug);

    if (nnls.info() != Eigen::Success) {
      error("NNLS failed for column " + to_string(j));
    }

    for (unsigned i = 0; i < ncols; ++i) {
      Gk(i, j) = g(i);
    }

    cout << j << " tolerance " << nnls.tolerance() << " iterations " << nnls.iterations() << " maxIterations " << nnls.maxIterations();
    cout << " info " << nnls.info() << endl;
    cout << "g" << endl << g.transpose() << endl;
    Eigen::VectorXd agb = A_aug * g - b_aug;
    cout << "|Ax-b|/|b| = " << agb.norm() / b.norm() << endl;
  }
}

}
}
