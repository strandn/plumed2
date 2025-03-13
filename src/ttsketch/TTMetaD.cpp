#include "TTHelper.h"
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Communicator.h"
#include "tools/File.h"
#include "tools/OpenMP.h"

using namespace std;
using namespace itensor;
using namespace PLMD::bias;

namespace PLMD {
namespace ttsketch {

class TTMetaD : public Bias {

private:
  struct Gaussian {
    bool multivariate;
    double height;
    vector<double> center;
    vector<double> sigma;
    vector<double> invsigma;
    Gaussian(const bool m, const double h, const vector<double>& c, const vector<double>& s)
      : multivariate(m), height(h), center(c), sigma(s), invsigma(s) {
      for(unsigned i = 0; i < invsigma.size(); ++i) {
        if(abs(invsigma[i]) > 1.e-20) {
          invsigma[i] = 1.0 / invsigma[i];
        } else {
          invsigma[i] = 0.0;
        }
      }
    }
  };
  double kbt_;
  int stride_;
  bool welltemp_;
  double biasf_;
  string fmt_;
  bool isFirstStep_;
  double height0_;
  vector<double> sigma0_;
  vector<Gaussian> hills_;
  // OFile hillsOfile_;
  // vector<unique_ptr<IFile>> ifiles_;
  // vector<string> ifilesnames_;
  bool walkers_mpi_;
  int mpi_size_;
  int mpi_rank_;
  unsigned d_;
  int sketch_rc_;
  int sketch_r_;
  double sketch_cutoff_;
  int sketch_stride_;
  double sketch_alpha_;
  vector<BasisFunc> sketch_basis_;
  unsigned sketch_count_;
  MPS vb_;
  // double vb_cutoff_;
  // double vb_rank_;
  double sketch_until_;
  bool frozen_;

  // void readGaussians(IFile *ifile);
  // void writeGaussian(const Gaussian& hill, OFile& file);
  double getHeight(const vector<double>& cv);
  double getBias(const vector<double>& cv);
  double getBiasAndDerivatives(const vector<double>& cv, vector<double>& der);
  double evaluateGaussian(const vector<double>& cv, const Gaussian& hill);
  double evaluateGaussianAndDerivatives(const vector<double>& cv, const Gaussian& hill, vector<double>& der, vector<double>& dp);
  // bool scanOneHill(IFile* ifile, vector<Value>& tmpvalues, vector<double>& center, vector<double>& sigma, double& height, bool& multivariate);
  void paraSketch();
  MPS createTTCoeff() const;
  pair<vector<ITensor>, IndexSet> intBasisSample(const IndexSet& is) const;
  tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is);
  tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMomentVb(const MPS& coeff);

public:
  explicit TTMetaD(const ActionOptions&);
  void calculate() override;
  void update() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(TTMetaD, "TTMETAD")

void TTMetaD::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory", "SIGMA", "the widths of the Gaussian hills");
  keys.add("compulsory", "PACE", "the frequency for hill addition");
  // keys.add("compulsory", "FILE", "HILLS", "a file in which the list of added hills is stored");
  keys.add("compulsory", "HEIGHT", "the heights of the Gaussian hills");
  keys.add("optional", "FMT", "specify format for HILLS files (useful for decrease the number of digits in regtests)");
  keys.add("optional", "BIASFACTOR", "use well tempered metadynamics and use this bias factor. Please note you must also specify temp");
  keys.add("optional", "TEMP", "the system temperature - this is only needed if you are doing well-tempered metadynamics");
  keys.addFlag("WALKERS_MPI", false, "To be used when gromacs + multiple walkers are used");
  keys.use("RESTART");
  keys.add("optional", "SKETCH_RANK", "Target rank for TTSketch algorithm - compulsory if CUTOFF is not specified");
  keys.add("optional", "SKETCH_CUTOFF", "Truncation error cutoff for singular value decomposition - compulsory if RANK is not specified");
  keys.add("compulsory", "SKETCH_INITRANK", "Initial rank for TTSketch algorithm");
  keys.add("compulsory", "SKETCH_PACE", "1e6", "The frequency for TT Vbias updates");
  keys.add("compulsory", "INTERVAL_MIN", "Lower limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "INTERVAL_MAX", "Upper limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "SKETCH_NBASIS", "20", "Number of basis functions per dimension");
  keys.add("compulsory", "SKETCH_ALPHA", "0.05", "Weight coefficient for random tensor train construction");
  // keys.add("optional", "VB_CUTOFF", "Convergence threshold for TT Vbias");
  // keys.add("optional", "VB_RANK", "Largest possible rank for TT Vbias");
  keys.add("optional", "SKETCH_UNTIL", "After this time, the bias potential freezes");
}

TTMetaD::TTMetaD(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  kbt_(0.0),
  stride_(0),
  welltemp_(false),
  biasf_(-1.0),
  isFirstStep_(true),
  height0_(numeric_limits<double>::max()),
  walkers_mpi_(false),
  mpi_size_(0),
  mpi_rank_(0),
  sketch_r_(0),
  sketch_cutoff_(0.0),
  sketch_count_(1),
  // vb_cutoff_(0.0),
  // vb_rank_(0),
  sketch_until_(numeric_limits<double>::max()),
  frozen_(false)
{
  this->d_ = getNumberOfArguments();
  if(this->d_ < 2) {
    error("Number of arguments must be at least 2");
  }
  parse("FMT", this->fmt_);
  parseVector("SIGMA", this->sigma0_);
  if(this->sigma0_.size() != d_) {
    error("number of arguments does not match number of SIGMA parameters");
  }
  parse("HEIGHT", this->height0_);
  parse("PACE", this->stride_);
  if(stride_ <= 0) {
    error("frequency for hill addition is nonsensical");
  }
  // string hillsfname = "HILLS";
  // parse("FILE", hillsfname);
  parse("BIASFACTOR", this->biasf_);
  if(this->biasf_ < 1.0 && this->biasf_ != -1.0) {
    error("well tempered bias factor is nonsensical");
  }
  this->kbt_ = getkBT();
  if(this->biasf_ >= 1.0) {
    if(this->kbt_ == 0.0) {
      error("Unless the MD engine passes the temperature to plumed, with well-tempered metad you must specify it using TEMP");
    }
    this->welltemp_ = true;
  }

  parseFlag("WALKERS_MPI", this->walkers_mpi_);
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
  if(nbasis % 2 == 0) {
    ++nbasis;
  }
  parse("SKETCH_ALPHA", this->sketch_alpha_);
  if(this->sketch_alpha_ <= 0.0 || this->sketch_alpha_ > 1.0) {
    error("SKETCH_ALPHA must be positive and no greater than 1");
  }
  for(unsigned i = 0; i < this->d_; ++i) {
    if(interval_max[i] <= interval_min[i]) {
      error("INTERVAL_MAX parameters need to be greater than respective INTERVAL_MIN parameters");
    }
    this->sketch_basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]), nbasis, false, 0.0, false));
  }
  if(this->walkers_mpi_) {
    this->mpi_size_ = multi_sim_comm.Get_size();
    this->mpi_rank_ = multi_sim_comm.Get_rank();
  }
  // parse("VB_CUTOFF", this->vb_cutoff_);
  // if(this->vb_cutoff_ < 0.0 || this->vb_cutoff_ >= 1.0) {
  //   error("VB_CUTOFF must be nonnegative and less than 1");
  // }
  // parse("VB_RANK", this->vb_rank_);
  // if(this->vb_rank_ < 0) {
  //   error("VB_RANK must be nonnegative");
  // }

  parse("SKETCH_UNTIL", this->sketch_until_);

  if(getRestart()) {

  }
}

void TTMetaD::calculate() {
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

void TTMetaD::update() {
  bool nowAddAHill;
  if(getStep() % this->stride_ == 0 && !isFirstStep_ && !this->frozen_) {
    nowAddAHill = true;
  } else {
    nowAddAHill = false;
  }

  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; ++i) {
    cv[i] = getArgument(i);
  }

  if(nowAddAHill) {
    double height = getHeight(cv);

    if(walkers_mpi_) {
      vector<double> all_cv(this->mpi_size_ * this->d_, 0.0);
      vector<double> all_sigma(this->mpi_size_ * this->sigma0_.size(), 0.0);
      vector<double> all_height(this->mpi_size_, 0.0);
      multi_sim_comm.Allgather(cv, all_cv);
      multi_sim_comm.Allgather(this->sigma0_, all_sigma);
      multi_sim_comm.Allgather(height * (this->biasf_ > 1.0 ? this->biasf_ / (this->biasf_ - 1.0) : 1.0), all_height);

      for(unsigned i = 0; i < this->mpi_size_; i++) {
        vector<double> cv_now(this->d_);
        vector<double> sigma_now(this->sigma0_.size());
        for(unsigned j = 0; j < this->d_; j++) {
          cv_now[j] = all_cv[i * this->d_ + j];
        }
        for(unsigned j = 0; j < this->sigma0_.size(); j++) {
          sigma_now[j] = all_sigma[i * this->sigma0_.size() + j];
        }
        double fact = (this->biasf_ > 1.0 ? (this->biasf_ - 1.0) / this->biasf_ : 1.0);
        Gaussian newhill = Gaussian(false, all_height[i] * fact, cv_now, sigma_now);
        this->hills_.push_back(newhill);
      }
    } else {
      Gaussian newhill = Gaussian(false, height, cv, this->sigma0_);
      this->hills_.push_back(newhill);
    }
  }

  bool nowAddATT;
  if(getStep() % this->sketch_stride_ == 0 && !this->isFirstStep_ && !this->frozen_) {
    nowAddATT = true;
  } else {
    nowAddATT = false;
    this->isFirstStep_ = false;
  }

  if(nowAddATT) {
    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      unsigned N = this->hills_.size();
      log << "Sample limits\n";
      for(unsigned i = 0; i < this->d_; ++i) {
        auto [large, small] = this->sketch_basis_[i].dom();
        for(unsigned j = 0; j < N; ++j) {
          if(this->hills_[j].center[i] > large) {
            large = this->hills_[j].center[i];
          }
          if(this->hills_[j].center[i] < small) {
            small = this->hills_[j].center[i];
          }
        }
        log << small << " " << large << "\n";
      }

      log << "\nEmpirical means:\n";
      Matrix<double> sigmahat(this->d_, this->d_);
      vector<double> muhat(this->d_, 0.0);
      for(unsigned k = 0; k < this->d_; ++k) {
        for(unsigned j = 0; j < N; ++j) {
          muhat[k] += this->hills_[j].center[k] / N;
        }
        log << muhat[k] << " ";
      }
      log << "\nEmpirical covariance matrix:\n";
      for(unsigned k = 0; k < this->d_; ++k) {
        for(unsigned l = k; l < this->d_; ++l) {
          sigmahat(k, l) = sigmahat(l, k) = 0.0;
          for(unsigned j = 0; j < N; ++j) {
            sigmahat(k, l) += (this->hills_[j].center[k] - muhat[k]) * (this->hills_[j].center[l] - muhat[l]) / (N - 1);
          }
          sigmahat(l, k) = sigmahat(k, l);
        }
      }
      matrixOut(log, sigmahat);

      // unsigned N = this->sketch_stride_ * this->mpi_size_ / this->stride_;
      // log << "Sample limits\n";
      // for(unsigned i = 0; i < this->d_; ++i) {
      //   auto [large, small] = this->sketch_basis_[i].dom();
      //   for(unsigned j = 0; j < N; ++j) {
      //     int jadj = this->hills_.size() - N + j;
      //     if(this->hills_[jadj].center[i] > large) {
      //       large = this->hills_[jadj].center[i];
      //     }
      //     if(this->hills_[jadj].center[i] < small) {
      //       small = this->hills_[jadj].center[i];
      //     }
      //   }
      //   log << small << " " << large << "\n";
      // }

      // log << "\nEmpirical means:\n";
      // Matrix<double> sigmahat(this->d_, this->d_);
      // vector<double> muhat(this->d_, 0.0);
      // for(unsigned k = 0; k < this->d_; ++k) {
      //   for(unsigned j = 0; j < N; ++j) {
      //     int jadj = this->hills_.size() - N + j;
      //     muhat[k] += this->hills_[jadj].center[k] / N;
      //   }
      //   log << muhat[k] << " ";
      // }
      // log << "\nEmpirical covariance matrix:\n";
      // for(unsigned k = 0; k < this->d_; ++k) {
      //   for(unsigned l = k; l < this->d_; ++l) {
      //     sigmahat(k, l) = sigmahat(l, k) = 0.0;
      //     for(unsigned j = 0; j < N; ++j) {
      //       int jadj = this->hills_.size() - N + j;
      //       sigmahat(k, l) += (this->hills_[jadj].center[k] - muhat[k]) * (this->hills_[jadj].center[l] - muhat[l]) / (N - 1);
      //     }
      //     sigmahat(l, k) = sigmahat(k, l);
      //   }
      // }
      // matrixOut(log, sigmahat);

      vector<double> A0(N);
      vector<vector<double>> x(N);
      for(unsigned i = 0; i < N; ++i) {
        x[i] = this->hills_[i].center;
        A0[i] = getBias(x[i]);
      }
      
      if(this->d_ == 3) {
        ofstream file;
        if(this->sketch_count_ == 2) {
          file.open("phi2phi3_0_0.txt");
        } else {
          file.open("phi2phi3_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, -1.2 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi3_1_0.txt");
        } else {
          file.open("phi2phi3_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, 1.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi4_0_0.txt");
        } else {
          file.open("phi2phi4_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, -1.2, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi4_1_0.txt");
        } else {
          file.open("phi2phi4_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, 1.0, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3phi4_0_0.txt");
        } else {
          file.open("phi3phi4_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, x, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3phi4_1_0.txt");
        } else {
          file.open("phi3phi4_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, x, y }) << endl;
          }
        }
        file.close();
      }

      if(this->d_ == 6) {
        ofstream file;
        if(this->sketch_count_ == 2) {
          file.open("phi2psi2_0_0.txt");
        } else {
          file.open("phi2psi2_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, -1.2, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2psi2_1_0.txt");
        } else {
          file.open("phi2psi2_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, 1.0, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi2phi3_0_0.txt");
        } else {
          file.open("psi2phi3_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, x, y, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi2phi3_1_0.txt");
        } else {
          file.open("psi2phi3_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, x, y, 0.0, 1.0, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3psi3_0_0.txt");
        } else {
          file.open("phi3psi3_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, x, y, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3psi3_1_0.txt");
        } else {
          file.open("phi3psi3_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, x, y, 1.0, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi3phi4_0_0.txt");
        } else {
          file.open("psi3phi4_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, x, y, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi3phi4_1_0.txt");
        } else {
          file.open("psi3phi4_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, x, y, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi4psi4_0_0.txt");
        } else {
          file.open("phi4psi4_0_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, 0.0, x, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi4psi4_1_0.txt");
        } else {
          file.open("phi4psi4_1_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, 0.0, x, y }) << endl;
          }
        }
        file.close();
      }

      log << "\nStarting TT-sketch...\n";
      log.flush();
      paraSketch();
      ++this->sketch_count_;

      this->hills_.clear();

      vector<double> diff(N);
      for(unsigned i = 0; i < N; ++i) {
        diff[i] = getBias(x[i]);
      }
      transform(diff.begin(), diff.end(), A0.begin(), diff.begin(), minus<double>());
      log << "Relative l2 error = " << sqrt(norm(diff) / norm(A0)) << "\n\n";
      log.flush();

      string ttfilename = "ttsketch.h5";
      if(this->walkers_mpi_) {
        ttfilename = "../" + ttfilename;
      }
      ttWrite(ttfilename, this->vb_, this->sketch_count_);
      
      if(this->d_ == 3) {
        ofstream file;
        if(this->sketch_count_ == 2) {
          file.open("phi2phi3_0.txt");
        } else {
          file.open("phi2phi3_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, -1.2 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi3_1.txt");
        } else {
          file.open("phi2phi3_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, 1.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi4_0.txt");
        } else {
          file.open("phi2phi4_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, -1.2, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2phi4_1.txt");
        } else {
          file.open("phi2phi4_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, 1.0, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3phi4_0.txt");
        } else {
          file.open("phi3phi4_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, x, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3phi4_1.txt");
        } else {
          file.open("phi3phi4_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, x, y }) << endl;
          }
        }
        file.close();
      }

      if(this->d_ == 6) {
        ofstream file;
        if(this->sketch_count_ == 2) {
          file.open("phi2psi2_0.txt");
        } else {
          file.open("phi2psi2_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, -1.2, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi2psi2_1.txt");
        } else {
          file.open("phi2psi2_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ x, y, 1.0, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi2phi3_0.txt");
        } else {
          file.open("psi2phi3_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, x, y, 0.0, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi2phi3_1.txt");
        } else {
          file.open("psi2phi3_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, x, y, 0.0, 1.0, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3psi3_0.txt");
        } else {
          file.open("phi3psi3_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, x, y, -1.2, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi3psi3_1.txt");
        } else {
          file.open("phi3psi3_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, x, y, 1.0, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi3phi4_0.txt");
        } else {
          file.open("psi3phi4_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, x, y, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("psi3phi4_1.txt");
        } else {
          file.open("psi3phi4_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, x, y, 0.0 }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi4psi4_0.txt");
        } else {
          file.open("phi4psi4_0.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, 0.0, x, y }) << endl;
          }
        }
        file.close();
        if(this->sketch_count_ == 2) {
          file.open("phi4psi4_1.txt");
        } else {
          file.open("phi4psi4_1.txt", ios_base::app);
        }
        for(int i = 0; i < 100; ++i) {
          double x = -M_PI + 2 * i * M_PI / 100;
          for(int j = 0; j < 100; ++j) {
            double y = -M_PI + 2 * j * M_PI / 100;
            file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, 0.0, x, y }) << endl;
          }
        }
        file.close();
      }
    }

    if(this->walkers_mpi_) {
      multi_sim_comm.Bcast(this->sketch_count_, 0);
      if(this->mpi_rank_ != 0) {
        this->hills_.clear();
        this->vb_ = ttRead("../ttsketch.h5", this->sketch_count_);
      }
    }
    if(getTime() >= this->sketch_until_) {
      this->frozen_ = true;
    }
  }
  if(getStep() % this->sketch_stride_ == 1 && !this->frozen_) {
    log << "Vbias update " << this->sketch_count_ << "...\n\n";
    log.flush();
  }
}

double TTMetaD::getHeight(const vector<double>& cv) {
  double height = this->height0_;
  if(this->welltemp_) {
    double vbias = getBias(cv);
    if(this->biasf_ > 1.0) {
      height = this->height0_ * exp(-vbias / (this->kbt_ * (this->biasf_ - 1.0)));
    } else {
      height = this->height0_ * exp(-vbias / this->kbt_);
    }
  }
  return height;
}

double TTMetaD::getBias(const vector<double>& cv) {
  double bias = length(this->vb_) == 0 ? 0.0 : ttEval(this->vb_, this->sketch_basis_, cv, false);
  unsigned nt = OpenMP::getNumThreads();
  #pragma omp parallel num_threads(nt)
  {
    #pragma omp for reduction(+:bias) nowait
    for(unsigned i = 0; i < hills_.size(); ++i) {
      bias += evaluateGaussian(cv, this->hills_[i]);
    }
  }
  return bias;
}

double TTMetaD::getBiasAndDerivatives(const vector<double>& cv, vector<double>& der) {
  double bias = length(this->vb_) == 0 ? 0.0 : ttEval(this->vb_, this->sketch_basis_, cv, false);
  if(length(this->vb_) != 0) {
    der = ttGrad(this->vb_, this->sketch_basis_, cv, false);
  }
  unsigned nt = OpenMP::getNumThreads();
  if(this->hills_.size() < 2 * nt || nt == 1) {
    vector<double> dp(this->d_);
    for(unsigned i = 0; i < this->hills_.size(); ++i) {
      bias += evaluateGaussianAndDerivatives(cv, this->hills_[i], der, dp);
    }
  } else {
    #pragma omp parallel num_threads(nt)
    {
      vector<double> omp_deriv(this->d_, 0.0);
      vector<double> dp(this->d_);
      #pragma omp for reduction(+:bias) nowait
      for(unsigned i = 0; i < this->hills_.size(); ++i) {
        bias += evaluateGaussianAndDerivatives(cv, this->hills_[i], omp_deriv, dp);
      }
      #pragma omp critical
      for(unsigned i = 0; i < this->d_; ++i) {
        der[i] += omp_deriv[i];
      }
    }
  }
  return bias;
}

double TTMetaD::evaluateGaussian(const vector<double>& cv, const Gaussian& hill) {
  double dp2 = 0.0;
  for(unsigned i = 0; i < this->d_; i++) {
    double dp = difference(i, hill.center[i], cv[i]) * hill.invsigma[i];
    dp2 += dp * dp;
  }
  dp2 *= 0.5;

  double bias = 0.0;
  if(dp2 < dp2cutoff) {
    bias = hill.height * exp(-dp2);
  }

  return bias;
}

double TTMetaD::evaluateGaussianAndDerivatives(const vector<double>& cv, const Gaussian& hill, vector<double>& der, vector<double>& dp) {
  double dp2 = 0.0;
  double bias = 0.0;
  for(unsigned i = 0; i < this->d_; i++) {
    dp[i] = difference(i, hill.center[i], cv[i]) * hill.invsigma[i];
    dp2 += dp[i] * dp[i];
  }
  dp2 *= 0.5;
  if(dp2 < dp2cutoff) {
    bias = hill.height * exp(-dp2);
    for(unsigned i = 0; i < this->d_; i++) {
      der[i] -= bias * dp[i] * hill.invsigma[i];
    }
  }

  return bias;
}

void TTMetaD::paraSketch() {
  unsigned N = this->hills_.size();
  auto coeff = createTTCoeff();
  auto [M, is] = intBasisSample(siteInds(coeff));
  MPS G(this->d_);

  auto [Bemp, envi_L, envi_R] = formTensorMoment(M, coeff, is);
  MPS Bemp_Vb;
  vector<ITensor> envi_L_Vb;
  vector<ITensor> envi_R_Vb;
  if(this->sketch_count_ != 1) {
    auto vbresult = formTensorMomentVb(coeff);
    Bemp_Vb = get<0>(vbresult);
    envi_L_Vb = get<1>(vbresult);
    envi_R_Vb = get<2>(vbresult);
    PrintData(Bemp);
    PrintData(Bemp_Vb);
    for(unsigned i = 1; i <= this->d_; ++i) {
      Bemp.ref(i) += Bemp_Vb(i);
      cout << "Bemp " << i << endl;
      PrintData(Bemp(i));
    }
  }
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
    Matrix<double> Lt, AMat, PMat, AMat_Vb;
    transpose(LMat, Lt);
    mult(Lt, RMat, AMat);

    if(this->sketch_count_ != 1) {
      auto ivb = linkIndex(this->vb_, core_id - 1);
      int rank_vb = dim(ivb);
      LMat = Matrix<double>(rank_vb, rank);
      RMat = Matrix<double>(rank_vb, rank);
      cout << "envi_L_Vb " << core_id - 1 << endl;
      PrintData(envi_L_Vb[core_id - 1]);
      cout << "envi_R_Vb " << core_id - 2 << endl;
      PrintData(envi_R_Vb[core_id - 2]);
      for(unsigned i = 1; i <= rank_vb; ++i) {
        for(int j = 1; j <= rank; ++j) {
          LMat(i - 1, j - 1) = envi_L_Vb[core_id - 1].elt(ivb = i, links(core_id - 1) = j);
          RMat(i - 1, j - 1) = envi_R_Vb[core_id - 2].elt(ivb = i, links(core_id - 1) = j);
        }
      }
      Matrix<double> AMat_Vb;
      transpose(LMat, Lt);
      mult(Lt, RMat, AMat_Vb);
      cout << "AMat " << core_id << " " << AMat.nrows() << " " << AMat.ncols() << endl;
      cout << "AMat_Vb " << core_id << " " << AMat_Vb.nrows() << " " << AMat_Vb.ncols() << endl;
    }

    ITensor A(prime(links(core_id - 1)), links(core_id - 1));
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= rank; ++j) {
        A.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat(i - 1, j - 1));
      }
    }
    if(this->sketch_count_ != 1) {
      ITensor A_Vb(prime(links(core_id - 1)), links(core_id - 1));
      for(int i = 1; i <= rank; ++i) {
        for(int j = 1; j <= rank; ++j) {
          A_Vb.set(prime(links(core_id - 1)) = i, links(core_id - 1) = j, AMat_Vb(i - 1, j - 1));
        }
      }
      cout << "A " << core_id << endl;
      PrintData(A);
      cout << "A_Vb " << core_id << endl;
      PrintData(A_Vb);
      A += A_Vb;
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

  G.ref(1) = Bemp(1) * V[1];
  for(unsigned core_id = 2; core_id <= this->d_; ++core_id) {
    int rank = dim(links(core_id - 1)), rank_trimmed = dim(links_trimmed[core_id - 2]);
    ITensor A = U[core_id - 1] * S[core_id - 1];
    ITensor Pinv(links_trimmed[core_id - 2], links(core_id - 1));
    Matrix<double> AMat(rank, rank_trimmed), PMat;
    for(int i = 1; i <= rank; ++i) {
      for(int j = 1; j <= rank_trimmed; ++j) {
        AMat(i - 1, j - 1) = A.elt(prime(links(core_id - 1)) = i, links_trimmed[core_id - 2] = j);
      }
    }
    pseudoInvert(AMat, PMat);

    for(int i = 1; i <= rank_trimmed; ++i) {
      for(int j = 1; j <= rank; ++j) {
        Pinv.set(links_trimmed[core_id - 2] = i, links(core_id - 1) = j, PMat(i - 1, j - 1));
      }
    }
    G.ref(core_id) = Pinv * Bemp(core_id);
    if(core_id != this->d_) {
      G.ref(core_id) *= V[core_id];
    }
  }

  log << "Final ranks ";
  for(unsigned i = 1; i < this->d_; ++i) {
    log << dim(linkIndex(G, i)) << " ";
  }
  log << "\n";
  log.flush();

  this->vb_ = G;
}

MPS TTMetaD::createTTCoeff() const {
  int n = this->sketch_basis_[0].nbasis();
  auto sites = SiteSet(this->d_, n);
  auto coeff = randomMPS(sites, this->sketch_rc_);
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

pair<vector<ITensor>, IndexSet> TTMetaD::intBasisSample(const IndexSet& is) const {
  unsigned N = this->hills_.size();
  int nb = this->sketch_basis_[0].nbasis();
  auto sites_new = SiteSet(this->d_, N);
  vector<ITensor> M;
  vector<Index> is_new;
  for(unsigned i = 1; i <= this->d_; ++i) {
    double L = (this->sketch_basis_[i - 1].dom().second - this->sketch_basis_[i - 1].dom().first) / 2;
    double a = (this->sketch_basis_[i - 1].dom().second + this->sketch_basis_[i - 1].dom().first) / 2;
    M.push_back(ITensor(sites_new(i), is(i)));
    is_new.push_back(sites_new(i));
    for(unsigned j = 1; j <= N; ++j) {
      double x = this->hills_[j - 1].center[i - 1];
      double w = this->hills_[j - 1].sigma[i - 1];
      double h = pow(this->hills_[j - 1].height, 1.0 / this->d_);
      for(int pos = 1; pos <= nb; ++pos) {
        double result = 0.0;
        if(pos == 1) {
          result = h * sqrt(M_PI / L) * w;
        } else if(pos % 2 == 0) {
          result = exp(-pow(M_PI * w * (pos / 2), 2) / (2 * pow(L, 2))) * h * sqrt(2 * M_PI / L) * w * cos(M_PI * (x - a) * (pos / 2) / L);
        } else {
          result = exp(-pow(M_PI * w * (pos / 2), 2) / (2 * pow(L, 2))) * h * sqrt(2 * M_PI / L) * w * sin(M_PI * (x - a) * (pos / 2) / L);
        }
        M.back().set(sites_new(i) = j, is(i) = pos, result);
      }
    }
  }
  return make_pair(M, IndexSet(is_new));
}

tuple<MPS, vector<ITensor>, vector<ITensor>> TTMetaD::formTensorMoment(const vector<ITensor>& M, const MPS& coeff, const IndexSet& is) {
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

tuple<MPS, vector<ITensor>, vector<ITensor>> TTMetaD::formTensorMomentVb(const MPS& coeff) {
  for(unsigned i = 1; i <= this->d_; ++i) {
    this->vb_.ref(i) *= delta(siteIndex(this->vb_, i), siteIndex(coeff, i));
  }
  vector<ITensor> envi_L(this->d_);
  envi_L[1] = coeff(1) * this->vb_(1);
  for(unsigned i = 2; i < this->d_; ++i) {
    envi_L[i] = envi_L[i - 1] * coeff(i) * this->vb_(i);
  }

  vector<ITensor> envi_R(this->d_);
  envi_R[this->d_ - 2] = coeff(this->d_) * this->vb_(this->d_);
  for(int i = this->d_ - 3; i >= 0; --i) {
    envi_R[i] = envi_R[i + 1] * coeff(i + 2) * this->vb_(i + 2);
  }

  MPS B(this->d_);
  B.ref(1) = this->vb_(1) * envi_R[0];
  for(unsigned core_id = 2; core_id < this->d_; ++core_id) {
    B.ref(core_id) = envi_L[core_id - 1] * this->vb_(core_id) * envi_R[core_id - 1];
  }
  B.ref(this->d_) = envi_L[this->d_ - 1] * this->vb_(this->d_);

  return make_tuple(B, envi_L, envi_R);
}

}
}
