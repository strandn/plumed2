#include "TTCross.h"
#include "TTHelper.h"
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Communicator.h"
#include "tools/File.h"
#include "tools/Stopwatch.h"
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
  vector<MPS> ttList_;
  TTCross aca_;
  vector<BasisFunc> basis_;
  vector<BasisFunc> aca_basis_;
  vector<vector<double>> samples_;
  vector<vector<double>> lastsamples_;
  vector<double> traj_;
  double vmax_;
  double alpha_;
  double lambda_;
  bool isFirstStep_;
  unsigned count_;
  double bf_;
  bool conv_;
  bool walkers_mpi_;
  int mpi_size_;
  int mpi_rank_;
  bool do_aca_;
  double vshift_;
  int output_2d_;
  unsigned max_samples_;
  OFile pivot_file_;
  ForwardDecl<Stopwatch> stopwatch_fwd;
  Stopwatch& stopwatch = *stopwatch_fwd;
  string samplesfname_;
  OFile samplesOfile_;

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
  keys.addFlag("NOCONV", false, "Specifies that TTSketch densities and gradients should not be smoothed via Gaussian kernels whenever evaluated");
  keys.addFlag("KERNEL_BASIS", false, "Specifies that local kernel basis should be used instead of Fourier basis");
  keys.addFlag("WALKERS_MPI", false, "To be used when gromacs + multiple walkers are used");
  keys.addFlag("DO_ACA", false, "Approximate Vbias not as explicit sum but rather as TTCross approximation");
  keys.addFlag("ACA_NOCONV", false, "Specifies that TTCross functions and gradients should not be smoothed via Gaussian kernels whenever evaluated");
  keys.addFlag("ACA_KERNEL_BASIS", false, "Specifies that local kernel basis should be used instead of Fourier basis for TTCross functions");
  keys.addFlag("ACA_AUTO_RANK", false, "Specifies that during TTCross, an optimal rank will be chosen based on error analysis");
  keys.add("optional", "RANK", "Target rank for TTSketch algorithm - compulsory if CUTOFF is not specified");
  keys.add("optional", "CUTOFF", "Truncation error cutoff for singular value decomposition - compulsory if RANK is not specified");
  keys.add("optional", "TEMP", "The system temperature");
  keys.add("optional", "VMAX", "Upper limit of Vbias across all CV space, in units of kT");
  keys.add("optional", "WIDTH", "Width of Gaussian kernels");
  keys.add("compulsory", "INITRANK", "Initial rank for TTSketch algorithm");
  keys.add("compulsory", "PACE", "1e6", "The frequency for Vbias updates");
  keys.add("compulsory", "SAMPLESTRIDE", "100", "The frequency with which samples are collected for density estimation");
  keys.add("compulsory", "INTERVAL_MIN", "Lower limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "INTERVAL_MAX", "Upper limits, outside the limits the system will not feel the biasing force");
  keys.add("compulsory", "NBASIS", "20", "Number of Fourier basis functions per dimension");
  keys.add("compulsory", "ALPHA", "0.05", "Weight coefficient for random tensor train construction");
  keys.add("optional", "LAMBDA", "Ratio of largest to smallest allowed density magnitudes");
  keys.add("optional", "BIASFACTOR", "For well-tempering");
  keys.use("RESTART");
  keys.add("optional", "FILE", "Name of the file where samples are stored");
  keys.add("optional", "ACA_CUTOFF", "Convergence threshold for TT-cross calculations");
  keys.add("optional", "ACA_RANK", "Largest possible rank for TT-cross calculations");
  keys.add("compulsory", "OUTPUT_2D", "0", "Number of bins per dimension for outputting 2D marginals of sketch densities - 0 for no output");
  keys.add("optional", "MAX_SAMPLES", "Limits the number of samples kept in memory");
}

TTSketch::TTSketch(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  r_(0),
  cutoff_(0.0),
  kbt_(0.0),
  vmax_(numeric_limits<double>::max()),
  lambda_(100.0),
  isFirstStep_(true),
  count_(1),
  bf_(1.0),
  conv_(true),
  walkers_mpi_(false),
  mpi_size_(0),
  mpi_rank_(0),
  do_aca_(false),
  vshift_(0.0),
  max_samples_(numeric_limits<unsigned>::max())
{
  bool noconv, kernel, aca_noconv, aca_kernel, aca_auto_rank;
  parseFlag("NOCONV", noconv);
  parseFlag("KERNEL_BASIS", kernel);
  parseFlag("WALKERS_MPI", this->walkers_mpi_);
  parseFlag("DO_ACA", this->do_aca_);
  parseFlag("ACA_NOCONV", aca_noconv);
  parseFlag("ACA_KERNEL_BASIS", aca_kernel);
  parseFlag("ACA_AUTO_RANK", aca_auto_rank);
  this->d_ = getNumberOfArguments();
  if(this->d_ < 2) {
    error("Number of arguments must be at least 2");
  }
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
  vector<double> w;
  parseVector("WIDTH", w);
  if(!noconv && w.size() != this->d_) {
    error("Number of arguments does not match number of WIDTH parameters");
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
  if(nbasis <= 1) {
    error("NBASIS must be greater than 1");
  }
  if(nbasis % 2 == 0) {
    ++nbasis;
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
    error("BIASFACTOR must be greater than 1");
  }
  for(unsigned i = 0; i < this->d_; ++i) {
    if(!noconv && w[i] <= 0.0) {
      error("Gaussian smoothing requires positive WIDTH");
    }
    if(interval_max[i] <= interval_min[i]) {
      error("INTERVAL_MAX parameters need to be greater than respective INTERVAL_MIN parameters");
    }
    double width = noconv ? 0.0 : w[i];
    this->basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]), nbasis, width, kernel));
    if(this->do_aca_) {
      this->aca_basis_.push_back(BasisFunc(make_pair(interval_min[i], interval_max[i]), nbasis, width, aca_kernel));
    }
  }
  this->conv_ = !noconv;

  double aca_cutoff = 0.0;
  parse("ACA_CUTOFF", aca_cutoff);
  if(this->do_aca_ && (aca_cutoff < 0.0 || aca_cutoff >= 1.0)) {
    error("TTCross requires ACA_CUTOFF that is nonnegative and less than 1");
  }
  int aca_rank = numeric_limits<int>::max();
  parse("ACA_RANK", aca_rank);
  if(this->do_aca_ && aca_rank <= 0) {
    error("TTCross requires positive ACA_RANK");
  }

  if(this->do_aca_) {
    vector<Value*> args(this->d_);
    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      this->pivot_file_.link(*this);
      this->pivot_file_.enforceSuffix("");
      this->pivot_file_.open("pivots.dat");
      this->pivot_file_.setHeavyFlush();
      for(unsigned i = 0; i < this->d_; ++i) {
        args[i] = getPntrToArgument(i);
        this->pivot_file_.setupPrintValue(args[i]);
      }
    }
    this->aca_ = TTCross(this->aca_basis_, this->basis_, getkBT(), aca_cutoff,
                         aca_rank, log, !aca_noconv, !noconv,
                         5 * (nbasis - 1), this->walkers_mpi_,
                         this->mpi_rank_, aca_auto_rank, this->pivot_file_,
                         args);
  }

  if(this->walkers_mpi_) {
    this->mpi_size_ = multi_sim_comm.Get_size();
    this->mpi_rank_ = multi_sim_comm.Get_rank();
  }

  parse("OUTPUT_2D", this->output_2d_);
  if(this->output_2d_ < 0) {
    error("OUTPUT_2D must be nonnegative");
  }

  parse("MAX_SAMPLES", this->max_samples_);
  if(this->max_samples_ <= 0) {
    error("MAX_SAMPLES must be positive");
  }

  this->samplesfname_ = "SAMPLES";
  parse("FILE", this->samplesfname_);
  if(this->walkers_mpi_) {
    this->samplesfname_ += "." + to_string(this->mpi_rank_);
  }

  if(getRestart()) {
    int npivots = 0;
    IFile samples_ifile;
    bool done = false;
    while(true) {
      string filename = this->samplesfname_ + "." + to_string(this->count_ - 1);
      if(samples_ifile.FileExist(filename)) {
        samples_ifile.open(filename);
      } else {
        break;
      }
      for(int i = 0; i <= this->pace_ / this->stride_; ++i) {
        vector<double> cv;
        vector<Value> tmpvalues;
        for(unsigned j = 0; j < this->d_; ++j) {
          if(!samples_ifile.scanField(&tmpvalues[j])) {
            done = true;
            break;
          }
          cv[j] = tmpvalues[j].get();
        }
        if(done) {
          break;
        }
        this->traj_.insert(this->traj_.end(), cv.begin(), cv.end());
        samples_ifile.scanField();
      }
      samples_ifile.close();
      if(done) {
        break;
      }
      if(this->walkers_mpi_) {
        vector<double> all_traj(this->mpi_size_ * this->traj_.size(), 0.0);
        multi_sim_comm.Allgather(this->traj_, all_traj);
        if(this->mpi_rank_ == 0) {
          for(int i = 0; i < this->mpi_size_; ++i) {
            for(unsigned j = 0; j < this->traj_.size() / this->d_; ++j) {
              vector<double> step(all_traj.begin() + i * this->traj_.size() + j * this->d_,
                                  all_traj.begin() + i * this->traj_.size() + (j + 1) * this->d_);
              this->samples_.push_back(step);
              this->lastsamples_.push_back(step);
              if(this->do_aca_) {
                this->aca_.addSample(step);
              }
            }
          }
        }
      } else {
        for(unsigned i = 0; i < this->traj_.size(); i += this->d_) {
          vector<double> step(this->traj_.begin() + i, this->traj_.begin() + i + this->d_);
          this->samples_.push_back(step);
          this->lastsamples_.push_back(step);
          if(this->do_aca_) {
            this->aca_.addSample(step);
          }
        }
      }
      this->traj_.clear();
      if(this->samples_.size() > this->max_samples_) {
        this->samples_.erase(this->samples_.begin(), this->samples_.begin() + (this->samples_.size() - this->max_samples_));
        if(this->do_aca_) {
          this->aca_.trimSamples(this->max_samples_);
        }
      }
      this->count_++;
    }
    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      if(this->do_aca_) {
        IFile pivot_ifile;
        if(pivot_ifile.FileExist("pivots.dat")) {
          pivot_ifile.open("pivots.dat");
        } else {
          error("The file pivots.dat cannot be found!");
        }
        bool done = false;
        while(true) {
          vector<double> cv;
          vector<Value> tmpvalues;
          for(unsigned i = 0; i < this->d_; ++i) {
            if(!pivot_ifile.scanField(&tmpvalues[i])) {
              done = true;
              break;
            }
            cv[i] = tmpvalues[i].get();
          }
          if(done) {
            break;
          }
          this->aca_.addPivot(cv);
          pivot_ifile.scanField();
          ++npivots;
        }
        pivot_ifile.close();
      }
    }

    if(this->walkers_mpi_) {
      multi_sim_comm.Bcast(this->count_, 0);
    }
    
    string ttfilename = "ttsketch.h5";
    if(this->walkers_mpi_) {
      ttfilename = "../" + ttfilename;
    }
    for(unsigned i = 2; i <= this->count_; ++i) {
      // try {
      //   this->ttList_.push_back(ttRead(ttfilename, i));
      // } catch(...) {
      //   this->count_ = i - 1;
      //   if(this->walkers_mpi_) {
      //     multi_sim_comm.Bcast(this->count_, 0);
      //   }
      //   break;
      // }
      this->ttList_.push_back(ttRead(ttfilename, i));
    }
    if(this->do_aca_) {
      // try {
      //   this->aca_.readVb(this->count_);
      // } catch(...) {
      //   --this->count_;
      //   if(this->walkers_mpi_) {
      //     multi_sim_comm.Bcast(this->count_, 0);
      //   }
      //   this->ttList_.pop_back();
      //   this->aca_.readVb(this->count_);
      // }
      this->aca_.readVb(this->count_);
    }

    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      if(!this->do_aca_) {
        double vpeak = 0.0;
        for(auto& s : this->samples_) {
          double bias = getBias(s);
          if(bias > vpeak) {
            vpeak = bias;
          }
        }
        this->vshift_ = max(vpeak - this->vmax_, 0.0);
        log << "  Vtop = " << vpeak << " Vshift = " << this->vshift_ << "\n";
      }
    }

    if(this->walkers_mpi_) {
      multi_sim_comm.Bcast(this->vshift_, 0);
    }
    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      log << "  restarting from step " << this->count_ << "\n";
      log << "  " << this->samples_.size() << " samples retrieved\n";
      if(this->do_aca_) {
        log << "  " << npivots << " pivots retrieved\n";
      }
    }
  }
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
    if(this->walkers_mpi_) {
      vector<double> all_traj(this->mpi_size_ * this->traj_.size(), 0.0);
      multi_sim_comm.Allgather(this->traj_, all_traj);
      if(this->mpi_rank_ == 0) {
        for(int i = 0; i < this->mpi_size_; ++i) {
          for(unsigned j = 0; j < this->traj_.size() / this->d_; ++j) {
            vector<double> step(all_traj.begin() + i * this->traj_.size() + j * this->d_,
                                all_traj.begin() + i * this->traj_.size() + (j + 1) * this->d_);
            this->samples_.push_back(step);
            this->lastsamples_.push_back(step);
            if(this->do_aca_) {
              this->aca_.addSample(step);
            }
          }
        }
      }
    } else {
      for(unsigned i = 0; i < this->traj_.size(); i += this->d_) {
        vector<double> step(this->traj_.begin() + i, this->traj_.begin() + i + this->d_);
        this->samples_.push_back(step);
        this->lastsamples_.push_back(step);
        if(this->do_aca_) {
          this->aca_.addSample(step);
        }
      }
    }
    this->traj_.clear();
    if(this->samples_.size() > this->max_samples_) {
      this->samples_.erase(this->samples_.begin(), this->samples_.begin() + (this->samples_.size() - this->max_samples_));
      if(this->do_aca_) {
        this->aca_.trimSamples(this->max_samples_);
      }
    }
  } else {
    nowAddATT = false;
    this->isFirstStep_ = false;
  }

  vector<double> cv(this->d_);
  for(unsigned i = 0; i < this->d_; ++i) {
    cv[i] = getArgument(i);
  }

  if(nowAddATT) {
    this->vshift_ = 0.0;
    if(!this->walkers_mpi_ || this->mpi_rank_ == 0) {
      unsigned N = this->lastsamples_.size();
      log << "Sample limits\n";
      for(unsigned i = 0; i < this->d_; ++i) {
        auto [large, small] = this->basis_[i].dom();
        for(unsigned j = 0; j < N; ++j) {
          if(this->lastsamples_[j][i] > large) {
            large = this->lastsamples_[j][i];
          }
          if(this->lastsamples_[j][i] < small) {
            small = this->lastsamples_[j][i];
          }
        }
        log << small << " " << large << "\n";
      }

      double hf = 1.0;
      double vmean = 0.0;
      if(this->bf_ > 1.0) {
        vector<double> vlist(N);
        for(unsigned i = 0; i < N; ++i) {
          vlist[i] = getBias(this->lastsamples_[i]);
        }
        vmean = accumulate(vlist.begin(), vlist.end(), 0.0) / N;
        hf = exp(-vmean / (this->kbt_ * (this->bf_ - 1)));
      }

      log << "\nForming TT-sketch density...\n";
      log.flush();
      paraSketch();

      log << "\nEmpirical means:\n";
      Matrix<double> sigmahat(this->d_, this->d_);
      vector<double> muhat(this->d_, 0.0);
      for(unsigned k = 0; k < this->d_; ++k) {
        for(unsigned j = 0; j < N; ++j) {
          muhat[k] += this->lastsamples_[j][k] / N;
        }
        log << muhat[k] << " ";
      }
      log << "\nEmpirical covariance matrix:\n";
      for(unsigned k = 0; k < this->d_; ++k) {
        for(unsigned l = k; l < this->d_; ++l) {
          sigmahat(k, l) = sigmahat(l, k) = 0.0;
          for(unsigned j = 0; j < N; ++j) {
            sigmahat(k, l) += (this->lastsamples_[j][k] - muhat[k]) * (this->lastsamples_[j][l] - muhat[l]) / (N - 1);
          }
          sigmahat(l, k) = sigmahat(k, l);
        }
      }
      matrixOut(log, sigmahat);
      auto [sigma, mu] = covMat(this->ttList_.back(), this->basis_);
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

      if(this->output_2d_ > 0) {
        for(unsigned k = 0; k < this->d_; ++k) {
          for(unsigned l = k + 1; l < this->d_; ++l) {
            vector<vector<double>> marginals(this->output_2d_, vector<double>(this->output_2d_, 0.0));
            marginal2d(this->ttList_.back(), this->basis_, k + 1, l + 1, marginals, false);
            string filename = "ttsketch_" + getPntrToArgument(k)->getName() + "_" + getPntrToArgument(l)->getName() + "_" +
                              to_string(this->count_ - 2) + ".dat";
            if(this->walkers_mpi_) {
              filename = "../" + filename;
            }
            OFile file;
            file.link(*this);
            file.enforceSuffix("");
            file.open(filename);
            file.setHeavyFlush();
            file.setupPrintValue(getPntrToArgument(k));
            file.setupPrintValue(getPntrToArgument(l));
            auto xdom = this->basis_[k].dom();
            auto ydom = this->basis_[l].dom();
            for(int i = 0; i < this->output_2d_; ++i) {
              for(int j = 0; j < this->output_2d_; ++j) {
                double x = xdom.first + i * (xdom.second - xdom.first) / this->output_2d_;
                double y = ydom.first + j * (ydom.second - ydom.first) / this->output_2d_;
                file.printField(getPntrToArgument(k), x);
                file.printField(getPntrToArgument(l), y);
                file.printField("hh" + getPntrToArgument(k)->getName() + getPntrToArgument(l)->getName(), marginals[i][j]);
                file.printField();
              }
            }
          }
        }
      }

      if(this->output_2d_ > 0) {
        for(unsigned k = 0; k < this->d_; ++k) {
          for(unsigned l = k + 1; l < this->d_; ++l) {
            vector<vector<double>> marginals(this->output_2d_, vector<double>(this->output_2d_, 0.0));
            marginal2d(this->ttList_.back(), this->basis_, k + 1, l + 1, marginals, true);
            string filename = "ttsketch_conv_" + getPntrToArgument(k)->getName() + "_" + getPntrToArgument(l)->getName() + "_" +
                              to_string(this->count_ - 2) + ".dat";
            if(this->walkers_mpi_) {
              filename = "../" + filename;
            }
            OFile file;
            file.link(*this);
            file.enforceSuffix("");
            file.open(filename);
            file.setHeavyFlush();
            file.setupPrintValue(getPntrToArgument(k));
            file.setupPrintValue(getPntrToArgument(l));
            auto xdom = this->basis_[k].dom();
            auto ydom = this->basis_[l].dom();
            for(int i = 0; i < this->output_2d_; ++i) {
              for(int j = 0; j < this->output_2d_; ++j) {
                double x = xdom.first + i * (xdom.second - xdom.first) / this->output_2d_;
                double y = ydom.first + j * (ydom.second - ydom.first) / this->output_2d_;
                file.printField(getPntrToArgument(k), x);
                file.printField(getPntrToArgument(l), y);
                file.printField("hh" + getPntrToArgument(k)->getName() + getPntrToArgument(l)->getName(), marginals[i][j]);
                file.printField();
              }
            }
          }
        }
      }

      double rhomax = 0.0;
      for(auto& s : this->lastsamples_) {
        double rho = ttEval(this->ttList_.back(), this->basis_, s, this->conv_);
        if(rho > rhomax) {
          rhomax = rho;
        }
      }
      this->ttList_.back() *= pow(this->lambda_, hf) / rhomax;
      if(this->do_aca_) {
        this->aca_.updateG(this->ttList_.back());
      }
      
      double vpeak = 0.0;
      vector<double> gradtop(this->d_, 0.0);
      vector<double> topsample;
      vector<vector<double>> topsamples(this->d_);
      if(this->do_aca_) {
        this->aca_.updateVshift(0.0);
        auto vtopresult = this->aca_.vtop();
        vpeak = vtopresult.first;
        topsample = vtopresult.second;
      } else {
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
      }
      this->vshift_ = max(vpeak - this->vmax_, 0.0);
      if(this->do_aca_) {
        this->aca_.updateVshift(this->vshift_);
      }
      log << "\n";
      if(this->bf_ > 1.0) {
        log << "Vmean = " << vmean << " Height = " << this->kbt_ * std::log(pow(this->lambda_, hf)) << "\n";
      }
      log << "Vtop = " << vpeak << " Vshift = " << this->vshift_ << "\n";
      for(unsigned j = 0; j < this->d_; ++j) {
        log << topsample[j] << " ";
      }
      log << "\n\n";
      log.flush();

      string ttfilename = "ttsketch.h5";
      if(this->walkers_mpi_) {
        ttfilename = "../" + ttfilename;
      }
      ttWrite(ttfilename, this->ttList_.back(), this->count_);

      if(this->do_aca_) {
        // if(this->d_ == 2) {
        //   ofstream file;
        //   if(this->count_ == 2) {
        //     file.open("F0.txt");
        //   } else {
        //     file.open("F0.txt", ios_base::app);
        //   }
        //   for(int i = 0; i < 100; ++i) {
        //     double x = -M_PI + 2 * i * M_PI / 100;
        //     for(int j = 0; j < 100; ++j) {
        //       double y = -M_PI + 2 * j * M_PI / 100;
        //       file << x << " " << y << " " << max(this->aca_.f({ x, y }), 0.0) << endl;
        //     }
        //   }
        //   file.close();
        // }
        
        this->aca_.updateVb();
        this->aca_.writeVb(this->count_);

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
        log << "\n";
      }

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

      this->lastsamples_.clear();
      
      // if(this->d_ == 2) {
      //   ofstream file, filex, filey;
      //   if(this->count_ == 2) {
      //     file.open("F.txt");
      //     filex.open("dFdx.txt");
      //     filey.open("dFdy.txt");
      //   } else {
      //     file.open("F.txt", ios_base::app);
      //     filex.open("dFdx.txt", ios_base::app);
      //     filey.open("dFdy.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       vector<double> der(this->d_, 0.0);
      //       double ene = getBiasAndDerivatives({ x, y }, der);
      //       file << x << " " << y << " " << ene << endl;
      //       filex << x << " " << y << " " << der[0] << endl;
      //       filey << x << " " << y << " " << der[1] << endl;
      //     }
      //   }
      //   file.close();
      //   filex.close();
      //   filey.close();
      // }

      // if(this->d_ == 3) {
      //   ofstream file;
      //   if(this->count_ == 2) {
      //     file.open("phi2phi3_0.txt");
      //   } else {
      //     file.open("phi2phi3_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, y, -1.2 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi2phi3_1.txt");
      //   } else {
      //     file.open("phi2phi3_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, y, 1.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi2phi4_0.txt");
      //   } else {
      //     file.open("phi2phi4_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, -1.2, y }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi2phi4_1.txt");
      //   } else {
      //     file.open("phi2phi4_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, 1.0, y }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi3phi4_0.txt");
      //   } else {
      //     file.open("phi3phi4_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ -1.2, x, y }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi3phi4_1.txt");
      //   } else {
      //     file.open("phi3phi4_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ 1.0, x, y }) << endl;
      //     }
      //   }
      //   file.close();
      // }
      // if(this->d_ == 6) {
      //   ofstream file;
      //   if(this->count_ == 2) {
      //     file.open("phi2psi2_0.txt");
      //   } else {
      //     file.open("phi2psi2_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, y, -1.2, 0.0, -1.2, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi2psi2_1.txt");
      //   } else {
      //     file.open("phi2psi2_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ x, y, 1.0, 0.0, -1.2, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("psi2phi3_0.txt");
      //   } else {
      //     file.open("psi2phi3_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ -1.2, x, y, 0.0, -1.2, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("psi2phi3_1.txt");
      //   } else {
      //     file.open("psi2phi3_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ 1.0, x, y, 0.0, 1.0, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi3psi3_0.txt");
      //   } else {
      //     file.open("phi3psi3_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ -1.2, 0.0, x, y, -1.2, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi3psi3_1.txt");
      //   } else {
      //     file.open("phi3psi3_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ 1.0, 0.0, x, y, 1.0, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("psi3phi4_0.txt");
      //   } else {
      //     file.open("psi3phi4_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, x, y, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("psi3phi4_1.txt");
      //   } else {
      //     file.open("psi3phi4_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, x, y, 0.0 }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi4psi4_0.txt");
      //   } else {
      //     file.open("phi4psi4_0.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ -1.2, 0.0, -1.2, 0.0, x, y }) << endl;
      //     }
      //   }
      //   file.close();
      //   if(this->count_ == 2) {
      //     file.open("phi4psi4_1.txt");
      //   } else {
      //     file.open("phi4psi4_1.txt", ios_base::app);
      //   }
      //   for(int i = 0; i < 100; ++i) {
      //     double x = -M_PI + 2 * i * M_PI / 100;
      //     for(int j = 0; j < 100; ++j) {
      //       double y = -M_PI + 2 * j * M_PI / 100;
      //       file << x << " " << y << " " << getBias({ 1.0, 0.0, 1.0, 0.0, x, y }) << endl;
      //     }
      //   }
      //   file.close();
      // }
    }

    if(this->walkers_mpi_) {
      multi_sim_comm.Bcast(this->count_, 0);
      multi_sim_comm.Bcast(this->vshift_, 0);
      if(this->mpi_rank_ != 0) {
        this->ttList_.push_back(ttRead("../ttsketch.h5", this->count_));
        if(this->do_aca_) {
          this->aca_.readVb(this->count_);
        }
      }
    }
    stopwatch.stop("Timing " + to_string(this->count_ - 1));
    log << stopwatch << "\n";
    log.flush();
  }

  if(getStep() % this->pace_ == 0) {
    if(!this->isFirstStep_) {
      this->samplesOfile_.flush();
      this->samplesOfile_.close();
      // if(this->mpi_rank_ == 0) {
      //   cout << 1 << " " << getStep() << " " << this->count_ << endl;
      // }
    }
    this->samplesOfile_.link(*this);
    this->samplesOfile_.enforceSuffix("");
    this->samplesOfile_.open(this->samplesfname_ + "." + to_string(this->count_ - 1));
    // if(!this->samplesOfile_.isOpen()) {
    //   plumed_merror("Failed to open samplesOfile: " + this->samplesfname_ + "." + to_string(this->count_ - 1));
    // }
    this->samplesOfile_.setHeavyFlush();
    for(unsigned i = 0; i < this->d_; ++i) {
      this->samplesOfile_.setupPrintValue(getPntrToArgument(i));
    }
    // if(this->mpi_rank_ == 0) {
    //   cout << 2 << " " << getStep() << " " << this->count_ << endl;
    // }
    // this->samplesOfile_.printField();
  }

  if(getStep() % this->stride_ == 0) {
    this->traj_.insert(this->traj_.end(), cv.begin(), cv.end());
    // if(!this->samplesOfile_.isOpen()) {
    //   plumed_merror("samplesOfile_ is not open at step " + std::to_string(getStep()));
    // }
    for(unsigned j = 0; j < this->d_; ++j) {
      this->samplesOfile_.printField(getPntrToArgument(j), cv[j]);
    }
    this->samplesOfile_.printField();
    // if(this->mpi_rank_ == 0) {
    //   cout << 3 << " " << getStep() << " " << this->count_ << endl;
    // }
  }

  if(getStep() % this->pace_ == 1) {
    log << "Vbias update " << this->count_ << "...\n\n";
    log.flush();
    stopwatch.start("Timing " + to_string(this->count_));
  }
}

double TTSketch::getBiasAndDerivatives(const vector<double>& cv, vector<double>& der) {
  double bias = getBias(cv);
  if(bias == 0.0) {
    return 0.0;
  }
  if(this->do_aca_) {
    der = ttGrad(this->aca_.vb(), this->aca_basis_, cv, this->aca_.conv());
  } else {
    for(auto& tt : this->ttList_) {
      double rho = ttEval(tt, this->basis_, cv, this->conv_);
      if(rho > 1.0) {
        auto deri = ttGrad(tt, this->basis_, cv, this->conv_);
        transform(deri.begin(), deri.end(), deri.begin(), bind(multiplies<double>(), placeholders::_1, this->kbt_ / rho));
        transform(der.begin(), der.end(), deri.begin(), der.begin(), plus<double>());
      }
    }
  }
  return bias;
}

double TTSketch::getBias(const vector<double>& cv) {
  if(this->do_aca_) {
    if(length(this->aca_.vb()) == 0) {
      return 0.0;
    }
    return max(ttEval(this->aca_.vb(), this->aca_basis_, cv, this->aca_.conv()), 0.0);
  } else {
    double bias = 0.0;
    for(auto& tt : this->ttList_) {
      bias += this->kbt_ * std::log(max(ttEval(tt, this->basis_, cv, this->conv_), 1.0));
    }
    return max(bias - this->vshift_, 0.0);
  }
}

void TTSketch::paraSketch() {
  unsigned N = this->lastsamples_.size();
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
    if(this->r_ > 0) {
      svd(A, U[core_id - 1], S[core_id - 1], V[core_id - 1], {"Cutoff=", this->cutoff_, "RightTags=", original_link_tags, "MaxDim=", this->r_});
    } else {
      svd(A, U[core_id - 1], S[core_id - 1], V[core_id - 1], {"Cutoff=", this->cutoff_, "RightTags=", original_link_tags});
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

  if(this->basis_[0].kernel()) {
    for(unsigned i = 1; i <= this->d_; ++i) {
      auto s = siteIndex(G, i);
      ITensor ginv(s, prime(s));
      for(int j = 1; j <= dim(s); ++j) {
        for(int l = 1; l <= dim(s); ++l) {
          ginv.set(s = j, prime(s) = l, this->basis_[i - 1].ginv()(j - 1, l - 1));
        }
      }
      G.ref(i) *= ginv;
      G.ref(i).noPrime();
    }
  }

  this->ttList_.push_back(G);
  ++this->count_;
}

MPS TTSketch::createTTCoeff() const {
  default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
  normal_distribution<double> distribution(0.0, 1.0);
  int n = this->basis_[0].nbasis();
  auto sites = SiteSet(this->d_, n);
  auto coeff = MPS(sites, this->rc_);
  for(int j = 1; j <= n; ++j) {
    for(int k = 1; k <= this->rc_; ++k) {
      coeff.ref(1).set(sites(1) = j, linkIndex(coeff, 1) = k, distribution(generator));
    }
  }
  for(unsigned i = 2; i <= this->d_ - 1; ++i) {
    for(int j = 1; j <= n; ++j) {
      for(int k = 1; k <= this->rc_; ++k) {
        for(int l = 1; l <= this->rc_; ++l) {
          coeff.ref(i).set(sites(i) = j, linkIndex(coeff, i - 1) = k, linkIndex(coeff, i) = l, distribution(generator));
        }
      }
    }
  }
  for(int j = 1; j <= n; ++j) {
    for(int k = 1; k <= this->rc_; ++k) {
      coeff.ref(this->d_).set(sites(this->d_) = j, linkIndex(coeff, this->d_ - 1) = k, distribution(generator));
    }
  }
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
  unsigned N = this->lastsamples_.size();
  int nb = this->basis_[0].nbasis();
  auto sites_new = SiteSet(this->d_, N);
  vector<ITensor> M;
  vector<Index> is_new;
  double h = pow(1.0 / N, 1.0 / this->d_);
  for(unsigned i = 1; i <= this->d_; ++i) {
    M.push_back(ITensor(sites_new(i), is(i)));
    is_new.push_back(sites_new(i));
    for(unsigned j = 1; j <= N; ++j) {
      double x = this->lastsamples_[j - 1][i - 1];
      for(int pos = 1; pos <= nb; ++pos) {
        M.back().set(sites_new(i) = j, is(i) = pos, h * this->basis_[i - 1](x, pos, false));
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
