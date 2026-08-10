/* Transition Path Process (TPP) bias.
 *
 * Drives trajectories along reactive paths using one of four drift modes
 * selected via MODE.  Symmetric xi regularization is used throughout:
 *
 *   q_xi = xi + (1-2*xi)*q,   q_xi in [xi, 1-xi]  for xi in (0, 0.5).
 *
 * ---- NEG_LOG_Q* modes — COEFF_FILE stores q (the committor) ----
 *
 *   MODE=NEG_LOG_Q (default, full Doob drift):
 *     force_i = 2*kBT * (1-2*xi) * dq/dCV_i / q_xi
 *     V_bias  = -2*kBT * log q_xi
 *
 *   MODE=NEG_LOG_Q_HALF (Picard modified drift, soc.pdf eq. 19):
 *     force_i = kBT * (1-2*xi) * dq/dCV_i / q_xi
 *     V_bias  = -kBT * log q_xi
 *
 * ---- HJB_PHI* modes — COEFF_FILE stores Phi_xi1 = -log q_xi1 ----
 *
 *   On-the-fly xi1→xi2 conversion (soc.pdf Sec. 6.1–6.2):
 *     q_xi2   = xi2 + [(1-2*xi2)/(1-2*xi1)] * (e^{-Phi_xi1} - xi1)
 *     -∇log q_xi2 = [(1-2*xi2)/(1-2*xi1)] * e^{-Phi_xi1} * ∇Phi_xi1 / q_xi2
 *
 *   MODE=HJB_PHI (full Doob drift, with optional xi conversion):
 *     force_i = -2*kBT * [(1-2*xi2)/(1-2*xi1)] * e^{-Phi_xi1}/q_xi2 * dPhi_xi1/dCV_i
 *     V_bias  = -2*kBT * log q_xi2
 *
 *   MODE=HJB_PHI_HALF (Picard modified drift):
 *     force_i = -kBT * [(1-2*xi2)/(1-2*xi1)] * e^{-Phi_xi1}/q_xi2 * dPhi_xi1/dCV_i
 *     V_bias  = -kBT * log q_xi2
 *
 *   When XI2 is omitted xi2=xi1, reducing to force_i = -factor*kBT * dPhi_xi1/dCV_i.
 *
 * Both are represented as cluster-product (rank-2) basis expansions using either
 * Fourier functions (default) or localized periodic Gaussian kernels (KERNEL_BASIS).
 * Coefficients and cluster indices are read from COEFF_FILE.
 */

#include "Bias.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Tools.h"
#include "ttsketch/BasisFunc.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace PLMD::bias;
using namespace PLMD::ttsketch;

namespace PLMD {
namespace tpp {

class TPP : public Bias {
private:
  // ---- kBT ----
  double kbt_;

  // ---- Drift mode ----
  enum class Mode { NEG_LOG_Q, NEG_LOG_Q_HALF, HJB_PHI, HJB_PHI_HALF };
  Mode mode_;

  // ---- Phi basis expansion ----
  unsigned d_;
  unsigned nbasis_;                             // basis functions per dimension
  bool kernel_;                                 // true: Gaussian kernel basis; false: Fourier
  vector<BasisFunc> basis_;
  vector<pair<unsigned,unsigned>> clusters_;    // (k1, k2): 0-based CV dimension indices
  vector<double> coeffs_;                       // length = nclusters * nbasis_^2

  // ---- Coverage check ----
  bool hasSrc_;
  bool doCoverageCheck_;
  vector<vector<double>> src_;
  double coverage_radius2_;

  // ---- Region/basin geometry — identical data structures to Committor.cpp ----
  struct Region {
    enum Type { RECT, SPHERE, ELLIPSE } type;
    vector<double> lower, upper;        // RECT
    vector<double> center, semi_axes;   // SPHERE / ELLIPSE
  };
  vector<Region> regions_;
  vector<vector<unsigned>> basinRegions_;   // basinRegions_[b] = 0-based region indices for basin b
  unsigned nbasins_;

  // ---- Periodicity cache — identical to Committor.cpp ----
  vector<bool>   argIsPeriodic_;
  vector<double> argPeriod_;

  // ---- xi regularization ----
  double xi1_;   // level used to compute Phi = -log q_xi1 in COEFF_FILE
  double xi2_;   // output level for NEG_LOG_Q* (on-the-fly conversion); default = xi1_

  // ---- helpers ----
  double periodicDiff(double x, double c, unsigned dim) const;
  bool   regionContains(const Region& reg, const vector<double>& x) const;
  int    whichBasin(const vector<double>& x) const;
  // Evaluates the Fourier cluster-product basis expansion and its gradient.
  // Returns the raw expansion value; interpretation (q or Phi) depends on MODE.
  double evalExpansion(const vector<double>& x, vector<double>& grad) const;

public:
  static void registerKeywords(Keywords& keys);
  explicit TPP(const ActionOptions& ao);
  void calculate() override;
};

PLUMED_REGISTER_ACTION(TPP, "TPP")

// ---------------------------------------------------------------------------
void TPP::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);

  keys.add("optional", "TEMP",
           "System temperature in energy units. Required if the MD engine does not pass temperature to PLUMED.");

  keys.addFlag("KERNEL_BASIS", false,
               "Use a localized periodic Gaussian kernel basis instead of the Fourier basis. "
               "Basis function 1 is the constant 1; functions 2..NBASIS are Gaussians centered on a "
               "uniform grid over the domain, summed over periodic images k=-1,0,1.");
  keys.add("optional", "KERNEL_DX",
           "Bandwidth (sigma) of each Gaussian kernel, one value per CV dimension. "
           "Defaults to the grid spacing (domain width / (NBASIS-1)) when not specified or set to 0. "
           "Only used when KERNEL_BASIS is set.");

  keys.add("optional", "MODE",
           "Drift mode. NEG_LOG_Q (default) and NEG_LOG_Q_HALF: COEFF_FILE holds the committor q; "
           "symmetric xi regularization applied on-the-fly. "
           "HJB_PHI and HJB_PHI_HALF: COEFF_FILE holds Phi_xi1 = -log q_xi1; "
           "optional on-the-fly xi1→xi2 conversion via XI2. "
           "Full-factor (NEG_LOG_Q, HJB_PHI) gives the Doob drift; "
           "half-factor (NEG_LOG_Q_HALF, HJB_PHI_HALF) gives the Picard modified drift.");

  // Domain bounds and basis size
  keys.add("compulsory", "DOMAIN_LL",
           "Lower bounds of the domain, one value per CV.");
  keys.add("compulsory", "DOMAIN_UL",
           "Upper bounds of the domain, one value per CV.");
  keys.add("compulsory", "NBASIS",
           "Number of basis functions per CV dimension. "
           "For Fourier basis: NBASIS functions (1 constant + pairs of cos/sin). "
           "For kernel basis (KERNEL_BASIS): NBASIS functions (1 constant + NBASIS-1 Gaussians). "
           "COEFF_FILE must contain exactly nclusters * NBASIS^2 values.");

  keys.add("compulsory", "CLUSTER1",
           "First CV dimension index (1-based) for each cluster. Length must equal the number of clusters and match CLUSTER2.");
  keys.add("compulsory", "CLUSTER2",
           "Second CV dimension index (1-based) for each cluster. k1 and k2 may be equal. Length must match CLUSTER1.");

  keys.add("compulsory", "COEFF_FILE", "COEFFS",
           "File containing basis expansion coefficients, one value per line. "
           "For NEG_LOG_Q* modes these are coefficients for q (the committor). "
           "For HJB_PHI* modes these are coefficients for Phi_xi1 = -log q_xi1. "
           "Total lines must equal nclusters * NBASIS^2. "
           "Must be consistent with KERNEL_BASIS: Fourier coefficients are not interchangeable "
           "with kernel basis coefficients.");

  // Source states for coverage check
  keys.add("optional", "SOURCE_FILE",
           "File containing source-state CV coordinates, one row per state. If omitted, the bias is applied everywhere.");
  keys.add("optional", "COVERAGE_RADIUS",
           "Coverage radius for the nearest-source-state check. Defaults to 2 * sqrt(domain_volume / Nsrc).");
  keys.addFlag("COVERAGE_CHECK", false,
               "Enable nearest-source-state coverage check. When set, the bias is zeroed outside the region covered by SOURCE_FILE points.");

  // Region geometry — identical keywords to Committor.cpp
  keys.add("numbered", "REGION_LL",
           "Lower limits for rectangular region #. "
           "For periodic CVs, REGION_LL > REGION_UL indicates a wrap-around interval.");
  keys.add("numbered", "REGION_UL",    "Upper limits for rectangular region #.");
  keys.add("numbered", "REGION_CENTER","Center of spherical or elliptical region #.");
  keys.add("numbered", "REGION_RADIUS","Radius of spherical region # (single value).");
  keys.add("numbered", "REGION_AXES",  "Semi-axis lengths of elliptical region # (one per CV).");

  // Basin keywords — identical to Committor.cpp
  keys.add("numbered", "BASIN",
           "Comma-separated list of 1-based region indices forming basin #. By convention BASIN1 is the source (A) and BASIN2 is the target (B).");

  // Legacy rectangular shorthand — identical to Committor.cpp
  keys.add("numbered", "BASIN_LL",
           "Lower limits for basin # (legacy rectangular shorthand, one value per CV).");
  keys.add("numbered", "BASIN_UL",
           "Upper limits for basin # (legacy rectangular shorthand, one value per CV).");

  // xi regularization
  keys.add("compulsory", "XI", "0.01",
           "Regularization parameter xi for symmetric regularization q_xi = xi + (1-2*xi)*q, "
           "so q_xi in [xi, 1-xi]. Must be in (0, 0.5). "
           "For NEG_LOG_Q* modes: xi applied directly to q from COEFF_FILE. "
           "For HJB_PHI* modes: xi1, the level at which Phi_xi1 in COEFF_FILE was computed.");
  keys.add("optional", "XI2",
           "Output regularization xi2 for HJB_PHI and HJB_PHI_HALF modes only: "
           "on-the-fly conversion from xi1 to xi2 via "
           "q_xi2 = xi2 + [(1-2*xi2)/(1-2*xi1)]*(exp(-Phi_xi1) - xi1). "
           "Defaults to XI (no conversion, i.e. xi2=xi1). Must satisfy 0 < xi2 <= xi1. "
           "Ignored for NEG_LOG_Q* modes.");
}

// ---------------------------------------------------------------------------
TPP::TPP(const ActionOptions& ao)
  : Action(ao),
    Bias(ao),
    kbt_(getkBT()),
    mode_(Mode::NEG_LOG_Q),
    d_(getNumberOfArguments()),
    nbasis_(0),
    kernel_(false),
    hasSrc_(false),
    doCoverageCheck_(false),
    coverage_radius2_(-1.0),
    nbasins_(0),
    xi1_(0.01),
    xi2_(0.01)
{
  if(d_ == 0) error("TPP: ARG must specify at least one CV");
  if(kbt_ == 0.0)
    error("TPP: temperature is required. Set TEMP or ensure the MD engine passes it to PLUMED.");

  // ---- Mode ----
  string mode_str = "NEG_LOG_Q";
  parse("MODE", mode_str);
  if     (mode_str == "NEG_LOG_Q")      mode_ = Mode::NEG_LOG_Q;
  else if(mode_str == "NEG_LOG_Q_HALF") mode_ = Mode::NEG_LOG_Q_HALF;
  else if(mode_str == "HJB_PHI")        mode_ = Mode::HJB_PHI;
  else if(mode_str == "HJB_PHI_HALF")   mode_ = Mode::HJB_PHI_HALF;
  else error("TPP: unknown MODE '" + mode_str +
             "'. Valid: NEG_LOG_Q, NEG_LOG_Q_HALF, HJB_PHI, HJB_PHI_HALF");

  // ---- Periodicity cache — identical to Committor.cpp ----
  argIsPeriodic_.resize(d_, false);
  argPeriod_.resize(d_, 0.0);
  for(unsigned i = 0; i < d_; ++i) {
    if(getPntrToArgument(i)->isPeriodic()) {
      argIsPeriodic_[i] = true;
      string smin, smax;
      getPntrToArgument(i)->getDomain(smin, smax);
      double dmin, dmax;
      Tools::convert(smin, dmin);
      Tools::convert(smax, dmax);
      argPeriod_[i] = dmax - dmin;
      log.printf("  CV %u is periodic with period %f\n", i, argPeriod_[i]);
    }
  }

  // ---- Kernel basis flag and bandwidth ----
  parseFlag("KERNEL_BASIS", kernel_);
  vector<double> kernel_dx;
  parseVector("KERNEL_DX", kernel_dx);
  if(kernel_dx.empty()) {
    kernel_dx.assign(d_, 0.0);   // 0 → BasisFunc uses grid spacing
  }
  if(kernel_dx.size() != d_)
    error("TPP: KERNEL_DX must have one entry per ARG (or be omitted entirely)");
  if(kernel_) {
    for(unsigned i = 0; i < d_; ++i)
      if(kernel_dx[i] < 0.0)
        error("TPP: KERNEL_DX values must be non-negative (0 = use grid spacing)");
  }

  // ---- Domain and basis functions ----
  vector<double> domain_ll, domain_ul;
  parseVector("DOMAIN_LL", domain_ll);
  parseVector("DOMAIN_UL", domain_ul);
  if(domain_ll.size() != d_) error("TPP: DOMAIN_LL must have one entry per ARG");
  if(domain_ul.size() != d_) error("TPP: DOMAIN_UL must have one entry per ARG");
  for(unsigned i = 0; i < d_; ++i)
    if(domain_ul[i] <= domain_ll[i])
      error("TPP: DOMAIN_UL[" + to_string(i) + "] must be strictly greater than DOMAIN_LL[" + to_string(i) + "]");

  int nbasis_in = 0;
  parse("NBASIS", nbasis_in);
  if(nbasis_in < 1) error("TPP: NBASIS must be >= 1");
  if(kernel_ && nbasis_in < 2) error("TPP: NBASIS must be >= 2 for KERNEL_BASIS (need at least one Gaussian center)");
  nbasis_ = static_cast<unsigned>(nbasis_in);
  log.printf("  NBASIS = %u %s basis functions per dimension\n",
             nbasis_, kernel_ ? "Gaussian kernel" : "Fourier");
  if(kernel_) {
    for(unsigned i = 0; i < d_; ++i) {
      double spacing = (domain_ul[i] - domain_ll[i]) / (nbasis_in - 1);
      double last_center = domain_ll[i] + (nbasis_in - 2) * spacing;
      double dx_used = (kernel_dx[i] > 0.0) ? kernel_dx[i] : spacing;
      log.printf("    dim %u: %d centers in [%f, %f], sigma=%f\n",
                 i, nbasis_in - 1, domain_ll[i], last_center, dx_used);
    }
  }

  basis_.reserve(d_);
  for(unsigned i = 0; i < d_; ++i)
    basis_.emplace_back(make_pair(domain_ll[i], domain_ul[i]),
                        nbasis_in, /*w=*/0.0, kernel_, kernel_dx[i]);

  // ---- Cluster dimension pairs ----
  vector<double> c1vec, c2vec;
  parseVector("CLUSTER1", c1vec);
  parseVector("CLUSTER2", c2vec);
  if(c1vec.size() != c2vec.size())
    error("TPP: CLUSTER1 and CLUSTER2 must have the same number of values");
  if(c1vec.empty()) error("TPP: CLUSTER1/CLUSTER2 must be non-empty");
  for(unsigned ic = 0; ic < c1vec.size(); ++ic) {
    int k1 = static_cast<int>(std::round(c1vec[ic]));
    int k2 = static_cast<int>(std::round(c2vec[ic]));
    if(k1 < 1 || k1 > static_cast<int>(d_))
      error("TPP: CLUSTER1[" + to_string(ic) + "]=" + to_string(k1) +
            " is out of range [1," + to_string(d_) + "]");
    if(k2 < 1 || k2 > static_cast<int>(d_))
      error("TPP: CLUSTER2[" + to_string(ic) + "]=" + to_string(k2) +
            " is out of range [1," + to_string(d_) + "]");
    clusters_.emplace_back(static_cast<unsigned>(k1 - 1),
                           static_cast<unsigned>(k2 - 1));   // store 0-based
  }
  log.printf("  %zu cluster(s):\n", clusters_.size());
  for(unsigned ic = 0; ic < clusters_.size(); ++ic)
    log.printf("    cluster %u: k1=%u, k2=%u\n", ic + 1,
               clusters_[ic].first + 1, clusters_[ic].second + 1);

  // ---- Load coefficient file ----
  const unsigned expected_ncoeffs = static_cast<unsigned>(clusters_.size()) * nbasis_ * nbasis_;
  string coeff_file;
  parse("COEFF_FILE", coeff_file);
  {
    ifstream fin(coeff_file);
    if(!fin) error("TPP: cannot open COEFF_FILE '" + coeff_file + "'");
    string line;
    int linenum = 0;
    while(getline(fin, line)) {
      ++linenum;
      auto pos = line.find('#');
      if(pos != string::npos) line = line.substr(0, pos);
      if(line.find_first_not_of(" \t\r\n") == string::npos) continue;
      istringstream iss(line);
      double c;
      if(!(iss >> c))
        error("TPP: could not parse coefficient in COEFF_FILE line " + to_string(linenum));
      coeffs_.push_back(c);
    }
  }
  if(coeffs_.size() != expected_ncoeffs)
    error("TPP: COEFF_FILE has " + to_string(coeffs_.size()) + " values but expected " +
          to_string(expected_ncoeffs) + " (nclusters=" + to_string(clusters_.size()) +
          " * NBASIS^2=" + to_string(nbasis_ * nbasis_) + ")");
  log.printf("  Loaded %u coefficients from '%s' (%zu clusters x %u^2 basis pairs)\n",
             expected_ncoeffs, coeff_file.c_str(), clusters_.size(), nbasis_);

  // ---- Optional source file for coverage check ----
  string src_file;
  parse("SOURCE_FILE", src_file);
  if(!src_file.empty()) {
    hasSrc_ = true;
    ifstream fin(src_file);
    if(!fin) error("TPP: cannot open SOURCE_FILE '" + src_file + "'");
    string line;
    int linenum = 0;
    while(getline(fin, line)) {
      ++linenum;
      auto pos = line.find('#');
      if(pos != string::npos) line = line.substr(0, pos);
      if(line.find_first_not_of(" \t\r\n") == string::npos) continue;
      istringstream iss(line);
      vector<double> row(d_);
      for(unsigned i = 0; i < d_; ++i)
        if(!(iss >> row[i]))
          error("TPP: bad source state in SOURCE_FILE line " + to_string(linenum));
      src_.push_back(row);
    }
    if(src_.empty()) error("TPP: SOURCE_FILE '" + src_file + "' contains no valid entries");
    log.printf("  Loaded %zu source states from '%s'\n", src_.size(), src_file.c_str());

    double cov_radius = -1.0;
    parse("COVERAGE_RADIUS", cov_radius);
    if(cov_radius > 0.0) {
      coverage_radius2_ = cov_radius * cov_radius;
    } else {
      double domain_vol = 1.0;
      for(unsigned i = 0; i < d_; ++i) domain_vol *= (domain_ul[i] - domain_ll[i]);
      coverage_radius2_ = 4.0 * domain_vol / static_cast<double>(src_.size());
    }
    log.printf("  Coverage radius = %f (radius^2 = %f)\n",
               sqrt(coverage_radius2_), coverage_radius2_);
  } else {
    log.printf("  No SOURCE_FILE given: bias gradient applied everywhere\n");
  }

  parseFlag("COVERAGE_CHECK", doCoverageCheck_);
  if(hasSrc_)
    log.printf("  Coverage check: %s\n", doCoverageCheck_ ? "enabled" : "disabled (bias applied everywhere)");

  // ---- xi regularization ----
  parse("XI", xi1_);
  if(xi1_ <= 0.0 || xi1_ >= 0.5)
    error("TPP: XI must be in (0, 0.5)");
  xi2_ = xi1_;  // default: no conversion
  parse("XI2", xi2_);
  if(xi2_ <= 0.0 || xi2_ > xi1_)
    error("TPP: XI2 must satisfy 0 < XI2 <= XI");
  if(mode_ == Mode::NEG_LOG_Q || mode_ == Mode::NEG_LOG_Q_HALF) {
    if(xi2_ != xi1_)
      log.printf("  WARNING: XI2 is ignored in NEG_LOG_Q / NEG_LOG_Q_HALF modes\n");
    log.printf("  xi = %e  (symmetric regularization applied to q)\n", xi1_);
  } else {
    log.printf("  xi1 (COEFF_FILE level) = %e\n", xi1_);
    log.printf("  xi2 (output level)     = %e%s\n", xi2_,
               (xi2_ < xi1_) ? "  (on-the-fly conversion active)" : "  (no conversion)");
  }

  // =====================================================================
  //  Region/basin parsing — identical to Committor.cpp.
  //  Try region mode first (REGION_* + BASIN#); fall back to legacy
  //  (BASIN_LL# / BASIN_UL#) when no REGION_* keywords are present.
  // =====================================================================

  for(unsigned r = 1;; ++r) {
    vector<double> rll, rul, rcenter, rradius, raxes;
    parseNumberedVector("REGION_LL",     r, rll);
    parseNumberedVector("REGION_UL",     r, rul);
    parseNumberedVector("REGION_CENTER", r, rcenter);
    parseNumberedVector("REGION_RADIUS", r, rradius);
    parseNumberedVector("REGION_AXES",   r, raxes);
    if(rll.empty() && rul.empty() && rcenter.empty() && rradius.empty() && raxes.empty()) break;

    Region reg;

    // --- Rectangular ---
    if(!rll.empty() || !rul.empty()) {
      if(rll.empty() || rul.empty())
        error("TPP: REGION_LL and REGION_UL must both be given for rectangular region " + to_string(r));
      if(rll.size() != d_)
        error("TPP: REGION_LL" + to_string(r) + " has wrong number of values (expected " + to_string(d_) + ")");
      if(rul.size() != d_)
        error("TPP: REGION_UL" + to_string(r) + " has wrong number of values (expected " + to_string(d_) + ")");
      if(!rcenter.empty() || !rradius.empty() || !raxes.empty())
        error("TPP: region " + to_string(r) + " mixes RECT keywords (LL/UL) with SPHERE/ELLIPSE keywords (CENTER/RADIUS/AXES)");
      for(unsigned i = 0; i < d_; ++i)
        if(rll[i] > rul[i] && !argIsPeriodic_[i])
          error("TPP: REGION_UL must be >= REGION_LL for non-periodic CV dimension " +
                to_string(i) + " in region " + to_string(r));
      reg.type  = Region::RECT;
      reg.lower = rll;
      reg.upper = rul;

    // --- Spherical ---
    } else if(!rcenter.empty() && !rradius.empty() && raxes.empty()) {
      if(rcenter.size() != d_)
        error("TPP: REGION_CENTER" + to_string(r) + " has wrong number of values (expected " + to_string(d_) + ")");
      if(rradius.size() != 1)
        error("TPP: REGION_RADIUS" + to_string(r) + " must be a single value");
      reg.type = Region::SPHERE;
      reg.center = rcenter;
      reg.semi_axes.assign(d_, rradius[0]);

    // --- Elliptical ---
    } else if(!rcenter.empty() && !raxes.empty() && rradius.empty()) {
      if(rcenter.size() != d_)
        error("TPP: REGION_CENTER" + to_string(r) + " has wrong number of values (expected " + to_string(d_) + ")");
      if(raxes.size() != d_)
        error("TPP: REGION_AXES" + to_string(r) + " has wrong number of values (expected " + to_string(d_) + ")");
      for(unsigned i = 0; i < d_; ++i)
        if(raxes[i] <= 0.0)
          error("TPP: REGION_AXES values must be positive in region " + to_string(r));
      reg.type      = Region::ELLIPSE;
      reg.center    = rcenter;
      reg.semi_axes = raxes;

    } else {
      error("TPP: could not determine region type for region " + to_string(r) +
            ". Use REGION_LL+REGION_UL (rect), REGION_CENTER+REGION_RADIUS (sphere), "
            "or REGION_CENTER+REGION_AXES (ellipse).");
    }
    regions_.push_back(reg);
  }

  bool hasRegions = !regions_.empty();

  if(hasRegions) {
    // --- Parse BASIN# keywords — identical to Committor.cpp ---
    for(unsigned b = 1;; ++b) {
      vector<double> bvec_d;
      parseNumberedVector("BASIN", b, bvec_d);
      if(bvec_d.empty()) break;
      vector<unsigned> bvec;
      for(unsigned k = 0; k < bvec_d.size(); ++k) {
        int idx = static_cast<int>(std::round(bvec_d[k]));
        if(idx < 1 || idx > static_cast<int>(regions_.size()))
          error("TPP: BASIN" + to_string(b) + " references region " + to_string(idx) +
                " which is out of range [1," + to_string(regions_.size()) + "]");
        bvec.push_back(static_cast<unsigned>(idx - 1));  // 0-based
      }
      basinRegions_.push_back(bvec);
      nbasins_ = b;
    }
    if(nbasins_ == 0)
      error("TPP: REGION_* keywords found but no BASIN# keywords to assign regions to basins");
  }

  // ---- Legacy mode — identical to Committor.cpp ----
  bool hasLegacy = false;
  if(!hasRegions) {
    for(unsigned b = 1;; ++b) {
      vector<double> tmpl, tmpu;
      parseNumberedVector("BASIN_LL", b, tmpl);
      parseNumberedVector("BASIN_UL", b, tmpu);
      if(tmpl.empty() && tmpu.empty()) break;
      if(tmpl.size() != d_)
        error("TPP: wrong number of values for BASIN_LL: should equal the number of arguments");
      if(tmpu.size() != d_)
        error("TPP: wrong number of values for BASIN_UL: should equal the number of arguments");
      for(unsigned i = 0; i < d_; ++i)
        if(tmpl[i] > tmpu[i] && !argIsPeriodic_[i])
          error("TPP: BASIN_UL must be >= BASIN_LL for non-periodic CVs");
      Region reg;
      reg.type  = Region::RECT;
      reg.lower = tmpl;
      reg.upper = tmpu;
      regions_.push_back(reg);
      basinRegions_.push_back(vector<unsigned>(1, static_cast<unsigned>(regions_.size() - 1)));
      nbasins_ = b;
    }
    hasLegacy = (nbasins_ > 0);
  }

  if(!hasLegacy && !hasRegions)
    error("TPP: no basins defined. Use BASIN_LL/BASIN_UL (legacy) or REGION_*/BASIN# (region mode).");

  // ---- Log region/basin configuration ----
  log.printf("  Number of regions: %u\n", static_cast<unsigned>(regions_.size()));
  for(unsigned r = 0; r < regions_.size(); ++r) {
    const Region& reg = regions_[r];
    switch(reg.type) {
    case Region::RECT:
      log.printf("  Region %u: RECT\n", r + 1);
      for(unsigned i = 0; i < d_; ++i) {
        if(reg.lower[i] > reg.upper[i])
          log.printf("    dim %u: [%f, domain_max] U [domain_min, %f]  (wrap-around)\n",
                     i, reg.lower[i], reg.upper[i]);
        else
          log.printf("    dim %u: [%f, %f]\n", i, reg.lower[i], reg.upper[i]);
      }
      break;
    case Region::SPHERE:
      log.printf("  Region %u: SPHERE  center=(", r + 1);
      for(unsigned i = 0; i < d_; ++i) log.printf("%s%f", i ? "," : "", reg.center[i]);
      log.printf(")  radius=%f\n", reg.semi_axes[0]);
      break;
    case Region::ELLIPSE:
      log.printf("  Region %u: ELLIPSE  center=(", r + 1);
      for(unsigned i = 0; i < d_; ++i) log.printf("%s%f", i ? "," : "", reg.center[i]);
      log.printf(")  semi_axes=(");
      for(unsigned i = 0; i < d_; ++i) log.printf("%s%f", i ? "," : "", reg.semi_axes[i]);
      log.printf(")\n");
      break;
    }
  }
  log.printf("  Number of basins: %u  (convention: BASIN1=A, BASIN2=B)\n", nbasins_);
  for(unsigned b = 0; b < nbasins_; ++b) {
    log.printf("  Basin %u = union of region(s):", b + 1);
    for(unsigned k = 0; k < basinRegions_[b].size(); ++k)
      log.printf(" %u", basinRegions_[b][k] + 1);
    log.printf("\n");
  }

  log.printf("  kBT = %f\n", kbt_);
  {
    const char* names[] = {"NEG_LOG_Q", "NEG_LOG_Q_HALF", "HJB_PHI", "HJB_PHI_HALF"};
    log.printf("  Mode: %s\n", names[static_cast<int>(mode_)]);
  }
  switch(mode_) {
  case Mode::NEG_LOG_Q:
    log.printf("  Drift: 2D*(1-2xi)*grad_q/q_xi  (Doob, q-expansion)\n");
    break;
  case Mode::NEG_LOG_Q_HALF:
    log.printf("  Drift: D*(1-2xi)*grad_q/q_xi   (Picard, q-expansion)\n");
    break;
  case Mode::HJB_PHI:
    log.printf("  Drift: -2D*fac*grad_Phi_xi1    (Doob, Phi-expansion)\n");
    break;
  case Mode::HJB_PHI_HALF:
    log.printf("  Drift: -D*fac*grad_Phi_xi1     (Picard, Phi-expansion)\n");
    break;
  }

  checkRead();
}

// ---------------------------------------------------------------------------
// Minimum-image signed displacement — identical to Committor.cpp.
// ---------------------------------------------------------------------------
double TPP::periodicDiff(double x, double c, unsigned dim) const {
  double dx = x - c;
  if(argIsPeriodic_[dim]) {
    const double period = argPeriod_[dim];
    const double half   = 0.5 * period;
    while(dx >  half) dx -= period;
    while(dx < -half) dx += period;
  }
  return dx;
}

// ---------------------------------------------------------------------------
// Test whether x lies inside a region — identical logic to Committor.cpp.
// ---------------------------------------------------------------------------
bool TPP::regionContains(const Region& reg, const vector<double>& x) const {
  switch(reg.type) {

  case Region::RECT:
    for(unsigned i = 0; i < d_; ++i) {
      if(argIsPeriodic_[i]) {
        if(reg.lower[i] <= reg.upper[i]) {
          double mid  = 0.5 * (reg.lower[i] + reg.upper[i]);
          double half = 0.5 * (reg.upper[i] - reg.lower[i]);
          if(std::abs(periodicDiff(x[i], mid, i)) >= half) return false;
        } else {
          double gap_mid  = 0.5 * (reg.upper[i] + reg.lower[i]);
          double gap_half = 0.5 * (reg.lower[i] - reg.upper[i]);
          if(std::abs(periodicDiff(x[i], gap_mid, i)) < gap_half) return false;
        }
      } else {
        if(x[i] <= reg.lower[i] || x[i] >= reg.upper[i]) return false;
      }
    }
    return true;

  case Region::SPHERE: {
    double r2 = 0.0;
    const double R2 = reg.semi_axes[0] * reg.semi_axes[0];
    for(unsigned i = 0; i < d_; ++i) {
      double dx = periodicDiff(x[i], reg.center[i], i);
      r2 += dx * dx;
      if(r2 > R2) return false;
    }
    return true;
  }

  case Region::ELLIPSE: {
    double s = 0.0;
    for(unsigned i = 0; i < d_; ++i) {
      double dx = periodicDiff(x[i], reg.center[i], i);
      double a  = reg.semi_axes[i];
      s += (dx * dx) / (a * a);
      if(s > 1.0) return false;
    }
    return true;
  }

  }
  return false;
}

// ---------------------------------------------------------------------------
// Returns the 0-based index of the first basin containing x, or -1 if none.
// ---------------------------------------------------------------------------
int TPP::whichBasin(const vector<double>& x) const {
  for(unsigned b = 0; b < nbasins_; ++b)
    for(unsigned k = 0; k < basinRegions_[b].size(); ++k)
      if(regionContains(regions_[basinRegions_[b][k]], x))
        return static_cast<int>(b);
  return -1;
}

// ---------------------------------------------------------------------------
// Evaluate the cluster-product Fourier basis expansion and its gradient.
//
// f(x) = sum_{ic,nk1,nk2} c[j] * psi_{nk1}(x[k1]) * psi_{nk2}(x[k2])
//
// Gradient at dimension i (product rule):
//   k1==i, k2!=i:  dpsi_{nk1}(x[i]) * psi_{nk2}(x[k2])
//   k1!=i, k2==i:  psi_{nk1}(x[k1]) * dpsi_{nk2}(x[i])
//   k1==i, k2==i:  dpsi_{nk1}(x[i])*psi_{nk2}(x[i]) + psi_{nk1}(x[i])*dpsi_{nk2}(x[i])
//   otherwise:     0
// ---------------------------------------------------------------------------
double TPP::evalExpansion(const vector<double>& x, vector<double>& grad) const {
  fill(grad.begin(), grad.end(), 0.0);
  double val = 0.0;
  const unsigned nclusters = static_cast<unsigned>(clusters_.size());

  for(unsigned ic = 0; ic < nclusters; ++ic) {
    const unsigned k1 = clusters_[ic].first;
    const unsigned k2 = clusters_[ic].second;

    for(unsigned nk1 = 1; nk1 <= nbasis_; ++nk1) {
      const double psi1  = basis_[k1](x[k1],  static_cast<int>(nk1), /*conv=*/false);
      const double dpsi1 = basis_[k1].grad(x[k1], static_cast<int>(nk1), /*conv=*/false);

      for(unsigned nk2 = 1; nk2 <= nbasis_; ++nk2) {
        const unsigned j = ic * nbasis_ * nbasis_ + (nk1 - 1) * nbasis_ + (nk2 - 1);
        const double c = coeffs_[j];

        const double psi2  = basis_[k2](x[k2],  static_cast<int>(nk2), /*conv=*/false);
        const double dpsi2 = basis_[k2].grad(x[k2], static_cast<int>(nk2), /*conv=*/false);

        val += c * psi1 * psi2;

        if(k1 == k2) {
          grad[k1] += c * (dpsi1 * psi2 + psi1 * dpsi2);
        } else {
          grad[k1] += c * dpsi1 * psi2;
          grad[k2] += c * psi1  * dpsi2;
        }
      }
    }
  }
  return val;
}

// ---------------------------------------------------------------------------
// Main calculation — called at every step (STRIDE handled by Bias base).
// ---------------------------------------------------------------------------
void TPP::calculate() {
  vector<double> x(d_);
  for(unsigned i = 0; i < d_; ++i) x[i] = getArgument(i);

  // ---- Check basin membership and stop ----
  int basin = whichBasin(x);
  if(basin >= 0) {
    for(unsigned i = 0; i < d_; ++i) setOutputForce(i, 0.0);
    setBias(0.0);
    if(basin == 0) {
      log.printf("TPP: trajectory returned to source basin A (BASIN1) — stopping\n");
      plumed.stop();
    } else {
      log.printf("TPP: trajectory reached target basin %d (BASIN%d) — success\n",
                 basin + 1, basin + 1);
      plumed.stop();
    }
    return;
  }

  // ---- Coverage check ----
  // We only need to know whether *any* source point is within coverage_radius2_,
  // so we exit the outer loop as soon as one is found (O(1) in the covered case).
  // The inner loop breaks early when the partial squared distance already exceeds
  // coverage_radius2_ (safe because adding more dimensions can only increase d2).
  if(hasSrc_ && doCoverageCheck_) {
    bool covered = false;
    for(const auto& s : src_) {
      double d2 = 0.0;
      for(unsigned i = 0; i < d_; ++i) {
        double dx = periodicDiff(x[i], s[i], i);
        d2 += dx * dx;
        if(d2 >= coverage_radius2_) break;
      }
      if(d2 < coverage_radius2_) { covered = true; break; }
    }
    if(!covered) {
      for(unsigned i = 0; i < d_; ++i) setOutputForce(i, 0.0);
      setBias(0.0);
      return;
    }
  }

  // ---- Compute force and bias ----
  vector<double> grad(d_, 0.0);

  if(mode_ == Mode::NEG_LOG_Q || mode_ == Mode::NEG_LOG_Q_HALF) {
    // COEFF_FILE stores q (the committor).
    // Symmetric xi regularization: q_xi = xi + (1-2*xi)*q_raw, q_xi in [xi, 1-xi].
    // grad log q_xi = (1-2*xi)*grad_q / q_xi
    // force_i = factor * kBT * (1-2*xi) * dq/dCV_i / q_xi
    double q_raw = evalExpansion(x, grad);

    // Clip Fourier artifacts to [0, 1]; zero gradient at the boundary
    if(q_raw <= 0.0) {
      q_raw = 0.0;
      fill(grad.begin(), grad.end(), 0.0);
    } else if(q_raw >= 1.0) {
      q_raw = 1.0;
      fill(grad.begin(), grad.end(), 0.0);
    }

    const double alpha  = 1.0 - 2.0 * xi1_;          // q_xi = xi1 + alpha*q_raw
    const double q_xi   = xi1_ + alpha * q_raw;        // in [xi1_, 1-xi1_] after clipping
    const double factor = (mode_ == Mode::NEG_LOG_Q) ? 2.0 : 1.0;

    for(unsigned i = 0; i < d_; ++i)
      setOutputForce(i, factor * kbt_ * alpha * grad[i] / q_xi);
    setBias(-factor * kbt_ * std::log(q_xi));

  } else {
    // COEFF_FILE stores Phi_xi1 = -log q_xi1.
    // On-the-fly conversion to xi2 (soc.pdf Sec. 6, symmetric regularization):
    //   q_xi2 = xi2 + [(1-2*xi2)/(1-2*xi1)] * (e^{-Phi_xi1} - xi1)
    //   -∇ log q_xi2 = [(1-2*xi2)/(1-2*xi1)] * e^{-Phi_xi1} * ∇Phi_xi1 / q_xi2
    //   force_i = factor * kBT * ∇ log q_xi2
    //           = -factor * kBT * [(1-2*xi2)/(1-2*xi1)] * e^{-Phi_xi1}/q_xi2 * dPhi_xi1/dCV_i
    double phi = evalExpansion(x, grad);

    // Clip phi to valid range [-log(1-xi1), -log(xi1)] (B = phi_lo, A = phi_hi)
    const double phi_lo = -std::log(1.0 - xi1_);
    const double phi_hi = -std::log(xi1_);
    if(phi < phi_lo) {
      phi = phi_lo;
      fill(grad.begin(), grad.end(), 0.0);
    } else if(phi > phi_hi) {
      phi = phi_hi;
      fill(grad.begin(), grad.end(), 0.0);
    }

    const double q_xi1  = std::exp(-phi);
    const double scale  = (1.0 - 2.0*xi2_) / (1.0 - 2.0*xi1_);
    double       q_xi2  = xi2_ + scale * (q_xi1 - xi1_);
    if(q_xi2 < xi2_) q_xi2 = xi2_;   // guard against floating-point rounding below xi2

    const double fac    = scale * q_xi1 / q_xi2;   // = (1-2*xi2)/(1-2*xi1) * e^{-phi} / q_xi2
    const double factor = (mode_ == Mode::HJB_PHI) ? 2.0 : 1.0;

    for(unsigned i = 0; i < d_; ++i)
      setOutputForce(i, -factor * kbt_ * fac * grad[i]);
    setBias(-factor * kbt_ * std::log(q_xi2));
  }
}

} // namespace tpp
} // namespace PLMD
