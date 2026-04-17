/* Transition Path Process (TPP) bias.
 *
 * Implements the Doob h-transform of overdamped Langevin with h = q (the committor):
 *
 *   dX = [-grad V(X) / gamma + 2D grad q(X) / q(X)] dt + sqrt(2D) dW
 *
 * The extra drift 2D grad q / q corresponds to the bias potential V_bias = -2kT log q,
 * giving force  F_i = +2kT dq/d(CV_i) / q  on each collective variable.
 *
 * The "bias" component output is -2kT log(q).  For importance-sampling reweighting,
 * the instantaneous log weight of a TPP frame relative to unbiased dynamics is
 *   log w = -bias / kT = 2 log q.
 *
 * The committor is represented as a cluster-product (rank-2) basis expansion:
 *   q(x) = sum_k c_k * phi_{n1_k}(x_1) * phi_{n2_k}(x_2) * ... * phi_{nd_k}(x_d)
 * where each phi_n is a 1-D Fourier basis function (from ttsketch::BasisFunc).
 * Coefficients and cluster indices are read from COEFF_FILE.
 *
 * By convention BASIN1 = source basin A and BASIN2 = target basin B, but any
 * number of basins can be defined and the simulation stops when any of them is
 * entered.  Region geometry and periodic-CV handling are identical to
 * src/generic/Committor.cpp.  The legacy BASIN_LL#/BASIN_UL# rectangular shorthand is also
 * supported.
 *
 * The committor is clipped to q_floor outside all basins to prevent division by
 * very small values near the boundary of A.  The bias force is only applied when the current
 * point is "covered" by the source sample cloud; outside that region the force
 * and bias value are both zero.
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

  // ---- Committor basis expansion ----
  unsigned d_;
  unsigned nbasis_;                             // Fourier basis functions per dimension
  vector<BasisFunc> basis_;
  // Each cluster is a pair of 0-based CV dimension indices (k1, k2).
  // The committor term for cluster ic and basis pair (nk1, nk2) is:
  //   c[ic*nbasis_^2 + (nk1-1)*nbasis_ + (nk2-1)] * phi_{nk1}(x[k1]) * phi_{nk2}(x[k2])
  // Nb = nclusters * (nbasis+1)^2.
  vector<pair<unsigned,unsigned>> clusters_;    // (k1, k2): 0-based CV dimension indices
  vector<double> coeffs_;                       // length = nclusters * nbasis_^2

  // ---- Coverage check ----
  bool hasSrc_;
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

  // ---- q clipping ----
  double q_floor_;

  // ---- helpers ----
  double periodicDiff(double x, double c, unsigned dim) const;
  bool   regionContains(const Region& reg, const vector<double>& x) const;
  // Returns the 0-based index of the first basin containing x, or -1 if none.
  int    whichBasin(const vector<double>& x) const;
  double evalCommittor(const vector<double>& x, vector<double>& grad_q) const;

public:
  static void registerKeywords(Keywords& keys);
  explicit TPP(const ActionOptions& ao);
  void calculate() override;
};

PLUMED_REGISTER_ACTION(TPP, "TPP")

// ---------------------------------------------------------------------------
void TPP::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");

  keys.add("optional", "TEMP",
           "System temperature in energy units. Required if the MD engine does not pass temperature to PLUMED.");

  // Domain bounds and basis size
  keys.add("compulsory", "DOMAIN_LL",
           "Lower bounds of the committor domain, one value per CV.");
  keys.add("compulsory", "DOMAIN_UL",
           "Upper bounds of the committor domain, one value per CV.");
  keys.add("compulsory", "NBASIS",
           "Number of Fourier basis functions per CV dimension. COEFF_FILE must contain exactly nclusters * NBASIS^2 values.");

  // Cluster dimension pairs — CLUSTER1 lists the first CV index of each cluster,
  // CLUSTER2 lists the second.  Both must have length nclusters.
  // E.g. for clusters [(1,1),(1,2),(2,2)]: CLUSTER1=1,1,2  CLUSTER2=1,2,2
  keys.add("compulsory", "CLUSTER1",
           "First CV dimension index (1-based) for each cluster, one value per cluster. Length must equal the number of clusters and match CLUSTER2.");
  keys.add("compulsory", "CLUSTER2",
           "Second CV dimension index (1-based) for each cluster, one value per cluster. k1 and k2 may be equal (same CV). Length must match CLUSTER1.");

  // Coefficient file
  keys.add("compulsory", "COEFF_FILE", "QCOEFFS",
           "File containing the committor expansion coefficients, one value per line. Total lines must equal nclusters * NBASIS^2. Ordering: outer loop over clusters (in the order CLUSTER1, CLUSTER2, ...), inner loops over nk1 then nk2, each from 1 to NBASIS.");

  // Source states for coverage check
  keys.add("optional", "SOURCE_FILE",
           "File containing source-state CV coordinates, one row per state. If omitted, the bias is applied everywhere without a coverage check.");
  keys.add("optional", "COVERAGE_RADIUS",
           "Coverage radius for the nearest-source-state check. Defaults to 2 * sqrt(domain_volume / Nsrc).");

  // Region geometry — identical keywords to Committor.cpp
  keys.add("numbered", "REGION_LL",
           "Lower limits for rectangular region #. "
           "For periodic CVs, setting REGION_LL > REGION_UL indicates a wrap-around interval.");
  keys.add("numbered", "REGION_UL",    "Upper limits for rectangular region #.");
  keys.add("numbered", "REGION_CENTER","Center of spherical or elliptical region #.");
  keys.add("numbered", "REGION_RADIUS","Radius of spherical region # (single value).");
  keys.add("numbered", "REGION_AXES",  "Semi-axis lengths of elliptical region # (one per CV).");
  // keys.reset_style("REGION_LL",     "optional");
  // keys.reset_style("REGION_UL",     "optional");
  // keys.reset_style("REGION_CENTER", "optional");
  // keys.reset_style("REGION_RADIUS", "optional");
  // keys.reset_style("REGION_AXES",   "optional");

  // Basin keywords — identical to Committor.cpp
  // By convention: BASIN1 = source basin A, BASIN2 = target basin B.
  keys.add("numbered", "BASIN",
           "Comma-separated list of 1-based region indices forming basin #. By convention BASIN1 is the source basin (A) and BASIN2 is the target (B). The simulation stops when any basin is entered.");
  // keys.reset_style("BASIN", "optional");

  // Legacy rectangular shorthand — identical to Committor.cpp
  keys.add("numbered", "BASIN_LL",
           "Lower limits for basin # (legacy rectangular shorthand, one value per CV).");
  keys.add("numbered", "BASIN_UL",
           "Upper limits for basin # (legacy rectangular shorthand, one value per CV).");
  // keys.reset_style("BASIN_LL", "optional");
  // keys.reset_style("BASIN_UL", "optional");

  keys.add("compulsory", "Q_FLOOR", "1.0e-8",
           "Minimum value to which q is clipped outside all basins, to avoid division by zero near basin boundaries.");
}

// ---------------------------------------------------------------------------
TPP::TPP(const ActionOptions& ao)
  : Action(ao),
    Bias(ao),
    kbt_(getkBT()),
    d_(getNumberOfArguments()),
    nbasis_(0),
    hasSrc_(false),
    coverage_radius2_(-1.0),
    nbasins_(0),
    q_floor_(1.0e-8)
{
  if(d_ == 0) error("TPP: ARG must specify at least one CV");
  if(kbt_ == 0.0)
    error("TPP: temperature is required. Set TEMP or ensure the MD engine passes it to PLUMED.");

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

  // ---- Domain and Fourier basis functions ----
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
  nbasis_ = static_cast<unsigned>(nbasis_in);
  log.printf("  NBASIS = %u Fourier basis functions per dimension\n", nbasis_);

  basis_.reserve(d_);
  for(unsigned i = 0; i < d_; ++i)
    basis_.emplace_back(make_pair(domain_ll[i], domain_ul[i]),
                        nbasis_in, /*w=*/0.0, /*kernel=*/false, /*dx=*/0.0);

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
  // Expected: nclusters * NBASIS^2 values, one per non-comment line.
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

  // ---- q_floor ----
  parse("Q_FLOOR", q_floor_);
  if(q_floor_ <= 0.0) error("TPP: Q_FLOOR must be positive");
  log.printf("  q_floor = %e\n", q_floor_);

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

  // ---- Log region/basin configuration — mirrors Committor.cpp style ----
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
  log.printf("  Bias potential: V_bias = -2*kBT*log(q)\n");

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
// Test whether x lies inside a region — identical logic to Committor.cpp
// regionContains (open-interval RECT; closed sphere/ellipse).
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
// Basin 0 (BASIN1) = source A; basin 1+ (BASIN2, ...) = targets.
// ---------------------------------------------------------------------------
int TPP::whichBasin(const vector<double>& x) const {
  for(unsigned b = 0; b < nbasins_; ++b)
    for(unsigned k = 0; k < basinRegions_[b].size(); ++k)
      if(regionContains(regions_[basinRegions_[b][k]], x))
        return static_cast<int>(b);
  return -1;
}

// ---------------------------------------------------------------------------
// Evaluate q(x) and grad_q(x) using the cluster-product expansion.
//
// For cluster ic with dimension pair (k1, k2) and basis pair (nk1, nk2), the term is:
//   c[j] * phi_{nk1}(x[k1]) * phi_{nk2}(x[k2])
// where j = ic*nbasis_^2 + (nk1-1)*nbasis_ + (nk2-1).
//
// Gradient at dimension i (product rule, all cases):
//   k1==i, k2!=i:  dphi_{nk1}(x[i]) * phi_{nk2}(x[k2])
//   k1!=i, k2==i:  phi_{nk1}(x[k1]) * dphi_{nk2}(x[i])
//   k1==i, k2==i:  dphi_{nk1}(x[i])*phi_{nk2}(x[i]) + phi_{nk1}(x[i])*dphi_{nk2}(x[i])
//   otherwise:     0
// ---------------------------------------------------------------------------
double TPP::evalCommittor(const vector<double>& x, vector<double>& grad_q) const {
  fill(grad_q.begin(), grad_q.end(), 0.0);
  double q = 0.0;
  const unsigned nclusters = static_cast<unsigned>(clusters_.size());

  for(unsigned ic = 0; ic < nclusters; ++ic) {
    const unsigned k1 = clusters_[ic].first;   // 0-based CV dimension
    const unsigned k2 = clusters_[ic].second;

    for(unsigned nk1 = 1; nk1 <= nbasis_; ++nk1) {
      const double phi1  = basis_[k1](x[k1],  static_cast<int>(nk1), /*conv=*/false);
      const double dphi1 = basis_[k1].grad(x[k1], static_cast<int>(nk1), /*conv=*/false);

      for(unsigned nk2 = 1; nk2 <= nbasis_; ++nk2) {
        const unsigned j = ic * nbasis_ * nbasis_ + (nk1 - 1) * nbasis_ + (nk2 - 1);
        const double c = coeffs_[j];

        const double phi2  = basis_[k2](x[k2],  static_cast<int>(nk2), /*conv=*/false);
        const double dphi2 = basis_[k2].grad(x[k2], static_cast<int>(nk2), /*conv=*/false);

        q += c * phi1 * phi2;

        if(k1 == k2) {
          // Both functions act on the same CV — full product rule
          grad_q[k1] += c * (dphi1 * phi2 + phi1 * dphi2);
        } else {
          grad_q[k1] += c * dphi1 * phi2;
          grad_q[k2] += c * phi1  * dphi2;
        }
      }
    }
  }
  return q;
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
      // BASIN1 = source A: trajectory returned to A — stop gracefully
      log.printf("TPP: trajectory returned to source basin A (BASIN1) — stopping\n");
      plumed.stop();
    } else {
      // BASIN2+ = target B (or other targets): trajectory completed successfully
      log.printf("TPP: trajectory reached target basin %d (BASIN%d) — success\n",
                 basin + 1, basin + 1);
      plumed.stop();
    }
    return;
  }

  // ---- Coverage check ----
  if(hasSrc_) {
    double d2_min = numeric_limits<double>::max();
    for(const auto& s : src_) {
      double d2 = 0.0;
      for(unsigned i = 0; i < d_; ++i) {
        double dx = periodicDiff(x[i], s[i], i);
        d2 += dx * dx;
        if(d2 >= d2_min) break;
      }
      if(d2 < d2_min) d2_min = d2;
    }
    if(d2_min >= coverage_radius2_) {
      for(unsigned i = 0; i < d_; ++i) setOutputForce(i, 0.0);
      setBias(0.0);
      return;
    }
  }

  // ---- Evaluate committor and gradient ----
  vector<double> grad_q(d_, 0.0);
  double q     = evalCommittor(x, grad_q);

  // Clamp q to [0, 1]: values outside this range are Fourier artifacts.
  // Zero the gradient in both clamped cases so no spurious force is applied.
  double q_eff;
  if(q < 0.0) {
    q_eff = q_floor_;
    fill(grad_q.begin(), grad_q.end(), 0.0);
  } else if(q > 1.0) {
    q_eff = 1.0;
    fill(grad_q.begin(), grad_q.end(), 0.0);
  } else {
    q_eff = max(q, q_floor_);
  }

  // V_bias = -2*kBT*log(q_eff)  =>  F_i = +2*kBT * grad_q[i] / q_eff
  for(unsigned i = 0; i < d_; ++i)
    setOutputForce(i, 2.0 * kbt_ * grad_q[i] / q_eff);

  setBias(-2.0 * kbt_ * std::log(q_eff));
}

} // namespace tpp
} // namespace PLMD
