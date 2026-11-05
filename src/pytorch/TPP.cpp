/* Transition Path Process (TPP) bias — PyTorch committor/HJB version.
 *
 * Drives trajectories along reactive paths using the Doob h-transform
 * drift built from a model supplied as a TorchScript-compiled PyTorch
 * model (a *.pt / *.ptc file produced by torch.jit.trace or
 * torch.jit.script), evaluated and differentiated via LibTorch autograd —
 * no basis-function expansion of any kind. This replaces the previous
 * Galerkin/fast-operator-learning implementation of this file (Fourier or
 * Gaussian-kernel cluster-product basis, coefficients from COEFF_FILE,
 * src/ttsketch/BasisFunc.h) with the same evaluate-model/take-gradient
 * approach used by PYTORCH_MODEL (src/pytorch/PytorchModel.cpp), combined
 * with the xi-regularized Doob-drift formulas from this project's
 * {doublewell,triplewell,aladip}_picard_tpp_nn.py scripts.
 *
 * Two different model representations are supported, selected by MODE:
 *
 *  - NEG_LOG_Q / NEG_LOG_Q_HALF ("q-modes"): the model outputs q itself
 *    (the committor, in [0,1]) — matching CommittorNet / MaskedQNet's raw
 *    output in the *_picard_tpp_nn.py scripts' richardson_init() stage.
 *    Regularization is ASYMMETRIC:
 *
 *      q_xi = xi + (1-xi)*q,   q_xi in [xi, 1]   for xi in (0, 1).
 *
 *    Only the A boundary (q=0) is lifted to xi; q=1 at B needs no
 *    regularization since log(1) is finite.
 *
 *      MODE=NEG_LOG_Q (full Doob drift):
 *        force_i = 2*kBT * (1-xi) * dq/dCV_i / q_xi
 *        V_bias  = -2*kBT * log q_xi
 *
 *      MODE=NEG_LOG_Q_HALF (Picard modified drift, soc.pdf eq. 19):
 *        force_i = kBT * (1-xi) * dq/dCV_i / q_xi
 *        V_bias  = -kBT * log q_xi
 *
 *  - PHI_HALF / PHI ("phi-modes"): the model outputs Phi = -log(q_xi)
 *    directly — matching BoundedPhiNet's raw output in the
 *    *_picard_tpp_nn.py scripts' main Picard loop (Phi(x) =
 *    phi_hi*sigmoid(net(x)), trained with boundary targets Phi=phi_hi at
 *    A and Phi=0 at B, phi_hi = -log(xi)). Substituting Phi for -log(q_xi)
 *    into the q-mode formulas above (grad(log q_xi) = -grad(Phi)) gives:
 *
 *      MODE=PHI (full Doob drift):
 *        force_i = -2*kBT * dPhi/dCV_i
 *        V_bias  =  2*kBT * Phi
 *
 *      MODE=PHI_HALF (Picard modified drift, soc.pdf eq. 19):
 *        force_i = -kBT * dPhi/dCV_i
 *        V_bias  =  kBT * Phi
 *
 *    No xi-regularization term appears in the phi-mode force/bias
 *    formulas themselves (Phi is already regularized by construction, via
 *    the network's own sigmoid-bounded training target) — XI is used only
 *    to derive the defensive clip range [0, -log(xi)] for a stray
 *    out-of-range forward pass (see calculate()), mirroring the clip to
 *    [0,1] used for the q-modes. This differs from the old (pre-PyTorch,
 *    basis-expansion) HJB_PHI/HJB_PHI_HALF modes, which existed to convert
 *    an already-regularized Phi_xi1 basis expansion between regularization
 *    levels on the fly; there is no such baked-in xi1 here, since the
 *    model is retrained from scratch each outer Picard iteration.
 *
 * Typical use (mirroring the *_picard_tpp_nn.py main loop): MODE=
 * NEG_LOG_Q_HALF with a previously-computed raw-committor model for the
 * very first biased iteration (warm start from richardson_init()), then
 * MODE=PHI_HALF with the current BoundedPhiNet export for every iteration
 * thereafter. MODE=PHI (full factor) is for downstream, post-convergence
 * Doob dynamics / enhanced sampling — a separate concern from the fitting
 * loop this mode list otherwise serves.
 *
 * The model is loaded once (MODEL keyword), frozen and optimized for
 * inference exactly as PYTORCH_MODEL does, and must take the bias's ARG
 * list as a single (1, n_arg) input and return a single scalar output.
 * Region/basin geometry, periodicity handling, basin-stop logic and
 * the optional nearest-source-state coverage check are all unchanged from
 * the previous implementation — none of that depended on BasisFunc.
 *
 * Removed keywords (no longer meaningful without a basis expansion):
 * DOMAIN_LL, DOMAIN_UL, NBASIS, CLUSTER1, CLUSTER2, KERNEL_BASIS,
 * KERNEL_DX, COEFF_FILE/COEFFS, XI2. Replaced by: MODEL.
 */

// Lives in src/pytorch/ (not src/bias/) specifically because this module's
// Makefile has no -Werror override: LibTorch's own c10/ATen headers are not
// -pedantic-clean under strict -std=c++17 (they rely on GNU variadic-macro
// extensions), which src/bias/Makefile's `CXXFLAGS += -Wshadow -Werror`
// turns into hard compile errors. src/pytorch/PytorchModel.cpp already
// compiles cleanly against these same headers for exactly this reason.
#ifdef __PLUMED_HAS_LIBTORCH

#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Exception.h"
#include "tools/Tools.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Note: Freezing a ScriptModule (torch::jit::freeze) works only in >=1.11.
// For 1.8 <= versions <=1.10 we need a hack — identical workaround to
// PytorchModel.cpp's own (see that file for the upstream references).
#if ( TORCH_VERSION_MAJOR == 2 || TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10 )
#define DO_TORCH_FREEZE_HACK
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#endif

using namespace std;
using namespace PLMD::bias;

namespace PLMD {
namespace tpp {

class TPP : public Bias {
private:
  // ---- kBT ----
  double kbt_;

  // ---- Drift mode ----
  enum class Mode { NEG_LOG_Q, NEG_LOG_Q_HALF, PHI_HALF, PHI };
  Mode mode_;

  // ---- PyTorch committor model ----
  unsigned d_;
  torch::jit::script::Module model_;
  torch::Device device_ = torch::kCPU;

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

  // ---- xi regularization (asymmetric: q_xi = xi + (1-xi)*q) ----
  // phiHi_ = -log(xi_), the phi-mode clip bound (see calculate()); derived
  // once from xi_ at construction time, same phi_hi = -log(xi) convention
  // used by BoundedPhiNet in the *_picard_tpp_nn.py scripts.
  double xi_;
  double phiHi_;

  // ---- helpers ----
  double periodicDiff(double x, double c, unsigned dim) const;
  bool   regionContains(const Region& reg, const vector<double>& x) const;
  int    whichBasin(const vector<double>& x) const;
  // Evaluates the model and its gradient at x via LibTorch autograd.
  // Returns the raw (pre-clip) model output — q for the q-modes, Phi for
  // the phi-modes (see MODE).
  double evalModel(const vector<double>& x, vector<double>& grad);

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

  keys.add("optional", "MODE",
           "Drift mode. Two model representations: q-modes (NEG_LOG_Q, NEG_LOG_Q_HALF) expect MODEL "
           "to output the committor q in [0,1]; phi-modes (PHI, PHI_HALF) expect MODEL to output "
           "Phi = -log(q_xi) directly. NEG_LOG_Q (default): full Doob drift, "
           "force_i = 2*kBT*(1-xi)*dq/dCV_i/q_xi. NEG_LOG_Q_HALF: Picard modified drift (half factor), "
           "q-valued model — typically used for the first biased iteration, warm-started from a "
           "previously-computed raw-committor model. PHI_HALF: Picard modified drift (half factor), "
           "phi-valued model — force_i = -kBT*dPhi/dCV_i; used for every iteration after the first. "
           "PHI: full Doob drift, phi-valued model — force_i = -2*kBT*dPhi/dCV_i; for downstream, "
           "post-convergence Doob dynamics, not the fitting loop.");

  keys.add("compulsory", "MODEL",
           "Path to a TorchScript-compiled model (torch.jit.trace / torch.jit.script output, "
           "e.g. a *.pt or *.ptc file). The model must accept a (1, n_arg) tensor of the bias's ARG "
           "values, in ARG order, and return a single scalar output, whose meaning depends on MODE: "
           "for NEG_LOG_Q/NEG_LOG_Q_HALF a committor q in [0,1] (matching this project's CommittorNet/ "
           "MaskedQNet raw output); for PHI/PHI_HALF the HJB potential Phi = -log(q_xi) directly "
           "(matching this project's BoundedPhiNet raw output), not a committor.");

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
           "Regularization parameter xi for asymmetric regularization q_xi = xi + (1-xi)*q, "
           "so q_xi in [xi, 1]. Must be in (0, 1). Only the A boundary (q=0) is regularized; "
           "q=1 at B needs no regularization.");
}

// ---------------------------------------------------------------------------
TPP::TPP(const ActionOptions& ao)
  : Action(ao),
    Bias(ao),
    kbt_(getkBT()),
    mode_(Mode::NEG_LOG_Q),
    d_(getNumberOfArguments()),
    hasSrc_(false),
    doCoverageCheck_(false),
    coverage_radius2_(-1.0),
    nbasins_(0),
    xi_(0.01),
    phiHi_(0.0)
{
  if(d_ == 0) error("TPP: ARG must specify at least one CV");
  if(kbt_ == 0.0)
    error("TPP: temperature is required. Set TEMP or ensure the MD engine passes it to PLUMED.");

  // ---- Mode ----
  string mode_str = "NEG_LOG_Q";
  parse("MODE", mode_str);
  if     (mode_str == "NEG_LOG_Q")      mode_ = Mode::NEG_LOG_Q;
  else if(mode_str == "NEG_LOG_Q_HALF") mode_ = Mode::NEG_LOG_Q_HALF;
  else if(mode_str == "PHI_HALF")       mode_ = Mode::PHI_HALF;
  else if(mode_str == "PHI")            mode_ = Mode::PHI;
  else error("TPP: unknown MODE '" + mode_str + "'. Valid: NEG_LOG_Q, NEG_LOG_Q_HALF, PHI_HALF, PHI");

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

  // ---- Load the TorchScript committor model ----
  {
    std::stringstream ss;
    ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
    std::string version;
    ss >> version;
    log.printf(("  LibTorch version: " + version + "\n").data());
  }

  string model_file;
  parse("MODEL", model_file);
  try {
    model_ = torch::jit::load(model_file, device_);
  } catch (const c10::Error& e) {
    std::ifstream infile(model_file);
    bool exist = infile.good();
    infile.close();
    if(exist)
      error("TPP: cannot load MODEL '" + model_file + "'. Please check that it is a TorchScript-compiled "
            "model (exported with torch.jit.trace or torch.jit.script).");
    else
      error("TPP: MODEL file '" + model_file + "' does not exist.");
  }

  model_.eval();
#ifdef DO_TORCH_FREEZE_HACK
  {
    bool optimize_numerics = true;
    auto out_mod = torch::jit::freeze_module(model_, {});
    auto graph = out_mod.get_method("forward").graph();
    OptimizeFrozenGraph(graph, optimize_numerics);
    model_ = out_mod;
  }
#else
  model_ = torch::jit::freeze(model_);
#endif
#if (TORCH_VERSION_MAJOR == 2 || TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10)
  model_ = torch::jit::optimize_for_inference(model_);
#endif

  // ---- Sanity-check the model's output shape with a dummy forward pass ----
  {
    std::vector<float> zeros(d_, 0.0f);
    torch::Tensor single_input = torch::tensor(zeros).view({1, static_cast<long>(d_)}).to(device_);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(single_input);
    torch::Tensor output = model_.forward(inputs).toTensor();
    if(output.numel() != 1)
      error("TPP: MODEL must return exactly one scalar output (q or Phi, depending on MODE), got " +
            to_string(output.numel()) + " values. Check the model's forward() signature.");
  }
  log.printf("  Loaded model from '%s'  (n_in=%u)\n", model_file.c_str(), d_);

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
      // No DOMAIN_LL/UL anymore (no basis to size); fall back to the
      // bounding box of the source states themselves for the default
      // coverage-radius volume estimate.
      vector<double> lo(d_,  numeric_limits<double>::infinity());
      vector<double> hi(d_, -numeric_limits<double>::infinity());
      for(const auto& s : src_)
        for(unsigned i = 0; i < d_; ++i) {
          lo[i] = min(lo[i], s[i]);
          hi[i] = max(hi[i], s[i]);
        }
      double domain_vol = 1.0;
      for(unsigned i = 0; i < d_; ++i) domain_vol *= max(hi[i] - lo[i], 1e-12);
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
  parse("XI", xi_);
  if(xi_ <= 0.0 || xi_ >= 1.0)
    error("TPP: XI must be in (0, 1)");
  phiHi_ = -std::log(xi_);
  if(mode_ == Mode::NEG_LOG_Q || mode_ == Mode::NEG_LOG_Q_HALF)
    log.printf("  xi = %e  (asymmetric regularization: q_xi = xi + (1-xi)*q)\n", xi_);
  else
    log.printf("  xi = %e  (used only to set the phi-mode clip bound phi_hi = -log(xi) = %f)\n",
               xi_, phiHi_);

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
    const char* names[] = {"NEG_LOG_Q", "NEG_LOG_Q_HALF", "PHI_HALF", "PHI"};
    log.printf("  Mode: %s\n", names[static_cast<int>(mode_)]);
  }
  switch(mode_) {
  case Mode::NEG_LOG_Q:
    log.printf("  Drift: 2*kBT*(1-xi)*grad_q/q_xi  (Doob, q-valued model)\n");
    break;
  case Mode::NEG_LOG_Q_HALF:
    log.printf("  Drift: kBT*(1-xi)*grad_q/q_xi    (Picard, q-valued model)\n");
    break;
  case Mode::PHI:
    log.printf("  Drift: -2*kBT*grad_Phi           (Doob, phi-valued model)\n");
    break;
  case Mode::PHI_HALF:
    log.printf("  Drift: -kBT*grad_Phi             (Picard, phi-valued model)\n");
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
// Evaluate the PyTorch committor model and its gradient at x via LibTorch
// autograd. Same forward/grad pattern as PytorchModel::calculate(), but
// for a single scalar output and only the one gradient we actually need.
// ---------------------------------------------------------------------------
double TPP::evalModel(const vector<double>& x, vector<double>& grad) {
  vector<float> xf(x.begin(), x.end());
  torch::Tensor input = torch::tensor(xf).view({1, static_cast<long>(d_)}).to(device_);
  input.set_requires_grad(true);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  torch::Tensor output = model_.forward(inputs).toTensor();

  double q = output.item<float>();

  auto gradient = torch::autograd::grad(
                    {output}, {input},
                    /*grad_outputs=*/{torch::ones_like(output)},
                    /*retain_graph=*/false,
                    /*create_graph=*/false)[0];

  auto gradient_cpu = gradient.to(torch::kCPU).contiguous();
  const float* gptr = gradient_cpu.data_ptr<float>();
  for(unsigned i = 0; i < d_; ++i) grad[i] = static_cast<double>(gptr[i]);

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
  double raw = evalModel(x, grad);

  if(mode_ == Mode::NEG_LOG_Q || mode_ == Mode::NEG_LOG_Q_HALF) {
    // q-modes: q_xi = xi + (1-xi)*q_raw, asymmetric regularization (see
    // file header). grad log q_xi = (1-xi)*grad_q / q_xi.
    // force_i = factor * kBT * (1-xi) * dq/dCV_i / q_xi

    // Clip model output to [0,1]; zero gradient at the boundary (same
    // defensive pattern as the old Fourier-artifact clipping — should
    // rarely trigger for a sigmoid/mask-constrained committor network, but
    // costs nothing to guard against a stray out-of-range forward pass).
    double q_raw = raw;
    if(q_raw <= 0.0) {
      q_raw = 0.0;
      fill(grad.begin(), grad.end(), 0.0);
    } else if(q_raw >= 1.0) {
      q_raw = 1.0;
      fill(grad.begin(), grad.end(), 0.0);
    }

    const double qalpha = 1.0 - xi_;                // q_xi = xi + qalpha*q_raw
    const double q_xi   = xi_ + qalpha * q_raw;      // in [xi_, 1] after clipping
    const double factor = (mode_ == Mode::NEG_LOG_Q) ? 2.0 : 1.0;

    for(unsigned i = 0; i < d_; ++i)
      setOutputForce(i, factor * kbt_ * qalpha * grad[i] / q_xi);
    setBias(-factor * kbt_ * std::log(q_xi));

  } else {
    // phi-modes: the model outputs Phi = -log(q_xi) directly (see file
    // header for the substitution that gives these formulas from the
    // q-mode ones above). No xi term in the force/bias itself — XI only
    // sets the defensive clip bound phi_hi = -log(xi), mirroring the
    // q-modes' clip to [0,1] for a stray out-of-range forward pass (should
    // rarely trigger for a sigmoid-bounded BoundedPhiNet).
    double phi_raw = raw;
    if(phi_raw <= 0.0) {
      phi_raw = 0.0;
      fill(grad.begin(), grad.end(), 0.0);
    } else if(phi_raw >= phiHi_) {
      phi_raw = phiHi_;
      fill(grad.begin(), grad.end(), 0.0);
    }

    const double factor = (mode_ == Mode::PHI) ? 2.0 : 1.0;

    for(unsigned i = 0; i < d_; ++i)
      setOutputForce(i, -factor * kbt_ * grad[i]);
    setBias(factor * kbt_ * phi_raw);
  }
}

} // namespace tpp
} // namespace PLMD

#endif // __PLUMED_HAS_LIBTORCH
