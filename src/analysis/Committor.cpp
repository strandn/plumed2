/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "core/ActionPilot.h"
#include "core/ActionWithArguments.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include <cmath>

namespace PLMD {
namespace generic {

//+PLUMEDOC PRINTANALYSIS COMMITTOR
/*
Does a committor analysis.

Supports three region shapes: rectangular (the original behavior), spherical
(hyperspherical in CV space), and elliptical (axis-aligned ellipsoids in CV
space).  Each basin can be a single region or a **union** of several regions,
which is useful when a metastable macro-state spans multiple local minima.

Periodic boundary conditions are handled automatically for periodic CVs
such as torsion angles.  For spheres and ellipses the minimum-image distance
is used.  For rectangles on periodic CVs, setting REGION_LL > REGION_UL
for a given CV dimension is interpreted as a wrap-around interval that
crosses the periodic boundary (e.g. the region near ±π for a torsion).

Two input modes are supported.  **Legacy mode** uses the original BASIN_LL /
BASIN_UL keywords and is fully backward compatible; every numbered pair
defines one rectangular basin.  **Region mode** lets you define an arbitrary
collection of regions with REGION_* keywords and then group them into
basins with BASIN keywords.  The two modes cannot be mixed.

\par Legacy syntax (backward compatible)

\plumedfile
TORSION ATOMS=1,2,3,4 LABEL=r1
TORSION ATOMS=2,3,4,5 LABEL=r2
COMMITTOR ...
  ARG=r1,r2
  STRIDE=10
  BASIN_LL1=0.15,0.20
  BASIN_UL1=0.25,0.40
  BASIN_LL2=-0.25,-0.40
  BASIN_UL2=-0.15,-0.20
... COMMITTOR
\endplumedfile

\par Spherical region

\plumedfile
TORSION ATOMS=1,2,3,4 LABEL=phi
TORSION ATOMS=2,3,4,5 LABEL=psi
COMMITTOR ...
  ARG=phi,psi
  STRIDE=10
  REGION_LL1=-3.0,0.5
  REGION_UL1=-0.8,3.1
  REGION_CENTER2=1.1,-0.7
  REGION_RADIUS2=0.5
  BASIN1=1
  BASIN2=2
... COMMITTOR
\endplumedfile

\par Union of regions (e.g. two minima in basin A)

\plumedfile
TORSION ATOMS=1,2,3,4 LABEL=phi
TORSION ATOMS=2,3,4,5 LABEL=psi
COMMITTOR ...
  ARG=phi,psi
  STRIDE=10
  REGION_LL1=-3.0,2.0
  REGION_UL1=-2.0,3.1
  REGION_CENTER2=-1.4,1.0
  REGION_RADIUS2=0.4
  REGION_CENTER3=1.1,-0.7
  REGION_AXES3=0.5,0.3
  BASIN1=1,2
  BASIN2=3
... COMMITTOR
\endplumedfile

\par Wrap-around rectangle on a periodic CV

For a torsion angle (period 2π on [−pi, pi]), a rectangle with
REGION_LL = 2.5 and REGION_UL = −2.5 in that dimension covers
the region [2.5, pi] ∪ [−pi, −2.5].

*/
//+ENDPLUMEDOC

class Committor :
  public ActionPilot,
  public ActionWithArguments
{
private:
  std::string file;
  OFile ofile;
  std::string fmt;

  // ---- Region geometry ----
  struct Region {
    enum Type { RECT, SPHERE, ELLIPSE };
    Type type;
    // RECT: lower/upper bounds per CV dimension
    std::vector<double> lower, upper;
    // SPHERE / ELLIPSE: center and semi-axis lengths per CV dimension
    // (for a sphere all semi_axes entries are equal to the radius)
    std::vector<double> center;
    std::vector<double> semi_axes;
  };

  std::vector<Region> regions;
  // Each basin is a union of region indices (0-based into regions[])
  std::vector< std::vector<unsigned> > basinRegions;
  unsigned nbasins;
  unsigned basin;   // 1-based index of the basin currently occupied (0 = none)
  bool doNotStop;

  // Periodicity cache (filled once in constructor)
  std::vector<bool>   argIsPeriodic;
  std::vector<double> argPeriod;  // period length; 0 when not periodic

  // Minimum-image signed displacement  x − c  respecting periodicity
  double periodicDiff(double x, double c, unsigned dim) const;

  // Test whether a point lies inside a region
  bool regionContains(const Region& reg, const std::vector<double>& args) const;

public:
  static void registerKeywords( Keywords& keys );
  explicit Committor(const ActionOptions& ao);
  void calculate() override;
  void apply() override {}
};

PLUMED_REGISTER_ACTION(Committor,"COMMITTOR")

void Committor::registerKeywords( Keywords& keys ) {
  Action::registerKeywords(keys);
  ActionPilot::registerKeywords(keys);
  ActionWithArguments::registerKeywords(keys);
  keys.use("ARG");

  // ---------- legacy rectangular-basin keywords (backward compatible) ----------
  keys.add("numbered", "BASIN_LL","List of lower limits for basin #. "
           "Legacy syntax: each numbered pair BASIN_LL / BASIN_UL defines one rectangular basin.");
  keys.add("numbered", "BASIN_UL","List of upper limits for basin #.");
  keys.reset_style("BASIN_LL","optional");
  keys.reset_style("BASIN_UL","optional");

  // ---------- new region keywords ----------
  keys.add("numbered", "REGION_LL",    "Lower limits for rectangular region #. "
           "For periodic CVs, setting REGION_LL > REGION_UL indicates a wrap-around interval.");
  keys.add("numbered", "REGION_UL",    "Upper limits for rectangular region #.");
  keys.add("numbered", "REGION_CENTER","Center of spherical or elliptical region #.");
  keys.add("numbered", "REGION_RADIUS","Radius of spherical region # (single value).");
  keys.add("numbered", "REGION_AXES",  "Semi-axis lengths of elliptical region # (one per CV).");

  // ---------- basin-as-union keyword ----------
  keys.add("numbered", "BASIN", "Comma-separated list of 1-based region indices that form basin #. "
           "A basin is the union of all listed regions.");

  // ---------- common keywords ----------
  keys.add("compulsory","STRIDE","1","the frequency with which the CVs are analyzed");
  keys.add("optional","FILE","the name of the file on which to output the reached basin");
  keys.add("optional","FMT","the format that should be used to output real numbers");
  keys.addFlag("NOSTOP",false,"if true do not stop the simulation when reaching a basin but just keep track of it");
}

// ---------------------------------------------------------------------------
//  Constructor – parse input and build regions + basins
// ---------------------------------------------------------------------------
Committor::Committor(const ActionOptions& ao):
  Action(ao),
  ActionPilot(ao),
  ActionWithArguments(ao),
  fmt("%f"),
  nbasins(0),
  basin(0),
  doNotStop(false)
{
  // --- output file ---
  ofile.link(*this);
  parse("FILE",file);
  if(file.length()>0) {
    ofile.open(file);
    log.printf("  on file %s\n",file.c_str());
  } else {
    log.printf("  on plumed log file\n");
    ofile.link(log);
  }
  parse("FMT",fmt);
  fmt=" "+fmt;
  log.printf("  with format %s\n",fmt.c_str());

  const unsigned nargs = getNumberOfArguments();

  // --- cache periodicity information for every CV ---
  argIsPeriodic.resize(nargs, false);
  argPeriod.resize(nargs, 0.0);
  for(unsigned i=0; i<nargs; ++i) {
    if( getPntrToArgument(i)->isPeriodic() ) {
      argIsPeriodic[i] = true;
      std::string smin, smax;
      getPntrToArgument(i)->getDomain(smin, smax);
      double dmin, dmax;
      Tools::convert(smin, dmin);
      Tools::convert(smax, dmax);
      argPeriod[i] = dmax - dmin;
    }
  }

  // =====================================================================
  //  Try region mode first: parse all REGION_* entries directly.
  //  Do NOT probe at index 1 first — parseNumberedVector consumes keywords
  //  and would deplete REGION_CENTER1/REGION_AXES1 before the loop runs.
  //  Mode is determined from whether any regions were successfully parsed.
  // =====================================================================
  for(unsigned r=1;; ++r) {
    std::vector<double> rll, rul, rcenter, rradius, raxes;
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
        error("COMMITTOR: REGION_LL and REGION_UL must both be given for rectangular region " + std::to_string(r));
      if(rll.size()!=nargs)
        error("COMMITTOR: REGION_LL" + std::to_string(r) + " has wrong number of values (expected " + std::to_string(nargs) + ")");
      if(rul.size()!=nargs)
        error("COMMITTOR: REGION_UL" + std::to_string(r) + " has wrong number of values (expected " + std::to_string(nargs) + ")");
      if(!rcenter.empty() || !rradius.empty() || !raxes.empty())
        error("COMMITTOR: region " + std::to_string(r) + " mixes RECT keywords (LL/UL) with SPHERE/ELLIPSE keywords (CENTER/RADIUS/AXES)");
      for(unsigned i=0; i<nargs; ++i) {
        if(rll[i] > rul[i] && !argIsPeriodic[i])
          error("COMMITTOR: REGION_UL must be >= REGION_LL for non-periodic CV dimension " + std::to_string(i) + " in region " + std::to_string(r));
      }
      reg.type  = Region::RECT;
      reg.lower = rll;
      reg.upper = rul;
    }
    // --- Spherical ---
    else if(!rcenter.empty() && !rradius.empty() && raxes.empty()) {
      if(rcenter.size()!=nargs)
        error("COMMITTOR: REGION_CENTER" + std::to_string(r) + " has wrong number of values (expected " + std::to_string(nargs) + ")");
      if(rradius.size()!=1)
        error("COMMITTOR: REGION_RADIUS" + std::to_string(r) + " must be a single value");
      reg.type      = Region::SPHERE;
      reg.center    = rcenter;
      reg.semi_axes.assign(nargs, rradius[0]);
    }
    // --- Elliptical ---
    else if(!rcenter.empty() && !raxes.empty() && rradius.empty()) {
      if(rcenter.size()!=nargs)
        error("COMMITTOR: REGION_CENTER" + std::to_string(r) + " has wrong number of values (expected " + std::to_string(nargs) + ")");
      if(raxes.size()!=nargs)
        error("COMMITTOR: REGION_AXES" + std::to_string(r) + " has wrong number of values (expected " + std::to_string(nargs) + ")");
      for(unsigned i=0; i<nargs; ++i) {
        if(raxes[i]<=0.0) error("COMMITTOR: REGION_AXES values must be positive in region " + std::to_string(r));
      }
      reg.type      = Region::ELLIPSE;
      reg.center    = rcenter;
      reg.semi_axes = raxes;
    }
    else {
      error("COMMITTOR: could not determine region type for region " + std::to_string(r) +
            ". Use REGION_LL+REGION_UL (rect), REGION_CENTER+REGION_RADIUS (sphere), or REGION_CENTER+REGION_AXES (ellipse).");
    }
    regions.push_back(reg);
  }

  bool hasRegions = !regions.empty();

  if(hasRegions) {
    // --- Parse BASIN keywords (unions of regions) ---
    for(unsigned b=1;; ++b) {
      std::vector<double> bvec_d;
      parseNumberedVector("BASIN", b, bvec_d);
      if(bvec_d.empty()) break;
      std::vector<unsigned> bvec;
      for(unsigned k=0; k<bvec_d.size(); ++k) {
        int idx = static_cast<int>(std::round(bvec_d[k]));
        if(idx<1 || idx>static_cast<int>(regions.size()))
          error("COMMITTOR: BASIN" + std::to_string(b) + " references region " + std::to_string(idx) +
                " which is out of range [1," + std::to_string(regions.size()) + "]");
        bvec.push_back(static_cast<unsigned>(idx-1)); // convert to 0-based
      }
      basinRegions.push_back(bvec);
      nbasins = b;
    }
    if(nbasins==0) error("COMMITTOR: REGION_* keywords found but no BASIN keywords to assign regions to basins");
  }

  // =====================================================================
  //  LEGACY MODE – fallback when no REGION_* keywords were present.
  //  Each BASIN_LL/BASIN_UL pair defines one rectangular basin directly.
  // =====================================================================
  bool hasLegacy = false;
  if(!hasRegions) {
    for(unsigned b=1;; ++b) {
      std::vector<double> tmpl, tmpu;
      parseNumberedVector("BASIN_LL", b, tmpl);
      parseNumberedVector("BASIN_UL", b, tmpu);
      if(tmpl.empty() && tmpu.empty()) break;
      if(tmpl.size()!=nargs) error("Wrong number of values for BASIN_LL: they should be equal to the number of arguments");
      if(tmpu.size()!=nargs) error("Wrong number of values for BASIN_UL: they should be equal to the number of arguments");

      for(unsigned i=0; i<nargs; ++i) {
        if(tmpl[i] > tmpu[i] && !argIsPeriodic[i])
          error("COMMITTOR: BASIN_UL must be >= BASIN_LL for non-periodic CVs");
      }

      Region reg;
      reg.type  = Region::RECT;
      reg.lower = tmpl;
      reg.upper = tmpu;
      regions.push_back(reg);

      std::vector<unsigned> bvec(1, static_cast<unsigned>(regions.size()-1));
      basinRegions.push_back(bvec);
      nbasins = b;
    }
    hasLegacy = (nbasins > 0);
  }

  if(!hasLegacy && !hasRegions)
    error("COMMITTOR: no basins defined.  Use BASIN_LL/BASIN_UL (legacy) or REGION_*/BASIN (region mode).");

  parseFlag("NOSTOP", doNotStop);
  checkRead();

  // --- Log the parsed configuration ---
  log.printf("  Number of regions: %u\n", static_cast<unsigned>(regions.size()));
  for(unsigned r=0; r<regions.size(); ++r) {
    const Region& reg = regions[r];
    switch(reg.type) {
    case Region::RECT:
      log.printf("  Region %u: RECT\n", r+1);
      for(unsigned i=0; i<nargs; ++i) {
        if(reg.lower[i] > reg.upper[i])
          log.printf("    dim %u: [%f, domain_max] U [domain_min, %f]  (wrap-around)\n", i, reg.lower[i], reg.upper[i]);
        else
          log.printf("    dim %u: [%f, %f]\n", i, reg.lower[i], reg.upper[i]);
      }
      break;
    case Region::SPHERE:
      log.printf("  Region %u: SPHERE  center=(", r+1);
      for(unsigned i=0; i<nargs; ++i) log.printf("%s%f", i?",":"", reg.center[i]);
      log.printf(")  radius=%f\n", reg.semi_axes[0]);
      break;
    case Region::ELLIPSE:
      log.printf("  Region %u: ELLIPSE  center=(", r+1);
      for(unsigned i=0; i<nargs; ++i) log.printf("%s%f", i?",":"", reg.center[i]);
      log.printf(")  semi_axes=(");
      for(unsigned i=0; i<nargs; ++i) log.printf("%s%f", i?",":"", reg.semi_axes[i]);
      log.printf(")\n");
      break;
    }
  }
  log.printf("  Number of basins: %u\n", nbasins);
  for(unsigned b=0; b<nbasins; ++b) {
    log.printf("  Basin %u = union of region(s):", b+1);
    for(unsigned k=0; k<basinRegions[b].size(); ++k)
      log.printf(" %u", basinRegions[b][k]+1);
    log.printf("\n");
  }
  if(doNotStop) log.printf("  NOSTOP: will track visited basins without stopping the simulation\n");

  // --- Periodicity summary ---
  for(unsigned i=0; i<nargs; ++i) {
    if(argIsPeriodic[i])
      log.printf("  CV %u is periodic with period %f\n", i, argPeriod[i]);
  }

  for(unsigned i=0; i<nargs; ++i) ofile.setupPrintValue( getPntrToArgument(i) );
}

// ---------------------------------------------------------------------------
//  Minimum-image signed displacement (x - c) for dimension dim
// ---------------------------------------------------------------------------
double Committor::periodicDiff(double x, double c, unsigned dim) const {
  double dx = x - c;
  if(argIsPeriodic[dim]) {
    const double period = argPeriod[dim];
    const double half   = 0.5 * period;
    while(dx >  half) dx -= period;
    while(dx < -half) dx += period;
  }
  return dx;
}

// ---------------------------------------------------------------------------
//  Test whether args[] lies inside a given region
// ---------------------------------------------------------------------------
bool Committor::regionContains(const Region& reg, const std::vector<double>& args) const {
  const unsigned nargs = args.size();

  switch(reg.type) {

  // ---- Rectangle ----
  case Region::RECT:
    for(unsigned i=0; i<nargs; ++i) {
      if(argIsPeriodic[i]) {
        if(reg.lower[i] <= reg.upper[i]) {
          // Normal (non-wrapping) interval on a periodic CV.
          // Recast as: distance from interval midpoint < half-width.
          double mid  = 0.5 * (reg.lower[i] + reg.upper[i]);
          double half = 0.5 * (reg.upper[i] - reg.lower[i]);
          double dx   = std::abs(periodicDiff(args[i], mid, i));
          if(dx >= half) return false;   // strict inequality (open interval)
        } else {
          // Wrap-around interval (lower > upper).
          // The "gap" that is NOT in the region is (upper, lower).
          double gap_mid  = 0.5 * (reg.upper[i] + reg.lower[i]);
          double gap_half = 0.5 * (reg.lower[i] - reg.upper[i]);
          double dx       = std::abs(periodicDiff(args[i], gap_mid, i));
          if(dx < gap_half) return false;  // inside the gap → outside the region
        }
      } else {
        // Non-periodic: simple open-interval check (matches original behavior)
        if(args[i] <= reg.lower[i] || args[i] >= reg.upper[i]) return false;
      }
    }
    return true;

  // ---- Sphere (all semi-axes equal) ----
  case Region::SPHERE: {
    double r2sum = 0.0;
    const double R2 = reg.semi_axes[0] * reg.semi_axes[0];
    for(unsigned i=0; i<nargs; ++i) {
      double dx = periodicDiff(args[i], reg.center[i], i);
      r2sum += dx * dx;
      if(r2sum > R2) return false;  // early exit
    }
    return true;   // r2sum <= R2  (closed ball, boundary included)
  }

  // ---- Ellipse (axis-aligned, semi-axes may differ) ----
  case Region::ELLIPSE: {
    double sum = 0.0;
    for(unsigned i=0; i<nargs; ++i) {
      double dx = periodicDiff(args[i], reg.center[i], i);
      double a  = reg.semi_axes[i];
      sum += (dx * dx) / (a * a);
      if(sum > 1.0) return false;  // early exit
    }
    return true;   // sum <= 1  (closed ellipsoid, boundary included)
  }

  } // end switch
  return false;
}

// ---------------------------------------------------------------------------
//  Main calculation – called every STRIDE steps
// ---------------------------------------------------------------------------
void Committor::calculate() {
  const unsigned nargs = getNumberOfArguments();

  // Gather current CV values
  std::vector<double> args(nargs);
  for(unsigned i=0; i<nargs; ++i) args[i] = getArgument(i);

  // For each basin, check if the point is in any of its constituent regions
  bool inonebasin = false;
  for(unsigned b=0; b<nbasins; ++b) {
    bool inThisBasin = false;
    for(unsigned k=0; k<basinRegions[b].size(); ++k) {
      if( regionContains(regions[ basinRegions[b][k] ], args) ) {
        inThisBasin = true;
        break;   // union: one hit is enough
      }
    }

    if(inThisBasin) {
      // Log a transition only when the basin identity changes
      if(basin != (b+1)) {
        basin = b+1;
        ofile.fmtField(" %f");
        ofile.printField("time", getTime());
        for(unsigned i=0; i<nargs; i++) {
          ofile.fmtField(fmt);
          ofile.printField( getPntrToArgument(i), getArgument(i) );
        }
        ofile.printField("basin", static_cast<int>(b+1));
        ofile.printField();
      }
      inonebasin = true;
      break;
    }
  }
  if(!inonebasin) basin = 0;

  // Stop the simulation if requested
  if(inonebasin && !doNotStop) {
    std::string num; Tools::convert(basin, num);
    std::string str = "COMMITTED TO BASIN " + num;
    ofile.addConstantField(str);
    ofile.printField();
    ofile.flush();
    plumed.stop();
  }
}

}
}
