#include "BasisFunc.h"
#include "TTCross.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/ActionShortcut.h"

using namespace std;
using namespace itensor;

namespace PLMD {
namespace ttsketch {

class TTFreeEnergy : public Action {
private:
  MPS vb_;
  vector<BasisFunc> basis_;
  vector<vector<double>> samples_;
  double kbt_;
  unsigned d_;
  int n_;
  double cutoff_;
  int maxrank_;
  vector<vector<vector<double>>> I_;
  vector<vector<vector<double>>> J_;
  vector<vector<double>> u_;
  vector<vector<double>> v_;
  vector<double> resfirst_;
  int aca_n_;
  double aca_epsabs_;
  double aca_epsrel_;
  int aca_limit_;
  int aca_key_;
public:
  explicit TTFreeEnergy(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void update() override { }
  void calculate() override { }
  void apply() override { }
};

PLUMED_REGISTER_ACTION(TTFreeEnergy,"TT_FES")

void TTFreeEnergy::registerKeywords(Keywords& keys) {
  ActionShortcut::registerKeywords(keys);
  keys.add("compulsory", "ARG", "Positions of arguments that you would like to make the free energy for");
  keys.add("compulsory", "TEMP", "The system temperature");
  keys.add("compulsory", "GRID_MIN", "The minimum to use for the grid");
  keys.add("compulsory", "GRID_MAX", "The maximum to use for the grid");
  keys.add("compulsory", "GRID_BIN", "The number of bins to use for the grid");
  keys.add("compulsory", "ACA_CUTOFF", "1.0e-6", "Convergence threshold for TT-cross calculations");
  keys.add("compulsory", "ACA_RANK", "50", "Largest possible rank for TT-cross calculations");
  keys.add("compulsory", "ACA_N", "10000000", "Size of integration workspace");
  keys.add("compulsory", "ACA_EPSABS", "1.0e-12", "Absolute error limit for integration");
  keys.add("compulsory", "ACA_EPSREL", "1.0e-8", "Relative error limit for integration");
  keys.add("compulsory", "ACA_LIMIT", "10000000", "Maximum number of subintervals for integration");
  keys.add("compulsory", "ACA_KEY", "6", "Integration rule");
}

TTFreeEnergy::TTFreeEnergy(const ActionOptions& ao) :
  Action(ao),
  ActionShortcut(ao)
{
  
}

}
}
