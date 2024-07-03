#include <Eigen/QR>
#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "itensor/all.h"
#include "BasisFunc.h"

using namespace std;
using namespace Eigen;
using namespace itensor;
using namespace PLMD::bias;

namespace PLMD {
namespace ttsketch {

class TTSketch : public Bias {

private:
  int rc_;
  double temp_;
  int pace_;
  int stride_;
  vector<MPS> rholist_;
  vector<double> rhomaxlist_;
  vector<BasisFunc> basis_;
  vector<vector<double>> samples_;
  double vmax_;
  double vshift_;
  int count_;

public:
  explicit TTSketch(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate();
  void paraSketch();
  MPS createTTCoeff(int n, int d) const;
  pair<vector<ITensor>, IndexSet> intBasisSample(IndexSet const& is) const;
  static tuple<MPS, vector<ITensor>, vector<ITensor>> formTensorMoment(vector<ITensor> const& M, MPS const& coeff, IndexSet const& is) const;
  double densEval(int step, vector<double> const& elements) const;
  vector<double> densGrad(int step, vector<double> const& elements) const;
};

PLUMED_REGISTER_ACTION(TTSketch, "TTSKETCH")

TTSketch::TTSketch(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
  vmax_(numeric_limits<double>::max()),
  vshift_(0.0),
  count_(0)
{

}

void TTSketch::registerKeywords(Keywords& keys) {

}

void TTSketch::calculate() {

}

void TTSketch::paraSketch() {

}

MPS TTSketch::createTTCoeff(int n, int d) {
  
}

pair<vector<ITensor>, IndexSet> TTSketch::intBasisSample(IndexSet const& is) const {

}

static tuple<MPS, vector<ITensor>, vector<ITensor>> TTSketch::formTensorMoment(vector<ITensor> const& M,
                                                                               MPS const& coeff,
                                                                               IndexSet const& is) const {
  
}

double TTSketch::densEval(int step, vector<double> const& elements) const {

}

vector<double> TTSketch::densGrad(int step, vector<double> const& elements) const {

}

}
}
