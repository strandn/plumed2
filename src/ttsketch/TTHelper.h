#ifndef __PLUMED_ttsketch_TTHelper_h
#define __PLUMED_ttsketch_TTHelper_h
#include "BasisFunc.h"
#include "tools/Matrix.h"
#include "itensor/all.h"

namespace PLMD {
namespace ttsketch {

void ttWrite(const std::string& filename, const itensor::MPS& tt, unsigned count);
itensor::MPS ttRead(const std::string& filename, unsigned count);
void ttSumWrite(const std::string& filename, const itensor::MPS& tt, unsigned count);
itensor::MPS ttSumRead(const std::string& filename, unsigned count);
double ttEval(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv);
std::vector<double> ttGrad(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv);
std::tuple<Matrix<double>, std::vector<double>, double> covMat(const itensor::MPS& tt, const std::vector<BasisFunc>& basis);
void marginal2d(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, int pos1, int pos2, std::vector<std::vector<double>>& grid, bool conv);

}
}

#endif
