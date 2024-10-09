#ifndef __PLUMED_ttsketch_TTHelper_h
#define __PLUMED_ttsketch_TTHelper_h
#include "BasisFunc.h"
#include "itensor/all.h"

namespace PLMD {
namespace ttsketch {

void ttWrite(const itensor::MPS& tt, unsigned count);
itensor::MPS ttRead(unsigned count);
double ttEval(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv);
std::vector<double> ttGrad(const itensor::MPS& tt, const std::vector<BasisFunc>& basis, const std::vector<double>& elements, bool conv);

}
}

#endif
