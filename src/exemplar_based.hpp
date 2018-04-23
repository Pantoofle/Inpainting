#ifndef exemplar_based_hpp_INCLUDED
#define exemplar_based_hpp_INCLUDED

#include <climits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "CImg.h"

#define GRAIN 256
#define CHANNEL 3
#define DIM 2

using namespace cimg_library;

enum class Pxl_stat : char { Known, Filled, Frontiere, Empty };

void ex_based_inpainting(const char *, const char *, const char *, int);

double norm(std::pair<double, double> v);
double vect_scal(std::pair<double, double> a, std::pair<double, double> b);

std::pair<double, double>
grad(CImg<int> &, std::vector<std::vector<Pxl_stat>> &, int, int, int);
std::pair<double, double> normal_vect(std::vector<std::vector<Pxl_stat>> &, int,
                                      int, int);

#endif // exemplar_based_hpp_INCLUDED
