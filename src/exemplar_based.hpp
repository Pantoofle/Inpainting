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

void ex_based_inpainting(const char *, const char *, const char *, int);

double norm(std::pair<double, double> v);
std::pair<double, double> vect_prod(std::pair<double, double> a,
                                    std::pair<double, double> b);

std::pair<double, double> grad(CImg<int> &, int, int);
std::pair<double, double> normal_vect(CImg<int> &, int, int);

#endif // exemplar_based_hpp_INCLUDED
