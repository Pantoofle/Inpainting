#ifndef exemplar_based_hpp_INCLUDED
#define exemplar_based_hpp_INCLUDED

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "CImg.h"

#define PATCH_SIZE 9
#define HALF_PATCH 4
#define GRAIN 256
#define CHANNEL 3
#define DIM 2

void ex_based_inpainting(const char *, const char *, const char *);

double norm(std::pair<double, double> v);
std::pair<double, double> vect_prod(std::pair<double, double> a,
                                    std::pair<double, double> b);

#endif // exemplar_based_hpp_INCLUDED