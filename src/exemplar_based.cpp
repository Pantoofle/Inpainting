#include "exemplar_based.hpp"

using namespace cimg_library;
using namespace std;

void ex_based_inpainting(const char *src_path, const char *mask_path,
                         const char *result_path) {

  // LOAD IMAGES AND MASK
  clog << "Loading image and mask" << endl;
  CImg<int> src(src_path);
  CImg<int> mask(mask_path);
  CImg<int> result(src_path);

  int width = src.width();
  int height = src.height();

  CImg<int> frontiere(width, height, 1, 1, 0);
  int frontiere_points = 0;

  if (src.width() != mask.width() || src.height() != mask.height())
    throw(runtime_error("Images must have the same size"));

  // The vector containing C(p)
  vector<vector<double>> confidence =
      vector<vector<double>>(width, vector<double>(height, 0.));

  // The vector containing D(p)
  vector<vector<double>> data =
      vector<vector<double>>(width, vector<double>(height, 0.));

  // The vector containing n(p)
  vector<vector<pair<double, double>>> normal_vect =
      vector<vector<pair<double, double>>>(
          width,
          vector<pair<double, double>>(height, pair<double, double>(0., 0.)));

  // The vector containing \delta I_p^{\bot}
  vector<vector<pair<double, double>>> grad =
      vector<vector<pair<double, double>>>(
          width,
          vector<pair<double, double>>(height, pair<double, double>(0., 0.)));

  // INITIATE CONFIDENCE/DATA/FRONTIERE
  clog << "Initiate the values." << endl;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      // If we are in the zone to fill
      if (mask(i, j, 0) == 0) {
        // Erase the previous value
        result(i, j, 0) = result(i, j, 1) = result(i, j, 2) = 0;
        // Check if at frontiere
        if ((i > 0 && mask(i - 1, j, 0) != 0) ||
            (i < width - 1 && mask(i + 1, j, 0) != 0) ||
            (j > 0 && mask(i, j - 1, 0) != 0) ||
            (j < height - 1 && mask(i, j + 1, 0) != 0)) {
          frontiere(i, j, 0) = 1;
          frontiere_points++;
          result(i, j, 0) = 255;
        }
      } else {
        // If we are in the known zone, we set the confidence to maximum
        confidence[i][j] = 1.;
      }
    }
  }

  // ALGO
  clog << "Starting main loop" << endl;
  while (frontiere_points != 0) {
    // Update P(p) for points in the front
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        // Keep only the frontiere points
        if (frontiere(i, j, 0) != 1)
          continue;

        // At the frontiere, update C(p) and D(p)

        // Update C(p)
        // Sum the confidence values on the patch
        double c = 0.;
        for (int a = -HALF_PATCH; a < HALF_PATCH; a++) {
          if (i + a < 0 || i + a >= width)
            continue;

          for (int b = -HALF_PATCH; b < HALF_PATCH; b++) {
            if (i + b < 0 || i + b >= width)
              continue;
            if (mask(i + a, j + b, 0) != 0.)
              c += confidence[i + a][j + b];
          }
        }
        // Compute the new C(p)
        confidence[i][j] = c / (PATCH_SIZE * PATCH_SIZE);

        // Update D(p)
        data[i][j] =
            norm(vect_prod(grad[i][j], normal_vect[i][j])) / (GRAIN * CHANNEL);
      }
    }
  }
  // END_ALGO

  // SAVE THE RESULT
  clog << "Save to result path" << endl;
  result.save(result_path);
  clog << "Done" << endl;
}

double norm(pair<double, double> v) {
  return sqrt(get<0>(v) * get<0>(v) + get<1>(v) * get<1>(v));
}

pair<double, double> vect_prod(pair<double, double> a, pair<double, double> b) {
  return pair<double, double>(get<0>(a) * get<0>(b), get<1>(a) * get<1>(b));
}
