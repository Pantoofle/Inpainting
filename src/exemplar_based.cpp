#include "exemplar_based.hpp"

using namespace cimg_library;
using namespace std;

void ex_based_inpainting(const char *src_path, const char *mask_path,
                         const char *result_path, int patch_size) {

  int half_patch = patch_size / 2;
  // LOAD IMAGES AND MASK
  clog << "Loading image and mask" << endl;
  CImg<int> src(src_path);
  CImg<int> mask(mask_path);
  CImg<int> result(src_path);

  clog << "Source :" << src_path << endl;
  clog << "Mask   :" << mask_path << endl;
  clog << "Result :" << result_path << endl;

  int width = src.width();
  int height = src.height();

  clog << "Width  : " << width << endl;
  clog << "Height : " << height << endl;

  vector<vector<Pxl_stat>> status = vector<vector<Pxl_stat>>(
      width, vector<Pxl_stat>(height, Pxl_stat::Known));
  int frontiere_points = 0;

  if (src.width() != mask.width() || src.height() != mask.height())
    throw(runtime_error("Images must have the same size"));

  // The vector containing C(p)
  vector<vector<double>> confidence =
      vector<vector<double>>(width, vector<double>(height, 0.));

  // INITIATE CONFIDENCE/DATA/FRONTIERE
  clog << "Initiate the values." << endl;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      // If we are in the zone to fill
      if (mask(i, j, 0) < 20) {
        // Erase the previous value
        result(i, j, 0, 0) = result(i, j, 0, 1) = result(i, j, 0, 2) = 0;
        status[i][j] = Pxl_stat::Empty;
        // Check if at frontiere
        if ((i > 0 && mask(i - 1, j, 0) != 0) ||
            (i < width - 1 && mask(i + 1, j, 0) != 0) ||
            (j > 0 && mask(i, j - 1, 0) != 0) ||
            (j < height - 1 && mask(i, j + 1, 0) != 0)) {
          status[i][j] = Pxl_stat::Frontiere;
          frontiere_points++;
        }
      } else {
        // If we are in the known zone, we set the confidence to maximum
        confidence[i][j] = 1.;
      }
    }
  }

  // for (int c = 0; c < CHANNEL; c++)
  // clog << "Pixel " << result(0, 0, 0, c) << endl;

  // pair<double, double> g = grad(result, status, 0, 0, half_patch);
  // clog << "Grad :" << get<0>(g) << " and " << get<1>(g) << endl;

  // while (true) {
  //}

  // result.save("../img/start.jpg");

  // ALGO
  clog << "Frontiere points : " << frontiere_points << endl;
  clog << "Starting main loop" << endl;
  while (frontiere_points != 0) {
    // Update P(p) for points in the front and find the argmax patch
    int max_i = -1;
    int max_j = -1;
    double max_P = -1.;

    //#pragma omp parallel for
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        // Keep only the frontiere points
        if (status[i][j] != Pxl_stat::Frontiere)
          continue;
        // Compute the confidence
        double c = 0.;
        for (int a = -half_patch; a <= half_patch; a++) {
          if (i + a < 0 || i + a >= width)
            continue;
          for (int b = -half_patch; b <= half_patch; b++) {
            if (j + b < 0 || j + b >= height)
              continue;
            c += confidence[i + a][j + b];
          }
        }
        confidence[i][j] = c / (patch_size * patch_size);

        double data;
        pair<double, double> g, nv;
        g = grad(result, status, i, j, half_patch);
        nv = normal_vect(status, i, j, half_patch);

        // clog << "Grad :" << get<0>(g) << ", " << get<1>(g) << endl;
        // clog << "NVec :" << get<0>(nv) << ", " << get<1>(nv) << endl;

        data = vect_scal(g, nv) / (GRAIN * CHANNEL);

        // if (data != 0)
        // cout << "Data : " << data << endl;

        double P = data * confidence[i][j];
        // clog << i << " - " << j << " -> P: " << P << " (c: " <<
        // confidence[i][j]
        //<< ", d:" << data << ")" << endl;

        if (P > max_P) {
          max_P = P;
          // clog << "Update max" << endl;
          max_i = i;
          max_j = j;
        }
      }
    }
    // clog << "Most interesting point : " << max_i << ", " << max_j << endl;

    // The maximal patch is at max_i, max_j
    // We look for the patch p' in the image that minimises d(\psi_p, \psi_p')
    // clog << "\tFind optimal patch" << endl;
    int opt_i = -1;
    int opt_j = -1;
    unsigned long long opt_d = LONG_LONG_MAX;

    //#pragma omp parallel for
    for (int i = half_patch; i < width - half_patch; i++) {
      for (int j = half_patch; j < height - half_patch; j++) {
        if (i == max_i && j == max_j)
          continue;
        // For each point, we compute the distance of the patch

        unsigned long long d = 0;
        for (int a = -half_patch; a <= half_patch; a++) {
          if (max_i + a < 0 || max_i + a >= width)
            continue;

          for (int b = -half_patch; b <= half_patch; b++) {
            if (max_j + b < 0 || max_j + b >= height)
              continue;
            if (status[i + a][j + b] != Pxl_stat::Known)
              goto skip;

            // If the pixel is already filled, compute the distance
            if (status[max_i + a][max_j + b] == Pxl_stat::Known ||
                status[max_i + a][max_j + b] == Pxl_stat::Filled) {
              for (int c = 0; c < CHANNEL; c++) {
                d += (result(i + a, j + b, 0, c) -
                      result(max_i + a, max_j + b, 0, c)) *
                     (result(i + a, j + b, 0, c) -
                      result(max_i + a, max_j + b, 0, c));
              } // for channel
            }   // if filled
          }     // for b
        }       // for a

        if (d < opt_d) {
          // clog << "At " << i << ", " << j << endl;
          // clog << "To " << max_i << ", " << max_j << endl;
          // clog << "New Dist : " << d << endl;
          // clog << "Old optimal Dist : " << opt_d << endl;

          opt_d = d;
          opt_i = i;
          opt_j = j;
        }
      // Jump here if there is an unknown pixel in the patch
      skip:
        continue;
      } // For j
    }   // For i

    // We have the optimal patch. Now, copy it
    // clog << "\tCopy patch" << endl;
    // clog << "\t\tFrom " << opt_i << ", " << opt_j << endl;
    // clog << "\t\tTo   " << max_i << ", " << max_j << endl;
    // clog << "\t\tDistance " << opt_d << endl;
    for (int a = -half_patch; a <= half_patch; a++) {
      if (opt_i + a < 0 || opt_i + a >= width)
        continue;
      if (max_i + a < 0 || max_i + a >= width)
        continue;

      for (int b = -half_patch; b <= half_patch; b++) {
        if (opt_j + b < 0 || opt_j + b >= height)
          continue;
        if (max_j + b < 0 || max_j + b >= height)
          continue;

        if (status[opt_i + a][opt_j + b] == Pxl_stat::Known &&
            status[max_i + a][max_j + b] != Pxl_stat::Known) {

          if (status[max_i + a][max_j + b] == Pxl_stat::Frontiere) {
            frontiere_points--;
          }
          status[max_i + a][max_j + b] = Pxl_stat::Filled;
          for (int c = 0; c < CHANNEL; c++)
            result(max_i + a, max_j + b, 0, c) =
                result(opt_i + a, opt_j + b, 0, c);
        }
      }
    }

    // Update the confidence value at the center
    // clog << "\tCompute confidence at the center" << endl;
    double c = 0.;
    for (int a = -half_patch; a <= half_patch; a++) {
      if (max_i + a < 0 || max_i + a >= width)
        continue;
      for (int b = -half_patch; b <= half_patch; b++) {
        if (max_j + b < 0 || max_j + b >= height)
          continue;
        c += confidence[max_i + a][max_j + b];
      }
    }
    confidence[max_i][max_j] = c / (patch_size * patch_size);

    // Propagates the confidence to the rest of the patch
    // clog << "\tUpdate the confidence in the patch" << endl;
    for (int a = -half_patch; a <= half_patch; a++) {
      if (max_i + a < 0 || max_i + a >= width)
        continue;

      for (int b = -half_patch; b <= half_patch; b++) {
        if (max_j + b < 0 || max_j + b >= height)
          continue;
        if (status[max_i + a][max_j + b] == Pxl_stat::Known)
          continue;
        confidence[max_i + a][max_j + b] = confidence[max_i][max_j];
      }
    }

    // Update the new frontiere, the contour of the patch
    // clog << "\tUpdate frontiere" << endl;
    for (int a = -half_patch - 1; a <= half_patch + 1; a++) {
      if (max_i + a < 0 || max_i + a >= width)
        continue;

      int x = max_i + a;
      for (int b = -half_patch - 1; b <= half_patch + 1; b++) {
        if (max_j + b < 0 || max_j + b >= height)
          continue;

        int y = max_j + b;
        if (status[x][y] != Pxl_stat::Empty)
          continue;

        if (y - 1 > 0 && (status[x][y - 1] == Pxl_stat::Filled ||
                          status[x][y - 1] == Pxl_stat::Known)) {
          frontiere_points++;
          status[x][y] = Pxl_stat::Frontiere;
          continue;
        }
        if (y + 1 < height && (status[x][y + 1] == Pxl_stat::Filled ||
                               status[x][y + 1] == Pxl_stat::Known)) {
          frontiere_points++;
          status[x][y] = Pxl_stat::Frontiere;
          continue;
        }
        if (x - 1 > 0 && (status[x - 1][y] == Pxl_stat::Filled ||
                          status[x - 1][y] == Pxl_stat::Known)) {
          frontiere_points++;
          status[x][y] = Pxl_stat::Frontiere;
          continue;
        }
        if (x + 1 > 0 && (status[x + 1][y] == Pxl_stat::Filled ||
                          status[x + 1][y] == Pxl_stat::Known)) {
          frontiere_points++;
          status[x][y] = Pxl_stat::Frontiere;
          continue;
        }
      }
    }
    clog << "Frontiere points left : " << frontiere_points << endl;
    result.save(result_path);

    // CImg<int> canvas(width, height, 1, 3, 0);
    // for (int i = 0; i < width; i++) {
    // for (int j = 0; j < height; j++) {
    // switch (status[i][j]) {
    // case Pxl_stat::Filled:
    // canvas(i, j, 0) = 255;
    // break;
    // case Pxl_stat::Frontiere:
    // canvas(i, j, 1) = 255;
    // break;
    // case Pxl_stat::Empty:
    // canvas(i, j, 2) = 255;
    // break;
    // case Pxl_stat::Known:
    // canvas(i, j, 2) = 0;
    // break;
    //}
    // if (i == max_i && j == max_j)
    // canvas(i, j, 1) = canvas(i, j, 2) = 255;
    // if (i == opt_i && j == opt_j)
    // canvas(i, j, 0) = canvas(i, j, 2) = 255;
    //}
    //}
    // canvas.save("../img/can1.jpg");
  }
  // END_ALGO

  // SAVE THE RESULT
  clog << "Save to result path" << endl;
  result.save(result_path);
  clog << "Done" << endl;
}

double norm(pair<double, double> v) {
  return sqrt(get<0>(v) * get<0>(v) + get<1>(v) * get<1>(v));
}

double vect_scal(pair<double, double> a, pair<double, double> b) {
  return get<0>(a) * get<0>(b) + get<1>(a) * get<1>(b);
}

pair<double, double> grad(CImg<int> &img, vector<vector<Pxl_stat>> &status,
                          int i, int j, int half_patch) {
  int gx = 0;
  int gy = 0;
  for (int a = -half_patch; a <= half_patch; a++) {
    if (i + a - 1 < 0 || i + a >= img.width())
      continue;

    for (int b = -half_patch; b <= half_patch; b++) {
      if (j + b - 1 < 0 || j + b >= img.height())
        continue;

      if (status[i + a][j + b] == Pxl_stat::Empty ||
          status[i + a][j + b] == Pxl_stat::Frontiere ||
          status[i + a - 1][j + b] == Pxl_stat::Empty ||
          status[i + a - 1][j + b] == Pxl_stat::Frontiere ||
          status[i + a][j + b - 1] == Pxl_stat::Empty ||
          status[i + a][j + b - 1] == Pxl_stat::Frontiere)
        continue;
      for (int c = 0; c < CHANNEL; c++) {

        // clog << "i':" << i + a << ", j':" << j + b << endl;
        // clog << "d1:"
        //<< abs(img(i + a, j + b, 0, c) - img(i + a - 1, j + b, 0, c))
        //<< endl;
        // clog << "d2:"
        //<< abs(img(i + a, j + b, 0, c) - img(i + a, j + b - 1, 0, c))
        //<< endl;

        gx += abs(img(i + a, j + b, 0, c) - img(i + a - 1, j + b, 0, c));
        gy += abs(img(i + a, j + b, 0, c) - img(i + a, j + b - 1, 0, c));
      }
    }
  }
  // if (gx != 0 or gy != 0)
  // clog << "Ping" << endl;
  return pair<double, double>((double)gy, (double)gx);
}

pair<double, double> normal_vect(vector<vector<Pxl_stat>> &status, int i, int j,
                                 int half_patch) {

  int nx = 0;
  int ny = 0;
  for (int a = -half_patch; a <= half_patch; a++) {
    if (i + a < 0 || i + a >= status.size())
      continue;

    for (int b = -half_patch; b <= half_patch; b++) {
      if (j + b < 0 || j + b >= (status[0]).size())
        continue;

      if (status[i + a][j + b] == Pxl_stat::Frontiere) {
        nx += abs(a);
        ny += abs(b);
      }
    }
  }

  // Normalize the vector
  double N = norm(pair<double, double>((double)nx, (double)ny));
  if (N != 0.)
    return pair<double, double>(((double)nx) / N, ((double)ny) / N);
  else
    return pair<double, double>(((double)nx), ((double)ny));
}
