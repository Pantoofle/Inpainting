#include "exemplar_based.hpp"

int main(int argc, char **argv) {
  ex_based_inpainting(argv[1], argv[2], argv[3], atoi(argv[4]));
  return 0;
}
