#include "function.h"

double sigmoidFunction(const double x, double alpha) {
    return 0.5 * (x * alpha / (1 + abs(x * alpha))) + 0.5;
}