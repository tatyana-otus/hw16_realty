#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>

using namespace dlib;

const int N = 7;

const std::array<size_t, N> precision { {6, 6, 0, 2, 2, 2, 0} };

typedef dlib::matrix<double, N, 1> sample_type;

typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;

typedef polynomial_kernel<sample_type> poly_kernel;
typedef radial_basis_kernel<sample_type> rbf_kernel;