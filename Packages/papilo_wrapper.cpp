#include <papilo/core/Presolve.hpp>
#include <papilo/io/MPSInput.hpp>
#include <papilo/io/MPSOutput.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>

namespace py = pybind11;
using namespace papilo;

std::string presolve_mps(const std::string& input_mps, const std::string& output_mps, const std::string& transform_file) {
    Papilo presolver;
    Problem<double> problem;

    // Read MPS
    MPSInput input(input_mps);
    input.read(problem);

    // Run presolve
    presolver.loadProblem(problem);
    presolver.presolve();

    // Write presolved problem
    Problem<double> presolved;
    presolver.getPresolvedProblem(presolved);
    MPSOutput output(output_mps);
    output.write(presolved);

    // Save transformation
    std::ofstream tf(transform_file, std::ios::binary);
    presolver.writePresolveTransformation(tf);
    tf.close();

    return "Presolve complete.";
}

std::vector<double> postsolve_solution(const std::vector<double>& x_reduced, const std::string& transform_file) {
    Papilo presolver;
    std::ifstream tf(transform_file, std::ios::binary);
    presolver.readPresolveTransformation(tf);

    std::vector<double> x_original;
    presolver.postsolve(x_reduced, x_original);
    return x_original;
}

PYBIND11_MODULE(papilo_bindings, m) {
    m.def("presolve_mps", &presolve_mps, "Presolve LP and save transformation");
    m.def("postsolve_solution", &postsolve_solution, "Recover original solution");
}
