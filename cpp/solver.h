#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Solution{
    vector<VectorXd> uStep;
    vector<double> tStep;
};

Solution solver(Mesh mesh,VectorXd u0,vector<double> tVec,vector<double> flux){

    Solution sol;
    VectorXd u = u0;
    double dt = tVec[1];
    vector<double> tStep{0};
    vector<VectorXd> uStep{u0};
    int end = round(tVec[0]/dt);

    // Flux and sparse solver

    auto Fx = [flux](VectorXd u){return flux[0]*u;};
    auto Fy = [flux](VectorXd u){return flux[1]*u;};
    
    SimplicialLDLT<SparseMatrix<double>> sparSolve;
    sparSolve.compute(mesh.M);

    // Solves with Euler scheme

    for(int i=0; i<end; i++){

        VectorXd fx = Fx(u);
        VectorXd fy = Fy(u);

        VectorXd F = mesh.flux(fx,fy);
        VectorXd S = mesh.Sx*fx+mesh.Sy*fy;
        u += dt*sparSolve.solve(S-F);

        for(int j=0; j<u.size(); j++){u[j] = abs(u[j]);}

        uStep.push_back(u);
        tStep.push_back((i+1)*dt);
    }

    sol.tStep = tStep;
    sol.uStep = uStep;
    return sol;
}
