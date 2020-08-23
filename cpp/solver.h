#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Data{

    double dt;
    double tMax;
    VectorXd u0;
    vector<double> flux;
    vector<vector<int>> fId;
};

vector<VectorXd> transport(Mesh mesh,Data data){

    double dt = data.dt;
    VectorXd u = data.u0;
    vector<VectorXd> U{data.u0};
    int end = round(data.tMax/dt);
    vector<double> flux = data.flux;
    vector<vector<int>> fId = data.fId;

    // BC and sparse solver

    auto Fx = [flux](VectorXd u){return flux[0]*u;};
    auto Fy = [flux](VectorXd u){return flux[1]*u;};
    
    vector<Face> face = mesh.precompute(fId);
    SimplicialLDLT<SparseMatrix<double>> sparSolve;
    sparSolve.compute(mesh.M);

    // Solves with Euler scheme

    for(int i=0; i<end; i++){

        vector<VectorXd> F{Fx(u),Fy(u)};
        VectorXd B = mesh.neumannVar(face,F);
        VectorXd S = mesh.Sx.transpose()*F[0]+mesh.Sy.transpose()*F[1];

        u += dt*sparSolve.solve(S-B);
        for(int j=0; j<u.size(); j++){u[j] = abs(u[j]);}
        U.push_back(u);
    }
    return U;
}
