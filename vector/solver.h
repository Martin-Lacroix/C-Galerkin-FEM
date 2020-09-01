#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Data{

    double E;
    double v;
    vector<int> nIdx;
    vector<int> nIdy;
    vector<Vector2d> bcNeu;
    vector<double> bcDirX;
    vector<double> bcDirY;
    vector<vector<int>> fId;
};

// Solves a static linear elsaticity equation ∇·σ(u) = 0

vector<VectorXd> elasticity(Mesh mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;

    // BC and sparse solver

    SparseLU<SparseMatrix<double>> solver;
    SparseMatrix<double> A = mesh.matrix2D(data.E,data.v);
    A = mesh.dirichletBC1(A,nIdx,0);
    A = mesh.dirichletBC1(A,nIdy,1);
    solver.compute(A);

    vector<Face> face = mesh.setFace(data.fId);
    VectorXd b = mesh.neumannBC(face,data.bcNeu);
    b = mesh.dirichletBC2(b,nIdx,data.bcDirX,0);
    b = mesh.dirichletBC2(b,nIdy,data.bcDirY,1);

    // Solves with Euler scheme

    VectorXd u = solver.solve(b);
    vector<VectorXd> U{u};
    return U;
}
