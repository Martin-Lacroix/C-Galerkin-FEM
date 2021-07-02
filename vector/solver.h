#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;
typedef SparseMatrix<double> SM;

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

// Creates a linear stiffness tensor

Matrix3d stiffness(double E,double v){

    Matrix3d D;
    D.row(0) = Vector3d {1-v,v,0};
    D.row(1) = Vector3d {v,1-v,0};
    D.row(2) = Vector3d {0,0,(1-2*v)/2};
    D *= E/((1+v)*(1-2*v));

    return D;
}

// Solves the equation of motion ∇·σ(u) = 0 in linear elasticity

VectorXd solve(Mesh &mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;
    Matrix3d D = stiffness(data.E,data.v);

    // Finite strain theory solver

    SparseLU<SM> solver;
    vector<Face> face = mesh.setFace(data.fId);
    VectorXd dB = mesh.neumannBC(face,data.bcNeu);
    dB = mesh.dirichletBC2(dB,nIdx,data.bcDirX,0);
    dB = mesh.dirichletBC2(dB,nIdy,data.bcDirY,1);

    SM K = mesh.matrix2D(D);
    K = mesh.dirichletBC1(K,nIdx,0);
    K = mesh.dirichletBC1(K,nIdy,1);

    solver.compute(K);
    VectorXd u = solver.solve(dB);
    mesh.update(u);

    return u;
}
