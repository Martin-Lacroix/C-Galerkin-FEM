#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;
typedef SparseMatrix<double> SM;

struct Data{

    int step;
    double E;
    double v;
    double Et;
    double ys;
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
    D.row(0) = Vector3d {1,v,0};
    D.row(1) = Vector3d {v,1,0};
    D.row(2) = Vector3d {0,0,(1-v)/2};
    D *= E/(1-v*v);

    return D;
}

// Solves the equation of motion ∇·σ(u) = 0 in linear elasto-plasticity

VectorXd newton(Mesh &mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;
    Matrix3d Dp = stiffness(data.Et,0.5);
    Matrix3d De = stiffness(data.E,data.v);

    // BC and sparse solver

    vector<Face> face = mesh.setFace(data.fId);
    VectorXd B = mesh.neumannBC(face,data.bcNeu);
    B = mesh.dirichletBC2(B,nIdx,data.bcDirX,0);
    B = mesh.dirichletBC2(B,nIdy,data.bcDirY,1);

    // Solves with Newton-Raphson

    VectorXd dB = B/data.step;
    VectorXd u(2*mesh.nNbr);
    SparseLU<SM> solver;
    u.setZero();

       for(int i=0; i<data.step; i++){

        SM K = mesh.matrix2D(De,Dp,u,data.ys);
        K = mesh.dirichletBC1(K,nIdx,0);
        K = mesh.dirichletBC1(K,nIdy,1);

        solver.compute(K);
        VectorXd du = solver.solve(dB);
        //mesh.update(du);
        u += du;
    }
    return u;
}
