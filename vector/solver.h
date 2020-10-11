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

VectorXd newton(Mesh &mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;
    Matrix3d D = stiffness(data.E,data.v);

    // BC and sparse solver
    
    vector<Vector2d> bcNeu = data.bcNeu;
    vector<Vector2d> dBC(bcNeu.size());
    for(int i=0; i<bcNeu.size(); i++){dBC[i] = bcNeu[i]/data.step;}

    // Updated Lagrangian solver

    VectorXd u(2*mesh.nNbr);
    SparseLU<SM> solver;
    u.setZero();

       for(int i=0; i<data.step+1; i++){

        vector<Face> face = mesh.setFace(data.fId);
        VectorXd dB = mesh.neumannBC(face,dBC);
        dB = mesh.dirichletBC2(dB,nIdx,data.bcDirX,0);
        dB = mesh.dirichletBC2(dB,nIdy,data.bcDirY,1);

        SM K = mesh.matrix2D(D);
        K = mesh.dirichletBC1(K,nIdx,0);
        K = mesh.dirichletBC1(K,nIdy,1);

        solver.compute(K);
        VectorXd du = solver.solve(dB);
        mesh.update(du);
        u += du;
    }
    return u;
}
