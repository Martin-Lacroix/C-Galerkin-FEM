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
    double du;
    vector<int> nIdx;
    vector<int> nIdy;
    vector<Vector2d> bcNeu;
    vector<double> bcDirX;
    vector<double> bcDirY;
    vector<vector<int>> fId;
};

// Solves a quasi-static linear elsaticity equation ∇·σ(u) = 0

vector<VectorXd> elasticity(Mesh &mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;

    // BC and sparse solver

    SparseLU<SM> solver;
    SM K = mesh.matrix2D(data.E,data.v);
    K = mesh.dirichletBC1(K,nIdx,0);
    K = mesh.dirichletBC1(K,nIdy,1);
    solver.compute(K);

    vector<Face> face = mesh.setFace(data.fId);
    VectorXd B = mesh.neumannBC(face,data.bcNeu);
    B = mesh.dirichletBC2(B,nIdx,data.bcDirX,0);
    B = mesh.dirichletBC2(B,nIdy,data.bcDirY,1);

    // Solves the system

    VectorXd u = solver.solve(B);
    VectorXd e = mesh.strain(u);
    vector<VectorXd> UE{u,e};
    return UE;
}

// Solves a quasi-static linear elsaticity equation ∇·σ(u) = 0

vector<VectorXd> newton(Mesh &mesh,Data data){

    vector<int> nIdx = data.nIdx;
    vector<int> nIdy = data.nIdy;

    // BC and sparse solver

    SparseLU<SM> solver;
    SM K = mesh.matrix2D(data.E,data.v);
    K = mesh.dirichletBC1(K,nIdx,0);
    K = mesh.dirichletBC1(K,nIdy,1);

    vector<Face> face = mesh.setFace(data.fId);
    VectorXd B = mesh.neumannBC(face,data.bcNeu);
    B = mesh.dirichletBC2(B,nIdx,data.bcDirX,0);
    B = mesh.dirichletBC2(B,nIdy,data.bcDirY,1);

    // Solves with Newton-Raphson

    double lam = 0;
    double dl = 1.0/data.step;
    VectorXd u(2*mesh.nNbr);
    VectorXd r(2*mesh.nNbr);
    u.setZero();

    auto force = [K,B](VectorXd u){return K*u-B;};
    SM J = mesh.jacobian(force,u,data.du);
    solver.compute(J);

    while(lam<1){

        // Predictor phase

        lam += dl;
        if(lam>1){lam = 1;}

        r = K*u-lam*B;
        u = solver.solve(J*u-r);

        // Corrector phase

        for(int i=0; i<5; i++){

            r = K*u-lam*B;
            u = solver.solve(J*u-r);
        }
    }

    VectorXd e = mesh.strain(u);
    vector<VectorXd> UE{u,e};
    return UE;
}