#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Data{

    double k;
    double dt;
    double tMax;
    VectorXd u0;
    vector<int> nId;
    vector<double> a;
    vector<double> bcDir;
    vector<double> bcNeu;
    vector<vector<int>> fId;
    function<VectorXd(MatrixXd)> fun;
    function<vector<VectorXd>(VectorXd)> flux;
};

vector<VectorXd> transport(Mesh mesh,Data data){

    VectorXd u = data.u0;
    vector<VectorXd> U{data.u0};
    int end = round(data.tMax/data.dt);

    // BC and sparse solver
    
    SparseLU<SparseMatrix<double>> solver;
    vector<Face> face = mesh.setFace(data.fId);
    SparseMatrix<double> Sx = mesh.matrix2D("Sx").transpose();
    SparseMatrix<double> Sy = mesh.matrix2D("Sy").transpose();
    SparseMatrix<double> M = mesh.matrix2D("M");
    solver.compute(M);

    // Solves with Euler scheme

    for(int i=0; i<end; i++){

        vector<VectorXd> F = data.flux(u);
        VectorXd B = mesh.neumannBC2(face,F);
        VectorXd S = Sx*F[0]+Sy*F[1];

        u += data.dt*solver.solve(S-B);
        for(int j=0; j<u.size(); j++){u[j] = abs(u[j]);}
        U.push_back(u);
    }
    return U;
}

vector<VectorXd> laplace(Mesh mesh,Data data){

    vector<double> bc = data.bcDir;
    vector<int> nId = data.nId;

    // BC and sparse solver
    
    VectorXd b = mesh.vector1D(data.fun);
    SparseLU<SparseMatrix<double>> solver;
    SparseMatrix<double> K = mesh.dirichletBC(mesh.matrix2D("K"),nId);
    for(int i; i<nId.size(); i++){b(nId[i]) = bc[i];}
    solver.compute(K);

    // Solves with Euler scheme

    VectorXd u = solver.solve(b);
    vector<VectorXd> U{u};
    return U;
}

vector<VectorXd> advection(Mesh mesh,Data data){

    double k = data.k;
    vector<double> a = data.a;
    vector<int> nId = data.nId;
    vector<double> bcDir = data.bcDir;
    vector<double> bcNeu = data.bcNeu;

    // BC and sparse solver

    SparseLU<SparseMatrix<double>> solver;
    SparseMatrix<double> K = mesh.matrix2D("K");
    SparseMatrix<double> Sx = mesh.matrix2D("Sx");
    SparseMatrix<double> Sy = mesh.matrix2D("Sy");
    SparseMatrix<double> A = mesh.dirichletBC(a[0]*Sx+a[1]*Sy+k*K,nId);
    solver.compute(A);

    VectorXd b = mesh.vector1D(data.fun);
    vector<Face> face = mesh.setFace(data.fId);
    VectorXd B = mesh.neumannBC1(face,bcNeu);
    
    for(int i; i<nId.size(); i++){
        b(nId[i]) = bcDir[i];
        B(nId[i]) = 0;
    }

    // Solves with Euler scheme

    VectorXd u = solver.solve(b+B);
    vector<VectorXd> U{u};
    return U;
}

vector<VectorXd> diffusion(Mesh mesh,Data data){

    VectorXd u = data.u0;
    vector<int> nId = data.nId;
    vector<VectorXd> U{data.u0};
    vector<double> bc = data.bcDir;
    int end = round(data.tMax/data.dt);

    // BC and sparse solver

    SparseLU<SparseMatrix<double>> solver;
    SparseMatrix<double> M = mesh.matrix2D("M");
    SparseMatrix<double> MK = M-data.k*data.dt*mesh.matrix2D("K");
    M = mesh.dirichletBC(M,data.nId);
    solver.compute(M);

    // Solves with Euler scheme

    for(int i=0; i<end; i++){

        VectorXd b = MK*u;
        for(int i; i<nId.size(); i++){b(nId[i]) = bc[i];}
        u = solver.solve(b);
        U.push_back(u);
    }
    return U;
}