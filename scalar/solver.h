#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;
typedef SparseMatrix<double> SM;

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

// Solves the transport equation du(t,x,y)/dt + ∇f(u) = 0

vector<VectorXd> transport(Mesh &mesh,Data data){

    VectorXd u = data.u0;
    vector<VectorXd> U{data.u0};
    int end = round(data.tMax/data.dt);

    // BC and sparse solver
    
    SparseLU<SM> solver;
    vector<Face> face = mesh.setFace(data.fId);
    SM Sx = mesh.matrix2D("Sx").transpose();
    SM Sy = mesh.matrix2D("Sy").transpose();
    SM M = mesh.matrix2D("M");
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

// Solves the laplace equation -Δu(x,y) = f(x,y)

vector<VectorXd> laplace(Mesh &mesh,Data data){

    vector<double> bc = data.bcDir;
    vector<int> nId = data.nId;

    // BC and sparse solver
    
    VectorXd b = mesh.vector1D(data.fun);
    SM K = mesh.dirichletBC(mesh.matrix2D("K"),nId);
    for(int i; i<nId.size(); i++){b(nId[i]) = bc[i];}
    SparseLU<SM> solver;
    solver.compute(K);

    // Solves with Euler scheme

    VectorXd u = solver.solve(b);
    vector<VectorXd> U{u};
    return U;
}

// Solves the steady advection-diffusion equation a·∇u(x,y) - kΔu(x,y) = 0

vector<VectorXd> advection(Mesh &mesh,Data data){

    double k = data.k;
    vector<double> a = data.a;
    vector<int> nId = data.nId;
    vector<double> bcDir = data.bcDir;
    vector<double> bcNeu = data.bcNeu;

    // BC and sparse solver

    SM K = mesh.matrix2D("K");
    SM Sx = mesh.matrix2D("Sx");
    SM Sy = mesh.matrix2D("Sy");
    SM A = mesh.dirichletBC(a[0]*Sx+a[1]*Sy+k*K,nId);
    SparseLU<SM> solver;
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

// Solves the unsteady diffusion equation du(t,x,y)/dt - kΔu(t,xy) = 0

vector<VectorXd> diffusion(Mesh &mesh,Data data){

    VectorXd u = data.u0;
    vector<int> nId = data.nId;
    vector<VectorXd> U{data.u0};
    vector<double> bc = data.bcDir;
    int end = round(data.tMax/data.dt);

    // BC and sparse solver

    SM M = mesh.matrix2D("M");
    SM MK = M-data.k*data.dt*mesh.matrix2D("K");
    M = mesh.dirichletBC(M,nId);
    SparseLU<SM> solver;
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