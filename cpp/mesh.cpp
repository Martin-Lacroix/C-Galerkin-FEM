#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"
#include <omp.h>

using namespace std;
using namespace Eigen;

// Triangle element

Triangle::Triangle(vector<vector<double>> nXY){

    Matrix2d invJ;
    Vector3d drN{-1,1,0};
    Vector3d dsN{-1,0,1};
    Vector4d wei{-27.0/96,25.0/96,25.0/96,25.0/96};
    vector<vector<double>> gRS{{1.0/3,1.0/3},{0.6,0.2},{0.2,0.6},{0.2,0.2}};

    // Computes Jacobian

    double J11 = drN(0)*nXY[0][0]+drN(1)*nXY[1][0]+drN(2)*nXY[2][0];
    double J12 = drN(0)*nXY[0][1]+drN(1)*nXY[1][1]+drN(2)*nXY[2][1];
    double J21 = dsN(0)*nXY[0][0]+dsN(1)*nXY[1][0]+dsN(2)*nXY[2][0];
    double J22 = dsN(0)*nXY[0][1]+dsN(1)*nXY[1][1]+dsN(2)*nXY[2][1];

    double detJ = J11*J22-J12*J21;

    invJ(0,0) = J22/detJ;
    invJ(1,1) = J11/detJ;
    invJ(0,1) = -J12/detJ;
    invJ(1,0) = -J21/detJ;

    N = MatrixXd(3,4);
    MatrixXd dxN(3,4);
    MatrixXd dyN(3,4);

    // Mass and stifness matrices

    for(int i=0; i<4; i++){
        
        N.col(i) = Vector3d {1-gRS[i][0]-gRS[i][1],gRS[i][0],gRS[i][1]};
        dxN.col(i) = drN*invJ(0,0)+dsN*invJ(0,1);
        dyN.col(i) = drN*invJ(1,0)+dsN*invJ(1,1);
    }

    M = N*wei.asDiagonal()*(N.transpose())*detJ;
    Sx = N*wei.asDiagonal()*(dxN.transpose())*detJ;
    Sy = N*wei.asDiagonal()*(dyN.transpose())*detJ;
}

// Quadrangle element

Quadrangle::Quadrangle(vector<vector<double>> nXY){

    VectorXd wei(9);
    vector<vector<double>> gRS;
    
    wei << 25.0/81,25.0/81,25.0/81,25.0/81,40.0/81,40.0/81,40.0/81,40.0/81,64.0/81;
    gRS.push_back({-sqrt(3.0/5),-sqrt(3.0/5)});
    gRS.push_back({sqrt(3.0/5),-sqrt(3.0/5)});
    gRS.push_back({-sqrt(3.0/5),sqrt(3.0/5)});
    gRS.push_back({sqrt(3.0/5),sqrt(3.0/5)});
    gRS.push_back({0,-sqrt(3.0/5)});
    gRS.push_back({-sqrt(3.0/5),0});
    gRS.push_back({sqrt(3.0/5),0});
    gRS.push_back({0,sqrt(3.0/5)});
    gRS.push_back({0,0});

    N = MatrixXd(4,9);
    MatrixXd dxN(4,9);
    MatrixXd dyN(4,9);
    MatrixXd drN(4,9);
    MatrixXd dsN(4,9);
    VectorXd invJ11(9);
    VectorXd invJ12(9);
    VectorXd invJ21(9);
    VectorXd invJ22(9);
    VectorXd detJ(9);

    dxN.setZero();
    dyN.setZero();

    // Computes Jacobian

    auto J11 = [nXY](double s){return ((s-1)*nXY[0][0]+(1-s)*nXY[1][0]+(s+1)*nXY[2][0]-(s+1)*nXY[3][0])/4;};
    auto J12 = [nXY](double s){return ((s-1)*nXY[0][1]+(1-s)*nXY[1][1]+(s+1)*nXY[2][1]-(s+1)*nXY[3][1])/4;};
    auto J21 = [nXY](double r){return ((r-1)*nXY[0][0]-(r+1)*nXY[1][0]+(r+1)*nXY[2][0]+(1-r)*nXY[3][0])/4;};
    auto J22 = [nXY](double r){return ((r-1)*nXY[0][1]-(r+1)*nXY[1][1]+(r+1)*nXY[2][1]+(1-r)*nXY[3][1])/4;};

    for (int i=0; i<9; i++){

        detJ(i) = J11(gRS[i][1])*J22(gRS[i][0])-J12(gRS[i][1])*J21(gRS[i][0]);

        invJ11(i) = J22(gRS[i][0])/detJ(i);
        invJ12(i) = -J12(gRS[i][1])/detJ(i);
        invJ21(i) = -J21(gRS[i][0])/detJ(i);
        invJ22(i) = J11(gRS[i][1])/detJ(i);

        drN.col(i) = Vector4d {(gRS[i][1]-1)/4,(1-gRS[i][1])/4,(gRS[i][1]+1)/4,-(gRS[i][1]+1)/4};
        dsN.col(i) = Vector4d {(gRS[i][0]-1)/4,-(gRS[i][0]+1)/4,(gRS[i][0]+1)/4,(1-gRS[i][0])/4};
    }

    // Local shape functions

    for(int i=0; i<9; i++){

        N(0,i) = (1-gRS[i][0])*(1-gRS[i][1])/4;
        N(1,i) = (1+gRS[i][0])*(1-gRS[i][1])/4;
        N(2,i) = (1+gRS[i][0])*(1+gRS[i][1])/4;
        N(3,i) = (1-gRS[i][0])*(1+gRS[i][1])/4;

        for(int j=0; j<4; j++){
            dxN(j,i) += drN(j,i)*invJ11(i)+dsN(j,i)*invJ12(i);
            dyN(j,i) += drN(j,i)*invJ12(i)+dsN(j,i)*invJ22(i);
        }
    }

    M = N*(wei.asDiagonal()*detJ).asDiagonal()*(N.transpose());
    Sx = N*(wei.asDiagonal()*detJ).asDiagonal()*(dxN.transpose());
    Sy = N*(wei.asDiagonal()*detJ).asDiagonal()*(dyN.transpose());
}

// Face element

Face::Face(vector<vector<double>> nXY,vector<int> fId_in){

    MatrixXd N(2,3);
    Vector3d wei{5.0/9,5.0/9,8.0/9};
    vector<double> gRS{-sqrt(3.0/5),sqrt(3.0/5),0};
    Vector2d v{nXY[1][0]-nXY[0][0],nXY[1][1]-nXY[0][1]};

    // Outer normal
    
    double detJ = sqrt(v(0)*v(0)+v(1)*v(1))/2;
    norm = {v(1)/(2*detJ),v(0)/(2*detJ)};

    // Other attributes

    for(int i=0; i<3; i++){N.col(i) = Vector2d {(1-gRS[i])/2,(1+gRS[i])/2};}
    M = N*wei.asDiagonal()*(N.transpose())*detJ;
    Nf = N*wei*detJ;
    fId = fId_in;
}

Mesh::Mesh(vector<vector<double>> nXY_in,vector<vector<int>> eId_in){

    nXY = nXY_in;
    eId = eId_in;
    nNbr = nXY.size();
    eNbr = eId.size();

    M = SparseMatrix<double>(nNbr,nNbr);
    Sx = SparseMatrix<double>(nNbr,nNbr);
    Sy = SparseMatrix<double>(nNbr,nNbr);

    typedef Triplet<double> T;
    vector<T> tripM;
    vector<T> tripSx;
    vector<T> tripSy;

    #pragma omp parallel
    {
        vector<T> tripMP;
        vector<T> tripSxP;
        vector<T> tripSyP;

        #pragma omp for
        for(int i=0; i<eNbr; i++){
            
            if(eId[i].size()==4){
                Quadrangle elem({nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]],nXY[eId[i][3]]});

                for(int j=0; j<4; j++){
                    for(int k=0; k<4; k++){

                        tripMP.push_back(T(eId[i][k],eId[i][j],elem.M(k,j)));
                        tripSxP.push_back(T(eId[i][k],eId[i][j],elem.Sx(k,j)));
                        tripSyP.push_back(T(eId[i][k],eId[i][j],elem.Sy(k,j)));
                    }
                }
            }

            if(eId[i].size()==3){
                Triangle elem({nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]]});

                for(int j=0; j<3; j++){
                    for(int k=0; k<3; k++){

                        tripMP.push_back(T(eId[i][k],eId[i][j],elem.M(k,j)));
                        tripSxP.push_back(T(eId[i][k],eId[i][j],elem.Sx(k,j)));
                        tripSyP.push_back(T(eId[i][k],eId[i][j],elem.Sy(k,j)));
                    }
                }
            }
        }

        #pragma omp critical
        {
        tripM.insert(tripM.end(),tripMP.begin(),tripMP.end());
        tripSx.insert(tripSx.end(),tripSxP.begin(),tripSxP.end());
        tripSy.insert(tripSy.end(),tripSyP.begin(),tripSyP.end());
        }
    }
    M.setFromTriplets(tripM.begin(),tripM.end());
    Sx.setFromTriplets(tripSx.begin(),tripSx.end());
    Sy.setFromTriplets(tripSy.begin(),tripSy.end());
}

// Precompute Neumann BC

vector<Face> Mesh::precompute(vector<vector<int>> fId){

    vector<Face> face;
    #pragma omp parallel
    {
        vector<Face> faceP;
        #pragma omp for

        for(int i=0; i<fId.size(); i++){
            Face L2({nXY[fId[i][0]],nXY[fId[i][1]]},fId[i]);
            faceP.push_back(L2);
        }
        #pragma omp critical
        face.insert(face.end(),faceP.begin(),faceP.end());
    }
    return face;
}

// Apply constant Neumann BC

VectorXd Mesh::neumannFix(vector<Face> face, double bc){

    VectorXd B(nNbr);B.setZero();
    for(int i=0; i<face.size(); i++){B(face[i].fId) += face[i].Nf*bc;}
    return B;
}

// Apply variable Neumann BC

VectorXd Mesh::neumannVar(vector<Face> face, vector<VectorXd> F){
    
    VectorXd B(nNbr);
    B.setZero();

    for(int i=0; i<face.size(); i++){
        
        Matrix2d flux;
        flux.col(0) = Vector2d {F[0](face[i].fId[0]),F[0](face[i].fId[1])};
        flux.col(1) = Vector2d {F[1](face[i].fId[0]),F[1](face[i].fId[1])};
        Vector2d bc = face[i].M*flux.cwiseAbs()*face[i].norm.cwiseAbs();
        B(face[i].fId[0]) += bc[0];
        B(face[i].fId[1]) += bc[1];
    }
    return B;
}