#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "mesh.h"
#include <omp.h>

using namespace std;
using namespace Eigen;

Triangle::Triangle(vector<vector<double>> nXY){

    Vector4d wei{-27.0/96,25.0/96,25.0/96,25.0/96};
    vector<vector<double>> gRS{{1.0/3,1.0/3},{0.6,0.2},{0.2,0.6},{0.2,0.2}};

    Matrix2d invJ;
    Vector3d drN{-1,1,0};
    Vector3d dsN{-1,0,1};

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
    Sx = dxN*wei.asDiagonal()*(N.transpose())*detJ;
    Sy = dyN*wei.asDiagonal()*(N.transpose())*detJ;
}

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
    Sx = dxN*(wei.asDiagonal()*detJ).asDiagonal()*(N.transpose());
    Sy = dyN*(wei.asDiagonal()*detJ).asDiagonal()*(N.transpose());
}

Face::Face(vector<vector<double>> nXY){

    N = MatrixXd(2,3);
    Vector3d wei{5.0/9,5.0/9,8.0/9};
    vector<double> gRS{-sqrt(3.0/5),sqrt(3.0/5),0};
    Vector3d v{nXY[1][0]-nXY[0][0],nXY[1][1]-nXY[0][1],0};

    // Outer normal

    norm = {v(1),v(0)};
    double detJ = sqrt(v(0)*v(0)+v(1)*v(1))/2;
    norm /= sqrt(norm(0)*norm(0)+norm(1)*norm(1));

    // Mass and local shape functions

    for(int i=0; i<3; i++){N.col(i) = Vector2d {(1-gRS[i])/2,(1+gRS[i])/2};}
    M = N*wei.asDiagonal()*(N.transpose())*detJ;
}

Vector2d Face::flux(Vector2d fx,Vector2d fy){

    Vector2d Fx = norm[0]*fx;
    Vector2d Fy = norm[1]*fy;

    Fx(0) = abs(Fx(0));
    Fx(1) = abs(Fx(1));
    Fy(0) = abs(Fy(0));
    Fy(1) = abs(Fy(1));

    Vector2d F = M*(Fx+Fy);
    return F;
}

Mesh::Mesh(vector<vector<double>> nXY_in,vector<vector<int>> eId_in,vector<vector<int>> fId_in){

    nXY = nXY_in;
    eId = eId_in;
    fId = fId_in;

    nNbr = nXY.size();
    eNbr = eId.size();
    fNbr = fId.size();

    M = SparseMatrix<double>(nNbr,nNbr);
    Sx = SparseMatrix<double>(nNbr,nNbr);
    Sy = SparseMatrix<double>(nNbr,nNbr);

    typedef Triplet<double> T;
    vector<T> tripM;
    vector<T> tripSx;
    vector<T> tripSy;

    #pragma omp parallel
    {
        vector<Face> faceP;
        #pragma omp for

        for(int i=0; i<fNbr; i++){

            Face L2({nXY[fId[i][0]],nXY[fId[i][1]]});
            faceP.push_back(L2);
        }

        #pragma omp critical
        faces.insert(faces.end(),faceP.begin(),faceP.end());
    }

    #pragma omp parallel
    {
        vector<T> tripMP;
        vector<T> tripSxP;
        vector<T> tripSyP;
        #pragma omp for
        
        for(int i=0; i<eNbr; i++){
            
            if(eId[i].size()==4){

                Quadrangle Q4({nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]],nXY[eId[i][3]]});

                for(int j=0; j<4; j++){

                    // Adds to the global mass matrix

                    tripMP.push_back(T(eId[i][0],eId[i][j],Q4.M(0,j)));
                    tripMP.push_back(T(eId[i][1],eId[i][j],Q4.M(1,j)));
                    tripMP.push_back(T(eId[i][2],eId[i][j],Q4.M(2,j)));
                    tripMP.push_back(T(eId[i][3],eId[i][j],Q4.M(3,j)));

                    // Adds to the global stifness matrices

                    tripSxP.push_back(T(eId[i][0],eId[i][j],Q4.Sx(0,j)));
                    tripSxP.push_back(T(eId[i][1],eId[i][j],Q4.Sx(1,j)));
                    tripSxP.push_back(T(eId[i][2],eId[i][j],Q4.Sx(2,j)));
                    tripSxP.push_back(T(eId[i][3],eId[i][j],Q4.Sx(3,j)));

                    tripSyP.push_back(T(eId[i][0],eId[i][j],Q4.Sy(0,j)));
                    tripSyP.push_back(T(eId[i][1],eId[i][j],Q4.Sy(1,j)));
                    tripSyP.push_back(T(eId[i][2],eId[i][j],Q4.Sy(2,j)));
                    tripSyP.push_back(T(eId[i][3],eId[i][j],Q4.Sy(3,j)));
                }
            }

            if(eId[i].size()==3){

                Triangle T3({nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]]});

                for(int j=0; j<3; j++){

                    // Adds to the global mass matrix

                    tripMP.push_back(T(eId[i][0],eId[i][j],T3.M(0,j)));
                    tripMP.push_back(T(eId[i][1],eId[i][j],T3.M(1,j)));
                    tripMP.push_back(T(eId[i][2],eId[i][j],T3.M(2,j)));

                    // Adds to the global stifness matrices

                    tripSxP.push_back(T(eId[i][0],eId[i][j],T3.Sx(0,j)));
                    tripSxP.push_back(T(eId[i][1],eId[i][j],T3.Sx(1,j)));
                    tripSxP.push_back(T(eId[i][2],eId[i][j],T3.Sx(2,j)));

                    tripSyP.push_back(T(eId[i][0],eId[i][j],T3.Sy(0,j)));
                    tripSyP.push_back(T(eId[i][1],eId[i][j],T3.Sy(1,j)));
                    tripSyP.push_back(T(eId[i][2],eId[i][j],T3.Sy(2,j)));
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

VectorXd Mesh::flux(VectorXd fx,VectorXd fy){
    
    VectorXd F(nNbr);
    F.setZero();

    for(int i=0; i<fNbr; i++){F(fId[i]) += faces[i].flux(fx(fId[i]),fy(fId[i]));}
    return F;
}
