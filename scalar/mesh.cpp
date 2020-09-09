#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <vector>
#include "mesh.h"
#include <omp.h>

using namespace std;
using namespace Eigen;
typedef Triplet<double> T;
typedef SparseMatrix<double> SM;

// Triangle or quadrangle 2D element

Elem::Elem(vector<vector<double>> nXY,int idx){

    index = idx;
    type = nXY.size();
    if(type==4){gPts = 9;}
    if(type==3){gPts = 4;}
    MatrixXd rs(gPts,2);
    w = VectorXd(gPts);

    if(type==4){
    
        w << 25.0/81,25.0/81,25.0/81,25.0/81,40.0/81,40.0/81,40.0/81,40.0/81,64.0/81;
        rs.row(0) = Vector2d {-sqrt(3.0/5),-sqrt(3.0/5)};
        rs.row(1) = Vector2d {sqrt(3.0/5),-sqrt(3.0/5)};
        rs.row(2) = Vector2d {-sqrt(3.0/5),sqrt(3.0/5)};
        rs.row(3) = Vector2d {sqrt(3.0/5),sqrt(3.0/5)};
        rs.row(4) = Vector2d {0,-sqrt(3.0/5)};
        rs.row(5) = Vector2d {-sqrt(3.0/5),0};
        rs.row(6) = Vector2d {sqrt(3.0/5),0};
        rs.row(7) = Vector2d {0,sqrt(3.0/5)};
        rs.row(8) = Vector2d {0,0};
    }

    if(type==3){

        w << -27.0/96,25.0/96,25.0/96,25.0/96;
        rs.row(0) = Vector2d {1.0/3,1.0/3};
        rs.row(1) = Vector2d {0.6,0.2};
        rs.row(2) = Vector2d {0.2,0.6};
        rs.row(3) = Vector2d {0.2,0.2};
    }

    N = MatrixXd(type,gPts);
    dxN = MatrixXd(type,gPts);
    dyN = MatrixXd(type,gPts);
    dxN.setZero();
    dyN.setZero();

    MatrixXd drN(type,gPts);
    MatrixXd dsN(type,gPts);
    VectorXd J11(gPts);J11.setZero();
    VectorXd J12(gPts);J12.setZero();
    VectorXd J21(gPts);J21.setZero();
    VectorXd J22(gPts);J22.setZero();
    VectorXd I(gPts);I.setOnes();
    VectorXd r = rs.col(0);
    VectorXd s = rs.col(1);

    if(type==4){

        N.row(0) = (I-r).asDiagonal()*(I-s)/4;
        N.row(1) = (I+r).asDiagonal()*(I-s)/4;
        N.row(2) = (I+r).asDiagonal()*(I+s)/4;
        N.row(3) = (I-r).asDiagonal()*(I+s)/4;

        for (int i=0; i<gPts; i++){
            drN.col(i) = Vector4d {(s(i)-1)/4,(1-s(i))/4,(s(i)+1)/4,-(s(i)+1)/4};
            dsN.col(i) = Vector4d {(r(i)-1)/4,-(r(i)+1)/4,(r(i)+1)/4,(1-r(i))/4};
        }
    }

    if(type==3){

        N.row(0) = I-r-s;
        N.row(1) = r;
        N.row(2) = s;

        for (int i=0; i<gPts; i++){
            drN.col(i) = Vector3d {-1,1,0};
            dsN.col(i) = Vector3d {-1,0,1};
        }
    }

    MatrixXd coord(type,2);
    for(int i=0; i<type; i++){coord.row(i) = Vector2d::Map(nXY[i].data());}
    xy = N.transpose()*coord;

    for(int i=0; i<type; i++){

        J11 += drN.row(i)*nXY[i][0];
        J12 += drN.row(i)*nXY[i][1];
        J21 += dsN.row(i)*nXY[i][0];
        J22 += dsN.row(i)*nXY[i][1];
    }

    detJ = J11.asDiagonal()*J22-J12.asDiagonal()*J21;
    VectorXd invJ11 = (J22.array().colwise()/detJ.array()).matrix();
    VectorXd invJ22 = (J11.array().colwise()/detJ.array()).matrix();
    VectorXd invJ12 = (J12.array().colwise()/detJ.array()).matrix()*(-1);
    VectorXd invJ21 = (J21.array().colwise()/detJ.array()).matrix()*(-1);

    for(int i=0; i<gPts; i++){
        for(int j=0; j<type; j++){

            dxN(j,i) += drN(j,i)*invJ11(i)+dsN(j,i)*invJ12(i);
            dyN(j,i) += drN(j,i)*invJ21(i)+dsN(j,i)*invJ22(i);
        }
    }
}

// Face 1D element

Face::Face(vector<vector<double>> nXY,vector<int> fId_in,int idx){

    index = idx;
    int type = 2;
    int gPts = 3;

    VectorXd I(gPts);
    N = MatrixXd(type,gPts);
    Matrix2d R{{0,1},{-1,0}};
    w = Vector3d {5.0/9,5.0/9,8.0/9};
    Vector3d rs{-sqrt(3.0/5),sqrt(3.0/5),0};
    Vector2d v{nXY[1][0]-nXY[0][0],nXY[1][1]-nXY[0][1]};

    v = R*v;
    I.setOnes();
    fId = fId_in;
    N.row(0) = (I-rs)/2;
    N.row(1) = (I+rs)/2;
    detJ = sqrt(v(0)*v(0)+v(1)*v(1))/2;
    norm = {v(0)/(2*detJ),v(1)/(2*detJ)};
}

// Builds the Elems of the mesh

Mesh::Mesh(vector<vector<double>> nXY_in,vector<vector<int>> eId_in){

    nXY = nXY_in;
    eId = eId_in;
    nNbr = nXY.size();
    eNbr = eId.size();

    #pragma omp parallel
    {
        vector<Elem> eListP;
        #pragma omp for

        for(int i=0; i<eNbr; i++){
            
            int type = eId[i].size();
            vector<vector<double>> nXYp;
            if(type==3){nXYp = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]]};}
            if(type==4){nXYp = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]],nXY[eId[i][3]]};}
            Elem elem(nXYp,i);
            eListP.push_back(elem);
        }
        #pragma omp critical
        eList.insert(eList.end(),eListP.begin(),eListP.end());
    }
    auto compare = [](const Elem& x,const Elem& y){return x.index<y.index;};
    sort(eList.begin(),eList.end(),compare);
}

// Computes the global matrices

SM Mesh::matrix2D(string name){

    SM Mat(nNbr,nNbr);
    vector<T> triplet;

    #pragma omp parallel
    {
        vector<T> trip;
        #pragma omp for

        for(int i=0; i<eNbr; i++){

            Elem elem = eList[i];
            MatrixXd A(elem.type,elem.type);
            VectorXd wJ = elem.w.asDiagonal()*elem.detJ;

            if(name=="M"){A = elem.N*wJ.asDiagonal()*elem.N.transpose();}
            if(name=="Sx"){A = elem.N*wJ.asDiagonal()*elem.dxN.transpose();}
            if(name=="Sy"){A = elem.N*wJ.asDiagonal()*elem.dyN.transpose();}
            if(name=="K"){A = elem.dxN*wJ.asDiagonal()*elem.dxN.transpose();}
            if(name=="K"){A += elem.dyN*wJ.asDiagonal()*elem.dyN.transpose();}

            for(int j=0; j<elem.type; j++){
                for(int k=0; k<elem.type; k++){
                    trip.push_back(T(eId[i][k],eId[i][j],A(k,j)));
                }
            }  
        }
        #pragma omp critical
        triplet.insert(triplet.end(),trip.begin(),trip.end());
    }
    Mat.setFromTriplets(triplet.begin(),triplet.end());
    return Mat;
}

// Integrates N*fun(x,y) over Elems

VectorXd Mesh::vector1D(function<VectorXd(MatrixXd)> fun){

    VectorXd F(nNbr);
    F.setZero();

    for(int i; i<eNbr; i++){

        Elem elem = eList[i];
        VectorXd wJ = elem.w.asDiagonal()*elem.detJ;
        VectorXd elF = elem.N*wJ.asDiagonal()*fun(elem.xy);
        for(int j=0; j<elem.type; j++){F(eId[i][j]) += elF[j];}
    }
    return F;
}

// Precompute faces for Neumann BC

vector<Face> Mesh::setFace(vector<vector<int>> fId){

    vector<Face> fList;
    #pragma omp parallel
    {
        vector<Face> fListP;
        #pragma omp for

        for(int i=0; i<fId.size(); i++){
            Face elem({nXY[fId[i][0]],nXY[fId[i][1]]},fId[i],i);
            fListP.push_back(elem);
        }
        #pragma omp critical
        fList.insert(fList.end(),fListP.begin(),fListP.end());
    }
    auto compare = [](const Face& x,const Face& y){return x.index<y.index;};
    sort(fList.begin(),fList.end(),compare);
    return fList;
}

// Apply constant Neumann BC

VectorXd Mesh::neumannBC1(vector<Face> &fList,vector<double> bc){

    VectorXd B(nNbr);
    B.setZero();

    for(int i=0; i<fList.size(); i++){
        
        Face elem = fList[i];
        Vector2d N = elem.N*elem.w*elem.detJ;
        B(elem.fId) += N*bc[i];
    }
    return B;
}

// Apply variable Neumann BC

VectorXd Mesh::neumannBC2(vector<Face> &fList,vector<VectorXd> F){
    
    VectorXd B(nNbr);
    B.setZero();

    for(int i=0; i<fList.size(); i++){

        Matrix2d flux;
        Face elem = fList[i];
        MatrixXd N = elem.N;

        Matrix2d M = N*elem.w.asDiagonal()*N.transpose()*elem.detJ;
        flux.col(0) = Vector2d {F[0](elem.fId[0]),F[0](elem.fId[1])};
        flux.col(1) = Vector2d {F[1](elem.fId[0]),F[1](elem.fId[1])};
        Vector2d bc = M*flux*elem.norm;

        B(elem.fId[0]) += bc[0];
        B(elem.fId[1]) += bc[1];
    }
    return B;
}

// Prepare the matrix for Dirichlet BC

SM Mesh::dirichletBC(SM A,vector<int> nId){

    VectorXd I(nNbr);
    I.setOnes();

    for(int i=0; i<nId.size(); i++){I(nId[i]) = 0;}
    SM Ad = I.asDiagonal()*A;
    Ad = Ad.pruned(1,1e-16);

    for(int i=0; i<nId.size(); i++){Ad.insert(nId[i],nId[i]) = 1;}
    return Ad.pruned(1,1e-16);
}