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

#include <iostream>

// Triangle or quadrangle 2D element

Elem::Elem(vector<vector<double>> nXY,int idx){

    index = idx;
    type = nXY.size();
    if(type==4){gPts = 9;}
    if(type==3){gPts = 4;}
    MatrixXd rs(gPts,2);
    MatrixXd xy(type,2);
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

    for(int i=0; i<type; i++){
        xy.row(i) = Vector2d::Map(nXY[i].data());
    }

    N = MatrixXd(type,gPts);
    dxN = {MatrixXd(type,gPts).setZero(),MatrixXd(type,type).setZero()};
    dyN = {MatrixXd(type,gPts).setZero(),MatrixXd(type,type).setZero()};

    VectorXd I(gPts);I.setOnes();
    vector<MatrixXd> drN = {MatrixXd(type,gPts),MatrixXd(type,type)};
    vector<MatrixXd> dsN = {MatrixXd(type,gPts),MatrixXd(type,type)};
    vector<VectorXd> J11 = {VectorXd(gPts).setZero(),VectorXd(type).setZero()};
    vector<VectorXd> J12 = {VectorXd(gPts).setZero(),VectorXd(type).setZero()};
    vector<VectorXd> J21 = {VectorXd(gPts).setZero(),VectorXd(type).setZero()};
    vector<VectorXd> J22 = {VectorXd(gPts).setZero(),VectorXd(type).setZero()};

    VectorXd x = xy.col(0);
    VectorXd y = xy.col(1);
    VectorXd r = rs.col(0);
    VectorXd s = rs.col(1);

    if(type==4){

        N.row(0) = (I-r).asDiagonal()*(I-s)/4;
        N.row(1) = (I+r).asDiagonal()*(I-s)/4;
        N.row(2) = (I+r).asDiagonal()*(I+s)/4;
        N.row(3) = (I-r).asDiagonal()*(I+s)/4;

        for (int i=0; i<gPts; i++){
            drN[0].col(i) = Vector4d {(s(i)-1)/4,(1-s(i))/4,(s(i)+1)/4,-(s(i)+1)/4};
            dsN[0].col(i) = Vector4d {(r(i)-1)/4,-(r(i)+1)/4,(r(i)+1)/4,(1-r(i))/4};
        }
        for (int i=0; i<type; i++){
            drN[1].col(i) = Vector4d {(y(i)-1)/4,(1-y(i))/4,(y(i)+1)/4,-(y(i)+1)/4};
            dsN[1].col(i) = Vector4d {(x(i)-1)/4,-(x(i)+1)/4,(x(i)+1)/4,(1-x(i))/4};
        }
    }

    if(type==3){

        N.row(0) = I-r-s;
        N.row(1) = r;
        N.row(2) = s;

        for (int i=0; i<gPts; i++){
            drN[0].col(i) = Vector3d {-1,1,0};
            dsN[0].col(i) = Vector3d {-1,0,1};
        }
        for (int i=0; i<type; i++){
            drN[1].col(i) = Vector3d {-1,1,0};
            dsN[1].col(i) = Vector3d {-1,0,1};
        }
    }

    for(int k=0; k<2; k++){
        for(int i=0; i<type; i++){

        J11[k] += drN[k].row(i)*x(i);
        J12[k] += drN[k].row(i)*y(i);
        J21[k] += dsN[k].row(i)*x(i);
        J22[k] += dsN[k].row(i)*y(i);
        }
    }

    for(int k=0; k<2; k++){

        detJ.push_back(J11[k].asDiagonal()*J22[k]-J12[k].asDiagonal()*J21[k]);
        VectorXd invJ11 = (J22[k].array().colwise()/detJ[k].array()).matrix();
        VectorXd invJ22 = (J11[k].array().colwise()/detJ[k].array()).matrix();
        VectorXd invJ12 = (J12[k].array().colwise()/detJ[k].array()).matrix()*(-1);
        VectorXd invJ21 = (J21[k].array().colwise()/detJ[k].array()).matrix()*(-1);

        for(int i=0; i<detJ[k].size(); i++){
            for(int j=0; j<type; j++){

                dxN[k](j,i) += drN[k](j,i)*invJ11(i)+dsN[k](j,i)*invJ12(i);
                dyN[k](j,i) += drN[k](j,i)*invJ21(i)+dsN[k](j,i)*invJ22(i);
            }
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

// Builds the elements of the mesh

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
            vector<vector<double>> nXY_el;
            if(type==3){nXY_el = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]]};}
            if(type==4){nXY_el = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]],nXY[eId[i][3]]};}
            Elem elem(nXY_el,i);
            eListP.push_back(elem);
        }
        #pragma omp critical
        eList.insert(eList.end(),eListP.begin(),eListP.end());
    }
    auto compare = [](const Elem& x,const Elem& y){return x.index<y.index;};
    sort(eList.begin(),eList.end(),compare);
}

// Computes the global matrix K

SM Mesh::matrix2D(double E,double v){

    SM Mat(2*nNbr,2*nNbr);
    vector<T> triplet;
    Matrix3d D;

    D.row(0) = Vector3d {1,v,0};
    D.row(1) = Vector3d {v,1,0};
    D.row(2) = Vector3d {0,0,(1-v)/2};
    D *= E/(1-v*v);

    #pragma omp parallel
    {
        vector<T> trip;
        #pragma omp for

        for(int i=0; i<eNbr; i++){

            Elem elem = eList[i];
            int type = elem.type;
            MatrixXd A(2*type,2*type);
            MatrixXd B(2*type,3);

            A.setZero();
            B.setZero();

            for(int j=0; j<elem.gPts; j++){

                B(seq(0,type-1),0) = elem.dxN[0].col(j);
                B(seq(0,type-1),2) = elem.dyN[0].col(j);
                B(seq(type,2*type-1),1) = elem.dyN[0].col(j);
                B(seq(type,2*type-1),2) = elem.dxN[0].col(j);
                A += elem.w(j)*B*D*B.transpose()*elem.detJ[0](j);
            }

            for(int j=0; j<type; j++){
                for(int k=0; k<type; k++){

                    trip.push_back(T(eId[i][k],eId[i][j],A(k,j)));
                    trip.push_back(T(eId[i][k]+nNbr,eId[i][j],A(type+k,j)));
                    trip.push_back(T(eId[i][k],eId[i][j]+nNbr,A(k,type+j)));
                    trip.push_back(T(eId[i][k]+nNbr,eId[i][j]+nNbr,A(type+k,type+j)));
                }
            }  
        }
        #pragma omp critical
        triplet.insert(triplet.end(),trip.begin(),trip.end());
    }
    Mat.setFromTriplets(triplet.begin(),triplet.end());
    return Mat;
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

VectorXd Mesh::neumannBC(vector<Face> &fList,vector<Vector2d> bc){

    VectorXd B(2*nNbr);
    B.setZero();

    for(int i=0; i<fList.size(); i++){
        
        MatrixXd S(4,2);
        Face elem = fList[i];
        Vector2d N = elem.N*elem.w*elem.detJ;

        S.setZero();
        S({0,1},0) = N;
        S({2,3},1) = N;

        Vector4d Be = S*bc[i];

        B(elem.fId[0]) += Be(0);
        B(elem.fId[1]) += Be(1);
        B(elem.fId[0]+nNbr) += Be(2);
        B(elem.fId[1]+nNbr) += Be(3);
    }
    return B;
}

// Prepare the matrix for Dirichlet BC

SM Mesh::dirichletBC1(SM A,vector<int> nId,int dim){

    int d = dim*nNbr;
    double tol = 1e-16;
    VectorXd I(2*nNbr);
    I.setOnes();

    for(int i=0; i<nId.size(); i++){I(nId[i]+d) = 0;}
    SM Ad = I.asDiagonal()*A;
    Ad = Ad.pruned(1,tol);

    for(int i=0; i<nId.size(); i++){Ad.insert(nId[i]+d,nId[i]+d) = 1;}
    return Ad.pruned(1,tol);
}

// Prepares the vector for Dirichlet BC

VectorXd Mesh::dirichletBC2(VectorXd b,vector<int> nId,vector<double> bc,int dim){

    int d = dim*nNbr;
    for(int i; i<nId.size(); i++){b(nId[i]+d) = bc[i];}
    return b;
}

// Evaluates the Jacobian matrix of a function

SM Mesh::jacobian(function<VectorXd(VectorXd)> fun,VectorXd u,double du){

    SM J(2*nNbr,2*nNbr);
    VectorXd F = fun(u);
    double tol = 1e-16;
    vector<T> triplet;

    #pragma omp parallel
    {
        VectorXd col;
        vector<T> trip;
        VectorXd uTemp = u;
        #pragma omp for

        for(int j=0; j<2*nNbr; j++){

            uTemp[j] += du;
            col = (fun(uTemp)-F)/du;
            uTemp[j] = u[j];

            for(int i=0; i<2*nNbr; i++){
                if(abs(col(i))>tol){trip.push_back(T(i,j,col(i)));}
            }
        }
        #pragma omp critical
        triplet.insert(triplet.end(),trip.begin(),trip.end());
    }
    J.setFromTriplets(triplet.begin(),triplet.end());
    return J;
}

// Computes the strain field

VectorXd Mesh::strain(VectorXd u){

    VectorXd du(3*nNbr);
    du.setZero();

    for(int i=0; i<eNbr; i++){

        Elem elem = eList[i];
        int type = elem.type;
        MatrixXd B(2*type,3);
        VectorXd uEl(2*type);
        B.setZero();

        for(int j=0; j<type; j++){

            uEl(j) = u(eId[i][j]);
            uEl(j+type) = u(eId[i][j]+nNbr);
        }

        for(int j=0; j<type; j++){

            B(seq(0,type-1),0) = elem.dxN[1].col(j);
            B(seq(0,type-1),2) = elem.dyN[1].col(j);
            B(seq(type,2*type-1),1) = elem.dyN[1].col(j);
            B(seq(type,2*type-1),2) = elem.dxN[1].col(j);

            Vector3d duEl = B.transpose()*uEl;
            du(eId[i][j]+2*nNbr) = duEl(2);
            du(eId[i][j]+nNbr) = duEl(1);
            du(eId[i][j]) = duEl(0);
        }
    }
    return du;
}