#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <vector>
#include "mesh.h"
#include <omp.h>

using namespace std;
using namespace Eigen;

// Triangle or quadrangle 2D element

Element::Element(vector<vector<double>> nXY,int idx){

    int node;
    index = idx;
    type = nXY.size();
    if(type==4){node = 9;}
    if(type==3){node = 4;}
    MatrixXd rs(node,2);
    VectorXd w(node);

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

    MatrixXd N(type,node);
    MatrixXd drN(type,node);
    MatrixXd dsN(type,node);
    MatrixXd dxN(type,node);dxN.setZero();
    MatrixXd dyN(type,node);dyN.setZero();
    VectorXd J11(node);J11.setZero();
    VectorXd J12(node);J12.setZero();
    VectorXd J21(node);J21.setZero();
    VectorXd J22(node);J22.setZero();
    VectorXd I(node);I.setOnes();
    VectorXd r = rs.col(0);
    VectorXd s = rs.col(1);

    if(type==4){

        N.row(0) = (I-r).asDiagonal()*(I-s)/4;
        N.row(1) = (I+r).asDiagonal()*(I-s)/4;
        N.row(2) = (I+r).asDiagonal()*(I+s)/4;
        N.row(3) = (I-r).asDiagonal()*(I+s)/4;

        for (int i=0; i<node; i++){
            drN.col(i) = Vector4d {(s(i)-1)/4,(1-s(i))/4,(s(i)+1)/4,-(s(i)+1)/4};
            dsN.col(i) = Vector4d {(r(i)-1)/4,-(r(i)+1)/4,(r(i)+1)/4,(1-r(i))/4};
        }
    }

    if(type==3){

        N.row(0) = I-r-s;
        N.row(1) = r;
        N.row(2) = s;

        for (int i=0; i<node; i++){
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

    VectorXd detJ = J11.asDiagonal()*J22-J12.asDiagonal()*J21;
    VectorXd invJ11 = (J22.array().colwise()/detJ.array()).matrix();
    VectorXd invJ22 = (J11.array().colwise()/detJ.array()).matrix();
    VectorXd invJ12 = (J12.array().colwise()/detJ.array()).matrix()*(-1);
    VectorXd invJ21 = (J21.array().colwise()/detJ.array()).matrix()*(-1);

    for(int i=0; i<node; i++){
        for(int j=0; j<type; j++){

            dxN(j,i) += drN(j,i)*invJ11(i)+dsN(j,i)*invJ12(i);
            dyN(j,i) += drN(j,i)*invJ21(i)+dsN(j,i)*invJ22(i);
        }
    }

    Ne = N*(w.asDiagonal()*detJ).asDiagonal();
    M = N*(w.asDiagonal()*detJ).asDiagonal()*(N.transpose());
    Sx = N*(w.asDiagonal()*detJ).asDiagonal()*(dxN.transpose());
    Sy = N*(w.asDiagonal()*detJ).asDiagonal()*(dyN.transpose());
    K = dxN*(w.asDiagonal()*detJ).asDiagonal()*(dxN.transpose());
    K += dyN*(w.asDiagonal()*detJ).asDiagonal()*(dyN.transpose());
}

// Face 1D element

Face::Face(vector<vector<double>> nXY,vector<int> fId_in){

    int type = 2;
    int node = 3;

    VectorXd I(node);
    MatrixXd N(type,node);
    Vector3d w{5.0/9,5.0/9,8.0/9};
    Vector3d rs{-sqrt(3.0/5),sqrt(3.0/5),0};
    Vector2d v{nXY[1][0]-nXY[0][0],nXY[1][1]-nXY[0][1]};

    I.setOnes();
    N.row(0) = (I-rs)/2;
    N.row(1) = (I+rs)/2;
    double detJ = sqrt(v(0)*v(0)+v(1)*v(1))/2;
    M = N*w.asDiagonal()*(N.transpose())*detJ;
    norm = {v(1)/(2*detJ),v(0)/(2*detJ)};
    Ne = N*w*detJ;
    fId = fId_in;
}

// Builds the elements of the mesh

Mesh::Mesh(vector<vector<double>> nXY_in,vector<vector<int>> eId_in){

    nXY = nXY_in;
    eId = eId_in;
    nNbr = nXY.size();
    eNbr = eId.size();

    #pragma omp parallel
    {
        vector<Element> eListP;
        #pragma omp for

        for(int i=0; i<eNbr; i++){
            
            int type = eId[i].size();
            vector<vector<double>> nXY_el;
            if(type==3){nXY_el = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]]};}
            if(type==4){nXY_el = {nXY[eId[i][0]],nXY[eId[i][1]],nXY[eId[i][2]],nXY[eId[i][3]]};}
            Element elem(nXY_el,i);
            eListP.push_back(elem);
        }
        #pragma omp critical
        eList.insert(eList.end(),eListP.begin(),eListP.end());
    }
    auto compare = [](const Element& x,const Element& y){return x.index<y.index;};
    sort(eList.begin(),eList.end(),compare);
}

// Computes the global matrices

SparseMatrix<double> Mesh::matrix2D(string name){

    SparseMatrix<double> A(nNbr,nNbr);
    typedef Triplet<double> T;
    vector<T> triplet;

    #pragma omp parallel
    {
        vector<T> trip;
        #pragma omp for

        for(int i=0; i<eNbr; i++){
            for(int j=0; j<eList[i].type; j++){
                for(int k=0; k<eList[i].type; k++){

                    if(name=="M"){trip.push_back(T(eId[i][k],eId[i][j],eList[i].M(k,j)));}
                    if(name=="K"){trip.push_back(T(eId[i][k],eId[i][j],eList[i].K(k,j)));}
                    if(name=="Sx"){trip.push_back(T(eId[i][k],eId[i][j],eList[i].Sx(k,j)));}
                    if(name=="Sy"){trip.push_back(T(eId[i][k],eId[i][j],eList[i].Sy(k,j)));}
                }
            }  
        }
        #pragma omp critical
        triplet.insert(triplet.end(),trip.begin(),trip.end());
    }
    A.setFromTriplets(triplet.begin(),triplet.end());
    return A;
}

// Integrates N*fun(x,y) over elements

VectorXd Mesh::vector1D(function<VectorXd(MatrixXd)> fun){

    VectorXd F(nNbr);
    F.setZero();

    for(int i; i<eNbr; i++){
        VectorXd elF = eList[i].Ne*fun(eList[i].xy);
        for(int j=0; j<eList[i].type; j++){F(eId[i][j]) += elF[j];}
    }
    return F;
}

// Precompute faces for Neumann BC

vector<Face> Mesh::setFace(vector<vector<int>> fId){

    vector<Face> face;
    #pragma omp parallel
    {
        vector<Face> faceP;
        #pragma omp for

        for(int i=0; i<fId.size(); i++){
            Face elem({nXY[fId[i][0]],nXY[fId[i][1]]},fId[i]);
            faceP.push_back(elem);
        }
        #pragma omp critical
        face.insert(face.end(),faceP.begin(),faceP.end());
    }
    return face;
}

// Apply constant Neumann BC

VectorXd Mesh::neumannBC1(vector<Face> face,vector<double> bc){

    VectorXd B(nNbr);B.setZero();
    for(int i=0; i<face.size(); i++){B(face[i].fId) += face[i].Ne*bc[i];}
    return B;
}

// Apply variable Neumann BC

VectorXd Mesh::neumannBC2(vector<Face> face,vector<VectorXd> F){
    
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

// Prepare the matrix for Dirichlet BC

SparseMatrix<double> Mesh::dirichletBC(SparseMatrix<double> A,vector<int> nId){

    VectorXd I(nNbr);
    I.setOnes();

    for(int i=0; i<nId.size(); i++){I(nId[i]) = 0;}
    SparseMatrix<double> Ad = I.asDiagonal()*A;
    Ad = Ad.pruned();

    for(int i=0; i<nId.size(); i++){Ad.insert(nId[i],nId[i]) = 1;}
    return Ad.pruned();
}