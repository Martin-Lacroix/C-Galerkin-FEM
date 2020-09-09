#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#ifndef MESH_H
#define MESH_H

using namespace std;
using namespace Eigen;
typedef SparseMatrix<double> SM;

class Elem{

    public:

    Elem(vector<vector<double>> nXY,int idx);
    double vonMises(VectorXd u,Matrix3d D);

    int type;
    int gPts;
    int index;

    vector<VectorXd> detJ;
    vector<MatrixXd> dxN;
    vector<MatrixXd> dyN;
    MatrixXd N;
    VectorXd w;
};

class Face{

    public:

    Face(vector<vector<double>> nXY,vector<int> fId,int idx);

    int index;
    double detJ;

    MatrixXd N;
    VectorXd w;
    Vector2d norm;
    vector<int> fId;
};

class Mesh{

    public:

    Mesh(vector<vector<double>> nXY,vector<vector<int>> eId);
    VectorXd dirichletBC2(VectorXd b,vector<int> nId,vector<double> bc,int dim);
    SM jacobian(function<VectorXd(VectorXd)> fun,VectorXd &u,double dx);
    VectorXd neumannBC(vector<Face> &fList,vector<Vector2d> bc);
    SM matrix2D(Matrix3d De,Matrix3d Dp,VectorXd &u,double ys);
    vector<Face> setFace(vector<vector<int>> fId);
    SM dirichletBC1(SM A,vector<int> nId,int dim);
    VectorXd strain(VectorXd u);
    void update(VectorXd &u);

    vector<Elem> eList;
    vector<int> plastic;
    vector<vector<int>> eId;
    vector<vector<double>> nXY;

    int nNbr;
    int eNbr;
};

#endif