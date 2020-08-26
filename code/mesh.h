#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#ifndef MESH_H
#define MESH_H

using namespace std;
using namespace Eigen;

class Element{

    public:

    Element(vector<vector<double>> nXY,int idx);

    int type;
    int index;
    MatrixXd M;
    MatrixXd K;
    MatrixXd Sx;
    MatrixXd Sy;
    MatrixXd Ne;
    MatrixXd xy;
};

class Face{

    public:

    Face(vector<vector<double>> nXY,vector<int> fId);

    Matrix2d M;
    Vector2d Ne;
    Vector2d norm;
    vector<int> fId;
};

class Mesh{

    public:

    Mesh(vector<vector<double>> nXY,vector<vector<int>> eId);

    SparseMatrix<double> dirichletBC(SparseMatrix<double> A,vector<int> nId);
    VectorXd neumannBC2(vector<Face> face,vector<VectorXd> F);
    VectorXd neumannBC1(vector<Face> face,vector<double> bc);
    VectorXd vector1D(function<VectorXd(MatrixXd)> fun);
    vector<Face> setFace(vector<vector<int>> fId);
    SparseMatrix<double> matrix2D(string name);

    vector<Element> eList;
    vector<vector<int>> eId;
    vector<vector<double>> nXY;

    int nNbr;
    int eNbr;
};

#endif