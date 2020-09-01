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
    int gPts;
    int index;

    VectorXd detJ;
    MatrixXd dxN;
    MatrixXd dyN;
    MatrixXd xy;
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

    SparseMatrix<double> dirichletBC1(SparseMatrix<double> A,vector<int> nId,int dim);
    VectorXd dirichletBC2(VectorXd b,vector<int> nId,vector<double> bc,int dim);
    VectorXd neumannBC(vector<Face> face,vector<Vector2d> bc);
    SparseMatrix<double> matrix2D(double E,double v);
    vector<Face> setFace(vector<vector<int>> fId);

    vector<Element> eList;
    vector<vector<int>> eId;
    vector<vector<double>> nXY;

    int nNbr;
    int eNbr;
};

#endif