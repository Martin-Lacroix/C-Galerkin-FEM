#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#ifndef MESH_H
#define MESH_H

using namespace std;
using namespace Eigen;

class Triangle{

    public:

    Triangle(vector<vector<double>> nXY);

    MatrixXd N;
    Matrix3d M;
    Matrix3d Sx;
    Matrix3d Sy;
};

class Quadrangle{

    public:

    Quadrangle(vector<vector<double>> nXY);

    MatrixXd N;
    Matrix4d M;
    Matrix4d Sx;
    Matrix4d Sy;
};

class Face{

    public:

    Face(vector<vector<double>> nXY,vector<int> fId);

    Matrix2d M;
    Vector2d Nf;
    Vector2d norm;
    vector<int> fId;
};

class Mesh{

    public:

    Mesh(vector<vector<double>> nXY,vector<vector<int>> eId);
    VectorXd neumannVar(vector<Face> face,vector<VectorXd> F);
    VectorXd neumannFix(vector<Face> face,double bc);
    vector<Face> precompute(vector<vector<int>> fId);

    int nNbr;
    int eNbr;

    SparseMatrix<double> M;
    SparseMatrix<double> Sx;
    SparseMatrix<double> Sy;

    vector<vector<int>> eId;
    vector<vector<double>> nXY;
};

#endif
