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

    Face(vector<vector<double>> nXY);
    Vector2d flux(Vector2d fx,Vector2d fy);

    Matrix2d M;
    MatrixXd N;
    Vector2d norm;
    double detJ;

    vector<vector<double>> nXY;
    Vector3d wei{5.0/9,5.0/9,8.0/9};
    vector<double> gRS{-sqrt(3.0/5),sqrt(3.0/5),0};
};

class Mesh{

    public:

    Mesh(vector<vector<double>> nXY,vector<vector<int>> eId,vector<vector<int>> fId);
    VectorXd flux(VectorXd fx,VectorXd fy);

    int nNbr;
    int eNbr;
    int fNbr;

    SparseMatrix<double> M;
    SparseMatrix<double> Sx;
    SparseMatrix<double> Sy;

    vector<vector<int>> eId;
    vector<vector<int>> fId;
    vector<vector<double>> nXY;
    vector<Quadrangle> elemsQ;
    vector<Triangle> elemsT;
    vector<Face> faces;
};

#endif