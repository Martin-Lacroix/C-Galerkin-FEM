#include <Eigen/Dense>
#include <iostream>
#include <direct.h>
#include "solver.h"
#include <fstream>
#include <vector>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Param{

    vector<vector<double>> nXY;
    vector<vector<int>> eId;
    vector<int> nId;
};

Param meshParam(int elem,int size,double xyMax){

    int k = 0;
    Param param;
    vector<int> idx;
    int node = size+1;
    vector<vector<int>> eId;
    vector<int> nId(4*size);
    vector<vector<double>> nXY(node*node,vector<double>(2));
    for(int i=size; i<node*size-1; i+=node){idx.push_back(i);}

    // Node coordinates

    for(int i=0; i<node; i++){
        for(int j=0; j<node; j++){
            nXY[i*node+j] = {j*xyMax/size,i*xyMax/size};
        }
    }

    // Element indices

    for(int i=0; i<node*size-1; i++){if(i!=idx[k]){

        if(elem==4){eId.push_back({i,i+1,i+node+1,i+node});}
        if(elem==3){eId.push_back({i,i+1,i+node});eId.push_back({i+1,i+node+1,i+node});}}
        else{k++;}
    }

    // Boundary node indices

    for(int i=0; i<size; i++){
        
        nId[i] = i;
        nId[size+i] = (i+1)*node-1;
        nId[2*size+i] = node*node-i-1;
        nId[3*size+i] = (size-i)*node;
    }

    param.nXY = nXY;
    param.eId = eId;
    param.nId = nId;
    return param;
}

// Solves the laplace equation Δu(x,y) = f(x,y)

int main(){

    int type = 3;
    int size = 50;
    double xyMax = 1;

    Param param = meshParam(type,size,xyMax);
    auto fun = [](MatrixXd xy){return (2*xy.col(0)).array().sin()+(2*xy.col(1)).array().sin();};
    vector<double> bc(param.nId.size(),0);
    
    Data data;
    data.fun = fun;
    data.bcDir = bc;
    data.nId = param.nId;

    // Mesh and solver

    Mesh mesh(param.nXY,param.eId);
    cout << "\nMesh: done" << endl;
    vector<VectorXd> u = laplace(mesh,data);
    cout << "Solver: done" << endl;

    // Writes the file

    mkdir("../output");
    ofstream solution("../output/solution.txt");
    ofstream nXY("../output/nXY.txt");

    for (int i=0; i<param.nXY.size(); i++){
        nXY << param.nXY[i][0] << "," << param.nXY[i][1] << "\n";
    }

    for (int i=0; i<u.size(); i++){
        for (int j=0; j<u[0].size()-1; j++){solution << u[i][j] << ",";}
        solution << u[i][u[0].size()-1] << "\n";
    }

    cout << "Writing: done\n" << endl;
    return 0;
}