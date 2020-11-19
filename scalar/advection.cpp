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
    vector<vector<int>> fId;
    vector<int> nId;
};

Param meshParam(int elem,int size,double xyMax){

    int k = 0;
    Param param;
    vector<int> idx;
    int node = size+1;
    vector<vector<int>> eId;
    vector<int> nId(2*size+1);
    vector<vector<int>> fId(2*size,vector<int>(2));
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

    // Boundary node and face indices

    for(int i=0; i<node; i++){nId[i] = i;}
    for(int i=0; i<size; i++){nId[node+i] = (i+1)*node;}
    for(int i=0; i<size; i++){
        
        fId[i] = {(i+1)*node-1,(i+2)*node-1};
        fId[size+i] = {node*node-1-i,node*node-2-i};
    }

    param.nXY = nXY;
    param.eId = eId;
    param.nId = nId;
    param.fId = fId;
    return param;
}

// Solves the steady advection-diffusion equation a·∇u(x,y) - kΔu(x,y) = 0

int main(){

    int type = 3;
    int size = 50;
    double xyMax = 1;

    Param param = meshParam(type,size,xyMax);
    auto fun = [](MatrixXd xy){return xy.col(0).array().pow(0);};
    vector<double> bcNeu(param.fId.size(),-0.1);
    vector<double> bcDir(param.nId.size(),0);

    Data data;
    data.k = 1;
    data.a = {3,3};
    data.fun = fun;
    data.bcDir = bcDir;
    data.bcNeu = bcNeu;
    data.nId = param.nId;
    data.fId = param.fId;

    // Mesh and solver

    Mesh mesh(param.nXY,param.eId);
    cout << "\nMesh: done" << endl;
    
    vector<VectorXd> u = advection(mesh,data);
    cout << "Solver: done" << endl;

    // Writes the file

    mkdir("output");
    ofstream solution("output/solution.txt");
    ofstream nXY("output/nXY.txt");

    for (int i=0; i<mesh.nXY.size(); i++){
        nXY << mesh.nXY[i][0] << "," << mesh.nXY[i][1] << "\n";
    }

    for (int i=0; i<u.size(); i++){
        for (int j=0; j<u[0].size()-1; j++){solution << u[i][j] << ",";}
        solution << u[i][u[0].size()-1] << "\n";
    }

    cout << "Writing: done\n" << endl;
    return 0;
}
