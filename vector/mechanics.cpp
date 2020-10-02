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

Param meshParam(int type,int size,double xyMax){

    int k = 0;
    Param param;
    vector<int> idx;
    int node = size+1;
    vector<int> nId(node);
    vector<vector<int>> eId;
    vector<vector<int>> fId(size,vector<int>(2));
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

        if(type==4){eId.push_back({i,i+1,i+node+1,i+node});}
        if(type==3){eId.push_back({i,i+1,i+node});eId.push_back({i+1,i+node+1,i+node});}}
        else{k++;}
    }

    // Boundary face indices

    for(int i=0; i<node; i++){nId[i] = i*node;}
    for(int i=0; i<size; i++){fId[i] = {(i+1)*node-1,(i+2)*node-1};}

    param.nXY = nXY;
    param.eId = eId;
    param.fId = fId;
    param.nId = nId;
    return param;
}

// Solves linear elsato-plasticity ∇·σ(u) = 0

int main(){

    int type = 4;
    int size = 50;
    double xyMax = 1;

    Param param = meshParam(type,size,xyMax);
    vector<Vector2d> bcNeu(param.fId.size());
    for(int i=0; i<param.fId.size(); i++){bcNeu[i] = Vector2d {0,0.1};}

    Data data;
    data.E = 2;
    data.v = 0.3;
    data.step = 100;
    data.bcNeu = bcNeu;
    data.fId = param.fId;
    data.nIdx = param.nId;
    data.nIdy = param.nId;
    data.bcDirX = vector<double> (param.nId.size(),0);
    data.bcDirY = vector<double> (param.nId.size(),0);

    // Mesh and solver

    Mesh mesh(param.nXY,param.eId);
    cout << "\nMesh: done" << endl;
    VectorXd u = newton(mesh,data);
    cout << "Solver: done" << endl;

    // Writes the file

    mkdir("../output");
    ofstream nXY("../output/nXY.txt");
    ofstream solution("../output/solution.txt");
    ofstream strain("../output/strain.txt");

    for (int i=0; i<mesh.nXY.size(); i++){
        nXY << mesh.nXY[i][0] << "," << mesh.nXY[i][1] << "\n";
    }

    for (int j=0; j<u.size()-1; j++){solution << u[j] << ",";}
    solution << u[u.size()-1] << "\n";
    cout << "Writing: done\n" << endl;
    return 0;
}
