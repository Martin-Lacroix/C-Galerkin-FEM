#include <Eigen/Dense>
#include <iostream>
#include <direct.h>
#include "solver.h"
#include <fstream>
#include <vector>
#include <time.h>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Param{

    vector<vector<double>> nXY;
    vector<vector<int>> eId;
    vector<vector<int>> fId;
};

Param meshParam(int elem,int size,double xyMax){

    int k = 0;
    Param param;
    vector<int> idx;
    int node = size+1;
    vector<vector<int>> eId;
    vector<vector<int>> fId(4*size,vector<int>(2));
    vector<vector<double>> nXY(node*node,vector<double>(2));
    for(int i=size; i<node*size-1; i+=node){idx.push_back(i);}

    // Node coordinates

    for(int i=0; i<node; i++){
        for(int j=0; j<node; j++){
            nXY[i*node+j] = {j*xyMax/size,i*xyMax/size};
        }
    }

    // Element indices

    for(int i=0; i<node*node-node-1; i++){if(i!=idx[k]){
        if(elem==4){eId.push_back({i,i+1,i+node+1,i+node});}
        if(elem==3){eId.push_back({i,i+1,i+node});eId.push_back({i+1,i+node+1,i+node});}}
        else{k++;}
    }

    // Boundary face indices

    for(int i=0; i<node-1; i++){
        
        fId[i] = {i,i+1};
        fId[(node-1)+i] = {(i+1)*node-1,(i+2)*node-1};
        fId[2*(node-1)+i] = {node*node-i-1,node*node-i-2};
        fId[3*(node-1)+i] = {(node-1-i)*node,(node-2-i)*node};
    }

    param.nXY = nXY;
    param.eId = eId;
    param.fId = fId;
    return param;
}

// Gaussian initial solution

VectorXd gaussian(vector<vector<double>> nXY,double xyMax){
    
    double mu = xyMax/2;
    int nNbr = nXY.size();
    VectorXd u(nXY.size());

    for(int i=0; i<nNbr; i++){u(i) = exp(-pow((nXY[i][0]-mu),2)/2-pow((nXY[i][1]-mu),2)/2);}
    u /= u.maxCoeff();
    return u;
}

int main(){

    int elem = 3;
    int size = 50;
    double xyMax = 10;
    vector<double> flux{6,-6};
    Param param = meshParam(elem,size,xyMax);
    VectorXd u0 = gaussian(param.nXY,xyMax);

    Data data;
    data.u0 = u0;
    data.tMax = 1;
    data.dt = 0.001;
    data.flux = {6,-6};
    data.fId = param.fId;

    // Mesh and solver

    const clock_t start1 = clock();
    Mesh mesh(param.nXY,param.eId);
    cout << "Mesh: " << float(clock()-start1)/CLOCKS_PER_SEC << " [sec]" << endl;

    const clock_t start2 = clock();
    vector<VectorXd> u = transport(mesh,data);
    cout << "Solver: " << float(clock()-start2)/CLOCKS_PER_SEC << " [sec]" << endl;

    // Writes the file

    mkdir("../output");
    ofstream solution("../output/solution.txt");
    ofstream nXY("../output/nXY.txt");

    for (int i=0; i<param.nXY.size(); i++){
        nXY << param.nXY[i][0] << "," << param.nXY[i][1] << "\n";
    }

    for (int i=0; i<u.size(); i++){
        for (int j=0; j<u0.size()-1; j++){solution << u[i][j] << ",";}
        solution << u[i][u0.size()-1] << "\n";
    }

    return 0;
}