#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "solver.h"
#include <vector>
#include <time.h>
#include "mesh.h"

using namespace std;
using namespace Eigen;

struct Struct{ 
    vector<vector<double>> nXY;
    vector<vector<int>> eId;
    vector<vector<int>> fId;
};

Struct meshParam(int elem,int size,float dx){

    int k = 0;
    Struct param;
    vector<int> idx;
    vector<vector<int>> eId;
    vector<vector<int>> fId(4*(size-1),vector<int>(2));
    vector<vector<double>> nXY(size*size,vector<double>(2));
    for(int i=size-1; i<size*size-size-1; i+=size){idx.push_back(i);}

    // Node coordinates

    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            nXY[i*size+j] = {j*dx,i*dx};
        }
    }

    // Element Q4 - nodes indices

    if(elem==4){

        for(int i=0; i<size*size-size-1; i++){
            if(i!=idx[k]){eId.push_back({i,i+1,i+size+1,i+size});}
            else{k++;}
        }
    }

    // Element T3 - nodes indices

    if(elem==3){

        for(int i=0; i<size*size-size-1; i++){
            if(i!=idx[k]){
                eId.push_back({i,i+1,i+size});
                eId.push_back({i+1,i+size+1,i+size});
            }

            else{k++;}
        }
    }

    // Boundary face indices

    for(int i=0; i<size-1; i++){
        
        fId[i] = {i,i+1};
        fId[(size-1)+i] = {(i+1)*size-1,(i+2)*size-1};
        fId[2*(size-1)+i] = {size*size-i-1,size*size-i-2};
        fId[3*(size-1)+i] = {(size-1-i)*size,(size-2-i)*size};
    }

    param.nXY = nXY;
    param.eId = eId;
    param.fId = fId;
    return param;
}

// Gaussian initial solution

VectorXd gaussian(vector<vector<double>> nXY,float dx, vector<double> center){
    
    float x = center[0]*dx;
    float y = center[1]*dx;
    VectorXd u(nXY.size());

    for(int i=0; i<nXY.size(); i++){
        u(i) = exp(-(nXY[i][0]-x)*(nXY[i][0]-x)/2-(nXY[i][1]-y)*(nXY[i][1]-y)/2);
    }

    return u/u.maxCoeff();
}

int main(){

    int elem = 4;
    int size = 60;
    float dx = 0.2;
    vector<double> flux{6,-6};
    vector<double> center{30,30};
    vector<double> tVec{1,0.001};

    // Mesh And Flux

    Struct param = meshParam(elem,size,dx);
    VectorXd u0 = gaussian(param.nXY,dx,center);

    ofstream solution("output/solution.txt");
    ofstream time("output/time.txt");
    ofstream nXY("output/nXY.txt");

    // Solver

    const clock_t start1 = clock();
    Mesh mesh(param.nXY,param.eId,param.fId);
    cout << "Mesh: " << float(clock()-start1)/CLOCKS_PER_SEC << endl;

    const clock_t start2 = clock();
    Solution sol = solver(mesh,u0,tVec,flux);
    cout << "Solver: " << float(clock()-start2)/CLOCKS_PER_SEC << endl;

    // Writes the file

    for (int i=0; i<sol.uStep.size(); i++){
        for (int j=0; j<u0.size(); j++){

            solution << sol.uStep[i][j] << ",";
        }
        solution << "\n";
    }

    for (int i=0; i<sol.tStep.size(); i++){time << sol.tStep[i] << "\n";}
    for (int i=0; i<param.nXY.size(); i++){nXY << param.nXY[i][0] << "," << param.nXY[i][1] << "\n";}

    return 0;
}