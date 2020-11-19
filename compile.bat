mkdir build
g++ -O2 -fopenmp -I C:\ProgramData\Eigen vector\mechanics.cpp vector\mesh.cpp -o build\mechanics.exe
g++ -O2 -fopenmp -I C:\ProgramData\Eigen scalar\advection.cpp scalar\mesh.cpp -o build\advection.exe
g++ -O2 -fopenmp -I C:\ProgramData\Eigen scalar\transport.cpp scalar\mesh.cpp -o build\transport.exe
g++ -O2 -fopenmp -I C:\ProgramData\Eigen scalar\diffusion.cpp scalar\mesh.cpp -o build\diffusion.exe
g++ -O2 -fopenmp -I C:\ProgramData\Eigen scalar\laplace.cpp scalar\mesh.cpp -o build\laplace.exe