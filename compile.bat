mkdir build

:: External library paths

set path-eigen=C:\Computing\msys64\mingw64\include\eigen3

:: Main code paths

set mechanics-cpp=vector\mechanics.cpp vector\mesh.cpp
set advection-cpp=scalar\advection.cpp scalar\mesh.cpp
set transport-cpp=scalar\transport.cpp scalar\mesh.cpp
set diffusion-cpp=scalar\diffusion.cpp scalar\mesh.cpp
set laplace-cpp=scalar\laplace.cpp scalar\mesh.cpp

:: GCC Compiler

g++ -O3 -fopenmp -I %path-eigen% %mechanics-cpp% -o build\mechanics.exe
g++ -O3 -fopenmp -I %path-eigen% %advection-cpp% -o build\advection.exe
g++ -O3 -fopenmp -I %path-eigen% %transport-cpp% -o build\transport.exe
g++ -O3 -fopenmp -I %path-eigen% %diffusion-cpp% -o build\diffusion.exe
g++ -O3 -fopenmp -I %path-eigen% %laplace-cpp% -o build\laplace.exe
