mkdir build
g++ -O2 -fopenmp -I C:\Computing\eigen-3.4-rc1 vector\mechanics.cpp vector\mesh.cpp -o build\mechanics.exe
g++ -O2 -fopenmp -I C:\Computing\eigen-3.4-rc1 scalar\advection.cpp scalar\mesh.cpp -o build\advection.exe
g++ -O2 -fopenmp -I C:\Computing\eigen-3.4-rc1 scalar\transport.cpp scalar\mesh.cpp -o build\transport.exe
g++ -O2 -fopenmp -I C:\Computing\eigen-3.4-rc1 scalar\diffusion.cpp scalar\mesh.cpp -o build\diffusion.exe
g++ -O2 -fopenmp -I C:\Computing\eigen-3.4-rc1 scalar\laplace.cpp scalar\mesh.cpp -o build\laplace.exe