# CGFEM-Example

Finite element code developped for **academic purpose** (and thus not optimized) to solve differential equations in two dimensions using triangle or quadrangle elements. The code aim at giving an intuitive example of a finite element implementation in C++ language.

## Use

First, make sure that the Eigen library is installed and that your GCC supports OpenMP. Then move to the code folder and compile the project by providing the path to Eigen to the compiler. The scripts laplace.cpp, transport.cpp, advection.cpp and diffusion.cpp are provided as example.
```css
g++ -O2 -fopenmp -I path-to-Eigen\Eigen example.cpp mesh.cpp -o example.exe
.\example.exe
```
You may run the executable and visualize the solution with plot.py. Note that the nodes of an element and the domain borders must be generated counterclockwise.
```css
.\example.exe
python plot.py
```

## Author

* Martin Lacroix
