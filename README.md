# CGFEM-Example

Finite element code developped for **academic purpose** (and thus not optimized) to solve differential equations in two dimensions using triangle or quadrangle elements. The code aim at giving an intuitive example of a finite element implementation in C++ language.

## Use

The nodes of an element and the domain border must be generated counterclockwise. First, make sure that the Eigen library is installed and that your GCC supports OpenMP. Then move to the code folder and compile the project by providing the path to Eigen to the compiler, you may then directly launch the executable and visualize the solution with plot.py.
```css
g++ -O2 -fopenmp -I path-to-Eigen\Eigen example.cpp mesh.cpp -o example.exe
.\example.exe
```
The different equation laplace.cpp, transport.cpp, advection.cpp and diffusion.cpp are provided as example cases.

## Author

* Martin Lacroix
