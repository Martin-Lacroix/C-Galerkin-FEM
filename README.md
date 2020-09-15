# CGFEM-Example

Finite element code developped for **Academic Purpose** (and thus not optimized) to solve differential equations in two dimensions using triangle or quadrangle elements. The code aims at giving an intuitive example of a continuous Galerkin finite element implementation in C++ language.

## Use

First, make sure that the Eigen library is installed and that your GCC supports OpenMP. Then move to one of the code folders and compile the project by providing the path to Eigen to the compiler. Some scripts such as transport.cpp or mechanics.cpp are provided as example.
```css
g++ -O2 -fopenmp -I path-to-Eigen\Eigen example.cpp mesh.cpp -o example.exe
```
You may run the executable and visualize the solution with plot.py. Note that the nodes of an element and the domain borders must be assigned counterclockwise as the face normals are oriented to the right.
```css
.\example.exe
python plot.py
```

## Author

* Martin Lacroix
