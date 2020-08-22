# CGFEM-Example

Finite element code developped for **academic purpose** (and thus not optimized) to solve differential equations in two dimensions using triangle or quadrangle elements. Both codes aim at giving an intuitive example of a finite element implementation in C++ and Python.

## Use

The nodes of an element and the domain border must be generated counterclockwise. For the Python code, move to the folder containing the source codes and directly launch one of the test cases in any Python 3 compiler, for instance:
```css
python diffusion.py
```
For the C++ code, make sure that the Eigen library is installed and that your GCC supports OpenMP. Then move to the code folder and compile the project by providing the path to Eigen to the compiler, you may directly launch example.exe and visualize the solution with plot.py.
```css
g++ -O2 -fopenmp -I C:\ProgramData\Eigen example.cpp mesh.cpp -o example.exe
.\example.exe
```

## Author

* Martin Lacroix
