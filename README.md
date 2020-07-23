# Finite Element

Finite element code developped for **academic purpose**. Both propgrams, the first written in C++ and the second in Python, solve a transport equation in two dimensions using triangle or quadrangle elements. The codes aim at giving an intuitive example of a finite element code implementation as well as the use of Eigen and Numpy sparse matrix libraries.

## Use

For the Python code, move to the file containing the source codes and directly launch run.py in any Python 3 compiler. The C++ code requires to be compiled with minGW and the path to Eigen, then you may directly launch run.exe from the output file and visualize the solution with plot.py.
```css
python run.py
```
```css
g++ -O2 -I C:\ProgramData\Eigen run.cpp mesh.cpp -o run.exe
.\run.exe
```

## Author

* Martin Lacroix
