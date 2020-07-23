# Finite Element

Finite element code developped for **academic purpose** to solve a transport equation in two dimensions using triangle or quadrangle elements. Both codes aim at giving an intuitive example of a finite element code implementation in python and C++. The nodes of an element and the domain border must be generated counterclockwise.

## Use

For the Python code, move to the file containing the source codes and directly launch example.py in any Python 3 compiler.
```css
python example.py
```
The C++ code requires to be compiled with minGW and the path to Eigen, then you may directly launch example.exe from the output file and visualize the solution with plot.py.
```css
g++ -O2 -I C:\ProgramData\Eigen run.cpp mesh.cpp -o run.exe
.\example.exe
```

## Author

* Martin Lacroix
