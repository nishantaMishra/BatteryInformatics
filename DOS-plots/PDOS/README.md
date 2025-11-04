# PDOS plotter

Program for plotting pdos from VASP calculations.

program requires .dat files generated from VASP output files using vaspkit or pymatgen.

usage: 
Navigate to the folder containing the calculation files and call the function as:

```bash
python pdos.py # inside the folder containing the .dat files
```
Or 
```bash
python pdos.py /path/to/folder/with/dat/files
```

The program will look for the following files in the specified folder:
- vaspout.h5
- DOSCAR, PROCAR, INCAR (if vaspout.h5 is not found)

# TODO:
- Independence from vaspkit to generate .dat files.

## Requirements 
- Python 3.x
- numpy
- matplotlib
- pymatgen
The program will try to install all the required packages if not already installed on first execution.
- vaspkit : program expects vaspkit to be installed and available in PATH to generate .dat files from VASP output files if not already present.
