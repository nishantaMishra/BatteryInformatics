# PDOS plotter

Program for plotting pdos from VASP calculations.

program requires .dat files generated from VASP output files using vaspkit or pymatgen.

usage: 
```bash
python pdos.py # inside the folder containing the .dat files
```
Or 
```bash
python pdos.py /path/to/folder/with/dat/files
```

## Requirements 
- Python 3.x
- numpy
- matplotlib
- pymatgen
The program will try to install all the required packages if not already installed on first execution.
