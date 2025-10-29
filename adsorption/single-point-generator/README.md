# single point generator

This program generates single point calculation input files from a series of VASP calculations with different surface/adsorbate configurations. It extracts the necessary parameters from the original calculations and creates new input files for single point energy calculations.

## A note on NGXF...

NGX,NGY and NGZ depend on `PREC` and `ENCUT` and lattie constant(from POSCAR) As far as these two tags are 
not changing between the tree calculation the NGswill stay constant amongst the three calculation

Also because we do not change size of simulation cell while rmoving the surface/adsorbate, these values stay same. 

Thus this program is not writing them.


However the NGX, NGY and NGZ values can be calculated from the folollowing formula if needed:

```python
#!/usr/bin/env python3
"""
Estimate FFT grid sizes (NGX, NGY, NGZ, NGXF, NGYF, NGZF)
that VASP will choose from POSCAR + INCAR.
https://vasp.at/wiki/NGX 
"""

import numpy as np
from pymatgen.io.vasp import Poscar

# ---- user inputs ----
POSCAR_FILE = "POSCAR"
INCAR_FILE = "INCAR"
# ----------------------

# --- physical constants ---
# ℏ²/(2m_e) in eV·Å²
HBAR2_OVER_2ME = 3.80998212

# --- read ENCUT and PREC ---
encut = 520.0
prec = "Accurate"
with open(INCAR_FILE) as f:
    for line in f:
        line = line.strip()
        if line.upper().startswith("ENCUT"):
            encut = float(line.split('=')[1].split()[0])
        elif line.upper().startswith("PREC"):
            prec = line.split('=')[1].strip().capitalize()

# --- read lattice vectors ---
pos = Poscar.from_file(POSCAR_FILE)
lattice = pos.structure.lattice.matrix   # 3×3 Å
a1, a2, a3 = lattice

# --- derive Gcut (Å⁻¹) ---
Gcut = np.sqrt(encut / HBAR2_OVER_2ME)

# --- multiplier from PREC ---
if prec.lower().startswith("acc"):
    c = 2.0
elif prec.lower().startswith("high"):
    c = 2.0
elif prec.lower().startswith("normal"):
    c = 1.5
else:
    c = 1.5  # safe default

# --- compute ideal grid counts (before rounding) ---
lengths = [np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)]
N_ideal = [c * Gcut * L / np.pi for L in lengths]

# --- helper: round to nearest FFT-friendly integer (factors 2,3,5) ---
def nearest_fft_integer(x):
    best = None
    min_err = 1e9
    for n in range(max(8, int(x*0.8)), int(x*1.2)+8):
        tmp = n
        for p in [2,3,5]:
            while tmp % p == 0 and tmp > 1:
                tmp //= p
        if tmp == 1:
            err = abs(n - x)
            if err < min_err:
                best, min_err = n, err
    return best if best else int(round(x))

NG = [nearest_fft_integer(n) for n in N_ideal]
NGF = [2*n for n in NG]

print("ENCUT = %.1f eV, PREC = %s" % (encut, prec))
print("Cell lengths (Å): a=%.3f  b=%.3f  c=%.3f" % tuple(lengths))
print("Estimated grids:")
print("  NGX, NGY, NGZ   = %3d  %3d  %3d" % tuple(NG))
print("  NGXF,NGYF,NGZF  = %3d  %3d  %3d" % tuple(NGF))
```

# Known Issues 
- incar_modification.md is created inside the calculation directory instead of the root directory of folder. 