# Plot for each element and each orbital

import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = 'vaspout.h5'

with h5py.File(file_path, 'r') as h5file:
    energies = h5file['results/electron_dos/energies'][()]
    dos_total = h5file['results/electron_dos/dos'][0]
    dos_partial = h5file['results/electron_dos/dospar'][0]
    efermi = h5file['results/electron_dos/efermi'][()]
    atom_species = h5file['results/positions/ion_types'][()]  # e.g. [b'Fe', b'Fe', b'O', b'O']

# Decode bytes to strings
atom_symbols = [s.decode('utf-8') for s in atom_species]

# Shift energies so that EF = 0 eV
energies -= efermi

# Identify unique elements
unique_elements = sorted(set(atom_symbols))

# Prepare a colour map for different orbital types
orbital_colours = {
    's': 'green',
    'p': 'red',
    'd': 'purple'
}

plt.figure(figsize=(10, 7))

# Plot total DOS
plt.plot(energies, dos_total, color='black', label='Total DOS', linewidth=1.2)

# Loop over each element and plot orbital contributions
for element in unique_elements:
    atom_indices = [j for j, el in enumerate(atom_symbols) if el == element]

    # Sum PDOS for atoms of this element
    pdos_element = np.sum(dos_partial[atom_indices, :, :], axis=0)  # shape: (n_orbitals, n_energies)

    # Extract orbital-resolved PDOS
    pdos_s = pdos_element[0, :]                      # s (index 0)
    pdos_p = np.sum(pdos_element[1:4, :], axis=0)    # p (1:4 → px, py, pz)
    pdos_d = np.sum(pdos_element[4:9, :], axis=0)    # d (4:9 → dxy, dyz, dz2, dxz, dx2–y2)

    # Plot each orbital contribution
    plt.plot(energies, pdos_s, color=orbital_colours['s'], linestyle='-', label=f'{element}-s')
    plt.plot(energies, pdos_p, color=orbital_colours['p'], linestyle='--', label=f'{element}-p')
    plt.plot(energies, pdos_d, color=orbital_colours['d'], linestyle='-.', label=f'{element}-d')

# Fermi level
plt.axvline(0, color='k', linestyle='--', label='Fermi Level')

# Labels, title, and legend
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States (States/eV)')
plt.title('Projected Density of States by Element and Orbital')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()
