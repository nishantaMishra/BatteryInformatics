# can plot for each element.

import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = 'vaspout.h5'

with h5py.File(file_path, 'r') as h5file:
    # Load DOS data
    energies = h5file['results/electron_dos/energies'][()]
    dos_total = h5file['results/electron_dos/dos'][0]
    dos_partial = h5file['results/electron_dos/dospar'][0]
    efermi = h5file['results/electron_dos/efermi'][()]

    # Extract per-atom element names (already stored as bytes)
    atom_species = h5file['results/positions/ion_types'][()]  # e.g. [b'Fe', b'Fe', b'O', b'O']

# Decode bytes to proper strings
atom_symbols = [s.decode('utf-8') for s in atom_species]

# Shift energy scale so Fermi level = 0 eV
energies -= efermi

# Identify unique elements
unique_elements = sorted(set(atom_symbols))

# Colour map for plotting
colours = plt.cm.tab10(np.linspace(0, 1, len(unique_elements)))

# Set up plot
plt.figure(figsize=(8, 6))

# Plot total DOS
plt.plot(energies, dos_total, color='black', label='Total DOS', linewidth=1.2)

# Plot PDOS for each element
for i, element in enumerate(unique_elements):
    atom_indices = [j for j, el in enumerate(atom_symbols) if el == element]
    pdos_element = np.sum(dos_partial[atom_indices, :, :], axis=(0, 1))
    plt.plot(energies, pdos_element, label=f'{element} (PDOS)', color=colours[i])

# Fermi level line
plt.axvline(0, color='k', linestyle='--', label='Fermi Level')

# Labels, title, and legend
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States (States/eV)')
plt.title('Projected Density of States by Element')
plt.legend()
plt.tight_layout()
plt.show()
