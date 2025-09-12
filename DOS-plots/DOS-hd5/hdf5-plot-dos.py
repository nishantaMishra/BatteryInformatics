# working as intended on Tuesday 01 October 2024 01:54:18 PM IST
# program for plotting electronic density of states
# it only requires .h5 file.
# 
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path to your .h5 file
file_path = 'vaspout.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as h5file:
    # Access the energy and DOS data
    energies = h5file['results/electron_dos/energies'][()]
    dos_total = h5file['results/electron_dos/dos'][0]  # Total DOS
    dos_partial = h5file['results/electron_dos/dospar'][0]  # Projected DOS (PDOS)
    efermi = h5file['results/electron_dos/efermi'][()]  # Fermi energy
    
    # Shift energies relative to Fermi level
    energies -= efermi

    # Sum over all atoms for each orbital
    pdos_s = np.sum(dos_partial[:, 0, :], axis=0)  # s-orbital
    pdos_p = np.sum(dos_partial[:, 1:4, :], axis=(0, 1))  # p-orbitals (px, py, pz)
    pdos_d = np.sum(dos_partial[:, 4:9, :], axis=(0, 1))  # d-orbitals (dxy, dyz, dz2, dxz, dx2-y2)

# Plotting the total DOS and PDOS
plt.figure(figsize=(8, 6))

# Plot total DOS
plt.plot(energies, dos_total, label='Total DOS', color='blue')

# Plot partial DOS for different orbitals
plt.plot(energies, pdos_s, label='s-orbital', color='green')
plt.plot(energies, pdos_p, label='p-orbitals', color='red')
plt.plot(energies, pdos_d, label='d-orbitals', color='purple')

# Add labels and title
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States (States/eV)')
plt.title('Total and Partial Density of States')

# Add a vertical line for the Fermi level (at 0 eV)
plt.axvline(0, color='k', linestyle='--', label='Fermi Level')

# Show legend and plot
plt.legend()
plt.show()

