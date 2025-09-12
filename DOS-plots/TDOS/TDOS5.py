import os
import numpy as np
import matplotlib.pyplot as plt

def read_tdos_file(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] == 3:
            # Spin-polarized data: energy, up, down
            energy = data[:, 0]
            tdos_up = data[:, 1]
            tdos_down = data[:, 2]
        elif data.shape[1] == 2:
            # Non-magnetic data: energy, total only
            energy = data[:, 0]
            tdos_up = data[:, 1]
            tdos_down = data[:, 1]
            print("Info: Detected non-magnetic TDOS (single column for DOS). Using identical UP and DOWN channels.")
        else:
            print(f"Error: Unexpected TDOS format in {file_path}. Expected 2 or 3 columns.")
            return None, None, None
        return energy, tdos_up, tdos_down
    except Exception as e:
        print(f"Error reading TDOS file: {e}")
        return None, None, None

def is_zero(x, tol=1e-5):
    return abs(x) < tol

def detect_bandgap_exact(energy, tdos_up, tdos_down):
    data = list(zip(energy, tdos_up, tdos_down))
    data.sort()  # ensure sorted by energy ascending

    min_e, min_up, min_down = data[0]
    if not (is_zero(min_up) and is_zero(min_down)):
        return None  # No bandgap detected

    fermi_idx = np.argmin(np.abs(energy))

    vbm = None
    for i in range(fermi_idx, -1, -1):
        if is_zero(tdos_up[i]) and is_zero(tdos_down[i]):
            vbm = energy[i]
        else:
            break

    cbm = None
    for i in range(fermi_idx, len(energy)):
        if is_zero(tdos_up[i]) and is_zero(tdos_down[i]):
            cbm = energy[i]
        else:
            break

    if vbm is None or cbm is None or cbm <= vbm:
        return None

    return vbm, cbm, cbm - vbm

def analyze_tdos(energy, tdos_up, tdos_down, symmetry_tol=0.01):
    print("\n----- Analysis Results -----")

    max_dos = max(np.max(np.abs(tdos_up)), np.max(np.abs(tdos_down)), 1e-8)
    sym_diff = np.mean(np.abs(tdos_up + tdos_down)) / max_dos
    if sym_diff < symmetry_tol:
        print("✅ Non-magnetic (TDOS-UP and DOWN are symmetric).")
    else:
        print("⚠️  Magnetic (TDOS-UP and DOWN differ noticeably).")

    bandgap_info = detect_bandgap_exact(energy, tdos_up, tdos_down)

    if bandgap_info:
        vbm, cbm, width = bandgap_info
        print(f"✅ Bandgap region: {vbm:.3f} to {cbm:.3f} eV → Width: {width:.3f} eV")
    else:
        print("⚠️  No bandgap detected.")

    print("-----------------------------\n")
    return bandgap_info

def plot_tdos(energy, tdos_up, tdos_down, directory):
    bandgap_info = analyze_tdos(energy, tdos_up, tdos_down)
    avg = 0.5 * (tdos_up + np.abs(tdos_down))

    plt.figure(figsize=(8, 6))
    plt.plot(energy, tdos_up, label='TDOS-UP', color='C0')
    plt.plot(energy, tdos_down, label='TDOS-DOWN', color='C1')
    plt.plot(energy, avg, label='AVG', color='C2', linestyle='--')
    plt.axvline(0, color='k', linestyle='--', label='Fermi (0 eV)')

    if bandgap_info:
        start, end, _ = bandgap_info
        plt.axvspan(start, end, color='red', alpha=0.2, label='Bandgap region')
        plt.text((start + end) / 2, 0.02, f"Gap = {(end - start):.2f} eV", ha='center', color='red')

    plt.title(f"Total Density of States — {os.path.basename(directory)}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    user_dir = input("Path to directory containing TDOS.dat: ").strip()
    path = os.path.join(user_dir, "TDOS.dat")
    energy, tdos_up, tdos_down = read_tdos_file(path)
    if energy is not None:
        plot_tdos(energy, tdos_up, tdos_down, user_dir)
