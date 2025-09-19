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
    # Deprecated legacy function retained for compatibility (not used now)
    return None

# --- New advanced bandgap detection utilities ---

def _find_zero_intervals(energy, tdos_up, tdos_down, zero_tol=1e-5):
    intervals = []
    in_zero = False
    start_e = None
    for e, u, d in zip(energy, tdos_up, tdos_down):
        if abs(u) < zero_tol and abs(d) < zero_tol:
            if not in_zero:
                in_zero = True
                start_e = e
        else:
            if in_zero:
                intervals.append((start_e, prev_e))
                in_zero = False
        prev_e = e
    if in_zero:
        intervals.append((start_e, prev_e))
    return [(s, e) for s, e in intervals if e > s]

def detect_bandgap_advanced(energy, tdos_up, tdos_down, zero_tol=1e-5,
                            min_gap=0.1, max_gap=7.0, fermi_window=1.0):
    """Detect bandgap possibly shifted from Fermi.

    Offset gap rule (updated):
      Accept if Fermi not inside zero-DOS interval BUT either VBM or CBM lies within
      ±fermi_window of 0 eV.
    """
    intervals = _find_zero_intervals(energy, tdos_up, tdos_down, zero_tol=zero_tol)
    if not intervals:
        return None
    candidates = []
    for s, e in intervals:
        w = e - s
        if w >= min_gap and w <= max_gap:
            candidates.append((s, e, w))
    if not candidates:
        return None
    fermi_gap = [c for c in candidates if c[0] <= 0 <= c[1]]
    if fermi_gap:
        fermi_gap.sort(key=lambda x: (-x[2], abs((x[0]+x[1])/2)))
        s, e, w = fermi_gap[0]
        return {'vbm': s, 'cbm': e, 'width': w, 'type': 'fermi_gap', 'message': 'Bandgap spans the Fermi level.'}
    # New edge-based offset detection
    edge_offset = []
    for s, e, w in candidates:
        vbm_dist = abs(s)
        cbm_dist = abs(e)
        if min(vbm_dist, cbm_dist) <= fermi_window:  # Either edge close to Fermi
            edge_offset.append((s, e, w, min(vbm_dist, cbm_dist)))
    if edge_offset:
        edge_offset.sort(key=lambda x: (x[3], -x[2]))  # prioritize closer edge then wider gap
        s, e, w, _ = edge_offset[0]
        return {'vbm': s, 'cbm': e, 'width': w, 'type': 'offset_gap',
                'message': 'Caution: Either the compound is metallic or there is bandgap shift.'}
    return None

def analyze_tdos(energy, tdos_up, tdos_down, symmetry_tol=0.01):
    print("\n----- Analysis Results -----")
    
    # Check if data is spin-polarized by comparing if up and down are identical arrays
    is_spin_polarized = not np.array_equal(tdos_up, tdos_down)
    
    if is_spin_polarized:
        max_dos = max(np.max(np.abs(tdos_up)), np.max(np.abs(tdos_down)), 1e-12)
        sym_diff = np.mean(np.abs(tdos_up + tdos_down)) / max_dos
        if sym_diff < symmetry_tol:
            print("✅ Non-magnetic (TDOS-UP and DOWN are symmetric).")
        else:
            print("⚠️  Magnetic (TDOS-UP and DOWN differ noticeably).")
    else:
        print("ℹ️  Non-spin-polarized calculation detected.")

    gap = detect_bandgap_advanced(energy, tdos_up, tdos_down)
    if gap:
        if gap['type'] == 'fermi_gap':
            print(f"✅ Bandgap detected: {gap['vbm']:.3f} to {gap['cbm']:.3f} eV → Width: {gap['width']:.3f} eV")
        else:
            print(f"⚠️  Offset bandgap candidate: {gap['vbm']:.3f} to {gap['cbm']:.3f} eV → Width: {gap['width']:.3f} eV")
            print("    Note: Fermi lies outside this gap region.")
        print(f"    {gap['message']}")
    else:
        print("⚠️  No bandgap detected (metallic or gap outside criteria).")

    print("-----------------------------\n")
    return gap

def plot_tdos(energy, tdos_up, tdos_down, directory):
    bandgap_info = analyze_tdos(energy, tdos_up, tdos_down)
    avg = 0.5 * (tdos_up + np.abs(tdos_down))

    plt.figure(figsize=(8, 6))
    plt.plot(energy, tdos_up, label='TDOS-UP', color='C0')
    plt.plot(energy, tdos_down, label='TDOS-DOWN', color='C1')
    plt.plot(energy, avg, label='AVG', color='C2', linestyle='--')
    plt.axvline(0, color='k', linestyle='--', label='Fermi (0 eV)')

    if bandgap_info:
        s = bandgap_info['vbm']; e_ = bandgap_info['cbm']
        plt.axvspan(s, e_, color='red', alpha=0.18,
                    label='Bandgap' if bandgap_info['type']=='fermi_gap' else 'Offset gap')
        y_text = 0.02 * max(1e-6, np.max([np.max(np.abs(tdos_up)), np.max(np.abs(tdos_down))]))
        plt.text(0.5*(s+e_), y_text, f"{bandgap_info['width']:.2f} eV", ha='center', color='red')
        if bandgap_info['type'] == 'offset_gap':
            plt.text(0.5*(s+e_), y_text*3, 'Offset from Fermi', ha='center', color='red', fontsize=8)

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
