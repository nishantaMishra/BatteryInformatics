#!/bin/bash

# Single Point Calculation Generator
# This script sets up directories and files for single point calculations

set -e  # Exit on any error

echo "=== Single Point Calculation Generator ==="
echo

# Get the relaxed system directory from user
read -p "Enter the path to the relaxed system directory: " relaxed_dir

# Check if directory exists
if [[ ! -d "$relaxed_dir" ]]; then
    echo "Error: Directory '$relaxed_dir' does not exist!"
    exit 1
fi

# Check for required files
required_files=("CONTCAR" "POTCAR" "KPOINTS" "INCAR")
for file in "${required_files[@]}"; do
    if [[ ! -f "$relaxed_dir/$file" ]]; then
        echo "Error: Required file '$file' not found in '$relaxed_dir'!"
        exit 1
    fi
done

echo "Found all required files in '$relaxed_dir'"
echo

# Store the script directory before changing directories
script_dir="$(dirname "$(realpath "$0")")"

# Change to the relaxed system directory so all temp files are created there
cd "$relaxed_dir"

# Create directories in the relaxed system directory
echo "Creating directories..."
mkdir -p mol_in_slab_box surf sys

# Copy files to all directories
echo "Copying files..."
for dir in mol_in_slab_box surf sys; do
    cp "CONTCAR" "$dir/POSCAR"
    cp "POTCAR" "$dir/POTCAR"
    cp "KPOINTS" "$dir/KPOINTS"
    cp "INCAR" "$dir/INCAR"
    echo "  Files copied to $dir/"
done

echo
echo "Now launching interactive z-cut determination..."
echo "Instructions:"
echo "  - Use the slider to set the z-cutoff value"
echo "  - Click atoms to select/exclude if needed"
echo "  - Close the window when satisfied with the selection"
echo

# Use poscar_splitter.py with absolute path
python3 "$script_dir/poscar_splitter.py" "CONTCAR"

# Check if the splitter was successful
if [[ ! -f "surf_atoms.tmp" ]] || [[ ! -f "mol_atoms.tmp" ]] || [[ ! -f "z_cutoff.tmp" ]]; then
    echo "Error: Failed to determine atom selections. Cleaning up..."
    rm -rf mol_in_slab_box surf sys
    rm -f surf_atoms.tmp mol_atoms.tmp z_cutoff.tmp
    exit 1
fi

# Read the results
z_cutoff=$(cat z_cutoff.tmp)
echo
echo "Z-cutoff determined: $z_cutoff Å"

# Generate POSCAR files
echo "Generating POSCAR files..."

# Generate surf/POSCAR (atoms below z-cutoff)
if [[ -s surf_atoms.tmp ]]; then
    python3 "$script_dir/extract_atoms.py" "CONTCAR" surf_atoms.tmp "surf/POSCAR"
    echo "  surf/POSCAR created (surface atoms below z-cutoff)"
else
    echo "  Warning: No surface atoms found below z-cutoff"
fi

# Generate mol_in_slab_box/POSCAR (atoms above z-cutoff)
if [[ -s mol_atoms.tmp ]]; then
    python3 "$script_dir/extract_atoms.py" "CONTCAR" mol_atoms.tmp "mol_in_slab_box/POSCAR"
    echo "  mol_in_slab_box/POSCAR created (molecule atoms above z-cutoff)"
else
    echo "  Warning: No molecule atoms found above z-cutoff"
fi

# sys/POSCAR remains unchanged (already copied)
echo "  sys/POSCAR ready (unchanged from original)"

# Extract grid parameters from OUTCAR if it exists
echo "Extracting grid parameters from OUTCAR..."
outcar_path="OUTCAR"
if [[ -f "$outcar_path" ]]; then
    grid_info=$(grep "support grid" "$outcar_path" | head -1)
    if [[ -n "$grid_info" ]]; then
        ngxf=$(echo "$grid_info" | grep -o "NGXF=\s*[0-9]*" | grep -o "[0-9]*")
        ngyf=$(echo "$grid_info" | grep -o "NGYF=\s*[0-9]*" | grep -o "[0-9]*")
        ngzf=$(echo "$grid_info" | grep -o "NGZF=\s*[0-9]*" | grep -o "[0-9]*")
        echo "  Found grid parameters: NGXF=$ngxf, NGYF=$ngyf, NGZF=$ngzf"
    else
        echo "  Warning: Could not find grid parameters in OUTCAR"
        ngxf=""; ngyf=""; ngzf=""
    fi
else
    echo "  Warning: OUTCAR not found, grid parameters will be empty"
    ngxf=""; ngyf=""; ngzf=""
fi

# Create incar_modification.md template
echo "Creating INCAR modification template..."
cat > incar_modification.md << EOF
# INCAR Modification Template
# Lines starting with # are comments and will be ignored
# Uncomment and modify the lines you want to change in INCAR files

SYSTEM = <directory-name>
NSW = 0
IBRION = -1
ISIF = 2
PREC = Accurate
EDIFF = 1E-6
LCHARG = .TRUE.
#LWAVE = .TRUE.
LVTOT = .TRUE.
LORBIT = 11
LAECHG = .TRUE.
NGXF = $ngxf
NGYF = $ngyf
NGZF = $ngzf

# Add any additional INCAR modifications below:
EOF

echo "  incar_modification.md created"
echo "  You can edit incar_modification.md to customize INCAR modifications"
echo "  Press Enter to continue with INCAR modifications, or Ctrl+C to exit and edit the file first"
read -p ""

# Modify INCAR files in each directory
echo "Modifying INCAR files..."
for dir in mol_in_slab_box surf sys; do
    if [[ -f "$dir/INCAR" ]] && [[ -f "incar_modification.md" ]]; then
        echo "  Modifying $dir/INCAR..."
        
        # Read modifications from incar_modification.md (skip comments)
        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            
            # Extract tag and value
            if [[ "$line" =~ ^[[:space:]]*([A-Z_]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
                tag="${BASH_REMATCH[1]}"
                value="${BASH_REMATCH[2]}"
                
                # Handle special case for SYSTEM tag
                if [[ "$tag" == "SYSTEM" ]]; then
                    if [[ "$value" == "<directory-name>" ]]; then
                        value="$dir"
                    fi
                fi
                
                # Update or add the tag in INCAR
                if grep -q "^[[:space:]]*$tag[[:space:]]*=" "$dir/INCAR"; then
                    # Tag exists, replace it
                    sed -i "s/^[[:space:]]*$tag[[:space:]]*=.*/$tag = $value/" "$dir/INCAR"
                else
                    # Tag doesn't exist, add it
                    echo "$tag = $value" >> "$dir/INCAR"
                fi
            fi
        done < incar_modification.md
        
        echo "    $dir/INCAR modified successfully"
    else
        echo "    Warning: $dir/INCAR or incar_modification.md not found"
    fi
done

# Clean up temporary files
rm -f surf_atoms.tmp mol_atoms.tmp z_cutoff.tmp

echo
echo "=== Setup Complete ==="
echo "Directories created:"
echo "  - mol_in_slab_box/  (molecule + slab box)"
echo "  - surf/             (surface atoms only)"
echo "  - sys/              (complete system)"
echo
echo "All directories contain POSCAR, POTCAR, KPOINTS, and modified INCAR files."
echo "INCAR modification template saved as incar_modification.md"
echo "Ready for single point calculations!"
