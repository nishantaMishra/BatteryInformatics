# Color Scheme Manager
# Handles loading and managing different color schemes for elements
# All color schemes now read directly from ElementColorSchemes.yaml

import os
import sys
import yaml

def rgb_to_hex(rgb_list):
    """Convert RGB list to hex string"""
    return '#{:02x}{:02x}{:02x}'.format(rgb_list[0], rgb_list[1], rgb_list[2])

def load_all_color_schemes():
    """Load all color schemes from YAML file"""
    # Get path to YAML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'ElementColorSchemes.yaml')
    
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            
            color_schemes = {}
            
            # Load each scheme from YAML and convert RGB to hex
            for scheme_name, elements in data.items():
                if scheme_name.lower() != 'extras':  # Skip extras section
                    scheme_colors = {}
                    for element, rgb in elements.items():
                        if isinstance(rgb, list) and len(rgb) == 3:
                            scheme_colors[element] = rgb_to_hex(rgb)
                    color_schemes[scheme_name.lower()] = scheme_colors
            
            return color_schemes
            
    except Exception as e:
        print(f"Warning: Could not load color schemes from YAML: {e}")
        return {}

# Load all color schemes from YAML
ALL_SCHEMES = load_all_color_schemes()

# Available color schemes with aliases
AVAILABLE_SCHEMES = {
    'vesta': ALL_SCHEMES.get('vesta', {}),
    'jmol': ALL_SCHEMES.get('jmol', {}), 
    'user': ALL_SCHEMES.get('user', {}),
    'custom': ALL_SCHEMES.get('user', {}),  # Alias for user
}

# Add CPK and MP as hardcoded fallbacks if not in YAML
if 'cpk' not in ALL_SCHEMES:
    AVAILABLE_SCHEMES['cpk'] = {
        'H': '#FFFFFF', 'C': '#000000', 'N': '#0000FF', 'O': '#FF0000',
        'F': '#00FF00', 'Cl': '#00FF00', 'Br': '#A52A2A', 'I': '#9400D3',
        'P': '#FFA500', 'S': '#FFFF00', 'B': '#FFC0CB', 'Li': '#800080',
        'Na': '#800080', 'Mg': '#228B22', 'Al': '#808080', 'Si': '#DAA520',
        'K': '#800080', 'Ca': '#808080', 'Ti': '#808080', 'Cr': '#808080',
        'Mn': '#808080', 'Fe': '#FFA500', 'Co': '#FFA500', 'Ni': '#A52A2A',
        'Cu': '#A52A2A', 'Zn': '#A52A2A', 'As': '#FFA500', 'Se': '#FFA500',
        'Mo': '#808080', 'Ag': '#808080', 'Sn': '#808080', 'Te': '#FFA500',
        'Ba': '#FFA500', 'W': '#808080', 'Au': '#DAA520', 'Hg': '#808080', 'Pb': '#808080'
    }
else:
    AVAILABLE_SCHEMES['cpk'] = ALL_SCHEMES['cpk']

if 'mp' not in ALL_SCHEMES:
    AVAILABLE_SCHEMES['mp'] = {
        'H': '#FFFFFF', 'He': '#FFC0CB', 'Li': '#800080', 'Be': '#228B22',
        'B': '#FFB6C1', 'C': '#808080', 'N': '#0000FF', 'O': '#FF0000',
        'F': '#90EE90', 'Ne': '#FFB6C1', 'Na': '#800080', 'Mg': '#228B22',
        'Al': '#FFA500', 'Si': '#F0E68C', 'P': '#FFA500', 'S': '#FFFF00',
        'Cl': '#00FF00', 'Ar': '#FFC0CB', 'K': '#800080', 'Ca': '#808080',
        'Ti': '#808080', 'Fe': '#FFA500', 'Cu': '#FFA500', 'Au': '#FFD700'
    }
else:
    AVAILABLE_SCHEMES['mp'] = ALL_SCHEMES['mp']

# Add materials as alias for MP
AVAILABLE_SCHEMES['materials'] = AVAILABLE_SCHEMES['mp']

def get_available_schemes():
    """Return list of available color scheme names"""
    return list(AVAILABLE_SCHEMES.keys())

def get_scheme_colors(scheme_name):
    """Get colors for a specific scheme"""
    scheme_name = scheme_name.lower()
    return AVAILABLE_SCHEMES.get(scheme_name, {})

def get_element_color(element, scheme_name):
    """Get color for a specific element in a specific scheme"""
    colors = get_scheme_colors(scheme_name)
    element_symbol = element.capitalize()
    return colors.get(element_symbol, None)

def apply_color_scheme(elements, plotting_info, scheme_name):
    """Apply color scheme to elements and return fill_colors dictionary"""
    fill_colors = {}
    scheme_colors = get_scheme_colors(scheme_name)
    
    if not scheme_colors:
        print(f"⚠ Color scheme '{scheme_name}' not found or empty")
        return fill_colors
    
    print(f"\n=== {scheme_name.upper()} Color Scheme Applied ===")
    
    for element in elements:
        # Allow color assignment for TDOS as well as elements
        if element in plotting_info:
            if element == 'tot':
                # Try multiple possible keys for TDOS in the YAML: 'TDOS', 'Tot', 'tot'
                tkeys = ['TDOS', 'Tot', 'tdos', 'tot']
                assigned = False
                for k in tkeys:
                    if k in scheme_colors:
                        fill_colors[element] = scheme_colors[k]
                        print(f"✓ TDOS assigned {scheme_name.upper()} color: {scheme_colors[k]}")
                        assigned = True
                        break
                if not assigned:
                    print(f"⚠ TDOS not found in {scheme_name.upper()} scheme, using default color")
            else:
                element_symbol = element.capitalize()
                if element_symbol in scheme_colors:
                    fill_colors[element] = scheme_colors[element_symbol]
                    print(f"✓ {element} assigned {scheme_name.upper()} color: {scheme_colors[element_symbol]}")
                else:
                    print(f"⚠ {element} not found in {scheme_name.upper()} scheme, using default color")
    
    return fill_colors

def list_schemes_info():
    """Print information about available color schemes"""
    print("\n" + "="*70)
    print("AVAILABLE COLOR SCHEMES")
    print("="*70)
    
    schemes_info = {
        'vesta': "VESTA 3 visualization software colors (from YAML)",
        'jmol': "Jmol molecular visualization software colors (from YAML)",
        'user': "User-defined custom colors (from YAML)",
        'custom': "Alias for User-defined custom colors",
        'cpk': "Classic Corey-Pauling-Koltun chemistry colors",
        'mp': "Materials Project database colors",
        'materials': "Alias for Materials Project colors"
    }
    
    for scheme, description in schemes_info.items():
        if scheme in AVAILABLE_SCHEMES and AVAILABLE_SCHEMES[scheme]:
            element_count = len(AVAILABLE_SCHEMES[scheme])
            print(f"• {scheme:<12} - {description} ({element_count} elements)")
        else:
            print(f"• {scheme:<12} - {description} (not available)")
    
    print("\nUsage examples:")
    print("  Ti s p d, O p tot --colour vesta")
    print("  --all -c jmol fill")
    print("  Ti d, O p --colour user --grid")
    print("  tot -c mp fill")
    print("="*70)