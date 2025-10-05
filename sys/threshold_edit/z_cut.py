#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from pymatgen.io.vasp import Poscar
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- User options ---
poscar_file = "CONTCAR"
show_z_cut = True
z_cut_guess = None  # e.g., 18.0  (optional fixed horizontal line)
show_cell = True    # Show unit cell boundaries
elev_angle = 0      # Initial elevation viewing angle (0 for direct view of x-y plane)
azim_angle = 90     # Initial azimuth viewing angle (90 for x-axis into screen, y-axis to left)
use_atomic_radii = True  # Scale markers by atomic radii
save_figure = False  # Option to save the figure

# Approximate atomic radii in Angstroms (for visualization)
atomic_radii = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Mo': 1.45, 'W': 1.35, 'Au': 1.36, 'Pt': 1.36, 'Pd': 1.39
}
# Default radius for elements not in the dictionary
default_radius = 1.0

# --- Read structure ---
pos = Poscar.from_file(poscar_file)
structure = pos.structure

# Get Cartesian coordinates and elements
coords = np.array([site.coords for site in structure])
elems = [str(site.specie) for site in structure]
z = coords[:, 2]

# --- Colour map by element ---
unique_elems = sorted(set(elems))
colours = plt.cm.tab10(np.linspace(0, 1, len(unique_elems)))
colour_map = {el: colours[i] for i, el in enumerate(unique_elems)}

# --- Create figure with more space for widgets ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.25)  # Make space for sliders

# Function to get marker sizes based on atomic radii
def get_marker_sizes(element, count, scale=80):
    """Generate marker sizes for the specified element, repeated count times"""
    if use_atomic_radii:
        size = atomic_radii.get(element, default_radius) * scale
    else:
        size = scale
    return size  # Return a scalar value instead of a list

# Function to draw the unit cell
def draw_unit_cell(ax, lattice):
    # Get lattice vectors
    a, b, c = lattice.matrix
    
    # Define the vertices of the unit cell
    vertices = np.array([
        [0, 0, 0], [a[0], a[1], a[2]], [b[0], b[1], b[2]], [c[0], c[1], c[2]],
        [a[0]+b[0], a[1]+b[1], a[2]+b[2]], [a[0]+c[0], a[1]+c[1], a[2]+c[2]],
        [b[0]+c[0], b[1]+c[1], b[2]+c[2]], [a[0]+b[0]+c[0], a[1]+b[1]+c[1], a[2]+b[2]+c[2]]
    ])
    
    # Define the faces of the unit cell
    faces = [
        [vertices[0], vertices[1], vertices[4], vertices[2]],  # Bottom face
        [vertices[0], vertices[1], vertices[5], vertices[3]],  # Front face
        [vertices[0], vertices[2], vertices[6], vertices[3]],  # Left face
        [vertices[7], vertices[4], vertices[1], vertices[5]],  # Right face
        [vertices[7], vertices[4], vertices[2], vertices[6]],  # Back face
        [vertices[7], vertices[5], vertices[3], vertices[6]]   # Top face
    ]
    
    # Create a Poly3DCollection
    cell = Poly3DCollection(faces, alpha=0.1, facecolor='gray', edgecolor='black', linewidths=1)
    ax.add_collection3d(cell)

# Add these global variables to store view state
view_limits = None  # Will store (xmin, xmax, ymin, ymax, zmin, zmax)
initial_limits_set = False  # Flag to track if we've set initial limits

# Function to update the plot based on z_cut value
def update_plot(z_value, show_above=True, show_below=True):
    global view_limits, initial_limits_set
    
    # Store current view limits before clearing the plot
    if not view_limits and initial_limits_set:
        view_limits = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()
        )
    
    ax.clear()
    
    # Calculate default view if needed
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0
    mid_x = (coords[:,0].max() + coords[:,0].min()) / 2.0
    mid_y = (coords[:,1].max() + coords[:,1].min()) / 2.0
    mid_z = (coords[:,2].max() + coords[:,2].min()) / 2.0
    
    # Set view limits - use stored limits if available, otherwise use defaults
    if view_limits and not ax.button_pressed:  # Don't restore if user is currently panning/zooming
        ax.set_xlim(view_limits[0])
        ax.set_ylim(view_limits[1])
        ax.set_zlim(view_limits[2])
    else:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        if not initial_limits_set:
            initial_limits_set = True
    
    ax.set_box_aspect([1,1,1])
    
    # Counter for visible atoms
    visible_atoms = 0
    visible_by_element = {}
    
    # Track if we have any visible elements for the legend
    has_visible_elements = False
    
    # Filter atoms based on z-cut and visibility options
    for el in unique_elems:
        idx = [i for i, e in enumerate(elems) if e == el]
        
        # Filter by z position
        if show_above and show_below:
            idx_to_show = idx
        elif show_above:
            idx_to_show = [i for i in idx if coords[i, 2] >= z_value]
        elif show_below:
            idx_to_show = [i for i in idx if coords[i, 2] < z_value]
        else:
            idx_to_show = []
        
        visible_atoms += len(idx_to_show)
        visible_by_element[el] = len(idx_to_show)
        
        if idx_to_show:
            has_visible_elements = True
            # Use a scalar size value for all points of this element
            size = get_marker_sizes(el, len(idx_to_show))
            ax.scatter(coords[idx_to_show, 0], coords[idx_to_show, 1], coords[idx_to_show, 2],
                      label=el, s=size, alpha=0.8, color=colour_map[el])
    
    # Draw unit cell if enabled
    if show_cell:
        draw_unit_cell(ax, structure.lattice)
    
    # Draw the z-cut plane
    if show_z_cut:
        x_max = np.max(coords[:, 0])
        y_max = np.max(coords[:, 1])
        xx, yy = np.meshgrid([0, x_max], [0, y_max])
        zz = np.ones_like(xx) * z_value
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='r')
        ax.text(x_max/2, y_max/2, z_value, f'z = {z_value:.2f} Å', color='r')
    
    # Set labels with customized positioning for better clarity in this view
    ax.set_xlabel("x (Å)", labelpad=10)
    ax.set_ylabel("y (Å)", labelpad=10)
    ax.set_zlabel("z (Å)", labelpad=10)
    
    # Only create a legend if there are visible elements
    if has_visible_elements:
        ax.legend()
    
    # Set view angle (modified to ensure consistent orientation)
    ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
    
    # Add structure info
    formula = structure.composition.reduced_formula
    num_atoms = len(structure)
    info_text = f"{formula} - {num_atoms} atoms"
    ax.set_title(f"Atomic positions: {info_text}")
    
    # Add text annotation for visible atom count
    atom_text = f"Visible: {visible_atoms}/{num_atoms} atoms"
    element_counts = ", ".join([f"{el}: {visible_by_element[el]}" for el in unique_elems])
    
    # Create a text box with counts
    ax.text2D(0.02, 0.98, atom_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add per-element counts below
    ax.text2D(0.02, 0.93, element_counts, transform=ax.transAxes, 
              fontsize=9, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    fig.canvas.draw_idle()

# Add a new function to capture view state changes from toolbar
def on_view_change(event):
    global view_limits
    # Update view_limits when user manually changes the view
    if hasattr(event, 'inaxes') and event.inaxes == ax:
        view_limits = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()
        )

# Connect the view change event handler
fig.canvas.mpl_connect('button_release_event', on_view_change)

# Initial z_cut value
if z_cut_guess is None:
    z_cut_guess = (np.max(z) + np.min(z)) / 2  # Default to middle of z-range

# Create sliders for interactive controls
z_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
z_slider = Slider(z_ax, 'Z-cut (Å)', np.min(z), np.max(z), valinit=z_cut_guess)

elev_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
elev_slider = Slider(elev_ax, 'Elevation', -90, 90, valinit=elev_angle)

azim_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
azim_slider = Slider(azim_ax, 'Azimuth', 0, 360, valinit=azim_angle)

# Create checkboxes for display options
check_ax = plt.axes([0.025, 0.05, 0.15, 0.15])
check = CheckButtons(check_ax, ['Above z-cut', 'Below z-cut', 'Show cell'], 
                    [True, True, show_cell])

# Update function for z-cut slider
def update_z(val):
    # Get the status of checkboxes by index instead of checking for labels in the status
    show_above = check.get_status()[0]  # First checkbox (Above z-cut)
    show_below = check.get_status()[1]  # Second checkbox (Below z-cut)
    show_cell_option = check.get_status()[2]  # Third checkbox (Show cell)
    update_plot(z_slider.val, show_above, show_below)
z_slider.on_changed(update_z)

# Update function for view angle sliders
def update_view(val):
    global view_limits
    # When changing view angles, we should clear stored limits
    # since the projection is changing
    view_limits = None
    ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
    fig.canvas.draw_idle()
elev_slider.on_changed(update_view)
azim_slider.on_changed(update_view)

# Update function for checkboxes
def update_options(label):
    # Get the status of checkboxes by index
    show_above = check.get_status()[0]  # First checkbox (Above z-cut)
    show_below = check.get_status()[1]  # Second checkbox (Below z-cut)
    show_cell_option = check.get_status()[2]  # Third checkbox (Show cell)
    global show_cell
    show_cell = show_cell_option
    update_plot(z_slider.val, show_above, show_below)
check.on_clicked(update_options)

# Save button
save_ax = plt.axes([0.025, 0.01, 0.15, 0.03])
save_button = plt.Button(save_ax, 'Save Figure')

def save_fig(event):
    plt.savefig('structure_visualization.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'structure_visualization.png'")
save_button.on_clicked(save_fig)

# Add a reset view button
reset_ax = plt.axes([0.025, 0.20, 0.15, 0.03])
reset_button = plt.Button(reset_ax, 'Reset View')

def reset_view(event):
    global view_limits
    view_limits = None  # Clear stored limits
    # Trigger a redraw with default limits
    update_plot(z_slider.val, check.get_status()[0], check.get_status()[1])
reset_button.on_clicked(reset_view)

# Add a standard view button
std_view_ax = plt.axes([0.025, 0.24, 0.15, 0.03])
std_view_button = plt.Button(std_view_ax, 'Standard View')

def set_standard_view(event):
    global view_limits
    view_limits = None  # Clear stored limits
    # Set to standard orientation (x-axis into screen, y-axis to left)
    elev_slider.set_val(0)
    azim_slider.set_val(90)
    # Trigger a redraw
    update_plot(z_slider.val, check.get_status()[0], check.get_status()[1])
std_view_button.on_clicked(set_standard_view)

# Initial plot
update_plot(z_cut_guess)

plt.show()
