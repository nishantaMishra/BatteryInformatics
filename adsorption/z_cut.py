#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
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

# Terminology settings - can be overridden by calling program
action_verb = "exclude"  # "exclude", "remove", "delete", etc.
action_past_tense = "excluded"  # "excluded", "removed", "deleted", etc.
undo_description = "exclusion"  # "exclusion", "removal", "deletion", etc.

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

# Track selected and excluded atoms
selected_atom = None
excluded_atoms = set()  # Atoms excluded from consideration
atom_scatter_objs = {}  # Dictionary to track scatter plot objects by atom index
# info_text_obj removed to avoid remove() crash after ax.clear()
scatter_to_indices = {}  # Map scatter collection id -> list of atom indices shown in it
exclusion_stack = []     # Stack to support undo

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
view_modified = False  # Track if view has been modified by navigation tools

# Function to update the plot based on z_cut value
def update_plot(z_value, show_above=True, show_below=True, preserve_view=True):
    global view_limits, initial_limits_set, atom_scatter_objs, scatter_to_indices, view_modified
    
    # Capture current view if needed before clearing
    if preserve_view and view_modified:
        current_view = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()
        )
    else:
        current_view = None

    # Do not try to remove text artists after clearing; just clear and redraw
    ax.clear()
    atom_scatter_objs = {}
    scatter_to_indices = {}

    # Calculate default view if needed
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0
    mid_x = (coords[:,0].max() + coords[:,0].min()) / 2.0
    mid_y = (coords[:,1].max() + coords[:,1].min()) / 2.0
    mid_z = (coords[:,2].max() + coords[:,2].min()) / 2.0
    
    # Set view limits - priority:
    # 1. Use current_view if we just captured it
    # 2. Use stored view_limits if available
    # 3. Use default calculated limits
    if current_view is not None:
        ax.set_xlim(current_view[0])
        ax.set_ylim(current_view[1])
        ax.set_zlim(current_view[2])
        view_limits = current_view  # Update stored limits
    elif view_limits and preserve_view:
        ax.set_xlim(view_limits[0])
        ax.set_ylim(view_limits[1])
        ax.set_zlim(view_limits[2])
    else:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        if not initial_limits_set:
            initial_limits_set = True
            view_limits = (
                ax.get_xlim(),
                ax.get_ylim(),
                ax.get_zlim()
            )
    
    ax.set_box_aspect([1,1,1])
    
    # Counter for visible atoms
    visible_atoms = 0
    visible_by_element = {}
    
    # Track if we have any visible elements for the legend
    has_visible_elements = False
    
    # Filter atoms based on z-cut, visibility options, and exclusion
    for el in unique_elems:
        idx = [i for i, e in enumerate(elems) if e == el]
        
        # Filter by z position and excluded atoms
        if show_above and show_below:
            idx_to_show = [i for i in idx if i not in excluded_atoms]
        elif show_above:
            idx_to_show = [i for i in idx if coords[i, 2] >= z_value and i not in excluded_atoms]
        elif show_below:
            idx_to_show = [i for i in idx if coords[i, 2] < z_value and i not in excluded_atoms]
        else:
            idx_to_show = []
        
        visible_atoms += len(idx_to_show)
        visible_by_element[el] = len(idx_to_show)
        
        if idx_to_show:
            has_visible_elements = True
            size = get_marker_sizes(el, len(idx_to_show))
            scatter = ax.scatter(
                coords[idx_to_show, 0], coords[idx_to_show, 1], coords[idx_to_show, 2],
                label=el, s=size, alpha=0.8, color=colour_map[el], picker=True
            )
            # Store the scatter object and accurate mapping from the collection points to atom indices
            scatter_to_indices[id(scatter)] = list(idx_to_show)
            for i in idx_to_show:
                atom_scatter_objs[i] = scatter
    
    # Draw excluded atoms (faded)
    for i in excluded_atoms:
        if ((show_above and coords[i, 2] >= z_value) or 
            (show_below and coords[i, 2] < z_value)):
            el = elems[i]
            size = get_marker_sizes(el, 1)
            ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                       s=size, alpha=0.25, color='gray', edgecolors=colour_map[el], linewidth=1)
    
    # Highlight currently selected atom (if not excluded and currently visible)
    if selected_atom is not None and selected_atom not in excluded_atoms:
        sel_ok = ((show_above and coords[selected_atom, 2] >= z_value) or
                  (show_below and coords[selected_atom, 2] < z_value) or
                  (show_above and show_below))
        if sel_ok:
            el = elems[selected_atom]
            x, y, zv = coords[selected_atom]
            ax.scatter([x], [y], [zv], s=get_marker_sizes(el, 1) * 1.5,
                       facecolors='none', edgecolors='k', linewidths=1.5, zorder=10)

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
    
    # Completely hide x and y axes to reduce distraction
    # Hide axis lines
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Remove tick marks
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Set pane edge colors to be invisible or very faint
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('w')
    # Hide grid
    ax.grid(False)
    
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
    atom_text = f"Visible: {visible_atoms}/{num_atoms} atoms (Excluded: {len(excluded_atoms)})"
    element_counts = ", ".join([f"{el}: {visible_by_element.get(el, 0)}" for el in unique_elems])
    
    # Create a text box with counts
    ax.text2D(0.02, 0.98, atom_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add per-element counts below
    ax.text2D(0.02, 0.93, element_counts, transform=ax.transAxes, 
              fontsize=9, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add selected atom info (persistent across redraws)
    if selected_atom is not None:
        el = elems[selected_atom]
        x, y, zv = coords[selected_atom]
        ax.text2D(0.5, 0.05, f"Atom #{selected_atom}: {el} at ({x:.3f}, {y:.3f}, {zv:.3f})",
                  transform=ax.transAxes, fontsize=10, horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Add instructions for atom selection - use dynamic terminology
    instructions = f"Click: Select atom | Backspace: {action_verb.title()} selected | Ctrl+Z: Undo last {undo_description}"
    ax.text2D(0.5, 0.02, instructions, transform=ax.transAxes, 
              fontsize=9, horizontalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    fig.canvas.draw_idle()

# Add functions to capture all view change events
def on_view_change(event):
    global view_limits, view_modified
    if hasattr(event, 'inaxes') and event.inaxes == ax:
        view_modified = True
        view_limits = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()
        )

def on_scroll(event):
    global view_modified
    if event.inaxes == ax:
        view_modified = True

def on_key_nav(event):
    global view_modified
    # Check for navigation keys
    if event.key in ['up', 'down', 'left', 'right', '+', '-']:
        view_modified = True

# Function to handle atom selection
def on_pick(event):
    global selected_atom
    scatter = event.artist
    inds = getattr(event, 'ind', None)
    # inds is a numpy array; avoid ambiguous truth-value checks
    if inds is None or (hasattr(inds, '__len__') and len(inds) == 0):
        return
    # Map picked point in this scatter collection back to the actual atom index
    idx_list = scatter_to_indices.get(id(scatter))
    if not idx_list:
        return
    picked = int(np.atleast_1d(inds)[0])
    if 0 <= picked < len(idx_list):
        selected_atom = idx_list[picked]
        # Preserve view when updating after picking
        update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=True)

def on_key(event):
    global selected_atom, excluded_atoms, exclusion_stack
    # Exclude currently selected atom
    if event.key == 'backspace' and selected_atom is not None:
        if selected_atom not in excluded_atoms:
            excluded_atoms.add(selected_atom)
            exclusion_stack.append(selected_atom)
            print(f"Atom #{selected_atom} ({elems[selected_atom]}) {action_past_tense} from consideration")
        update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=True)

    # Undo last exclusion (support both 'ctrl+z' and 'z' as a fallback)
    elif event.key in ('ctrl+z', 'z', 'Z'):
        if exclusion_stack:
            last_excluded = exclusion_stack.pop()
            excluded_atoms.discard(last_excluded)
            print(f"Restored atom #{last_excluded}")
            update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=True)
    # Check for navigation keys
    elif event.key in ['up', 'down', 'left', 'right', '+', '-']:
        on_key_nav(event)

# Connect all event handlers
fig.canvas.mpl_connect('button_release_event', on_view_change)
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('key_press_event', on_key)

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
    update_plot(z_slider.val, show_above, show_below, preserve_view=True)
z_slider.on_changed(update_z)

# Update function for view angle sliders
def update_view(val):
    global view_limits, view_modified
    # When changing view angles via sliders, reset the view_modified flag
    # since we're explicitly setting the view
    view_modified = False
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
    update_plot(z_slider.val, show_above, show_below, preserve_view=True)
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
    global view_limits, view_modified
    view_limits = None  # Clear stored limits
    view_modified = False  # Reset the modification flag
    # Trigger a redraw with default limits
    update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=False)
reset_button.on_clicked(reset_view)

# Add a standard view button
std_view_ax = plt.axes([0.025, 0.24, 0.15, 0.03])
std_view_button = plt.Button(std_view_ax, 'Standard View')

def set_standard_view(event):
    global view_limits, view_modified
    view_limits = None  # Clear stored limits
    view_modified = False  # Reset the modification flag
    # Set to standard orientation (x-axis into screen, y-axis to left)
    elev_slider.set_val(0)
    azim_slider.set_val(90)
    # Trigger a redraw
    update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=False)
std_view_button.on_clicked(set_standard_view)

# Add a button to clear all exclusions
clear_excl_ax = plt.axes([0.025, 0.28, 0.15, 0.03])
clear_excl_button = plt.Button(clear_excl_ax, 'Clear Exclusions')

def clear_exclusions(event):
    global excluded_atoms, exclusion_stack
    excluded_atoms = set()
    exclusion_stack = []
    print(f"All atom {undo_description}s cleared")
    update_plot(z_slider.val, check.get_status()[0], check.get_status()[1], preserve_view=True)
clear_excl_button.on_clicked(clear_exclusions)

# Function to get included atoms (for external use)
def get_included_atoms():
    """Return list of atom indices that are not excluded"""
    return [i for i in range(len(coords)) if i not in excluded_atoms]

# Initial plot
update_plot(z_cut_guess)

plt.show()
plt.show()