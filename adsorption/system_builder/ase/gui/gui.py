# fmt: off

import pickle
import subprocess
import sys
from functools import partial
from time import time

import numpy as np

import ase.gui.ui as ui
from ase import Atoms, __version__
from ase.gui.defaults import read_defaults
from ase.gui.i18n import _
from ase.gui.images import Images
from ase.gui.nanoparticle import SetupNanoparticle
from ase.gui.nanotube import SetupNanotube
from ase.gui.observer import Observers
from ase.gui.save import save_dialog
from ase.gui.settings import Settings
from ase.gui.status import Status
from ase.gui.surfaceslab import SetupSurfaceSlab
from ase.gui.view import View


class GUIObservers:
    def __init__(self):
        self.new_atoms = Observers()
        self.set_atoms = Observers()
        self.change_atoms = Observers()


class GUI(View):
    ARROWKEY_SCAN = 0
    ARROWKEY_MOVE = 1
    ARROWKEY_ROTATE = 2

    def __init__(self, images=None,
                 rotations='',
                 show_bonds=False, expr=None):

        if not isinstance(images, Images):
            images = Images(images)

        self.images = images

        # Ordinary observers seem unused now, delete?
        self.observers = []
        self.obs = GUIObservers()

        self.config = read_defaults()
        if show_bonds:
            self.config['show_bonds'] = True

        menu = self.get_menu_data()

        self.window = ui.ASEGUIWindow(close=self.exit, menu=menu,
                                      config=self.config, scroll=self.scroll,
                                      scroll_event=self.scroll_event,
                                      press=self.press, move=self.move,
                                      release=self.release,
                                      resize=self.resize,
                                      open_callback=self.open)

        super().__init__(rotations)
        self.status = Status(self)

        # Undo/Redo stacks and flags
        self.undo_stack = []
        self.redo_stack = []
        self._is_restoring_state = False
        self._pan_undo_recorded = False
        self._max_undo = 200

        self.subprocesses = []  # list of external processes
        self.movie_window = None
        self.simulation = {}  # Used by modules on Calculate menu.
        self.module_state = {}  # Used by modules to store their state.

        self.arrowkey_mode = self.ARROWKEY_SCAN
        self.move_atoms_mask = None

        self.set_frame(len(self.images) - 1, focus=True)

        # Used to move the structure with the mouse
        self.prev_pos = None
        self.last_scroll_time = time()
        self.orig_scale = self.scale

        if len(self.images) > 1:
            self.movie()

        if expr is None:
            expr = self.config['gui_graphs_string']

        if expr is not None and expr != '' and len(self.images) > 1:
            self.plot_graphs(expr=expr, ignore_if_nan=True)

        # Tabs initialization
        self.tabs = {}  # Dictionary to store tabs and their associated Images
        self.current_tab = None  # Track the currently active tab

        # Add a tab control to the GUI
        self.tab_control = ui.TabControl(self.window.win, self.switch_tab)
        self.tab_control.pack(side='top', fill='x')

        # Do not create any initial tab
        self.images = None  # No images loaded initially
        self.current_tab = None

        self.tab_view_settings = {}  # Store view settings for each tab
        self.tab_selection_state = {}  # Store selection state for each tab

        # Pan mode state: when True, canvas cursor becomes a 4-spoked
        # "move" cursor (Tk 'fleur') and user can drag to pan.
        self.pan_mode = False
        self.last_pan_pos = None

    @property
    def moving(self):
        return self.arrowkey_mode != self.ARROWKEY_SCAN

    def run(self):
        self.window.run()

    def toggle_move_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_MOVE)

    def toggle_rotate_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_ROTATE)

    def toggle_arrowkey_mode(self, mode):
        # If not currently in given mode, activate it.
        # Else, deactivate it (go back to SCAN mode)
        assert mode != self.ARROWKEY_SCAN

        if self.arrowkey_mode == mode:
            self.arrowkey_mode = self.ARROWKEY_SCAN
            self.move_atoms_mask = None
        else:
            self.arrowkey_mode = mode
            self.move_atoms_mask = self.images.selected.copy()

        self.draw()

    def step(self, key):
        d = {'Home': -10000000,
             'Page-Up': -1,
             'Page-Down': 1,
             'End': 10000000}[key]
        i = max(0, min(len(self.images) - 1, self.frame + d))
        self.set_frame(i)
        if self.movie_window is not None:
            self.movie_window.frame_number.value = i

    def copy_image(self, key=None):
        self.images._images.append(self.atoms.copy())
        self.images.filenames.append(None)

        if self.movie_window is not None:
            self.movie_window.frame_number.scale.configure(to=len(self.images))
        self.step('End')

    def _do_zoom(self, x):
        """Utility method for zooming"""
        self.scale *= x
        self.draw()

    def zoom(self, key):
        """Zoom in/out on keypress or clicking menu item"""
        # Record view change
        self._push_undo('view_zoom')
        x = {'+': 1.2, '-': 1 / 1.2}[key]
        self._do_zoom(x)

    def scroll_event(self, event):
        """Zoom in/out when using mouse wheel"""
        SHIFT = event.modifier == 'shift'
        x = 1.0
        if event.button == 4 or event.delta > 0:
            x = 1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01
        elif event.button == 5 or event.delta < 0:
            x = 1.0 / (1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01)
        self._do_zoom(x)

    def settings(self):
        return Settings(self)

    def scroll(self, event):
        shift = 0x1
        ctrl = 0x4
        alt_l = 0x8  # Also Mac Command Key
        mac_option_key = 0x10

        use_small_step = bool(event.state & shift)
        rotate_into_plane = bool(event.state & (ctrl | alt_l | mac_option_key))

        # When we're moving atoms, prefer in-plane movement: do not
        # interpret Up/Down as rotate-into-plane. Only allow
        # rotate_into_plane when in ROTATE arrowkey mode.
        if self.arrowkey_mode != self.ARROWKEY_ROTATE:
            rotate_into_plane = False

        # Normalize key names: handle common variants like 'KP_Up',
        # 'KeyPad-Up' etc. Prefer the already-normalized `event.key`
        # (set by `ui.bind`), but fall back to `event.keysym` when
        # necessary. Strip known keypad prefixes and lowercase.
        key = getattr(event, 'key', None)
        if not key:
            # Some event objects may provide `keysym` instead of `key`.
            key = getattr(event, 'keysym', '')
        key = str(key)
        # Accept keys like 'ArrowLeft' / 'ArrowRight' (common on some platforms)
        if key.lower().startswith('arrow'):
            key = key[len('arrow'):]
        # Strip common keypad prefixes
        for prefix in ('kp_', 'kp', 'keypad_', 'keypad-'):
            if key.lower().startswith(prefix):
                key = key[len(prefix):]
                break
        key = key.strip().lower()

        dxdydz = {'up': (0, 1 - rotate_into_plane, rotate_into_plane),
                  'down': (0, -1 + rotate_into_plane, -rotate_into_plane),
                  'right': (1 - rotate_into_plane, 0, rotate_into_plane),
                  'left': (-1 + rotate_into_plane, 0, -rotate_into_plane)}.get(key, None)

        # Get scroll direction using shift + right mouse button
        # event.type == '6' is mouse motion, see:
        # http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-types.html
        if event.type == '6':
            cur_pos = np.array([event.x, -event.y])
            # Continue scroll if button has not been released
            if self.prev_pos is None or time() - self.last_scroll_time > .5:
                self.prev_pos = cur_pos
                self.last_scroll_time = time()
            else:
                dxdy = cur_pos - self.prev_pos
                dxdydz = np.append(dxdy, [0])
                self.prev_pos = cur_pos
                self.last_scroll_time = time()

        if dxdydz is None:
            return

        vec = 0.1 * np.dot(self.axes, dxdydz)
        if use_small_step:
            vec *= 0.1

        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            # Record atomic positions before modifying
            self._push_undo('atoms_move')
            if self.move_atoms_mask is None:
                self.move_atoms_mask = self.images.selected.copy()
            self.atoms.positions[self.move_atoms_mask[:len(self.atoms)]] += vec
            self.set_frame()
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            # Record atomic positions before modifying
            self._push_undo('atoms_rotate')
            if self.move_atoms_mask is None:
                self.move_atoms_mask = self.images.selected.copy()
            mask = self.move_atoms_mask[:len(self.atoms)]
            if np.any(mask):
                center = self.atoms.positions[mask].mean(axis=0)
                tmp_atoms = self.atoms[mask]
                tmp_atoms.positions -= center
                tmp_atoms.rotate(50 * np.linalg.norm(vec), vec)
                self.atoms.positions[mask] = tmp_atoms.positions + center
            self.set_frame()
        else:
            # View panning via scroll
            self._push_undo('view_pan')
            scale = self.orig_scale / (3 * self.scale)
            self.center -= vec * scale
            self.draw()

    def delete_selected_atoms(self, widget=None, data=None):
        import ase.gui.ui as ui
        nselected = sum(self.images.selected)
        if nselected and ui.ask_question(_('Delete atoms'),
                                         _('Delete selected atoms?')):
            self.really_delete_selected_atoms()

    def really_delete_selected_atoms(self):
        mask = self.images.selected[:len(self.atoms)]
        del self.atoms[mask]

        # Will remove selection in other images, too
        self.images.selected[:] = False
        self.set_frame()
        self.draw()

    def constraints_window(self):
        from ase.gui.constraints import Constraints
        return Constraints(self)

    def set_selected_atoms(self, selected):
        newmask = np.zeros(len(self.images.selected), bool)
        newmask[selected] = True

        if np.array_equal(newmask, self.images.selected):
            return

        # (By creating newmask, we can avoid resetting the selection in
        # case the selected indices are invalid)
        self.images.selected[:] = newmask
        self.draw()

    def select_all(self, key=None):
        self.images.selected[:] = True
        self.draw()

    def invert_selection(self, key=None):
        self.images.selected[:] = ~self.images.selected
        self.draw()

    def select_constrained_atoms(self, key=None):
        self.images.selected[:] = ~self.images.get_dynamic(self.atoms)
        self.draw()

    def select_immobile_atoms(self, key=None):
        if len(self.images) > 1:
            R0 = self.images[0].positions
            for atoms in self.images[1:]:
                R = atoms.positions
                self.images.selected[:] = ~(np.abs(R - R0) > 1.0e-10).any(1)
        self.draw()

    def movie(self):
        from ase.gui.movie import Movie
        self.movie_window = Movie(self)

    def plot_graphs(self, key=None, expr=None, ignore_if_nan=False):
        from ase.gui.graphs import Graphs
        g = Graphs(self)
        if expr is not None:
            g.plot(expr=expr, ignore_if_nan=ignore_if_nan)

    def pipe(self, task, data):
        process = subprocess.Popen([sys.executable, '-m', 'ase.gui.pipe'],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        pickle.dump((task, data), process.stdin)
        process.stdin.close()
        # Either process writes a line, or it crashes and line becomes ''
        line = process.stdout.readline().decode('utf8').strip()

        if line != 'GUI:OK':
            if line == '':  # Subprocess probably crashed
                line = _('Failure in subprocess')
            self.bad_plot(line)
        else:
            self.subprocesses.append(process)
        return process

    def bad_plot(self, err, msg=''):
        ui.error(_('Plotting failed'), '\n'.join([str(err), msg]).strip())

    def neb(self):
        from ase.utils.forcecurve import fit_images
        try:
            forcefit = fit_images(self.images)
        except Exception as err:
            self.bad_plot(err, _('Images must have energies and forces, '
                                 'and atoms must not be stationary.'))
        else:
            self.pipe('neb', forcefit)

    def bulk_modulus(self):
        try:
            v = [abs(np.linalg.det(atoms.cell)) for atoms in self.images]
            e = [self.images.get_energy(a) for a in self.images]
            from ase.eos import EquationOfState
            eos = EquationOfState(v, e)
            plotdata = eos.getplotdata()
        except Exception as err:
            self.bad_plot(err, _('Images must have energies '
                                 'and varying cell.'))
        else:
            self.pipe('eos', plotdata)

    def reciprocal(self):
        if self.atoms.cell.rank != 3:
            self.bad_plot(_('Requires 3D cell.'))
            return None

        cell = self.atoms.cell.uncomplete(self.atoms.pbc)
        bandpath = cell.bandpath(npoints=0)
        return self.pipe('reciprocal', bandpath)

    def add_tab(self, filename, images):
        """Add a new tab with the filename as the title and associated Images."""
        # Save current tab view/selection before switching, so it doesn't get
        # reset when a new tab is added.
        if self.current_tab is not None and self.current_tab in self.tabs:
            try:
                self.tab_view_settings[self.current_tab] = {
                    'scale': self.scale,
                    'center': self.center.copy(),
                    'axes': self.axes.copy()
                }
            except Exception:
                # Be lenient if any attribute is missing
                self.tab_view_settings[self.current_tab] = {
                    'scale': getattr(self, 'scale', 1.0),
                    'center': getattr(self, 'center', np.zeros(3)).copy(),
                    'axes': getattr(self, 'axes', np.eye(3)).copy()
                }
            if hasattr(self, 'images') and hasattr(self.images, 'selected'):
                self.tab_selection_state[self.current_tab] = self.images.selected.copy()

        if not isinstance(images, Images):
            images = Images(images)

        # Ensure this Images object has its own independent selection array
        if hasattr(images, 'selected'):
            images.selected = np.zeros(len(images.selected), bool)

        tab_title = filename.split('/')[-1]  # Extract the file name from the path
        # Pass full filepath so the TabControl can show it on hover
        tab_id = self.tab_control.add_tab(tab_title, filepath=filename)
        self.tabs[tab_id] = images

        # Make the new tab current and associate a view for it (inherit current
        # GUI view so the new tab starts with the same view but remains independent).
        self.current_tab = tab_id
        self.images = images

        try:
            self.tab_view_settings[tab_id] = {
                'scale': self.scale,
                'center': self.center.copy(),
                'axes': self.axes.copy()
            }
        except Exception:
            self.tab_view_settings[tab_id] = {
                'scale': getattr(self, 'scale', 1.0),
                'center': getattr(self, 'center', np.zeros(3)).copy(),
                'axes': getattr(self, 'axes', np.eye(3)).copy()
            }

        # Initialize selection state for this tab
        self.tab_selection_state[tab_id] = images.selected.copy()

        # Now set frame and redraw (the switch_tab logic restores settings when
        # switching between already added tabs; for a freshly added tab we keep
        # the inherited view we stored above).
        self.set_frame(len(self.images) - 1, focus=True)
        self.draw()
        
        # Ensure the main window has focus after adding a new tab
        # This is critical for arrow key functionality
        try:
            self.window.win.focus_force()
        except Exception:
            pass

    def switch_tab(self, tab_id):
        """Switch to the specified tab."""
        if self.current_tab is not None and self.current_tab in self.tabs:
            # Save the current view settings for the current tab
            self.tab_view_settings[self.current_tab] = {
                'scale': self.scale,
                'center': self.center.copy(),
                'axes': self.axes.copy()
            }
            # Save the current selection state by copying the actual selection array
            if hasattr(self.images, 'selected'):
                self.tab_selection_state[self.current_tab] = self.images.selected.copy()

        if tab_id in self.tabs:
            self.current_tab = tab_id
            self.images = self.tabs[tab_id]
            
            # Restore the saved selection state for the new tab
            if tab_id in self.tab_selection_state:
                # Ensure we have the right size array
                saved_selection = self.tab_selection_state[tab_id]
                if len(self.images.selected) != len(saved_selection):
                    # Resize if needed
                    new_selection = np.zeros(len(self.images.selected), bool)
                    min_len = min(len(new_selection), len(saved_selection))
                    new_selection[:min_len] = saved_selection[:min_len]
                    self.images.selected[:] = new_selection
                else:
                    self.images.selected[:] = saved_selection
            else:
                # Default: no selection
                self.images.selected[:] = False
            
            # Also reset the move_atoms_mask if in move/rotate mode
            if self.arrowkey_mode != self.ARROWKEY_SCAN:
                self.move_atoms_mask = self.images.selected.copy()
            
            self.set_frame(len(self.images) - 1, focus=True)

            # Restore the saved view settings for the new tab, if available
            if tab_id in self.tab_view_settings:
                settings = self.tab_view_settings[tab_id]
                self.scale = settings['scale']
                self.center = settings['center']
                self.axes = settings['axes']
            else:
                # Default view settings if no saved settings exist
                self.scale = 1.0
                self.center = np.zeros(3)
                self.axes = np.eye(3)

            self.draw()
            
            # Ensure the main window has focus after switching tabs
            # This is critical for arrow key functionality
            try:
                self.window.win.focus_force()
            except Exception:
                pass

    def open(self, button=None, filename=None):
        # Prefer the native OS file dialog when available for a familiar
        # experience. Fall back to the bundled ASEFileChooser if needed.
        # Try desktop-native pickers first (Linux: zenity/kdialog), then
        # tkinter's askopenfilename, then ASEFileChooser as a final fallback.
        filename = filename or ''
        if not filename:
            try:
                import subprocess
                import shutil

                # zenity is common on GNOME, kdialog on KDE
                if shutil.which('zenity'):
                    proc = subprocess.run(['zenity', '--file-selection'],
                                           capture_output=True, text=True)
                    if proc.returncode == 0:
                        filename = proc.stdout.strip()
                elif shutil.which('kdialog'):
                    proc = subprocess.run(['kdialog', '--getopenfilename'],
                                           capture_output=True, text=True)
                    if proc.returncode == 0:
                        filename = proc.stdout.strip()
            except Exception:
                filename = filename or ''

        if not filename:
            try:
                from tkinter.filedialog import askopenfilename
                filename = filename or askopenfilename(title=_('Open ...'))
                format = None
            except Exception:
                chooser = ui.ASEFileChooser(self.window.win)
                filename = filename or chooser.go()
                format = chooser.format
        else:
            format = None

        if filename:
            try:
                new_images = Images()
                new_images.read([filename], slice(None), format)
                self.add_tab(filename, new_images)  # Use filename as tab title
            except Exception as err:
                ui.show_io_error(filename, err)
                return  # Hmm.  Is self.images in a consistent state?
            self.set_frame(len(self.images) - 1, focus=True)

    def modify_atoms(self, key=None):
        from ase.gui.modify import ModifyAtoms
        return ModifyAtoms(self)

    def add_atoms(self, key=None):
        from ase.gui.add import AddAtoms
        return AddAtoms(self)

    def cell_editor(self, key=None):
        from ase.gui.celleditor import CellEditor
        return CellEditor(self)

    def atoms_editor(self, key=None):
        from ase.gui.atomseditor import AtomsEditor
        return AtomsEditor(self)

    def quick_info_window(self, key=None):
        from ase.gui.quickinfo import info
        info_win = ui.Window(_('Quick Info'))
        info_win.add(info(self))

        # Update quickinfo window when we change frame
        def update(window):
            exists = window.exists
            if exists:
                # Only update if we exist
                window.things[0].text = info(self)
            return exists
        self.attach(update, info_win)
        return info_win

    def surface_window(self):
        return SetupSurfaceSlab(self)

    def nanoparticle_window(self):
        return SetupNanoparticle(self)

    def nanotube_window(self):
        return SetupNanotube(self)

    def new_atoms(self, atoms):
        "Set a new atoms object."
        rpt = getattr(self.images, 'repeat', None)
        self.images.repeat_images(np.ones(3, int))
        self.images.initialize([atoms])
        self.frame = 0  # Prevent crashes
        self.images.repeat_images(rpt)
        self.set_frame(frame=0, focus=True)
        self.obs.new_atoms.notify()

    def exit(self, event=None):
        for process in self.subprocesses:
            process.terminate()
        self.window.close()

    def new(self, key=None):
        subprocess.Popen([sys.executable, '-m', 'ase', 'gui'])

    def save(self, key=None):
        return save_dialog(self)

    def external_viewer(self, name):
        from ase.visualize import view
        return view(list(self.images), viewer=name)

    def selected_atoms(self):
        selection_mask = self.images.selected[:len(self.atoms)]
        return self.atoms[selection_mask]

    # Clipboard API (restored to fix Edit menu callbacks)
    @property
    def clipboard(self):
        from ase.gui.clipboard import AtomsClipboard
        return AtomsClipboard(self.window.win)

    def cut_atoms_to_clipboard(self, event=None):
        self.copy_atoms_to_clipboard(event)
        self.really_delete_selected_atoms()

    def copy_atoms_to_clipboard(self, event=None):
        atoms = self.selected_atoms()
        self.clipboard.set_atoms(atoms)

    def paste_atoms_from_clipboard(self, event=None):
        try:
            atoms = self.clipboard.get_atoms()
        except Exception as err:
            ui.error(
                'Cannot paste atoms',
                'Pasting currently works only with the ASE JSON format.\n\n'
                f'Original error:\n\n{err}')
            return

        if self.atoms == Atoms():
            self.atoms.cell = atoms.cell
            self.atoms.pbc = atoms.pbc
        self.paste_atoms_onto_existing(atoms)

    def paste_atoms_onto_existing(self, atoms):
        selection = self.selected_atoms()
        if len(selection):
            paste_center = selection.positions.sum(axis=0) / len(selection)
            atoms = atoms.copy()
            atoms.cell = (1, 1, 1)
            atoms.center(about=paste_center)

        self.add_atoms_and_select(atoms)
        self.move_atoms_mask = self.images.selected.copy()
        self.arrowkey_mode = self.ARROWKEY_MOVE
        self.draw()

    def add_atoms_and_select(self, new_atoms):
        atoms = self.atoms
        atoms += new_atoms

        if len(atoms) > self.images.maxnatoms:
            self.images.initialize(list(self.images),
                                   self.images.filenames)

        selected = self.images.selected
        selected[:] = False
        selected[len(atoms) - len(new_atoms):len(atoms)] = True

        self.set_frame()
        self.draw()

    def align_view_along_axis(self, axis_type='a', reciprocal=False):
        """Align view along crystallographic axis.
        
        Args:
            axis_type: 'a', 'b', or 'c' for the crystallographic axis
            reciprocal: If True, use reciprocal lattice vector (a*, b*, c*)
        """
        # Record view change
        self._push_undo('view_align')
        if self.atoms.cell.rank != 3:
            ui.error(_('Error'), _('Requires 3D cell for axis alignment.'))
            return
        
        axis_map = {'a': 0, 'b': 1, 'c': 2}
        axis_idx = axis_map.get(axis_type.lower(), 0)
        
        if reciprocal:
            # Get reciprocal lattice vectors
            cell = self.atoms.cell.reciprocal()
            axis_vector = cell[axis_idx]
        else:
            # Get direct lattice vectors
            axis_vector = self.atoms.cell[axis_idx]
        
        # Normalize the axis vector
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        
        # We want to look along this axis, so set it as the z-direction
        # Create a rotation matrix that aligns axis_vector with [0, 0, 1]
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # If vectors are already aligned, no rotation needed
        if np.allclose(axis_vector, z_axis):
            self.axes = np.eye(3)
        elif np.allclose(axis_vector, -z_axis):
            self.axes = np.diag([1, -1, -1])
        else:
            # Rotation axis is perpendicular to both vectors
            rot_axis = np.cross(axis_vector, z_axis)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            
            # Rotation angle
            cos_angle = np.dot(axis_vector, z_axis)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Rodrigues rotation formula
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                         [rot_axis[2], 0, -rot_axis[0]],
                         [-rot_axis[1], rot_axis[0], 0]])
            
            R = (np.eye(3) + np.sin(angle) * K + 
                 (1 - np.cos(angle)) * np.dot(K, K))
            
            self.axes = R.T
        
        self.draw()

    def align_along_a(self, key=None):
        """Align view along a-axis (direct lattice)."""
        self.align_view_along_axis('a', reciprocal=False)

    def align_along_b(self, key=None):
        """Align view along b-axis (direct lattice)."""
        self.align_view_along_axis('b', reciprocal=False)

    def align_along_c(self, key=None):
        """Align view along c-axis (direct lattice)."""
        self.align_view_along_axis('c', reciprocal=False)

    def align_along_a_star(self, key=None):
        """Align view along a*-axis (reciprocal lattice)."""
        self.align_view_along_axis('a', reciprocal=True)

    def align_along_b_star(self, key=None):
        """Align view along b*-axis (reciprocal lattice)."""
        self.align_view_along_axis('b', reciprocal=True)

    def align_along_c_star(self, key=None):
        """Align view along c*-axis (reciprocal lattice)."""
        self.align_view_along_axis('c', reciprocal=True)

    def wrap_atoms(self, key=None):
        """Wrap atoms around the unit cell."""
        # Record positions before wrapping
        self._push_undo('atoms_wrap')
        for atoms in self.images:
            atoms.wrap()
        self.set_frame()

    def get_menu_data(self):
        M = ui.MenuItem
        return [
            (_('_File'),
             [M(_('_Open'), self.open, 'Ctrl+O'),
              M(_('_New'), self.new, 'Ctrl+N'),
              M(_('_Save'), self.save, 'Ctrl+S'),
              M('---'),
              M(_('_Quit'), self.exit, 'Ctrl+Q')]),

            (_('_Edit'),
             [
              # Undo/Redo at the top
              M(_('_Undo'), self.undo, 'Ctrl+Z'),
              M(_('_Redo'), self.redo, 'Ctrl+Y'),
              M('---'),
              M(_('Select _all'), self.select_all),
              M(_('_Invert selection'), self.invert_selection),
              M(_('Select _constrained atoms'), self.select_constrained_atoms),
              M(_('Select _immobile atoms'), self.select_immobile_atoms),
              # M('---'),
              M(_('_Cut'), self.cut_atoms_to_clipboard, 'Ctrl+X'),
              M(_('_Copy'), self.copy_atoms_to_clipboard, 'Ctrl+C'),
              M(_('_Paste'), self.paste_atoms_from_clipboard, 'Ctrl+V'),
              M('---'),
              M(_('Hide selected atoms'), self.hide_selected),
              M(_('Show selected atoms'), self.show_selected),
              M('---'),
              # Remove Ctrl+Y from Modify to free it for Redo
              M(_('_Modify'), self.modify_atoms),
              M(_('_Add atoms'), self.add_atoms, 'Ctrl+A'),
              M(_('_Delete selected atoms'), self.delete_selected_atoms,
                'Backspace'),
              M(_('Edit _cell …'), self.cell_editor, 'Ctrl+E'),
              M(_('Edit _atoms …'), self.atoms_editor, 'A'),
              M('---'),
              M(_('_First image'), self.step, 'Home'),
              M(_('_Previous image'), self.step, 'Page-Up'),
              M(_('_Next image'), self.step, 'Page-Down'),
              M(_('_Last image'), self.step, 'End'),
              M(_('Append image copy'), self.copy_image)]),

            (_('_View'),
             [M(_('Show _unit cell'), self.toggle_show_unit_cell, 'Ctrl+U',
                value=self.config['show_unit_cell']),
              M(_('Show _axes'), self.toggle_show_axes,
                value=self.config['show_axes']),
              M(_('Show _bonds'), self.toggle_show_bonds, 'Ctrl+B',
                value=self.config['show_bonds']),
              M(_('Show _velocities'), self.toggle_show_velocities, 'Ctrl+G',
                value=False),
              M(_('Show _forces'), self.toggle_show_forces, 'Ctrl+F',
                value=False),
              M(_('Show _magmoms'), self.toggle_show_magmoms,
                value=False),
              M(_('Show _Labels'), self.show_labels,
                choices=[_('_None'),
                         _('Atom _Index'),
                         _('_Magnetic Moments'),  # XXX check if exist
                         _('_Element Symbol'),
                         _('_Initial Charges'),  # XXX check if exist
                         ]),
              M('---'),
              M(_('Quick Info ...'), self.quick_info_window, 'Ctrl+I'),
              M(_('Repeat ...'), self.repeat_window, 'R'),
              M(_('Rotate ...'), self.rotate_window),
              M(_('Colors ...'), self.colors_window, 'C'),
              # TRANSLATORS: verb
              M(_('Focus'), self.focus, 'F'),
              M(_('Zoom in'), self.zoom, '+'),
              M(_('Zoom out'), self.zoom, '-'),
              M(_('Change View'),
                submenu=[
                    M(_('Reset View'), self.reset_view, '='),
                    M('---'),
                    M(_('xy-plane'), self.set_view, 'Z'),
                    M(_('yz-plane'), self.set_view, 'X'),
                    M(_('zx-plane'), self.set_view, 'Y'),
                    M(_('yx-plane'), self.set_view, 'Shift+Z'),
                    M(_('zy-plane'), self.set_view, 'Shift+X'),
                    M(_('xz-plane'), self.set_view, 'Shift+Y'),
                    M('---'),
                    M(_('a2,a3-plane'), self.set_view, 'I'),
                    M(_('a3,a1-plane'), self.set_view, 'J'),
                    M(_('a1,a2-plane'), self.set_view, 'K'),
                    M(_('a3,a2-plane'), self.set_view, 'Shift+I'),
                    M(_('a1,a3-plane'), self.set_view, 'Shift+J'),
                    M(_('a2,a1-plane'), self.set_view, 'Shift+K'),
                    M('---'),
                    # Toggle interactive pan mode: change cursor and allow drag-to-pan
                    M(_('Pan'), self.toggle_pan_mode, 'P'),
                    M('---'),
                    M(_('Along a-axis'), self.align_along_a),
                    M(_('Along b-axis'), self.align_along_b),
                    M(_('Along c-axis'), self.align_along_c),
                    M(_('Along a*-axis'), self.align_along_a_star),
                    M(_('Along b*-axis'), self.align_along_b_star),
                    M(_('Along c*-axis'), self.align_along_c_star
                    )]),
              M(_('Settings ...'), self.settings),
              M('---'),
              M(_('VMD'), partial(self.external_viewer, 'vmd')),
              M(_('RasMol'), partial(self.external_viewer, 'rasmol')),
              M(_('xmakemol'), partial(self.external_viewer, 'xmakemol')),
              M(_('avogadro'), partial(self.external_viewer, 'avogadro'))]),

            (_('_Tools'),
             [M(_('Graphs ...'), self.plot_graphs),
              M(_('Movie ...'), self.movie),
              M(_('Constraints ...'), self.constraints_window),
              M(_('Render scene ...'), self.render_window),
              M(_('_Move selected atoms'), self.toggle_move_mode, 'Ctrl+M'),
              M(_('_Rotate selected atoms'), self.toggle_rotate_mode,
                'Ctrl+R'),
              M(_('NE_B plot'), self.neb),
              M(_('B_ulk Modulus'), self.bulk_modulus),
              M(_('Reciprocal space ...'), self.reciprocal),
              M(_('Wrap atoms'), self.wrap_atoms, 'Ctrl+W')]),

            # TRANSLATORS: Set up (i.e. build) surfaces, nanoparticles, ...
            (_('_Setup'),
             [M(_('_Surface slab'), self.surface_window, disabled=False),
              M(_('_Nanoparticle'),
                self.nanoparticle_window),
              M(_('Nano_tube'), self.nanotube_window)]),

            # (_('_Calculate'),
            # [M(_('Set _Calculator'), self.calculator_window, disabled=True),
            #  M(_('_Energy and Forces'), self.energy_window, disabled=True),
            #  M(_('Energy Minimization'), self.energy_minimize_window,
            #    disabled=True)]),

            (_('_Helped'),
             [M(_('_About'), partial(
                 ui.about, 'ASE-GUI',
                 version=__version__,
                 webpage='https://ase-lib.org/ase/gui/gui.html')),
              M(_('Webpage ...'), webpage)])]

    def attach(self, function, *args, **kwargs):
        self.observers.append((function, args, kwargs))

    def call_observers(self):
        # Use function return value to determine if we keep observer
        self.observers = [(function, args, kwargs) for (function, args, kwargs)
                          in self.observers if function(*args, **kwargs)]

    def repeat_poll(self, callback, ms, ensure_update=True):
        """Invoke callback(gui=self) every ms milliseconds.

        This is useful for polling a resource for updates to load them
        into the GUI.  The GUI display will be hence be updated after
        each call; pass ensure_update=False to circumvent this.

        Polling stops if the callback function raises StopIteration.

        Example to run a movie manually, then quit::

            from ase.collections import g2
            from ase.gui.gui import GUI

            names = iter(g2.names)

            def main(gui):
                try:
                    name = next(names)
                except StopIteration:
                    gui.window.win.quit()
                else:
                    atoms = g2[name]
                    gui.images.initialize([atoms])

            gui = GUI()
            gui.repeat_poll(main, 30)
            gui.run()"""

        def callbackwrapper():
            try:
                callback(gui=self)
            except StopIteration:
                pass
            finally:
                # Reinsert self so we get called again:
                self.window.win.after(ms, callbackwrapper)

            if ensure_update:
                self.set_frame()
                self.draw()

        self.window.win.after(ms, callbackwrapper)


    # --- Undo/Redo helpers ---
    def _snapshot_state(self):
        try:
            pos = self.atoms.positions.copy() if len(self.atoms) else None
        except Exception:
            pos = None
        try:
            sel = self.images.selected.copy()
        except Exception:
            sel = None
        return {
            'frame': int(getattr(self, 'frame', 0)),
            'positions': pos,
            'axes': getattr(self, 'axes', np.eye(3)).copy(),
            'center': getattr(self, 'center', np.zeros(3)).copy(),
            'scale': float(getattr(self, 'scale', 1.0)),
            'selection': sel
        }

    def _apply_state(self, state):
        self._is_restoring_state = True
        try:
            # Restore frame first (does not change axes/center/scale)
            target_frame = state.get('frame', self.frame)
            if 0 <= target_frame < len(self.images):
                # set_frame triggers draw; we'll redraw at end anyway
                self.set_frame(target_frame)

            # Restore positions if matching length
            pos = state.get('positions')
            if pos is not None and len(self.atoms) == len(pos):
                self.atoms.positions[:] = pos

            # Restore view parameters
            axes = state.get('axes')
            center = state.get('center')
            scale = state.get('scale')
            if axes is not None:
                self.axes = axes
            if center is not None:
                self.center = center
            if scale is not None:
                self.scale = scale

            # Restore selection if compatible
            sel = state.get('selection')
            if sel is not None and hasattr(self.images, 'selected'):
                if len(self.images.selected) == len(sel):
                    self.images.selected[:] = sel
                else:
                    # Resize safely
                    new_sel = np.zeros_like(self.images.selected, dtype=bool)
                    n = min(len(new_sel), len(sel))
                    new_sel[:n] = sel[:n]
                    self.images.selected[:] = new_sel

            # After restoring, redraw
            self.draw()
        finally:
            self._is_restoring_state = False

    def _push_undo(self, reason=''):
        if self._is_restoring_state:
            return
        try:
            self.undo_stack.append(self._snapshot_state())
            if len(self.undo_stack) > self._max_undo:
                self.undo_stack = self.undo_stack[-self._max_undo:]
            self.redo_stack.clear()
        except Exception:
            # Be robust; never block user interaction
            pass

    def undo(self, key=None):
        if not self.undo_stack:
            return
        # Push current to redo, restore last undo
        current = self._snapshot_state()
        state = self.undo_stack.pop()
        self.redo_stack.append(current)
        self._apply_state(state)

    def redo(self, key=None):
        if not self.redo_stack:
            return
        current = self._snapshot_state()
        state = self.redo_stack.pop()
        self.undo_stack.append(current)
        self._apply_state(state)
    # --- end Undo/Redo helpers ---

    def run(self):
        self.window.run()

    def toggle_move_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_MOVE)

    def toggle_rotate_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_ROTATE)

    def toggle_arrowkey_mode(self, mode):
        # If not currently in given mode, activate it.
        # Else, deactivate it (go back to SCAN mode)
        assert mode != self.ARROWKEY_SCAN

        if self.arrowkey_mode == mode:
            self.arrowkey_mode = self.ARROWKEY_SCAN
            self.move_atoms_mask = None
        else:
            self.arrowkey_mode = mode
            self.move_atoms_mask = self.images.selected.copy()

        self.draw()

    def step(self, key):
        d = {'Home': -10000000,
             'Page-Up': -1,
             'Page-Down': 1,
             'End': 10000000}[key]
        i = max(0, min(len(self.images) - 1, self.frame + d))
        self.set_frame(i)
        if self.movie_window is not None:
            self.movie_window.frame_number.value = i

    def copy_image(self, key=None):
        self.images._images.append(self.atoms.copy())
        self.images.filenames.append(None)

        if self.movie_window is not None:
            self.movie_window.frame_number.scale.configure(to=len(self.images))
        self.step('End')

    def _do_zoom(self, x):
        """Utility method for zooming"""
        self.scale *= x
        self.draw()

    def zoom(self, key):
        """Zoom in/out on keypress or clicking menu item"""
        # Record view change
        self._push_undo('view_zoom')
        x = {'+': 1.2, '-': 1 / 1.2}[key]
        self._do_zoom(x)

    def scroll_event(self, event):
        """Zoom in/out when using mouse wheel"""
        SHIFT = event.modifier == 'shift'
        x = 1.0
        if event.button == 4 or event.delta > 0:
            x = 1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01
        elif event.button == 5 or event.delta < 0:
            x = 1.0 / (1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01)
        self._do_zoom(x)

    def settings(self):
        return Settings(self)

    def scroll(self, event):
        shift = 0x1
        ctrl = 0x4
        alt_l = 0x8  # Also Mac Command Key
        mac_option_key = 0x10

        use_small_step = bool(event.state & shift)
        rotate_into_plane = bool(event.state & (ctrl | alt_l | mac_option_key))

        # When we're moving atoms, prefer in-plane movement: do not
        # interpret Up/Down as rotate-into-plane. Only allow
        # rotate_into_plane when in ROTATE arrowkey mode.
        if self.arrowkey_mode != self.ARROWKEY_ROTATE:
            rotate_into_plane = False

        # Normalize key names: handle common variants like 'KP_Up',
        # 'KeyPad-Up' etc. Prefer the already-normalized `event.key`
        # (set by `ui.bind`), but fall back to `event.keysym` when
        # necessary. Strip known keypad prefixes and lowercase.
        key = getattr(event, 'key', None)
        if not key:
            # Some event objects may provide `keysym` instead of `key`.
            key = getattr(event, 'keysym', '')
        key = str(key)
        # Accept keys like 'ArrowLeft' / 'ArrowRight' (common on some platforms)
        if key.lower().startswith('arrow'):
            key = key[len('arrow'):]
        # Strip common keypad prefixes
        for prefix in ('kp_', 'kp', 'keypad_', 'keypad-'):
            if key.lower().startswith(prefix):
                key = key[len(prefix):]
                break
        key = key.strip().lower()

        dxdydz = {'up': (0, 1 - rotate_into_plane, rotate_into_plane),
                  'down': (0, -1 + rotate_into_plane, -rotate_into_plane),
                  'right': (1 - rotate_into_plane, 0, rotate_into_plane),
                  'left': (-1 + rotate_into_plane, 0, -rotate_into_plane)}.get(key, None)

        # Get scroll direction using shift + right mouse button
        # event.type == '6' is mouse motion, see:
        # http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-types.html
        if event.type == '6':
            cur_pos = np.array([event.x, -event.y])
            # Continue scroll if button has not been released
            if self.prev_pos is None or time() - self.last_scroll_time > .5:
                self.prev_pos = cur_pos
                self.last_scroll_time = time()
            else:
                dxdy = cur_pos - self.prev_pos
                dxdydz = np.append(dxdy, [0])
                self.prev_pos = cur_pos
                self.last_scroll_time = time()

        if dxdydz is None:
            return

        vec = 0.1 * np.dot(self.axes, dxdydz)
        if use_small_step:
            vec *= 0.1

        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            # Record atomic positions before modifying
            self._push_undo('atoms_move')
            if self.move_atoms_mask is None:
                self.move_atoms_mask = self.images.selected.copy()
            self.atoms.positions[self.move_atoms_mask[:len(self.atoms)]] += vec
            self.set_frame()
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            # Record atomic positions before modifying
            self._push_undo('atoms_rotate')
            if self.move_atoms_mask is None:
                self.move_atoms_mask = self.images.selected.copy()
            mask = self.move_atoms_mask[:len(self.atoms)]
            if np.any(mask):
                center = self.atoms.positions[mask].mean(axis=0)
                tmp_atoms = self.atoms[mask]
                tmp_atoms.positions -= center
                tmp_atoms.rotate(50 * np.linalg.norm(vec), vec)
                self.atoms.positions[mask] = tmp_atoms.positions + center
            self.set_frame()
        else:
            # View panning via scroll
            self._push_undo('view_pan')
            scale = self.orig_scale / (3 * self.scale)
            self.center -= vec * scale
            self.draw()

    def delete_selected_atoms(self, widget=None, data=None):
        import ase.gui.ui as ui
        nselected = sum(self.images.selected)
        if nselected and ui.ask_question(_('Delete atoms'),
                                         _('Delete selected atoms?')):
            self.really_delete_selected_atoms()

    def really_delete_selected_atoms(self):
        mask = self.images.selected[:len(self.atoms)]
        del self.atoms[mask]

        # Will remove selection in other images, too
        self.images.selected[:] = False
        self.set_frame()
        self.draw()

    def constraints_window(self):
        from ase.gui.constraints import Constraints
        return Constraints(self)

    def set_selected_atoms(self, selected):
        newmask = np.zeros(len(self.images.selected), bool)
        newmask[selected] = True

        if np.array_equal(newmask, self.images.selected):
            return

        # (By creating newmask, we can avoid resetting the selection in
        # case the selected indices are invalid)
        self.images.selected[:] = newmask
        self.draw()

    def select_all(self, key=None):
        self.images.selected[:] = True
        self.draw()

    def invert_selection(self, key=None):
        self.images.selected[:] = ~self.images.selected
        self.draw()

    def select_constrained_atoms(self, key=None):
        self.images.selected[:] = ~self.images.get_dynamic(self.atoms)
        self.draw()

    def select_immobile_atoms(self, key=None):
        if len(self.images) > 1:
            R0 = self.images[0].positions
            for atoms in self.images[1:]:
                R = atoms.positions
                self.images.selected[:] = ~(np.abs(R - R0) > 1.0e-10).any(1)
        self.draw()

    def movie(self):
        from ase.gui.movie import Movie
        self.movie_window = Movie(self)

    def plot_graphs(self, key=None, expr=None, ignore_if_nan=False):
        from ase.gui.graphs import Graphs
        g = Graphs(self)
        if expr is not None:
            g.plot(expr=expr, ignore_if_nan=ignore_if_nan)

    def pipe(self, task, data):
        process = subprocess.Popen([sys.executable, '-m', 'ase.gui.pipe'],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        pickle.dump((task, data), process.stdin)
        process.stdin.close()
        # Either process writes a line, or it crashes and line becomes ''
        line = process.stdout.readline().decode('utf8').strip()

        if line != 'GUI:OK':
            if line == '':  # Subprocess probably crashed
                line = _('Failure in subprocess')
            self.bad_plot(line)
        else:
            self.subprocesses.append(process)
        return process

    def bad_plot(self, err, msg=''):
        ui.error(_('Plotting failed'), '\n'.join([str(err), msg]).strip())

    def neb(self):
        from ase.utils.forcecurve import fit_images
        try:
            forcefit = fit_images(self.images)
        except Exception as err:
            self.bad_plot(err, _('Images must have energies and forces, '
                                 'and atoms must not be stationary.'))
        else:
            self.pipe('neb', forcefit)

    def bulk_modulus(self):
        try:
            v = [abs(np.linalg.det(atoms.cell)) for atoms in self.images]
            e = [self.images.get_energy(a) for a in self.images]
            from ase.eos import EquationOfState
            eos = EquationOfState(v, e)
            plotdata = eos.getplotdata()
        except Exception as err:
            self.bad_plot(err, _('Images must have energies '
                                 'and varying cell.'))
        else:
            self.pipe('eos', plotdata)

    def reciprocal(self):
        if self.atoms.cell.rank != 3:
            self.bad_plot(_('Requires 3D cell.'))
            return None

        cell = self.atoms.cell.uncomplete(self.atoms.pbc)
        bandpath = cell.bandpath(npoints=0)
        return self.pipe('reciprocal', bandpath)

    def add_tab(self, filename, images):
        """Add a new tab with the filename as the title and associated Images."""
        # Save current tab view/selection before switching, so it doesn't get
        # reset when a new tab is added.
        if self.current_tab is not None and self.current_tab in self.tabs:
            try:
                self.tab_view_settings[self.current_tab] = {
                    'scale': self.scale,
                    'center': self.center.copy(),
                    'axes': self.axes.copy()
                }
            except Exception:
                # Be lenient if any attribute is missing
                self.tab_view_settings[self.current_tab] = {
                    'scale': getattr(self, 'scale', 1.0),
                    'center': getattr(self, 'center', np.zeros(3)).copy(),
                    'axes': getattr(self, 'axes', np.eye(3)).copy()
                }
            if hasattr(self, 'images') and hasattr(self.images, 'selected'):
                self.tab_selection_state[self.current_tab] = self.images.selected.copy()

        if not isinstance(images, Images):
            images = Images(images)

        # Ensure this Images object has its own independent selection array
        if hasattr(images, 'selected'):
            images.selected = np.zeros(len(images.selected), bool)

        tab_title = filename.split('/')[-1]  # Extract the file name from the path
        # Pass full filepath so the TabControl can show it on hover
        tab_id = self.tab_control.add_tab(tab_title, filepath=filename)
        self.tabs[tab_id] = images

        # Make the new tab current and associate a view for it (inherit current
        # GUI view so the new tab starts with the same view but remains independent).
        self.current_tab = tab_id
        self.images = images

        try:
            self.tab_view_settings[tab_id] = {
                'scale': self.scale,
                'center': self.center.copy(),
                'axes': self.axes.copy()
            }
        except Exception:
            self.tab_view_settings[tab_id] = {
                'scale': getattr(self, 'scale', 1.0),
                'center': getattr(self, 'center', np.zeros(3)).copy(),
                'axes': getattr(self, 'axes', np.eye(3)).copy()
            }

        # Initialize selection state for this tab
        self.tab_selection_state[tab_id] = images.selected.copy()

        # Now set frame and redraw (the switch_tab logic restores settings when
        # switching between already added tabs; for a freshly added tab we keep
        # the inherited view we stored above).
        self.set_frame(len(self.images) - 1, focus=True)
        self.draw()
        
        # Ensure the main window has focus after adding a new tab
        # This is critical for arrow key functionality
        try:
            self.window.win.focus_force()
        except Exception:
            pass

    def switch_tab(self, tab_id):
        """Switch to the specified tab."""
        if self.current_tab is not None and self.current_tab in self.tabs:
            # Save the current view settings for the current tab
            self.tab_view_settings[self.current_tab] = {
                'scale': self.scale,
                'center': self.center.copy(),
                'axes': self.axes.copy()
            }
            # Save the current selection state by copying the actual selection array
            if hasattr(self.images, 'selected'):
                self.tab_selection_state[self.current_tab] = self.images.selected.copy()

        if tab_id in self.tabs:
            self.current_tab = tab_id
            self.images = self.tabs[tab_id]
            
            # Restore the saved selection state for the new tab
            if tab_id in self.tab_selection_state:
                # Ensure we have the right size array
                saved_selection = self.tab_selection_state[tab_id]
                if len(self.images.selected) != len(saved_selection):
                    # Resize if needed
                    new_selection = np.zeros(len(self.images.selected), bool)
                    min_len = min(len(new_selection), len(saved_selection))
                    new_selection[:min_len] = saved_selection[:min_len]
                    self.images.selected[:] = new_selection
                else:
                    self.images.selected[:] = saved_selection
            else:
                # Default: no selection
                self.images.selected[:] = False
            
            # Also reset the move_atoms_mask if in move/rotate mode
            if self.arrowkey_mode != self.ARROWKEY_SCAN:
                self.move_atoms_mask = self.images.selected.copy()
            
            self.set_frame(len(self.images) - 1, focus=True)

            # Restore the saved view settings for the new tab, if available
            if tab_id in self.tab_view_settings:
                settings = self.tab_view_settings[tab_id]
                self.scale = settings['scale']
                self.center = settings['center']
                self.axes = settings['axes']
            else:
                # Default view settings if no saved settings exist
                self.scale = 1.0
                self.center = np.zeros(3)
                self.axes = np.eye(3)

            self.draw()
            
            # Ensure the main window has focus after switching tabs
            # This is critical for arrow key functionality
            try:
                self.window.win.focus_force()
            except Exception:
                pass

    def open(self, button=None, filename=None):
        # Prefer the native OS file dialog when available for a familiar
        # experience. Fall back to the bundled ASEFileChooser if needed.
        # Try desktop-native pickers first (Linux: zenity/kdialog), then
        # tkinter's askopenfilename, then ASEFileChooser as a final fallback.
        filename = filename or ''
        if not filename:
            try:
                import subprocess
                import shutil

                # zenity is common on GNOME, kdialog on KDE
                if shutil.which('zenity'):
                    proc = subprocess.run(['zenity', '--file-selection'],
                                           capture_output=True, text=True)
                    if proc.returncode == 0:
                        filename = proc.stdout.strip()
                elif shutil.which('kdialog'):
                    proc = subprocess.run(['kdialog', '--getopenfilename'],
                                           capture_output=True, text=True)
                    if proc.returncode == 0:
                        filename = proc.stdout.strip()
            except Exception:
                filename = filename or ''

        if not filename:
            try:
                from tkinter.filedialog import askopenfilename
                filename = filename or askopenfilename(title=_('Open ...'))
                format = None
            except Exception:
                chooser = ui.ASEFileChooser(self.window.win)
                filename = filename or chooser.go()
                format = chooser.format
        else:
            format = None

        if filename:
            try:
                new_images = Images()
                new_images.read([filename], slice(None), format)
                self.add_tab(filename, new_images)  # Use filename as tab title
            except Exception as err:
                ui.show_io_error(filename, err)
                return  # Hmm.  Is self.images in a consistent state?
            self.set_frame(len(self.images) - 1, focus=True)

    def modify_atoms(self, key=None):
        from ase.gui.modify import ModifyAtoms
        return ModifyAtoms(self)

    def add_atoms(self, key=None):
        from ase.gui.add import AddAtoms
        return AddAtoms(self)

    def cell_editor(self, key=None):
        from ase.gui.celleditor import CellEditor
        return CellEditor(self)

    def atoms_editor(self, key=None):
        from ase.gui.atomseditor import AtomsEditor
        return AtomsEditor(self)

    def quick_info_window(self, key=None):
        from ase.gui.quickinfo import info
        info_win = ui.Window(_('Quick Info'))
        info_win.add(info(self))

        # Update quickinfo window when we change frame
        def update(window):
            exists = window.exists
            if exists:
                # Only update if we exist
                window.things[0].text = info(self)
            return exists
        self.attach(update, info_win)
        return info_win

    def surface_window(self):
        return SetupSurfaceSlab(self)

    def nanoparticle_window(self):
        return SetupNanoparticle(self)

    def nanotube_window(self):
        return SetupNanotube(self)

    def new_atoms(self, atoms):
        "Set a new atoms object."
        rpt = getattr(self.images, 'repeat', None)
        self.images.repeat_images(np.ones(3, int))
        self.images.initialize([atoms])
        self.frame = 0  # Prevent crashes
        self.images.repeat_images(rpt)
        self.set_frame(frame=0, focus=True)
        self.obs.new_atoms.notify()

    def exit(self, event=None):
        for process in self.subprocesses:
            process.terminate()
        self.window.close()

    def new(self, key=None):
        subprocess.Popen([sys.executable, '-m', 'ase', 'gui'])

    def save(self, key=None):
        return save_dialog(self)

    def external_viewer(self, name):
        from ase.visualize import view
        return view(list(self.images), viewer=name)

    def selected_atoms(self):
        selection_mask = self.images.selected[:len(self.atoms)]
        return self.atoms[selection_mask]

    # Clipboard API (restored to fix Edit menu callbacks)
    @property
    def clipboard(self):
        from ase.gui.clipboard import AtomsClipboard
        return AtomsClipboard(self.window.win)

    def cut_atoms_to_clipboard(self, event=None):
        self.copy_atoms_to_clipboard(event)
        self.really_delete_selected_atoms()

    def copy_atoms_to_clipboard(self, event=None):
        atoms = self.selected_atoms()
        self.clipboard.set_atoms(atoms)

    def paste_atoms_from_clipboard(self, event=None):
        try:
            atoms = self.clipboard.get_atoms()
        except Exception as err:
            ui.error(
                'Cannot paste atoms',
                'Pasting currently works only with the ASE JSON format.\n\n'
                f'Original error:\n\n{err}')
            return

        if self.atoms == Atoms():
            self.atoms.cell = atoms.cell
            self.atoms.pbc = atoms.pbc
        self.paste_atoms_onto_existing(atoms)

    def paste_atoms_onto_existing(self, atoms):
        selection = self.selected_atoms()
        if len(selection):
            paste_center = selection.positions.sum(axis=0) / len(selection)
            atoms = atoms.copy()
            atoms.cell = (1, 1, 1)
            atoms.center(about=paste_center)

        self.add_atoms_and_select(atoms)
        self.move_atoms_mask = self.images.selected.copy()
        self.arrowkey_mode = self.ARROWKEY_MOVE
        self.draw()

    def add_atoms_and_select(self, new_atoms):
        atoms = self.atoms
        atoms += new_atoms

        if len(atoms) > self.images.maxnatoms:
            self.images.initialize(list(self.images),
                                   self.images.filenames)

        selected = self.images.selected
        selected[:] = False
        selected[len(atoms) - len(new_atoms):len(atoms)] = True

        self.set_frame()
        self.draw()

    def align_view_along_axis(self, axis_type='a', reciprocal=False):
        """Align view along crystallographic axis.
        
        Args:
            axis_type: 'a', 'b', or 'c' for the crystallographic axis
            reciprocal: If True, use reciprocal lattice vector (a*, b*, c*)
        """
        # Record view change
        self._push_undo('view_align')
        if self.atoms.cell.rank != 3:
            ui.error(_('Error'), _('Requires 3D cell for axis alignment.'))
            return
        
        axis_map = {'a': 0, 'b': 1, 'c': 2}
        axis_idx = axis_map.get(axis_type.lower(), 0)
        
        if reciprocal:
            # Get reciprocal lattice vectors
            cell = self.atoms.cell.reciprocal()
            axis_vector = cell[axis_idx]
        else:
            # Get direct lattice vectors
            axis_vector = self.atoms.cell[axis_idx]
        
        # Normalize the axis vector
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        
        # We want to look along this axis, so set it as the z-direction
        # Create a rotation matrix that aligns axis_vector with [0, 0, 1]
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # If vectors are already aligned, no rotation needed
        if np.allclose(axis_vector, z_axis):
            self.axes = np.eye(3)
        elif np.allclose(axis_vector, -z_axis):
            self.axes = np.diag([1, -1, -1])
        else:
            # Rotation axis is perpendicular to both vectors
            rot_axis = np.cross(axis_vector, z_axis)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            
            # Rotation angle
            cos_angle = np.dot(axis_vector, z_axis)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Rodrigues rotation formula
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                         [rot_axis[2], 0, -rot_axis[0]],
                         [-rot_axis[1], rot_axis[0], 0]])
            
            R = (np.eye(3) + np.sin(angle) * K + 
                 (1 - np.cos(angle)) * np.dot(K, K))
            
            self.axes = R.T
        
        self.draw()

    def align_along_a(self, key=None):
        """Align view along a-axis (direct lattice)."""
        self.align_view_along_axis('a', reciprocal=False)

    def align_along_b(self, key=None):
        """Align view along b-axis (direct lattice)."""
        self.align_view_along_axis('b', reciprocal=False)

    def align_along_c(self, key=None):
        """Align view along c-axis (direct lattice)."""
        self.align_view_along_axis('c', reciprocal=False)

    def align_along_a_star(self, key=None):
        """Align view along a*-axis (reciprocal lattice)."""
        self.align_view_along_axis('a', reciprocal=True)

    def align_along_b_star(self, key=None):
        """Align view along b*-axis (reciprocal lattice)."""
        self.align_view_along_axis('b', reciprocal=True)

    def align_along_c_star(self, key=None):
        """Align view along c*-axis (reciprocal lattice)."""
        self.align_view_along_axis('c', reciprocal=True)

    def wrap_atoms(self, key=None):
        """Wrap atoms around the unit cell."""
        # Record positions before wrapping
        self._push_undo('atoms_wrap')
        for atoms in self.images:
            atoms.wrap()
        self.set_frame()

    def get_menu_data(self):
        M = ui.MenuItem
        return [
            (_('_File'),
             [M(_('_Open'), self.open, 'Ctrl+O'),
              M(_('_New'), self.new, 'Ctrl+N'),
              M(_('_Save'), self.save, 'Ctrl+S'),
              M('---'),
              M(_('_Quit'), self.exit, 'Ctrl+Q')]),

            (_('_Edit'),
             [
              # Undo/Redo at the top
              M(_('_Undo'), self.undo, 'Ctrl+Z'),
              M(_('_Redo'), self.redo, 'Ctrl+Y'),
              M('---'),
              M(_('Select _all'), self.select_all),
              M(_('_Invert selection'), self.invert_selection),
              M(_('Select _constrained atoms'), self.select_constrained_atoms),
              M(_('Select _immobile atoms'), self.select_immobile_atoms),
              # M('---'),
              M(_('_Cut'), self.cut_atoms_to_clipboard, 'Ctrl+X'),
              M(_('_Copy'), self.copy_atoms_to_clipboard, 'Ctrl+C'),
              M(_('_Paste'), self.paste_atoms_from_clipboard, 'Ctrl+V'),
              M('---'),
              M(_('Hide selected atoms'), self.hide_selected),
              M(_('Show selected atoms'), self.show_selected),
              M('---'),
              # Remove Ctrl+Y from Modify to free it for Redo
              M(_('_Modify'), self.modify_atoms),
              M(_('_Add atoms'), self.add_atoms, 'Ctrl+A'),
              M(_('_Delete selected atoms'), self.delete_selected_atoms,
                'Backspace'),
              M(_('Edit _cell …'), self.cell_editor, 'Ctrl+E'),
              M(_('Edit _atoms …'), self.atoms_editor, 'A'),
              M('---'),
              M(_('_First image'), self.step, 'Home'),
              M(_('_Previous image'), self.step, 'Page-Up'),
              M(_('_Next image'), self.step, 'Page-Down'),
              M(_('_Last image'), self.step, 'End'),
              M(_('Append image copy'), self.copy_image)]),

            (_('_View'),
             [M(_('Show _unit cell'), self.toggle_show_unit_cell, 'Ctrl+U',
                value=self.config['show_unit_cell']),
              M(_('Show _axes'), self.toggle_show_axes,
                value=self.config['show_axes']),
              M(_('Show _bonds'), self.toggle_show_bonds, 'Ctrl+B',
                value=self.config['show_bonds']),
              M(_('Show _velocities'), self.toggle_show_velocities, 'Ctrl+G',
                value=False),
              M(_('Show _forces'), self.toggle_show_forces, 'Ctrl+F',
                value=False),
              M(_('Show _magmoms'), self.toggle_show_magmoms,
                value=False),
              M(_('Show _Labels'), self.show_labels,
                choices=[_('_None'),
                         _('Atom _Index'),
                         _('_Magnetic Moments'),  # XXX check if exist
                         _('_Element Symbol'),
                         _('_Initial Charges'),  # XXX check if exist
                         ]),
              M('---'),
              M(_('Quick Info ...'), self.quick_info_window, 'Ctrl+I'),
              M(_('Repeat ...'), self.repeat_window, 'R'),
              M(_('Rotate ...'), self.rotate_window),
              M(_('Colors ...'), self.colors_window, 'C'),
              # TRANSLATORS: verb
              M(_('Focus'), self.focus, 'F'),
              M(_('Zoom in'), self.zoom, '+'),
              M(_('Zoom out'), self.zoom, '-'),
              M(_('Change View'),
                submenu=[
                    M(_('Reset View'), self.reset_view, '='),
                    M('---'),
                    M(_('xy-plane'), self.set_view, 'Z'),
                    M(_('yz-plane'), self.set_view, 'X'),
                    M(_('zx-plane'), self.set_view, 'Y'),
                    M(_('yx-plane'), self.set_view, 'Shift+Z'),
                    M(_('zy-plane'), self.set_view, 'Shift+X'),
                    M(_('xz-plane'), self.set_view, 'Shift+Y'),
                    M('---'),
                    M(_('a2,a3-plane'), self.set_view, 'I'),
                    M(_('a3,a1-plane'), self.set_view, 'J'),
                    M(_('a1,a2-plane'), self.set_view, 'K'),
                    M(_('a3,a2-plane'), self.set_view, 'Shift+I'),
                    M(_('a1,a3-plane'), self.set_view, 'Shift+J'),
                    M(_('a2,a1-plane'), self.set_view, 'Shift+K'),
                    M('---'),
                    # Toggle interactive pan mode: change cursor and allow drag-to-pan
                    M(_('Pan'), self.toggle_pan_mode, 'P'),
                    M('---'),
                    M(_('Along a-axis'), self.align_along_a),
                    M(_('Along b-axis'), self.align_along_b),
                    M(_('Along c-axis'), self.align_along_c),
                    M(_('Along a*-axis'), self.align_along_a_star),
                    M(_('Along b*-axis'), self.align_along_b_star),
                    M(_('Along c*-axis'), self.align_along_c_star
                    )]),
              M(_('Settings ...'), self.settings),
              M('---'),
              M(_('VMD'), partial(self.external_viewer, 'vmd')),
              M(_('RasMol'), partial(self.external_viewer, 'rasmol')),
              M(_('xmakemol'), partial(self.external_viewer, 'xmakemol')),
              M(_('avogadro'), partial(self.external_viewer, 'avogadro'))]),

            (_('_Tools'),
             [M(_('Graphs ...'), self.plot_graphs),
              M(_('Movie ...'), self.movie),
              M(_('Constraints ...'), self.constraints_window),
              M(_('Render scene ...'), self.render_window),
              M(_('_Move selected atoms'), self.toggle_move_mode, 'Ctrl+M'),
              M(_('_Rotate selected atoms'), self.toggle_rotate_mode,
                'Ctrl+R'),
              M(_('NE_B plot'), self.neb),
              M(_('B_ulk Modulus'), self.bulk_modulus),
              M(_('Reciprocal space ...'), self.reciprocal),
              M(_('Wrap atoms'), self.wrap_atoms, 'Ctrl+W')]),

            # TRANSLATORS: Set up (i.e. build) surfaces, nanoparticles, ...
            (_('_Setup'),
             [M(_('_Surface slab'), self.surface_window, disabled=False),
              M(_('_Nanoparticle'),
                self.nanoparticle_window),
              M(_('Nano_tube'), self.nanotube_window)]),

            # (_('_Calculate'),
            # [M(_('Set _Calculator'), self.calculator_window, disabled=True),
            #  M(_('_Energy and Forces'), self.energy_window, disabled=True),
            #  M(_('Energy Minimization'), self.energy_minimize_window,
            #    disabled=True)]),

            (_('_Helped'),
             [M(_('_About'), partial(
                 ui.about, 'ASE-GUI',
                 version=__version__,
                 webpage='https://ase-lib.org/ase/gui/gui.html')),
              M(_('Webpage ...'), webpage)])]

    def attach(self, function, *args, **kwargs):
        self.observers.append((function, args, kwargs))

    def call_observers(self):
        # Use function return value to determine if we keep observer
        self.observers = [(function, args, kwargs) for (function, args, kwargs)
                          in self.observers if function(*args, **kwargs)]

    def repeat_poll(self, callback, ms, ensure_update=True):
        """Invoke callback(gui=self) every ms milliseconds.

        This is useful for polling a resource for updates to load them
        into the GUI.  The GUI display will be hence be updated after
        each call; pass ensure_update=False to circumvent this.

        Polling stops if the callback function raises StopIteration.

        Example to run a movie manually, then quit::

            from ase.collections import g2
            from ase.gui.gui import GUI

            names = iter(g2.names)

            def main(gui):
                try:
                    name = next(names)
                except StopIteration:
                    gui.window.win.quit()
                else:
                    atoms = g2[name]
                    gui.images.initialize([atoms])

            gui = GUI()
            gui.repeat_poll(main, 30)
            gui.run()"""

        def callbackwrapper():
            try:
                callback(gui=self)
            except StopIteration:
                pass
            finally:
                # Reinsert self so we get called again:
                self.window.win.after(ms, callbackwrapper)

            if ensure_update:
                self.set_frame()
                self.draw()

        self.window.win.after(ms, callbackwrapper)


    # --- Added pan utilities ---
    def toggle_pan_mode(self, key=None):
        """Toggle pan mode and update canvas cursor to a four-spoked asterisk."""
        self.pan_mode = not getattr(self, "pan_mode", False)
        try:
            canvas = getattr(self.window, "canvas", None)
            if canvas is not None:
                canvas.configure(cursor='fleur' if self.pan_mode else '')
        except Exception:
            pass
        try:
            # ensure GUI keeps focus for keyboard shortcuts
            self.window.win.focus_force()
        except Exception:
            pass
        return self.pan_mode

    def press(self, event):
        """Handle mouse press. Start panning if pan_mode is active."""
        if getattr(self, "pan_mode", False):
            try:
                # Store start position (pixels) and original center
                self._pan_start = np.array([event.x, event.y])
                self._pan_orig_center = self.center.copy()
                self._pan_undo_recorded = False
                # Ensure focus so keyboard shortcuts still work
                try:
                    self.window.win.focus_force()
                except Exception:
                    pass
            except Exception:
                # If anything goes wrong, ignore and fall back
                self._pan_start = None
            return
        # Default behaviour - delegate to View.press
        try:
            return super().press(event)
        except Exception:
            return None

    def move(self, event):
        """Handle mouse motion. If panning, compute displacement and update center."""
        if getattr(self, "pan_mode", False) and getattr(self, "_pan_start", None) is not None:
            try:
                # Record undo only once per pan gesture
                if not self._pan_undo_recorded:
                    self._push_undo('view_pan_mode')
                    self._pan_undo_recorded = True

                cur = np.array([event.x, event.y])
                delta = cur - self._pan_start  # pixels: +x right, +y down

                # Convert pixel delta to a world displacement.
                # Map pixel fraction to world by scaling with original scale and axes.
                try:
                    w, h = self.window.size
                except Exception:
                    w, h = 450.0, 450.0

                # normalized screen displacement (right, up)
                frac = np.array([delta[0] / max(1.0, w), -delta[1] / max(1.0, h), 0.0])

                # Choose a sensible mapping to world coordinates using axes and scale.
                # The factor controls sensitivity; tune if needed.
                sensitivity = max(0.5, self.orig_scale)  # avoid too small numbers
                world_vec = (frac[0] * self.axes[:, 0] + frac[1] * self.axes[:, 1]) * sensitivity

                # Apply pan relative to original center so panning is smooth while dragging
                self.center = self._pan_orig_center - world_vec
                self.draw()
            except Exception:
                pass
            return

        # Not panning: delegate to View.move
        try:
            return super().move(event)
        except Exception:
            return None

    def release(self, event):
        """End panning on mouse release (cleanup)."""
        if getattr(self, "pan_mode", False) and getattr(self, "_pan_start", None) is not None:
            try:
                del self._pan_start
                del self._pan_orig_center
                self._pan_undo_recorded = False
            except Exception:
                pass
            return
        try:
            return super().release(event)
        except Exception:
            return None
    # --- end added pan utilities ---

def webpage():
    import webbrowser
    webbrowser.open('https://ase-lib.org/ase/gui/gui.html')
