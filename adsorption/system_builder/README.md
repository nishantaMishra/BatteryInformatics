Basically, edited codes of ase gui. 

While development execute
```bash
python3 -m ase gui
```

Execute the command in the project directory to open the GUI.


Added features:
- Multiple tabs support
- Opens native file browser dialogs for loading and saving files.
- Presets for view configurations. I, J and K.
- Panning with keypress "P"  (Double right click to trigger panning.)
- Supports undo/redo of atom movements with Ctrl+Z / Ctrl+Y
- Tab location on hovering the tab bar.
- Support for XDATCAR.
- Axes gizmo: Alignment of structure by clicking the axes.
- Ctrl+W to close tab.

# Known Issues
1. `ase_dragdrop_gui.py` could be moved inside the directory.

# To do
1. Make colour schemes work. Add interactive colour changing.

