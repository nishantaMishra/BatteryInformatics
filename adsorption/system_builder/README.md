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
- Panning with keypress "P"  
- Supports undo/redo of atom movements with Ctrl+Z / Ctrl+Y
- Tab location on hoverng the tab bar.

# Known Issues

1. Loaded tab still affects the other tabs.
But, that can be presented as a feature because the newly loaded structure is ready to be compared with the previous one.

2. Currently, there is no way to close tabs. Just open a new instance of the program. Ctrl+W will be implemented in the future.
3. `ase_dragdrop_gui.py` could be moved inside the directory.

# To do
1. Make colour schemes work. Add interactive colour changing.
2. Support for XDATCAR.
3. Alignment of structure by clicking the axes. 

