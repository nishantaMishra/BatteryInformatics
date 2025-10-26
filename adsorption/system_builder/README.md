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

# Known Issues

Loaded tab still affects the other tabs.
Maybe that's not a bug but a feature because the newly loaded structure is ready to be compared with the previous one.