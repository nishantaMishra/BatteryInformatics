# fmt: off

"""Dialog for saving one or more configurations."""

import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.io.formats import (
    filetype,
    get_ioformat,
    parse_filename,
    string2index,
    write,
)

text = _("""\
Append name with "@n" in order to write image
number "n" instead of the current image. Append
"@start:stop" or "@start:stop:step" if you want
to write a range of images. You can leave out
"start" and "stop" so that "name@:" will give
you all images. Negative numbers count from the
last image. Examples: "name@-1": last image,
"name@-2:": last two.""")


def save_dialog(gui, filename=None):
    # Try to get default directory from currently opened file
    initialdir = None
    initialfile = None
    if not filename and hasattr(gui.images, 'filenames'):
        current_filename = gui.images.filenames[gui.frame]
        if current_filename:
            import os
            initialdir = os.path.dirname(os.path.abspath(current_filename))
            initialfile = os.path.basename(current_filename)

    if not filename:
        # Try desktop-native pickers first (zenity/kdialog on Linux),
        # then tkinter's asksaveasfilename, then SaveFileDialog.
        try:
            import subprocess
            import shutil
            import os

            if shutil.which('zenity'):
                args = ['zenity', '--file-selection', '--save', '--confirm-overwrite']
                if initialfile:
                    args.extend(['--filename', os.path.join(initialdir or '', initialfile)])
                proc = subprocess.run(args, capture_output=True, text=True)
                if proc.returncode == 0:
                    filename = proc.stdout.strip()
            elif shutil.which('kdialog'):
                args = ['kdialog', '--getsavefilename']
                if initialdir:
                    args.append(initialdir)
                if initialfile:
                    args[-1] = os.path.join(args[-1] if len(args) > 2 else '.', initialfile)
                proc = subprocess.run(args, capture_output=True, text=True)
                if proc.returncode == 0:
                    filename = proc.stdout.strip()
        except Exception:
            pass

        # Fall back to tkinter file dialog
        if not filename:
            try:
                from tkinter.filedialog import asksaveasfilename
                kwargs = {'title': _('Save ...')}
                if initialdir:
                    kwargs['initialdir'] = initialdir
                if initialfile:
                    kwargs['initialfile'] = initialfile
                filename = asksaveasfilename(**kwargs)
            except Exception:
                pass

        # Final fallback to SaveFileDialog
        if not filename:
            dialog = ui.SaveFileDialog(gui.window.win, _('Save ...'))
            ui.Text(text).pack(dialog.top)
            filename = dialog.go()

    if not filename:
        return

    filename, index = parse_filename(filename)
    if index is None:
        index = slice(gui.frame, gui.frame + 1)
    elif isinstance(index, str):
        try:
            index = string2index(index)
        except Exception:
            # If string2index fails, default to current frame
            index = slice(gui.frame, gui.frame + 1)
    elif isinstance(index, slice):
        pass
    else:
        if index < 0:
            index += len(gui.images)
        index = slice(index, index + 1)
    format = filetype(filename, read=False)
    io = get_ioformat(format)

    extra = {}
    remove_hidden = False
    if format in ['png', 'eps', 'pov']:
        bbox = np.empty(4)
        size = gui.window.size / gui.scale
        bbox[0:2] = np.dot(gui.center, gui.axes[:, :2]) - size / 2
        bbox[2:] = bbox[:2] + size
        extra['rotation'] = gui.axes
        extra['show_unit_cell'] = gui.window['toggle-show-unit-cell']
        extra['bbox'] = bbox
        colors = gui.get_colors(rgb=True)
        extra['colors'] = [rgb for rgb, visible
                           in zip(colors, gui.images.visible)
                           if visible]
        remove_hidden = True

    images = [gui.images.get_atoms(i, remove_hidden=remove_hidden)
              for i in range(*index.indices(len(gui.images)))]

    if len(images) > 1 and io.single:
        # We want to write multiple images, but the file format does not
        # support it.  The solution is to write multiple files, inserting
        # a number in the file name before the suffix.
        j = filename.rfind('.')
        filename = filename[:j] + '{0:05d}' + filename[j:]
        for i, atoms in enumerate(images):
            write(filename.format(i), atoms, **extra)
    else:
        try:
            write(filename, images, **extra)
        except Exception as err:
            from ase.gui.ui import showerror
            showerror(_('Error'), err)
            raise
