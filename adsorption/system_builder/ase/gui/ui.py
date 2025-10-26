# fmt: off

# type: ignore
import platform
import re
import tkinter as tk
import tkinter.ttk as ttk
from collections import namedtuple
from functools import partial
from tkinter.filedialog import LoadFileDialog, SaveFileDialog
from tkinter.messagebox import askokcancel as ask_question
from tkinter.messagebox import showerror, showinfo, showwarning

import numpy as np

from ase.gui.i18n import _

__all__ = [
    'error', 'ask_question', 'MainWindow', 'LoadFileDialog', 'SaveFileDialog',
    'ASEGUIWindow', 'Button', 'CheckButton', 'ComboBox', 'Entry', 'Label',
    'Window', 'MenuItem', 'RadioButton', 'RadioButtons', 'Rows', 'Scale',
    'showinfo', 'showwarning', 'SpinBox', 'Text']


def error(title, message=None):
    if message is None:
        message = title
        title = _('Error')
    return showerror(title, message)


def about(name, version, webpage):
    text = [name,
            '',
            _('Version') + ': ' + version,
            _('Web-page') + ': ' + webpage]
    win = Window(_('About'))
    win.add(Text('\n'.join(text)))


def helpbutton(text):
    return Button(_('Help'), helpwindow, text)


def helpwindow(text):
    win = Window(_('Help'))
    win.add(Text(text))


class BaseWindow:
    def __init__(self, title, close=None):
        self.title = title
        if close:
            self.win.protocol('WM_DELETE_WINDOW', close)
        else:
            self.win.protocol('WM_DELETE_WINDOW', self.close)

        self.things = []
        self.exists = True

    def close(self):
        self.win.destroy()
        self.exists = False

    def title(self, txt):
        self.win.title(txt)

    title = property(None, title)

    def add(self, stuff, anchor='w'):  # 'center'):
        if isinstance(stuff, str):
            stuff = Label(stuff)
        elif isinstance(stuff, list):
            stuff = Row(stuff)
        stuff.pack(self.win, anchor=anchor)
        self.things.append(stuff)


class Window(BaseWindow):
    def __init__(self, title, close=None):
        self.win = tk.Toplevel()
        super().__init__(title, close)


class Widget:
    def pack(self, parent, side='top', anchor='center'):
        widget = self.create(parent)
        widget.pack(side=side, anchor=anchor)
        if not isinstance(self, (Rows, RadioButtons)):
            pass

    def grid(self, parent):
        widget = self.create(parent)
        widget.grid()

    def create(self, parent):
        self.widget = self.creator(parent)
        return self.widget

    @property
    def active(self):
        return self.widget['state'] == 'normal'

    @active.setter
    def active(self, value):
        self.widget['state'] = ['disabled', 'normal'][bool(value)]


class Row(Widget):
    def __init__(self, things):
        self.things = things

    def create(self, parent):
        self.widget = tk.Frame(parent)
        for thing in self.things:
            if isinstance(thing, str):
                thing = Label(thing)
            thing.pack(self.widget, 'left')
        return self.widget

    def __getitem__(self, i):
        return self.things[i]


class Label(Widget):
    def __init__(self, text='', color=None):
        self.creator = partial(tk.Label, text=text, fg=color)

    @property
    def text(self):
        return self.widget['text']

    @text.setter
    def text(self, new):
        self.widget.config(text=new)


class Text(Widget):
    def __init__(self, text):
        self.creator = partial(tk.Text, height=text.count('\n') + 1)
        s = re.split('<(.*?)>', text)
        self.text = [(s[0], ())]
        i = 1
        tags = []
        while i < len(s):
            tag = s[i]
            if tag[0] != '/':
                tags.append(tag)
            else:
                tags.pop()
            self.text.append((s[i + 1], tuple(tags)))
            i += 2

    def create(self, parent):
        widget = Widget.create(self, parent)
        widget.tag_configure('sub', offset=-6)
        widget.tag_configure('sup', offset=6)
        widget.tag_configure('c', foreground='blue')
        for text, tags in self.text:
            widget.insert('insert', text, tags)
        widget.configure(state='disabled', background=parent['bg'])
        widget.bind("<1>", lambda event: widget.focus_set())
        return widget


class Button(Widget):
    def __init__(self, text, callback, *args, **kwargs):
        self.callback = partial(callback, *args, **kwargs)
        self.creator = partial(tk.Button,
                               text=text,
                               command=self.callback)


class CheckButton(Widget):
    def __init__(self, text, value=False, callback=None):
        self.text = text
        self.var = tk.BooleanVar(value=value)
        self.callback = callback

    def create(self, parent):
        self.check = tk.Checkbutton(parent, text=self.text,
                                    var=self.var, command=self.callback)
        return self.check

    @property
    def value(self):
        return self.var.get()


class SpinBox(Widget):
    def __init__(self, value, start, end, step, callback=None,
                 rounding=None, width=6):
        self.callback = callback
        self.rounding = rounding
        self.creator = partial(tk.Spinbox,
                               from_=start,
                               to=end,
                               increment=step,
                               command=callback,
                               width=width)
        self.initial = str(value)

    def create(self, parent):
        self.widget = self.creator(parent)
        bind_enter(self.widget, lambda event: self.callback())
        self.value = self.initial
        return self.widget

    @property
    def value(self):
        x = self.widget.get().replace(',', '.')
        if '.' in x:
            return float(x)
        if x == 'None':
            return None
        return int(x)

    @value.setter
    def value(self, x):
        self.widget.delete(0, 'end')
        if '.' in str(x) and self.rounding is not None:
            try:
                x = round(float(x), self.rounding)
            except (ValueError, TypeError):
                pass
        self.widget.insert(0, x)


# Entry and ComboBox use same mechanism (since ttk ComboBox
# is a subclass of tk Entry).
def _set_entry_value(widget, value):
    widget.delete(0, 'end')
    widget.insert(0, value)


class Entry(Widget):
    def __init__(self, value='', width=20, callback=None):
        self.creator = partial(tk.Entry,
                               width=width)
        if callback is not None:
            self.callback = lambda event: callback()
        else:
            self.callback = None
        self.initial = value

    def create(self, parent):
        self.entry = self.creator(parent)
        self.value = self.initial
        if self.callback:
            bind_enter(self.entry, self.callback)
        return self.entry

    @property
    def value(self):
        return self.entry.get()

    @value.setter
    def value(self, x):
        _set_entry_value(self.entry, x)


class Scale(Widget):
    def __init__(self, value, start, end, callback):
        def command(val):
            callback(int(val))

        self.creator = partial(tk.Scale,
                               from_=start,
                               to=end,
                               orient='horizontal',
                               command=command)
        self.initial = value

    def create(self, parent):
        self.scale = self.creator(parent)
        self.value = self.initial
        return self.scale

    @property
    def value(self):
        return self.scale.get()

    @value.setter
    def value(self, x):
        self.scale.set(x)


class RadioButtons(Widget):
    def __init__(self, labels, values=None, callback=None, vertical=False):
        self.var = tk.IntVar()

        if callback:
            def callback2():
                callback(self.value)
        else:
            callback2 = None

        self.values = values or list(range(len(labels)))
        self.buttons = [RadioButton(label, i, self.var, callback2)
                        for i, label in enumerate(labels)]
        self.vertical = vertical

    def create(self, parent):
        self.widget = frame = tk.Frame(parent)
        side = 'top' if self.vertical else 'left'
        for button in self.buttons:
            button.create(frame).pack(side=side)
        return frame

    @property
    def value(self):
        return self.values[self.var.get()]

    @value.setter
    def value(self, value):
        self.var.set(self.values.index(value))

    def __getitem__(self, value):
        return self.buttons[self.values.index(value)]


class RadioButton(Widget):
    def __init__(self, label, i, var, callback):
        self.creator = partial(tk.Radiobutton,
                               text=label,
                               var=var,
                               value=i,
                               command=callback)


if ttk is not None:
    class ComboBox(Widget):
        def __init__(self, labels, values=None, callback=None):
            self.values = values or list(range(len(labels)))
            self.callback = callback
            self.creator = partial(ttk.Combobox,
                                   values=labels)

        def create(self, parent):
            widget = Widget.create(self, parent)
            widget.current(0)
            if self.callback:
                def callback(event):
                    self.callback(self.value)
                widget.bind('<<ComboboxSelected>>', callback)

            return widget

        @property
        def value(self):
            return self.values[self.widget.current()]

        @value.setter
        def value(self, val):
            _set_entry_value(self.widget, val)
else:
    # Use Entry object when there is no ttk:
    def ComboBox(labels, values, callback):
        return Entry(values[0], callback=callback)


class Rows(Widget):
    def __init__(self, rows=None):
        self.rows_to_be_added = rows or []
        self.creator = tk.Frame
        self.rows = []

    def create(self, parent):
        widget = Widget.create(self, parent)
        for row in self.rows_to_be_added:
            self.add(row)
        self.rows_to_be_added = []
        return widget

    def add(self, row):
        if isinstance(row, str):
            row = Label(row)
        elif isinstance(row, list):
            row = Row(row)
        row.grid(self.widget)
        self.rows.append(row)

    def clear(self):
        while self.rows:
            del self[0]

    def __getitem__(self, i):
        return self.rows[i]

    def __delitem__(self, i):
        widget = self.rows.pop(i).widget
        widget.grid_remove()
        widget.destroy()

    def __len__(self):
        return len(self.rows)


class MenuItem:
    def __init__(self, label, callback=None, key=None,
                 value=None, choices=None, submenu=None, disabled=False):
        self.underline = label.find('_')
        self.label = label.replace('_', '')

        is_macos = platform.system() == 'Darwin'

        if key:
            parts = key.split('+')
            modifiers = []
            key_char = None

            for part in parts:
                if part in ('Alt', 'Shift'):
                    modifiers.append(part)
                elif part == 'Ctrl':
                    modifiers.append('Control')
                elif len(part) == 1 and 'Shift' in modifiers:
                    # If shift and letter, uppercase
                    key_char = part
                else:
                    # Lower case
                    key_char = part.lower()

            if is_macos:
                modifiers = ['Command' if m == 'Alt' else m for m in modifiers]

            if modifiers and key_char:
                self.keyname = f"<{'-'.join(modifiers)}-{key_char}>"
            else:
                # Handle special non-modifier keys
                self.keyname = {
                    'Home': '<Home>',
                    'End': '<End>',
                    'Page-Up': '<Prior>',
                    'Page-Down': '<Next>',
                    'Backspace': '<BackSpace>'
                }.get(key, key.lower())
        else:
            self.keyname = None

        if key:
            def callback2(event=None):
                callback(key)

            callback2.__name__ = callback.__name__
            self.callback = callback2
        else:
            self.callback = callback

        if is_macos and key is not None:
            self.key = key.replace('Alt', 'Command')
        else:
            self.key = key
        self.value = value
        self.choices = choices
        self.submenu = submenu
        self.disabled = disabled

    def addto(self, menu, window, stuff=None):
        callback = self.callback
        if self.label == '---':
            menu.add_separator()
        elif self.value is not None:
            var = tk.BooleanVar(value=self.value)
            stuff[self.callback.__name__.replace('_', '-')] = var

            menu.add_checkbutton(label=self.label,
                                 underline=self.underline,
                                 command=self.callback,
                                 accelerator=self.key,
                                 var=var)

            def callback(key):  # noqa: F811
                var.set(not var.get())
                self.callback()

        elif self.choices:
            submenu = tk.Menu(menu)
            menu.add_cascade(label=self.label, menu=submenu)
            var = tk.IntVar()
            var.set(0)
            stuff[self.callback.__name__.replace('_', '-')] = var
            for i, choice in enumerate(self.choices):
                submenu.add_radiobutton(label=choice.replace('_', ''),
                                        underline=choice.find('_'),
                                        command=self.callback,
                                        value=i,
                                        var=var)
        elif self.submenu:
            submenu = tk.Menu(menu)
            menu.add_cascade(label=self.label,
                             menu=submenu)
            for thing in self.submenu:
                thing.addto(submenu, window)
        else:
            state = 'normal'
            if self.disabled:
                state = 'disabled'
            menu.add_command(label=self.label,
                             underline=self.underline,
                             command=self.callback,
                             accelerator=self.key,
                             state=state)
        if self.key:
            window.bind(self.keyname, callback)


class MainWindow(BaseWindow):
    def __init__(self, title, close=None, menu=[]):
        # Try to use TkinterDnD if available so we can support native
        # drag-and-drop on the canvas. Fall back to normal tk.Tk().
        try:
            # Local import to avoid hard dependency
            from tkinterdnd2 import TkinterDnD
            self.win = TkinterDnD.Tk()
            self._dnd_available = True
        except Exception:
            self.win = tk.Tk()
            self._dnd_available = False
        super().__init__(title, close)

        # self.win.tk.call('tk', 'scaling', 3.0)
        # self.win.tk.call('tk', 'scaling', '-displayof', '.', 7)

        self.menu = {}

        if menu:
            self.create_menu(menu)

    def create_menu(self, menu_description):
        menu = tk.Menu(self.win)
        self.win.config(menu=menu)

        for label, things in menu_description:
            submenu = tk.Menu(menu)
            menu.add_cascade(label=label.replace('_', ''),
                             underline=label.find('_'),
                             menu=submenu)
            for thing in things:
                thing.addto(submenu, self.win, self.menu)

    def resize_event(self):
        # self.scale *= sqrt(1.0 * self.width * self.height / (w * h))
        self.draw()
        self.configured = True

    def run(self):
        # Workaround for nasty issue with tkinter on Mac:
        # https://gitlab.com/ase/ase/issues/412
        #
        # It is apparently a compatibility issue between Python and Tkinter.
        # Some day we should remove this hack.
        while True:
            try:
                tk.mainloop()
                break
            except UnicodeDecodeError:
                pass

    def __getitem__(self, name):
        return self.menu[name].get()

    def __setitem__(self, name, value):
        return self.menu[name].set(value)


def bind(callback, modifier=None):
    def handle(event):
        event.button = event.num
        event.key = event.keysym.lower()
        event.modifier = modifier
        callback(event)
    return handle


class ASEFileChooser(LoadFileDialog):
    def __init__(self, win, formatcallback=lambda event: None):
        from ase.io.formats import all_formats, get_ioformat
        super().__init__(win, _('Open ...'))
        labels = [_('Automatic')]
        values = ['']

        def key(item):
            return item[1][0]

        for format, (description, code) in sorted(all_formats.items(),
                                                  key=key):
            io = get_ioformat(format)
            if io.can_read and description != '?':
                labels.append(_(description))
                values.append(format)

        self.format = None

        def callback(value):
            self.format = value

        Label(_('Choose parser:')).pack(self.top)
        formats = ComboBox(labels, values, callback)
        formats.pack(self.top)


def show_io_error(filename, err):
    showerror(_('Read error'),
              _(f'Could not read {filename}: {err}'))


class ASEGUIWindow(MainWindow):
    def __init__(self, close, menu, config,
                 scroll, scroll_event,
                 press, move, release, resize,
                 open_callback=None):
        super().__init__('ASE-GUI', close, menu)

        self.size = np.array([450, 450])

        self.fg = config['gui_foreground_color']
        self.bg = config['gui_background_color']

        self.canvas = tk.Canvas(self.win,
                                width=self.size[0],
                                height=self.size[1],
                                bg=self.bg,
                                highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Optional callback invoked when files are dropped. The GUI
        # (ase.gui.gui.GUI) will pass its `open` method here so drops
        # can open files directly in the running GUI.
        self.open_callback = open_callback

        # Create status label early so we can report DnD availability.
        self.status = tk.Label(self.win, text='', anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # Try to register drag-and-drop if tkinterdnd2 is available.
        # This is optional; absence simply means DnD is not supported.
        # Try tkinterdnd2 first (preferred). If not available try the
        # Tcl 'tkdnd' package. If neither is available, we leave DnD
        # disabled but show a short hint in the status line.
        dnd_enabled = False
        try:
            from tkinterdnd2 import DND_FILES
            if hasattr(self.canvas, 'drop_target_register'):
                self.canvas.drop_target_register(DND_FILES)
                # Register a Tcl wrapper callback that accepts the raw %D
                tcl_cb = self.win.register(self._on_drop_from_tcl)
                try:
                    # Bind the virtual event to call our tcl wrapper with %D
                    self.win.tk.call('bind', self.canvas._w, '<<Drop>>', f'{tcl_cb} %D')
                except Exception:
                    # Fallback to widget's dnd_bind if direct bind fails
                    try:
                        self.canvas.dnd_bind('<<Drop>>', self._on_drop)
                    except Exception:
                        pass
                dnd_enabled = True
                self.update_status_line(_('Drag-and-drop enabled (tkinterdnd2)'))
                self._dnd_backend = 'tkinterdnd2'
        except Exception:
            # Try Tcl tkdnd package as a fallback
            try:
                # package require tkdnd
                self.win.tk.call('package', 'require', 'tkdnd')
                # register the canvas as drop target for any type
                try:
                    self.win.tk.call('tkdnd::drop_target_register', self.canvas._w, '*')
                except Exception:
                    try:
                        self.win.tk.call('tkdnd::drop_target_register', self.canvas._w, 'DND_Files')
                    except Exception:
                        pass

                # Bind using a Tcl wrapper to receive raw %D
                try:
                    tcl_cb = self.win.register(self._on_drop_from_tcl)
                    self.win.tk.call('bind', self.canvas._w, '<<Drop>>', f'{tcl_cb} %D')
                    dnd_enabled = True
                    self.update_status_line(_('Drag-and-drop enabled (tkdnd)'))
                    self._dnd_backend = 'tkdnd'
                except Exception:
                    # Could not bind; fall through
                    pass
            except Exception:
                # No tkdnd available either
                pass

        if not dnd_enabled:
            # Provide a helpful hint without being intrusive
            try:
                # Only update status line if empty
                if self.status.cget('text') == '':
                    self.update_status_line(_('Drag-and-drop disabled — install tkinterdnd2 to enable'))
            except Exception:
                pass

        self.canvas.bind('<ButtonPress>', bind(press))
        for button in range(1, 4):
            self.canvas.bind(f'<B{button}-Motion>', bind(move))
        self.canvas.bind('<ButtonRelease>', bind(release))
        self.canvas.bind('<Control-ButtonRelease>', bind(release, 'ctrl'))
        self.canvas.bind('<Shift-ButtonRelease>', bind(release, 'shift'))
        self.canvas.bind('<Configure>', resize)
        if not config['swap_mouse']:
            for button in (2, 3):
                self.canvas.bind(f'<Shift-B{button}-Motion>',
                                 bind(scroll))
        else:
            self.canvas.bind('<Shift-B1-Motion>',
                             bind(scroll))

        self.win.bind('<MouseWheel>', bind(scroll_event))
        # Keep a general key binding, but also bind arrow keys at the
        # global level so widgets (e.g. ttk.Notebook) that may take focus
        # do not swallow Left/Right/Up/Down before the GUI can handle them.
        self.win.bind('<Key>', bind(scroll))
        # Bind arrow keys (and common keypad variants) to ensure they are
        # delivered regardless of which widget currently has focus.
        for key in ('<Left>', '<Right>', '<Up>', '<Down>',
                    '<KP_Left>', '<KP_Right>', '<KP_Up>', '<KP_Down>'):
            try:
                self.win.bind_all(key, bind(scroll))
            except Exception:
                # Some platforms may not support certain event names; ignore.
                pass
        self.win.bind('<Shift-Key>', bind(scroll, 'shift'))
        self.win.bind('<Control-Key>', bind(scroll, 'ctrl'))
        
        # Ensure the canvas has initial focus for arrow key handling
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    def update_status_line(self, text):
        self.status.config(text=text)

    def _on_drop(self, event):
        # Many DnD backends supply a 'data' attribute on the event; if we
        # were called with a full tkinter Event object just extract the
        # string and delegate to the shared parser.
        data = getattr(event, 'data', '') or ''
        return self._handle_drop_data(data)

    def _on_drop_from_tcl(self, data):
        """Called via a Tcl-registered callback with the raw %D string.

        This avoids problems where Tk's bind substitution contains codes
        that tkinter's Event builder doesn't understand (e.g. '%#').
        """
        # Tcl passes bytes sometimes; ensure string
        try:
            if isinstance(data, bytes):
                data = data.decode('utf8')
        except Exception:
            pass
        return self._handle_drop_data(data)

    def _handle_drop_data(self, data):
        """Parse a raw drop data string and call open_callback for each file."""
        files = []
        import re, shlex

        # Extract braced groups first: {path with spaces}
        brace_matches = re.findall(r'{([^}]*)}', data)
        if brace_matches:
            files = brace_matches
        else:
            # Try to treat as a shell-like list (handles quoted items)
            try:
                files = shlex.split(data)
            except Exception:
                files = [s for s in data.split() if s]

        if not files:
            return 'break'

        for f in files:
            if not f:
                continue
            if f.startswith('file://'):
                f = f[7:]
            try:
                if callable(self.open_callback):
                    self.open_callback(filename=f)
            except Exception:
                # Swallow errors here; GUI.open has its own error reporting
                pass
        return 'break'

    def run(self):
        MainWindow.run(self)

    def click(self, name):
        self.callbacks[name]()

    def clear(self):
        self.canvas.delete(tk.ALL)

    def update(self):
        self.canvas.update_idletasks()

    def circle(self, color, selected, *bbox):
        if selected:
            outline = '#004500'
            width = 3
        else:
            outline = 'black'
            width = 1
        self.canvas.create_oval(*tuple(int(x) for x in bbox), fill=color,
                                outline=outline, width=width)

    def arc(self, color, selected, start, extent, *bbox):
        if selected:
            outline = '#004500'
            width = 3
        else:
            outline = 'black'
            width = 1
        self.canvas.create_arc(*tuple(int(x) for x in bbox),
                               start=start,
                               extent=extent,
                               fill=color,
                               outline=outline,
                               width=width)

    def line(self, bbox, width=1):
        self.canvas.create_line(*tuple(int(x) for x in bbox), width=width,
                                fill='black')

    def text(self, x, y, txt, anchor=tk.CENTER, color='black'):
        anchor = {'SE': tk.SE}.get(anchor, anchor)
        self.canvas.create_text((x, y), text=txt, anchor=anchor, fill=color)

    def after(self, time, callback):
        id = self.win.after(int(time * 1000), callback)
        # Quick'n'dirty object with a cancel() method:
        return namedtuple('Timer', 'cancel')(lambda: self.win.after_cancel(id))


def bind_enter(widget, callback):
    """Preferred incantation for binding Return/Enter.

    Bindings work differently on different OSes.  This ensures that
    keypad and normal Return work the same on Linux particularly."""

    widget.bind('<Return>', callback)
    widget.bind('<KP_Enter>', callback)


class TabControl(Widget):
    """A simple tab control widget."""
    def __init__(self, parent, switch_callback):
        self.switch_callback = switch_callback
        self.tabs = {}
        self.notebook = ttk.Notebook(parent)
        # Prevent the Notebook from taking focus so arrow keys are handled
        # by the main window (where atom-move/rotate handlers are bound).
        try:
            self.notebook.configure(takefocus=0)
        except Exception:
            # Older ttk versions may not support takefocus; ignore.
            pass

        # Mapping tab_id -> filepath (for tooltip display)
        self.filepaths = {}
        self._current_hover_index = None
        self._tooltip = None

        # Bind notebook events for tab change and hover handling
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        # Motion and leave for hover-tooltips
        try:
            self.notebook.bind("<Motion>", self._on_motion)
            self.notebook.bind("<Leave>", self._hide_tooltip)
        except Exception:
            pass

    def pack(self, **kwargs):
        self.notebook.pack(**kwargs)

    def add_tab(self, title, filepath=None):
        """Add a new tab with title. Optionally provide filepath for hover tooltip.
        Returns a tab_id integer used by the GUI to reference this tab."""
        frame = tk.Frame(self.notebook)
        try:
            frame.configure(takefocus=0)
        except Exception:
            pass
        self.notebook.add(frame, text=title)
        tab_id = len(self.tabs)
        self.tabs[tab_id] = frame
        if filepath is not None:
            self.filepaths[tab_id] = filepath
        return tab_id

    def _on_tab_change(self, event):
        """Handle tab change events by calling the provided switch callback
        with the currently selected tab id."""
        try:
            # notebook.select() returns internal tab id; index(...) gives integer index
            sel_index = self.notebook.index(self.notebook.select())
        except Exception:
            sel_index = None
        if sel_index is not None and self.switch_callback is not None:
            try:
                self.switch_callback(sel_index)
            except Exception:
                # Don't propagate exceptions from callback
                pass

    def _on_motion(self, event):
        """Show tooltip with full filepath when hovering a tab label."""
        try:
            idx = None
            # Determine tab index at pointer location
            try:
                idx = self.notebook.index(f"@{event.x},{event.y}")
            except Exception:
                idx = None

            if idx is None:
                self._hide_tooltip(event)
                return

            if idx == self._current_hover_index:
                return  # still on same tab

            self._current_hover_index = idx
            self._hide_tooltip(event)

            filepath = self.filepaths.get(idx)
            if not filepath:
                return

            # Create simple tooltip window
            self._tooltip = tk.Toplevel(self.notebook)
            # No window decorations
            try:
                self._tooltip.wm_overrideredirect(True)
            except Exception:
                pass
            lbl = tk.Label(self._tooltip, text=filepath, background="#ffffe0",
                           relief='solid', borderwidth=1, justify='left',
                           font=("TkDefaultFont", 9))
            lbl.pack(ipadx=4, ipady=2)

            # Place tooltip near mouse pointer
            try:
                x = self.notebook.winfo_rootx() + event.x + 12
                y = self.notebook.winfo_rooty() + event.y + 20
                self._tooltip.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass
        except Exception:
            # Guard against any unexpected errors in hover handling
            self._hide_tooltip(event)

    def _hide_tooltip(self, event=None):
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except Exception:
                pass
            self._tooltip = None
        self._current_hover_index = None
