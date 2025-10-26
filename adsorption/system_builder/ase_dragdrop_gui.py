"""
Simple drag-and-drop launcher for ASE GUI.

Usage:
  python3 ase_dragdrop_gui.py
Drop one or more structure files onto the window to open them in ASE GUI
(each file will be opened by launching: python -m ase.gui <file>).
"""

import sys
import subprocess
from pathlib import Path

# Try imports in order: PyQt6, PyQt5, PySide6, PySide2
QtWidgets = None
QtCore = None
try:
    from PyQt6 import QtWidgets, QtCore
    from PyQt6.QtGui import QGuiApplication
except Exception:
    try:
        from PyQt5 import QtWidgets, QtCore
    except Exception:
        try:
            from PySide6 import QtWidgets, QtCore
        except Exception:
            try:
                from PySide2 import QtWidgets, QtCore
            except Exception:
                QtWidgets = None

if QtWidgets is None:
    print("Error: PyQt5/6 or PySide2/6 is required. Install with: pip install pyqt5  (or pyqt6/pyside2/pyside6)")
    sys.exit(1)


class DropWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASE Drag & Drop Launcher")
        self.resize(560, 200)
        self.setAcceptDrops(True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        label = QtWidgets.QLabel(
            "Drop structure files here to open them in ASE GUI.\n"
            "Each dropped file will launch: python -m ase.gui <file>\n\n"
            "You can also launch an empty ASE GUI with the button below."
        )
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter if hasattr(QtCore.Qt, "AlignmentFlag") else QtCore.Qt.AlignCenter)
        layout.addWidget(label)

        btn = QtWidgets.QPushButton("Open empty ASE GUI")
        btn.clicked.connect(self.open_empty_ase_gui)
        layout.addWidget(btn)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        # Accept if there are urls (files)
        if mime.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        files = []
        if mime.hasUrls():
            for url in mime.urls():
                # local file paths only
                path = Path(QtCore.QUrl(url).toLocalFile()) if hasattr(QtCore, "QUrl") else Path(url.toLocalFile())
                if path.exists():
                    files.append(str(path))
        if not files:
            self.status.setText("No valid files dropped.")
            return
        self.status.setText(f"Launching ASE GUI for {len(files)} file(s)...")
        for f in files:
            self.launch_ase_for_file(f)
        self.status.setText(f"Launched ASE GUI for {len(files)} file(s).")

    def launch_ase_for_file(self, filepath):
        # Use same Python interpreter
        cmd = [sys.executable, "-m", "ase.gui", filepath]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            self.status.setText(f"Failed to launch ASE GUI: {e}")

    def open_empty_ase_gui(self):
        cmd = [sys.executable, "-m", "ase.gui"]
        try:
            subprocess.Popen(cmd)
            self.status.setText("Opened empty ASE GUI.")
        except Exception as e:
            self.status.setText(f"Failed to launch ASE GUI: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = DropWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
