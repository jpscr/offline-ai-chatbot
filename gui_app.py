"""GUI entry point. Run: python gui_app.py

The CLI in main.py still works independently for testing without the GUI.
"""

import sys

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def load_stylesheet(app: QApplication) -> None:
    try:
        with open("style.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass


def main() -> None:
    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
