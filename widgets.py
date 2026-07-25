"""Reusable widgets: chat message bubbles and telemetry metric bars."""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

import config


class ChatBubble(QFrame):
    """One chat message. `source_tag` ("local docs"/"world knowledge") is
    shown as a badge on assistant messages only."""

    def __init__(
        self,
        text: str,
        role: str,
        source_tag: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("chatBubbleUser" if role == "user" else "chatBubbleAssistant")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMaximumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        if source_tag:
            tag_label = QLabel(source_tag.upper())
            tag_label.setObjectName(
                "sourceTagLocal" if source_tag == "local docs" else "sourceTagWorld"
            )
            layout.addWidget(tag_label)

        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(message_label)


class MetricBar(QWidget):
    """Labelled progress bar that recolors (via a QSS dynamic property) as
    the value crosses warning/critical thresholds."""

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.bar = QProgressBar()
        self.bar.setObjectName("metricBar")
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)

    def set_value(self, percent: float) -> None:
        percent = max(0.0, min(100.0, percent))
        self.bar.setValue(int(percent))
        self.value_label.setText(f"{percent:.0f}%")

        if percent >= config.METRIC_CRITICAL_THRESHOLD:
            level = "critical"
        elif percent >= config.METRIC_WARNING_THRESHOLD:
            level = "warning"
        else:
            level = "good"

        self.bar.setProperty("level", level)
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)
