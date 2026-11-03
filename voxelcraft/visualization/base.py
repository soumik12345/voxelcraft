"""Shared lifecycle and local dataset navigation for Gradio visualization apps."""

from abc import ABC, abstractmethod
from pathlib import Path
from random import choice
from typing import Any

import gradio as gr
from gradio.themes import Base as BaseTheme


class BaseVisualizationApp(ABC):
    """Extend discovery and content creation to add another 3D dataset viewer.

    Instances hold dataset configuration and bounded read caches, never a user's
    selection. Gradio component values provide the per-session state.
    """

    title = "3D dataset viewer"
    description = "Explore objects and their associated metadata."
    css = """
    .gradio-container { max-width: 1500px !important; margin: auto; }
    .gradio-container h1 { font-size: 1.75rem; letter-spacing: -0.025em; }
    .gradio-container .prose { overflow-wrap: anywhere; }
    #object-view { min-height: 440px; }
    @media (max-width: 640px) { #object-view { min-height: 340px; } }
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise ValueError(f"Dataset directory does not exist: {self.data_dir}")
        self.sample_ids = tuple(self.discover_samples())
        self._sample_id_set = frozenset(self.sample_ids)
        self._app: gr.Blocks | None = None

    @abstractmethod
    def discover_samples(self) -> list[str]:
        """Return stable sample identifiers relative to ``data_dir``."""

    @abstractmethod
    def create_content(self, app: gr.Blocks) -> None:
        """Create dataset controls and callbacks inside the active Blocks context."""

    def sample_directory(self, sample_id: str) -> Path:
        """Resolve only indexed samples; reject traversal and escaping symlinks."""
        if sample_id not in self._sample_id_set:
            raise ValueError("Unknown sample. Select a house from the list.")
        directory = (self.data_dir / sample_id).resolve()
        if not directory.is_relative_to(self.data_dir) or not directory.is_dir():
            raise ValueError("Sample directory is unavailable or outside the dataset.")
        return directory

    def adjacent_sample(self, sample_id: str | None, offset: int) -> str | None:
        if not self.sample_ids:
            return None
        index = (
            self.sample_ids.index(sample_id) if sample_id in self._sample_id_set else 0
        )
        return self.sample_ids[(index + offset) % len(self.sample_ids)]

    def random_sample(self) -> str | None:
        return choice(self.sample_ids) if self.sample_ids else None

    def build(self) -> gr.Blocks:
        """Build once, allowing embedding, testing, or a subsequent ``launch``."""
        if self._app is None:
            with gr.Blocks(
                title=self.title, analytics_enabled=False, delete_cache=(3600, 3600)
            ) as app:
                gr.Markdown(f"# {self.title}\n\n{self.description}")
                self.create_content(app)
            self._app = app
        return self._app

    def launch(self, **kwargs: Any) -> Any:
        """Launch locally by default; callers can override Gradio launch options."""
        options = {
            "server_name": "127.0.0.1",
            "share": False,
            "allowed_paths": [str(self.data_dir)],
            "theme": BaseTheme(primary_hue="emerald", neutral_hue="slate"),
            "css": self.css,
        }
        options.update(kwargs)
        return self.build().queue(default_concurrency_limit=2).launch(**options)
