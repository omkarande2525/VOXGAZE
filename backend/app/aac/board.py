"""
VoxGaze AAC MVP: Word Board & Phrase Builder

Defines standard AAC word tiles, tile coordinate hit-testing,
phrase composition, undo, clear, and phrase formatting for offline speech.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TileCategory(str, Enum):
    PRONOUN = "PRONOUN"
    ACTION = "ACTION"
    NEED = "NEED"
    RESPONSE = "RESPONSE"
    SOCIAL = "SOCIAL"
    CONTROL = "CONTROL"


@dataclass(frozen=True)
class WordTile:
    """Definition of an individual AAC tile."""
    id: str
    text: str
    label: str
    icon: str
    category: TileCategory
    # Normalized bounding box coordinates (0.0 to 1.0)
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0


# Standard 11 core vocabulary words specified by product requirements
CORE_VOCABULARY: List[Tuple[str, str, str, TileCategory]] = [
    ("I", "I", "👤", TileCategory.PRONOUN),
    ("want", "want", "🤲", TileCategory.ACTION),
    ("water", "water", "💧", TileCategory.NEED),
    ("food", "food", "🍲", TileCategory.NEED),
    ("help", "help", "🆘", TileCategory.ACTION),
    ("yes", "yes", "✅", TileCategory.RESPONSE),
    ("no", "no", "❌", TileCategory.RESPONSE),
    ("pain", "pain", "⚡", TileCategory.NEED),
    ("rest", "rest", "🛏️", TileCategory.NEED),
    ("toilet", "toilet", "🚻", TileCategory.NEED),
    ("thank you", "thank you", "🙏", TileCategory.SOCIAL),
]


class PhraseBuilder:
    """Manages the phrase composition buffer with undo and clear functionality."""

    def __init__(self, max_words: int = 20):
        self.max_words = max_words
        self._words: List[str] = []
        self._history: List[List[str]] = [[]]

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def phrase_text(self) -> str:
        """Assembles words into a natural, spoken-ready phrase string."""
        if not self._words:
            return ""
        raw = " ".join(self._words)
        # Capitalize first letter and add period
        return raw[0].upper() + raw[1:] + "."

    @property
    def is_empty(self) -> bool:
        return len(self._words) == 0

    def add_word(self, word: str) -> bool:
        """Appends a word to the active phrase."""
        clean = word.strip()
        if not clean:
            return False
        if len(self._words) >= self.max_words:
            return False
        self._words.append(clean)
        self._history.append(list(self._words))
        return True

    def remove_last(self) -> Optional[str]:
        """Removes the most recent word from the phrase (Backspace/Undo)."""
        if not self._words:
            return None
        removed = self._words.pop()
        self._history.append(list(self._words))
        return removed

    def clear(self) -> None:
        """Clears the entire phrase buffer."""
        if self._words:
            self._words.clear()
            self._history.append([])


class AACBoard:
    """
    Arranges AAC word tiles into a structured accessible grid
    and maps normalized gaze coordinates (x, y in [0, 1]) to tiles.
    """

    def __init__(self):
        self.tiles: List[WordTile] = []
        self.phrase_builder = PhraseBuilder()
        self._init_tiles()

    def _init_tiles(self) -> None:
        """Initializes tiles in a 4-column accessible layout with generous margins."""
        # Standard layout: 4 columns x 3 rows (11 core vocabulary tiles + 1 spacer/action)
        cols = 4
        rows = 3
        # Padding & spacing in normalized screen space
        margin_x = 0.04
        margin_y = 0.12  # Space for top phrase bar
        bottom_margin = 0.12  # Space for bottom status & controls
        avail_w = 1.0 - (2 * margin_x)
        avail_h = 1.0 - margin_y - bottom_margin
        
        cell_w = avail_w / cols
        cell_h = avail_h / rows
        gap_x = cell_w * 0.08
        gap_y = cell_h * 0.10

        tile_w = cell_w - gap_x
        tile_h = cell_h - gap_y

        for idx, (t_id, text, icon, cat) in enumerate(CORE_VOCABULARY):
            r = idx // cols
            c = idx % cols
            x0 = margin_x + (c * cell_w) + (gap_x / 2)
            y0 = margin_y + (r * cell_h) + (gap_y / 2)
            x1 = x0 + tile_w
            y1 = y0 + tile_h

            self.tiles.append(
                WordTile(
                    id=t_id,
                    text=text,
                    label=text.capitalize(),
                    icon=icon,
                    category=cat,
                    x_min=round(x0, 4),
                    y_min=round(y0, 4),
                    x_max=round(x1, 4),
                    y_max=round(y1, 4),
                )
            )

    def find_tile_at(self, norm_x: float, norm_y: float) -> Optional[WordTile]:
        """Returns the tile at normalized coordinate (norm_x, norm_y) if any."""
        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            return None
        for tile in self.tiles:
            if tile.x_min <= norm_x <= tile.x_max and tile.y_min <= norm_y <= tile.y_max:
                return tile
        return None

    def get_tile_by_id(self, tile_id: str) -> Optional[WordTile]:
        """Finds a tile by exact identifier or text."""
        for t in self.tiles:
            if t.id.lower() == tile_id.lower() or t.text.lower() == tile_id.lower():
                return t
        return None
