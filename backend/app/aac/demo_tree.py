"""
VoxGaze AAC Demo: Pure Binary Gaze-Tree State Machine (Task D1)

Pure state machine without camera or I/O dependencies.
Navigates an 8-leaf hierarchical binary vocabulary tree loaded from JSON config
using real-timestamped directional control signals (LEFT, RIGHT, CENTER, NONE).

SPECIFICATIONS & INVARIANTS:
1. Pure State Machine: Operates on explicit (timestamp_seconds, direction) stream with no assumed frame rate.
2. Binary Decision Tree: Root -> LEFT=TALK, RIGHT=ALERT & ACTIONS; TALK -> REPLY (YES|NO), NEEDS (I NEED|WATER);
   ALERT & ACTIONS -> ALERT (HELP|PAIN), ACTIONS (SPEAK|UNDO).
3. Arming Requirement: A directional decision counts ONLY after the user was at CENTER for >= CENTER_ARM_MS.
   Prevents one continuous hold from cascading across multiple hierarchy levels.
4. Continuous Hold & Grace Window: Must hold the same direction for >= HOLD_MS. A transient dropout
   or glitch <= GRACE_MS does not reset the hold timer.
5. Mandatory Confirmation: Reaching a leaf enters CONFIRM screen (LEFT=YES, RIGHT=NO).
   Only YES adds the word or executes the action. NO or CONFIRM_TIMEOUT_S returns to ROOT.
6. Safety Abstention: NONE for >= TRACKING_LOSS_MS pauses tracking, disarms, and resets hold.
   UP, DOWN, or invalid inputs never trigger transitions.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class DemoTreeNode:
    """Represents a node in the binary vocabulary tree."""
    label: str
    left: Optional["DemoTreeNode"] = None
    right: Optional["DemoTreeNode"] = None
    word: Optional[str] = None
    action: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.word is not None or self.action is not None or (self.left is None and self.right is None)

    def get_leaves(self) -> List[Dict[str, Any]]:
        """Recursively collects all leaf definitions in the subtree."""
        if self.is_leaf:
            leaf_info: Dict[str, Any] = {"label": self.label}
            if self.word is not None:
                leaf_info["word"] = self.word
                leaf_info["kind"] = "word"
            elif self.action is not None:
                leaf_info["action"] = self.action
                leaf_info["kind"] = "action"
            return [leaf_info]

        leaves = []
        if self.left is not None:
            leaves.extend(self.left.get_leaves())
        if self.right is not None:
            leaves.extend(self.right.get_leaves())
        return leaves


@dataclass(frozen=True)
class DemoScreen:
    """Current presentation state exposed to UI and telemetry."""
    kind: str  # "node" or "confirm"
    title: str
    left_label: str
    right_label: str
    candidate: Optional[str] = None
    candidate_type: Optional[str] = None  # "word" or "action"


@dataclass(frozen=True)
class TreeEvent:
    """Diagnostic event record."""
    t: float
    type: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(self.t, 3), "type": self.type, "detail": self.detail}


def parse_tree_node(data: Dict[str, Any]) -> DemoTreeNode:
    """Recursively constructs a DemoTreeNode hierarchy from JSON dictionary."""
    if "word" in data:
        return DemoTreeNode(
            label=data.get("label", data["word"]),
            word=data["word"],
        )
    if "action" in data:
        return DemoTreeNode(
            label=data.get("label", data["action"]),
            action=data["action"],
        )

    label = data.get("label", "NODE")
    left_node = parse_tree_node(data["left"]) if "left" in data else None
    right_node = parse_tree_node(data["right"]) if "right" in data else None

    # Use child labels if leaf
    if left_node and left_node.is_leaf and not left_node.label:
        left_node.label = left_node.word or left_node.action or "LEFT"
    if right_node and right_node.is_leaf and not right_node.label:
        right_node.label = right_node.word or right_node.action or "RIGHT"

    return DemoTreeNode(
        label=label,
        left=left_node,
        right=right_node,
    )


class DemoStateMachine:
    """
    Pure state machine driving the gaze-tree interface.
    Consumes (timestamp_s, direction) samples and manages navigation, arming,
    hold progress, confirmation, phrase assembly, and tracking safety pauses.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        if config_dict is None:
            config_dict = self._load_config_file(config_path)

        self.config = config_dict
        timing = self.config.get("timing", {})
        self.hold_s: float = float(timing.get("hold_ms", 1000)) / 1000.0
        self.center_arm_s: float = float(timing.get("center_arm_ms", 500)) / 1000.0
        self.grace_s: float = float(timing.get("grace_ms", 150)) / 1000.0
        self.confirm_timeout_s: float = float(timing.get("confirm_timeout_s", 10))
        self.tracking_loss_s: float = float(timing.get("tracking_loss_ms", 300)) / 1000.0

        tree_data = self.config.get("tree", {})
        if not tree_data:
            raise ValueError("Config missing 'tree' definition.")
        self.root_node: DemoTreeNode = parse_tree_node(tree_data)

        # State machine variables
        self.current_node: DemoTreeNode = self.root_node
        self.state: str = "NODE"  # "NODE" or "CONFIRM"
        self.confirm_leaf: Optional[DemoTreeNode] = None
        self.confirm_enter_t: Optional[float] = None

        self.armed: bool = False
        self.center_since: Optional[float] = None
        self.hold_dir: Optional[str] = None
        self.hold_start: Optional[float] = None
        self.last_dir_ts: Optional[float] = None

        self.lost_since: Optional[float] = None
        self.paused: bool = False

        self.phrase: List[str] = []
        self.events: List[TreeEvent] = []
        self._current_t: float = 0.0

    def _load_config_file(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Loads JSON configuration from given path or standard search locations."""
        candidates = []
        if config_path:
            candidates.append(Path(config_path))
        candidates.extend([
            Path(__file__).resolve().parent / "demo_config.json",
            Path(__file__).resolve().parent.parent.parent.parent / "demo_config.json",
            Path("demo_config.json"),
        ])
        for p in candidates:
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(f"Could not locate demo_config.json in search paths: {candidates}")

    def _emit(self, t: float, event_type: str, detail: str = "") -> None:
        self.events.append(TreeEvent(t=t, type=event_type, detail=detail))

    def _clear_hold(self) -> None:
        self.hold_dir = None
        self.hold_start = None
        self.last_dir_ts = None

    def on_sample(self, t: float, direction: str) -> None:
        """
        Processes a single observation sample (t in seconds, direction string).
        direction can be LEFT, RIGHT, CENTER, or NONE.
        """
        self._current_t = t
        d = direction.upper().strip() if direction else "NONE"

        # 1. Check confirm screen timeout
        EPS = 1e-4
        if self.state == "CONFIRM" and self.confirm_enter_t is not None:
            if (t - self.confirm_enter_t) >= (self.confirm_timeout_s - EPS):
                self._emit(t, "timeout", f"Confirmation timed out after {self.confirm_timeout_s}s")
                self._cancel_confirm(t)
                return

        # 2. Handle Tracking Loss (NONE)
        if d == "NONE":
            if self.lost_since is None:
                self.lost_since = t

            if (t - self.lost_since) >= (self.tracking_loss_s - EPS):
                if not self.paused:
                    self.paused = True
                    self._emit(t, "tracking_lost", f"Tracking lost for {self.tracking_loss_s:.2f}s")
                self.armed = False
                self._clear_hold()
                self.center_since = None
            elif self.hold_dir is not None and self.last_dir_ts is not None and (t - self.last_dir_ts) > (self.grace_s + EPS):
                self._emit(t, "hold_cancel", f"{self.hold_dir} dropout exceeded grace")
                self._clear_hold()
            return

        # Tracking restored if it was lost
        if self.paused:
            self.paused = False
            self._emit(t, "tracking_restored", "Tracking resumed")
        self.lost_since = None

        # 3. Handle CENTER (Rest and Arming)
        if d == "CENTER":
            if self.center_since is None:
                self.center_since = t

            if (t - self.center_since) >= (self.center_arm_s - EPS):
                if not self.armed:
                    self.armed = True
                    self._emit(t, "armed", f"Armed at {self.current_node.label}")

            # Check if an active hold expired past grace window
            if self.hold_dir is not None and self.last_dir_ts is not None and (t - self.last_dir_ts) > (self.grace_s + EPS):
                self._emit(t, "hold_cancel", f"{self.hold_dir} hold interrupted by center")
                self._clear_hold()
            return

        # 4. Handle Directional Commands (LEFT / RIGHT)
        if d in ("LEFT", "RIGHT"):
            self.center_since = None

            if not self.armed:
                # Must be armed at center first; ignore directional input
                if self.hold_dir is not None and self.last_dir_ts is not None and (t - self.last_dir_ts) > (self.grace_s + EPS):
                    self._clear_hold()
                return

            # Start or continue hold
            if self.hold_dir != d:
                self.hold_dir = d
                self.hold_start = t
                self._emit(t, "hold_start", d)
            self.last_dir_ts = t

            # Check if hold duration met
            if self.hold_start is not None and (t - self.hold_start) >= (self.hold_s - EPS):
                self._commit(t, d)
                self.armed = False
                self._clear_hold()
            return

        # 5. Unsupported/Ignored directions (e.g. UP, DOWN, ABSTAIN)
        self.center_since = None
        if self.hold_dir is not None and self.last_dir_ts is not None and (t - self.last_dir_ts) > self.grace_s:
            self._clear_hold()

    def _commit(self, t: float, d: str) -> None:
        """Executes the committed directional transition."""
        if self.state == "CONFIRM":
            if d == "LEFT":
                # Confirm YES
                leaf = self.confirm_leaf
                self._emit(t, "confirm_yes", leaf.word or leaf.action or "YES")
                if leaf is not None:
                    if leaf.action == "SPEAK":
                        phrase_str = " ".join(self.phrase)
                        self._emit(t, "speak_request", phrase_str)
                    elif leaf.action == "UNDO":
                        if self.phrase:
                            removed = self.phrase.pop()
                            self._emit(t, "undo", removed)
                        else:
                            self._emit(t, "undo", "EMPTY_PHRASE")
                    elif leaf.word is not None:
                        self.phrase.append(leaf.word)
                        self._emit(t, "word_added", leaf.word)

                self._return_to_root()

            elif d == "RIGHT":
                # Cancel NO
                self._emit(t, "confirm_no", "Cancelled by user")
                self._return_to_root()

        else:
            # Navigation in NODE hierarchy
            self._emit(t, "decision", f"{self.current_node.label} -> {d}")
            child = self.current_node.left if d == "LEFT" else self.current_node.right
            if child is None:
                return

            if child.is_leaf:
                # Reached leaf: enter confirmation screen
                self.state = "CONFIRM"
                self.confirm_leaf = child
                self.confirm_enter_t = t
                cand_name = child.word or child.action or "LEAF"
                self._emit(t, "confirm_screen", f"Confirm {cand_name}")
            else:
                self.current_node = child

    def _cancel_confirm(self, t: float) -> None:
        """Aborts active confirmation screen back to ROOT."""
        self._return_to_root()

    def _return_to_root(self) -> None:
        """Resets hierarchy back to ROOT node while preserving phrase buffer."""
        self.state = "NODE"
        self.current_node = self.root_node
        self.confirm_leaf = None
        self.confirm_enter_t = None
        self.armed = False
        self._clear_hold()
        self.center_since = None

    def reset(self, t: Optional[float] = None) -> None:
        """Explicit escape / reset action (e.g. Esc key)."""
        cur_t = t if t is not None else self._current_t
        self._emit(cur_t, "reset", "Reset to root")
        self._return_to_root()
        self.paused = False
        self.lost_since = None

    @property
    def hold_progress(self) -> float:
        """Normalized progress of current active directional hold [0.0, 1.0]."""
        if self.hold_dir is None or self.hold_start is None or not self.armed:
            return 0.0
        ts = self.last_dir_ts if self.last_dir_ts is not None else self.hold_start
        elapsed = max(0.0, ts - self.hold_start)
        return min(1.0, elapsed / self.hold_s)

    @property
    def current_screen(self) -> DemoScreen:
        """Returns structured metadata of the active UI screen."""
        if self.state == "CONFIRM" and self.confirm_leaf is not None:
            leaf = self.confirm_leaf
            cand_str = leaf.word if leaf.word is not None else leaf.action
            cand_type = "action" if leaf.action is not None else "word"

            if leaf.action == "SPEAK":
                title = f"SPEAK '{' '.join(self.phrase)}'?" if self.phrase else "SPEAK EMPTY PHRASE?"
            elif leaf.action == "UNDO":
                title = f"UNDO '{self.phrase[-1]}'?" if self.phrase else "UNDO (NO WORDS)?"
            else:
                title = f"ADD '{leaf.word}'?"

            return DemoScreen(
                kind="confirm",
                title=title,
                left_label="YES",
                right_label="NO",
                candidate=cand_str,
                candidate_type=cand_type,
            )

        left_lbl = self.current_node.left.label if self.current_node.left else ""
        right_lbl = self.current_node.right.label if self.current_node.right else ""
        return DemoScreen(
            kind="node",
            title=self.current_node.label,
            left_label=left_lbl,
            right_label=right_lbl,
            candidate=None,
            candidate_type=None,
        )

    def get_leaves(self) -> List[Dict[str, Any]]:
        """Returns list of all available leaf items in the configured tree."""
        return self.root_node.get_leaves()

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary matching Section 17.4 /api/demo/state."""
        screen = self.current_screen
        return {
            "screen": {
                "title": screen.title,
                "left_label": screen.left_label,
                "right_label": screen.right_label,
                "kind": screen.kind,
                "candidate": screen.candidate,
                "candidate_type": screen.candidate_type,
            },
            "hold": {
                "direction": self.hold_dir,
                "progress": round(self.hold_progress, 3),
            },
            "armed": self.armed,
            "tracking": "paused" if self.paused else "ok",
            "phrase": list(self.phrase),
            "phrase_text": " ".join(self.phrase),
        }
