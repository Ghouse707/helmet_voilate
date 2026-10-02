"""
Temporal violation tracker — two-phase design.

Phase 1 — Confirmation
    A NO_HELMET detection must appear N_CONFIRM_FRAMES times within
    a CONFIRM_WINDOW-frame sliding window before it is trusted.

Phase 2 — Collection
    After confirmation the track enters a collection window
    (PLATE_COLLECT_FRAMES long).  During this window the pipeline runs
    plate detection every frame and accumulates the best crop.
    At the end of the window the pipeline finalises ONE database record.

Only one database row is ever written per track, regardless of how many
frames the violation is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

# ── tuneable constants ──────────────────────────────────────────────────────
CONFIRM_WINDOW       = 15   # frames: sliding window for confirmation
N_CONFIRM_FRAMES     = 4    # detections needed within that window
MAX_GAP              = 90   # frames before a non-confirmed track is discarded
PLATE_COLLECT_FRAMES = 60   # frames to search for the best plate after confirm

# ── status tokens returned by process_nohel() ──────────────────────────────
STATUS_PENDING    = "PENDING"     # confirmation still accumulating
STATUS_CONFIRMED  = "CONFIRMED"   # just crossed threshold — start collecting
STATUS_COLLECTING = "COLLECTING"  # inside the collection window
STATUS_EXPIRED    = "EXPIRED"     # collection window just ended — save now
STATUS_DONE       = "DONE"        # already saved — ignore


@dataclass
class TrackState:
    center: tuple[float, float]
    last_seen_frame: int
    # -- confirmation bookkeeping --
    last_nohel_frame: int = -9999
    nohel_window_start: int = 0
    nohel_count: int = 0
    # -- collection bookkeeping --
    confirmed: bool = False
    confirmed_frame: int = 0
    # -- terminal state --
    saved: bool = False


class EventTracker:
    """
    Thread-unsafe (single-threaded pipeline use only).

    Typical call pattern per frame
    --------------------------------
    for each NO_HELMET detection:
        status, key = tracker.process_nohel(track_id, center, frame_num, conf)
        if status == STATUS_CONFIRMED:
            active_collections[key] = CollectionData(...)
        elif status == STATUS_COLLECTING and key in active_collections:
            active_collections[key].candidate_box = updated_box

    # run plate detection for every active collection (caller's job)

    for key in tracker.check_expired(frame_num):
        cdata = active_collections.pop(key, None)
        if cdata:
            _finalize_and_save(cdata)
        tracker.mark_saved(key)
    """

    def __init__(
        self,
        max_distance:       float = 150,
        max_gap:            int   = MAX_GAP,
        confirm_window:     int   = CONFIRM_WINDOW,
        n_confirm:          int   = N_CONFIRM_FRAMES,
        collect_frames:     int   = PLATE_COLLECT_FRAMES,
    ):
        self.max_distance   = max_distance
        self.max_gap        = max_gap
        self.confirm_window = confirm_window
        self.n_confirm      = n_confirm
        self.collect_frames = collect_frames

        self._tracks:    dict[str, TrackState] = {}
        self._id_to_key: dict[int, str]         = {}

    # ── internal helpers ────────────────────────────────────────────────────

    def _expire_old(self, frame_num: int) -> None:
        """Remove stale *unconfirmed* tracks to keep the dict small."""
        stale = [
            k for k, s in self._tracks.items()
            if not s.confirmed and frame_num - s.last_seen_frame > self.max_gap
        ]
        for k in stale:
            del self._tracks[k]
        self._id_to_key = {
            tid: key for tid, key in self._id_to_key.items()
            if key in self._tracks
        }

    def _resolve_key(
        self,
        track_id,
        center: tuple[float, float],
        frame_num: int,
    ) -> str:
        """
        Return the canonical track key for this detection.

        Priority:
        1. Existing ByteTrack ID mapping
        2. Nearest spatial match within max_distance
        3. New key based on track_id or position
        """
        key = self._id_to_key.get(track_id) if track_id is not None else None

        if key is None or key not in self._tracks:
            neighbours = []
            for k, state in self._tracks.items():
                gap = frame_num - state.last_seen_frame
                if gap <= self.max_gap:
                    dist = hypot(
                        center[0] - state.center[0],
                        center[1] - state.center[1],
                    )
                    if dist <= self.max_distance:
                        neighbours.append((dist, k))
            if neighbours:
                key = min(neighbours)[1]
            elif track_id is not None:
                key = f"id:{track_id}"
            else:
                key = f"xy:{frame_num}:{center[0]:.0f}:{center[1]:.0f}"

        if track_id is not None:
            self._id_to_key[track_id] = key

        return key

    # ── public API ──────────────────────────────────────────────────────────

    def process_nohel(
        self,
        track_id,
        center: tuple[float, float],
        frame_num: int,
        conf: float = 1.0,
    ) -> tuple[str, str]:
        """
        Register one NO_HELMET detection and return *(status, track_key)*.

        Call once per NO_HELMET detection per frame.
        """
        self._expire_old(frame_num)
        key = self._resolve_key(track_id, center, frame_num)
        old = self._tracks.get(key)

        # Confirmation is a one-time transition. Later frames only refresh the
        # active rider position; they do not count the same violation again.
        if old is not None and old.confirmed:
            old.center = center
            old.last_seen_frame = frame_num
            if old.saved:
                return STATUS_DONE, key
            if frame_num - old.confirmed_frame <= self.collect_frames:
                return STATUS_COLLECTING, key
            return STATUS_EXPIRED, key

        # ── Build updated state ─────────────────────────────────────────────
        if old is None:
            state = TrackState(
                center=center,
                last_seen_frame=frame_num,
                last_nohel_frame=frame_num,
                nohel_window_start=frame_num,
                nohel_count=1,
            )
        else:
            frames_since_nohel = frame_num - old.last_nohel_frame
            if frames_since_nohel <= self.confirm_window:
                nohel_count   = old.nohel_count + 1
                window_start  = old.nohel_window_start
            else:
                # Gap too wide — restart window
                nohel_count   = 1
                window_start  = frame_num

            state = TrackState(
                center=center,
                last_seen_frame=frame_num,
                last_nohel_frame=frame_num,
                nohel_window_start=window_start,
                nohel_count=nohel_count,
                confirmed=old.confirmed,
                confirmed_frame=old.confirmed_frame,
                saved=old.saved,
            )

        self._tracks[key] = state

        # ── Status determination ────────────────────────────────────────────
        if state.saved:
            return STATUS_DONE, key

        if state.confirmed:
            # Already past threshold — are we still inside the collection window?
            if frame_num - state.confirmed_frame <= self.collect_frames:
                return STATUS_COLLECTING, key
            # Collection window passed (rare: track kept emitting NO_HELMET after expiry)
            return STATUS_EXPIRED, key

        if state.nohel_count >= self.n_confirm:
            # Just crossed the threshold for the first time
            state.confirmed       = True
            state.confirmed_frame = frame_num
            return STATUS_CONFIRMED, key

        return STATUS_PENDING, key

    def update_seen(
        self,
        track_id,
        center: tuple[float, float],
        frame_num: int,
    ) -> None:
        """
        Update position for a track that is NOT a NO_HELMET detection
        (helmeted riders, generic persons).  Keeps the spatial index
        fresh for association but does NOT increment the violation counter.
        """
        self._expire_old(frame_num)
        key = self._resolve_key(track_id, center, frame_num)
        old = self._tracks.get(key)
        if old is None:
            self._tracks[key] = TrackState(center=center, last_seen_frame=frame_num)
        else:
            old.center          = center
            old.last_seen_frame = frame_num

    def check_expired(self, frame_num: int) -> list[str]:
        """
        Return the keys of all confirmed-but-unsaved tracks whose
        plate-collection window has just expired.

        Call every frame so tracks are finalised even when the rider
        disappears from the scene.
        """
        return [
            key for key, state in self._tracks.items()
            if (state.confirmed
                and not state.saved
                and frame_num - state.confirmed_frame > self.collect_frames)
        ]

    def get_all_pending(self) -> list[str]:
        """Return every confirmed-but-unsaved track (used at end-of-video)."""
        return [
            key for key, state in self._tracks.items()
            if state.confirmed and not state.saved
        ]

    def mark_saved(self, key: str) -> None:
        """Permanently mark a track so it is never saved again."""
        if key in self._tracks:
            self._tracks[key].saved = True
