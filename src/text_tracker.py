"""
src/text_tracker.py
Multi-box IoU text tracker linking text regions across frames and identifying best frames.
"""

import logging
from typing import List, Optional, Set, Dict

from src.models import TrackedBox, TextEvent

logger = logging.getLogger(__name__)

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

class TextTracker:
    """
    Multi-box IoU tracker for text regions across consecutive video frames.
    """
    def __init__(self, iou_threshold: float = 0.5, max_gap_seconds: float = 0.5):
        self.iou_threshold = iou_threshold
        self.max_gap_seconds = max_gap_seconds
        self.next_track_id = 1
        self.active_tracks: List[TextEvent] = []
        self.completed_tracks: List[TextEvent] = []

    def update(self, timestamp: float, detected_boxes: List[TrackedBox]):
        """
        Update active tracks with new detected boxes at timestamp.
        Enforces one-to-one assignment and ages out tracks where gap > max_gap_seconds.
        """
        # Age out inactive tracks
        still_active: List[TextEvent] = []
        for track in self.active_tracks:
            if timestamp - track.end > self.max_gap_seconds:
                self.completed_tracks.append(track)
            else:
                still_active.append(track)
        self.active_tracks = still_active

        assigned_tracks: Set[int] = set()

        for box in detected_boxes:
            best_match_track: Optional[TextEvent] = None
            best_iou = 0.0

            for track in self.active_tracks:
                if track.track_id in assigned_tracks:
                    continue
                if not track.boxes:
                    continue
                
                last_box = track.boxes[-1]
                iou = compute_iou(box.bbox, last_box.bbox)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_track = track

            if best_match_track is not None:
                best_match_track.boxes.append(box)
                best_match_track.end = timestamp
                assigned_tracks.add(best_match_track.track_id)
            else:
                # Create new track
                new_track = TextEvent(
                    track_id=self.next_track_id,
                    start=timestamp,
                    end=timestamp,
                    boxes=[box],
                    best_frame_timestamp=timestamp
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)
                assigned_tracks.add(new_track.track_id)

    def update_best_frame(self, track_id: int, timestamp: float, text: str, query_relevance: float):
        """
        Update ONLY the specified track_id with its best frame by query relevance (ocr_conf * text_sim).
        """
        for track in self.active_tracks + self.completed_tracks:
            if track.track_id == track_id:
                if query_relevance > track.best_relevance:
                    track.best_relevance = query_relevance
                    track.best_frame_timestamp = timestamp
                    track.best_text = text
                break

    def finalize(self) -> List[TextEvent]:
        """
        Finalize all tracks and return complete list of tracked text events.
        """
        all_tracks = self.completed_tracks + self.active_tracks
        self.completed_tracks = []
        self.active_tracks = []
        return all_tracks
