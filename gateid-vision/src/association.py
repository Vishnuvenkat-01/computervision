"""
Geometric and spatial person-to-card association engine for GateID Vision.

Associates detected ID cards with tracked people using anatomical torso heuristics
and crowd overlap ambiguity detection.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents an axis-aligned bounding box [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def intersection_area(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def containment_ratio(self, other: "BoundingBox") -> float:
        """Calculates what fraction of `other` is contained within `self`."""
        if other.area <= 0.0:
            return 0.0
        return self.intersection_area(other) / other.area


@dataclass
class TrackedPerson:
    """Represents a person detected and tracked across frames."""
    track_id: int
    bbox: BoundingBox
    confidence: float


@dataclass
class DetectedCard:
    """Represents a college ID card detected by the custom detector."""
    card_id: int
    bbox: BoundingBox
    confidence: float


@dataclass
class AssociationMatch:
    """Result of associating a tracked person with candidate ID cards."""
    track_id: int
    status: str  # "associated", "unassociated", "ambiguous_overlap"
    card: Optional[DetectedCard] = None
    association_confidence: float = 0.0
    reason: str = ""
    candidate_card_ids: List[int] = field(default_factory=list)


class AssociationEngine:
    """
    Associates detected college ID cards with tracked people.

    Limitations:
    - Bounding boxes approximate 3D human anatomy in a 2D camera plane.
    - When multiple people walk shoulder-to-shoulder, their 2D torso regions
      may intersect, causing card attribution ambiguity. The engine explicitly
      identifies and flags these cases rather than making an arbitrary guess.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("association", {})
        self.torso_y_min_ratio = float(cfg.get("torso_y_min_ratio", 0.15))
        self.torso_y_max_ratio = float(cfg.get("torso_y_max_ratio", 0.65))
        self.torso_x_margin_ratio = float(cfg.get("torso_x_margin_ratio", 0.10))
        self.ambiguity_iou_threshold = float(cfg.get("ambiguity_iou_threshold", 0.20))

    def compute_torso_region(self, person_box: BoundingBox) -> BoundingBox:
        """
        Calculates the anatomical chest/torso boundary for a person.
        Standard college lanyards and badges reside between the neck (~15% height)
        and upper waist (~65% height).
        """
        w = person_box.width
        h = person_box.height
        margin_x = w * self.torso_x_margin_ratio

        tx1 = person_box.x1 - margin_x
        tx2 = person_box.x2 + margin_x
        ty1 = person_box.y1 + (h * self.torso_y_min_ratio)
        ty2 = person_box.y1 + (h * self.torso_y_max_ratio)

        return BoundingBox(x1=tx1, y1=ty1, x2=tx2, y2=ty2)

    def associate(
        self,
        people: List[TrackedPerson],
        cards: List[DetectedCard],
    ) -> Dict[int, AssociationMatch]:
        """
        Associates each tracked person with detected ID cards.

        Returns a dictionary mapping track_id -> AssociationMatch.
        """
        results: Dict[int, AssociationMatch] = {}

        if not people:
            return results

        # 1. Compute torso boxes for all tracked people
        torso_map: Dict[int, BoundingBox] = {
            p.track_id: self.compute_torso_region(p.bbox) for p in people
        }

        # 2. For each card, find all candidate people whose torso covers it
        # Map: card_id -> list of (track_id, containment_ratio, is_center_inside)
        card_to_people: Dict[int, List[Tuple[int, float, bool]]] = {
            c.card_id: [] for c in cards
        }

        for card in cards:
            cx, cy = card.bbox.center
            for person in people:
                torso = torso_map[person.track_id]
                containment = torso.containment_ratio(card.bbox)
                center_inside = torso.contains_point(cx, cy)

                # Consider a plausible match if card center is inside or containment > 25%
                if center_inside or containment >= 0.25:
                    card_to_people[card.card_id].append(
                        (person.track_id, containment, center_inside)
                    )

        # 3. Detect ambiguity: cards plausible for more than one person in crowd overlap
        ambiguous_cards: Dict[int, List[int]] = {}
        valid_card_assignments: Dict[int, int] = {}  # card_id -> track_id

        for card in cards:
            candidates = card_to_people[card.card_id]
            if len(candidates) > 1:
                # Multiple people overlap this card's torso position
                # Check if overlap is high between candidate torsos
                candidate_ids = [c[0] for c in candidates]
                ambiguous_cards[card.card_id] = candidate_ids
                logger.debug(
                    "Card %s is ambiguous across candidate persons: %s",
                    card.card_id,
                    candidate_ids,
                )
            elif len(candidates) == 1:
                track_id, containment, center_inside = candidates[0]
                valid_card_assignments[card.card_id] = track_id

        # 4. Invert to track_id -> assigned cards / ambiguous cards
        person_cards: Dict[int, List[DetectedCard]] = {p.track_id: [] for p in people}
        person_ambiguous_cards: Dict[int, List[int]] = {p.track_id: [] for p in people}

        card_dict = {c.card_id: c for c in cards}

        for card_id, track_id in valid_card_assignments.items():
            person_cards[track_id].append(card_dict[card_id])

        for card_id, cand_ids in ambiguous_cards.items():
            for tid in cand_ids:
                if tid in person_ambiguous_cards:
                    person_ambiguous_cards[tid].append(card_id)

        # 5. Build final association match for each person
        for person in people:
            tid = person.track_id
            assigned = person_cards[tid]
            ambig = person_ambiguous_cards[tid]

            if assigned:
                # Select highest confidence card if multiple associated
                best_card = max(assigned, key=lambda c: c.confidence)
                torso = torso_map[tid]
                assoc_conf = min(
                    person.confidence,
                    best_card.confidence,
                    max(0.5, torso.containment_ratio(best_card.bbox)),
                )
                results[tid] = AssociationMatch(
                    track_id=tid,
                    status="associated",
                    card=best_card,
                    association_confidence=float(assoc_conf),
                    reason=f"ID card {best_card.card_id} geometrically verified in torso region",
                    candidate_card_ids=[c.card_id for c in assigned],
                )
            elif ambig:
                results[tid] = AssociationMatch(
                    track_id=tid,
                    status="ambiguous_overlap",
                    card=None,
                    association_confidence=0.0,
                    reason=f"Card detection {ambig} overlaps multiple individuals in crowd",
                    candidate_card_ids=ambig,
                )
            else:
                results[tid] = AssociationMatch(
                    track_id=tid,
                    status="unassociated",
                    card=None,
                    association_confidence=0.0,
                    reason="No ID card detected in torso region",
                    candidate_card_ids=[],
                )

        return results
