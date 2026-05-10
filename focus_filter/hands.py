"""
hands.py – Handdetektering och gestanalys för Focus Filter
============================================================
Ansvarar för att köra mediapipe Hands på kameraframes och
tolka resultatet till enkla Python-datastrukturer som resten
av koden kan arbeta med utan att veta om mediapipe.

Innehåller:
    HandDetector – klass som kapslar in mediapipe Hands-sessionen

Hur det hänger ihop:
    main.py skapar ett HandDetector-objekt och anropar
    detector.detect(frame) varje frame. Resultatet är en lista
    med HandData-dicts som GestureState konsumerar i state.py.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import cv2
import mediapipe as mp

from config import (
    DETECT_WIDTH, HANDEDNESS_MIN_SCORE, SMOOTH_ALPHA,
)

_mp_hands = mp.solutions.hands


def _finger_extended(lm, tip: int, pip: int, margin: float = 0.02) -> bool:
    """Kontrollerar om ett finger är utsträckt.

    Jämför fingerspetsens y-koordinat med PIP-ledens (mellanleden).
    Om spetsen är tillräckligt mycket OVANFÖR leden (lägre y = högre
    på skärmen) räknas fingret som utsträckt.

    Args:
        lm    : Landmärks-objekt för en hand.
        tip   : Landmärks-index för fingerspetsen.
        pip   : Landmärks-index för PIP-leden.
        margin: Extra marginal för att undvika falska positiver.

    Returns:
        True om fingret är utsträckt.
    """
    return lm.landmark[tip].y < lm.landmark[pip].y - margin


class HandDetector:
    """Kapslar in mediapipe Hands och returnerar tolkade handdata.

    Kör detection på en nedskalad kopia av frame för bättre prestanda,
    men returnerar koordinater skalade tillbaka till original-upplösningen.
    Koordinater mjukas ut med exponentiell glidande medelvärde (EMA)
    för att minska skakighet.

    Attributes:
        _hands : mediapipe Hands-instans.
        _smooth: Dict med EMA-historik per hand (nyckel = "Label_i").
    """

    def __init__(self) -> None:
        """Skapar mediapipe Hands-sessionen med optimerade inställningar."""
        self._hands = _mp_hands.Hands(
            max_num_hands           = 2,
            model_complexity        = 0,
            min_detection_confidence= 0.5,
            min_tracking_confidence = 0.3,
        )
        self._smooth: dict = {}

    def close(self) -> None:
        """Frigör mediapipe-resurser. Anropas av main.py vid avslut."""
        self._hands.close()

    def detect(self, frame) -> list[dict]:
        """Kör handdetektering på frame och returnerar handdata.

        Varje dict i resultatlistan innehåller:
            idx_ext (bool)  : Pekfingret utsträckt.
            mid_ext (bool)  : Långfingret utsträckt.
            idx     (tuple) : Pekfingertoppens pixelkoordinat (x, y).
            mid     (tuple) : Långfingertoppens pixelkoordinat (x, y).
            idx_n   (tuple) : Normaliserad koordinat för pekfingret (0–1).
            mid_n   (tuple) : Normaliserad koordinat för långfingret (0–1).

        Args:
            frame: BGR-kamerabild.

        Returns:
            Lista med 0–2 hand-dicts. Tom lista = inga händer hittade.
        """
        h, w = frame.shape[:2]

        # Skala ned för snabbare detection
        if w > DETECT_WIDTH:
            scale = DETECT_WIDTH / w
            small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        hands_data: list[dict] = []
        seen_keys: set[str]    = set()

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, (lm, info) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                cls = info.classification[0]
                if cls.score < HANDEDNESS_MIN_SCORE:
                    continue

                key   = f"{cls.label}_{i}"
                idx_n = (lm.landmark[8].x,  lm.landmark[8].y)
                mid_n = (lm.landmark[12].x, lm.landmark[12].y)

                # EMA-utjämning för mjukare koordinater
                prev = self._smooth.get(key)
                if prev is not None:
                    pi, pm = prev
                    idx_n = (
                        SMOOTH_ALPHA * pi[0] + (1 - SMOOTH_ALPHA) * idx_n[0],
                        SMOOTH_ALPHA * pi[1] + (1 - SMOOTH_ALPHA) * idx_n[1],
                    )
                    mid_n = (
                        SMOOTH_ALPHA * pm[0] + (1 - SMOOTH_ALPHA) * mid_n[0],
                        SMOOTH_ALPHA * pm[1] + (1 - SMOOTH_ALPHA) * mid_n[1],
                    )
                self._smooth[key] = (idx_n, mid_n)
                seen_keys.add(key)

                hands_data.append({
                    "idx_ext": _finger_extended(lm, 8,  6),
                    "mid_ext": _finger_extended(lm, 12, 10),
                    "idx"    : (int(idx_n[0] * w), int(idx_n[1] * h)),
                    "mid"    : (int(mid_n[0] * w), int(mid_n[1] * h)),
                    "idx_n"  : idx_n,
                    "mid_n"  : mid_n,
                })

        # Rensa gamla handar ur smooth-historiken
        for k in list(self._smooth):
            if k not in seen_keys:
                del self._smooth[k]

        return hands_data
