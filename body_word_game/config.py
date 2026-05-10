"""
config.py – Konfiguration för Body Word Game
=============================================
Samlar ALLA konstanter på ett ställe så att du aldrig behöver leta
i spellogiken när du vill justera hastighet, storlek eller färger.

Hur det hänger ihop:
    Alla andra moduler importerar härifrån. Ingen annan modul
    definierar egna konstanter – allt styrs härifrån.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import cv2
import mediapipe as mp

# ── Kamera ────────────────────────────────────────────────────
CAM_WIDTH  : int = 960   # Önskad kamerabredd i pixlar
CAM_HEIGHT : int = 540   # Önskad kamerahöjd i pixlar
CAM_FPS    : int = 30    # Bildfrekvens

# ── Spelbalans ────────────────────────────────────────────────
BASE_SPEED        : float = 130.0  # px/sek vid referensavstånd
BASE_INTERVAL     : float = 2.0    # sekunder mellan nytt ord vid referensavstånd
BASE_CATCH_RADIUS : int   = 58     # fångstradie i px vid referensavstånd

# ── Avståndskalibrering ───────────────────────────────────────
# Normaliserad axelbredd (0–1) som mediapipe rapporterar på "bra" avstånd.
# Ju närmre kameran, desto bredare axlar → högre dfactor → snabbare spel.
REF_SHOULDER_WIDTH : float = 0.27

# ── Skelett-rendering ─────────────────────────────────────────
BONE_COLOR     : tuple = (255, 255, 255)  # Vit (BGR)
BONE_THICKNESS : int   = 2                # Linjetjocklek i px
JOINT_R        : int   = 6               # Ledkulornas radier i px

# ── Ordtypsfärger (BGR) ───────────────────────────────────────
# Nycklarna MÅSTE matcha nycklarna i words.py → WORDS.
# Bugg i original: svenska nycklar matchade inte engelska WORDS-nycklar
# vilket gjorde att inga färger visades. Fixat här.
TYPE_COLOR : dict = {
    "ADJECTIVE" : (80, 200, 255),   # Ljusblå
    "NOUN"      : (80, 255, 160),   # Mintgrön
    "VERB"      : (80, 130, 255),   # Lila-blå
}

# ── Fångstpunkter ─────────────────────────────────────────────
# Vilka kroppspunkter som aktivt fångar ord.
# Handleder, armbågar och vrister ger en naturlig rörelseyta.
_PL = mp.solutions.pose.PoseLandmark
CATCH_LM_IDS : list = [
    _PL.LEFT_WRIST,
    _PL.RIGHT_WRIST,
    _PL.LEFT_ELBOW,
    _PL.RIGHT_ELBOW,
    _PL.LEFT_ANKLE,
    _PL.RIGHT_ANKLE,
]

# ── Typsnitt ──────────────────────────────────────────────────
FONT : int = cv2.FONT_HERSHEY_DUPLEX
