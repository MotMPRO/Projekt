"""
config.py – Konfiguration för Focus Filter
===========================================
Alla konstanter samlade på ett ställe. Justera värdena här för att
ändra hur mjuk handspårningen är, hur många frames en gest måste
hållas för att aktiveras, och vid vilken upplösning detection sker.

Hur det hänger ihop:
    Alla andra moduler importerar härifrån.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

# ── Kamera ────────────────────────────────────────────────────
CAM_WIDTH  : int = 640
CAM_HEIGHT : int = 480
CAM_FPS    : int = 30
WINDOW_NAME: str = "Focus Filter"

# ── Handspårning ──────────────────────────────────────────────
# Interpolationsfaktor för att mjuka ut koordinater (0 = ingen mjukning,
# 1 = ingen rörelse alls). Originalet hade 0.7 vilket kan kännas tregt;
# justera nedåt för snabbare respons.
SMOOTH_ALPHA         : float = 0.7

# Minimalt confidence-poäng för att lita på "vänster/höger"-klassningen.
HANDEDNESS_MIN_SCORE : float = 0.8

# Antal frames två-hands-resize-gesten måste hållas innan den aktiveras.
# Förhindrar oavsiktlig resize vid snabba rörelser.
RESIZE_HOLD_FRAMES   : int   = 3

# Upplösning som mediapipe detection körs på. Lägre = snabbare,
# men kan missa händer som är långt från kameran.
DETECT_WIDTH         : int   = 320
