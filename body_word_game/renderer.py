"""
renderer.py – Renderingslager för Body Word Game
=================================================
All OpenCV-ritning är samlad här. Modulen vet ingenting om spellogik –
den tar emot data och ritar. Det gör det enkelt att byta ut utseendet
utan att röra spelet.

Innehåller:
    draw_text()     – text med kontur
    draw_panel()    – halvtransparent bakgrundsruta
    draw_skeleton() – mediapipe-skelett på kamerabilden

Klasser:
    FallingWord – ett fallande ord med egen position, hastighet och ritning

Hur det hänger ihop:
    game.py skapar FallingWord-instanser och anropar draw_skeleton().
    main.py anropar draw_text() för "ingen kropp"-meddelandet.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import cv2
import mediapipe as mp
import numpy as np

from config import (
    BONE_COLOR, BONE_THICKNESS, JOINT_R,
    CATCH_LM_IDS, BASE_CATCH_RADIUS,
    TYPE_COLOR, FONT,
)

_pose = mp.solutions.pose


# ─────────────────────────────────────────────────────────────
#  PRIMITIVER
# ─────────────────────────────────────────────────────────────

def draw_text(
    frame : np.ndarray,
    text  : str,
    pos   : tuple[int, int],
    scale : float,
    color : tuple[int, int, int],
    stroke: int = 3,
) -> None:
    """Ritar text med svart kontur så att den syns mot alla bakgrunder.

    Varför dubbel putText: första anropet ritar en tjock svart "skugga",
    andra anropet ritar den tunna färgade texten ovanpå. Ger läsbarhet
    oavsett om bakgrunden är ljus eller mörk.

    Args:
        frame  : BGR-bildbuffer (ändras in-place).
        text   : Strängen att visa.
        pos    : (x, y) – textens nedre vänstra hörn.
        scale  : Textstorlek som multiplikator.
        color  : BGR-färg för texten.
        stroke : Extra tjocklek på konturen (px).
    """
    thick = max(1, int(scale * 2))
    cv2.putText(frame, text, pos, FONT, scale,
                (0, 0, 0), thick + stroke, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale,
                color, thick, cv2.LINE_AA)


def draw_panel(
    frame : np.ndarray,
    x1    : int,
    y1    : int,
    x2    : int,
    y2    : int,
    alpha : float = 0.6,
) -> None:
    """Ritar en halvtransparent svart rektangel (HUD-bakgrund).

    Används för topp- och bottompanelerna. Arbetar direkt på ROI-slicen
    för att slippa kopiera hela bilden.

    Args:
        frame       : BGR-bildbuffer (ändras in-place).
        x1,y1,x2,y2: Rektangelns hörn i pixlar.
        alpha       : Hur mörk panelen är (0 = genomskinlig, 1 = svart).
    """
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, dst=roi)


# ─────────────────────────────────────────────────────────────
#  SKELETT
# ─────────────────────────────────────────────────────────────

def draw_skeleton(
    frame    : np.ndarray,
    landmarks,
    w        : int,
    h        : int,
) -> None:
    """Ritar spelarens skelett (ben, leder, fångstcirklar) på bilden.

    Varje led ritas med svart yttre cirkel + vit inre cirkel för kontrast.
    Fångstpunkter (handleder, armbågar, vrister) markeras med en extra
    cyan ring vars radie = halva fångstradiusen, som visuell feedback.

    Args:
        frame    : BGR-bildbuffer (ändras in-place).
        landmarks: mediapipe PoseLandmarks från senaste frame.
        w, h     : Bildens bredd och höjd i pixlar.
    """
    lm = landmarks.landmark

    # Ben – linjer mellan landmärken enligt mediapipes anslutningsschema
    for a_id, b_id in _pose.POSE_CONNECTIONS:
        a, b = lm[a_id], lm[b_id]
        if a.visibility < 0.35 or b.visibility < 0.35:
            continue
        ax, ay = int(a.x * w), int(a.y * h)
        bx, by = int(b.x * w), int(b.y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 0, 0),
                 BONE_THICKNESS + 3, cv2.LINE_AA)   # Svart kontur
        cv2.line(frame, (ax, ay), (bx, by), BONE_COLOR,
                 BONE_THICKNESS, cv2.LINE_AA)         # Vit linje

    # Leder – fylld cirkel vid varje synligt landmärke
    for pt in lm:
        if pt.visibility < 0.35:
            continue
        px, py = int(pt.x * w), int(pt.y * h)
        cv2.circle(frame, (px, py), JOINT_R + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), JOINT_R, (255, 255, 255), -1, cv2.LINE_AA)

    # Fångstcirklar – cyan ring på de aktiva fångstpunkterna
    for lm_id in CATCH_LM_IDS:
        pt = lm[lm_id]
        if pt.visibility > 0.35:
            px, py = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (px, py),
                       int(BASE_CATCH_RADIUS * 0.45),
                       (0, 220, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  FALLANDE ORD
# ─────────────────────────────────────────────────────────────

class FallingWord:
    """Representerar ett ord som faller nedåt och kan fångas av spelaren.

    Varje instans hanterar sin egen rörelse, kollisionsyta och rendering.
    Game-klassen skapar och lagrar instanserna; denna klass vet ingenting
    om spelregler – bara om sin egen position och hur den ritas.

    Attributes:
        text    : Ordets textsträng.
        wtype   : Ordklass ("ADJECTIVE", "NOUN", "VERB").
        x       : Horisontell position (px), oförändrad under fallet.
        y       : Vertikal position (px), ökar varje frame.
        speed   : Fallhastighet i px/sek.
        scale   : Textstorlek-multiplikator.
        frame_h : Skärmhöjd – behövs för done()-kontrollen.
        caught  : True när spelaren har fångat ordet.
        flash   : Sekunder kvar av fångst-animation.
    """

    def __init__(
        self,
        text    : str,
        wtype   : str,
        x       : int,
        speed   : float,
        scale   : float,
        frame_h : int,
    ) -> None:
        """Skapar ett nytt fallande ord ovanför skärmens överkant.

        Startar på y = -50 (utanför skärmen) så att det glider in
        naturligt utan att "poppa" fram.
        """
        self.text    = text
        self.wtype   = wtype
        self.x       = x
        self.y       = float(-50)
        self.speed   = speed
        self.scale   = scale
        self.frame_h = frame_h
        self.caught  = False
        self.flash   = 0.0

    def update(self, dt: float) -> None:
        """Uppdaterar position och flash-timer en frame framåt.

        Fångade ord slutar falla men flash-timern räknas ner
        tills animationen är klar och done() returnerar True.

        Args:
            dt: Tid sedan förra frame i sekunder.
        """
        if not self.caught:
            self.y += self.speed * dt
        if self.flash > 0:
            self.flash -= dt

    def done(self) -> bool:
        """Returnerar True om ordet ska tas bort från listan.

        Två fall: antingen har det fallit nedanför skärmen,
        eller har det fångats och flash-animationen är slut.
        """
        return self.y > self.frame_h + 80 or (self.caught and self.flash <= 0)

    def draw(self, frame: np.ndarray, quest: str) -> None:
        """Ritar ordet på bilden, anpassat efter om det är quest-typen.

        Quest-ord: ljusa, lite större, grön glow-bakgrund.
        Övriga ord: grå, lite mindre.
        Fångade ord: grön flash-text ersätter normalvisningen.

        Args:
            frame : BGR-bildbuffer (ändras in-place).
            quest : Den ordklass spelaren ska fånga just nu.
        """
        x, y     = self.x, int(self.y)
        is_quest = (self.wtype == quest)
        col      = (255, 255, 255) if is_quest else (160, 160, 160)
        scale    = self.scale * (1.1 if is_quest else 0.82)

        # Fångst-flash: byt ut hela ordet mot en grön bekräftelse
        if self.caught and self.flash > 0:
            draw_text(frame, "CAUGHT! " + self.text.upper(),
                      (x - 60, y), scale * 1.25, (50, 255, 80), stroke=4)
            return

        # Grön glow-rektangel bakom quest-ord för att de ska sticka ut.
        # Bugfix: använd samma tjocklek som draw_text för korrekt boxstorlek.
        if is_quest:
            thick = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(self.text, FONT, scale, thick + 3)
            pad = 7
            ov  = frame.copy()
            cv2.rectangle(ov,
                          (x - pad,      y - th - pad),
                          (x + tw + pad, y + pad),
                          (0, 50, 0), -1)
            cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)

        # Själva ordtexten
        draw_text(frame, self.text, (x, y), scale, col,
                  stroke=3 if is_quest else 2)

        # Liten ordklassetikett under ordet
        tc = TYPE_COLOR.get(self.wtype, (200, 200, 200))
        draw_text(frame, self.wtype, (x, y + 16),
                  max(0.28, scale * 0.4), tc, stroke=2)
