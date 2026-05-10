"""
game.py – Spellogik för Body Word Game
========================================
Innehåller all spellogik: spawn, fångst, poäng och HUD-rendering.
Håller ingen kamerakod – det sköter main.py.

Innehåller:
    distance_factor() – skattar spelarens avstånd via axelbredd
    Game              – huvudklass som äger spelstatus

Hur det hänger ihop:
    main.py skapar ett Game-objekt och anropar per frame:
        game.check_catch(pose_lm, w, h, dfactor)
        game.update(dt, w, h, dfactor)
        game.draw_hud(frame, dfactor)

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import random
import time

import mediapipe as mp
import numpy as np

from config import (
    BASE_CATCH_RADIUS, BASE_INTERVAL, BASE_SPEED,
    CATCH_LM_IDS, REF_SHOULDER_WIDTH, TYPE_COLOR,
)
from renderer import FallingWord, draw_panel, draw_text
from words import ALL_TYPES, WORDS

_pose = mp.solutions.pose


# ─────────────────────────────────────────────────────────────
#  AVSTÅNDSSKATTNING
# ─────────────────────────────────────────────────────────────

def distance_factor(pose_lm) -> float:
    """Skattar spelarens avstånd till kameran via normaliserad axelbredd.

    Mediapipe ger koordinater i 0–1. Ju närmre kameran, desto bredare
    verkar axlarna. Faktorn 1.0 = referensavstånd (REF_SHOULDER_WIDTH).
    Resultatet klipps till [0.3, 2.5] för att undvika extremvärden som
    bryter spelbalansen.

    Varför axelbredd: det är den mest stabila mätpunkten på kroppen –
    inte lika känslig för rotation som t.ex. nosen.

    Args:
        pose_lm: mediapipe PoseLandmarks-objekt.

    Returns:
        float – >1.0 nära kameran, <1.0 långt bort, 1.0 vid fel.
    """
    lm = pose_lm.landmark
    ls = lm[_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[_pose.PoseLandmark.RIGHT_SHOULDER]
    if ls.visibility < 0.5 or rs.visibility < 0.5:
        return 1.0   # Axlarna syns inte – fall tillbaka på neutralt värde
    shoulder_w = abs(rs.x - ls.x)
    return max(0.3, min(2.5, shoulder_w / REF_SHOULDER_WIDTH))


# ─────────────────────────────────────────────────────────────
#  SPELKLASS
# ─────────────────────────────────────────────────────────────

class Game:
    """Hanterar all spelstatus och logik för Body Word Game.

    Ansvarar för:
        - Spawna fallande ord med rätt timing och svårighet
        - Detektera när spelaren fångar ett ord (kollision)
        - Räkna poäng och byta quest-klass
        - Rita hela HUD:en (ord, panel, feedback, avstånd)

    Attributes:
        score      : Aktuell poäng.
        quest      : Ordklass spelaren ska fånga just nu.
        words      : Lista med aktiva FallingWord-instanser.
        last_spawn : Unix-tid för senaste spawn.
        fb_text    : Feedbacktext (visas efter fångst).
        fb_color   : BGR-färg för feedbacktexten.
        fb_timer   : Sekunder kvar att visa feedbacken.
    """

    def __init__(self) -> None:
        """Skapar spelet och startar direkt med reset()."""
        self.reset()

    def reset(self) -> None:
        """Återställer all speldata till startläge.

        Nollställer poäng, väljer ny slumpmässig quest och rensar
        alla aktiva ord och feedback. Anropas av R-tangenten i main.py.
        """
        self.score      : int              = 0
        self.quest      : str              = random.choice(ALL_TYPES)
        self.words      : list[FallingWord] = []
        self.last_spawn : float            = time.time()
        self.fb_text    : str              = ""
        self.fb_color   : tuple            = (255, 255, 255)
        self.fb_timer   : float            = 0.0

    # ── Spawn ─────────────────────────────────────────────────

    def _maybe_spawn(self, w: int, h: int, dfactor: float) -> None:
        """Skapar ett nytt fallande ord om det är dags.

        Spawn-intervallet minskar när dfactor är hög (spelaren nära)
        → svårare. Quest-ordklassen spawnar 40% av gångerna för att
        spelaren alltid ska ha något att sikta på utan att det blir
        för lätt.

        Args:
            w, h    : Skärmdimensioner i pixlar.
            dfactor : Avståndsfaktor från distance_factor().
        """
        interval = BASE_INTERVAL / max(0.3, dfactor)
        if time.time() - self.last_spawn < interval:
            return
        self.last_spawn = time.time()

        wtype  = self.quest if random.random() < 0.40 else random.choice(ALL_TYPES)
        text   = random.choice(WORDS[wtype])
        speed  = BASE_SPEED * max(0.4, dfactor) * random.uniform(0.85, 1.15)
        scale  = 0.82 * max(0.55, dfactor)
        margin = 90
        # Bugfix: skydda mot för smal skärm (t.ex. kamerafel) → hoppa spawn
        if w <= margin * 2:
            return
        x = random.randint(margin, w - margin)
        self.words.append(FallingWord(text, wtype, x, speed, scale, h))

    # ── Fångst ────────────────────────────────────────────────

    def check_catch(
        self,
        pose_lm,
        w       : int,
        h       : int,
        dfactor : float,
    ) -> None:
        """Testar om en kroppspunkt rör vid ett fallande ord.

        Cirkulär kollision: om avståndet² < radie² är ordet fångat.
        Radien skalas med dfactor så att avstånd till kameran inte
        ger orättvis fördel (nära kamera → fysiskt mindre rörelseyta).

        Args:
            pose_lm : mediapipe PoseLandmarks.
            w, h    : Skärmdimensioner i pixlar.
            dfactor : Avståndsfaktor från distance_factor().
        """
        catch_r = int(BASE_CATCH_RADIUS / max(0.4, dfactor))
        lm      = pose_lm.landmark

        # Bygg lista med pixelpositioner för synliga fångstpunkter
        catchers: list[tuple[int, int]] = [
            (int(lm[lm_id].x * w), int(lm[lm_id].y * h))
            for lm_id in CATCH_LM_IDS
            if lm[lm_id].visibility > 0.35
        ]

        for word in self.words:
            if word.caught:
                continue
            wx, wy = word.x, int(word.y)
            for cx, cy in catchers:
                if (cx - wx) ** 2 + (cy - wy) ** 2 < catch_r ** 2:
                    self._register_catch(word)
                    break   # Bara en fångst per ord per frame

    def _register_catch(self, word: FallingWord) -> None:
        """Hanterar ett fångat ord: poäng, feedback och ny quest.

        Rätt ordklass → +1 poäng, ny quest (aldrig samma som förra).
        Fel ordklass  → feedback utan poängändring.

        Args:
            word: Det FallingWord som spelaren precis rörde.
        """
        word.caught = True
        word.flash  = 0.75

        if word.wtype == self.quest:
            self.score   += 1
            self.fb_text  = f"+1  '{word.text}' is a {word.wtype}!"
            self.fb_color = (50, 255, 100)
            # Ny quest – alltid en ANNAN klass än den precis fångade.
            # Bugfix: om bara en ordklass finns (words.py saknar andra)
            # behålls nuvarande quest istället för att krascha med choice([]).
            others = [q for q in ALL_TYPES if q != self.quest]
            self.quest = random.choice(others) if others else self.quest
        else:
            self.fb_text  = f"WRONG!  '{word.text}' is a {word.wtype}"
            self.fb_color = (60, 80, 255)

        self.fb_timer = 2.2

    # ── Uppdatering ───────────────────────────────────────────

    def update(self, dt: float, w: int, h: int, dfactor: float) -> None:
        """Uppdaterar all spellogik för en frame.

        Ordning:
            1. Försök spawna nytt ord.
            2. Flytta alla ord nedåt.
            3. Ta bort ord som är klara (fallit utanför eller fångade).
            4. Räkna ned feedback-timern.

        Args:
            dt      : Sekunder sedan förra frame.
            w, h    : Skärmdimensioner i pixlar.
            dfactor : Avståndsfaktor från distance_factor().
        """
        self._maybe_spawn(w, h, dfactor)
        for word in self.words:
            word.update(dt)
        self.words = [wo for wo in self.words if not wo.done()]
        if self.fb_timer > 0:
            self.fb_timer -= dt

    # ── HUD ───────────────────────────────────────────────────

    def draw_hud(self, frame: np.ndarray, dfactor: float) -> None:
        """Ritar hela spelets HUD ovanpå kamerabilden.

        Lager (bakre → främre):
            1. Fallande ord
            2. Mörk toppanel
            3. Quest-text (mitten), poäng (höger), avstånd (vänster)
            4. Feedbacktext med fade-effekt
            5. Mörk bottompanel med kontrollhints

        Args:
            frame   : BGR-bildbuffer (ändras in-place).
            dfactor : Avståndsfaktor – styr avståndsstatus-texten.
        """
        h, w = frame.shape[:2]

        # Ord
        for word in self.words:
            word.draw(frame, self.quest)

        # Toppanel
        draw_panel(frame, 0, 0, w, 68)
        quest_col = TYPE_COLOR.get(self.quest, (255, 255, 100))
        draw_text(frame, f"CATCH: {self.quest}",
                  (w // 2 - 150, 50), 1.1, quest_col, stroke=4)
        draw_text(frame, f"Score: {self.score}",
                  (w - 200, 48), 0.9, (255, 255, 255), stroke=2)

        # Avståndsstatus
        if dfactor > 1.35:
            dist_str, dist_col = "CLOSE >> step back!", (60, 60, 255)
        elif dfactor < 0.72:
            dist_str, dist_col = "FAR << step closer", (255, 165, 60)
        else:
            dist_str, dist_col = "Good distance", (60, 220, 60)
        draw_text(frame, dist_str, (16, 48), 0.52, dist_col, stroke=2)

        # Feedback med fade (timern används som alpha)
        if self.fb_timer > 0 and self.fb_text:
            fade = min(1.0, self.fb_timer)
            col  = tuple(int(c * fade) for c in self.fb_color)
            draw_text(frame, self.fb_text,
                      (w // 2 - 280, h // 2 + 50), 0.9, col, stroke=3)

        # Bottompanel
        draw_panel(frame, 0, h - 30, w, h, alpha=0.55)
        draw_text(frame,
                  "Catch with hands / elbows / feet  |  Q = quit  |  R = restart",
                  (14, h - 10), 0.42, (190, 190, 190), stroke=2)
