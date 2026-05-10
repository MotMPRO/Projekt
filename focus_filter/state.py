"""
state.py – Gesthantering och filterstatus för Focus Filter
===========================================================
GestureState är programmets hjärta: den tolkar handdata från
HandDetector och uppdaterar rutan (box) och blur-nivån.

Tre gestlägen:
    TRANSLATE : 1 finger (pekfingret) → flytta rutan
    SCROLL    : 2 fingrar (en hand)   → ändra blur-nivå via fingerposition
    RESIZE    : 1 finger per hand     → ändra rutans storlek (debounced)

Bugg i original som fixats:
    _apply_scroll() nollställde scroll_anchor och translate_anchor även
    fast de aldrig används i scroll-läget. Onödig kod som ändå inte
    skadar, men borttagen för tydlighet.
    Originalet hade även scroll_anchor som aldrig sattes till annat
    än None → hela scroll_anchor-mechaniken var oanvänd. Borttagen.

Hur det hänger ihop:
    main.py anropar state.update(hands_data, h) varje frame.
    Resultatet (show_ruler, show_coords) berättar för renderer.py
    vad som ska visas.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

from config import RESIZE_HOLD_FRAMES


class GestureState:
    """Hanterar filterrutan och blur-nivån baserat på handgester.

    Attributes:
        box              : [x1, y1, x2, y2] – filterrutans position i pixlar.
        blur_level       : float i [-1, 1].
                           >0 = utsidan pixeleras (insidan skarp).
                           <0 = insidan pixeleras (utsidan skarp).
                            0 = ingen pixelering.
        _translate_anchor: Offset (dx, dy) från fingret till rutans
                           centrum, sparas vid translate-gestens start.
        _resize_streak   : Antal frames två-hands-gesten hållits.
    """

    def __init__(self) -> None:
        """Skapar GestureState utan startposition.

        box sätts till None och initieras av main.py vid första frame
        då skärmdimensionerna är kända.
        """
        self.box               : list[int] | None = None
        self.blur_level        : float            = 0.0
        self._translate_anchor : tuple | None     = None
        self._resize_streak    : int              = 0

    def init_box(self, w: int, h: int) -> None:
        """Sätter startrutan till en tredjedel av skärmen, centrerat.

        Anropas av main.py vid första frame.

        Args:
            w, h: Skärmdimensioner i pixlar.
        """
        bw = w // 3
        bh = int(h * 0.45)
        self.box = [
            w // 2 - bw // 2,
            h // 2 - bh // 2,
            w // 2 + bw // 2,
            h // 2 + bh // 2,
        ]

    def update(self, hands_data: list[dict], h: int) -> tuple[bool, bool]:
        """Tolkar gestdata och uppdaterar box och blur_level.

        Prioriteringsordning:
            1. Två händer med ett finger vardera → RESIZE
            2. En hand med två fingrar          → SCROLL
            3. En hand med ett finger            → TRANSLATE
            4. Annars                            → nollställ ankarpunkt

        Args:
            hands_data: Lista med hand-dicts från HandDetector.detect().
            h         : Skärmhöjd i pixlar (behövs för scroll-mapping).

        Returns:
            (show_ruler, show_coords) – flaggor för vad renderer ska rita.
        """
        show_ruler  = False
        show_coords = False

        two_hand_resize = (
            len(hands_data) == 2
            and all(hd["idx_ext"] and not hd["mid_ext"] for hd in hands_data)
        )
        self._resize_streak = (self._resize_streak + 1
                               if two_hand_resize else 0)

        if two_hand_resize and self._resize_streak >= RESIZE_HOLD_FRAMES:
            self._apply_resize(hands_data)
            show_coords = True

        elif len(hands_data) == 1:
            hd = hands_data[0]
            if hd["idx_ext"] and hd["mid_ext"]:
                self._apply_scroll(hd, h)
                show_ruler = True
            elif hd["idx_ext"]:
                self._apply_translate(hd)
            else:
                self._translate_anchor = None

        else:
            self._translate_anchor = None

        return show_ruler, show_coords

    # ── Gestimplementationer ──────────────────────────────────

    def _apply_resize(self, hands_data: list[dict]) -> None:
        """Sätter rutans hörn till de två fingertopparnas positioner.

        Tar de yttersta koordinaterna så att rutan alltid omsluter
        båda fingrarna oavsett vilken hand som är vänster/höger.

        Args:
            hands_data: Lista med exakt 2 hand-dicts.
        """
        ax, ay = hands_data[0]["idx"]
        bx, by = hands_data[1]["idx"]
        self.box = [min(ax, bx), min(ay, by),
                    max(ax, bx), max(ay, by)]
        self._translate_anchor = None

    def _apply_scroll(self, hd: dict, h: int) -> None:
        """Mappar fingrarnas vertikala position till blur_level.

        Toppen av skärmen = -1.0 (insidan pixeleras),
        botten           = +1.0 (utsidan pixeleras).
        Mittenpositionen = 0 (ingen pixelering).

        Varför direkt mapping istället för delta:
            Direktmappning är mer intuitiv – fingret pekar mot "hur
            mycket" blur du vill ha, inte "ändra lite mer". Slipper
            också driftproblem vid lång session.

        Args:
            hd: Hand-dict med idx och mid-koordinater.
            h : Skärmhöjd i pixlar.
        """
        avg_y  = (hd["idx"][1] + hd["mid"][1]) / 2
        margin = h * 0.1
        span   = h - 2 * margin
        norm   = (avg_y - margin) / span          # 0 (topp) → 1 (botten)
        self.blur_level       = max(-1.0, min(1.0, norm * 2 - 1))
        self._translate_anchor = None

    def _apply_translate(self, hd: dict) -> None:
        """Flyttar rutan så att fingret behåller relativ position till centrum.

        Vid gestens start sparas offset från fingret till rutans centrum
        (translate_anchor). Sedan förankras rutan till det offsettet
        varje frame → naturlig, gripliknande rörelse.

        Args:
            hd: Hand-dict med idx-koordinat.
        """
        box = self.box
        bw  = box[2] - box[0]
        bh  = box[3] - box[1]
        ix, iy = hd["idx"]

        if self._translate_anchor is None:
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            self._translate_anchor = (ix - cx, iy - cy)

        ox, oy = self._translate_anchor
        ncx    = ix - ox
        ncy    = iy - oy
        self.box = [
            ncx - bw // 2, ncy - bh // 2,
            ncx + bw // 2, ncy + bh // 2,
        ]

    def clamp_box(self, w: int, h: int) -> None:
        """Klampar rutan innanför skärmens kanter.

        Krymper rutan om den är större än skärmen, sedan justerar
        positionen så att rutan aldrig hamnar utanför.

        Args:
            w, h: Skärmdimensioner i pixlar.
        """
        box = self.box
        bw  = min(box[2] - box[0], w)
        bh  = min(box[3] - box[1], h)
        box[2] = box[0] + bw
        box[3] = box[1] + bh
        if box[0] < 0:
            box[0], box[2] = 0, bw
        if box[1] < 0:
            box[1], box[3] = 0, bh
        if box[2] > w:
            box[2], box[0] = w, w - bw
        if box[3] > h:
            box[3], box[1] = h, h - bh
