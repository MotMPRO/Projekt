"""
renderer.py – Renderingsfunktioner för Focus Filter
=====================================================
All OpenCV-ritning som inte tillhör en specifik entitet.
Modulen är helt utan sidoeffekter på spelstatus – den tar
emot data och ritar, inget mer.

Innehåller:
    pixelate()         – pixelerar ett bildblock
    render_frame()     – applicerar pixelering på rätt delar av bilden
    draw_box()         – ritar filterrutans kant + hörnmarkörer
    draw_fingertips()  – ritar synliga fingertoppar
    draw_ruler()       – vertikal linjal med blur-markör och status-pill
    draw_coord_label() – koordinatetikett vid en fingertipp

Hur det hänger ihop:
    main.py anropar dessa funktioner i rätt ordning varje frame
    baserat på flaggorna från GestureState.update().

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  BILDFILTER
# ─────────────────────────────────────────────────────────────

def pixelate(img: np.ndarray, strength: float) -> np.ndarray:
    """Pixelerar en bild genom att krympa och förstora med INTER_NEAREST.

    Metoden är snabb och ger karaktäristisk pixelering utan blur.
    Strength 0 = ingen effekt, 1 = max pixelering (4 block).

    Args:
        img     : BGR-bildarray.
        strength: Pixeleringsgrad 0.0–1.0.

    Returns:
        Ny pixelerad bild med samma dimensioner som img.
    """
    if strength <= 0.02:
        return img
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    # Färre block = grövre pixelering
    blocks = max(4, int(80 - strength * 75))
    small  = cv2.resize(img, (blocks, max(1, int(blocks * h / w))),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def render_frame(
    frame     : np.ndarray,
    box       : list[int],
    blur_level: float,
) -> np.ndarray:
    """Applicerar pixelering på utsidan, insidan eller båda.

    blur_level > 0 → utsidan pixeleras (insidan är fokus-rutan)
    blur_level < 0 → insidan pixeleras (utsidan är fokus)
    blur_level = 0 → originalbild

    Varför två separata inside/outside: det gör det möjligt att
    ha blur på båda sidor simultant om blur_level justeras till
    ett mixat läge i framtiden.

    Args:
        frame     : BGR-kamerabild (modifieras EJ – kopia returneras).
        box       : [x1, y1, x2, y2] – filterrutans position.
        blur_level: float i [-1, 1].

    Returns:
        Ny bildbuffer med pixelering applicerad.
    """
    x1, y1, x2, y2 = box
    inside  = max(0.0,  blur_level)
    outside = max(0.0, -blur_level)

    if outside > 0.02:
        output = pixelate(frame, outside)
        if inside > 0.02:
            output[y1:y2, x1:x2] = pixelate(frame[y1:y2, x1:x2], inside)
        else:
            output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    else:
        if inside > 0.02:
            output = frame.copy()
            output[y1:y2, x1:x2] = pixelate(frame[y1:y2, x1:x2], inside)
        else:
            output = frame
    return output


# ─────────────────────────────────────────────────────────────
#  RUTANS KANT
# ─────────────────────────────────────────────────────────────

def draw_box(output: np.ndarray, box: list[int]) -> None:
    """Ritar filterrutans kant och hörnmarkörer.

    Tunna vita linjer + svarta cirklar med vita centra i hörnen
    ger ett rent, tydligt utseende mot alla bakgrunder.

    Args:
        output: BGR-bildbuffer (ändras in-place).
        box   : [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(output, (cx, cy), 7, (0, 0, 0), 2)
        cv2.circle(output, (cx, cy), 5, (255, 255, 255), -1)


# ─────────────────────────────────────────────────────────────
#  FINGERTOPPAR
# ─────────────────────────────────────────────────────────────

def draw_fingertips(output: np.ndarray, hands_data: list[dict]) -> None:
    """Ritar vita punkter på utstrackta fingertoppar.

    Visuell feedback som bekräftar att gesten detekterats korrekt.

    Args:
        output    : BGR-bildbuffer (ändras in-place).
        hands_data: Lista med hand-dicts från HandDetector.
    """
    for hd in hands_data:
        if hd["idx_ext"]:
            cv2.circle(output, hd["idx"], 9, (0, 0, 0), 2)
            cv2.circle(output, hd["idx"], 5, (255, 255, 255), -1)
        if hd["mid_ext"]:
            cv2.circle(output, hd["mid"], 9, (0, 0, 0), 2)
            cv2.circle(output, hd["mid"], 5, (255, 255, 255), -1)


# ─────────────────────────────────────────────────────────────
#  LINJAL
# ─────────────────────────────────────────────────────────────

def _status_info(level: float) -> tuple[tuple, str]:
    """Beräknar färg och text för status-pillret baserat på blur_level.

    Grön = svag effekt, gul = medium, röd = stark.

    Args:
        level: blur_level i [-1, 1].

    Returns:
        (BGR-färg, statustext)
    """
    if level > 0.05:
        return _level_color_text(level, "BLUR")
    if level < -0.05:
        return _level_color_text(-level, "FOCUS")
    return (80, 220, 80), "NEUTRAL"


def _level_color_text(magnitude: float, label: str) -> tuple[tuple, str]:
    """Väljer färg (grön/gul/röd) efter styrka och bygger procenttext.

    Args:
        magnitude: Effektens styrka (0–1).
        label    : "BLUR" eller "FOCUS".

    Returns:
        (BGR-färg, "LABEL XX%")
    """
    pct  = int(magnitude * 100)
    text = f"{label} {pct}%"
    if magnitude < 0.33:
        return (80, 220, 80),  text   # Grön
    if magnitude < 0.66:
        return (40, 220, 240), text   # Gul
    return (60, 60, 240), text        # Röd


def _rounded_rect(
    img      : np.ndarray,
    p1       : tuple[int, int],
    p2       : tuple[int, int],
    color    : tuple,
    radius   : int = 12,
    thickness: int = -1,
) -> None:
    """Ritar en rundad rektangel (fylld eller bara kant).

    Används för status-pillret och koordinatetiketterna.
    Implementationen kombinerar rektanglar och ellipsbågar.

    Args:
        img      : BGR-bildbuffer (ändras in-place).
        p1, p2   : Övre vänstra / nedre högra hörnet.
        color    : BGR-färg.
        radius   : Hörnradie i pixlar.
        thickness: -1 för fylld, >0 för kantlinje.
    """
    x1, y1 = p1
    x2, y2 = p2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius),
                180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius),
                270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),
                90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),
                0,   0, 90, color, thickness)


def draw_ruler(img: np.ndarray, level: float) -> None:
    """Ritar vertikal linjal med rörlig markör och status-pill.

    Linjalen visas till vänster i bild under scroll-gesten.
    Markörens position på linjalen speglar blur_level direkt.

    Args:
        img  : BGR-bildbuffer (ändras in-place).
        level: blur_level i [-1, 1].
    """
    h  = img.shape[0]
    rx = 55
    top = int(h * 0.18)
    bot = int(h * 0.82)
    mid = (top + bot) // 2

    # Linjalen
    cv2.line(img, (rx, top), (rx, bot), (255, 255, 255), 2)
    cv2.line(img, (rx - 12, top), (rx + 12, top), (255, 255, 255), 2)
    cv2.line(img, (rx - 12, bot), (rx + 12, bot), (255, 255, 255), 2)
    cv2.line(img, (rx - 8,  mid), (rx + 8,  mid), (255, 255, 255), 1)
    for i in range(1, 10):
        ty = top + (bot - top) * i // 10
        tw = 6 if i % 5 else 9
        cv2.line(img, (rx - tw, ty), (rx + tw, ty), (255, 255, 255), 1)

    # Rörlig markör
    marker_y = int(mid + level * (bot - top) / 2)
    cv2.circle(img, (rx, marker_y), 12, (0, 0, 0), -1)
    cv2.circle(img, (rx, marker_y), 9,  (255, 255, 255), -1)

    # Status-pill
    color, text = _status_info(level)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad          = 10
    bx1 = rx + 22
    by1 = marker_y - th // 2 - pad
    bx2 = bx1 + tw + 2 * pad
    by2 = marker_y + th // 2 + pad
    _rounded_rect(img, (bx1, by1), (bx2, by2), color,    radius=14, thickness=-1)
    _rounded_rect(img, (bx1, by1), (bx2, by2), (255, 255, 255), radius=14, thickness=1)
    cv2.putText(img, text, (bx1 + pad, marker_y + th // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  KOORDINATETIKETT
# ─────────────────────────────────────────────────────────────

def draw_coord_label(
    img  : np.ndarray,
    point: tuple[int, int],
    nx   : float,
    ny   : float,
) -> None:
    """Ritar x/y-koordinat i en rundad ruta bredvid en fingertipp.

    Används vid resize-gesten för att ge feedback om fingrarnas
    exakta normaliserade position.

    Args:
        img  : BGR-bildbuffer (ändras in-place).
        point: Pixelkoordinat att placera etiketten bredvid.
        nx,ny: Normaliserade koordinater att visa (0.0–1.0).
    """
    text = f"x:{nx:.2f}  y:{ny:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad  = 8
    px, py = point
    bx1  = px + 14
    by1  = py - th // 2 - pad
    bx2  = bx1 + tw + 2 * pad
    by2  = py + th // 2 + pad
    _rounded_rect(img, (bx1, by1), (bx2, by2), (0, 0, 0),
                  radius=10, thickness=-1)
    _rounded_rect(img, (bx1, by1), (bx2, by2), (255, 255, 255),
                  radius=10, thickness=1)
    cv2.putText(img, text, (bx1 + pad, py + th // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
