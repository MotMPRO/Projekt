"""
main.py – Startpunkt för Focus Filter
=======================================
Öppnar kameran, kör handdetektering och ritar filtret varje frame.
All gest- och filterlogik delegeras till GestureState (state.py),
HandDetector (hands.py) och renderer.py.

Gester:
    1 finger (pekfinger)          → flytta rutan
    2 fingrar (pekfinger + lång)  → justera blur-nivå (scroll)
    1 finger per hand             → resize (håll 3 frames)

Kontroller:
    Q – avsluta

Hur det hänger ihop:
    main.py äger kameran och fönstret. Allt annat delegeras.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import cv2

from config import CAM_FPS, CAM_HEIGHT, CAM_WIDTH, WINDOW_NAME
from hands import HandDetector
from renderer import (
    draw_box, draw_coord_label, draw_fingertips,
    draw_ruler, render_frame,
)
from state import GestureState


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    """Öppnar webbkameran med specificerade dimensioner och FPS.

    Args:
        index: Kameraindex (0 = standard).

    Returns:
        Öppnat VideoCapture-objekt.

    Raises:
        SystemExit: Om kameran inte kan öppnas.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        raise SystemExit(1)
    return cap


def _run_loop(
    cap     : cv2.VideoCapture,
    detector: HandDetector,
    state   : GestureState,
) -> None:
    """Kör huvudloopen tills Q trycks.

    Per frame:
        1. Läs och spegla kamerabild.
        2. Initiera box vid första frame.
        3. Detektera händer.
        4. Uppdatera geststate.
        5. Rendera filter + UI-element.
        6. Visa fönster.

    Args:
        cap     : Öppnat VideoCapture-objekt.
        detector: Initierad HandDetector.
        state   : Initierad GestureState.
    """
    initialized = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Initiera box en gång vid första frame (dimensioner nu kända)
        if not initialized:
            state.init_box(w, h)
            initialized = True

        # Handdetektering
        hands_data = detector.detect(frame)

        # Gestlogik
        show_ruler, show_coords = state.update(hands_data, h)
        state.clamp_box(w, h)

        # Rendera pixelering
        output = render_frame(frame, state.box, state.blur_level)

        # Ruta + fingertoppar (alltid synliga)
        draw_box(output, state.box)
        draw_fingertips(output, hands_data)

        # Kontextuella UI-element
        if show_ruler:
            draw_ruler(output, state.blur_level)
        if show_coords:
            for hd in hands_data:
                if hd["idx_ext"]:
                    draw_coord_label(output, hd["idx"],
                                     hd["idx_n"][0], hd["idx_n"][1])

        cv2.imshow(WINDOW_NAME, output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    """Initierar alla komponenter och startar Focus Filter."""
    cap      = _open_camera()
    detector = HandDetector()
    state    = GestureState()

    print("[INFO] Focus Filter startar – Q = avsluta")
    print("[INFO] 1 finger = flytta | 2 fingrar = blur | 2 händer = resize")

    try:
        _run_loop(cap, detector, state)
    finally:
        # finally garanterar att resurser frigörs även vid krasch
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Avslutad")


if __name__ == "__main__":
    main()
