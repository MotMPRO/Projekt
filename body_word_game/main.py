"""
main.py – Startpunkt för Body Word Game
=========================================
Initierar kamera och MediaPipe Pose, kör huvudloopen och
stänger av allt korrekt vid avslut.

Hur det hänger ihop:
    main.py känner till kamera + fönster. Spellogik och rendering
    delegeras helt till Game (game.py) och draw_skeleton / draw_text
    (renderer.py). Tanken: byt ut main.py om du vill köra på
    en annan plattform – resten av koden är orörd.

Kontroller:
    Q – avsluta
    R – starta om

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

import time

import cv2
import mediapipe as mp

from config import CAM_FPS, CAM_HEIGHT, CAM_WIDTH, FONT
from game import Game, distance_factor
from renderer import draw_skeleton, draw_text


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    """Öppnar och konfigurerar webbkameran.

    Använder DirectShow + MJPG för låg latens på Windows.
    På Linux/macOS ignoreras dessa flaggor tyst av OpenCV.

    Args:
        index: Kameraindex (0 = standard).

    Returns:
        Öppnat VideoCapture-objekt.

    Raises:
        SystemExit: Om kameran inte kan öppnas.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        raise SystemExit(1)
    return cap


def _run_loop(cap: cv2.VideoCapture, game: Game, pose) -> None:
    """Kör spelets huvudloop tills Q trycks.

    Per frame:
        1. Läs + spegla kamerabilden.
        2. Kör mediapipe Pose-detection.
        3. Om kropp hittad: uppdatera dfactor, rita skelett, kontrollera fångst.
        4. Uppdatera spellogik och rita HUD.
        5. Visa fönster och hantera tangentbord.

    dt (tidsdelta) begränsas till 0.1 s för att undvika hopp om
    en frame tar ovanligt lång tid (t.ex. vid bakgrundslast).

    Args:
        cap  : Öppnat VideoCapture-objekt.
        game : Initierad Game-instans.
        pose : Aktivt mediapipe Pose-kontextobjekt.
    """
    prev_time  = time.time()
    dfac       = 1.0
    read_fails = 0   # Räknar på varandra följande kamerafel

    while True:
        ok, frame = cap.read()
        if not ok:
            # Bugfix: räkna misslyckade läsningar – avsluta vid ihållande fel
            # för att undvika oändlig loop om kameran kopplas ur.
            read_fails += 1
            if read_fails > 30:
                print("[ERROR] Kameran svarar inte – avslutar")
                break
            continue
        read_fails = 0   # Återställ räknaren vid lyckad läsning

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        now       = time.time()
        dt        = min(now - prev_time, 0.1)
        prev_time = now

        # Pose-detection (mediapipe kräver RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)

        if result.pose_landmarks:
            dfac = distance_factor(result.pose_landmarks)
            draw_skeleton(frame, result.pose_landmarks, w, h)
            game.check_catch(result.pose_landmarks, w, h, dfac)
        else:
            draw_text(frame,
                      "Stand in front of the camera – upper body must be visible",
                      (w // 2 - 370, h // 2), 0.72, (100, 200, 255), stroke=3)

        game.update(dt, w, h, dfac)
        game.draw_hud(frame, dfac)

        cv2.imshow("Body Word Game – Catch the right word class!", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            game.reset()
            print("[INFO] Game restarted")


def main() -> None:
    """Initierar alla komponenter och startar spelet.

    Skapar kamera, Game och mediapipe Pose-session,
    kör loopen och frigör resurser vid avslut.
    """
    cap  = _open_camera()
    game = Game()

    with mp.solutions.pose.Pose(
        model_complexity         = 0,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.4,
    ) as pose:
        print("[INFO] Starting – stand in front of the camera!")
        print("[INFO] Q = quit  |  R = restart")
        _run_loop(cap, game, pose)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Game over – final score: {game.score}")


if __name__ == "__main__":
    main()
