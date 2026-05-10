# Amin Bachar – Gestigenkänning med Python och MediaPipe

**Namn:** Amin Bachar  
**Datum:** 10 maj 2026

---

## Om projektet

Det här projektet använder en webbkamera och MediaPipe för att styra saker med kroppen och händerna i realtid. Jag har byggt tre separata program som alla bygger på samma idé – kameran ser dig och du styr saker med rörelser.

---

## Två program

### 1. Body Word Game (`body_word_game/`)
Ett ordklassspel där ord faller ner på skärmen och du fångar rätt ord med händerna, armbågarna eller fötterna. MediaPipe Pose detekterar hela kroppen. Rätt ordklass ger poäng och en ny uppgift.

**Kontroller:**
| Knapp | Funktion |
|-------|----------|
| `Q` | Avsluta |
| `R` | Starta om |

---

### 2. Focus Filter (`focus_filter/`)
Ett filter där du styr en pixelerings-ruta med handen. Du kan flytta rutan, ändra storlek och justera hur stark pixeleringen är – allt med handrörelser framför kameran.

**Gester:**
| Gest | Funktion |
|------|----------|
| 1 finger (pekfinger) | Flytta rutan |
| 2 fingrar (pekfinger + långfinger) | Justera blur-nivå |
| 1 finger per hand (2 händer) | Ändra storlek på rutan |

**Kontroller:**
| Knapp | Funktion |
|-------|----------|
| `Q` | Avsluta |

---

## Hur man kör programmen

Varje program körs för sig från sin egen mapp.

```bash
# Body Word Game
cd body_word_game
python main.py

# Focus Filter
cd focus_filter
python main.py
```

---

## Vad som behövs installeras

```bash
pip install opencv-python mediapipe numpy
```

---

## Filstruktur

```
body_word_game/
    main.py       – startar spelet, kameran och loopen
    game.py       – spellogik, poäng och kollision
    renderer.py   – all ritning med OpenCV
    config.py     – alla inställningar
    words.py      – alla ord sorterade per ordklass

focus_filter/
    main.py       – startar filtret och kameran
    state.py      – gestlogik och filterrutan
    hands.py      – handdetektering med MediaPipe
    renderer.py   – pixelering, linjal och ritfunktioner
    config.py     – alla inställningar
```
