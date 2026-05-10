"""
words.py – Ordbanker för Body Word Game
========================================
Innehåller alla ord sorterade per ordklass.
Lägg till fler ord direkt i respektive lista – resten sköter sig självt.

Hur det hänger ihop:
    game.py importerar WORDS och ALL_TYPES för att spawna och
    kategorisera fallande ord. Nycklarna i WORDS MÅSTE matcha
    nycklarna i TYPE_COLOR (config.py) för att färgerna ska stämma.

Författare : Amin Bachar
Datum      : 10 maj 2026
"""

# Varje nyckel = en ordklass. Listorna kan utökas fritt.
WORDS : dict[str, list[str]] = {
    "ADJECTIVE": [
        "fast", "big", "small", "cold", "warm", "happy", "sad",
        "strong", "weak", "soft", "hard", "young", "old", "new",
        "funny", "quiet", "bright", "dark", "heavy", "lazy",
        "brave", "clever", "sharp", "smooth", "wild",
    ],
    "NOUN": [
        "dog", "cat", "house", "car", "sun", "moon", "book", "tree",
        "flower", "water", "fire", "air", "earth", "child", "city",
        "bird", "lake", "mountain", "bridge", "school", "door",
        "window", "table", "chair", "storm",
    ],
    "VERB": [
        "run", "jump", "sing", "dance", "write", "read",
        "eat", "sleep", "play", "work", "fly", "swim", "talk",
        "listen", "see", "hear", "think", "laugh", "cry", "smile",
        "rest", "fight", "win", "fall", "move",
    ],
}

# Härledd lista – uppdateras automatiskt om nya klasser läggs till i WORDS.
ALL_TYPES : list[str] = list(WORDS.keys())
