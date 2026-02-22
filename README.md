# Audio Assembly — Automatyczny montaż audio (PL)

Narzędzie CLI + Web UI do automatycznego montażu plików audio w języku polskim.
Usuwa wypełniacze mowy (yyy, eee), skraca zbyt długie pauzy i wykrywa powtórzenia/restarty zdań.

## Funkcje

- **Web UI** — drag & drop, podgląd na żywo, odtwarzacz przed/po, pobieranie wyników
- **CLI** — pełna kontrola z wiersza poleceń
- **Tryb A: CLEAN_FILLERS_AND_PAUSES** — usuwa wypełniacze (yyy, eee, em, ymm...) i skraca pauzy
- **Tryb B: REMOVE_REPETITIONS_CONTEXTUAL** — wykrywa i usuwa powtórzenia słów, fraz i restarty zdań
- Trzy tryby maskowania cięć: **ROOMTONE**, **FADE**, **CUT**
- Transkrypcja PL z timestampami na poziomie słów (faster-whisper)
- Raport JSON (EDL) z listą wszystkich operacji
- Transkrypcja z oznaczonymi fragmentami usuniętymi

## Wymagania

- Python 3.9+
- libsndfile (dla soundfile)
- ffmpeg (opcjonalnie, do obsługi MP3/M4A)

## Instalacja

```bash
pip install numpy scipy soundfile click rich

# Opcjonalnie: transkrypcja Whisper
pip install faster-whisper
```

Lub z setuptools:

```bash
pip install -e .

# Z transkrypcją:
pip install -e ".[transcription]"
```

## Użycie

### Web UI (zalecane)

```bash
python run_web.py
# lub z podaniem portu:
python run_web.py 8080
```

Otwórz http://localhost:5000 w przeglądarce. Interfejs umożliwia:
- Przeciągnięcie pliku audio (drag & drop)
- Wybór trybu (A/B) i wszystkich parametrów
- Podgląd postępu w czasie rzeczywistym (SSE)
- Porównanie oryginału z wynikiem (odtwarzacz)
- Tabelę edycji ze statusami i timestampami
- Pobranie wszystkich plików wyjściowych

### CLI

### Tryb A — Czyszczenie wypełniaczy i pauz

```bash
python -m audio_assembly wywiad.wav --mode A
```

### Tryb B — Usuwanie powtórzeń

```bash
python -m audio_assembly wywiad.wav --mode B
```

### Pełna konfiguracja

```bash
python -m audio_assembly wywiad.wav \
    --mode A \
    --output-dir ./wynik/ \
    --silence-threshold 3.0 \
    --min-gap 120 \
    --crossfade 60 \
    --soften-mode ROOMTONE \
    --soften-ms 120 \
    --max-pause 0.4 \
    --whisper-model base \
    --verbose
```

### Obróbka fragmentu nagrania

```bash
python -m audio_assembly wywiad.wav --mode A \
    --start-offset 60.0 \
    --end-offset 755.0
```

### Tryb ścisły (pomija niepewne edycje)

```bash
python -m audio_assembly wywiad.wav --mode B --strict --confidence-threshold 0.8
```

## Parametry

| Parametr | Domyślnie | Opis |
|---|---|---|
| `--mode` | A | Tryb: A=wypełniacze+pauzy, B=powtórzenia |
| `--output-dir` | ./output/ | Katalog wyjściowy |
| `--silence-threshold` | 3.0 | Próg ciszy (sek) — pauzy dłuższe są skracane |
| `--min-gap` | 120 | Min. przerwa do edycji (ms) |
| `--crossfade` | 60 | Crossfade na cięciach (ms) |
| `--soften-mode` | ROOMTONE | Maskowanie cięć: CUT / FADE / ROOMTONE |
| `--soften-ms` | 120 | Długość softenu (ms) |
| `--max-pause` | 0.4 | Max pauza po edycji (sek) |
| `--start-offset` | — | Początek fragmentu do obróbki (sek) |
| `--end-offset` | — | Koniec fragmentu do obróbki (sek) |
| `--whisper-model` | base | Model Whisper: tiny/base/small/medium/large-v3 |
| `--strict` | off | Pomija edycje o niskiej pewności |
| `--confidence-threshold` | 0.75 | Próg pewności (0.0–1.0) |
| `--verbose` | off | Szczegółowe logowanie |

## Pliki wyjściowe

Dla pliku wejściowego `wywiad.wav` w trybie A:

```
output/
├── wywiad_modeA.wav              # Zmontowane audio (WAV 48kHz/24-bit)
├── wywiad_modeA_edl.json         # Lista cięć (EDL)
├── wywiad_modeA_transcript.txt   # Transkrypcja z oznaczeniami
└── wywiad_modeA_report.json      # Raport przetwarzania
```

## Format EDL/JSON

```json
{
  "version": "1.0",
  "format": "audio_assembly_edl",
  "metadata": {
    "input_file": "wywiad.wav",
    "mode": "CLEAN_FILLERS_AND_PAUSES",
    "config": {
      "silence_threshold_sec": 3.0,
      "min_gap_to_edit_ms": 120,
      "crossfade_ms": 60,
      "soften_mode": "ROOMTONE",
      "soften_ms": 120,
      "max_pause_after_edit_sec": 0.4
    }
  },
  "edits": [
    {
      "type": "filler_removal",
      "start_sec": 1.8,
      "end_sec": 2.5,
      "duration_sec": 0.7,
      "confidence": 0.95,
      "status": "applied",
      "description": "Filler word: 'yyy'",
      "original_text": "yyy",
      "replacement_text": ""
    },
    {
      "type": "pause_shortening",
      "start_sec": 4.7,
      "end_sec": 7.78,
      "duration_sec": 3.08,
      "confidence": 1.0,
      "status": "applied",
      "description": "Shorten pause from 3.5s to 0.4s",
      "original_text": "",
      "replacement_text": ""
    }
  ],
  "summary": {
    "total_edits": 3,
    "applied": 3,
    "review": 0,
    "skipped": 0,
    "total_duration_removed_sec": 4.48
  }
}
```

### Opis pól EDL

| Pole | Typ | Opis |
|---|---|---|
| `type` | string | Typ edycji: `filler_removal`, `pause_shortening`, `repetition_removal`, `sentence_restart_removal` |
| `start_sec` | float | Początek cięcia (sekundy od początku nagrania) |
| `end_sec` | float | Koniec cięcia (sekundy) |
| `duration_sec` | float | Czas trwania cięcia |
| `confidence` | float | Pewność wykrycia (0.0–1.0) |
| `status` | string | `applied` = zastosowano, `review` = do przeglądu, `skipped` = pominięto |
| `description` | string | Opis operacji |
| `original_text` | string | Usunięty tekst (z transkrypcji) |
| `replacement_text` | string | Tekst zastępujący (jeśli dotyczy) |

## Tryby maskowania cięć

- **ROOMTONE** (domyślny): wstawia próbkę tła (room tone) z pobliskiego fragmentu ciszy. Najbardziej naturalne brzmienie.
- **FADE**: fade-out/fade-in wokół punktu cięcia. Dobre dla czystych nagrań.
- **CUT**: twarde cięcie z krótkim crossfade. Najszybsze, ale może być słyszalne.

## Testy

```bash
# Testy jednostkowe
python tests/test_unit.py

# Demo z plikami syntetycznymi
python tests/test_demo.py
```

## Architektura

```
audio_assembly/
├── __init__.py        # Wersja pakietu
├── __main__.py        # Punkt wejścia: python -m audio_assembly
├── cli.py             # Interfejs CLI (click + rich)
├── models.py          # Modele danych, konfiguracja, enumy
├── audio_io.py        # Ładowanie i eksport audio
├── vad.py             # Voice Activity Detection, wykrywanie ciszy
├── transcription.py   # Transkrypcja Whisper (PL, word-level)
├── fillers.py         # Tryb A: wykrywanie wypełniaczy i pauz
├── repetitions.py     # Tryb B: wykrywanie powtórzeń i restartów
├── renderer.py        # Renderowanie audio (crossfade, soften)
├── report.py          # Generowanie raportów EDL/JSON
└── pipeline.py        # Główny pipeline przetwarzania
```

### Pipeline

1. Wczytaj audio → normalizacja do 16kHz mono (analiza) + oryginalna jakość (eksport)
2. VAD — wykrywanie aktywności głosowej i pauz
3. Transkrypcja PL — word-level timestamps (faster-whisper)
4. Budowa kandydatów do edycji
5. Zastosowanie edycji wg wybranego trybu
6. Rendering audio z crossfade i soften_mode
7. Zapis plików wyjściowych + raport

## Wyniki demo

### Demo 1: Tryb A (wypełniacze + pauzy)
- Czas oryginału: 13.0s
- Czas po montażu: 8.9s
- Usunięto: 4.5s (2 wypełniacze, 1 pauza)

### Demo 2: Tryb B (powtórzenia)
- Czas oryginału: 10.0s
- Czas po montażu: 9.4s
- Usunięto: 0.8s (2 powtórzenia)
