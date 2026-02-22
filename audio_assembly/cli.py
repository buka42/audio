"""CLI interface for Audio Assembly."""

import logging
import os
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .models import Config, ProcessingMode, SoftenMode
from .pipeline import process_audio
from .report import format_report_summary

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--mode", "-m",
    type=click.Choice(["A", "B"], case_sensitive=False),
    default="A",
    help="Tryb przetwarzania: A=CLEAN_FILLERS_AND_PAUSES, B=REMOVE_REPETITIONS_CONTEXTUAL",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="Katalog wyjściowy (domyślnie: ./output/)",
)
@click.option(
    "--silence-threshold", "-s",
    type=float,
    default=3.0,
    show_default=True,
    help="Próg ciszy w sekundach - pauzy dłuższe będą skracane",
)
@click.option(
    "--min-gap",
    type=int,
    default=120,
    show_default=True,
    help="Minimalna długość przerwy do edycji (ms)",
)
@click.option(
    "--crossfade",
    type=int,
    default=60,
    show_default=True,
    help="Długość crossfade na cięciach (ms)",
)
@click.option(
    "--soften-mode",
    type=click.Choice(["CUT", "FADE", "ROOMTONE"], case_sensitive=False),
    default="ROOMTONE",
    show_default=True,
    help="Tryb maskowania cięć",
)
@click.option(
    "--soften-ms",
    type=int,
    default=120,
    show_default=True,
    help="Długość softenu wokół cięcia (ms)",
)
@click.option(
    "--max-pause",
    type=float,
    default=0.4,
    show_default=True,
    help="Maksymalna pauza po edycji (sekundy)",
)
@click.option(
    "--start-offset",
    type=float,
    default=None,
    help="Początek fragmentu do obróbki (sekundy)",
)
@click.option(
    "--end-offset",
    type=float,
    default=None,
    help="Koniec fragmentu do obróbki (sekundy)",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]),
    default="base",
    show_default=True,
    help="Rozmiar modelu Whisper do transkrypcji",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Tryb ścisły: pomija edycje o niskiej pewności zamiast oznaczać do przeglądu",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="Próg pewności dla automatycznego stosowania edycji (0.0-1.0)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Szczegółowe logowanie",
)
def main(
    input_file,
    mode,
    output_dir,
    silence_threshold,
    min_gap,
    crossfade,
    soften_mode,
    soften_ms,
    max_pause,
    start_offset,
    end_offset,
    whisper_model,
    strict,
    confidence_threshold,
    verbose,
):
    """Audio Assembly - Automatyczny montaż audio (PL)

    Przetwarza plik audio (WAV/MP3/M4A) i generuje zmontowaną wersję
    z usuniętymi wypełniaczami, skróconymi pauzami lub usuniętymi
    powtórzeniami.

    INPUT_FILE: Ścieżka do pliku audio wejściowego
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold]Audio Assembly[/bold] - Montaż audio PL\n"
        f"Plik: {input_file}\n"
        f"Tryb: {'A - Czyszczenie wypełniaczy i pauz' if mode == 'A' else 'B - Usuwanie powtórzeń'}",
        title="🎙️ Start",
    ))

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), "output")

    # Build config
    config = Config(
        mode=ProcessingMode.CLEAN_FILLERS_AND_PAUSES if mode == "A" else ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL,
        silence_threshold_sec=silence_threshold,
        min_gap_to_edit_ms=min_gap,
        crossfade_ms=crossfade,
        soften_mode=SoftenMode[soften_mode.upper()],
        soften_ms=soften_ms,
        max_pause_after_edit_sec=max_pause,
        start_offset_sec=start_offset,
        end_offset_sec=end_offset,
        whisper_model=whisper_model,
        strict_mode=strict,
        confidence_threshold=confidence_threshold,
    )

    try:
        report = process_audio(input_file, output_dir, config)
        summary = format_report_summary(report)
        console.print()
        console.print(Panel(summary, title="Wyniki", border_style="green"))

    except Exception as e:
        console.print(f"\n[bold red]Błąd:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
