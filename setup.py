"""Setup script for Audio Assembly."""

from setuptools import setup, find_packages

setup(
    name="audio-assembly",
    version="1.0.0",
    description="Automatyczny montaż audio PL - usuwanie wypełniaczy, pauz i powtórzeń",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "flask>=3.0.0",
    ],
    extras_require={
        "transcription": ["faster-whisper>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "audio-assembly=audio_assembly.cli:main",
        ],
    },
)
