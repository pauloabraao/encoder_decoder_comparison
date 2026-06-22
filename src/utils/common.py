"""Funções compartilhadas pelos pipelines encoder e decoder."""

import csv
import os
import re
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    """Colapsa espaços em branco consecutivos em um único espaço e apara as bordas."""
    return re.sub(r"\s+", " ", text).strip()


def save_metrics_to_csv(
    modelo: str,
    tempo_total_s: float,
    peak_memory_mb: float,
    cpu_user_time_s: float,
    cpu_system_time_s: float,
    metrics_file: str,
) -> None:
    """Acrescenta uma linha de métricas de desempenho ao CSV, criando o cabeçalho se necessário."""
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "modelo",
                "tempo_total_s",
                "peak_memory_mb",
                "cpu_user_time_s",
                "cpu_system_time_s",
            ])
        writer.writerow([
            modelo,
            round(tempo_total_s, 2),
            round(peak_memory_mb, 2),
            round(cpu_user_time_s, 2),
            round(cpu_system_time_s, 2),
        ])
