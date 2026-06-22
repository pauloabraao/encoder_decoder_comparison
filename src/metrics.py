"""
metrics.py — Pipeline de avaliação de extração de informação estruturada.

Suporta modelos Encoder-only e Decoder-only.
Produz métricas globais por modelo (sem CSVs mensais intermediários).
"""

import os
import json
import csv
from pathlib import Path
from difflib import SequenceMatcher
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

JSONS_ROOT = "jsons"
OUTPUT_BASE = "results/accuracy_results"

SIMILARITY_THRESHOLD = 0.8

MONTHS = ["01_2025"]

GROUND_TRUTH_ROOT = "data/ground_truth"

# Campos canônicos do schema — únicos campos avaliados
SCHEMA_FIELDS = [
    "document_id.number",
    "document_id.ig",
    "contracting_party.name",
    "contracting_party.cpf_cnpj",
    "contracted_party.name",
    "contracted_party.cpf_cnpj",
    "validity",
    "global_value",
    "signature.date",
    "signature.signatories.contracting_party",
    "signature.signatories.contracted_party",
    "legal_unit.name",
    "legal_unit.role",
]

N_SCHEMA_FIELDS = len(SCHEMA_FIELDS)


# ---------------------------------------------------------------------------
# Utilitários de similaridade
# ---------------------------------------------------------------------------

def similarity(a, b) -> float:
    """Similaridade SequenceMatcher entre dois valores (0.0–1.0)."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


# ---------------------------------------------------------------------------
# Carregamento e normalização de JSONs
# ---------------------------------------------------------------------------

def flatten_json(data: dict, parent_key: str = "") -> dict:
    """Achata um JSON aninhado com chaves separadas por ponto."""
    items = {}
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key))
        else:
            items[new_key] = v
    return items


def normalize_to_schema(data: dict) -> dict[str, object]:
    """
    Retorna um dicionário com exatamente os campos de SCHEMA_FIELDS.
    Campos ausentes no JSON de entrada são preenchidos com None.
    Campos extras (inventados pelo modelo) são descartados.
    """
    flat = flatten_json(data)
    return {field: flat.get(field) for field in SCHEMA_FIELDS}


def load_prediction(path: Path) -> dict | None:
    """
    Carrega um JSON de predição.
    Retorna None se o arquivo não existir ou for inválido (JSON malformado).
    JSONs com schema inválido (mas JSON válido) são aceitos normalmente.
    """
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_ground_truth(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Avaliação de um par (ground truth × predição)
# ---------------------------------------------------------------------------

def evaluate_document(
    gt_norm: dict[str, object],
    pred_norm: dict[str, object] | None,
) -> dict:
    """
    Avalia um par ground truth × predição sobre os campos canônicos.

    Retorna:
        tp, fp, fn por documento
        per_field: dict field → (tp, fp, fn, sim)
        similarities: lista de floats (uma por campo)
        coverage: fração de campos GT presentes na predição
    """
    per_field: dict[str, dict] = {}
    similarities: list[float] = []
    tp = fp = fn = 0

    # Campos do GT que estão preenchidos (não None)
    gt_filled_fields = sum(1 for v in gt_norm.values() if v is not None)

    # Campos do GT presentes na predição (preenchidos em ambos)
    coverage_count = 0

    for field in SCHEMA_FIELDS:
        gt_val = gt_norm[field]
        pred_val = pred_norm[field] if pred_norm is not None else None

        # Contabilizar coverage
        if gt_val is not None and pred_val is not None:
            coverage_count += 1

        if gt_val is None and pred_val is None:
            # Concordância perfeita em campo vazio — não penalizar
            sim = 1.0
            field_tp, field_fp, field_fn = 1, 0, 0

        elif gt_val is None and pred_val is not None:
            # Modelo inventou um campo que não existe no GT
            sim = 0.0
            field_tp, field_fp, field_fn = 0, 1, 0

        elif gt_val is not None and pred_val is None:
            # Modelo não extraiu um campo que existe no GT
            sim = 0.0
            field_tp, field_fp, field_fn = 0, 0, 1

        else:
            # Ambos preenchidos — medir similaridade
            sim = similarity(gt_val, pred_val)
            if sim >= SIMILARITY_THRESHOLD:
                field_tp, field_fp, field_fn = 1, 0, 0
            else:
                field_tp, field_fp, field_fn = 0, 1, 1

        tp += field_tp
        fp += field_fp
        fn += field_fn
        similarities.append(sim)

        per_field[field] = {
            "tp": field_tp,
            "fp": field_fp,
            "fn": field_fn,
            "sim": sim,
        }

    # Coverage: razão de campos GT preenchidos que foram preenchidos na predição
    coverage = coverage_count / gt_filled_fields if gt_filled_fields > 0 else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_field": per_field,
        "similarities": similarities,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Avaliação global de um modelo
# ---------------------------------------------------------------------------

def _iter_prediction_files(pred_model_dir: Path, filename: str) -> tuple[Path, bool]:
    """
    Gera o caminho e a flag is_invalid para um arquivo de predição.
    Busca primeiro na raiz do modelo/mês, depois em schema_invalid/.
    Retorna (None, False) se não encontrado em lugar nenhum.
    """
    direct = pred_model_dir / filename
    if direct.exists():
        return direct, False

    invalid = pred_model_dir / "schema_invalid" / filename
    if invalid.exists():
        return invalid, True

    return None, False


def evaluate_model_global(
    model_name: str,
    jsons_folder: Path,
    is_decoder: bool,
) -> dict | None:
    """
    Avalia um modelo somando TP/FP/FN de todos os documentos de todos os meses.
    Retorna dict com métricas globais do modelo (ou None se sem dados).
    """
    total_tp = total_fp = total_fn = 0
    all_similarities: list[float] = []
    all_coverages: list[float] = []

    # Acumuladores por campo
    field_tp: dict[str, int] = {f: 0 for f in SCHEMA_FIELDS}
    field_fp: dict[str, int] = {f: 0 for f in SCHEMA_FIELDS}
    field_fn: dict[str, int] = {f: 0 for f in SCHEMA_FIELDS}

    # Validade (apenas decoder)
    total_jsons = valid_jsons = invalid_jsons = 0

    found_any = False

    for month in MONTHS:
        gt_dir = Path(GROUND_TRUTH_ROOT) / month
        pred_month_dir = jsons_folder / model_name / month

        if not gt_dir.exists():
            continue

        gt_files = sorted(gt_dir.glob("*.json"))

        for gt_path in gt_files:
            filename = gt_path.name

            gt_data = load_ground_truth(gt_path)
            if gt_data is None:
                continue  # GT inválido — pular

            gt_norm = normalize_to_schema(gt_data)
            found_any = True

            # Localizar predição
            if not pred_month_dir.exists():
                pred_path, is_invalid = None, False
            else:
                pred_path, is_invalid = _iter_prediction_files(pred_month_dir, filename)

            # Contabilizar validade para decoders
            if is_decoder:
                total_jsons += 1
                if pred_path is None:
                    pass  # Arquivo ausente — não conta em valid nem invalid
                elif is_invalid:
                    invalid_jsons += 1
                else:
                    valid_jsons += 1

            # Carregar predição
            if pred_path is not None:
                pred_data = load_prediction(pred_path)
            else:
                pred_data = None

            if pred_data is not None:
                pred_norm = normalize_to_schema(pred_data)
            else:
                # Arquivo ausente ou JSON malformado: predição vazia
                pred_norm = None

            result = evaluate_document(gt_norm, pred_norm)

            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            all_similarities.extend(result["similarities"])
            all_coverages.append(result["coverage"])

            for field, metrics in result["per_field"].items():
                field_tp[field] += metrics["tp"]
                field_fp[field] += metrics["fp"]
                field_fn[field] += metrics["fn"]

    if not found_any:
        return None

    # Métricas F1 Micro Global
    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall_micro    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    coverage_mean = sum(all_coverages) / len(all_coverages) if all_coverages else 0.0
    sim_mean      = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0

    result_dict = {
        "model": model_name,
        "precision_micro": round(precision_micro, 4),
        "recall_micro":    round(recall_micro, 4),
        "f1_micro":        round(f1_micro, 4),
        "coverage":        round(coverage_mean, 4),
        "similaridade_media": round(sim_mean, 4),
    }

    if is_decoder:
        # Se nenhum arquivo foi contado (todos ausentes), evitar divisão por zero
        validity_rate      = valid_jsons   / total_jsons if total_jsons > 0 else 0.0
        schema_error_rate  = invalid_jsons / total_jsons if total_jsons > 0 else 0.0
        effective_extraction_rate = round(f1_micro * validity_rate, 4)

        result_dict["effective_extraction_rate"] = effective_extraction_rate
        result_dict["_validity"] = {
            "total_jsons":        total_jsons,
            "valid_jsons":        valid_jsons,
            "invalid_jsons":      invalid_jsons,
            "validity_rate":      round(validity_rate, 4),
            "schema_error_rate":  round(schema_error_rate, 4),
        }

    result_dict["_field_stats"] = {
        field: {
            "tp": field_tp[field],
            "fp": field_fp[field],
            "fn": field_fn[field],
        }
        for field in SCHEMA_FIELDS
    }

    return result_dict


# ---------------------------------------------------------------------------
# Salvamento de resultados
# ---------------------------------------------------------------------------

def save_model_results(all_results: list[dict], output_dir: Path, is_decoder: bool):
    """Salva encoder_results.csv ou decoder_results.csv."""
    if not all_results:
        print("[WARNING] Nenhum resultado para salvar.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_decoder:
        filename = output_dir / "decoder_results.csv"
        fieldnames = [
            "model",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "coverage",
            "similaridade_media",
            "effective_extraction_rate",
        ]
    else:
        filename = output_dir / "encoder_results.csv"
        fieldnames = [
            "model",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "coverage",
            "similaridade_media",
        ]

    rows = []
    for r in all_results:
        row = {k: r[k] for k in fieldnames if k in r}
        rows.append(row)

    # Ordenar por f1_micro desc
    rows.sort(key=lambda x: x.get("f1_micro", 0), reverse=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados salvos em: {filename}")


def save_field_metrics(all_results: list[dict], output_dir: Path):
    """Salva field_metrics.csv com métricas por campo por modelo."""
    if not all_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        model = r["model"]
        for field in SCHEMA_FIELDS:
            stats = r["_field_stats"][field]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            rows.append({
                "model":     model,
                "field":     field,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
            })

    filename = output_dir / "field_metrics.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "field", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Métricas por campo salvas em: {filename}")


def save_decoder_validity_metrics(all_results: list[dict], output_dir: Path):
    """Salva decoder_json_validity.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        v = r.get("_validity", {})
        rows.append({
            "model":             r["model"],
            "total_jsons":       v.get("total_jsons", 0),
            "valid_jsons":       v.get("valid_jsons", 0),
            "invalid_jsons":     v.get("invalid_jsons", 0),
            "validity_rate":     v.get("validity_rate", 0.0),
            "schema_error_rate": v.get("schema_error_rate", 0.0),
        })

    # Ordenar por validity_rate desc
    rows.sort(key=lambda x: x.get("validity_rate", 0), reverse=True)

    filename = output_dir / "decoder_json_validity.csv"
    fieldnames = [
        "model",
        "total_jsons",
        "valid_jsons",
        "invalid_jsons",
        "validity_rate",
        "schema_error_rate",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Validade dos decoders salva em: {filename}")


# ---------------------------------------------------------------------------
# Entrypoint principal
# ---------------------------------------------------------------------------

def get_metrics(model: Literal["encoder", "decoder"]):
    """Ponto de entrada principal. `model` deve ser "encoder" ou "decoder"."""
    is_decoder = model == "decoder"
    jsons_folder = Path(JSONS_ROOT) / f"{model}_results"
    output_dir   = Path(OUTPUT_BASE) / ("decoder" if is_decoder else "encoder")

    if not jsons_folder.exists():
        print(f"[ERROR] Pasta de resultados não encontrada: {jsons_folder}")
        return

    model_names = sorted(
        d for d in os.listdir(jsons_folder)
        if os.path.isdir(jsons_folder / d)
    )

    if not model_names:
        print(f"[WARNING] Nenhum modelo encontrado em: {jsons_folder}")
        return

    all_results = []

    for model_name in model_names:
        print(f"\nAvaliando modelo: {model_name}")
        result = evaluate_model_global(model_name, jsons_folder, is_decoder)
        if result:
            all_results.append(result)
        else:
            print(f"  [WARNING] Sem dados para {model_name}")

    if not all_results:
        print("[WARNING] Nenhum resultado gerado.")
        return

    # Salvar resultados
    save_model_results(all_results, output_dir, is_decoder)
    save_field_metrics(all_results, output_dir)

    if is_decoder:
        save_decoder_validity_metrics(all_results, output_dir)

    # Exibir resumo no terminal
    print("\n" + "=" * 60)
    print("Resultados finais:")
    print("=" * 60)

    display_cols = [
        "model", "precision_micro", "recall_micro", "f1_micro",
        "coverage", "similaridade_media",
    ]
    if is_decoder:
        display_cols.append("effective_extraction_rate")

    df = pd.DataFrame([{k: r[k] for k in display_cols} for r in all_results])
    df = df.sort_values("f1_micro", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    if mode not in ("encoder", "decoder"):
        print("Uso: python metrics.py [encoder|decoder]")
        sys.exit(1)

    get_metrics(mode)