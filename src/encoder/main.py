"""Pipeline encoder: extração de campos via QA extrativo (BERT) por documento."""

import os
import json
import logging
import re
import time

import psutil
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

from src.metrics import get_metrics
from src.utils.common import normalize_whitespace, save_metrics_to_csv

MODELS = [
    "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
]

MONTHS = ["01_2025"]

QUESTIONS_FILE = "src/encoder/questions.json"
METRICS_FILE = "results/performance_results/encoder_processing_metrics.csv"

# Score mínimo do QA para aceitar uma resposta; abaixo disso o campo vira None.
SCORE_THRESHOLD = 0.3

device = 0 if torch.cuda.is_available() else -1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_date(date_str):
    """Converte uma data por extenso em português para o formato DD/MM/YYYY."""
    if not date_str or not isinstance(date_str, str):
        return date_str

    text = date_str.strip().lower()
    text = re.sub(r"[,\.;]", " ", text)
    text = re.sub(r"\s+", " ", text)

    months = {
        "janeiro": "01", "jan": "01",
        "fevereiro": "02", "fev": "02",
        "março": "03", "mar": "03",
        "abril": "04", "abr": "04",
        "maio": "05",
        "junho": "06", "jun": "06",
        "julho": "07", "jul": "07",
        "agosto": "08", "ago": "08",
        "setembro": "09", "set": "09",
        "outubro": "10", "out": "10",
        "novembro": "11", "nov": "11",
        "dezembro": "12", "dez": "12",
    }

    match = re.search(r"(\d{1,2})\s*(?:de)?\s*([a-zç]+)\s*(?:de)?\s*(\d{4})", text)
    if not match:
        return date_str

    day, month_str, year = match.groups()
    month = months.get(month_str)
    if not month:
        return date_str

    try:
        return f"{str(int(day)).zfill(2)}/{month}/{year}"
    except ValueError:
        return date_str


def convert_empty_strings_to_none(data):
    """Substitui strings vazias por None recursivamente em dicts e listas."""
    if isinstance(data, dict):
        return {k: convert_empty_strings_to_none(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_empty_strings_to_none(item) for item in data]
    if isinstance(data, str) and data.strip() == "":
        return None
    return data


def build_json_output(answers_dict):
    """Monta o JSON final do schema a partir das respostas de cada pergunta."""
    output = {
        "document_type": "EXTRATO DE CONTRATO",
        "document_id": {
            "number": answers_dict.get("document_number"),
            "ig": answers_dict.get("document_ig"),
        },
        "contracting_party": {
            "name": answers_dict.get("contracting_name"),
            "cpf_cnpj": answers_dict.get("contracting_cnpj"),
        },
        "contracted_party": {
            "name": answers_dict.get("contracted_name"),
            "cpf_cnpj": answers_dict.get("contracted_cnpj"),
        },
        "validity": answers_dict.get("validity"),
        "global_value": answers_dict.get("global_value"),
        "signature": {
            "date": normalize_date(answers_dict.get("signature_date")),
            "signatories": {
                "contracting_party": answers_dict.get("signatory_contracting"),
                "contracted_party": answers_dict.get("signatory_contracted"),
            },
        },
        "legal_unit": {
            "name": answers_dict.get("legal_name"),
            "role": answers_dict.get("legal_role"),
        },
    }
    return convert_empty_strings_to_none(output)


def get_model_path(model_name: str) -> str:
    """Usa o modelo local em models/ se existir; caso contrário, busca no HuggingFace Hub."""
    local_path = f"models/{model_name}"
    if os.path.exists(local_path):
        return local_path

    logging.info(f"Modelo não encontrado localmente. Usando {model_name} do HuggingFace Hub.")
    return model_name


def run_inference_for_model(model_name: str, questions_map: dict) -> None:
    model_path = get_model_path(model_name)
    logging.info(f"Carregando modelo: {model_path}")

    qa_pipeline = pipeline("question-answering", model=model_path, device=device)

    for month in MONTHS:
        proc = psutil.Process()
        start_time = time.time()
        start_cpu = proc.cpu_times()
        peak_mem = proc.memory_info().rss

        logging.info(f"[{model_name}] Processando mês: {month}")

        input_dir = f"data/processed/{month}"
        safe_model_name = model_name.replace("/", "_")
        output_dir = f"jsons/encoder_results/{safe_model_name}/{month}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(input_dir):
            logging.warning(f"Pasta não encontrada: {input_dir}")
            continue

        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".md"))
        logging.info(f"{len(files)} arquivos encontrados em {month}")

        results, failed_files = [], []

        for file_name in tqdm(files, desc=f"Extraindo Dados [{model_name}][{month}]"):
            try:
                input_path = os.path.join(input_dir, file_name)
                with open(input_path, "r", encoding="utf-8") as f:
                    context = normalize_whitespace(f.read())

                current_answers = {}
                for key, question_text in questions_map.items():
                    try:
                        result = qa_pipeline(
                            question=question_text,
                            context=context,
                            max_seq_len=384,
                            doc_stride=128,
                            handle_impossible_answer=True,
                            max_answer_len=100,
                        )
                        if result["score"] < SCORE_THRESHOLD:
                            current_answers[key] = None
                        else:
                            current_answers[key] = result["answer"].strip()
                    except Exception as e:
                        logging.error(f"Erro na pergunta {key} do arquivo {file_name}: {e}")
                        current_answers[key] = None

                output_json = build_json_output(current_answers)

                file_id = file_name.replace(".md", "").split("_")[-1]
                output_name = f"extrato_contrato_{file_id}.json"
                output_path = os.path.join(output_dir, output_name)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=4)

                results.append(output_name)

            except Exception as e:
                logging.exception(f"Erro não tratado ao processar {file_name}: {e}")
                failed_files.append(file_name)

        elapsed = time.time() - start_time
        end_cpu = proc.cpu_times()
        peak_mem = max(peak_mem, proc.memory_info().rss)

        save_metrics_to_csv(
            modelo=model_name,
            tempo_total_s=elapsed,
            peak_memory_mb=peak_mem / 1024 / 1024,
            cpu_user_time_s=end_cpu.user - start_cpu.user,
            cpu_system_time_s=end_cpu.system - start_cpu.system,
            metrics_file=METRICS_FILE,
        )
        logging.info(f"Métricas salvas em: {METRICS_FILE}")
        logging.info(f"[{model_name}] Mês {month} concluído.")


if __name__ == "__main__":
    questions_map = load_questions(QUESTIONS_FILE)

    for model_name in MODELS:
        run_inference_for_model(model_name, questions_map)

    logging.info("Processamento concluído com sucesso!")

    try:
        get_metrics("encoder")
    except Exception as e:
        logging.warning(f"Erro ao processar métricas de accuracy: {e}")
