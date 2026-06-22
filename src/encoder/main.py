import os
import json
import logging
import re
import time
import csv

import psutil
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from src.metrics import get_metrics

# ===============================
# CONFIGURAÇÕES
# ===============================

# Modelos do HuggingFace para teste
MODELS = [
    "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
]

MONTHS = ["01_2025"]

QUESTIONS_FILE = "src/encoder/questions.json"

METRICS_FILE = "results/performance_results/encoder_processing_metrics.csv"

# Detectar GPU
device = 0 if torch.cuda.is_available() else -1

# ===============================
# LOGGING
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(text):
    """
    Limpeza essencial para garantir que o contexto de teste
    seja idêntico ao formato visto pelo modelo no treino.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_date(date_str):
    """
    Converte datas em português para o formato DD/MM/YYYY.
    """

    if not date_str or not isinstance(date_str, str):
        return date_str

    text = date_str.strip().lower()
    text = re.sub(r'[,\.;]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

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
        "dezembro": "12", "dez": "12"
    }

    pattern = r'(\d{1,2})\s*(?:de)?\s*([a-zç]+)\s*(?:de)?\s*(\d{4})'

    match = re.search(pattern, text)

    if match:
        day, month_str, year = match.groups()

        month = months.get(month_str)

        if month:
            try:
                day = str(int(day)).zfill(2)
                return f"{day}/{month}/{year}"

            except:
                return date_str

    return date_str

def convert_empty_strings_to_none(data):

    if isinstance(data, dict):
        return {
            k: convert_empty_strings_to_none(v)
            for k, v in data.items()
        }

    elif isinstance(data, list):
        return [
            convert_empty_strings_to_none(item)
            for item in data
        ]

    elif isinstance(data, str) and data.strip() == "":
        return None

    else:
        return data

def build_json_output(answers_dict):

    output = {
        "document_type": "EXTRATO DE CONTRATO",

        "document_id": {
            "number": answers_dict.get("document_number"),
            "ig": answers_dict.get("document_ig")
        },

        "contracting_party": {
            "name": answers_dict.get("contracting_name"),
            "cpf_cnpj": answers_dict.get("contracting_cnpj")
        },

        "contracted_party": {
            "name": answers_dict.get("contracted_name"),
            "cpf_cnpj": answers_dict.get("contracted_cnpj")
        },

        "validity": answers_dict.get("validity"),

        "global_value": answers_dict.get("global_value"),

        "signature": {
            "date": normalize_date(
                answers_dict.get("signature_date")
            ),

            "signatories": {
                "contracting_party":
                    answers_dict.get("signatory_contracting"),

                "contracted_party":
                    answers_dict.get("signatory_contracted")
            }
        },

        "legal_unit": {
            "name": answers_dict.get("legal_name"),
            "role": answers_dict.get("legal_role")
        }
    }

    return convert_empty_strings_to_none(output)


# ===============================
# MÉTRICAS
# ===============================

def save_metrics_to_csv(
    modelo: str,
    tempo_total_s: float,
    peak_memory_mb: float,
    cpu_user_time_s: float,
    cpu_system_time_s: float,
    metrics_file: str,
) -> None:
    """Salva as métricas em um arquivo CSV sem sobrescrever dados anteriores."""
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)

    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            # Escrever cabeçalho se o arquivo não existe
            writer.writerow([
                "modelo",
                "tempo_total_s",
                "peak_memory_mb",
                "cpu_user_time_s",
                "cpu_system_time_s",
            ])

        # Escrever a linha de dados
        writer.writerow([
            modelo,
            round(tempo_total_s, 2),
            round(peak_memory_mb, 2),
            round(cpu_user_time_s, 2),
            round(cpu_system_time_s, 2),
        ])

    logging.info(f"Métricas salvas em: {metrics_file}")


# ===============================
# CARREGAMENTO DE MODELOS
# ===============================

def get_model_path(model_name: str) -> str:
    """
    Retorna o caminho do modelo.
    Se for um caminho local válido, retorna o caminho.
    Caso contrário, assume que é um modelo do HuggingFace.
    """
    local_path = f"models/{model_name}"
    
    # Verificar se existe localmente
    if os.path.exists(local_path):
        return local_path
    
    # Caso contrário, usar o nome direto (HuggingFace hub)
    logging.info(f"Modelo não encontrado localmente. Usando {model_name} do HuggingFace Hub.")
    return model_name


# ===============================
# INFERÊNCIA
# ===============================

def run_inference_for_model(model_name: str, questions_map: dict) -> None:

    model_path = get_model_path(model_name)

    logging.info(f"Carregando modelo: {model_path}")

    qa_pipeline = pipeline(
        "question-answering",
        model=model_path,
        device=device
    )

    for month in MONTHS:

        proc = psutil.Process()
        start_time = time.time()
        start_cpu = proc.cpu_times()
        peak_mem = proc.memory_info().rss

        logging.info(f"[{model_name}] Processando mês: {month}")

        INPUT_DIR = f"data/processed/{month}"

        # Sanitizar o nome do modelo para uso em caminho
        safe_model_name = model_name.replace("/", "_")

        OUTPUT_DIR = (
            f"jsons/enconder_results/{safe_model_name}/{month}"
        )

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(INPUT_DIR):
            logging.warning(f"Pasta não encontrada: {INPUT_DIR}")
            continue

        files = sorted([
            f for f in os.listdir(INPUT_DIR)
            if f.endswith(".md")
        ])

        logging.info(
            f"{len(files)} arquivos encontrados em {month}"
        )

        results, failed_files = [], []

        for file_name in tqdm(
            files,
            desc=f"Extraindo Dados [{model_name}][{month}]"
        ):

            try:
                input_path = os.path.join(INPUT_DIR, file_name)

                with open(input_path, "r", encoding="utf-8") as f:
                    raw_context = f.read()

                context = normalize_text(raw_context)

                current_answers = {}

                for key, question_text in questions_map.items():

                    try:
                        result = qa_pipeline(
                            question=question_text,
                            context=context,
                            max_seq_len=384,
                            doc_stride=128,
                            handle_impossible_answer=True,
                            max_answer_len=100
                        )

                        if result["score"] < 0.3:
                            current_answers[key] = None
                        else:
                            current_answers[key] = result["answer"].strip()

                    except Exception as e:
                        logging.error(
                            f"Erro na pergunta {key} "
                            f"do arquivo {file_name}: {e}"
                        )
                        current_answers[key] = None

                output_json = build_json_output(current_answers)

                file_id = (
                    file_name
                    .replace(".md", "")
                    .split("_")[-1]
                )

                output_name = f"extrato_contrato_{file_id}.json"
                output_path = os.path.join(OUTPUT_DIR, output_name)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=4)

                results.append(output_name)

            except Exception as e:
                logging.exception(
                    f"Erro não tratado ao processar {file_name}: {e}"
                )
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

        logging.info(f"[{model_name}] Mês {month} concluído.")


# ===============================
# EXECUÇÃO
# ===============================

if __name__ == "__main__":

    questions_map = load_questions(QUESTIONS_FILE)

    for model_name in MODELS:
        run_inference_for_model(model_name, questions_map)

    logging.info("Processamento concluído com sucesso!")

    try:
        get_metrics("enconder")
    except Exception as e:
        logging.warning(f"Erro ao processar métricas de accuracy: {e}")
