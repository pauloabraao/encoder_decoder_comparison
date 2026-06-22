import re
import json
import requests
import time
import psutil
import logging
from typing import Optional, Dict, Any, Callable
import os
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm
from src.decoder.schema import ContractExtract

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURAÇÕES ====================
OLLAMA_URL = 'http://localhost:11434/api/generate'

MODELS = [
    #'mistral:7b'
    #'llama3.2:1b',
    'deepseek-r1:1.5b'
    #'deepseek-r1:1.5b',
    # 'qwen2.5vl:3b',
]

MONTHS = ["06_2025"]

MAX_RETRIES = 3

# ==================== OLLAMA ====================

def run_ollama(prompt: str, model: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "options": {"temperature": 0.1}},
        stream=True,
    )
    resposta = ''
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode(errors='ignore'))
                resposta += data.get('response', '')
            except Exception:
                continue
    return resposta


def answer_question_ollama(question: str, context: str, model: str) -> dict:
    try:
        result = run_ollama(prompt=question + '\n\n' + context, model=model)
        return {"answer": result}
    except Exception as e:
        logger.error(f"Erro ao processar a pergunta: {e}")
        return {"answer": None}


# ==================== FILE UTILITIES ====================

def load_context_from_file(file_path: str) -> Optional[str]:
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    return None


def load_all_markdown_files(folder_path: str) -> list[tuple[str, str]]:
    """Retorna uma lista de tuplas (nome_arquivo, conteúdo) ordenada alfabeticamente."""
    markdown_files = []
    if not os.path.exists(folder_path):
        logger.error(f"Pasta não encontrada: {folder_path}")
        return markdown_files

    # Obter todos os arquivos .md em ordem crescente
    md_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.md')])
    
    for file_name in md_files:
        file_path = os.path.join(folder_path, file_name)
        content = load_context_from_file(file_path)
        if content:
            markdown_files.append((file_name, content))

    logger.info(f"Total de arquivos carregados: {len(markdown_files)}")
    return markdown_files


# ==================== EXTRACTION ====================

def filter_answer(context: str, model: str) -> str:
    question = """
    Retorne SOMENTE um objeto JSON válido, sem explicações ou comentários. 
    { 
        "document_id": {
            "number": "",
            "ig": ""
        },
        "contracting_party": {
            "name": "",
            "cpf_cnpj": ""
        },
        "contracted_party": {
            "name": "",
            "cpf_cnpj": ""
        },
        "validity": "",
        "global_value": "",
        "signature": {
            "date": "",
            "signatories": {
            "contracting_party": "",
            "contracted_party": ""
            }
        },
        "legal_unit": {
            "name": "",
            "role": ""
        }
    }
    - Instruções:
        - Não inclua campos que não estejam na estrutura acima
        - As datas deve estar no formato DD/MM/YYYY.
        - Caso alguma informação não esteja no contexto dado, no campo referente retorne null.
        - Inclua no objeto "contracting_party" as informações da pessoa jurídica contratante.
        - Inclua no objeto "contracted_party" as informações da pessoa jurídica contratada.
    Exemplo 01:
        Contexto:   EXTRATO DE CONTRATO Nº DO DOCUMENTO 023/2025
                    IG:1368314|SACC:1363604
                    CONTRATANTE: A SECRETARIA DO DESENVOLVIMENTO AGRÁRIO - SDA, inscrita no CNPJ nº 07.954.563/0001-68. CONTRATADA: ROBERIO MATEUS DE ARAUJO - ME, inscrito (a) no CNPJ/MF sob o nº 22.267.632/0001-59.VIGÊNCIA: O prazo de vigência do presente contrato será de 12 (doze) meses a partir da data de sua publicação no DOE. VALOR GLOBAL: no valor de R$ 104.578,80 (cento e quatro mil, quinhentos e setenta e oito reais e oitenta centavos). DATA DA ASSINATURA: 07 de abril de 2025. SIGNATÁRIOS: MOISÉS BRAZ RICARDO Secretário do Desenvolvimento Agrário (CONTRATANTE) e ROBERIO MATEUS DE ARAUJO
                    Representante da Contratada (CONTRATADA).
                    Anna Karinne Nery Veras
                    COORDENADORA DA ASSESSORIA JURÍDICA
                    ...
        Resposta:
            {
                "document_type": "EXTRATO DE CONTRATO",
                "document_id": {
                    "number": "023/2025",
                    "ig": "1363604"
                },
                "contracting_party": {
                    "name": "SECRETARIA DO DESENVOLVIMENTO AGRÁRIO-SDA",
                    "cpf_cnpj": "07.954.563/0001-68"
                },
                "contracted_party": {
                    "name": "ROBERIO MATEUS DE ARAUJO - ME",
                    "cpf_cnpj": "22.267.632/0001-59"
                },
                "validity": "12 (doze) meses",
                "global_value": "R$ 104.578,80",
                "signature": {
                    "date": "07/04/2025",
                    "signatories": {
                    "contracting_party": "MOISÉS BRAZ RICARDO",
                    "contracted_party": "ROBERIO MATEUS DE ARAUJO"
                    }
                },
                "legal_unit": {
                    "name": "Anna Karinne Nery Veras",
                    "role": "Coordenadora da Assessoria Jurídica"
                }
            }

    Exemplo 02:

        Contexto:
            EXTRATO DE CONTRATO
            Nº DO DOCUMENTO 22001.128382/2024-43|SACC:1363604
            CONTRATANTE: O Estado do Ceará, através da Secretaria da Educação/EEMTI Gerardo Majella, inscrita(o) no CNPJ sob o nº 07.954.514/0555-31, doravante denominada(o) CONTRATANTE, neste ato representada(o) pela
            Diretora Escolar ARIANNY NASCIMENTO DE SOUSA CONTRATADA: T.F CONSTRUCOES LTDA, inscrita no CPF/CNPJ sob o nº 54.095.907/0001-00. VIGÊNCIA: O prazo de vigência é de 365 (trezentos e sessenta e cinco) dias corridos, VALOR GLOBAL: R$ 51.280,00 (cinquenta e um mil, duzentos e oitenta reais) DATA DA ASSINATURA:  SIGNATÁRIOS: CONTRATANTE – ARIANNY NASCIMENTO DE SOUSA CONTRATADA - Rodrigo Ferreira Torres e TESTEMUNHAS: 01- Italo Norberto Marinho, 02- Maria Roniely Pinheiro. 28 de maio de 2025.
            Marcos Felipe Vicente
            COORDENADOR/ASJUR

        Resposta:
            {
                "document_type": "EXTRATO DE CONTRATO",
                "document_id": {
                    "number": "22001.128382/2024-43",
                    "ig": null
                },
                "contracting_party": {
                    "name": "Secretaria da Educação/EEMTI Gerardo Majella",
                    "cpf_cnpj": "07.954.514/0555-31"
                },
                "contracted_party": {
                    "name": "T.F CONSTRUCOES LTDA",
                    "cpf_cnpj": "54.095.907/0001-00"
                },
                "validity": "365 (trezentos e sessenta e cinco) dias",
                "global_value": "R$ 51.280,00",
                "signature": {
                    "date": null,
                    "signatories": {
                    "contracting_party": "ARIANNY NASCIMENTO DE SOUSA",
                    "contracted_party": "Rodrigo Ferreira Torres"
                    }
                },
                "legal_unit": {
                    "name": "Marcos Felipe Vicente",
                    "role": "COORDENADOR/ASJUR"
                }
            }
    """
    res = answer_question_ollama(question, context, model)
    return res['answer']


def _extract_raw_json(text: str) -> Optional[str]:
    """Strips markdown fences and returns the outermost JSON object string."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    match = re.search(r'\{[\s\S]*\}', text)
    return match.group(0) if match else None


def extrair_json_para_dicionario(context: str, model: str, progress_callback: Optional[Callable[[int], None]] = None) -> tuple[Optional[dict], bool]:
    """
    Tries up to MAX_RETRIES times to get a valid JSON from the model.
    Continues retrying even if JSON is valid but out of schema.

    Returns:
        (dict, schema_valid):
            dict  — extracted data, {} on total failure, None never returned
            bool  — True if the dict also passes ContractExtract validation
    """
    last_raw = ""
    last_valid_data = {}  # Store the last valid JSON in case schema fails

    for attempt in range(1, MAX_RETRIES + 1):
        # Mostrar log efêmero de retentativa
        if progress_callback:
            progress_callback(attempt)
        else:
            # Fallback se não tiver callback
            sys.stdout.write(f"\r  Tentativa {attempt}/{MAX_RETRIES}...")
            sys.stdout.flush()

        # On retry, append the bad output so the model can self-correct
        if attempt > 1 and last_raw:
            resposta_slm = filter_answer(
                context + (
                    f"\n\n[CORREÇÃO] Sua resposta anterior não era um JSON válido:\n"
                    f"{last_raw}\n"
                    f"Corrija e retorne SOMENTE o objeto JSON válido."
                ),
                model,
            )
        else:
            resposta_slm = filter_answer(context, model)

        last_raw = resposta_slm

        json_str = _extract_raw_json(resposta_slm)
        if not json_str:
            logger.warning(f"Tentativa {attempt}: nenhum bloco JSON encontrado.")
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Tentativa {attempt}: JSON inválido — {e}")
            continue

        if not isinstance(data, dict):
            logger.warning(f"Tentativa {attempt}: JSON não é um objeto.")
            continue

        # JSON is structurally valid — now check schema
        try:
            ContractExtract.model_validate(data)
            logger.info("JSON válido e conforme o schema.")
            return data, True
        except ValidationError as e:
            logger.warning(f"Tentativa {attempt}: JSON válido mas fora do schema — {e}")
            last_valid_data = data
            # Continue tentando até a última tentativa
            if attempt == MAX_RETRIES:
                logger.error(f"Terceira tentativa falhou na validação do schema. Retornando JSON inválido.")
                return data, False
            continue

    logger.error(f"Todas as {MAX_RETRIES} tentativas falharam. Salvando JSON vazio.")
    return {}, False


# ==================== FILE PROCESSING ====================

def process_markdown_file(
    file_name: str,
    context: str,
    output_dir: str,
    model: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Optional[str]:

    data, schema_valid = extrair_json_para_dicionario(context, model, progress_callback)

    # Always inject document_type at the top
    output_data = {"document_type": "EXTRATO DE CONTRATO", **data}

    if not schema_valid:
        # Salva em schema_invalid: JSON válido fora do schema OU JSON inválido
        target_dir = os.path.join(output_dir, "schema_invalid")
    else:
        target_dir = output_dir

    os.makedirs(target_dir, exist_ok=True)

    output_filename = file_name.replace('.md', '.json')
    output_path = os.path.join(target_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    status = "schema_invalid" if not schema_valid else ("ok")
    logger.info(f"Salvo [{status}]: {output_path}")
    return output_path


# ==================== METRICS ====================

def save_metrics_to_csv(
    modelo: str,
    tempo_total_s: float,
    peak_memory_mb: float,
    cpu_user_time_s: float,
    cpu_system_time_s: float,
    metrics_file: str,
) -> None:
    """
    Salva as métricas em um arquivo CSV sem sobrescrever dados anteriores.
    """
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

    logger.info(f"Métricas salvas em: {metrics_file}")


# ==================== MAIN ====================

def main(markdown_folder: str, output_dir: str, model: str) -> None:
    process = psutil.Process()
    start_time = time.time()
    start_cpu = process.cpu_times()
    peak_mem = process.memory_info().rss

    logger.info("=" * 60)
    logger.info(f"INICIANDO PROCESSAMENTO — modelo: {model}")
    logger.info("=" * 60)

    markdown_files = load_all_markdown_files(markdown_folder)
    if not markdown_files:
        logger.error("Nenhum arquivo .md encontrado!")
        return

    results, failed_files = [], []

    # Criar callback para mostrar tentativas efêmeras
    def attempt_callback(attempt: int) -> None:
        sys.stdout.write(f"\r  Tentativa {attempt}/{MAX_RETRIES}...")
        sys.stdout.flush()

    # Usar tqdm para barra de progresso
    for file_name, content in tqdm(markdown_files, desc="Processando arquivos", unit="arquivo"):
        try:
            output = process_markdown_file(file_name, content, output_dir, model, attempt_callback)
            (results if output else failed_files).append(file_name)
            # Limpar a linha de tentativa ao terminar arquivo
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
        except Exception as e:
            logger.exception(f"Exceção ao processar {file_name}: {e}")
            failed_files.append(file_name)
            # Limpar a linha de tentativa
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    end_cpu = process.cpu_times()
    peak_mem = max(peak_mem, process.memory_info().rss)

    logger.info("=" * 60)
    logger.info("RESUMO")
    logger.info(f"  Total lidos          : {len(markdown_files)}")
    logger.info(f"  Sucesso              : {len(results)}")
    logger.info(f"  Falhas               : {len(failed_files)}")
    if failed_files:
        logger.info(f"  Arquivos falhados    : {', '.join(failed_files)}")
    logger.info(f"  Tempo total          : {elapsed:.2f}s")
    logger.info(f"  Peak Memory Usage    : {peak_mem / 1024 / 1024:.2f} MB")
    logger.info(f"  CPU user             : {end_cpu.user - start_cpu.user:.2f}s")
    logger.info(f"  CPU system           : {end_cpu.system - start_cpu.system:.2f}s")
    logger.info("=" * 60)
    
    # ==================== SALVAR MÉTRICAS ====================

    # Preparar caminho de saída
    metrics_csv = "results/performance_results/decoder_processing_metrics.csv"

    # Salvar métricas
    save_metrics_to_csv(
        modelo=model,
        tempo_total_s=elapsed,
        peak_memory_mb=peak_mem / 1024 / 1024,
        cpu_user_time_s=end_cpu.user - start_cpu.user,
        cpu_system_time_s=end_cpu.system - start_cpu.system,
        metrics_file=metrics_csv,
    )


if __name__ == "__main__":
    for model in MODELS:
        model_name = model.replace(":", "_").replace(".", "-")
        for month in MONTHS:
            main(
                markdown_folder=f"data/processed/{month}",
                output_dir=f"jsons/decoder_results/{model_name}/{month}",
                model=model,
            )