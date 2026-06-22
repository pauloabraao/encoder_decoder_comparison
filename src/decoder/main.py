"""Pipeline decoder: extração de campos gerando JSON via LLM local (Ollama)."""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import psutil
import requests
from pydantic import ValidationError
from tqdm import tqdm

from src.decoder.schema import ContractExtract
from src.utils.common import normalize_whitespace, save_metrics_to_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "deepseek-r1:1.5b",
]

MONTHS = ["06_2025"]

MAX_RETRIES = 3


def run_ollama(prompt: str, model: str) -> str:
    """Envia o prompt ao Ollama e concatena os fragmentos da resposta em streaming."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "options": {"temperature": 0.1}},
        stream=True,
    )
    answer = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode(errors="ignore"))
            answer += data.get("response", "")
        except Exception:
            continue
    return answer


def answer_question_ollama(question: str, context: str, model: str) -> dict:
    try:
        result = run_ollama(prompt=question + "\n\n" + context, model=model)
        return {"answer": result}
    except Exception as e:
        logger.error(f"Erro ao processar a pergunta: {e}")
        return {"answer": None}


def load_context_from_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        return None
    with open(file_path, encoding="utf-8") as f:
        return normalize_whitespace(f.read())


def load_all_markdown_files(folder_path: str) -> list[tuple[str, str]]:
    """Retorna tuplas (nome_arquivo, conteúdo) de todos os .md, em ordem alfabética."""
    if not os.path.exists(folder_path):
        logger.error(f"Pasta não encontrada: {folder_path}")
        return []

    markdown_files = []
    for file_name in sorted(f for f in os.listdir(folder_path) if f.endswith(".md")):
        content = load_context_from_file(os.path.join(folder_path, file_name))
        if content:
            markdown_files.append((file_name, content))

    logger.info(f"Total de arquivos carregados: {len(markdown_files)}")
    return markdown_files


def filter_answer(context: str, model: str) -> str:
    """Monta o prompt few-shot com o schema-alvo e consulta o modelo."""
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
    return answer_question_ollama(question, context, model)["answer"]


def _extract_raw_json(text: str) -> Optional[str]:
    """Remove cercas markdown e retorna o objeto JSON mais externo do texto."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def extract_json_dict(
    context: str,
    model: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[Optional[dict], bool]:
    """
    Consulta o modelo até MAX_RETRIES vezes em busca de um JSON válido.
    Continua tentando mesmo quando o JSON é válido mas fora do schema.

    Retorna (dados, schema_valido): o dict extraído ({} em falha total) e um
    booleano indicando se ele também passou na validação de ContractExtract.
    """
    last_raw = ""

    for attempt in range(1, MAX_RETRIES + 1):
        if progress_callback:
            progress_callback(attempt)
        else:
            sys.stdout.write(f"\r  Tentativa {attempt}/{MAX_RETRIES}...")
            sys.stdout.flush()

        # Na retentativa, anexa a saída anterior para o modelo se autocorrigir.
        if attempt > 1 and last_raw:
            resposta_slm = filter_answer(
                context + (
                    "\n\n[CORREÇÃO] Sua resposta anterior não era um JSON válido:\n"
                    f"{last_raw}\n"
                    "Corrija e retorne SOMENTE o objeto JSON válido."
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

        try:
            ContractExtract.model_validate(data)
            logger.info("JSON válido e conforme o schema.")
            return data, True
        except ValidationError as e:
            logger.warning(f"Tentativa {attempt}: JSON válido mas fora do schema — {e}")
            if attempt == MAX_RETRIES:
                logger.error("Última tentativa falhou na validação do schema. Retornando JSON fora do schema.")
                return data, False
            continue

    logger.error(f"Todas as {MAX_RETRIES} tentativas falharam. Salvando JSON vazio.")
    return {}, False


def process_markdown_file(
    file_name: str,
    context: str,
    output_dir: str,
    model: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Optional[str]:
    data, schema_valid = extract_json_dict(context, model, progress_callback)

    # document_type é sempre fixo e fica no topo do objeto.
    output_data = {"document_type": "EXTRATO DE CONTRATO", **data}

    # JSON fora do schema (ou inválido) vai para a subpasta schema_invalid/.
    target_dir = output_dir if schema_valid else os.path.join(output_dir, "schema_invalid")
    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, file_name.replace(".md", ".json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    status = "ok" if schema_valid else "schema_invalid"
    logger.info(f"Salvo [{status}]: {output_path}")
    return output_path


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

    def attempt_callback(attempt: int) -> None:
        sys.stdout.write(f"\r  Tentativa {attempt}/{MAX_RETRIES}...")
        sys.stdout.flush()

    for file_name, content in tqdm(markdown_files, desc="Processando arquivos", unit="arquivo"):
        try:
            output = process_markdown_file(file_name, content, output_dir, model, attempt_callback)
            (results if output else failed_files).append(file_name)
        except Exception as e:
            logger.exception(f"Exceção ao processar {file_name}: {e}")
            failed_files.append(file_name)
        finally:
            # Limpa a linha efêmera de "Tentativa N/M".
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

    save_metrics_to_csv(
        modelo=model,
        tempo_total_s=elapsed,
        peak_memory_mb=peak_mem / 1024 / 1024,
        cpu_user_time_s=end_cpu.user - start_cpu.user,
        cpu_system_time_s=end_cpu.system - start_cpu.system,
        metrics_file="results/performance_results/decoder_processing_metrics.csv",
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
