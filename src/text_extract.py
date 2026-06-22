"""Extrai cada Extrato de Contrato de PDFs do Diário Oficial para arquivos .md."""

import fitz  # PyMuPDF
import os
import re


def extract_aditivos_to_md(pdf_path, output_dir, start_index=1):
    """Salva cada extrato encontrado no PDF como extrato_contrato_N.md. Retorna a quantidade."""
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)

    pattern = re.compile(r"(EXTRATO DE CONTRATO.*?\*\*\* \*\*\* \*\*\*)", re.DOTALL)
    matches = pattern.findall(full_text)

    for i, extrato in enumerate(matches):
        idx = start_index + i
        filename = f"extrato_contrato_{idx}.md"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(extrato.strip())

    return len(matches)


def extract_from_folder(folder_path, output_dir, start_index=1):
    """Processa todos os PDFs da pasta, numerando os extratos de forma contínua."""
    total = 0
    current_index = start_index

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            count = extract_aditivos_to_md(pdf_path, output_dir, start_index=current_index)
            current_index += count
            total += count

    return total


if __name__ == "__main__":
    pdf_folder = r"data\raw\01_2025"
    output_folder = r"data\processed\01_2025"

    total = extract_from_folder(pdf_folder, output_folder, start_index=1)
    print(f"{total} extratos salvos em '{output_folder}'")