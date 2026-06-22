import fitz  # PyMuPDF
import os
import re


def extract_aditivos_to_md(pdf_path, output_dir, start_index=1):
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    full_text = []
    for page in doc:
        full_text.append(page.get_text("text"))

    full_text = "\n".join(full_text)

    pattern = re.compile(
        r"(EXTRATO DE CONTRATO.*?\*\*\* \*\*\* \*\*\*)",
        re.DOTALL
    )

    matches = pattern.findall(full_text)

    for i, extrato in enumerate(matches):
        idx = start_index + i
        filename = f"extrato_contrato_{idx}.md"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(extrato.strip())

    return len(matches)


def extract_from_folder(folder_path, output_dir, start_index=1):
    total = 0
    current_index = start_index

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            count = extract_aditivos_to_md(
                pdf_path,
                output_dir,
                start_index=current_index
            )

            current_index += count
            total += count

    return total


if __name__ == "__main__":
    
    # # --- opção 1: um único PDF ---
    # pdf_file = r"data\raw\fevereiro\do20250227p01.pdf"
    output_folder = r"data\processed\01_2025"

    # total = extract_aditivos_to_md(pdf_file, output_folder, start_index=48)
    # print(f"{total} extratos salvos em '{output_folder}'")

    # --- opção 2: todos PDFs de uma pasta ---
    pdf_folder = r"data\raw\01_2025"

    total = extract_from_folder(pdf_folder, output_folder, start_index=1)
    print(f"{total} extratos salvos em '{output_folder}'")