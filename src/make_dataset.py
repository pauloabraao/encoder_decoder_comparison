"""Compila os JSONs anotados (um por pergunta) no dataset SQuAD 2.0 de treino."""

import os
import json

BASE_PATH = "jsons"
OUTPUT_FILE = "bert_qa_dataset_final_v2.json"
SKIP_FOLDERS = ["VERIFICAR_MANUALMENTE", "logs", "encoder_results"]


def generate_bert_qa_dataset(base_path):
    # Agrupa as perguntas por contexto: { file_id: { "context": ..., "qas": [] } }
    grouped_data = {}

    if not os.path.exists(base_path):
        print(f"[ERRO] Pasta {base_path} não encontrada.")
        return

    categories = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"Iniciando compilação de {len(categories)} categorias...")

    for folder in categories:
        if folder in SKIP_FOLDERS:
            continue

        folder_path = os.path.join(base_path, folder)

        for filename in os.listdir(folder_path):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                file_id = data.get("file")
                context = data.get("context", "")
                question = data.get("question", "")
                answer_text = data.get("answer", "")
                answer_start = data.get("answer_start")
                is_impossible = data.get("is_impossible", False)

                if not file_id or not context:
                    continue

                if file_id not in grouped_data:
                    grouped_data[file_id] = {"context": context, "qas": []}

                qa_entry = {
                    "id": f"{folder}_{filename.split('.')[0]}",
                    "question": question,
                    "is_impossible": is_impossible,
                    "answers": [],
                }

                # Só anexa a resposta quando ela existe e foi localizada no texto.
                if not is_impossible and answer_start is not None:
                    qa_entry["answers"].append({
                        "text": str(answer_text),
                        "answer_start": answer_start,
                    })

                grouped_data[file_id]["qas"].append(qa_entry)

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

    # Converte para o formato SQuAD: lista de parágrafos dentro de 'data'.
    squad_paragraphs = []
    for file_id, content in grouped_data.items():
        squad_paragraphs.append({
            "title": file_id,
            "paragraphs": [
                {
                    "context": content["context"],
                    "qas": content["qas"]
                }
            ]
        })

    squad_format = {"version": "v2.0", "data": squad_paragraphs}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(squad_format, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("DATASET GERADO COM SUCESSO!")
    print(f"Arquivo: {OUTPUT_FILE}")
    print(f"Contextos Únicos: {len(grouped_data)}")
    print(f"Total de Perguntas/Respostas: {sum(len(v['qas']) for v in grouped_data.values())}")
    print("="*50)

if __name__ == "__main__":
    generate_bert_qa_dataset(BASE_PATH)