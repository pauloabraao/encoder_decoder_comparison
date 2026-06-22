import os
import json
from pathlib import Path

# ===============================
# CONFIGURAÇÕES
# ===============================
BASE_PATH = "jsons"  # Pasta raiz onde estão as categorias
OUTPUT_FILE = "bert_qa_dataset_final_v2.json"
SKIP_FOLDERS = ["VERIFICAR_MANUALMENTE", "logs", "enconder_results"]

def generate_bert_qa_dataset(base_path):
    # Dicionário para agrupar por arquivo (contexto único)
    # Estrutura: { "file_name": { "context": "...", "qas": [] } }
    grouped_data = {}
    
    # 1. Lista todas as subpastas dinamicamente
    if not os.path.exists(base_path):
        print(f"[ERRO] Pasta {base_path} não encontrada.")
        return

    categories = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print(f"Iniciando compilação de {len(categories)} categorias...")

    for folder in categories:
        # Pular pastas de controle ou de erro
        if folder in SKIP_FOLDERS:
            continue

        folder_path = os.path.join(base_path, folder)
        
        # 2. Percorrer arquivos JSON dentro da categoria
        for filename in os.listdir(folder_path):
            if not filename.endswith(".json"):
                continue
                
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_id = data.get("file")
                context = data.get("context", "")
                question = data.get("question", "")
                answer_text = data.get("answer", "")
                # Usar o answer_start que o script de sanitização calculou
                answer_start = data.get("answer_start")
                is_impossible = data.get("is_impossible", False)

                # Validação básica
                if not file_id or not context:
                    continue

                # Se o contexto ainda não existe no dicionário, cria a estrutura base
                if file_id not in grouped_data:
                    grouped_data[file_id] = {
                        "context": context,
                        "qas": []
                    }
                
                # Montar estrutura da pergunta (Formato SQuAD 2.0)
                qa_entry = {
                    "id": f"{folder}_{filename.split('.')[0]}",
                    "question": question,
                    "is_impossible": is_impossible,
                    "answers": []
                }

                # Se a resposta existe e foi localizada no texto, adiciona ao array
                if not is_impossible and answer_start is not None:
                    qa_entry["answers"].append({
                        "text": str(answer_text),
                        "answer_start": answer_start
                    })
                
                grouped_data[file_id]["qas"].append(qa_entry)

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

    # 3. Transformar o dicionário no formato final (SQuAD)
    # O SQuAD espera uma lista de parágrafos dentro de 'data'
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

    squad_format = {
        "version": "v2.0",
        "data": squad_paragraphs
    }

    # 4. Salvar o arquivo final
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(squad_format, f, ensure_ascii=False, indent=2)
    
    print("="*50)
    print(f"DATASET GERADO COM SUCESSO!")
    print(f"Arquivo: {OUTPUT_FILE}")
    print(f"Contextos Únicos: {len(grouped_data)}")
    print(f"Total de Perguntas/Respostas: {sum(len(v['qas']) for v in grouped_data.values())}")
    print("="*50)

if __name__ == "__main__":
    generate_bert_qa_dataset(BASE_PATH)