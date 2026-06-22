# Extração de Informação Estruturada de Extratos de Contrato

Projeto de TCC que **compara duas abordagens** para extrair campos estruturados
(JSON) de "Extratos de Contrato" publicados no Diário Oficial do Estado do Ceará:

- **Pipeline Encoder** — QA extrativo com modelos BERT (uma pergunta por campo).
- **Pipeline Decoder** — geração direta de JSON com LLMs locais via Ollama.

Ambos produzem o **mesmo schema** de saída e são avaliados pelo mesmo módulo de
métricas (`src/metrics.py`), permitindo um comparativo justo de qualidade e custo.

## Schema de saída

Os dois pipelines extraem os mesmos 13 campos canônicos:

```json
{
  "document_type": "EXTRATO DE CONTRATO",
  "document_id":       { "number": null, "ig": null },
  "contracting_party": { "name": null, "cpf_cnpj": null },
  "contracted_party":  { "name": null, "cpf_cnpj": null },
  "validity": null,
  "global_value": null,
  "signature": {
    "date": null,
    "signatories": { "contracting_party": null, "contracted_party": null }
  },
  "legal_unit": { "name": null, "role": null }
}
```

## Estrutura do repositório

```
src/
├── encoder/
│   ├── main.py          # Pipeline encoder (QA extrativo com BERT)
│   └── questions.json   # Uma pergunta em PT por campo do schema
├── decoder/
│   ├── main.py          # Pipeline decoder (geração de JSON via Ollama)
│   └── schema.py        # Modelos Pydantic (ContractExtract) p/ validação
├── utils/
│   ├── common.py        # Helpers compartilhados (whitespace, métricas CSV)
│   └── normalizar_markdown.py  # Normaliza .md em uma única linha
├── metrics.py           # Avaliação (F1 micro, coverage, similaridade, validade)
├── make_dataset.py      # Compila os JSONs anotados no dataset SQuAD 2.0
├── text_extract.py      # Extrai extratos de PDFs do Diário Oficial p/ .md
├── analyze_md_characters.py    # Estatísticas de tamanho dos .md
├── training.ipynb       # Fine-tuning do modelo encoder (Colab)
└── graphs.ipynb         # Gráficos dos resultados

data/
├── raw/<mês>/           # PDFs do Diário Oficial
├── processed/<mês>/     # Extratos em .md (entrada dos pipelines)
└── ground_truth/<mês>/  # JSONs de referência p/ avaliação

jsons/
├── encoder_results/<modelo>/<mês>/   # Saídas do pipeline encoder
└── decoder_results/<modelo>/<mês>/   # Saídas do pipeline decoder
                                      # (JSON fora do schema vai em schema_invalid/)

results/
├── accuracy_results/{encoder,decoder}/   # Métricas de qualidade
└── performance_results/                  # Métricas de desempenho (tempo/memória/CPU)
```

> Os meses são identificados como `MM_AAAA` (ex.: `01_2025`). A lista de meses
> processados é configurada nas constantes `MONTHS` de cada script.

## Pré-requisitos

```bash
pip install -r requirements.txt
```

- O **pipeline encoder** usa `transformers`/`torch` (GPU é detectada
  automaticamente; cai para CPU se indisponível).
- O **pipeline decoder** requer o [Ollama](https://ollama.com) rodando localmente
  (`ollama serve`) com o modelo desejado já baixado (`ollama pull <modelo>`).
- Os scripts são executados **a partir da raiz do repositório** (usam imports
  absolutos `src.*` e caminhos relativos como `data/processed/...`).

## Pipeline Encoder (`src/encoder/main.py`)

Trata a extração como **Question Answering extrativo**: para cada documento, faz
uma pergunta por campo e usa o trecho do texto retornado como resposta.

Fluxo:
1. Carrega as perguntas de `src/encoder/questions.json` (uma por campo do schema).
2. Para cada `.md` em `data/processed/<mês>/`, normaliza o texto
   (`normalize_whitespace`) e o usa como contexto.
3. Roda o `pipeline("question-answering")` do HuggingFace para cada pergunta.
   Respostas com `score < SCORE_THRESHOLD` (0.3) são descartadas (viram `null`).
4. Monta o JSON do schema (`build_json_output`); datas por extenso são convertidas
   para `DD/MM/YYYY` (`normalize_date`) e strings vazias viram `null`.
5. Salva em `jsons/encoder_results/<modelo>/<mês>/` e registra métricas de
   desempenho em `results/performance_results/encoder_processing_metrics.csv`.
6. Ao final, dispara a avaliação de qualidade (`get_metrics("encoder")`).

O modelo padrão é `pierreguillou/bert-base-cased-squad-v1.1-portuguese`; modelos
em `models/<nome>` são usados localmente quando existirem. Configure `MODELS` e
`MONTHS` no topo do arquivo.

```bash
python -m src.encoder.main
```

## Pipeline Decoder (`src/decoder/main.py`)

Trata a extração como **geração de JSON**: envia o documento ao LLM com um prompt
few-shot que descreve o schema-alvo e exige somente um objeto JSON.

Fluxo:
1. Carrega os `.md` de `data/processed/<mês>/` (`load_all_markdown_files`).
2. Para cada documento, `filter_answer` monta o prompt few-shot (estrutura-alvo +
   instruções + 2 exemplos) e consulta o Ollama via `run_ollama` (streaming,
   `temperature=0.1`).
3. `extract_json_dict` tenta até `MAX_RETRIES` (3) vezes:
   - extrai o objeto JSON da resposta (`_extract_raw_json`, removendo cercas
     markdown);
   - valida sintaxe (`json.loads`) e schema (`ContractExtract.model_validate`);
   - em caso de falha, **reenvia a saída anterior** pedindo autocorreção.
4. `process_markdown_file` injeta `document_type` e salva o resultado em
   `jsons/decoder_results/<modelo>/<mês>/`. JSON **fora do schema** (ou inválido)
   vai para a subpasta `schema_invalid/`, o que permite medir a taxa de validade.
5. Registra métricas de desempenho em
   `results/performance_results/decoder_processing_metrics.csv`.

Configure `MODELS` (nomes de modelos do Ollama, ex.: `deepseek-r1:1.5b`),
`MONTHS` e `OLLAMA_URL` no topo do arquivo.

```bash
ollama serve            # em outro terminal
python -m src.decoder.main
```

## Avaliação (`src/metrics.py`)

Compara as predições de cada modelo com `data/ground_truth/<mês>/` sobre os 13
campos canônicos e produz métricas **globais por modelo**:

- **Precision / Recall / F1 micro** — um campo conta como acerto quando a
  similaridade textual (`SequenceMatcher`) ≥ `SIMILARITY_THRESHOLD` (0.8); acordo
  em campo vazio também conta como acerto.
- **Coverage** — fração dos campos preenchidos no ground truth que o modelo
  também preencheu.
- **Similaridade média** — média das similaridades campo a campo.
- **Validade de schema** (apenas decoder) — `validity_rate` e `schema_error_rate`
  a partir da contagem de arquivos em `schema_invalid/`, além da
  `effective_extraction_rate` (F1 × validade).

Saídas em `results/accuracy_results/{encoder,decoder}/`: `*_results.csv`,
`field_metrics.csv` (métricas por campo) e, para o decoder,
`decoder_json_validity.csv`.

```bash
python src/metrics.py encoder    # ou: decoder
```

## Treinamento do encoder (`src/training.ipynb`)

Notebook (pensado para o **Google Colab**) que faz o **fine-tuning** de um modelo
encoder para QA extrativo. O exemplo usa `huawei-noah/TinyBERT_General_4L_312D`.

Etapas:
1. **Setup** — instala dependências, monta o Google Drive e define os caminhos de
   checkpoints e do CSV de métricas.
2. **Dataset** — carrega `bert_qa_dataset_final_v2.json` (formato SQuAD 2.0,
   gerado por `make_dataset.py`), achata para uma lista de `(context, question,
   answers)` e faz split 90/10 (treino/validação).
3. **Tokenização** (`prepare_train_features`) — tokeniza pergunta+contexto com
   **janela deslizante** (`max_length=384`, `stride=128`,
   `return_overflowing_tokens`) e converte os spans de resposta em
   `start_positions`/`end_positions` via mapeamento de offsets; respostas
   ausentes apontam para o token `[CLS]` (caso "impossível" do SQuAD 2.0).
4. **Treinamento** — `Trainer` com `TrainingArguments`: 3 épocas, `lr=2e-5`,
   batch 8 × `gradient_accumulation_steps=2` (efetivo 16), `fp16`, avaliação e
   checkpoint por época (`save_total_limit=2`, `load_best_model_at_end`). O
   `MetricsLogCallback` grava as métricas de cada log em CSV no Drive.
5. **Exportação** — salva o `final_model` no Drive e consolida o histórico de
   treino (loss/eval_loss por época) em `training_metrics.csv`.

O modelo treinado pode então ser colocado em `models/<nome>` e referenciado em
`MODELS` no pipeline encoder.

## Fluxo completo (ponta a ponta)

1. `python src/text_extract.py` — extrai os extratos dos PDFs para `.md`.
2. `python -m src.utils.normalizar_markdown` — normaliza os `.md` (opcional).
3. (Opcional) `python src/make_dataset.py` + `src/training.ipynb` — gera o dataset
   e treina um encoder próprio.
4. `python -m src.encoder.main` **e/ou** `python -m src.decoder.main` — executa os
   pipelines de extração.
5. `python src/metrics.py encoder|decoder` — avalia os resultados.
6. `src/graphs.ipynb` — gera os gráficos comparativos.
