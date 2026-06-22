"""Normaliza arquivos .md em uma única linha, colapsando quebras e espaços."""

import re
from pathlib import Path


def collapse_whitespace(text: str) -> str:
    """Remove quebras de linha e colapsa espaços consecutivos em um único espaço."""
    return re.sub(r"\s+", " ", re.sub(r"\n+", " ", text)).strip()


def process_md_files(base_path):
    """Normaliza, no lugar, todos os .md encontrados recursivamente em base_path."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Erro: O caminho '{base_path}' não existe!")
        return

    processed = errors = 0
    for arquivo in base_path.rglob("*.md"):
        try:
            print(f"Processando: {arquivo}")
            with open(arquivo, "r", encoding="utf-8") as f:
                conteudo = f.read()

            linhas_originais = len(conteudo.splitlines())
            conteudo_modificado = collapse_whitespace(conteudo)

            with open(arquivo, "w", encoding="utf-8") as f:
                f.write(conteudo_modificado)

            processed += 1
            print(f"  ✓ Processado: {linhas_originais} linhas -> 1 linha")
        except Exception as e:
            errors += 1
            print(f"  ✗ Erro ao processar {arquivo}: {e}")

    print("\n" + "=" * 50)
    print("RESUMO DO PROCESSAMENTO:")
    print(f"  Arquivos processados: {processed}")
    print(f"  Erros: {errors}")
    print(f"  Total: {processed + errors}")


def process_md_files_with_backup(base_path):
    """Igual a process_md_files, mas salva o original em backup_md/ antes de sobrescrever."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Erro: O caminho '{base_path}' não existe!")
        return

    backup_dir = base_path / "backup_md"
    backup_dir.mkdir(exist_ok=True)

    processed = 0
    for arquivo in base_path.rglob("*.md"):
        if "backup_md" in str(arquivo):
            continue
        try:
            print(f"Processando: {arquivo}")
            with open(arquivo, "r", encoding="utf-8") as f:
                conteudo_original = f.read()

            backup_path = backup_dir / f"{arquivo.name}.backup"
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(conteudo_original)

            with open(arquivo, "w", encoding="utf-8") as f:
                f.write(collapse_whitespace(conteudo_original))

            processed += 1
            print(f"  ✓ Arquivo modificado e backup salvo em: {backup_path}")
        except Exception as e:
            print(f"  ✗ Erro ao processar {arquivo}: {e}")

    print(f"\nProcessamento concluído! {processed} arquivos modificados.")
    print(f"Backups salvos em: {backup_dir}")


if __name__ == "__main__":
    caminho = "../../data/processed"
    print(f"Processando arquivos .md em: {caminho}")
    process_md_files_with_backup(caminho)
