import os
import statistics
from pathlib import Path
from collections import defaultdict


PROCESSED_DIR = "data/processed"
MONTHS = ["01_2025", "02_2025", "03_2025", "04_2025", "05_2025", "06_2025"]


def count_characters(file_path: str) -> int:
    """Conta o número de caracteres em um arquivo"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return len(content)
    except Exception as e:
        print(f"[ERRO] Lendo {file_path}: {e}")
        return 0


def analyze_md_files():
    """Analisa a média de caracteres dos arquivos .md"""
    
    print("="*70)
    print("ANÁLISE DE CARACTERES - Arquivos .md em data/processed")
    print("="*70)
    
    all_char_counts = []
    month_stats = defaultdict(list)
    
    total_files = 0
    total_chars = 0
    
    # Processar cada mês
    for month in MONTHS:
        month_dir = os.path.join(PROCESSED_DIR, month)
        
        if not os.path.exists(month_dir):
            print(f"\n[WARNING] Pasta não encontrada: {month_dir}")
            continue
        
        # Encontrar todos os arquivos .md
        md_files = list(Path(month_dir).glob("*.md"))
        
        if not md_files:
            print(f"\n[INFO] Nenhum arquivo .md encontrado em: {month_dir}")
            continue
        
        print(f"\n{month}:")
        print(f"  Arquivos encontrados: {len(md_files)}")
        
        month_chars = 0
        
        for md_file in md_files:
            char_count = count_characters(str(md_file))
            month_chars += char_count
            all_char_counts.append(char_count)
            total_chars += char_count
            total_files += 1
        
        month_stats[month] = {
            "files": len(md_files),
            "total_chars": month_chars,
            "avg_chars": month_chars / len(md_files) if md_files else 0
        }
        
        print(f"  Total de caracteres: {month_chars:,}")
        print(f"  Média por arquivo: {month_stats[month]['avg_chars']:,.2f}")
    
    # Resumo geral
    print("\n" + "="*70)
    print("RESUMO GERAL (Todos os meses):")
    print("="*70)
    print(f"Total de arquivos: {total_files}")
    print(f"Total de caracteres: {total_chars:,}")
    print(f"Média geral por arquivo: {total_chars / total_files:,.2f}" if total_files > 0 else "Nenhum arquivo encontrado")
    
    # Estatísticas adicionais
    if all_char_counts:
        print(f"\nMínimo de caracteres: {min(all_char_counts):,}")
        print(f"Máximo de caracteres: {max(all_char_counts):,}")
        print(f"Mediana: {sorted(all_char_counts)[len(all_char_counts) // 2]:,}")
        print(f"Desvio padrão: {statistics.stdev(all_char_counts):,.2f}")
    
    print("="*70)


if __name__ == "__main__":
    analyze_md_files()
