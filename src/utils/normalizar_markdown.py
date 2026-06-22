import os
import re
from pathlib import Path

def processar_arquivos_md(caminho_base):
    """
    Lê todos os arquivos .md dentro de todas as pastas do caminho passado,
    remove quebras de linhas e salva os arquivos.
    
    Args:
        caminho_base (str): Caminho inicial para buscar os arquivos .md
    """
    
    # Converte para objeto Path para facilitar manipulação
    caminho_base = Path(caminho_base)
    
    # Verifica se o caminho existe
    if not caminho_base.exists():
        print(f"Erro: O caminho '{caminho_base}' não existe!")
        return
    
    # Contadores para estatísticas
    arquivos_processados = 0
    erros = 0
    
    # Busca por todos os arquivos .md recursivamente
    for arquivo in caminho_base.rglob("*.md"):
        try:
            print(f"Processando: {arquivo}")
            
            # Lê o conteúdo do arquivo
            with open(arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            # Conta linhas originais
            linhas_originais = len(conteudo.splitlines())
            
            # Remove quebras de linhas
            # Opção 1: Remove TODAS as quebras de linha (incluindo parágrafos vazios)
            conteudo_modificado = re.sub(r'\n+', ' ', conteudo)
            
            # Opção 2: Remove quebras de linha mas mantém um espaço entre parágrafos
            # Descomente a linha abaixo se preferir esta opção:
            # conteudo_modificado = re.sub(r'\n\s*\n', '\n\n', re.sub(r'(?<!\n)\n(?!\n)', ' ', conteudo))
            
            # Remove espaços extras que podem ter sido criados
            conteudo_modificado = re.sub(r'\s+', ' ', conteudo_modificado).strip()
            
            # Conta linhas após modificação
            linhas_finais = len(conteudo_modificado.splitlines())
            
            # Salva o arquivo modificado
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.write(conteudo_modificado)
            
            arquivos_processados += 1
            print(f"  ✓ Processado: {linhas_originais} linhas -> {linhas_finais} linhas")
            
        except Exception as e:
            erros += 1
            print(f"  ✗ Erro ao processar {arquivo}: {str(e)}")
    
    # Exibe resumo final
    print("\n" + "="*50)
    print(f"RESUMO DO PROCESSAMENTO:")
    print(f"  Arquivos processados: {arquivos_processados}")
    print(f"  Erros: {erros}")
    print(f"  Total: {arquivos_processados + erros}")

def processar_com_backup(caminho_base):
    """
    Versão que cria backup dos arquivos originais antes de modificar.
    
    Args:
        caminho_base (str): Caminho inicial para buscar os arquivos .md
    """
    caminho_base = Path(caminho_base)
    
    if not caminho_base.exists():
        print(f"Erro: O caminho '{caminho_base}' não existe!")
        return
    
    backup_dir = caminho_base / "backup_md"
    backup_dir.mkdir(exist_ok=True)
    
    arquivos_processados = 0
    
    for arquivo in caminho_base.rglob("*.md"):
        # Pula arquivos dentro do diretório de backup
        if "backup_md" in str(arquivo):
            continue
            
        try:
            print(f"Processando: {arquivo}")
            
            # Lê o conteúdo original
            with open(arquivo, 'r', encoding='utf-8') as f:
                conteudo_original = f.read()
            
            # Cria backup
            backup_path = backup_dir / f"{arquivo.name}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(conteudo_original)
            
            # Remove quebras de linha
            conteudo_modificado = re.sub(r'\n+', ' ', conteudo_original)
            conteudo_modificado = re.sub(r'\s+', ' ', conteudo_modificado).strip()
            
            # Salva arquivo modificado
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.write(conteudo_modificado)
            
            arquivos_processados += 1
            print(f"  ✓ Arquivo modificado e backup salvo em: {backup_path}")
            
        except Exception as e:
            print(f"  ✗ Erro ao processar {arquivo}: {str(e)}")
    
    print(f"\nProcessamento concluído! {arquivos_processados} arquivos modificados.")
    print(f"Backups salvos em: {backup_dir}")

if __name__ == "__main__":
    # Exemplo de uso
    
    # Opção 1: Sem backup (mais rápido, mas sem recuperação)
    # caminho = input("Digite o caminho para processar: ").strip()
    # processar_arquivos_md(caminho)
    
    # Opção 2: Com backup (recomendado)
    caminho = "../../data/processed"
    
    if not caminho:
        caminho = "../../data/processed"  # Usa diretório atual se não informar nada
    
    print(f"Processando arquivos .md em: {caminho}")
    print("Aguarde...")
    
    # Escolha qual função usar:
    # processar_arquivos_md(caminho)  # Sem backup
    processar_com_backup(caminho)      # Com backup