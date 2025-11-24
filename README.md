# RAG em PDF via Terminal

CLI simples em Python que lê um PDF local, divide em trechos, busca por similaridade (FAISS), reranqueia com CrossEncoder e responde com o modelo de chat da Mistral. Funciona em loop: você informa um PDF, faz perguntas, pode gerar um índice rápido das páginas ou trocar de arquivo sem reiniciar.

## Principais recursos
- Chat em linha de comando, reutilizando o mesmo índice enquanto você pergunta.
- Busca híbrida: embeddings `intfloat/multilingual-e5-base` + FAISS (Inner Product) + reranqueamento com `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Prompt compacto para `mistral-small-latest`, com contagem de tokens exibida a cada resposta.
- Comando opcional `gerar indice` que pede para o Mistral criar títulos curtos por página.
- Comando `trocar pdf` para abrir outro arquivo no meio da sessão.

## Requisitos
- Python 3.10+ recomendado
- Chave da API da Mistral exportada em `MISTRAL_API_KEY`
- Dependencias Python:
  - `pypdf`
  - `sentence_transformers`
  - `faiss-cpu`
  - `mistralai`
  - `numpy`

## Instalação rápida
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pypdf sentence_transformers faiss-cpu mistralai numpy
```

## Configuração da API
```bash
export MISTRAL_API_KEY="sua-chave-aqui"  # Windows: set MISTRAL_API_KEY=sua-chave-aqui
```

## Como usar
```bash
python rag.py
```
1) Informe o caminho do PDF quando solicitado (pode estar na mesma pasta).  
2) Pergunte sobre o documento. Use `sair` para encerrar.

Comandos aceitos no chat:
- Perguntas livres: "Qual o resumo do capítulo 1?"
- `gerar indice`: cria títulos curtos para cada página (usa a API da Mistral).
- `trocar pdf`: volta a perguntar o caminho de um novo PDF.
- `sair`: finaliza o script.

Exemplo rápido (Linux/macOS):
```bash
export MISTRAL_API_KEY="minha-chave"
python rag.py
# caminho do PDF: /caminho/para/documento.pdf
# pergunta: Quais são as conclusoes?
```

## O que acontece por trás
- Leitura: `carregar_pdf` extrai texto página a página e guarda offsets.
- Chunking: `divisor_trechos` cria janelas de 2000 caracteres com overlap de 400 e mantém a página de origem.
- Indexação: embeddings normalizados via `SentenceTransformer`, índice FAISS de produto interno.
- Rerank: busca top-30 no FAISS e reranqueia com CrossEncoder, retornando os melhores trechos.
- Geração: `prompt_mistral` constrói o contexto e `mistral` envia para o modelo chat, exibindo uso de tokens.

## Ajustes rápidos
- Tamanho dos trechos: altere `chunk_size` e `overlap` em `divisor_trechos`.
- Candidatos e top final: ajuste `k_base` e `k_final` em `busca_rerank`.
- Temperatura do modelo: mude `temperature` em `mistral`.

## Problemas comuns
- **Chave não configurada:** erro "Defina a API MISTRAL" indica `MISTRAL_API_KEY` ausente ou vazia.
- **PDF sem texto extraível:** `pypdf` precisa de texto incorporado; use OCR se o PDF for imagem.
- **Tempo/memória:** PDFs grandes geram muitos chunks; reduza `chunk_size` ou limite páginas.
