# RAG em PDF via Terminal

CLI em Python que lê um PDF local, divide em trechos, indexa no FAISS, reranqueia com CrossEncoder e responde via chat da Mistral. Funciona em loop: você informa um PDF, faz perguntas, pode gerar um índice rápido das páginas, ler páginas específicas ou trocar de arquivo sem reiniciar.

## Requisitos
- Python 3.10+
- Variável de ambiente `MISTRAL_API_KEY` com a chave da Mistral
- Dependências: `pypdf`, `sentence_transformers`, `faiss-cpu`, `mistralai`, `numpy`

## Instalação rápida
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pypdf sentence_transformers faiss-cpu mistralai numpy
```

Configure a chave da API:
```bash
export MISTRAL_API_KEY="sua-chave-aqui"  # Windows: set MISTRAL_API_KEY=sua-chave-aqui
```

## Como usar
```bash
python rag.py
```
1) Informe o caminho do PDF (pode estar na mesma pasta).  
2) Faça perguntas ou use um dos comandos abaixo.  
3) Digite `sair` para encerrar.

Comandos do chat:
- Perguntas livres: "Qual o resumo do capítulo 1?"
- `gerar indice`: cria títulos curtos para cada página (consome API da Mistral).
- `ler pagina`: pede um número e mostra o texto completo daquela página.
- `trocar pdf`: volta a pedir um novo arquivo e reinicia o índice.
- `sair`: finaliza o script.

Exemplo rápido:
```bash
export MISTRAL_API_KEY="minha-chave"
python rag.py
# caminho do PDF: /caminho/para/documento.pdf
# pergunta: Quais são as conclusoes?
```

## Como funciona
- Chunking: `divisor_trechos` cria janelas de 2000 caracteres com overlap de 400 e registra a página de origem.
- Busca híbrida: embeddings `intfloat/multilingual-e5-base` normalizados + índice FAISS (Inner Product).
- Rerank: top-30 candidatos reranqueados com `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Geração: prompt compacto para `mistral-small-latest` com contagem de tokens impressa.
- Leitura direta: `ler pagina` usa o texto extraído bruto de cada página quando disponível.

## Ajustes rápidos
- Tamanho dos trechos: altere `chunk_size` e `overlap` em `divisor_trechos`.
- Quantidade de candidatos: ajuste `k_base` e `k_final` em `busca_rerank`.
- Temperatura do modelo: modifique `temperature` na chamada `mistral`.

## Problemas comuns
- **Chave não configurada:** "Defina a API MISTRAL" indica `MISTRAL_API_KEY` ausente ou vazia.
- **PDF sem texto extraível:** `pypdf` precisa de texto incorporado; use OCR se o arquivo for imagem.
- **Tempo/memória:** PDFs grandes geram muitos chunks; reduza `chunk_size`, limite páginas ou troque de PDF com `trocar pdf`.
