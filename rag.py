import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from mistralai import Mistral
import textwrap
import time

# LLM variables
mistral_model = "mistral-small-latest"
embedding_model_name = "intfloat/multilingual-e5-base"
crossencoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# API
mistral_api_key = os.getenv("MISTRAL_API_KEY")
total_pages = 0  # filled when loading the PDF
page_texts = []  # raw text for each page
embed_model = None
index = None
cross = None
chunks = []
chunk_texts = []


def load_pdf(file_path):
    reader = PdfReader(file_path)  # opens the pdf
    texts = []  # stores the text from each page
    offsets = []  # stores the starting index for each page
    cursor = 0  # accumulated character position
    for page in reader.pages:  # iterate through pages
        page_text = page.extract_text() or ""  # extract page text
        offsets.append(cursor)  # record where the page starts
        texts.append(page_text)  # add the text for this page
        cursor += len(page_text) + 1  # advance cursor +1 for newline
    full_text = "\n".join(texts)  # concatenate pages with line breaks
    return full_text, offsets, texts  # return full text, offsets, and per-page texts


def find_page(start_idx, page_offsets):
    page = 0  # page index
    for i, off in enumerate(page_offsets):  # iterate over offsets
        if start_idx < off:  # reached next page start
            break
        page = i  # update current page
    return page  # return page index


def split_chunks(text, page_offsets, chunk_size=2000, overlap=400):
    if chunk_size <= overlap:  # validate chunk/overlap limits
        raise ValueError("Document smaller than the minimum chunk allowed")
    chunks = []  # list to receive chunks
    step = chunk_size - overlap  # stride considering overlap
    for start in range(0, len(text), step):  # iterate text
        end = start + chunk_size  # calculate chunk end
        chunk_text = text[start:end]  # slice text
        page = find_page(start, page_offsets)  # find originating page
        chunks.append({"page": page, "text": chunk_text})  # store page with chunk text
    return chunks  # return chunks
    print("PDF WITH MARKED CHUNKS")


def rag(file_path):
    global total_pages, page_texts
    pdf_text, offsets, page_texts = load_pdf(file_path)  # load full text
    total_pages = len(offsets)
    chunks = split_chunks(pdf_text, offsets, chunk_size=2000, overlap=400)
    chunk_texts = [c["text"] for c in chunks]
    embed_model = SentenceTransformer(embedding_model_name)
    print("EMBEDDING MODEL LOADED")
    embed_texts = [f"passage: {txt}" for txt in chunk_texts]
    embs = embed_model.encode(embed_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    d = embs.shape[1]  # vector dimension for faiss
    index = faiss.IndexFlatIP(d)  # create faiss index
    index.add(embs)  # add all vectors to faiss index
    print("FAISS INDEX CREATED")
    cross = CrossEncoder(crossencoder_model)
    print("CROSS-ENCODER LOADED")
    return embed_model, index, cross, chunks, chunk_texts


def search_rerank(question, k_base=30, k_final=3):  # search faiss and rerank top 3 among top 30
    global embed_model, index, cross, chunks, chunk_texts
    results = []
    question_fmt = f"query: {question}"  # expected by e5 for queries
    q_emb = embed_model.encode([question_fmt], convert_to_numpy=True, normalize_embeddings=True).astype("float32")  # question embedding
    scores, idxs = index.search(q_emb, k_base)  # candidate retrieval
    pairs = [(question, chunk_texts[i]) for i in idxs[0]]  # build question/chunk pairs for rerank
    rerank_scores = cross.predict(pairs)  # more accurate cross-encoder score
    reranked = sorted(zip(rerank_scores, idxs[0]), key=lambda x: x[0], reverse=True)[:k_final]  # sort by score
    for score, i in reranked:
        results.append({
            "score": float(score),  # cross-encoder score
            "page": chunks[i]["page"],  # source page
            "text": chunks[i]["text"],  # chunk text
        })
    return results  # return ranked chunks


def build_mistral_prompt(question, contexts):
    blocks = []  # accumulate formatted chunks with page number
    for ctx in contexts:  # iterate selected contexts
        blocks.append(f"[Page {ctx['page']}] {ctx['text']}")  # record page and text
    context_text = "\n\n".join(blocks)  # join chunks separated by blank lines
    return (
        "Summarize or answer the question using only the provided context. "  # instruction to the model
        "If there is no answer, say you don't know.\n\n"
        f"Question: {question}\n\n"  # insert user question
        f"Context:\n{context_text}"  # insert PDF excerpts
    )


def mistral_chat(question, contexts):
    api_key = mistral_api_key
    if not api_key:  # validate if key is set
        raise RuntimeError("Set the MISTRAL_API_KEY environment variable")  # alert to configure
    client = Mistral(api_key=api_key)  # instantiate Mistral client
    print("MISTRAL MODEL LOADED")
    prompt_text = build_mistral_prompt(question, contexts)  # build prompt with question and contexts
    resp = client.chat.complete(
        model=mistral_model,  # chosen model
        messages=[{"role": "user", "content": prompt_text}],  # user message
        temperature=0.2,  # low temperature for faithful responses
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage  # return response text


def generate_chunk_title(chunk_text, attempts=5, base_delay=2):
    api_key = mistral_api_key
    client = Mistral(api_key=api_key)

    prompt = (
        "Generate ONLY ONE SHORT TITLE for the text below.\n"
        "Title rules:\n"
        "- It must have between 2 and 5 words.\n"
        "- Do not describe long sentences.\n"
        "- Do not write explanations.\n"
        "- Do not use colons.\n\n"
        f"Text:\n{chunk_text}"
    )

    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            status = getattr(e, "http_status", None) or getattr(e, "status_code", None)
            if (status == 429) or ("rate limit" in msg) or ("too many requests" in msg):
                if attempt == attempts:
                    raise
                wait_time = base_delay * (2 ** (attempt - 1))
                print(f"Rate limit; waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time)
                continue
            raise


def generate_index(chunks):
    displayed_pages = set()
    for c in chunks:
        page = c["page"]
        if page in displayed_pages:
            continue
        displayed_pages.add(page)
        chunk_text = c["text"]
        title = generate_chunk_title(chunk_text)
        print(f"Page {page}: {title}")


def format_text(text, width=80):
    lines = []
    for paragraph in text.splitlines():
        for line in textwrap.wrap(paragraph, width):
            if len(line) < width - len(line):
               deficit = width - len(line)
               gaps = line.count("")
               parts = line.split("")
               base, remainder = divmod(deficit, gaps)
               for i in range(remainder):
                   parts[i] += " "
               line = (" " * base).join(parts)
            lines.append(line)
    return "\n".join(lines)


def show_page(chunks, human_page, trim=None):
    page_idx = human_page - 1  # align human count (1…) with internal index (0…)
    if total_pages and (page_idx < 0 or page_idx >= total_pages):
        print(f"The document has only {total_pages} pages.")
        return
    if page_texts and 0 <= page_idx < len(page_texts):
        text = page_texts[page_idx]
        if trim and len(text) > trim:
            text = text[:trim].rstrip() + "..."
        print(f"\nPage {human_page} (full text):\n{text}")
        return
    page_chunks = [c for c in chunks if c["page"] == page_idx]
    if not page_chunks:
        print(f"No chunk found on page {human_page}")
        return
    for i, c in enumerate(page_chunks, 1):
        text = c["text"]
        text = format_text(text, width=80)
        if trim and len(text) > trim:
            text = text[:trim].rstrip() + "..."
        print(f"\nChunk {i} (page {human_page}):\n{text}")


def choose_page():
    chosen_page = input("Enter the page number you want to read (e.g., 1, 2, 3...): ")
    try:
        num = int(chosen_page)
        if num <= 0:
            raise ValueError
    except ValueError:
        print("Enter a valid page number (>= 1).")
        return
    if total_pages and num > total_pages:
        print(f"The document has only {total_pages} pages.")
        return
    show_page(chunks, human_page=num)   


def chat_menu():
    while True:
        print("\n=== Function Menu ===")
        print("1) Load PDF")
        print("2) Chat with the PDF")
        print("3) Generate Semantic Index")
        print("4) Show specific page")
        print("5) Exit")

        option = input("Choose a menu number: ").strip()

        if option == "1":
            file_path = input("Enter the PDF path or type 'back': ").strip().strip('"')
            if file_path == "back":
                return chat_menu()
            if not file_path:
                print("No file provided.")
                return
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                return
            global embed_model, index, cross, chunks, chunk_texts
            embed_model, index, cross, chunks, chunk_texts = rag(file_path)
        elif option == "2":
            if not chunks:
                print("Load a PDF first")
                return chat_menu()
            while True:
                question = input("Type your question or 'back':  ").strip().lower()
                if question == "back":
                    return chat_menu()
                results = search_rerank(question)
                answer, usage = mistral_chat(question, results)
                print("\nANSWER (Mistral):\n")
                print(answer)
                print(f"\nTokens - input: {usage.prompt_tokens}, output: {usage.completion_tokens}, total: {usage.total_tokens}")
        elif option == "3":
            if not chunks:
                print("Load a PDF first")
                continue
            generate_index(chunks)
        elif option == "4":
            if not chunks:
                print("Load a PDF first")
                continue
            choose_page()
        elif option == "5":
            break
        
if __name__ == "__main__":  # run main when executed directly
    chat_menu()  # start main flow
