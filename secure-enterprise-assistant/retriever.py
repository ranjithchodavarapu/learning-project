"""
retriever.py — Keyword (TF-IDF) retrieval that only ever sees allowed documents.

Pure Python, no vector database. Project 1 built the full hybrid pipeline;
this project is about SECURITY, so retrieval is kept small and readable.

THE SECURITY POINT:
  search() receives ONLY the documents the user is allowed to read (already
  filtered by rbac.py). It builds its statistics from those documents alone.

  Why IDF over allowed chunks only: IDF ("how rare is this word?") normally
  uses the whole corpus. If it did here, the scores would depend on the text
  of restricted documents. A user could probe with queries and learn which
  words appear in documents they cannot read. Computing IDF only from allowed
  chunks removes that side channel.

LINE BY LINE:
  STOPWORDS
      Very common words ("the", "is") that carry no meaning. Removing them
      keeps scores focused on content words.

  tokenize(text)
      re.findall(r"[a-z0-9]+", text.lower()) = lowercase, then pull out runs of
      letters and digits. Punctuation disappears.

  MAX_CHUNK_CHARS = 200   (deliberately small)
      Chunks are about one paragraph. This is a SECURITY choice. The chunk
      guard drops a whole chunk when it finds an injection. With big chunks,
      one poisoned paragraph would take its harmless neighbours down with it
      (and the user loses legitimate answers). With paragraph-sized chunks
      the poisoned text is isolated and only that chunk is removed.
      Trade-off: smaller chunks carry less context per chunk. For a
      general-purpose RAG you would use larger chunks plus paragraph-level
      scanning; here we keep it simple and strict.

  split_into_chunks(doc)
      Splits the body on blank lines (paragraphs), then merges paragraphs
      only while the chunk stays under MAX_CHUNK_CHARS. Each chunk keeps the
      parent doc_id, title and classification.

  search(query, docs, top_k)
      1. Build the list of chunks from the allowed docs only.
      2. n = number of chunks;  df[t] = in how many chunks term t appears.
      3. idf(t) = log((n + 1) / (df[t] + 1)) + 1.
         The +1 inside avoids division by zero; the +1 at the end keeps the
         weight positive even for words found in every chunk.
      4. tf-weight = 1 + log(count). Repeating a word helps, but with
         diminishing returns (log), so keyword stuffing does not dominate.
      5. score = sum over query terms of tf-weight * idf.
      6. Drop zero scores, sort high to low, return top_k.
      Chunks with score 0 are never returned, so an unrelated question gets
      NO context, and the assistant answers "no information" instead of
      stuffing random text into the prompt.
"""

import math
import re
from collections import Counter
from typing import List

from models import Chunk, Document

MAX_CHUNK_CHARS = 200

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "what", "who", "how", "do", "does", "i", "me", "my",
    "can", "we", "our", "it", "this", "that", "with", "at", "by", "from", "about",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def split_into_chunks(doc: Document) -> List[Chunk]:
    paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=text,
            classification=doc.classification,
        )
        for i, text in enumerate(chunks)
    ]


class Retriever:
    def __init__(self, documents: List[Document]):
        # Chunk every document once. Chunks are looked up by doc_id later.
        self._chunks_by_doc = {d.doc_id: split_into_chunks(d) for d in documents}

    def search(self, query: str, allowed_docs: List[Document], top_k: int) -> List[Chunk]:
        chunks = [c for d in allowed_docs for c in self._chunks_by_doc.get(d.doc_id, [])]
        if not chunks:
            return []

        token_lists = [tokenize(c.text) for c in chunks]
        n = len(chunks)

        df: Counter = Counter()
        for tokens in token_lists:
            df.update(set(tokens))

        query_terms = set(tokenize(query))
        scored = []
        for chunk, tokens in zip(chunks, token_lists):
            tf = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term in tf:
                    idf = math.log((n + 1) / (df[term] + 1)) + 1
                    score += (1 + math.log(tf[term])) * idf
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
