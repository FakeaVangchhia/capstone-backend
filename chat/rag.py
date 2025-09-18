import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
try:
    # Prefer maintained package if installed
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
# Use the new dataset for improved coverage
DATA_PATH = BASE_DIR / "cmru_dataset.json"
CHROMA_DIR = BASE_DIR / ".chroma"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cmru_docs")


class RAGEngine:
    def __init__(self) -> None:
        self._ready = False
        self._docs: List[Dict[str, str]] = []
        self._retriever: object | None = None
        self._vectorstore: Chroma | None = None
        self._data: Dict | None = None

        # Prefer Groq if GROQ_API_KEY present; model can be overridden via OPENAI_MODEL
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._openai_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # LLM client (lazy init). We'll use langchain_groq if key is present
        self._llm = None

    def ensure_ready(self) -> None:
        if self._ready:
            return
        self._load_documents()
        self._build_vectorstore_and_retriever()
        self._init_llm()
        self._ready = True

    def _load_documents(self) -> None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keep raw tree for deterministic lookups (e.g., rankings)
        self._data = data

        chunks: List[Dict[str, str]] = []

        def push(text: str, path: str):
            text = (text or "").strip()
            if text:
                chunks.append({"id": f"{path}-{len(chunks)+1}", "text": text, "source": path})

        def walk(node, path: str):
            if node is None:
                return
            if isinstance(node, (str, int, float)):
                push(str(node), path)
                return
            if isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]")
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    next_path = f"{path}.{k}" if path else k
                    if isinstance(v, (str, int, float)):
                        push(f"{k}: {v}", next_path)
                    else:
                        walk(v, next_path)

        walk(data, "")

        # Further chunk long strings to keep context focused
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        split_chunks: List[Dict[str, str]] = []
        for ch in chunks:
            for i, piece in enumerate(splitter.split_text(ch["text"])):
                split_chunks.append({
                    "id": f"{ch['id']}-p{i+1}",
                    "text": piece,
                    "source": ch["source"],
                })

        self._docs = split_chunks or chunks

    def _build_vectorstore_and_retriever(self) -> None:
        # Build Chroma persistent vector store; on failure, keep BM25 as fallback
        texts = [d["text"] for d in self._docs]
        metadatas = [{"id": d["id"], "source": d["source"]} for d in self._docs]

        # Ensure persistence dir exists
        try:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            # Use a lightweight local embedding model by default
            embed_model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

            # Initialize or connect to collection
            self._vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )

            # Detect if collection is empty and needs indexing
            needs_index = False
            try:
                stats = self._vectorstore.get(limit=1)
                if not stats or len(stats.get("ids", [])) == 0:
                    needs_index = True
            except Exception:
                needs_index = True

            if needs_index and len(texts) > 0:
                # Upsert all docs
                self._vectorstore.add_texts(texts=texts, metadatas=metadatas)
                try:
                    self._vectorstore.persist()
                except Exception:
                    pass

            # Build retriever from vector store
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 6})
            return
        except Exception:
            # If Chroma setup fails, fall back to BM25
            pass

        # BM25 fallback
        try:
            from langchain_core.documents import Document
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = 6
            self._retriever = bm25
        except Exception:
            self._retriever = None

    def _init_llm(self) -> None:
        if self._groq_api_key:
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(api_key=self._groq_api_key, model=self._openai_model, temperature=0.2)
            except Exception:
                # If groq client import fails, keep llm as None to use fallback
                self._llm = None
        else:
            self._llm = None

    def _retrieve(self, query: str, k: int = 6) -> List[Tuple[float, Dict[str, str]]]:
        self.ensure_ready()
        scored: List[Tuple[float, Dict[str, str]]] = []
        try:
            if self._retriever is not None:
                # Support both VectorStoreRetriever and BM25Retriever
                docs = self._retriever.get_relevant_documents(query)
                for rank, d in enumerate(docs[:k]):
                    score = max(0.0, 1.0 - (rank * 0.1))
                    meta = getattr(d, "metadata", {}) or {}
                    text = getattr(d, "page_content", "")
                    scored.append((score, {"id": meta.get("id", ""), "text": text, "source": meta.get("source", "")}))
                if scored:
                    return scored
        except Exception:
            # fall through to keyword fallback
            pass

        # Keyword fallback over self._docs when BM25 is not available
        q = (query or "").lower()
        if not q:
            return []
        keywords = [w for w in q.replace("?", " ").replace(",", " ").split() if len(w) >= 2]
        def score_text(t: str) -> int:
            tl = t.lower()
            return sum(tl.count(w) for w in keywords)
        # rank documents by simple keyword frequency
        ranked = sorted(self._docs, key=lambda d: score_text(d["text"]), reverse=True)[:k]
        out: List[Tuple[float, Dict[str, str]]] = []
        for rank, d in enumerate(ranked):
            sc = max(0.0, 1.0 - rank * 0.1)
            out.append((sc, {"id": d.get("id", ""), "text": d.get("text", ""), "source": d.get("source", "")}))
        return out

    def answer(self, question: str) -> Dict:
        # Handle common greetings and casual conversation
        question_lower = question.lower().strip()
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
        
        if any(greeting in question_lower for greeting in greetings):
            return {
                "answer": "Hello! I'm CollegeGPT, your CMR University assistant. I can help you with information about programmes, campuses, admissions, rankings, and more. What would you like to know?",
                "sources": [],
            }
        
        # Handle identity / capability / help questions BEFORE consulting RAG data
        identity_triggers = [
            'who are you', 'what are you', 'who r u', 'your name', 'about you',
            'tell me about yourself', 'are you a bot', 'what can you do', 'help',
            'what do you do', 'who is this', 'who is that', 'introduce yourself'
        ]
        if any(trigger in question_lower for trigger in identity_triggers):
            return {
                "answer": (
                    "I'm CollegeGPT — a friendly AI assistant for CMR University. "
                    "I can answer questions about programmes, campuses, admissions, rankings, placements, and contact details. "
                    "Ask me something like: admissions status, campus locations, or notable rankings."
                ),
                "sources": [],
            }

        # Handle thank you messages
        if any(word in question_lower for word in ['thank', 'thanks', 'thx']):
            return {
                "answer": "You're welcome! Feel free to ask me anything else about CMR University.",
                "sources": [],
            }

        # Deterministic handler for ranking-related queries (with common misspellings)
        if any(w in question_lower for w in [
            "rank", "ranking", "rankings", "rated", "nirf", "iirf",
            "rand", "randking", "rankding"
        ]):
            try:
                self.ensure_ready()
                rankings = []
                if isinstance(self._data, dict):
                    rankings = self._data.get("rankings", []) or []
                if isinstance(rankings, list) and len(rankings) > 0:
                    # Build a friendly, conversational summary from the dataset
                    bullets = []
                    for item in rankings[:4]:
                        if not isinstance(item, dict):
                            continue
                        r = str(item.get("rank", "")).strip()
                        cat = str(item.get("category", "")).strip()
                        src = str(item.get("source", "")).strip()
                        bullet = " • ".join([p for p in [r, cat, src] if p])
                        if bullet:
                            bullets.append(f"- {bullet}")
                    if bullets:
                        answer = (
                            "Here's how CMR University is rated, based on the official info I have:\n"
                            + "\n".join(bullets)
                            + "\n\nWant details on a specific ranking or source? Happy to share more."
                        )
                        return {"answer": answer, "sources": [{"id": "rankings", "preview": "Rankings from cmru_dataset.json"}]}
            except Exception:
                # fall through to retrieval/LLM path
                pass

        # Improved handler for admissions (including MCA)
        if any(w in question_lower for w in [
            "admission", "admissions", "apply", "application", "process", "how to get in", "how can i apply"
        ]):
            try:
                self.ensure_ready()
                
                # Check if user is asking about MCA specifically
                mentions_mca = any(w in question_lower for w in ["mca", "master of computer applications"])
                
                lines = []
                
                # Provide meaningful admission process steps
                lines.append("Here's the admission process for CMR University:")
                lines.append("- Visit the official CMR University website")
                lines.append("- Fill out the online application form")
                lines.append("- Upload required documents (academic transcripts, ID proof, etc.)")
                lines.append("- Pay the application fee (non-refundable)")
                lines.append("- Submit your application and wait for confirmation")
                lines.append("- Attend counseling/interview if required")
                
                lines.append("")
                lines.append("Important notes:")
                lines.append("- Application fee is non-refundable")
                lines.append("- Use the same email ID throughout the process")
                lines.append("- Admissions for AY 2025-26 are currently open")
                
                if mentions_mca:
                    lines.append("- For MCA program details, please check the specific eligibility criteria")
                
                lines.append("")
                lines.append("For more information and to apply, visit: https://www.cmr.edu.in/campus/")
                lines.append("Admissions Helpline: 93429 00078")

                prefix = "For MCA admission," if mentions_mca else ""
                answer = f"{prefix} here's what you need to know about admissions at CMR University:\n" + "\n".join(lines)
                
                return {
                    "answer": answer,
                    "sources": [
                        {"id": "admissions", "preview": "Admissions information from CMR University dataset"}
                    ],
                }
            except Exception:
                # fall through to retrieval/LLM path
                pass
        
        hits = self._retrieve(question, 6)
        if not hits:
            return {
                "answer": (
                    "I couldn't find relevant information in the CMR University data. "
                    "Ask about programmes, campuses, rankings, admissions, or contact details."
                ),
                "sources": [],
            }

        # Prepare sources and context snippets (limit for brevity)
        sources: List[Dict[str, str]] = []
        snippets: List[str] = []
        for score, ch in hits[:5]:
            snippets.append(f"- {ch['text']}")
        for score, ch in hits[:2]:
            text = ch["text"]
            preview = text[:99] + "…" if len(text) > 100 else text
            sources.append({"id": ch["id"], "preview": preview})

        # Use Groq LLM via LangChain if available
        if self._llm is not None:
            try:
                context_text = "\n".join(snippets)
                if len(context_text) > 4000:
                    context_text = context_text[:4000] + "\n…"

                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are CollegeGPT, a warm and natural-sounding assistant for CMR University. "
                        "Answer ONLY using the provided context. Keep it conversational and helpful. "
                        "Prefer 2–4 short sentences or clear bullets. Avoid sounding robotic. "
                        "If the answer isn't in the context, say you don't know and suggest what to ask instead."
                    )),
                    ("user", (
                        "Question: {question}\n\n"
                        "Context from university data:\n{context}\n\n"
                        "Write a concise, friendly answer now."
                    )),
                ])

                chain = prompt | self._llm
                llm_resp = chain.invoke({"question": question, "context": context_text})
                llm_text = (getattr(llm_resp, "content", None) or str(llm_resp) or "").strip()
                if llm_text:
                    words = llm_text.split()
                    if len(words) > 140:
                        llm_text = " ".join(words[:140]) + " …"
                    return {"answer": llm_text, "sources": sources}
            except Exception:
                # On any failure, fall back to extractive answer below
                pass

        # Fallback: extractive, non-LLM answer (conversational)
        if len(snippets) > 0:
            top_snippet = snippets[0].replace("- ", "")
            if len(top_snippet) > 200:
                top_snippet = top_snippet[:200] + "..."
            answer = (
                f"Here's what I found: {top_snippet}\n\n"
                "If you want, I can pull more details or narrow it down."
            )
        else:
            answer = (
                "I couldn't find a clear answer in the dataset. "
                "Try asking about a specific area, programme, or ranking source."
            )
        
        return {"answer": answer, "sources": sources}


rag = RAGEngine()