import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

# LangChain imports (BM25 requires no external embeddings, lightweight and fast)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "cmr_university_data.json"


class RAGEngine:
    def __init__(self) -> None:
        self._ready = False
        self._docs: List[Dict[str, str]] = []
        self._retriever: BM25Retriever | None = None
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
        self._build_retriever()
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

    def _build_retriever(self) -> None:
        try:
            # BM25Retriever works directly on strings, no embeddings required
            texts = [d["text"] for d in self._docs]
            metadata = [{"id": d["id"], "source": d["source"]} for d in self._docs]
            # LangChain BM25Retriever expects list of Documents; construct minimally
            from langchain_core.documents import Document

            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadata)]
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = 6
            self._retriever = retriever
        except Exception:
            # If BM25 cannot be constructed (e.g., missing dependency), fall back to simple keyword ranking
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
                docs = self._retriever.get_relevant_documents(query)
                # BM25Retriever doesn't expose scores by default; treat order as score rank
                for rank, d in enumerate(docs[:k]):
                    score = max(0.0, 1.0 - (rank * 0.1))
                    scored.append((score, {"id": d.metadata.get("id", ""), "text": d.page_content, "source": d.metadata.get("source", "")}))
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

        # Deterministic handler for ranking-related queries
        if any(w in question_lower for w in ["rank", "ranking", "rankings", "rated", "nirf", "iirf"]):
            try:
                self.ensure_ready()
                rankings = []
                if isinstance(self._data, dict):
                    rankings = self._data.get("rankings", []) or []
                if isinstance(rankings, list) and len(rankings) > 0:
                    # Build a concise summary
                    lines = []
                    for item in rankings[:4]:
                        if not isinstance(item, dict):
                            continue
                        r = str(item.get("rank", "")).strip()
                        cat = str(item.get("category", "")).strip()
                        src = str(item.get("source", "")).strip()
                        bullet = " • ".join([p for p in [r, cat, src] if p])
                        if bullet:
                            lines.append(f"- {bullet}")
                    if lines:
                        answer = "Here are CMR University's notable rankings and ratings:\n" + "\n".join(lines)
                        return {"answer": answer, "sources": [{"id": "rankings", "preview": "CMR University rankings from campus data"}]}
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
                        "You are CollegeGPT, a friendly and concise assistant for CMR University. "
                        "Answer ONLY using the provided context. Keep answers under 120 words. "
                        "Use a short paragraph and up to 3 bullets if useful. "
                        "If the answer isn't in the context, say you don't know."
                    )),
                    ("user", (
                        "Question: {question}\n\n"
                        "Context from university data:\n{context}\n\n"
                        "Write a concise, student-friendly answer now."
                    )),
                ])

                chain = prompt | self._llm
                llm_resp = chain.invoke({"question": question, "context": context_text})
                llm_text = (getattr(llm_resp, "content", None) or str(llm_resp) or "").strip()
                if llm_text:
                    words = llm_text.split()
                    if len(words) > 120:
                        llm_text = " ".join(words[:120]) + " …"
                    return {"answer": llm_text, "sources": sources}
            except Exception:
                # On any failure, fall back to extractive answer below
                pass

        # Fallback: extractive, non-LLM answer (much more concise)
        if len(snippets) > 0:
            # Just return the most relevant snippet, not all of them
            top_snippet = snippets[0].replace("- ", "")
            if len(top_snippet) > 200:
                top_snippet = top_snippet[:200] + "..."
            answer = f"{top_snippet}\n\nFor more details, please ask a more specific question."
        else:
            answer = "I found some information but couldn't extract a clear answer. Please try rephrasing your question."
        
        return {"answer": answer, "sources": sources}


rag = RAGEngine()
