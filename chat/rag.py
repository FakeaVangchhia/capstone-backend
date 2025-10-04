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
SYLLABUS_PATH = BASE_DIR / "mca_syllabus.json"
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

    # Public method to upsert arbitrary text chunks (e.g., newly uploaded docs)
    def upsert_texts(self, texts: List[str], source: str = "uploaded") -> int:
        if not texts:
            return 0
        # Ensure vector store exists
        if self._vectorstore is None or self._retriever is None:
            # initialize with empty docs
            self._docs = []
            self._build_vectorstore_and_retriever()
        try:
            metadatas = [{"id": f"{source}:{i}", "source": source} for i, _ in enumerate(texts)]
            self._vectorstore.add_texts(texts=texts, metadatas=metadatas)
            try:
                self._vectorstore.persist()
            except Exception:
                pass
            # refresh retriever
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 6})
            return len(texts)
        except Exception:
            return 0

    def _load_documents(self) -> None:
        # Load university data
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keep raw tree for deterministic lookups (e.g., rankings)
        self._data = data

        # Load MCA syllabus data
        syllabus_data = None
        try:
            with open(SYLLABUS_PATH, "r", encoding="utf-8") as f:
                syllabus_data = json.load(f)
            self._syllabus_data = syllabus_data
        except Exception:
            self._syllabus_data = None

        chunks: List[Dict[str, str]] = []

        def push(text: str, path: str, source_type: str = "university"):
            text = (text or "").strip()
            if text:
                chunks.append({
                    "id": f"{source_type}:{path}-{len(chunks)+1}", 
                    "text": text, 
                    "source": path,
                    "source_type": source_type
                })

        def walk(node, path: str, source_type: str = "university"):
            if node is None:
                return
            if isinstance(node, (str, int, float)):
                push(str(node), path, source_type)
                return
            if isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]", source_type)
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    next_path = f"{path}.{k}" if path else k
                    if isinstance(v, (str, int, float)):
                        push(f"{k}: {v}", next_path, source_type)
                    else:
                        walk(v, next_path, source_type)

        # Process university data
        walk(data, "", "university")
        
        # Process syllabus data
        if syllabus_data:
            walk(syllabus_data, "", "syllabus")

        # Further chunk long strings to keep context focused
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        split_chunks: List[Dict[str, str]] = []
        for ch in chunks:
            for i, piece in enumerate(splitter.split_text(ch["text"])):
                split_chunks.append({
                    "id": f"{ch['id']}-p{i+1}",
                    "text": piece,
                    "source": ch["source"],
                    "source_type": ch.get("source_type", "university"),
                })

        self._docs = split_chunks or chunks

    def _build_vectorstore_and_retriever(self) -> None:
        # Build Chroma persistent vector store; on failure, keep BM25 as fallback
        texts = [d["text"] for d in self._docs]
        metadatas = [{"id": d["id"], "source": d["source"], "source_type": d.get("source_type", "university")} for d in self._docs]

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
        
        # Use word boundaries to avoid false matches (like "hi" in "machine")
        import re
        greeting_patterns = [r'\b' + re.escape(greeting) + r'\b' for greeting in greetings]
        if any(re.search(pattern, question_lower) for pattern in greeting_patterns):
            return {
                "answer": "Hey there! 👋 I'm your CMR University Study Buddy - think of me as your friendly AI companion here to help with everything university-related! Whether you need info about courses, campus life, study tips, or just want to chat about university stuff, I'm here for you. What's on your mind today?",
                "sources": [],
            }
        
        # Handle emotional/feeling questions to clarify AI nature
        feeling_triggers = [
            'how are you feeling', 'are you sad', 'are you happy', 'do you feel', 'your feelings',
            'are you okay', 'how do you feel', 'are you tired', 'are you bored', 'emotions',
            'mood', 'are you angry', 'are you excited', 'do you have feelings'
        ]
        if any(trigger in question_lower for trigger in feeling_triggers):
            return {
                "answer": (
                    "I'm an AI designed specifically for CMR University, so I don't have feelings or emotions like humans do! 😊 "
                    "But I'm always here and ready to help you with any questions about CMR University - whether it's about programmes, admissions, campus life, or anything else related to your studies. "
                    "What would you like to know about CMR University?"
                ),
                "sources": [],
            }

        # Handle academic/study preparation queries FIRST (before identity triggers)
        study_prep_triggers = [
            'first test', 'second test', 'mid term', 'end term', 'exam preparation',
            'what should i study', 'test preparation', 'exam topics', 'syllabus',
            'unit 1', 'unit 2', 'unit 3', 'unit 4', 'unit 5', 'module',
            'dsa', 'data structures', 'algorithms', 'database', 'java', 'python',
            'machine learning', 'artificial intelligence', 'operating system',
            'prepare for', 'topics for', 'study for', 'help me prepare', 'preparation guide'
        ]
        if any(trigger in question_lower for trigger in study_prep_triggers):
            return self._handle_study_preparation_query(question)

        # Handle identity / capability / help questions (but exclude study-related help)
        identity_triggers = [
            'who are you', 'what are you', 'who r u', 'your name', 'about you',
            'tell me about yourself', 'are you a bot', 'what can you do',
            'what do you do', 'who is this', 'who is that', 'introduce yourself'
        ]
        # Only trigger if it's a general help request, not study-specific
        if any(trigger in question_lower for trigger in identity_triggers) or (
            'help' in question_lower and not any(study_word in question_lower for study_word in ['prepare', 'study', 'exam', 'test'])
        ):
            return {
                "answer": (
                    "I'm your CMR University Study Buddy! 🎓✨ Think of me as your personal AI companion designed to help you succeed at CMR University. "
                    "I can help with:\n"
                    "📚 Course info & academic guidance\n"
                    "🏫 Campus details & facilities\n"
                    "📝 Admissions & application process\n"
                    "🏆 Rankings & achievements\n"
                    "💼 Placements & career guidance\n"
                    "🏠 Hostel & campus life info\n"
                    "📞 Contact details for different departments\n\n"
                    "Plus, I'm here to chat and support you through your university journey! What would you like to know?"
                ),
                "sources": [],
            }

        # Handle thank you messages
        if any(word in question_lower for word in ['thank', 'thanks', 'thx']):
            return {
                "answer": "You're so welcome! 😊 That's what study buddies are for! I'm always here whenever you need help with anything CMR University related. Keep crushing it! 💪",
                "sources": [],
            }

        # Handle study-related queries and academic support
        study_triggers = [
            'study tips', 'how to study', 'exam preparation', 'study schedule', 'study plan',
            'time management', 'study methods', 'study techniques', 'academic help',
            'struggling with studies', 'study motivation', 'exam stress', 'study habits'
        ]
        if any(trigger in question_lower for trigger in study_triggers):
            return {
                "answer": (
                    "Great question! As your study buddy, here are some effective study tips for CMR University students:\n\n"
                    "📚 **Study Strategies:**\n"
                    "• Use the Pomodoro Technique (25 min study + 5 min break)\n"
                    "• Create a consistent study schedule that works with your class timings\n"
                    "• Form study groups with classmates - collaborative learning is powerful!\n"
                    "• Use active recall instead of just re-reading notes\n\n"
                    "🎯 **CMR University Resources:**\n"
                    "• Take advantage of the library facilities on campus\n"
                    "• Join study groups in common areas - great for networking too!\n"
                    "• Use the high-speed Wi-Fi in hostels for online resources\n\n"
                    "💡 **Pro Tips:**\n"
                    "• Break big topics into smaller chunks\n"
                    "• Teach concepts to others - it reinforces your own learning\n"
                    "• Don't forget to take care of your mental health!\n\n"
                    "Need specific help with any subject or study challenge? I'm here to brainstorm solutions with you! 🤝"
                ),
                "sources": [],
            }

        # Handle campus life and social queries
        campus_life_triggers = [
            'campus life', 'student life', 'activities', 'clubs', 'events', 'social life',
            'making friends', 'hostel life', 'what to do on campus', 'bored', 'lonely',
            'extracurricular', 'sports', 'cultural activities'
        ]
        if any(trigger in question_lower for trigger in campus_life_triggers):
            return {
                "answer": (
                    "Campus life at CMR University is pretty awesome! 🌟 Here's what you can look forward to:\n\n"
                    "🏫 **Amazing Campus Features:**\n"
                    "• Beautiful 60-acre lakefront campus in North Bangalore\n"
                    "• International standard indoor sports complex\n"
                    "• Separate cricket, athletics, and football grounds\n"
                    "• Spaces for cultural activities and events\n\n"
                    "🏠 **Hostel Life:**\n"
                    "• Boys and girls hostels in vibrant Banaswadi area\n"
                    "• High-speed internet and dining facilities\n"
                    "• Single, double, triple, and four-sharing options\n"
                    "• Indoor games and recreational facilities\n\n"
                    "🎯 **Get Involved:**\n"
                    "• Join the LEAP program (Learning Engagement & Advancement Programme)\n"
                    "• Participate in cross-disciplinary learning activities\n"
                    "• Take advantage of the Office of Student Affairs programs\n\n"
                    "The key is to put yourself out there and join activities that interest you. You'll make great friends and memories! Want specific info about any particular aspect of campus life? 😊"
                ),
                "sources": [{"id": "campus_life", "preview": "Campus life information from CMR University dataset"}],
            }

        # Handle career and placement queries
        career_triggers = [
            'placement', 'placements', 'job', 'career', 'internship', 'companies',
            'recruiters', 'salary', 'employment', 'career guidance', 'job opportunities'
        ]
        if any(trigger in question_lower for trigger in career_triggers):
            return {
                "answer": (
                    "Great to see you thinking about your career! 🚀 CMR University has solid placement support:\n\n"
                    "💼 **Placement Highlights:**\n"
                    "• 200+ recruiting companies\n"
                    "• Pre-placement modules from first semester\n"
                    "• Strong emphasis on experiential learning and internships\n"
                    "• Training for competitive exams (CAT, GRE, TOEFL, CMAT, Bank PO)\n\n"
                    "📞 **Placement Team Contacts:**\n"
                    "• Lakeside Campus: Mr. Gopalakrishna M (9513444634)\n"
                    "• City Campus: Mr. Saravana Prasad V (9600545614)\n"
                    "• OMBR Campus: Mr. Parthiban S (7904552643)\n\n"
                    "🎯 **My Advice:**\n"
                    "• Start building your skills early - don't wait till final year!\n"
                    "• Participate in internships and projects\n"
                    "• Network with seniors and alumni\n"
                    "• Keep your resume updated and practice interview skills\n\n"
                    "Want specific guidance for your field of study? I can help you plan your career path! 💪"
                ),
                "sources": [{"id": "placements", "preview": "Placement information from CMR University dataset"}],
            }

        # Handle motivation and encouragement requests
        motivation_triggers = [
            'motivate me', 'motivation', 'encourage', 'feeling down', 'stressed',
            'overwhelmed', 'give up', 'difficult', 'hard', 'struggling', 'demotivated',
            'inspire me', 'need encouragement', 'feeling low', 'can\'t do this'
        ]
        if any(trigger in question_lower for trigger in motivation_triggers):
            return {
                "answer": (
                    f"Hey, I hear you! 🤗 University can be challenging, but you're not alone in this journey.\n\n"
                    f"{self.get_study_motivation()}\n\n"
                    "Remember:\n"
                    "• Every CMR University student faces challenges - it's part of growth! 🌱\n"
                    "• You have amazing resources: great faculty, supportive peers, and excellent facilities\n"
                    "• Take it one day at a time, one assignment at a time\n"
                    "• Don't hesitate to reach out to classmates, seniors, or faculty for help\n\n"
                    "You chose CMR University for a reason - trust in your decision and in yourself! "
                    "What specific challenge can I help you tackle today? 💪"
                ),
                "sources": [],
            }

        # Handle study tips requests
        study_tip_triggers = [
            'study tip', 'study tips', 'tip of the day', 'daily tip', 'study advice',
            'how to be better student', 'improve studying', 'study better'
        ]
        if any(trigger in question_lower for trigger in study_tip_triggers):
            return {
                "answer": (
                    f"{self.get_study_tip_of_day()}\n\n"
                    "Want more personalized study advice? Tell me:\n"
                    "• What subject are you working on?\n"
                    "• What specific challenge are you facing?\n"
                    "• Are you a visual, auditory, or hands-on learner?\n\n"
                    "I'm here to help you develop the perfect study strategy for your learning style! 📚✨"
                ),
                "sources": [],
            }

        # Handle program/course specific queries
        program_triggers = [
            'mca', 'master of computer applications', 'btech', 'bachelor of technology',
            'bba', 'bachelor of business administration', 'bcom', 'bachelor of commerce',
            'bdes', 'bachelor of design', 'barch', 'bachelor of architecture',
            'course', 'program', 'programme', 'degree', 'major', 'specialization'
        ]
        if any(trigger in question_lower for trigger in program_triggers):
            # Let this fall through to RAG retrieval for specific program info
            pass

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

        # Improved handler for admissions (including MCA) - Study buddy approach
        if any(w in question_lower for w in [
            "admission", "admissions", "apply", "application", "process", "how to get in", "how can i apply"
        ]):
            try:
                self.ensure_ready()
                
                # Check if user is asking about MCA specifically
                mentions_mca = any(w in question_lower for w in ["mca", "master of computer applications"])
                
                answer_parts = []
                
                if mentions_mca:
                    answer_parts.append("Awesome choice considering MCA at CMR University! 🎓 It's a great program for tech careers.")
                else:
                    answer_parts.append("Thinking about joining CMR University? Great choice! 🌟 Let me walk you through the admission process:")
                
                answer_parts.append("")
                answer_parts.append("📝 **Step-by-Step Admission Process:**")
                answer_parts.append("1. Visit the official CMR University website")
                answer_parts.append("2. Fill out the online application form carefully")
                answer_parts.append("3. Upload all required documents (transcripts, ID proof, etc.)")
                answer_parts.append("4. Pay the application fee (heads up - it's non-refundable)")
                answer_parts.append("5. Submit and wait for confirmation")
                answer_parts.append("6. Attend counseling/interview if called")
                
                answer_parts.append("")
                answer_parts.append("⚠️ **Important Tips from Your Study Buddy:**")
                answer_parts.append("• Use the same email throughout - don't change it!")
                answer_parts.append("• Keep digital copies of all documents handy")
                answer_parts.append("• Admissions for AY 2025-26 are currently open")
                answer_parts.append("• Apply early - don't wait till the last minute!")
                
                if mentions_mca:
                    answer_parts.append("• Check MCA-specific eligibility criteria carefully")
                
                answer_parts.append("")
                answer_parts.append("📞 **Need Help?**")
                answer_parts.append("• Website: https://www.cmr.edu.in/campus/")
                answer_parts.append("• Admissions Helpline: 93429 00078")
                answer_parts.append("")
                answer_parts.append("You've got this! The application process might seem overwhelming, but take it step by step. Feel free to ask if you need help with any specific part! 💪")

                answer = "\n".join(answer_parts)
                
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
                    "I couldn't find relevant information in the CMR University data for that question. 😊 "
                    "Try asking about programmes, campuses, rankings, admissions, or contact details - I'm here to help with all things CMR University!"
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
                        "You are the CMR University Study Buddy, a friendly AI companion designed to support students at CMR University. "
                        "You're like a helpful senior student who knows everything about the university and genuinely cares about helping others succeed. "
                        "Be warm, encouraging, and conversational. Use emojis to be friendly and relatable. "
                        "Answer using the provided context about CMR University, but frame responses as a supportive study partner would. "
                        "Give practical advice, be encouraging about challenges, and celebrate successes. "
                        "Keep responses conversational (2-4 sentences or clear bullets). "
                        "If you don't know something, suggest alternatives or direct them to the right resources. "
                        "Always maintain a positive, supportive tone that makes students feel they have a reliable friend at university."
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

        # Fallback: extractive, non-LLM answer (conversational and supportive)
        if len(snippets) > 0:
            top_snippet = snippets[0].replace("- ", "")
            if len(top_snippet) > 200:
                top_snippet = top_snippet[:200] + "..."
            answer = (
                f"Here's what I found for you: {top_snippet}\n\n"
                "Need me to dig deeper into any specific part? I'm here to help you get exactly what you need! 😊"
            )
        else:
            answer = (
                "Hmm, I couldn't find specific info about that in my CMR University knowledge base. 🤔 "
                "But don't worry! Try asking about:\n"
                "• Specific programs or courses\n"
                "• Campus facilities or hostel info\n"
                "• Admissions or placement details\n"
                "• Study tips or academic support\n\n"
                "I'm your study buddy - we'll figure this out together! 💪"
            )
        
        return {"answer": answer, "sources": sources}

    def get_study_motivation(self) -> str:
        """Return motivational messages for students"""
        motivations = [
            "You've got this! 💪 Every expert was once a beginner. Keep pushing forward!",
            "Remember why you started! 🌟 Your future self will thank you for the effort you put in today.",
            "Progress, not perfection! 📈 Every small step counts towards your bigger goals.",
            "Believe in yourself! 🚀 You're capable of amazing things at CMR University.",
            "Tough times don't last, but tough students do! 💎 You're stronger than you think.",
            "Your education is an investment in yourself! 📚 The best investment you'll ever make.",
            "Success is not final, failure is not fatal - it's the courage to continue that counts! ⭐",
            "You're not just studying, you're building your future! 🏗️ Keep building strong!"
        ]
        import random
        return random.choice(motivations)

    def get_study_tip_of_day(self) -> str:
        """Return daily study tips"""
        tips = [
            "💡 **Tip of the Day:** Use the 2-minute rule - if something takes less than 2 minutes, do it now!",
            "💡 **Tip of the Day:** Create a dedicated study space in your hostel room for better focus.",
            "💡 **Tip of the Day:** Review your notes within 24 hours of class - it boosts retention by 60%!",
            "💡 **Tip of the Day:** Take regular breaks! Your brain needs rest to process information effectively.",
            "💡 **Tip of the Day:** Teach someone else what you learned - it's the best way to solidify knowledge.",
            "💡 **Tip of the Day:** Use the campus library during peak hours - the study atmosphere is contagious!",
            "💡 **Tip of the Day:** Start assignments early and break them into smaller tasks.",
            "💡 **Tip of the Day:** Join study groups - collaborative learning makes difficult topics easier!"
        ]
        import random
        return random.choice(tips)

    def _handle_study_preparation_query(self, question: str) -> Dict:
        """Handle specific study preparation and syllabus queries"""
        question_lower = question.lower().strip()
        
        # Ensure RAG is ready (this loads syllabus data)
        self.ensure_ready()
        
        # Check if syllabus data is available
        if not hasattr(self, '_syllabus_data') or not self._syllabus_data:
            return {
                "answer": (
                    "I'd love to help you with study preparation! 📚 However, I don't have access to the detailed syllabus right now. "
                    "But here's what I can suggest:\n\n"
                    "🎯 **General Test Prep Strategy:**\n"
                    "• Start with fundamental concepts first\n"
                    "• Focus on the first 2-3 units/modules for early tests\n"
                    "• Practice coding problems if it's a programming subject\n"
                    "• Review lecture notes and textbook examples\n\n"
                    "For specific MCA subjects like DSA, Database, Java, etc., I can give you targeted study tips! "
                    "What subject are you preparing for? 🤔"
                ),
                "sources": [],
            }

        # Handle DSA (Data Structures and Algorithms) queries
        if any(term in question_lower for term in ['dsa', 'data structures', 'algorithms']):
            return self._get_dsa_study_guide(question_lower)
        
        # Handle Database queries - more comprehensive detection
        if any(term in question_lower for term in ['database', 'dbms', 'sql', 'db ', 'prepare for database']):
            return self._get_database_study_guide(question_lower)
        
        # Handle Java queries
        if any(term in question_lower for term in ['java', 'advanced java']):
            return self._get_java_study_guide(question_lower)
        
        # Handle Machine Learning queries - more comprehensive detection
        if any(term in question_lower for term in ['machine learning', 'ml', 'python', 'second semester', 'topics for second']):
            return self._get_ml_study_guide(question_lower)
        
        # Handle Operating System queries
        if any(term in question_lower for term in ['operating system', 'os']):
            return self._get_os_study_guide(question_lower)
        
        # General syllabus query - this should always return syllabus-based response
        return self._get_general_study_guide(question_lower)

    def _get_dsa_study_guide(self, question_lower: str) -> Dict:
        """Provide DSA-specific study guidance"""
        if any(term in question_lower for term in ['first test', 'unit 1', 'unit 2', 'module 1', 'module 2']):
            return {
                "answer": (
                    "Great! For your DSA first test, focus on these key topics from the syllabus: 📊\n\n"
                    "🎯 **Unit 1: Introduction to Data Structures (9 Hours)**\n"
                    "• Review of basic data structures\n"
                    "• Abstract data types (ADT)\n"
                    "• Big Oh, Small Oh, Omega and Theta notations\n"
                    "• Solving recurrence relations\n"
                    "• Master theorems\n\n"
                    "🎯 **Unit 2: Hashing and Priority Queues (9 Hours)**\n"
                    "• Hash functions and collision resolution\n"
                    "• Separate chaining, open addressing\n"
                    "• Linear probing, quadratic probing, double hashing\n"
                    "• Priority queues using heaps\n"
                    "• Heap insertion and deletion\n\n"
                    "💡 **Study Tips:**\n"
                    "• Practice time complexity analysis problems\n"
                    "• Implement basic hash table operations in C\n"
                    "• Understand heap operations thoroughly\n"
                    "• Solve recurrence relation examples\n\n"
                    "Focus on these first two units - they're typically covered in the first test! Need help with any specific topic? 🤓"
                ),
                "sources": [{"id": "mca_syllabus", "preview": "Advanced Data Structures using C - First Semester"}],
            }
        
        return {
            "answer": (
                "For DSA preparation, here's the complete roadmap based on your MCA syllabus: 🗺️\n\n"
                "📚 **All 5 Units to Master:**\n"
                "1. **Introduction** - ADT, complexity analysis, recurrence relations\n"
                "2. **Hashing & Priority Queues** - Hash functions, heaps\n"
                "3. **Trees** - BST, AVL, Red-Black, B-trees, Tries\n"
                "4. **Graphs** - Traversals, shortest paths, MST algorithms\n"
                "5. **Algorithm Design** - Greedy, divide & conquer, dynamic programming\n\n"
                "🎯 **Lab Practice Focus:**\n"
                "• Implement all major data structures in C\n"
                "• Practice Dijkstra's, Prim's, and Kruskal's algorithms\n"
                "• Work on tree implementations (AVL, Red-Black)\n\n"
                "Which specific unit or topic would you like me to break down further? 💪"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "Complete DSA syllabus - MCA First Semester"}],
        }

    def _get_database_study_guide(self, question_lower: str) -> Dict:
        """Provide Database-specific study guidance"""
        if any(term in question_lower for term in ['first test', 'unit 1', 'unit 2', 'module 1', 'module 2']):
            return {
                "answer": (
                    "Perfect! For your Database first test, concentrate on these foundational topics: 🗄️\n\n"
                    "🎯 **Unit 1: Relational Databases (9 Hours)**\n"
                    "• Relational model concepts\n"
                    "• SQL queries and operations\n"
                    "• Normalization techniques\n"
                    "• Query processing and optimization\n"
                    "• Transaction processing basics\n"
                    "• Concurrency control and recovery\n\n"
                    "🎯 **Unit 2: Parallel and Distributed Databases (9 Hours)**\n"
                    "• Database system architectures\n"
                    "• Client-server architectures\n"
                    "• Parallel systems and I/O parallelism\n"
                    "• Distributed database concepts\n"
                    "• Distributed transactions and commit protocols\n\n"
                    "💡 **Study Tips:**\n"
                    "• Practice SQL queries extensively\n"
                    "• Understand normalization with examples\n"
                    "• Learn transaction ACID properties\n"
                    "• Study concurrency control mechanisms\n\n"
                    "These first two units are crucial for your foundation! Need help with SQL practice or normalization? 📝"
                ),
                "sources": [{"id": "mca_syllabus", "preview": "Advanced Database Management Systems - First Semester"}],
            }
        
        return {
            "answer": (
                "Here's your complete Database study roadmap from the MCA syllabus: 🎯\n\n"
                "📚 **All 5 Units Coverage:**\n"
                "1. **Relational Databases** - SQL, normalization, transactions\n"
                "2. **Parallel & Distributed** - Architectures, distributed transactions\n"
                "3. **Object & Object-Relational** - ODMG, ODL, OQL\n"
                "4. **Emerging Technologies** - XML, Cloud databases, GIS\n"
                "5. **NoSQL Databases** - MongoDB, Cassandra, Neo4j\n\n"
                "🛠️ **Lab Practice:**\n"
                "• SQL queries with joins and subqueries\n"
                "• PL/SQL functions, procedures, and triggers\n"
                "• NoSQL database implementations\n"
                "• Data warehouse and OLAP operations\n\n"
                "Which database topic would you like to dive deeper into? 🤔"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "Complete Database syllabus - MCA First Semester"}],
        }

    def _get_java_study_guide(self, question_lower: str) -> Dict:
        """Provide Java-specific study guidance"""
        return {
            "answer": (
                "Here's your Java study plan based on the MCA Advanced Java syllabus: ☕\n\n"
                "📚 **Complete Java Roadmap (Second Semester):**\n"
                "1. **Multithreading & Networking** - Thread model, synchronization, sockets\n"
                "2. **Database Programming** - JDBC, connection pooling, transactions\n"
                "3. **Web Programming** - Servlets, JSP, session tracking, JSTL\n"
                "4. **Web Services & EJB** - SOAP, REST, JAX-WS, Enterprise beans\n"
                "5. **Frameworks** - Spring IoC/DI, AOP, MVC, Hibernate ORM\n\n"
                "🛠️ **Lab Practice Focus:**\n"
                "• Multithreading and networking programs\n"
                "• JDBC database connectivity\n"
                "• Servlet and JSP web applications\n"
                "• Spring and Hibernate integration\n\n"
                "💡 **Pro Tips:**\n"
                "• Start with core multithreading concepts\n"
                "• Practice JDBC extensively\n"
                "• Build small web applications\n"
                "• Understand Spring dependency injection\n\n"
                "Which Java topic needs your immediate attention? 🚀"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "Advanced Java Programming - MCA Second Semester"}],
        }

    def _get_ml_study_guide(self, question_lower: str) -> Dict:
        """Provide Machine Learning study guidance"""
        if any(term in question_lower for term in ['first test', 'unit 1', 'unit 2', 'module 1', 'module 2']):
            return {
                "answer": (
                    "Excellent! For your Machine Learning first test, focus on these fundamentals: 🤖\n\n"
                    "🎯 **Unit 1: Introduction to Machine Learning (9 Hours)**\n"
                    "• What is Machine Learning?\n"
                    "• Types: Supervised, Unsupervised, Reinforcement Learning\n"
                    "• The Machine Learning Process\n"
                    "• First Application: The Iris Dataset\n\n"
                    "🎯 **Unit 2: Supervised Learning (9 Hours)**\n"
                    "• **Regression:** Linear, Multiple, Polynomial, SVR\n"
                    "• **Classification:** Logistic Regression, KNN, SVM, Naive Bayes\n"
                    "• Decision Trees and Random Forest\n"
                    "• Model evaluation techniques\n\n"
                    "💡 **Study Strategy:**\n"
                    "• Understand the difference between regression and classification\n"
                    "• Practice with Python and scikit-learn\n"
                    "• Work through the Iris dataset example\n"
                    "• Implement basic algorithms from scratch\n\n"
                    "These first two units are perfect for your first test! Want help with any specific algorithm? 📊"
                ),
                "sources": [{"id": "mca_syllabus", "preview": "Machine Learning using Python - MCA Second Semester"}],
            }
        
        return {
            "answer": (
                "Here's your complete ML study roadmap from the MCA syllabus: 🧠\n\n"
                "📚 **All 5 Units to Master:**\n"
                "1. **Introduction** - ML types, process, Iris dataset\n"
                "2. **Supervised Learning** - Regression, classification algorithms\n"
                "3. **Unsupervised Learning** - Clustering, dimensionality reduction\n"
                "4. **Neural Networks & Deep Learning** - Perceptrons, CNNs, RNNs\n"
                "5. **NLP & Computer Vision** - Text processing, image classification\n\n"
                "🐍 **Python Lab Practice:**\n"
                "• Implement linear and logistic regression\n"
                "• K-means clustering and PCA\n"
                "• Neural networks from scratch\n"
                "• Work with real datasets\n\n"
                "Which ML concept would you like me to explain in detail? 🎯"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "Complete ML syllabus - MCA Second Semester"}],
        }

    def _get_os_study_guide(self, question_lower: str) -> Dict:
        """Provide Operating System study guidance"""
        if any(term in question_lower for term in ['first test', 'unit 1', 'unit 2', 'module 1', 'module 2']):
            return {
                "answer": (
                    "Great choice! For your OS first test, master these core concepts: 💻\n\n"
                    "🎯 **Unit 1: Introduction (9 Hours)**\n"
                    "• What is an Operating System?\n"
                    "• System types: Batch, Time-sharing, Real-time\n"
                    "• OS components and services\n"
                    "• System calls and programs\n"
                    "• Virtual machines\n\n"
                    "🎯 **Unit 2: Process Management (9 Hours)**\n"
                    "• Process concept and lifecycle\n"
                    "• Process scheduling algorithms\n"
                    "• Threads and interprocess communication\n"
                    "• Critical section problem\n"
                    "• Synchronization: Semaphores, monitors\n\n"
                    "💡 **Study Focus:**\n"
                    "• Understand process vs thread concepts\n"
                    "• Practice scheduling algorithm calculations\n"
                    "• Learn synchronization mechanisms\n"
                    "• Study deadlock prevention techniques\n\n"
                    "These first two units are fundamental! Need help with process scheduling or synchronization? ⚙️"
                ),
                "sources": [{"id": "mca_syllabus", "preview": "Operating System - MCA First Semester"}],
            }
        
        return {
            "answer": (
                "Here's your complete Operating System study guide: 🖥️\n\n"
                "📚 **All 5 Units Overview:**\n"
                "1. **Introduction** - OS types, components, system calls\n"
                "2. **Process Management** - Scheduling, threads, synchronization\n"
                "3. **Memory Management** - Paging, segmentation, virtual memory\n"
                "4. **Storage Management** - File systems, allocation methods\n"
                "5. **Protection & Security** - Access control, authentication\n\n"
                "🎯 **Key Topics to Master:**\n"
                "• CPU scheduling algorithms (FCFS, SJF, Round Robin)\n"
                "• Memory management techniques\n"
                "• File system implementation\n"
                "• Deadlock handling strategies\n\n"
                "Which OS concept would you like me to break down further? 🔧"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "Complete OS syllabus - MCA First Semester"}],
        }

    def _get_general_study_guide(self, question_lower: str) -> Dict:
        """Provide general study guidance based on syllabus"""
        return {
            "answer": (
                "I can help you with study preparation for your MCA subjects! 📚✨\n\n"
                "🎓 **MCA Subjects I can guide you with:**\n"
                "• **First Semester:** DSA, Database, Operating Systems, Statistics, AI\n"
                "• **Second Semester:** Machine Learning, Advanced Java, Distributed Computing\n"
                "• **Third Semester:** Full Stack Development, .NET, Network Security\n"
                "• **Fourth Semester:** Cloud Computing, Blockchain, IoT\n\n"
                "💡 **Just ask me things like:**\n"
                "• \"What should I study for DSA first test?\"\n"
                "• \"Help me prepare for Database exam\"\n"
                "• \"Java programming topics for second semester\"\n"
                "• \"Machine Learning unit 1 and 2 topics\"\n\n"
                "Which subject would you like to focus on? I'll give you a detailed study plan! 🎯"
            ),
            "sources": [{"id": "mca_syllabus", "preview": "MCA Complete Syllabus - All Semesters"}],
        }


rag = RAGEngine()