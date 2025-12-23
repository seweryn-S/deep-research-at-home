from __future__ import annotations

from src.pipe_shared import *  # noqa: F401,F403
from src.pipe_state import PipeStateMixin
from src.pipe_research_state import PipeResearchStateMixin
from src.pipe_text import PipeTextMixin
from src.pipe_compression import PipeCompressionMixin
from src.pipe_semantics import PipeSemanticsMixin
from src.pipe_fetching import PipeFetchingMixin
from src.pipe_citations import PipeCitationsMixin
from src.pipe_searching import PipeSearchingMixin
from src.pipe_llm import PipeLLMMixin
from src.pipe_interactive import PipeInteractiveMixin
from src.pipe_synthesis import PipeSynthesisMixin
from src.pipe_entry import PipeEntryMixin

class Pipe(
    PipeStateMixin,
    PipeResearchStateMixin,
    PipeTextMixin,
    PipeCompressionMixin,
    PipeSemanticsMixin,
    PipeFetchingMixin,
    PipeCitationsMixin,
    PipeSearchingMixin,
    PipeLLMMixin,
    PipeInteractiveMixin,
    PipeSynthesisMixin,
    PipeEntryMixin,
):
    _active_conversations: Set[str] = set()
    _conversation_lock: Optional[asyncio.Lock] = None

    class Valves(BaseModel):
        ENABLED: bool = Field(default=True, description="Enable Deep Research pipe")
        RESEARCH_MODEL: str = Field(default="gemma3:12b")
        SYNTHESIS_MODEL: str = Field(default="gemma3:27b")
        EMBEDDING_MODEL: str = Field(default="granite-embedding:30m")
        QUALITY_FILTER_MODEL: str = Field(default="gemma3:4b")
        QUALITY_FILTER_ENABLED: bool = Field(default=True)
        QUALITY_SIMILARITY_THRESHOLD: float = Field(default=0.60)
        MAX_CYCLES: int = Field(default=15, ge=3, le=50)
        MIN_CYCLES: int = Field(default=10, ge=1, le=10)
        EXPORT_RESEARCH_DATA: bool = Field(default=True)
        SEARCH_RESULTS_PER_QUERY: int = Field(default=3)
        EXTRA_RESULTS_PER_QUERY: int = Field(default=3)
        SUCCESSFUL_RESULTS_PER_QUERY: int = Field(default=1)
        CHUNK_LEVEL: int = Field(default=2)
        COMPRESSION_LEVEL: int = Field(default=4)
        LOCAL_INFLUENCE_RADIUS: int = Field(default=3)
        QUERY_WEIGHT: float = Field(default=0.5)
        FOLLOWUP_WEIGHT: float = Field(default=0.5)
        TEMPERATURE: float = Field(default=0.7)
        SYNTHESIS_TEMPERATURE: float = Field(default=0.6)
        EMBEDDING_API_BASE: str = Field(default="http://localhost:11434")
        SEARCH_URL: str = Field(default="http://192.168.1.1:8888/search?q=")
        MAX_FAILED_RESULTS: int = Field(default=6)
        EXTRACT_CONTENT_ONLY: bool = Field(default=True)
        PDF_MAX_PAGES: int = Field(default=25)
        HANDLE_PDFS: bool = Field(default=True)
        RELEVANCY_SNIPPET_LENGTH: int = Field(default=2000)
        DOMAIN_PRIORITY: str = Field(default="")
        CONTENT_PRIORITY: str = Field(default="")
        DOMAIN_MULTIPLIER: float = Field(default=1.3)
        KEYWORD_MULTIPLIER_PER_MATCH: float = Field(default=1.1)
        MAX_KEYWORD_MULTIPLIER: float = Field(default=2.0)
        INTERACTIVE_RESEARCH: bool = Field(default=True)
        USER_PREFERENCE_THROUGHOUT: bool = Field(default=True)
        SEMANTIC_TRANSFORMATION_STRENGTH: float = Field(default=0.7)
        TRAJECTORY_MOMENTUM: float = Field(default=0.6)
        GAP_EXPLORATION_WEIGHT: float = Field(default=0.4)
        STEPPED_SYNTHESIS_COMPRESSION: bool = Field(default=True)
        MAX_RESULT_TOKENS: int = Field(default=4000)
        COMPRESSION_SETPOINT: int = Field(default=4000)
        REPEATS_BEFORE_EXPANSION: int = Field(default=3)
        REPEAT_WINDOW_FACTOR: float = Field(default=0.95)
        VERIFY_CITATIONS: bool = Field(default=True)
        OUTPUT_LANGUAGE: str = Field(default="auto")
        THREAD_WORKERS: int = Field(default=50)
        PARALLEL_SESSIONS: int = Field(default=2)
        DEBUG_LLM: bool = Field(default=False)
        DEBUG_SEARCH: bool = Field(default=False)
        DEBUG_TIMING: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.state_manager = ResearchStateManager()
        self.search_client = SearchClient(self)
        self.content_processor = ContentProcessor(self)
        self.conversation_id = None
        self.embedding_cache = EmbeddingCache()
        self.transformation_cache = TransformationCache()
        self.is_pdf_content = False
        self.research_date = datetime.now().strftime("%Y-%m-%d")
        self.trajectory_accumulator = None
        self.embedding_dim = None
        self._conversation_ctx = contextvars.ContextVar("conversation_id", default=None)
        self._conversation_executors = {}
        cls = type(self)
        if cls._conversation_lock is None:
            cls._conversation_lock = asyncio.Lock()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}", "name": f"{name}"}]
