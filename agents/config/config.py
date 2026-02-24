import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class AgentConfig:
    """
    Configuration for the Research Agent and its dependencies.
    """
    
    # ── LLM Configuration ─────────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # ── Vector Store Configuration ────────────────────────────────────────
    # Default to in-memory for testing, but typically would be a URL
    QDRANT_LOCATION: str = os.getenv("QDRANT_LOCATION", ":memory:") 
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "deep_research")
    
    # ── Embedding Configuration ───────────────────────────────────────────
    # FastEmbed models
    DENSE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL: str = "prithivida/Splade_pp_en_v1"
    
    # ── Planner Configuration ─────────────────────────────────────────────
    # Max steps to allow in a plan (though LLM usually adheres to prompt instruction)
    MAX_PLAN_STEPS: int = 7
    
    # ── Tool Configuration ────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    ARXIV_MAX_RESULTS: int = 5
    
    @property
    def is_local_vector_store(self) -> bool:
        """Check if Qdrant is running in local/memory mode."""
        return self.QDRANT_LOCATION == ":memory:" or self.QDRANT_LOCATION.startswith("path:")


@dataclass
class ReportConfig:
    """Configuration for report depth and breadth (V2 architecture)."""
    name: str
    num_plan_steps: int      # Max initial topics from Planner
    max_depth: int           # Max depth of the BFS research tree
    max_probes: int          # Max gap children generated per node (before pruning)
    max_tool_pairs: int      # Max [tool, query/url] pairs per populate/resolve
    dupe_threshold: float   # Cosine similarity threshold for deduplication
    rag_top_k: int           # Number of vector store results for gap analysis
    writer_mode: str = "single"   # "single" = one-shot get_write; "bottomup" = per-node + exec summary
    writer_rag_top_k: int = 8    # Vector store results per section when writer_mode is bottomup (unused in single)
    writer_section_max_tokens: int = 4096   # Max tokens per section in bottom-up mode (enables longer sections)
    writer_report_max_tokens: int = 16384  # Max tokens for single-shot full report
    max_tavily_calls: int = 0   # Per-job cap on Tavily API calls (0 = no cap). After cap, use DuckDuckGo.

    @classmethod
    def COMPACT(cls):
        return cls(name="Compact", num_plan_steps=2, max_depth=2, max_probes=2, max_tool_pairs=2, dupe_threshold=0.92, rag_top_k=3, writer_mode="single", writer_rag_top_k=5, writer_section_max_tokens=2048, writer_report_max_tokens=8192, max_tavily_calls=10)

    @classmethod
    def STANDARD(cls):
        return cls(name="Standard", num_plan_steps=5, max_depth=4, max_probes=3, max_tool_pairs=3, dupe_threshold=0.85, rag_top_k=5, writer_mode="bottomup", writer_rag_top_k=8, writer_section_max_tokens=4096, writer_report_max_tokens=16384, max_tavily_calls=25)

    @classmethod
    def DEEP(cls):
        return cls(name="Deep", num_plan_steps=10, max_depth=6, max_probes=5, max_tool_pairs=4, dupe_threshold=0.6, rag_top_k=8, writer_mode="bottomup", writer_rag_top_k=10, writer_section_max_tokens=6144, writer_report_max_tokens=32768, max_tavily_calls=50)

