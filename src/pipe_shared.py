from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import math
import random
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from types import SimpleNamespace

try:
    from open_webui.constants import TASKS
    from open_webui.main import generate_chat_completions
    from open_webui.models.users import User
except Exception:  # pragma: no cover - used only outside OpenWebUI runtime
    TASKS = SimpleNamespace(DEFAULT="default")

    def generate_chat_completions(*args, **kwargs):
        return {}

    class User:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

from src.constants import COMPRESSION_RATIO_MAP, PROMPTS
from src.content import ContentProcessor
from src.search import SearchClient
from src.state import ResearchStateManager, TrajectoryAccumulator
from src.utils.embeddings import (
    EmbeddingCache,
    TransformationCache,
    apply_semantic_transformation,
    get_embedding,
)
from src.utils.logger import setup_logger
from src.utils.text import (
    chunk_text,
    clean_text_formatting,
    truncate_for_log,
)

PIPE_NAME = "Deep Research at Home"
name = PIPE_NAME
logger = logging.getLogger(PIPE_NAME)
