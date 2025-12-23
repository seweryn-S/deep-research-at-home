import sys
from pathlib import Path
from types import SimpleNamespace


# Ensure project root is on path so `import pipe` works even when pytest is run elsewhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Minimal stubs for open_webui dependencies used in tests
if "open_webui" not in sys.modules:
    sys.modules["open_webui"] = SimpleNamespace()


# constants.TASKS.DEFAULT
constants = SimpleNamespace(TASKS=SimpleNamespace(DEFAULT="default"))
sys.modules["open_webui.constants"] = constants


# main.generate_chat_completions
def _generate_chat_completions(*args, **kwargs):
    return {}


sys.modules["open_webui.main"] = SimpleNamespace(
    generate_chat_completions=_generate_chat_completions
)


# models.users.User
class _User:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


sys.modules["open_webui.models"] = SimpleNamespace(users=SimpleNamespace(User=_User))
sys.modules["open_webui.models.users"] = SimpleNamespace(User=_User)


# routers.retrieval.process_web_search / SearchForm
class _SearchForm:
    def __init__(self, query: str | None = None, queries=None, **kwargs):
        self.query = query
        if queries is None and query is not None:
            queries = [query]
        self.queries = queries or []


async def _process_web_search(request, search_form, user=None):
    # Return minimal empty search result structure
    return {"docs": [], "filenames": []}


sys.modules["open_webui.routers"] = SimpleNamespace(
    retrieval=SimpleNamespace(
        process_web_search=_process_web_search, SearchForm=_SearchForm
    )
)
sys.modules["open_webui.routers.retrieval"] = SimpleNamespace(
    process_web_search=_process_web_search, SearchForm=_SearchForm
)
