# Stan Pipe (per-konwersacja)

Pipe utrzymuje stan „rozmowy badawczej” w pamięci procesu, izolowany per-konwersacja. Ten dokument opisuje:
- gdzie stan żyje,
- jak wygląda (klucze i typy),
- jak przebiegają przejścia między fazami,
- na co uważać przy modyfikacjach.

## Gdzie jest stan i jak jest indeksowany

- Implementacja: `src/state.py` (`ResearchStateManager` + `default_conversation_state()`).
- Dostęp: `PipeStateMixin.get_state()` i `PipeStateMixin.update_state()` w `src/pipe_state.py`.
- Klucz konwersacji: `conversation_id` jest budowany z `user_id` oraz `chat_id/chatId/...` z payloadu OpenWebUI (fallback do `messages[0].id`), zob. `PipeStateMixin._resolve_conversation_id()` w `src/pipe_state.py`.

W praktyce:
- stan jest **in-memory** (brak persystencji po restarcie procesu),
- stan może zawierać typy pythonowe (np. `set()`), bo to nie jest serializowane do JSON.

## Życie stanu (lifecycle)

Najważniejsze punkty sterujące:

- **Start wywołania**: `PipeEntryMixin.pipe()` w `src/pipe_entry.py` ustawia kontekst rozmowy i inicjalizuje mapy śledzenia.
- **Nowa rozmowa**: jeśli `len(messages) <= 2` i nie czekamy na feedback, pipe wykonuje `reset_state(conversation_id)`.
- **Interaktywny outline**: ustawiane jest `waiting_for_outline_feedback=True` + `outline_feedback_data`, a `pipe()` zwraca `""` (pauza do kolejnej wiadomości).
- **Cleanup**: na końcu (po syntezie) `PipeStateMixin._cleanup_conversation_resources(conversation_id)` czyści zasoby per-konwersacja (m.in. usuwa stan tej rozmowy).

## Klucze stanu (podstawowy „kontrakt”)

Poniższa tabela bazuje na `default_conversation_state()` (`src/state.py`). W praktyce część pól jest uzupełniana „leniwe” w trakcie działania.

| Klucz | Typ | Po co | Gdzie ustawiane / używane |
|---|---|---|---|
| `research_completed` | `bool` | oznacza, że raport został wygenerowany | ustawiane w `src/pipe_entry.py` po syntezie; używane w `emit_status` (żeby ograniczać statusy po zakończeniu) |
| `prev_comprehensive_summary` | `str` | treść ostatniego raportu (podstawa do follow-up) | ustawiane w `src/pipe_entry.py` po syntezie; używane w `is_follow_up_query` (`src/pipe_interactive.py`) |
| `waiting_for_outline_feedback` | `bool` | flaga pauzy po outline | ustawiane w `process_user_outline_feedback` (`src/pipe_interactive.py`); sprawdzane w `src/pipe_entry.py` |
| `outline_feedback_data` | `dict \| None` | dane do kontynuacji po feedback | ustawiane w `process_user_outline_feedback`; odczytywane w `process_outline_feedback_continuation` oraz `pipe()` |
| `research_state` | `dict` | snapshot „wejścia” do cykli (outline, tematy, embedding) | ustawiane w `initialize_research_state` (`src/pipe_research_state.py`) oraz po feedback (`src/pipe_entry.py`) |
| `follow_up_mode` | `bool` | czy bieżące wywołanie to follow-up | ustawiane w `src/pipe_entry.py` |
| `user_preferences` | `dict` | preferencje kierunku badań (PDV) | ustawiane w `process_outline_feedback_continuation`; używane w priorytetyzacji i generowaniu query |
| `research_dimensions` | `dict \| None` | PCA-wymiary + coverage | inicjalizowane/aktualizowane w `src/pipe_semantics.py` |
| `research_trajectory` | `list[float] \| None` | wektor „momentum” badań | wyliczany w cyklach w `src/pipe_entry.py`, używany w priorytetyzacji i transformacjach |
| `semantic_transformations` | `dict \| None` | macierz transformacji embeddingów | tworzona w `src/pipe_semantics.py` / `src/pipe_interactive.py` |
| `pdv_alignment_history` | `list[float]` | historia dopasowania do preferencji | używana do adaptacji wag w priorytetyzacji |
| `gap_coverage_history` | `list[float]` | historia „gap coverage” | używana do adaptacji `gap_weight` |
| `active_outline` | `list[str]` | tematy do dalszego research | wyliczane/aktualizowane w cyklach w `src/pipe_entry.py` |
| `cycle_summaries` | `list[str]` | streszczenia per-cykl | dopisywane po analizie LLM w `src/pipe_entry.py` |
| `completed_topics` | `set[str]` | tematy uznane za zakończone | aktualizowane w `src/pipe_entry.py` |
| `irrelevant_topics` | `set[str]` | tematy do odrzucenia | aktualizowane w `src/pipe_entry.py` |
| `results_history` | `list[dict]` | pełna historia wyników | dopisywana w `process_query`/cyklach |
| `search_history` | `list` | historia query/zdarzeń | wykorzystywana do trajectory (zależnie od implementacji) |
| `url_selected_count` | `dict[url->int]` | ile razy URL faktycznie użyto | aktualizowane w `src/pipe_searching.py` |
| `url_considered_count` | `dict[url->int]` | ile razy URL rozważono | aktualizowane w `src/pipe_fetching.py` |
| `url_token_counts` | `dict[url->int]` | tokeny na URL | aktualizowane w `src/pipe_searching.py` |
| `master_source_table` | `dict[url->source]` | metadane źródeł (title, type, preview) | uzupełniane w `src/pipe_searching.py` i `src/pipe_fetching.py` |
| `global_citation_map` | `dict[url->int]` | globalne ID cytowań `[n]` | budowane w `src/pipe_synthesis.py` i dopinane w `src/pipe_entry.py` |
| `section_synthesized_content` | `dict[section->text]` | wygenerowane sekcje raportu | budowane w `src/pipe_synthesis.py` i finalnie synchronizowane w `src/pipe_entry.py` |
| `section_citations` | `dict[section->list]` | cytowania per-sekcja | budowane w `src/pipe_synthesis.py` oraz uzupełniane dodatkowymi formatami w `src/pipe_entry.py` |
| `verified_citations` / `flagged_citations` | `list` | wyniki weryfikacji | aktualizowane w `src/pipe_citations.py` / `src/pipe_synthesis.py` |
| `citation_fixes` | `list` | ślad modyfikacji cytowań (np. strike-through) | aktualizowane w `src/pipe_citations.py` |
| `memory_stats` | `dict` | liczniki tokenów | inicjalizowane w `src/state.py`, aktualizowane w `src/pipe_research_state.py` i `src/pipe_entry.py` |

## Klucze leniwie tworzone (poza default state)

Te pola nie są w `default_conversation_state()`, ale pojawiają się w toku pracy:

- `url_results_cache`: cache treści po URL (HTML/PDF) – `src/pipe_fetching.py`.
- `domain_session_map`: per-domain cookies + rate limiting – `src/pipe_fetching.py`.
- `topic_alignment_cache`: cache obliczeń podobieństw/alignments w priorytetyzacji – `src/pipe_interactive.py`.
- `subtopic_relevance_cache`: cache doboru źródeł pod subtopic – `src/pipe_synthesis.py`.
- `latest_dimension_coverage`: snapshot coverage do wyświetlenia w UI – `src/pipe_entry.py` i `src/pipe_semantics.py`.
- `verification_results`: agregat wyników weryfikacji do notki w raporcie – `src/pipe_entry.py` i `src/pipe_citations.py`.

## Gotchas (ważne przy poprawkach)

- **Follow-up vs cleanup**: kod ma tryb follow-up (opiera się o `prev_comprehensive_summary`), ale na końcu syntezy wykonywany jest cleanup, który usuwa stan per-konwersacja. Jeśli chcesz „prawdziwe” follow-up, trzeba świadomie zdecydować, co ma zostać zachowane (np. przenieść summary poza reset albo zmienić cleanup).
- **`set()` w stanie**: `completed_topics` i `irrelevant_topics` są `set`. To wygodne w runtime, ale jeśli kiedyś będziesz to serializować do JSON, trzeba to znormalizować do list.
- **Kontekst rozmowy**: `conversation_id` jest zależny od `chat_id`. Jeśli upstream zmieni format payloadu lub nie dostarczysz `chat_id`, fallback może powodować „nową” rozmowę i reset stanu.

