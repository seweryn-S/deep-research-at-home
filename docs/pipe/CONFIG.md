# Konfiguracja Pipe (`Valves`)

Pipe jest konfigurowany przez `Valves` w `src/pipe_impl.py`. W OpenWebUI te pola są zwykle mapowane na UI (suwaki/pola) lub konfigurację pipe.

## Zasady ogólne

- `Valves` wpływają na **koszt**, **czas**, **jakość** i **stabilność**.
- Najbezpieczniej zmieniać parametry dotyczące liczby wyników / cykli / temperatury. Najbardziej ryzykowne są zmiany modeli i endpointów.

## 1) Tryb i współbieżność

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `ENABLED` | `True` | wyłącza/uruchamia pipe (gating w `src/pipe_entry.py`) |
| `PARALLEL_SESSIONS` | `2` | limit równoległych rozmów badawczych (sloty w `src/pipe_state.py`) |
| `THREAD_WORKERS` | `50` | wątki w executorze per-konwersacja (m.in. dla zadań CPU/IO w tle) |

## 2) Modele i temperatury

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `RESEARCH_MODEL` | `gemma3:12b` | model do generowania query, analizy cykli, części logiki pomocniczej |
| `SYNTHESIS_MODEL` | `gemma3:27b` | model do pisania raportu (sekcje/intro/abstract/conclusion) i review |
| `QUALITY_FILTER_MODEL` | `gemma3:4b` | model do filtrowania jakości (dla niskiej podobieństwowo wyników) |
| `EMBEDDING_MODEL` | `granite-embedding:30m` | model embeddingów dla semantyki/rankingu/kompresji |
| `TEMPERATURE` | `0.7` | temperatura dla zadań research (query generation, analiza, itd.) |
| `SYNTHESIS_TEMPERATURE` | `0.6` | temperatura dla syntezy (zwykle niższa dla spójności) |
| `EMBEDDING_API_BASE` | `http://localhost:11434` | baza endpointu `/v1/embeddings` (OpenAI-compatible) |

## 3) Wyszukiwanie i dobór wyników

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `SEARCH_URL` | `http://192.168.1.1:8888/search?q=` | fallback search (gdy OpenWebUI retrieval nie zwróci wyników) |
| `SEARCH_RESULTS_PER_QUERY` | `3` | bazowa liczba wyników na query |
| `EXTRA_RESULTS_PER_QUERY` | `3` | dodatkowe wyniki, rosną gdy powtarzają się URL-e |
| `SUCCESSFUL_RESULTS_PER_QUERY` | `1` | ile „dobrych” wyników faktycznie zbieramy na query |
| `MAX_FAILED_RESULTS` | `6` | ile porażek (np. brak treści) dopuszczamy zanim przerwiemy iterację po wynikach |
| `REPEATS_BEFORE_EXPANSION` | `3` | po ilu powtórkach URL zwiększać pulę wyników (w `src/search.py`) |
| `QUALITY_FILTER_ENABLED` | `True` | włącza filtr jakości dla wyników o niskiej podobieństwowości |
| `QUALITY_SIMILARITY_THRESHOLD` | `0.60` | próg similarity poniżej którego uruchamia się filtr jakości |

## 4) Fetch, ekstrakcja treści i PDF

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `HANDLE_PDFS` | `True` | włącza obsługę PDF (ekstrakcja tekstu) |
| `PDF_MAX_PAGES` | `25` | limit stron przy ekstrakcji PDF (w implementacji ekstraktora) |
| `EXTRACT_CONTENT_ONLY` | `True` | czy zwracać „czysty tekst” (vs raw HTML) w fetch pipeline |
| `RELEVANCY_SNIPPET_LENGTH` | `2000` | gdy snippet jest krótszy, pipe próbuje fetchować URL (`src/pipe_searching.py`) |
| `MAX_RESULT_TOKENS` | `4000` | limit tokenów na wynik (używane też do ograniczania cache w fetch) |

## 5) Kompresja i chunkowanie

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `CHUNK_LEVEL` | `2` | agresywność chunkowania tekstu (patrz `src/pipe_text.py`) |
| `COMPRESSION_LEVEL` | `4` | poziom kompresji (mapowany przez `COMPRESSION_RATIO_MAP`) |
| `LOCAL_INFLUENCE_RADIUS` | `3` | promień lokalnej podobieństwowości w kompresji (sąsiednie chunki) |
| `STEPPED_SYNTHESIS_COMPRESSION` | `True` | czy wykonać kompresję wyników przed syntezą |
| `COMPRESSION_SETPOINT` | `4000` | target tokenów podczas stepped compression |
| `REPEAT_WINDOW_FACTOR` | `0.95` | kontrola „okna” przy obróbce powtórek URL (sliding window) |

## 6) Priorytetyzacja, semantyka i preferencje użytkownika

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `INTERACTIVE_RESEARCH` | `True` | czy zatrzymywać się po outline i pytać o feedback |
| `USER_PREFERENCE_THROUGHOUT` | `True` | czy stosować preferencje (PDV) konsekwentnie w cyklach |
| `SEMANTIC_TRANSFORMATION_STRENGTH` | `0.7` | siła transformacji embeddingów (PDV/trajectory/gap) |
| `TRAJECTORY_MOMENTUM` | `0.6` | waga trajectory w priorytetyzacji tematów |
| `GAP_EXPLORATION_WEIGHT` | `0.4` | waga „gap vector” (wyżej na początku, potem fade-out) |
| `QUERY_WEIGHT` | `0.5` | mieszanie query vs (inne sygnały) w scoringu kompresji |
| `FOLLOWUP_WEIGHT` | `0.5` | mieszanie query vs summary w kompresji, gdy `summary_embedding` istnieje |

Parametry „priorytetu treści” (używane przy scoringu wyników):

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `DOMAIN_PRIORITY` | `""` | lista domen/fragmentów domen do promowania |
| `CONTENT_PRIORITY` | `""` | słowa-klucze promujące treści |
| `DOMAIN_MULTIPLIER` | `1.3` | mnożnik dla dopasowanych domen |
| `KEYWORD_MULTIPLIER_PER_MATCH` | `1.1` | mnożnik per trafienie słowa-klucza |
| `MAX_KEYWORD_MULTIPLIER` | `2.0` | górny limit mnożnika słów-kluczy |

## 7) Cytowania, język i eksport

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `VERIFY_CITATIONS` | `True` | uruchamia weryfikację cytowań (kosztowna, ale poprawia jakość) |
| `OUTPUT_LANGUAGE` | `auto` | wymusza język odpowiedzi user-facing (PL/EN/...) |
| `EXPORT_RESEARCH_DATA` | `True` | zapisuje export (wyniki/query/URL/content) do pliku po zakończeniu |

## 8) Debug

| Zmienna | Domyślnie | Efekt |
|---|---:|---|
| `DEBUG_LLM` | `False` | loguje request/response LLM (uwaga na rozmiar logów) |
| `DEBUG_SEARCH` | `False` | dodatkowe logi wyszukiwania i filtrowania |
| `DEBUG_TIMING` | `False` | logi czasów krytycznych kroków (`timed(...)`) |

