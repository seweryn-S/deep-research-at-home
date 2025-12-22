Distributed under the Apache-2.0 License. See LICENSE for details.

- Wersja 0.3.0: Refaktoryzacja „Deep Research at Home”: uproszczony, modularny Pipe z pełną obsługą stanu, cache’y, fetchowania/kompresji/weryfikacji cytowań oraz testami stubbingującymi OpenWebUI — łatwiejsza analiza i dalszy rozwój.
- Wersja 0.2.3: przebudowane cytowania i bibliografia w sekcji „Comprehensive Answer” – spójne mapowanie numerów `[n]` na źródła, zamiana cytatów w treści na przypisy Markdown (`[^n]` → `[^n]: ...`), usunięcie surowych znaczników HTML i nadmiarowych list źródeł generowanych przez model.
- Wersja 0.2.2: poprawiona normalizacja odpowiedzi modeli rozumujących – strumieniowe pola `reasoning_content` są buforowane osobno i nie są już doklejane do `content`, co zapobiega wyciekowi wewnętrznych instrukcji do sekcji „Comprehensive Answer”; w razie braku `content` używany jest `reasoning_content` jako bezpieczny fallback.
- Wersja 0.2.1: limit równoległych sesji konfigurowalny (`PARALLEL_SESSIONS`, domyślnie 2) z licznikami w komunikatach; per-sesyjny executor wątków i sprzątanie stanu (w tym `url_results_cache`) po zakończeniu badań, żeby nie trzymać pamięci między rozmowami.
- Wersja 0.2.0: wycięty słownik MIT/prebuildy embeddingów; etykiety preferencji (PDV) tworzy lokalny LLM na podstawie wybranych tematów; dodany `DEBUG_LLM` do logowania zapytań/odpowiedzi oraz parsowanie strumieni `data:` do pełnego tekstu; wymuszanie języka na wyjściach user-facing (`OUTPUT_LANGUAGE`).
- Konfiguracja: `OLLAMA_URL` zastąpione przez `EMBEDDING_API_BASE`, dodany parametr `OUTPUT_LANGUAGE`, domyślny `THREAD_WORKERS` podniesiony do 50, śledzenie `embedding_dim` i większy domyślny wymiar w `TrajectoryAccumulator` (1024) z fallbackami bazującymi na wykrytym wymiarze.
- Funkcje pomocnicze: `count_tokens` uproszczone do heurystyki słów, `pipes()` zwraca identyfikator bez sufiksu `-pipe`.
- Wektory: `get_embedding` korzysta z endpointu OpenAI `/v1/embeddings`, cache’uje wymiary i usuwa ładowanie prebudowanych słowników; generowanie wymiarów badań (PCA) zapisuje teksty oraz embeddingi tematów do późniejszej interpretacji.
- Semantyka wymiarów: nazewnictwo PCA przeniesione z dopasowania słownika do wywołań LLM na podstawie reprezentatywnych tematów; dodane bezpieczne fallbacki przy brakach lub niezgodnościach wymiarów.
- UI i język: wymuszanie języka odpowiedzi dla komunikatów użytkownika (`OUTPUT_LANGUAGE`), lokalizacja outline/analizy na PL lub EN, nowe zwijane `<details>` dla wyników i analiz, komunikaty wyników wyszukiwania pakowane w collapsible blokach.

## Testing

Create a virtual environment and install test dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
pytest
```

Note: In restricted environments you may see a `joblib` warning about falling back to
serial mode (permission denied when creating semaphores). Tests still pass; if you want
to silence the warning, run `JOBLIB_MULTIPROCESSING=0 pytest`.
