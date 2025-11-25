Distributed under the Apache-2.0 License. See LICENSE for details.

- Konfiguracja: `OLLAMA_URL` zastąpione przez `EMBEDDING_API_BASE`, dodany parametr `OUTPUT_LANGUAGE`, domyślny `THREAD_WORKERS` podniesiony do 50, śledzenie `embedding_dim` i większy domyślny wymiar w `TrajectoryAccumulator` (1024) z fallbackami bazującymi na wykrytym wymiarze.
- Funkcje pomocnicze: `count_tokens` uproszczone do heurystyki słów, `pipes()` zwraca identyfikator bez sufiksu `-pipe`.
- Wektory: `get_embedding` korzysta z endpointu OpenAI `/v1/embeddings`, cache’uje wymiary i usuwa ładowanie prebudowanych słowników; generowanie wymiarów badań (PCA) zapisuje teksty oraz embeddingi tematów do późniejszej interpretacji.
- Semantyka wymiarów: nazewnictwo PCA przeniesione z dopasowania słownika do wywołań LLM na podstawie reprezentatywnych tematów; dodane bezpieczne fallbacki przy brakach lub niezgodnościach wymiarów.
- UI i język: wymuszanie języka odpowiedzi dla komunikatów użytkownika (`OUTPUT_LANGUAGE`), lokalizacja outline/analizy na PL lub EN, nowe zwijane `<details>` dla wyników i analiz, komunikaty wyników wyszukiwania pakowane w collapsible blokach.
