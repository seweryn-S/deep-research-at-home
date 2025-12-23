# Deep Research at Home – Pipe (OpenWebUI)

Ten katalog dokumentuje implementację Pipe „Deep Research at Home” w tym repo, tak żeby po czasie szybko zrozumieć:
- jak działa cały algorytm,
- gdzie są punkty rozszerzeń,
- gdzie najbezpieczniej robić poprawki,
- które elementy są specyficzne dla runtime OpenWebUI.

## TL;DR

- Edytuj kod: `src/*`
- Generuj single-file dla OpenWebUI: `python3 build_pipe.py` → `pipe.py`
- Nie edytuj ręcznie: `pipe.py` (autogenerowany)

## Mapa plików (najważniejsze)

- `src/main.py`: lokalny entrypoint developerski (cienki wrapper eksportujący `Pipe`).
- `src/pipe_impl.py`: klasa `Pipe` + `Valves` (konfiguracja) + składanie miksinów.
- `src/pipe_entry.py`: główna integracja z OpenWebUI – metoda `pipe(...)` (wejście/wyjście, cykle badawcze, synteza).
- `src/pipe_state.py`: izolacja stanu per-konwersacja, limity równoległości, cleanup.
- `src/state.py`: domyślny shape stanu (`default_conversation_state`) + `ResearchStateManager`.
- `src/search.py`: wyszukiwanie (`SearchClient`) – najpierw OpenWebUI retrieval, potem fallback HTTP (`SEARCH_URL`).
- `src/pipe_searching.py`: przetwarzanie pojedynczego wyniku (w tym fetch, token-limity, obsługa powtórek URL).
- `src/pipe_fetching.py` + `src/content.py`: pobieranie treści (HTML/PDF) i ekstrakcja tekstu.
- `src/pipe_semantics.py`: embeddingi + cache + semantyczne transformacje (trajectory/gap/PDV).
- `src/pipe_compression.py`: kompresja wyników (local similarity / eigendecomposition) + limity.
- `src/pipe_citations.py`: cytowania + weryfikacja + linkowanie `[n]` → przypisy `[^n]`.
- `src/pipe_synthesis.py`: generowanie outline do syntezy, pisanie sekcji, bibliografia, review i poprawki.
- `build_pipe.py`: bundling modułów `src/*` do `pipe.py` (OpenWebUI wymaga jednego pliku).

## Jak czytać resztę dokumentacji

- `docs/pipe/ALGORITHM.md`: algorytm end-to-end + pseudokod + diagramy.
- `docs/pipe/STATE.md`: „kontrakt” stanu (klucze, przepływ, kto ustawia).
- `docs/pipe/CONFIG.md`: opis `Valves` (co kontrolują i jak wpływają na flow).

## Szybkie komendy

- Bundling: `python3 build_pipe.py`
- Sprawdzenie zgodności bundla: `python3 build_pipe.py --check --diff`
- Testy: `pytest`

