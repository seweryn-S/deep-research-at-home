# ADR 0001: Bundling `src/*` do pojedynczego `pipe.py`

- Status: accepted
- Data: 2025-12-23

## Kontekst

OpenWebUI oczekuje, że Pipe będzie dostarczony jako pojedynczy plik Pythona (np. `pipe.py`).
Jednocześnie utrzymywanie całej logiki w jednym pliku:
- utrudnia czytanie i refaktor,
- komplikuje testy,
- utrudnia izolowanie odpowiedzialności (fetch/search/synthesis/citations),
- zwiększa ryzyko regresji przy „małych” zmianach.

## Decyzja

Kod źródłowy utrzymujemy modularnie w `src/`, a artefakt kompatybilny z OpenWebUI generujemy do `pipe.py` przez bundler:

- generator: `build_pipe.py`
- wejścia bundla: `FILES_TO_BUNDLE` w `build_pipe.py`
- zasada: `pipe.py` jest autogenerowany i **nie jest edytowany ręcznie**

## Konsekwencje

### Plusy

- Czytelna struktura: miksiny dzielą odpowiedzialności (`pipe_entry`, `pipe_fetching`, `pipe_synthesis`, itd.).
- Łatwiejsze testowanie i analiza: importy działają lokalnie bez OpenWebUI.
- Mniejsze diffy i prostszy review zmian.

### Minusy / ryzyka

- Możliwy „drift” między `src/*` a `pipe.py`, jeśli ktoś zapomni wygenerować bundla.
- Bundler usuwa importy `src.*` (`build_pipe.py`), więc w `pipe.py` nie można polegać na runtime importach z `src`.

### Mitigacje

- Używaj `python3 build_pipe.py --check --diff` (np. w CI).
- Traktuj `src/*` jako single source of truth; `pipe.py` jako artefakt.

## Alternatywy rozważane

1) Trzymać całość w `pipe.py` (odrzucone: utrzymanie i refaktor stają się zbyt kosztowne).
2) Dynamiczne importy z `src/` w runtime OpenWebUI (odrzucone: środowisko/pluginy OpenWebUI i wymaganie single-file).

