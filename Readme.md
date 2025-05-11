# Aircraft Damage Detection & Captioning

Projekt demonstruje:
1. **Klasyfikację binarną uszkodzeń lotniczych** przy użyciu transfer learning (VGG16).  
2. **Generowanie podpisów (captioning)** zdjęć uszkodzonych części za pomocą modelu BLIP.

---

## Struktura projektu

```
classification_captioning_app/
├── Readme.md
├── LICENSE
└── src/
    ├── __init__.py
    ├── captioning.py
    ├── data_utils.py
    ├── main.py
    ├── models.py
    └── trainer.py
```

---

## Szybki start

1. **Klonowanie repozytorium**  
   ```bash
   git clone https://github.com/TWOJ_UZYTKOWNIK/aircraft-damage-captioning.git
   cd aircraft-damage-captioning
   ```  

2. **Utwórz i aktywuj środowisko:**  
    ```
    python -m venv .venv
    source .venv/bin/activate      # Linux/macOS
    .venv\Scripts\activate         # Windows
    ```  

3. **Instalacja zależności**  
`pip install -r requirements.txt`  

4. **Uruchomienie pełnego pipeline’u (schemat)**  
    python src/data_utils.py  
        - pobiera i wypakowuje dane  
    python src/main.py  
        - trenuje model klasyfikacji  
        - ewaluacja na zbiorze testowym  
    python src/captioning.py  
        - generuje przykładowy podpis do zdjęcia  
    python src/main.py  
        - zbiera wszystkie te funkcjonalności w całość (jak to main ;) )  

5. **Dependencies**
    - tensorflow ≥ 2.5  
    - numpy  
    - matplotlib  
    - Pillow  
    - transformers (Hugging Face)  
    - torch  

6. **Wyniki**
    - Dokładność klasyfikatora: wyświetlana po treningu  
    - Przykładowe podpisy BLIP: drukowane w konsoli  