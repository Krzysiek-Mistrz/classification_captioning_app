# Aircraft Damage Detection & Captioning

Prosty projekt demonstrujący:
1. **Klasyfikację binarną uszkodzeń lotniczych** przy użyciu transfer learning na bazie VGG16.  
2. **Generowanie podpisów (captioning)** zdjęć uszkodzonych części za pomocą modelu BLIP.

---

## 📂 Struktura projektu

├── data/  │ ├── train/  │ ├── valid/  │ └── test/  │ ├── main.py  ├── models.py  ├── utils.py  ├── requirements.txt  └── README.md  

---

## 🚀 Szybki start

1. **Klonowanie repozytorium**
   ```bash
   git clone https://github.com/TWOJ_UZYTKOWNIK/aircraft-damage-captioning.git
   cd aircraft-damage-captioning

    Utwórz i aktywuj środowisko (opcjonalnie, np. conda lub venv)

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

Instalacja zależności

pip install -r requirements.txt

Uruchomienie pełnego pipeline’u

    python main.py

        pobiera i wypakowuje dane

        trenuje model klasyfikacji

        ewaluacja na zbiorze testowym

        generuje podpisy wybranych obrazków

🛠️ Dependencies

    tensorflow ≥ 2.x

    numpy

    matplotlib

    Pillow

    transformers (Hugging Face)

    torch (PyTorch)

📈 Wyniki

    Dokładność klasyfikatora: wyświetlana po treningu

    Przykładowe podpisy BLIP: drukowane w konsoli, np.

