# Aircraft Damage Detection & Captioning

Prosty projekt demonstrujący:
1. **Klasyfikację binarną uszkodzeń lotniczych** przy użyciu transfer learning (VGG16).  
2. **Generowanie podpisów (captioning)** zdjęć uszkodzonych części za pomocą modelu BLIP.

---

## 📂 Struktura projektu

aircraft-damage-captioning/ ├── data/ # dane pobierane automatycznie │ ├── train/ # zestaw treningowy │ ├── valid/ # zestaw walidacyjny │ └── test/ # zestaw testowy ├── src/ # kod źródłowy │ ├── data_utils.py # pobieranie, ekstrakcja, generatory │ ├── models.py # architektury sieci │ ├── trainer.py # trening, ewaluacja, wykresy │ ├── captioning.py # generowanie captionów BLIP │ └── main.py # uruchomienie całego pipeline'u ├── requirements.txt # zależności └── README.md # ten plik

---

## 🚀 Szybki start

1. **Klonowanie repozytorium**  
   ```bash
   git clone https://github.com/TWOJ_UZYTKOWNIK/aircraft-damage-captioning.git
   cd aircraft-damage-captioning

    Utwórz i aktywuj środowisko

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

Instalacja zależności

pip install -r requirements.txt

Uruchomienie pełnego pipeline’u

    python src/main.py

        pobiera i wypakowuje dane

        trenuje model klasyfikacji

        ewaluacja na zbiorze testowym

        generuje przykładowy podpis do zdjęcia

🛠️ Dependencies

    tensorflow ≥ 2.5

    numpy

    matplotlib

    Pillow

    transformers (Hugging Face)

    torch

📈 Wyniki

    Dokładność klasyfikatora: wyświetlana po treningu

    Przykładowe podpisy BLIP: drukowane w konsoli