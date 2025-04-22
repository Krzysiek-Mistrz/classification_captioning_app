from pathlib import Path
from src.data_utils import set_seed, download_and_extract, build_data_generators
from src.models import build_classifier
from src.trainer import compile_and_train, evaluate, plot_history
from src.captioning import generate_caption

def main():
    #data prep
    set_seed(42)
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1"
    base = Path("data/aircraft_damage_dataset_v1")
    download_and_extract(url, base)
    train_gen, valid_gen, test_gen = build_data_generators(base, img_size=(224,224), batch_size=32)

    #classifier
    model = build_classifier(input_shape=(224,224,3))
    history = compile_and_train(model, train_gen, valid_gen, lr=1e-4, epochs=5)
    plot_history(history)
    loss, acc = evaluate(model, test_gen)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    #captioning
    sample_img = base/"test"/"dent"/"144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"
    caption = generate_caption(sample_img)
    print("Caption:", caption)

if __name__ == "__main__":
    main()
