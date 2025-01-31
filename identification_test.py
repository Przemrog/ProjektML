import torch
from torchvision import transforms, models
from PIL import Image
import torch_directml
import os
import matplotlib.pyplot as plt
import numpy as np

# Ustawienie urządzenia do obliczeń (DirectML)
device = torch_directml.device()
print(f"Using device: {device}")

# Funkcja do wczytania modelu
# model_path - ścieżka do zapisanego modelu
# num_classes - liczba klas w zbiorze danych
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False) # Tworzenie modelu ResNet18
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes) # Dostosowanie warstwy wyjściowej
    model.load_state_dict(torch.load(model_path, map_location=device)) # Wczytanie wag modelu
    model = model.to(device) # Przeniesienie modelu na urządzenie
    model.eval() # Ustawienie modelu w tryb ewaluacji
    return model

# Funkcja do wczytania i przetworzenia obrazu
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB") # Wczytanie obrazu i konwersja do RGB
    image = transform(image).unsqueeze(0) # Przekształcenie obrazu i dodanie wymiaru batcha
    return image.to(device)

# Funkcja do wykonania predykcji
def predict_fruit(model, image_tensor, class_names):
    with torch.no_grad():  # Wyłączenie obliczania gradientów dla oszczędności pamięci
        output = model(image_tensor) # Przekazanie obrazu przez model
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100 # Obliczenie prawdopodobieństw
        top_probs, top_idxs = torch.topk(probabilities, 5) # Wybranie 5 największych prawdopodobieństw
        top_classes = [class_names[idx] for idx in top_idxs.cpu().numpy()] # Przetłumaczenie indeksów na nazwy klas
    return top_classes, top_probs.cpu().numpy()

# Funkcja do wizualizacji wyników
def visualize_results(image_path, top_classes, top_probs):
    image = Image.open(image_path).convert("RGB")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Image")
    
    ax[1].barh(top_classes[::-1], top_probs[::-1], color='skyblue') # Wykres słupkowy
    ax[1].set_xlabel("Confidence")
    ax[1].set_title("Top 5 Accuracy")
    plt.tight_layout()
    plt.show()

# Funkcja do wczytania nazw klas
def load_class_names(file_path="class_names.txt"):
    with open(file_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names

def main():
    model_path = "best_model.pth" # Ścieżka do zapisanego modelu
    image_path = "apple3.jpg"   # Ścieżka do obrazu do klasyfikacji
    class_names = load_class_names() # Nazwy klas
    num_classes = len(class_names) # Liczba klas

    print("Loading model")
    model = load_model(model_path, num_classes)
    
    print("Preprocessing image")
    image_tensor = preprocess_image(image_path)
    
    print("Predicting")
    top_classes, top_probs = predict_fruit(model, image_tensor, class_names)
    
    print("Visualization")
    visualize_results(image_path, top_classes, top_probs)

if __name__ == "__main__":
    main()
