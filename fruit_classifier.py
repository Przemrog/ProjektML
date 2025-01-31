import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch_directml
import os
import time

# Ustawienie urządzenia do obliczeń (DirectML)
device = torch_directml.device()
print(f"Using device: {device}")

# Ścieżki do zbioru danych
data_dir = "fruits-360"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Test")

# Transformacje dla obrazów (zmiana rozmiaru, normalizacja)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Wczytanie zbiorów danych
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Liczba klas w zbiorze danych
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# Ustawienie wielkości partii (batch size)
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Wczytanie wstępnie wytrenowanego modelu ResNet18
model = models.resnet18(pretrained=True)

# Zamrożenie wszystkich warstw modelu, aby nie były trenowane
for param in model.parameters():
    param.requires_grad = False

# Zamiana ostatniej warstwy na nową z odpowiednią liczbą klas
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc.requires_grad = True  # Umożliwienie trenowania nowej warstwy

# Przeniesienie modelu na wybrane urządzenie (DirectML)
model = model.to(device)

# Ustawienie funkcji kosztu oraz optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Liczba epok treningu
num_epochs = 5
best_accuracy = 0.0

# Pętla treningowa
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()  # Ustawienie modelu w tryb treningowy
    running_loss = 0.0
    start_time = time.time()
    
    total_batches = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()  # Wyzerowanie gradientów
        
        outputs = model(inputs)  # Przepuszczenie danych przez sieć
        loss = criterion(outputs, labels)  # Obliczenie straty
        loss.backward()  # Obliczenie gradientów
        optimizer.step()  # Aktualizacja wag
        
        running_loss += loss.item() * inputs.size(0)
        
        # Wypisanie informacji co 10 batchy
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch {batch_idx + 1}/{total_batches} | "
                f"Batch Loss: {loss.item():.4f} | "
                f"Elapsed Time: {time.time() - start_time:.2f}s"
            )
    
    # Obliczenie straty
    epoch_loss = running_loss / len(train_dataset)
    
    # Ewaluacja modelu na zbiorze testowym
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Wyłączenie obliczania gradientów dla oszczędności pamięci
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Pobranie klasy z największym prawdopodobieństwem
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Obliczenie dokładności na zbiorze testowym
    accuracy = 100 * correct / total
    epoch_time = time.time() - start_time
    
    # Wypisanie podsumowania
    print(
        f"Epoch {epoch+1} Summary | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )
    
    # Zapisanie najlepszego modelu
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

print(f"\nTraining complete. Best accuracy: {best_accuracy:.2f}%")