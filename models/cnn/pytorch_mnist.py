import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# ---------------- Transform (Augmentation nur Training) ----------------
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------------- Dataset (PyTorch sauber) ----------------
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   
            nn.ReLU(),
            nn.MaxPool2d(2),                   
            nn.Conv2d(32, 64, kernel_size=3),  
            nn.ReLU(),
            nn.MaxPool2d(2),                   
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = ConvolutionNeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.3,
    patience=1
)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():.4f}  [{batch * len(X)}/{size}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    f = [0.0 for _ in range(10)]
    ff = np.array([[0.0 for _ in range(10)] for _ in range(10)])

    test_loss = 0
    correct = 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            
            if pred.argmax(1) == y:
                correct += (pred.argmax(1) == y).sum().item()
            else:
                f[y] += 1
                ff[y][pred.argmax(1)] += 1


    test_loss /= len(dataloader)
    correct /= size

    scheduler.step(test_loss)
    s = sum(f)
    for i in range(len(f)):
        f[i] /= s
    for i in range(len(ff)):
        s = sum(ff[i])
        for j in range(len(ff[0])):
            ff[i][j] /= s

    
    print(f"Test Accuracy: {100 * correct:.2f}%, Avg loss: {test_loss:.4f}")
    print(f"Falsche Vorhersagen in Prozent:")
    print(f"0 wurde {100*f[0]:.2f}% falsch vorhergesagt,")
    print(f"1 wurde {100*f[1]:.2f}% falsch vorhergesagt,")
    print(f"2 wurde {100*f[2]:.2f}% falsch vorhergesagt,")
    print(f"3 wurde {100*f[3]:.2f}% falsch vorhergesagt,")
    print(f"4 wurde {100*f[4]:.2f}% falsch vorhergesagt,")
    print(f"5 wurde {100*f[5]:.2f}% falsch vorhergesagt,")
    print(f"6 wurde {100*f[6]:.2f}% falsch vorhergesagt,")
    print(f"7 wurde {100*f[7]:.2f}% falsch vorhergesagt,")
    print(f"8 wurde {100*f[8]:.2f}% falsch vorhergesagt,")
    print(f"9 wurde {100*f[9]:.2f}% falsch vorhergesagt,")
    

    print(f"Welche Zahlen wurden prozentual wie falsch vorhergesagt:")
    print(f"0 wurde am häufigsten als {ff[0].argmax()} mit {100*ff[0][ff[0].argmax()]}% falsch vorhergesagt")
    print(f"1 wurde am häufigsten als {ff[1].argmax()} mit {100*ff[1][ff[1].argmax()]}% falsch vorhergesagt")
    print(f"2 wurde am häufigsten als {ff[2].argmax()} mit {100*ff[2][ff[2].argmax()]}% falsch vorhergesagt")
    print(f"3 wurde am häufigsten als {ff[3].argmax()} mit {100*ff[3][ff[3].argmax()]}% falsch vorhergesagt")
    print(f"4 wurde am häufigsten als {ff[4].argmax()} mit {100*ff[4][ff[4].argmax()]}% falsch vorhergesagt")
    print(f"5 wurde am häufigsten als {ff[5].argmax()} mit {100*ff[5][ff[5].argmax()]}% falsch vorhergesagt")
    print(f"6 wurde am häufigsten als {ff[6].argmax()} mit {100*ff[6][ff[6].argmax()]}% falsch vorhergesagt")
    print(f"7 wurde am häufigsten als {ff[7].argmax()} mit {100*ff[7][ff[7].argmax()]}% falsch vorhergesagt")
    print(f"8 wurde am häufigsten als {ff[8].argmax()} mit {100*ff[8][ff[8].argmax()]}% falsch vorhergesagt")
    print(f"9 wurde am häufigsten als {ff[9].argmax()} mit {100*ff[9][ff[9].argmax()]}% falsch vorhergesagt")




epochs = 20

for t in range(epochs):
    print(f"\nEpoch {t+1}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

print("Done!")

torch.save(model.state_dict(), "CNN.pth")
