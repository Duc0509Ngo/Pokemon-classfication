import  torch
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from PIL import Image
from categories import categories
import io
def test(image_path, deploy=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")

    model = models.densenet121()
    model.eval()
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(categories))
    model.to(device)

    checkpoint = torch.load("../trained_models/pokemon/best.pt")
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])


    test_transform = Compose([
        Resize(size=(224, 224)),
        CenterCrop(size=200),
        ToTensor(),
        Normalize(mean=[0.59565047, 0.59168748, 0.56110077], std=[0.23401408, 0.22395668, 0.23227972])
    ])
    if deploy:
        image = Image.open(io.BytesIO(image_path)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        result = model(image)

        _, prediction = torch.max(result, dim=1)
        prediction = categories[prediction.item()]

    return prediction

if __name__ == "__main__":

    print(test("../10.png"))






