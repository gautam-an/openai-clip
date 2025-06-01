import os
import clip
import torch
from torchvision.datasets import CIFAR100

device = "cpu"
model, preprocess = clip.load('ViT-B/32', device)

dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
index = 1200
image, class_number = dataset[index]
class_name = dataset.classes[class_number]

print(f"The image at index {index} is a: {class_name}")

image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(10)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")