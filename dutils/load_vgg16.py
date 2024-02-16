import torchvision
model = torchvision.models.vgg16(pretrained=True)
print(model)
