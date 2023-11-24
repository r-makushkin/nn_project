from torchvision import transforms as T


preprocessing_func = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

def preprocess(img):
    return preprocessing_func(img)