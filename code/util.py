import torch

def load_model(path):
    if torch.cuda.is_available():
        model = torch.load(path)
        print('loaded model onto gpu')
    else:
        model = torch.load(path, map_location=lambda storage, loc: storage)
        print('loaded model onto cpu')
    return model