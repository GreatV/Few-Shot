import torch
from few_shot.models import FewShotClassifier


if __name__ == "__main__":
    model =FewShotClassifier(3, 3, 1600)
    x = torch.randn(1, 3, 80, 80)
    try:
        torch.export.export(model, (x, ))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e
