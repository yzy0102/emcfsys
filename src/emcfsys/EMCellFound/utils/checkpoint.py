import torch
import os

def save_model(model, path):
    torch.save(model, path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_pretrained(model, ckpt_path, device):
    print("Loading pretrained:", ckpt_path)
    obj = torch.load(ckpt_path, map_location=device)

    # ① 如果是完整模型：包含 model.state_dict()
    if hasattr(obj, "state_dict"):
        print("Loaded a full model object.")
        sd = obj.state_dict()

    # ② 如果是纯 state_dict（OrderedDict）
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # 如果是 {"model": state_dict, ... }
        if "model" in obj and isinstance(obj["model"], dict):
            print("Loaded checkpoint with key 'model'")
            sd = obj["model"]

        # 如果是 {"state_dict": state_dict, ... }
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            print("Loaded checkpoint with key 'state_dict'")
            sd = obj["state_dict"]

        # 直接是 state_dict
        else:
            print("Loaded raw state_dict")
            sd = obj

    else:
        raise ValueError("Unrecognized checkpoint format")

    # 统一 load
    model.load_state_dict(sd, strict=True)
    print("Pretrained weights loaded successfully.")
    return model