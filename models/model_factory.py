from models.light_tst import LightweightTSClassifier
from models.crossformer import SimpleCrossformer  # 需自行实现Crossformer简化版
from models.patchtst import PatchTSTClassifier    # 需自行实现PatchTST分类版

def get_model(model_name, config):
    """根据名称返回模型实例"""
    if model_name == "LightweightTS":
        return LightweightTSClassifier(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            d_model=config["d_model"]
        )
    elif model_name == "Crossformer":
        return SimpleCrossformer(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            d_model=config["d_model"]
        )
    elif model_name == "PatchTST":
        return PatchTSTClassifier(
            input_dim=config["input_dim"],
            patch_len=config["patch_len"],
            num_classes=config["num_classes"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")