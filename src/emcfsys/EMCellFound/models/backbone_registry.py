# backbone_registry.py

# BACKBONE_REGISTRY = {
#                         "resnet34", "resnet50", "resnet101", 
                        
#                          "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                         
#                          "efficientnet_b0", "efficientnet_b2", 
#                          "efficientnet_b4" , "efficientnet_b6", 
#                          "efficientnet_b7",
                         
#                          "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l",
                         
#                          "rexnetr_200.sw_in12k", "rexnetr_300.sw_in12k", 
                         
#                          "vit_small_patch16_dinov3.lvd1689m", "vit_base_patch16_dinov3.lvd1689m",
#                          "vit_large_patch16_dinov3.lvd1689m", "vit_huge_patch16_dinov3.lvd1689m"
# }


"""
backbone_registry.py

功能：
- 管理 backbone 注册表（内存 + JSON 持久化）
- 支持两种类型条目：type == "timm" 或 type == "local"
- 提供 register_backbone(...) / list_backbones() / load_backbone(...) 等函数
- 对 timm model 做自动验证；对 local .py 做动态导入（要求提供 create_backbone callable）

持久化位置：默认 ~/.emcfsys/backbones.json （可通过 env 修改）
"""

import os
import json
import importlib.util
from typing import List, Dict, Optional, Tuple
import traceback

DEFAULT_JSON = os.environ.get("EMCFSYS_BACKBONE_JSON", os.path.expanduser("~/.emcfsys/backbones.json"))

# registry entries: list of dicts: {"name": str, "type": "timm"|"local", "source": str}
# - for timm: source is "timm"
# - for local: source is absolute path to .py file
_default_builtin = [
    {"name": "resnet34", "type": "timm", "source": "timm"},
    {"name": "resnet50", "type": "timm", "source": "timm"},
    {"name": "resnet101", "type": "timm", "source": "timm"},
    
    
    {"name": "convnext_tiny", "type": "timm", "source": "timm"},
    {"name": "convnext_small", "type": "timm", "source": "timm"},
    {"name": "convnext_base", "type": "timm", "source": "timm"},
    {"name": "convnext_large", "type": "timm", "source": "timm"},
    
    {"name": "efficientnet_b0", "type": "timm", "source": "timm"},
    {"name": "efficientnet_b2", "type": "timm", "source": "timm"},
    {"name": "efficientnet_b4", "type": "timm", "source": "timm"},
    {"name": "efficientnet_b6", "type": "timm", "source": "timm"},
    {"name": "efficientnet_b7", "type": "timm", "source": "timm"},
    
    {"name": "efficientnetv2_s", "type": "timm", "source": "timm"},
    {"name": "efficientnetv2_m", "type": "timm", "source": "timm"},
    {"name": "efficientnetv2_l", "type": "timm", "source": "timm"},
    
    {"name": "rexnetr_200.sw_in12k", "type": "timm", "source": "timm"},
    {"name": "rexnetr_300.sw_in12k", "type": "timm", "source": "timm"},
    
    {"name": "vit_small_patch16_dinov3.lvd1689m", "type": "timm", "source": "timm"},
    {"name": "vit_base_patch16_dinov3.lvd1689m", "type": "timm", "source": "timm"},
    {"name": "vit_large_patch16_dinov3.lvd1689m", "type": "timm", "source": "timm"},
    {"name": "vit_huge_patch16_dinov3.lvd1689m", "type": "timm", "source": "timm"},
    
    {"name": "swin_tiny_patch4_window7_224", "type": "timm", "source": "timm"},
    {"name": "swin_small_patch4_window7_224", "type": "timm", "source": "timm"},
    {"name": "swin_base_patch4_window7_224", "type": "timm", "source": "timm"},
    {"name": "swin_large_patch4_window7_224", "type": "timm", "source": "timm"},
    {"name": "swin_large_patch4_window12_384", "type": "timm", "source": "timm"},
    {"name": "swin_large_patch4_window12_384_in22k", "type": "timm", "source": "timm"},
    
    {"name": "hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k", "type": "timm", "source": "timm"},
    {"name": "hiera_base_abswin_256.sbb2_e200_in12k_ft_in1k", "type": "timm", "source": "timm"},
    {"name": "hiera_large_abswin_256.sbb2_e200_in12k_ft_in1k", "type": "timm", "source": "timm"},
    {"name": "hiera_huge_abswin_256.sbb2_e200_in12k_ft_in1k", "type": "timm", "source": "timm"},
    {"name": "hiera_giant_abswin_256.sbb2_e200_in12k_ft_in1k", "type": "timm", "source": "timm"},


]

_registry: List[Dict] = []


def _ensure_json_dir(json_path: str):
    d = os.path.dirname(json_path)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_from_json(json_path: str = DEFAULT_JSON):
    global _registry
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    _registry = data
                    return
        except Exception:
            print("Warning: failed to load backbone registry json:", json_path)
            print(traceback.format_exc())
    # fallback to builtin
    _registry = list(_default_builtin)


def _save_to_json(json_path: str = DEFAULT_JSON):
    _ensure_json_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_registry, f, indent=2, ensure_ascii=False)


# init at import
_load_from_json(DEFAULT_JSON)


def list_backbones() -> List[Dict]:
    """返回 registry 的拷贝（list of dict）"""
    return list(_registry)


def list_backbone_names() -> List[str]:
    return [e["name"] for e in _registry]


def find_entry_by_name(name: str) -> Optional[Dict]:
    for e in _registry:
        if e["name"] == name:
            return e
    return None


# Validation helpers
def _validate_timm_backbone(name: str) -> Tuple[bool, str]:
    """尝试调用 timm.create_model(name, pretrained=False, features_only=True)，返回(success, msg)"""
    try:
        import timm
        # 注意：我们不加载 pretrained 权重（避免下载），但要构建 model 以验证接口
        _ = timm.create_model(name, pretrained=False, features_only=True)
        return True, "timm model created successfully"
    except Exception as e:
        return False, f"timm.create_model failed: {e}"


def _validate_local_py(path: str) -> Tuple[bool, str]:
    """验证 local .py 文件是否存在且内部含 create_backbone callable"""
    if not os.path.exists(path):
        return False, "file not found"
    try:
        spec = importlib.util.spec_from_file_location("local_backbone_validation", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        # accept either create_backbone or build_backbone
        if hasattr(mod, "create_backbone") and callable(getattr(mod, "create_backbone")):
            return True, "local module has create_backbone"
        if hasattr(mod, "build_backbone") and callable(getattr(mod, "build_backbone")):
            return True, "local module has build_backbone"
        return False, "local module must define callable create_backbone(...) or build_backbone(...)"
    except Exception as e:
        return False, f"Failed to import local module: {e}"


# Public API
def register_backbone(name: str, local_py: Optional[str] = None, validate: bool = True, persist: bool = True) -> Tuple[bool, str]:
    """
    注册一个 backbone。
    - 若 local_py is None: treat as timm name
    - 若 local_py provided: treat as local module and store absolute path
    返回 (success, message)
    """
    name = name.strip()
    if not name:
        return False, "Empty name"

    existing = find_entry_by_name(name)
    if existing:
        return False, f"Backbone '{name}' already exists in registry"

    if local_py:
        local_py = os.path.abspath(local_py)
        ok, msg = _validate_local_py(local_py) if validate else (True, "skipped validation")
        if not ok:
            return False, f"Local backbone validation failed: {msg}"
        entry = {"name": name, "type": "local", "source": local_py}
    else:
        ok, msg = _validate_timm_backbone(name) if validate else (True, "skipped validation")
        if not ok:
            return False, f"Timm validation failed: {msg}"
        entry = {"name": name, "type": "timm", "source": "timm"}

    _registry.append(entry)
    if persist:
        try:
            _save_to_json(DEFAULT_JSON)
        except Exception:
            print("Warning: failed to persist backbone registry")
    return True, f"Registered backbone '{name}' (type={entry['type']})"


def unregister_backbone(name: str, persist: bool = True) -> Tuple[bool, str]:
    entry = find_entry_by_name(name)
    if not entry:
        return False, f"Backbone '{name}' not found"
    _registry.remove(entry)
    if persist:
        _save_to_json(DEFAULT_JSON)
    return True, f"Unregistered backbone '{name}'"


def clear_registry(persist: bool = True) -> None:
    global _registry
    _registry = list(_default_builtin)
    if persist:
        _save_to_json(DEFAULT_JSON)


def load_backbone_factory(name: str):
    """
    根据 registry 条目返回一个 callable factory。
    - 若 type == "timm"，返回 (lambda **kw: timm.create_model(name, **kw))
    - 若 type == "local"，动态导入模块并返回其 create_backbone/build_backbone 函数
    """
    entry = find_entry_by_name(name)
    if not entry:
        raise ValueError(f"Backbone '{name}' is not registered")

    if entry["type"] == "timm":
        import timm
        def factory(pretrained=False, features_only=True, out_indices=None, **kwargs):
            # out_indices may be passed; timm will validate.
            return timm.create_model(name, pretrained=pretrained, features_only=features_only, out_indices=out_indices, **kwargs)
        return factory

    # local
    path = entry["source"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local backbone file not found: {path}")

    spec = importlib.util.spec_from_file_location(f"local_backbone_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    factory = None
    # Prefer create_backbone, then build_backbone
    if hasattr(mod, "create_backbone") and callable(mod.create_backbone):
        factory = getattr(mod, "create_backbone")
    elif hasattr(mod, "build_backbone") and callable(mod.build_backbone):
        factory = getattr(mod, "build_backbone")
    else:
        raise RuntimeError("Local module must expose callable create_backbone(...) or build_backbone(...)")

    return factory


# convenience wrapper to attempt instantiate a backbone (for validation or runtime)
def try_create_backbone(name: str, pretrained=False, features_only=True, out_indices=None, **kwargs):
    factory = load_backbone_factory(name)
    return factory(pretrained=pretrained, features_only=features_only, out_indices=out_indices, **kwargs)
