"""
luna-core 플러그인 매니저: 외부 플러그인 등록/로딩/상태 관리
"""
import importlib
import importlib.util
import os
import sys
import traceback
from typing import Dict, Any, List, Optional

class CorePluginManager:
    """
    luna-core 플러그인 매니저
    """
    def __init__(self, plugins_path: str, config: Dict[str, Any], import_namespace: Optional[str] = "plugins"):
        self.plugins_path = os.path.abspath(plugins_path)
        if os.path.basename(self.plugins_path) == "plugins":
            self.repo_root = os.path.dirname(self.plugins_path)
            self.plugins_root = self.plugins_path
        else:
            self.repo_root = self.plugins_path
            self.plugins_root = os.path.join(self.repo_root, "plugins")
        self.namespace = import_namespace
        self.config = config
        self.plugins: Dict[str, Any] = {}

        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

    def _plugin_dir(self, name: str) -> Optional[str]:
        d = os.path.join(self.plugins_root, name)
        return d if os.path.isdir(d) else None

    def _plugin_file(self, name: str) -> Optional[str]:
        d = self._plugin_dir(name)
        if not d:
            return None
        f = os.path.join(d, f"{name}_plugin.py")
        return f if os.path.isfile(f) else None

    def discover_plugins(self) -> List[str]:
        names: List[str] = []
        if os.path.isdir(self.plugins_root):
            for n in os.listdir(self.plugins_root):
                d = os.path.join(self.plugins_root, n)
                f = os.path.join(d, f"{n}_plugin.py")
                if os.path.isdir(d) and os.path.isfile(f):
                    names.append(n)
        return sorted(names)

    def _load_by_namespace(self, name: str):
        if not self.namespace:
            raise ImportError("namespace not set")
        module_path = f"{self.namespace}.{name}.{name}_plugin"
        return importlib.import_module(module_path)

    def _load_by_filepath(self, name: str):
        file_path = self._plugin_file(name)
        if not file_path:
            raise FileNotFoundError(f"플러그인 파일을 찾을 수 없습니다: {name}")
        mod_name = f"_luna_plugin_{name}"
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"spec 생성 실패: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore
        return module

    def load_plugin(self, name: str):
        try:
            module = None
            if self.namespace:
                try:
                    module = self._load_by_namespace(name)
                except Exception:
                    module = None
            if module is None:
                module = self._load_by_filepath(name)

            cls = getattr(module, "Plugin", None)
            if cls is None:
                raise AttributeError("Plugin 클래스가 없습니다.")
            inst = cls(name, self.config.get(name, {}))
            self.plugins[name] = inst
            return inst

        except Exception as e:
            print(f"[L.U.N.A. Plugin Manager] 플러그인 로딩 실패: {name}: {e}")
            traceback.print_exc()
            return None

    def enable_plugin(self, name: str):
        p = self.plugins.get(name)
        if p:
            p.enable()

    def disable_plugin(self, name: str):
        p = self.plugins.get(name)
        if p:
            p.disable()

    def get_plugin_status(self, name: str) -> Optional[Dict[str, Any]]:
        p = self.plugins.get(name)
        if p:
            return p.get_status()
        return None

    def update_plugin_config(self, name: str, new_config: Dict[str, Any]):
        p = self.plugins.get(name)
        if p:
            p.update_config(new_config)