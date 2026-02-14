"""
Centralized configuration management
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from YAML or JSON file
        
        Args:
            config_path: path to config file (YAML or JSON)
                        if None, uses data/config/config.yaml
        """
        self.config_path = config_path or "data/config/config.yaml"
        self.data: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  Config file not found: {self.config_path}")
            self.data = self._get_default_config()
            return
        
        ext = Path(self.config_path).suffix.lower()
        
        try:
            if ext == '.yaml' or ext == '.yml':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.data = yaml.safe_load(f) or {}
            elif ext == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                print(f"⚠️  Unsupported config format: {ext}")
                self.data = self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            self.data = self._get_default_config()
    
    def save(self):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.data, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get(self, key: str, default=None) -> Any:
        """Get config value"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set config value"""
        self.data[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all config data"""
        return self.data
    
    def set_all(self, data: Dict[str, Any]):
        """Set all config data and save"""
        self.data = data
        self.save()
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any):
        self.data[key] = value
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "project_name": "HG Camera Counter",
            "version": "0.1.0",
            "supabase": {
                "url": os.getenv("SUPABASE_URL", ""),
                "key": os.getenv("SUPABASE_ANON_KEY", ""),
                "branch_code": os.getenv("BRANCH_CODE", "DEMO"),
            },
            "cameras": {
                "Camera_01": {
                    "rtsp_url": os.getenv("CAM1_URL", ""),
                    "enabled": True,
                    "note": ""
                },
                "Camera_02": {
                    "rtsp_url": os.getenv("CAM2_URL", ""),
                    "enabled": True,
                    "note": ""
                }
            },
            "yolo": {
                "model": "yolov8m.pt",
                "conf": 0.35,
                "iou": 0.5,
                "imgsz": 640,
            },
            "runtime": {
                "target_fps": 10,
                "max_workers": 4,
            },
            "paths": {
                "models": "models",
                "zones": "data/zones",
                "staff_gallery": "data/staff_gallery",
                "reports": "reports",
                "snapshots": "snapshots",
                "logs": "logs",
            }
        }
