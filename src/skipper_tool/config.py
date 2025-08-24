#!/usr/bin/env python3
"""
Configuration management for skipper_tool.

Supports loading configuration from:
1. Command line arguments (highest priority)
2. .skipperrc files (TOML format)
3. Built-in defaults (lowest priority)

Configuration file search order:
1. ./.skipperrc (current directory)
2. ~/.skipperrc (home directory)
3. ~/.config/skipper/config.toml (XDG config)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Handle TOML import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. Install with: pip install tomli"
        )


# Default configuration
DEFAULT_CONFIG = {
    "models": {
        "screenshot_model": "gemini-2.5-flash",
        "ui_element_model": "gemini-2.5-pro",
        "yolo_model_path": None,
    },
    "browser": {
        "cdp_url": "http://localhost:9222",
        "context_index": 0,
        "page_index": 0,
    },
    "screenshot": {
        "padding_pixels": 50,
        "format": "PNG",
        "quality": 100,
    },
    "ui_interaction": {
        "click_delay_seconds": 1.0,
        "scroll_distance": 600,
        "mouse_scale_factor": 0.5,
    },
    "api": {
        "gemini_api_key": None,
        "skipper_api_key": None,
        "gemini_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "skipper_api_url": "https://nate-3--omni-parser-api-fastapi-app.modal.run",
        "request_timeout_seconds": 30,
        "health_check_timeout_seconds": 10,
    },
    "debug": {
        "enabled": False,
        "folder": None,
        "log_file": "debug.log",
        "images_subfolder": "images",
        "log_rotation": "10 MB",
        "log_retention": "7 days",
        "log_compression": "gz",
    },
    "annotation": {
        "text_padding": 5,
        "max_font_size": 30,
        "box_overlay_ratio": 0.02,
    },
    "temp": {
        "window_data_dir": "/tmp",
        "cache_dir": "/tmp/omniparser_cache",
    },
    "llm_analysis": {
        "image_reduction_factor": 0.5,
        "prompt_truncation_length": 500,
        "response_truncation_length": 300,
    },
    "omniparser": {
        "model_url": "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt?download=true",
        "device": "auto",
    },
}


def find_config_file() -> Optional[Path]:
    """
    Search for configuration file in standard locations.

    Returns:
        Path to config file if found, None otherwise
    """
    search_paths = [
        Path.cwd() / ".skipperrc",
        Path.home() / ".skipperrc",
        Path.home() / ".config" / "skipper" / "config.toml",
    ]

    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    return None


def load_toml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load and parse a TOML configuration file.

    Args:
        file_path: Path to TOML file

    Returns:
        Parsed configuration dictionary

    Raises:
        ValueError: If file cannot be parsed
    """
    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in {file_path}: {e}")
    except OSError as e:
        raise ValueError(f"Cannot read {file_path}: {e}")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file and environment variables.

    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    # Try to load from config file
    config_file = find_config_file()
    if config_file:
        try:
            file_config = load_toml_file(config_file)
            config = deep_merge(config, file_config)
        except ValueError as e:
            # Print warning but continue with defaults
            print(f"Warning: {e}", file=sys.stderr)

    # Environment variable overrides
    env_overrides = _get_env_overrides()
    if env_overrides:
        config = deep_merge(config, env_overrides)

    return config


def get_api_key(config: Dict[str, Any], key_name: str, env_var: str) -> Optional[str]:
    """
    Get API key with priority: environment variable > config file.

    Args:
        config: Configuration dictionary
        key_name: Key name in config (e.g., 'gemini_api_key')
        env_var: Environment variable name (e.g., 'GEMINI_API_KEY')

    Returns:
        API key if found, None otherwise
    """
    # Environment variable takes precedence
    env_key = os.getenv(env_var)
    if env_key:
        return env_key

    # Fall back to config file
    return config.get("api", {}).get(key_name)


def _get_env_overrides() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.

    Returns:
        Dictionary with environment variable overrides
    """
    overrides = {}

    # API keys - these take precedence over config file
    if gemini_key := os.getenv("GEMINI_API_KEY"):
        overrides.setdefault("api", {})["gemini_api_key"] = gemini_key

    if skipper_key := os.getenv("SKIPPER_API_KEY"):
        overrides.setdefault("api", {})["skipper_api_key"] = skipper_key

    # API URLs
    if skipper_url := os.getenv("SKIPPER_API_URL"):
        overrides.setdefault("api", {})["skipper_api_url"] = skipper_url

    return overrides


def merge_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Merge configuration with command line arguments.
    Command line arguments take highest precedence.

    Args:
        config: Configuration dictionary
        args: Parsed command line arguments

    Returns:
        Merged configuration
    """
    # Create a copy to avoid modifying the original
    result = config.copy()

    # Map CLI args to config structure
    if hasattr(args, "debug_folder") and args.debug_folder:
        result["debug"]["enabled"] = True
        result["debug"]["folder"] = args.debug_folder

    if hasattr(args, "screenshot_model") and args.screenshot_model:
        result["models"]["screenshot_model"] = args.screenshot_model

    if hasattr(args, "ui_element_model") and args.ui_element_model:
        result["models"]["ui_element_model"] = args.ui_element_model

    if hasattr(args, "yolo_model_path") and args.yolo_model_path:
        result["models"]["yolo_model_path"] = args.yolo_model_path

    if hasattr(args, "return_view") and args.return_view is not None:
        result["ui_interaction"]["return_view"] = args.return_view

    return result


def create_sample_config(
    gemini_api_key: Optional[str] = None, skipper_api_key: Optional[str] = None
) -> str:
    """
    Create a sample configuration file content with all options commented out.

    Args:
        gemini_api_key: Optional Gemini API key to include
        skipper_api_key: Optional Skipper API key to include

    Returns:
        Sample configuration as string
    """
    # Build API section dynamically
    api_section = "[api]\n# API configuration and keys\n"

    if gemini_api_key:
        api_section += f'gemini_api_key = "{gemini_api_key}"\n'
    else:
        api_section += '# gemini_api_key = "your-gemini-api-key-here"\n'

    if skipper_api_key:
        api_section += f'skipper_api_key = "{skipper_api_key}"\n'
    else:
        api_section += '# skipper_api_key = "your-skipper-api-key-here"\n'

    api_section += """# gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
# skipper_api_url = "https://nate-3--omni-parser-api-fastapi-app.modal.run"
# request_timeout_seconds = 30
# health_check_timeout_seconds = 10"""

    return f"""# Skipper Configuration File
# All values shown are defaults - uncomment and modify as needed

[models]
# LLM models for different tasks
# screenshot_model = "gemini-2.5-flash"
# ui_element_model = "gemini-2.5-pro" 
# yolo_model_path = ""  # Path to local YOLO model (optional)

[browser]
# Browser connection settings
# cdp_url = "http://localhost:9222"
# context_index = 0
# page_index = 0

[screenshot]
# Screenshot processing settings
# padding_pixels = 50
# format = "PNG"
# quality = 100

[ui_interaction]
# UI element interaction settings
# click_delay_seconds = 1.0
# scroll_distance = 600
# mouse_scale_factor = 0.5  # Coordinates are halved before clicking

{api_section}

[debug]
# Debug and logging settings
# enabled = false
# folder = ""  # Empty = no debug output
# log_file = "debug.log"
# images_subfolder = "images"
# log_rotation = "10 MB"
# log_retention = "7 days"
# log_compression = "gz"

[annotation]
# Visual annotation settings
# text_padding = 5
# max_font_size = 30
# box_overlay_ratio = 0.02

[temp]
# Temporary file settings
# window_data_dir = "/tmp"
# cache_dir = "/tmp/omniparser_cache"

[llm_analysis]
# LLM response processing
# image_reduction_factor = 0.5  # Shrink images by 50% for LLM
# prompt_truncation_length = 500
# response_truncation_length = 300

[omniparser]
# OmniParser model settings
# model_url = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt?download=true"
# device = "auto"  # "auto", "cuda", "cpu"
"""


def init_config_file(
    file_path: Optional[Path] = None,
    gemini_api_key: Optional[str] = None,
    skipper_api_key: Optional[str] = None,
) -> Path:
    """
    Create a sample configuration file.

    Args:
        file_path: Path where to create the config file.
                  If None, creates ~/.skipperrc
        gemini_api_key: Optional Gemini API key to include
        skipper_api_key: Optional Skipper API key to include

    Returns:
        Path to created config file

    Raises:
        FileExistsError: If config file already exists
        OSError: If file cannot be created
    """
    if file_path is None:
        file_path = Path.home() / ".skipperrc"

    if file_path.exists():
        raise FileExistsError(f"Config file already exists: {file_path}")

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        f.write(create_sample_config(gemini_api_key, skipper_api_key))

    return file_path
