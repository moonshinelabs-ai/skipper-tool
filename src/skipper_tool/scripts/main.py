#!/usr/bin/env python3
import argparse
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI
from PIL import Image
from playwright.sync_api import sync_playwright

from skipper_tool.config import (
    get_api_key,
    init_config_file,
    load_config,
    merge_with_args,
)
from skipper_tool.keystrokes import process_typing
from skipper_tool.profiling import set_debug_folder
from skipper_tool.reduce import (
    determine_ui_click_element,
    reduce_screenshot_with_llm,
)


def create_timestamp():
    """Create a timestamp string for filenames."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def setup_logging(debug_folder):
    """Setup loguru logging configuration with debug folder.

    Args:
        debug_folder: Path to debug folder for file logging (can be None)
    """
    # Remove default handler
    logger.remove()

    # Add console handler for INFO+ (human readable with colors)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # Only add file logging if debug_folder is provided
    if debug_folder:
        # Use single combined log file for all debug output
        log_file = Path(debug_folder) / "debug.log"

        # Ensure debug folder exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            sink=str(log_file),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

        logger.info(f"Debug logging enabled - writing to {log_file}")


def add_screenshot_padding(screenshot_bytes, padding_pixels):
    """Add whitespace padding to the top and bottom of a screenshot.

    Args:
        screenshot_bytes: Screenshot bytes from Playwright
        padding_pixels: Number of pixels to add to top and bottom

    Returns:
        Tuple of (padded_image_as_bytes, padded_pil_image)
    """
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(screenshot_bytes))

    # Get original dimensions
    width, height = img.size

    # Create new image with padding
    new_height = height + (2 * padding_pixels)
    padded_img = Image.new("RGB", (width, new_height), color="white")

    # Paste original image with top padding
    padded_img.paste(img, (0, padding_pixels))

    # Convert back to bytes
    output_bytes = io.BytesIO()
    padded_img.save(output_bytes, format="PNG")
    return output_bytes.getvalue(), padded_img


class SkipperTool:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.windows = {}
        self.llm_client = None

        # Handle optional debug folder
        if config["debug"]["enabled"] and config["debug"]["folder"]:
            # Use the configured debug folder
            self.debug_folder = Path(config["debug"]["folder"])
            self.images_folder = self.debug_folder / config["debug"]["images_subfolder"]

            # Setup logging first (before creating debug folder to ensure logging captures everything)
            setup_logging(self.debug_folder)

            # Create debug folder and images subfolder
            Path(self.debug_folder).mkdir(parents=True, exist_ok=True)
            self.images_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Debug folder created/verified: {self.debug_folder}")
            logger.debug(f"Images folder created/verified: {self.images_folder}")

            # Set debug folder for profiling module
            set_debug_folder(str(self.debug_folder))

            logger.debug("SkipperTool initialized")
            logger.debug(f"Debug folder: {self.debug_folder}")
            logger.debug(f"Screenshot model: {config['models']['screenshot_model']}")
            logger.debug(f"UI element model: {config['models']['ui_element_model']}")
            logger.debug(f"YOLO model path: {config['models']['yolo_model_path']}")
            logger.debug(
                f"SKIPPER_API_KEY: {'set' if os.getenv('SKIPPER_API_KEY') else 'not set'}"
            )
        else:
            self.debug_folder = None
            self.images_folder = None
            # Set up basic console-only logging
            logger.remove()
            logger.add(
                sink=lambda msg: print(msg, end=""),
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
                colorize=True,
            )

        self._get_llm_client()

    def _get_llm_client(self):
        if not self.llm_client:
            api_key = get_api_key(self.config, "gemini_api_key", "GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not found. Set GEMINI_API_KEY environment variable or add gemini_api_key to your .skipperrc config file."
                )
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url=self.config["api"]["gemini_base_url"],
            )
        return self.llm_client

    def _get_window_data_file(self, window_id):
        return Path(
            f"{self.config['temp']['window_data_dir']}/skipper_window_{window_id}.json"
        )

    def _save_window_data(self, window_id, data):
        with open(self._get_window_data_file(window_id), "w") as f:
            json.dump(data, f)

    def _load_window_data(self, window_id):
        file_path = self._get_window_data_file(window_id)
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return json.load(f)

    def open_window(self, window_name):
        # Just return a random window ID without doing anything else
        window_id = str(uuid.uuid4())[:5]
        logger.debug(f"Opening window: {window_name}")
        logger.info(f"Created window ID: {window_id}")
        logger.debug(f"Window ID {window_id} created for window name: {window_name}")
        return window_id

    def view_window(self, p):
        logger.debug("Viewing window")
        browser = p.chromium.connect_over_cdp(self.config["browser"]["cdp_url"])
        page = browser.contexts[self.config["browser"]["context_index"]].pages[
            self.config["browser"]["page_index"]
        ]

        title = page.title()
        url = page.url
        logger.debug(f"Page details - Title: {title}, URL: {url}")

        # Take a screenshot of the rendered page
        screenshot_bytes = page.screenshot()
        # Add padding to the screenshot
        padded_screenshot_bytes, padded_screenshot_img = add_screenshot_padding(
            screenshot_bytes, self.config["screenshot"]["padding_pixels"]
        )

        if self.debug_folder:
            timestamp = create_timestamp()
            screenshot_path = self.images_folder / f"screenshot_{timestamp}.png"
            with open(screenshot_path, "wb") as f:
                f.write(padded_screenshot_bytes)
            logger.debug(f"Screenshot saved to: {screenshot_path}")

        # Use LLM to analyze screenshot and provide description
        screenshot_analysis = reduce_screenshot_with_llm(
            self.llm_client,
            padded_screenshot_img,
            title,
            url,
            debug_folder=self.debug_folder,
            model=self.config["models"]["screenshot_model"],
        )

        # Log title and screenshot analysis
        logger.info(f"Page title: {title}")
        logger.info(f"Page URL: {url}")
        logger.info(f"Screenshot analysis:\n{screenshot_analysis}")

    def navigate_window(self, p, url):
        browser = p.chromium.connect_over_cdp(self.config["browser"]["cdp_url"])
        page = browser.contexts[self.config["browser"]["context_index"]].pages[
            self.config["browser"]["page_index"]
        ]
        page.goto(url)
        return self.view_window(p)

    def execute_command(self, p, prompt, command_type, return_view=False):
        # Get existing browser context or create new one
        browser = p.chromium.connect_over_cdp(self.config["browser"]["cdp_url"])
        page = browser.contexts[self.config["browser"]["context_index"]].pages[
            self.config["browser"]["page_index"]
        ]

        if command_type == "type":
            process_typing(page, prompt)
            return self.view_window(p)
        elif command_type == "scroll":
            if "up" in prompt.lower():
                page.mouse.wheel(0, -self.config["ui_interaction"]["scroll_distance"])
            elif "down" in prompt.lower():
                page.mouse.wheel(0, self.config["ui_interaction"]["scroll_distance"])
            else:
                raise ValueError("Can't determine scroll direction")
            return self.view_window(p)

        screenshot_bytes = page.screenshot()
        # Add padding to the screenshot
        padded_screenshot_bytes, padded_screenshot_img = add_screenshot_padding(
            screenshot_bytes, self.config["screenshot"]["padding_pixels"]
        )

        try:
            from skipper_tool.omniparser.client import OmniParser
        except ImportError:
            logger.error("OmniParser not available. Check installation.")
            raise

        # Initialize OmniParser client (will automatically choose local or remote mode)
        if self.config["api"]["skipper_api_key"]:
            os.environ["SKIPPER_API_KEY"] = self.config["api"]["skipper_api_key"]

        omni_client = OmniParser(
            yolo_model_path=self.config["models"]["yolo_model_path"],
        )
        logger.debug(f"Using {omni_client.mode} OmniParser execution")

        pil_image = padded_screenshot_img
        predictions = omni_client.predict(pil_image)
        annotated_img = predictions["annotated_image"]
        annotated_img_bytes = io.BytesIO()
        annotated_img.save(annotated_img_bytes, format="PNG")

        if self.debug_folder:
            timestamp = create_timestamp()
            annotated_img_path = self.images_folder / f"annotated_image_{timestamp}.png"
            annotated_img.save(annotated_img_path)
            unannotated_img_path = (
                self.images_folder / f"unannotated_image_{timestamp}.png"
            )
            padded_screenshot_img.save(unannotated_img_path)
            logger.debug(f"Annotated image saved to: {annotated_img_path}")
            logger.debug(f"Unannotated image saved to: {unannotated_img_path}")

        # Use LLM to determine which element to click to take the action
        action_element = determine_ui_click_element(
            self.llm_client,
            annotated_img,
            prompt,
            model=self.config["models"]["ui_element_model"],
        )
        action_element_dict = json.loads(
            action_element.replace("```json", "").replace("```", "")
        )
        logger.info(f"Selected element: {action_element_dict['element_id']}")
        logger.info(f"Reasoning: {action_element_dict['reasoning']}")
        x1, y1, x2, y2 = predictions["boxes"][int(action_element_dict["element_id"])]
        logger.debug(f"Element box: {x1}, {y1}, {x2}, {y2}")
        x_mid, y_mid = int((x1 + x2) / 2), int((y1 + y2) / 2)
        y_mid -= self.config["screenshot"]["padding_pixels"]  # Subtract padding
        logger.info(f"Clicking at coordinates: ({x_mid}, {y_mid})")
        scale_factor = self.config["ui_interaction"]["mouse_scale_factor"]
        page.mouse.click(int(x_mid * scale_factor), int(y_mid * scale_factor))
        time.sleep(self.config["ui_interaction"]["click_delay_seconds"])

        return self.view_window(p)


def main():
    parser = argparse.ArgumentParser(description="Skipper - Desktop interaction tool")

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command for creating config file
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--config",
        action="store_true",
        help="Create sample .skipperrc configuration file",
    )
    init_parser.add_argument(
        "--path",
        help="Path for config file (default: ~/.skipperrc)",
        required=False,
    )

    # Open command
    open_parser = subparsers.add_parser("open", help="Open a window")
    open_parser.add_argument("window_name", help="Name of the window to open")

    # View command
    _ = subparsers.add_parser("view", help="View window state")

    # Navigate command
    navigate_parser = subparsers.add_parser("navigate", help="Navigate to a URL")
    navigate_parser.add_argument("--url", required=True, help="URL to navigate to")

    # Command command
    command_parser = subparsers.add_parser("command", help="Execute command in window")
    command_parser.add_argument("--prompt", required=True, help="Command prompt")
    command_parser.add_argument(
        "--command_type",
        required=True,
        help="Type of command to execute (type, scroll, click)",
    )
    command_parser.add_argument(
        "--no-return-view", action="store_true", help="Don't return view after command"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle init command
    if args.command == "init":
        if args.config:
            try:
                # Prompt for API keys (optional)
                print("\nOptional: Enter API keys (press Enter to skip):")
                gemini_key = input("Gemini API key: ").strip() or None
                skipper_key = input("Skipper API key: ").strip() or None

                config_path = Path(args.path) if args.path else None
                created_path = init_config_file(config_path, gemini_key, skipper_key)
                print(f"\nCreated configuration file: {created_path}")
                if gemini_key or skipper_key:
                    print("API keys have been saved to the config file.")
                print("Edit the file to customize other settings.")
                return
            except FileExistsError as e:
                print(f"Error: {e}")
                return
            except OSError as e:
                print(f"Error creating config file: {e}")
                return
        else:
            print("Use --config to create a configuration file")
            return

    # Load configuration
    config = load_config()
    config = merge_with_args(config, args)

    skipper_tool = SkipperTool(config)

    try:
        with sync_playwright() as p:
            if args.command == "open":
                logger.debug(f"Executing open command for window: {args.window_name}")
                skipper_tool.open_window(args.window_name)
            elif args.command == "view":
                logger.debug("Executing view command")
                skipper_tool.view_window(p)
            elif args.command == "navigate":
                logger.debug("Executing navigate command")
                skipper_tool.navigate_window(p, args.url)
            elif args.command == "command":
                logger.debug(
                    f"Executing command: {args.prompt} (type: {args.command_type})"
                )
                skipper_tool.execute_command(
                    p,
                    args.prompt,
                    args.command_type,
                    not args.no_return_view,
                )
                logger.info("Command executed successfully")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
