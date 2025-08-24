"""Keystroke processing functionality for handling special key combinations."""

import re

from playwright.sync_api import Page


def process_typing(page: Page, text: str):
    """Process typing with support for bracketed keys and regular text.

    Extracts content inside angle brackets and sends it directly to page.keyboard.press.
    Regular text outside brackets is also sent to page.keyboard.press.

    Example:
        "<ControlOrMeta+A><Delete>Hello world" will result in:
        - page.keyboard.press("ControlOrMeta+A")
        - page.keyboard.press("Delete")
        - page.keyboard.press("Hello world")

    Args:
        page: Playwright page object
        text: Text to type, may contain bracketed keys like <ControlOrMeta+A>, <Delete>
    """
    from loguru import logger

    logger.info(f"Processing typing: {text}")

    # Pattern to match bracketed content and capture both brackets and content
    pattern = r"(<[^>]+>)"
    parts = re.split(pattern, text)

    for part in parts:
        if not part:  # Skip empty parts
            continue

        if part.startswith("<") and part.endswith(">"):
            # Extract content inside brackets
            key_content = part[1:-1]  # Remove < and >
            logger.info(f"Pressing key: {key_content}")
            page.keyboard.press(key_content)
        else:
            # Regular text - send directly to keyboard.press
            logger.info(f"Pressing text: {part}")
            page.keyboard.type(part)
