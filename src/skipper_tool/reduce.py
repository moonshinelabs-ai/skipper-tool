import base64
import io
import re
import time
from textwrap import dedent

from bs4 import BeautifulSoup
from loguru import logger

from skipper_tool.profiling import log_model_io, profile


def remove_base64_from_html(html_content):
    """Remove long base64 strings from HTML content to reduce token usage."""
    # Pattern to match data URLs with base64 content
    # Matches data:mediatype;base64,<base64-string>
    base64_pattern = r"data:[^;]*;base64,[A-Za-z0-9+/]{50,}={0,2}"

    # Replace base64 data URLs with a placeholder
    cleaned_html = re.sub(base64_pattern, "data:removed-base64-content", html_content)

    return cleaned_html


def remove_non_content_tags(html_content):
    """Remove HTML tags that don't contain useful content or structure."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Tags to remove completely (including their content)
        tags_to_remove = [
            "meta",  # Meta tags
            "style",  # CSS styles
            "script",  # JavaScript
            "link",  # External resources (CSS, etc.)
            "noscript",  # No-script content
            "head",  # Head section (if it somehow got through)
            "title",  # Page title (we get this separately)
            "code",  # Code blocks
            "svg",  # SVGs
        ]

        # Remove specified tags
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        soup_str = str(soup)
        # Remove HTML comments
        soup_str = re.sub(r"<!--.*?-->", "", soup_str, flags=re.DOTALL)
        return soup_str

    except Exception as e:
        logger.warning(f"Failed to parse HTML with BeautifulSoup: {e}")
        # Fallback to regex-based removal if BeautifulSoup fails
        fallback_html = html_content

        # Remove meta tags
        fallback_html = re.sub(r"<meta[^>]*>", "", fallback_html, flags=re.IGNORECASE)

        # Remove style tags and their content
        fallback_html = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            fallback_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove script tags and their content
        fallback_html = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            fallback_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove link tags
        fallback_html = re.sub(r"<link[^>]*>", "", fallback_html, flags=re.IGNORECASE)

        # Remove HTML comments
        fallback_html = re.sub(r"<!--.*?-->", "", fallback_html, flags=re.DOTALL)

        return fallback_html


def reduce_screenshot_with_llm(
    client, image, title, url, debug_folder=None, model="gemini-2.5-flash"
):
    """Use LLM to analyze a screenshot and provide description of page content.

    Args:
        client: The LLM client
        image: PIL Image object of the screenshot
        title: Page title
        url: Page URL
        debug_folder: Optional debug folder path
    """
    try:
        screenshot_bytes_io = io.BytesIO()

        # Shrink the image by 50%
        image = image.resize((image.width // 2, image.height // 2))
        image.save(screenshot_bytes_io, format="PNG")

        # Convert PIL image to bytes
        screenshot_bytes = screenshot_bytes_io.getvalue()

        # Convert screenshot bytes to base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = dedent(f"""
        Here is a screenshot of a website. I'd like you to recreate, as best you can, the semantic/structural outline of the page. Do your best to minimize the amount of html you return, but don't lose any information.
        
        - It doesn't need to do anything, and should just be pure html.
        - Do not include any JS or CSS.
        - It should, however, include all the relevant text.
        - If the page has multiple modal elements, such as a foreground element that renders the background occluded, you can just create the html for the foreground element and describe the background element in english (but don't need to include all text).
        - The page title is {title} and the page url is {url}.
        """)

        # Use the OpenAI client with vision capabilities and profiling
        start_time = time.time()
        with profile(
            "Screenshot analysis",
            model,
            image_size_bytes=len(screenshot_bytes),
            image_base64_chars=len(screenshot_base64),
            url=url,
            title=title,
        ):
            response = client.chat.completions.create(
                model=model,
                reasoning_effort="none",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing web page screenshots and describing their content, layout, and functionality in detail.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_base64}"
                                },
                            },
                        ],
                    },
                ],
            )

        output_content = response.choices[0].message.content
        runtime = time.time() - start_time

        # Log I/O for debugging
        if debug_folder:
            log_model_io(
                "Screenshot analysis",
                input_data=prompt,
                output_data=output_content,
                url=url,
                title=title,
                image_size_bytes=len(screenshot_bytes),
            )

        # Log inference results
        from skipper_tool.inference_logger import log_vision_call

        log_vision_call(
            model_name=model,
            operation="Screenshot analysis",
            input_prompt=prompt,
            image_data={
                "size_bytes": len(screenshot_bytes),
                "base64_chars": len(screenshot_base64),
                "format": "PNG",
                "compressed": True,
            },
            output_response=output_content,
            runtime_seconds=runtime,
            url=url,
            title=title,
        )

        # Debug information is now logged via profiling module

        return output_content

    except Exception as e:
        logger.error(f"Error with LLM screenshot processing: {e}")
        # Fallback to basic description if LLM fails
        fallback_content = (
            f"Screenshot of page: {title} ({url}) - Unable to analyze with LLM"
        )

        # Error info is now logged via profiling module

        return fallback_content


def determine_ui_click_element(client, image, action, model="gemini-2.5-flash"):
    """Use Gemini 2.5 Pro to analyze a screenshot and determine which element to click to take an action.

    Args:
        client: The LLM client
        image: PIL Image object of the screenshot
        action: The action to take on the page
    """
    try:
        screenshot_bytes_io = io.BytesIO()

        # Shrink the image by 50%
        image = image.resize((image.width // 2, image.height // 2))
        image.save(screenshot_bytes_io, format="PNG")

        # Convert PIL image to bytes
        screenshot_bytes = screenshot_bytes_io.getvalue()

        # Convert screenshot bytes to base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = dedent(f"""
        I am going to give you a screenshot that is annotated with a bunch of bounding boxes representing page elements. I will also give you an action I want to take on the page. Your job is to determine which element I should click to take that action. Return json with the following format:

        {{
            'element_id': 'the id of the element to click, can be None if you don't think there is a good element to click',
            'reasoning': 'your reasoning for why you chose this element, or why you think there is no good element to click'
        }}

        Action: {action}
        """)

        # Use the client with specified model and profiling
        start_time = time.time()
        with profile(
            "UI element selection",
            model,
            action=action,
            image_size_bytes=len(screenshot_bytes),
            image_base64_chars=len(screenshot_base64),
        ):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing web page screenshots and identifying UI elements, components, and interactive features.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_base64}"
                                },
                            },
                        ],
                    },
                ],
            )

        output_content = response.choices[0].message.content
        runtime = time.time() - start_time

        # Log I/O for debugging
        log_model_io(
            "UI element selection",
            input_data=prompt,
            output_data=output_content,
            action=action,
            image_size_bytes=len(screenshot_bytes),
        )

        # Log inference results
        from skipper_tool.inference_logger import log_vision_call

        log_vision_call(
            model_name=model,
            operation="UI element selection",
            input_prompt=prompt,
            image_data={
                "size_bytes": len(screenshot_bytes),
                "base64_chars": len(screenshot_base64),
                "format": "PNG",
                "compressed": True,
            },
            output_response=output_content,
            runtime_seconds=runtime,
            action=action,
        )

        return output_content

    except Exception as e:
        logger.error(f"Error with UI element determination: {e}")
        # Simple fallback
        return f"Unable to analyze screenshot - Error: {str(e)}"
