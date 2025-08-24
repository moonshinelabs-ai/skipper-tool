import argparse
import os
from pathlib import Path

import numpy as np
from loguru import logger

from skipper_tool.omniparser.client import OmniParser


def main():
    parser = argparse.ArgumentParser(
        description="Annotate screenshots using OmniParser to detect UI elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.png
  %(prog)s input.png --output-dir ./results
        """,
    )

    # Required arguments
    parser.add_argument("yolo_path", help="Path to the YOLO model file")
    parser.add_argument("image_path", help="Path to the input image file")

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory for annotated images (default: current directory)",
    )

    parser.add_argument(
        "--annotated-output",
        help="Filename for the annotated output image (default: annotated_<input_filename>)",
    )

    parser.add_argument(
        "--no-regular", action="store_true", help="Skip creating regular annotation"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed element information",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.image_path):
        logger.error(f"Image file '{args.image_path}' not found")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate default output filenames if not provided
    input_path = Path(args.image_path)
    input_stem = input_path.stem
    input_suffix = input_path.suffix

    if not args.annotated_output:
        annotated_filename = f"annotated_{input_stem}{input_suffix}"
    else:
        annotated_filename = args.annotated_output

    annotated_path = output_dir / annotated_filename

    omni_parser = OmniParser(yolo_model_path=args.yolo_path)

    logger.info(f"Processing image: {args.image_path}")

    # Get predictions and annotated image
    predictions = omni_parser.predict(args.image_path)
    logger.info(f"Found {len(predictions['boxes'])} UI elements")

    # Write predictions to numpy array
    np.save(output_dir / f"{input_stem}.npy", predictions)

    # Save annotated image if created
    if not args.no_regular and "annotated_image" in predictions:
        omni_parser.save_annotated_image(
            predictions["annotated_image"], str(annotated_path)
        )
        logger.info(f"Saved annotated image: {annotated_path}")

    # Print element details if verbose
    if args.verbose:
        logger.info("\nDetected elements:")
        for i, content in enumerate(predictions["parsed_content"]):
            logger.info(f"  Element {i}: {content}")


if __name__ == "__main__":
    main()
