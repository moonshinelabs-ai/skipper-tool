#!/usr/bin/env python3
"""
Utility script to analyze saved model inference logs.

This script provides tools to view, filter, and analyze the inference results
saved by the InferenceLogger.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.text import Text


def load_inference_files(log_dir: Path) -> Dict[str, List[dict]]:
    """Load all inference files from the log directory."""
    inference_data = {"llm_calls": [], "vision_calls": [], "yolo_predictions": []}

    for category in inference_data.keys():
        category_dir = log_dir / category
        if category_dir.exists():
            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        data["_file_path"] = str(json_file)
                        inference_data[category].append(data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading {json_file}: {e}")

    return inference_data


def print_summary(inference_data: Dict[str, List[dict]], console: Console):
    """Print a summary table of all inference logs."""
    table = Table(title="Inference Log Summary")

    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Total Runtime (s)", justify="right", style="green")
    table.add_column("Avg Runtime (s)", justify="right", style="yellow")

    for category, data in inference_data.items():
        if data:
            if category == "vision_calls":
                # Group vision calls by operation name
                operations = {}
                for item in data:
                    operation = item.get("operation", "unknown")
                    if operation not in operations:
                        operations[operation] = []
                    operations[operation].append(item)

                # Add a row for each operation
                for operation, operation_data in operations.items():
                    total_runtime = sum(
                        item.get("runtime_seconds", 0) for item in operation_data
                    )
                    avg_runtime = total_runtime / len(operation_data)

                    table.add_row(
                        f"Vision ({operation})",
                        str(len(operation_data)),
                        f"{total_runtime:.2f}",
                        f"{avg_runtime:.2f}",
                    )
            else:
                total_runtime = sum(item.get("runtime_seconds", 0) for item in data)
                avg_runtime = total_runtime / len(data)

                table.add_row(
                    category.replace("_", " ").title(),
                    str(len(data)),
                    f"{total_runtime:.2f}",
                    f"{avg_runtime:.2f}",
                )

    console.print(table)


def print_detailed_logs(
    inference_data: Dict[str, List[dict]], category: str, console: Console
):
    """Print detailed logs for a specific category."""
    data = inference_data.get(category, [])
    if not data:
        console.print(f"No {category} found.")
        return

    for i, item in enumerate(data):
        console.print(f"\n[bold cyan]{'=' * 50}")
        console.print(f"[bold cyan]{category.replace('_', ' ').title()} #{i + 1}")
        console.print(f"[bold cyan]{'=' * 50}")

        console.print(f"[bold]Timestamp:[/bold] {item.get('timestamp', 'N/A')}")
        console.print(f"[bold]Model:[/bold] {item.get('model_name', 'N/A')}")
        console.print(f"[bold]Operation:[/bold] {item.get('operation', 'N/A')}")
        console.print(f"[bold]Runtime:[/bold] {item.get('runtime_seconds', 0):.3f}s")

        if "input" in item:
            input_data = item["input"]
            if "prompt" in input_data:
                prompt = input_data["prompt"]
                console.print(
                    f"[bold]Input Length:[/bold] {input_data.get('prompt_length', len(prompt))} chars"
                )
                console.print("[bold]Input Prompt:[/bold]")
                console.print(
                    Text(
                        prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        style="dim",
                    )
                )

            if "image" in input_data:
                console.print(f"[bold]Image Info:[/bold] {input_data['image']}")

        if "output" in item:
            output_data = item["output"]
            if "response" in output_data:
                response = output_data["response"]
                console.print(
                    f"[bold]Output Length:[/bold] {output_data.get('response_length', len(response))} chars"
                )
                console.print("[bold]Output Response:[/bold]")
                console.print(
                    Text(
                        response[:300] + "..." if len(response) > 300 else response,
                        style="dim",
                    )
                )

            if "predictions" in output_data:
                predictions = output_data["predictions"]
                console.print(
                    f"[bold]Predictions:[/bold] {predictions.get('num_detections', 0)} detections"
                )
                console.print(f"[bold]Prediction Details:[/bold] {predictions}")

        if item.get("metadata"):
            console.print(f"[bold]Metadata:[/bold] {item['metadata']}")


def filter_by_model(
    inference_data: Dict[str, List[dict]], model_name: str
) -> Dict[str, List[dict]]:
    """Filter inference data by model name."""
    filtered_data = {}
    for category, data in inference_data.items():
        filtered_data[category] = [
            item
            for item in data
            if item.get("model_name", "").lower() == model_name.lower()
        ]
    return filtered_data


def filter_by_operation(
    inference_data: Dict[str, List[dict]], operation: str
) -> Dict[str, List[dict]]:
    """Filter inference data by operation type."""
    filtered_data = {}
    for category, data in inference_data.items():
        filtered_data[category] = [
            item
            for item in data
            if operation.lower() in item.get("operation", "").lower()
        ]
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Analyze model inference logs")
    parser.add_argument("log_dir", help="Directory containing inference logs")
    parser.add_argument(
        "--category",
        choices=["llm_calls", "vision_calls", "yolo_predictions"],
        help="Show detailed logs for specific category",
    )
    parser.add_argument("--model", help="Filter by model name")
    parser.add_argument("--operation", help="Filter by operation name")
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only summary table"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return

    console = Console()

    # Load inference data
    console.print(f"Loading inference logs from {log_dir}...")
    inference_data = load_inference_files(log_dir)

    # Apply filters
    if args.model:
        inference_data = filter_by_model(inference_data, args.model)
        console.print(f"Filtered by model: {args.model}")

    if args.operation:
        inference_data = filter_by_operation(inference_data, args.operation)
        console.print(f"Filtered by operation: {args.operation}")

    # Show summary
    print_summary(inference_data, console)

    # Show detailed logs if requested
    if not args.summary_only:
        if args.category:
            print_detailed_logs(inference_data, args.category, console)
        else:
            for category in ["llm_calls", "vision_calls", "yolo_predictions"]:
                if inference_data[category]:
                    print_detailed_logs(inference_data, category, console)


if __name__ == "__main__":
    main()
