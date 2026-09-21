"""
main.py — Entry point for the AI Data Analyst.

HOW TO RUN:
  python main.py --file data.csv
  python main.py --file data.csv --question "What are the top selling products?"
  python main.py --demo    (uses built-in sample data)

LINE BY LINE:
  argparse               → standard Python CLI argument library
  --file                 → path to the CSV to analyse
  --question             → optional custom question for the LLM
                           if given, the LLM focuses its code on answering this
  --demo                 → generates sample data and analyses it
                           useful for testing without having a real CSV

  args.demo              → if --demo flag is set:
    _create_sample_csv() → generates a realistic sample CSV
    pipeline.run(...)    → runs the full pipeline on it

  _create_sample_csv()   → creates a sample CSV with realistic data
    numpy.random         → generates random numbers
    np.random.seed(42)   → seed = 42 makes random numbers reproducible
                           same seed always gives same "random" data
    np.random.randint()  → random integers in a range
    np.random.choice()   → random selection from a list
    np.random.normal()   → numbers from a normal distribution
                           mean=50000 = centre of distribution
                           std=15000 = spread (standard deviation)
    pd.DataFrame(data)   → creates a DataFrame from a dict
    df.to_csv(path, index=False)
                         → saves to CSV
                         → index=False: don't save row numbers as a column
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from pipeline import DataAnalystPipeline

console = Console()


def main():
    # ── Parse command-line arguments ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="AI Data Analyst — upload CSV, get insights and charts"
    )
    parser.add_argument(
        "--file", type=str,
        help="Path to CSV file to analyse",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with built-in sample sales data",
    )
    args = parser.parse_args()

    # ── Get CSV path ──────────────────────────────────────────────────────
    if args.demo:
        csv_path = _create_sample_csv()
        console.print(f"[green]✓ Sample data created: {csv_path}[/green]")

    elif args.file:
        csv_path = args.file

    else:
        # Interactive mode — ask user
        csv_path = Prompt.ask("[bold cyan]Enter path to your CSV file[/bold cyan]")

    # ── Run the pipeline ──────────────────────────────────────────────────
    pipeline = DataAnalystPipeline()
    result = pipeline.run(csv_path)

    if result is None:
        console.print("[red]Pipeline failed. Check the file path and try again.[/red]")
        return

    if result.success:
        console.print("\n[bold green]✓ Done! Check the outputs/ folder for:[/bold green]")
        console.print("  • Chart PNG files")
        console.print("  • Markdown report")
    else:
        console.print("\n[yellow]⚠ Analysis completed with errors — partial results available[/yellow]")


def _create_sample_csv() -> str:
    """
    Create a realistic sample sales dataset for demo purposes.

    Uses numpy random with a fixed seed (42) so the data is
    always the same — reproducible results.

    Returns path to the created CSV file.
    """
    import numpy as np
    import pandas as pd

    # Fixed seed = same "random" data every time
    # Without this, each run would produce different data
    np.random.seed(42)

    n = 500   # number of rows

    data = {
        "date": pd.date_range(start="2023-01-01", periods=n, freq="D").astype(str),
        "product": np.random.choice(
            ["Laptop", "Phone", "Tablet", "Watch", "Headphones"], n
        ),
        "region": np.random.choice(
            ["North", "South", "East", "West"], n
        ),
        "sales_rep": np.random.choice(
            ["Alice", "Bob", "Carol", "David", "Eve"], n
        ),
        # Normal distribution: most values near mean, few outliers
        "revenue": np.random.normal(loc=50000, scale=15000, size=n).round(2),
        "units_sold": np.random.randint(1, 50, n),
        "customer_rating": np.round(np.random.uniform(3.0, 5.0, n), 1),
        # Some nulls in discount column — real-world data has missing values
        "discount_pct": np.where(
            np.random.random(n) > 0.3,   # 70% have a discount
            np.random.randint(0, 30, n),  # discount 0-30%
            None                           # 30% have no discount (null)
        ),
    }

    df = pd.DataFrame(data)

    # Save to sample_data folder
    path = "sample_data/sales_data.csv"
    Path("sample_data").mkdir(exist_ok=True)
    df.to_csv(path, index=False)

    return path


if __name__ == "__main__":
    main()
