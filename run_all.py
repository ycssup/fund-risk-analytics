import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_PATH = "data/sample_nav_data.xlsx"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_BENCHMARK_CANDIDATES = (
    "data/benchmark.xlsx",
    "data/benchmark.csv",
    "data/benchmark.xls",
    "data/benchmark_CSI 300.xlsx",
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the full pipeline runner.
    """
    parser = argparse.ArgumentParser(
        description="Run the full Fund Risk Analytics pipeline: analysis plus PDF report generation."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the fund NAV file (xlsx or csv). Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to the benchmark data file. If omitted, standard project benchmark files will be checked.",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Optional friendly benchmark name, for example 'CSI 300 Index'.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory root. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def resolve_benchmark_path(benchmark_arg: str | None, project_root: Path) -> Path:
    """
    Resolve the benchmark path from CLI input or standard project locations.
    """
    if benchmark_arg:
        return Path(benchmark_arg).expanduser().resolve()

    for candidate in DEFAULT_BENCHMARK_CANDIDATES:
        candidate_path = (project_root / candidate).resolve()
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Benchmark file not found. Pass --benchmark or add one of: "
        + ", ".join(DEFAULT_BENCHMARK_CANDIDATES)
    )


def validate_file(file_path: Path, label: str) -> None:
    """
    Validate that a required input path exists and points to a file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")


def format_command(command: list[str]) -> str:
    """
    Format a command for readable terminal logging.
    """
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str], project_root: Path) -> None:
    """
    Run a subprocess command and stream output directly to the terminal.
    """
    print(f">>> Running: {format_command(command)}")
    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command[1]}"
        )


def build_run_analysis_command(
    python_executable: str,
    project_root: Path,
    fund_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    benchmark_name: str | None,
) -> list[str]:
    """
    Build the command for scripts/run_analysis.py.
    """
    command = [
        python_executable,
        str(project_root / "scripts" / "run_analysis.py"),
        "--input",
        str(fund_path),
        "--benchmark",
        str(benchmark_path),
        "--output",
        str(output_dir),
    ]
    if benchmark_name:
        command.extend(["--benchmark-name", benchmark_name])
    return command


def build_generate_report_command(
    python_executable: str,
    project_root: Path,
    fund_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    report_path: Path,
    benchmark_name: str | None,
) -> list[str]:
    """
    Build the command for scripts/generate_report.py.
    """
    command = [
        python_executable,
        str(project_root / "scripts" / "generate_report.py"),
        "--input",
        str(fund_path),
        "--benchmark",
        str(benchmark_path),
        "--output",
        str(report_path),
        "--output-dir",
        str(output_dir),
    ]
    if benchmark_name:
        command.extend(["--benchmark-name", benchmark_name])
    return command


def main() -> None:
    """
    Validate inputs and run the complete analytics pipeline.
    """
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    fund_path = Path(args.input).expanduser().resolve()
    benchmark_path = resolve_benchmark_path(args.benchmark, project_root)
    output_dir = Path(args.output).expanduser().resolve()
    charts_dir = output_dir / "charts"
    reports_dir = output_dir / "reports"
    report_path = reports_dir / "fund_risk_report.pdf"

    validate_file(fund_path, "Fund file")
    validate_file(benchmark_path, "Benchmark file")

    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Fund Risk Analytics Pipeline")
    print(f"Fund file: {fund_path}")
    print(f"Benchmark file: {benchmark_path}")
    if args.benchmark_name:
        print(f"Benchmark name: {args.benchmark_name}")
    print(f"Output directory: {output_dir}")

    python_executable = sys.executable

    run_analysis_command = build_run_analysis_command(
        python_executable=python_executable,
        project_root=project_root,
        fund_path=fund_path,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        benchmark_name=args.benchmark_name,
    )
    run_command(run_analysis_command, project_root=project_root)

    generate_report_command = build_generate_report_command(
        python_executable=python_executable,
        project_root=project_root,
        fund_path=fund_path,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        report_path=report_path,
        benchmark_name=args.benchmark_name,
    )
    run_command(generate_report_command, project_root=project_root)

    print("Pipeline completed successfully")
    print(f"Charts folder: {charts_dir}")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
