"""
This module contains functions for getting run reports from Langfuse.
⚠️ It is created to be executed manually.

Running this file:
```
make shell
# Get summary of a run
python src/seer/automation/codegen/evals/datasets.py run-summary --dataset-name <dataset-name> --run-name <run-name>
# Get details of a run
python src/seer/automation/codegen/evals/datasets.py run-details --dataset-name <dataset-name> --run-name <run-name>
```
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Self

import click
from click import option
from langfuse import Langfuse
from langfuse.api.resources.commons.errors import NotFoundError
from langfuse.api.resources.commons.types.dataset_run_item import DatasetRunItem
from langfuse.api.resources.commons.types.dataset_run_with_items import DatasetRunWithItems

from seer.automation.codegen.evals.models import EvalItemInput, EvalItemOutput
from seer.automation.codegen.models import BugPrediction


@click.group()
def main():
    pass


@main.command()
@option("--dataset-name", type=str, required=True)
@option("--run-name", type=str, required=True)
def run_summary(dataset_name: str, run_name: str):
    """
    Generate and display a summary report for a specific dataset run.

    Args:
        dataset_name (str): Name of the dataset to analyze
        run_name (str): Name of the specific run to analyze
    """
    langfuse = Langfuse()
    try:
        run = langfuse.get_dataset_run(dataset_name, run_name)
    except NotFoundError as e:
        click.echo(f"❌ Run {run_name} not found: {e}")
        return
    summary = calculate_run_summary(langfuse, run)

    # Print report on the run
    evaluated_successfully_percentage = (
        summary.items_evaluated_successfully / max(summary.items_count, 1) * 100
    )
    evaluated_with_errors_percentage = 100 - evaluated_successfully_percentage

    click.echo("\n" + "=" * 42)
    click.echo(f"Run {run_name}")
    click.echo("=" * 42 + "\n")

    # Summary table
    summary_table = create_table(
        title="Summary",
        subtitle=f"Total items: {summary.items_count}",
        headers=["Metric", "Count", "Percentage"],
        rows=[
            [
                "Evaluated successfully",
                str(summary.items_evaluated_successfully),
                f"{evaluated_successfully_percentage:.2f}%",
            ],
            [
                "Evaluated with errors",
                str(summary.items_count - summary.items_evaluated_successfully),
                f"{evaluated_with_errors_percentage:.2f}%",
            ],
        ],
        alignments=["left", "right", "right"],
    )
    click.echo("\n".join(summary_table))

    # Predicted bugs count distribution
    predicted_bugs_count_distribution = create_distribution_chart(
        title="Predicted Bugs Count Distribution (All Items)",
        metric_label="Predicted Bugs",
        distribution=summary.predicted_bugs_count_distribution,
    )
    click.echo("\n".join(predicted_bugs_count_distribution))

    # Positive items statistics
    positive_items_table = create_table(
        title="Positive Items Statistics (items with expected bugs)",
        subtitle=f"Total items: {summary.positive_items_summary.items_count}",
        headers=["Metric", "Avg", "Total"],
        rows=[
            [
                "Total bugs expected",
                f"{summary.positive_items_summary.total_bugs_expected / summary.positive_items_summary.items_count:.2f}",
                f"{summary.positive_items_summary.total_bugs_expected:.2f}",
            ],
            [
                "Total bugs found",
                f"{summary.positive_items_summary.total_bugs_found / summary.positive_items_summary.items_count:.2f}",
                f"{summary.positive_items_summary.total_bugs_found:.2f}",
            ],
            [
                "Total predicted bugs",
                f"{summary.positive_items_summary.total_predicted_bugs / summary.positive_items_summary.items_count:.2f}",
                f"{summary.positive_items_summary.total_predicted_bugs:.2f}",
            ],
        ],
        alignments=["left", "right", "right"],
    )
    click.echo("\n".join(positive_items_table))

    # Quality metrics
    quality_metrics_table = create_table(
        title="Quality Metrics",
        subtitle=None,
        headers=["Metric", "Score"],
        rows=[
            ["Precision", f"{summary.positive_items_summary.precision:.2f}"],
            ["Recall", f"{summary.positive_items_summary.recall:.2f}"],
            ["F1 Score", f"{summary.positive_items_summary.f1_score:.2f}"],
            ["Avg Content Match", f"{summary.positive_items_summary.avg_content_match:.2f}"],
            ["Avg Location Match", f"{summary.positive_items_summary.avg_location_match:.2f}"],
        ],
        alignments=["left", "right"],
    )
    click.echo("\n".join(quality_metrics_table))

    # Negative items statistics
    negative_items_table = create_table(
        title="Negative Items Statistics (items without expected bugs)",
        subtitle=f"Total items: {summary.negative_items_summary.items_count}",
        headers=["Metric", "Avg", "Total"],
        rows=[
            [
                "Total predicted bugs",
                (
                    f"{summary.negative_items_summary.total_predicted_bugs / summary.negative_items_summary.items_count:.2f}"
                    if summary.negative_items_summary.items_count > 0
                    else "0.00"
                ),
                f"{summary.negative_items_summary.total_predicted_bugs:.2f}",
            ],
        ],
        alignments=["left", "right", "right"],
    )
    click.echo("\n".join(negative_items_table))

    # False positives distribution
    false_positives_distribution = create_distribution_chart(
        title="False Positives Distribution",
        metric_label="False Positives",
        distribution=summary.negative_items_summary.false_positives_distribution,
    )
    click.echo("\n".join(false_positives_distribution))


@main.command()
@option("--dataset-name", type=str, required=True)
@option("--run-name", type=str, required=True)
@option("--format", type=click.Choice(["md"]), required=True, default="md")
def run_details(dataset_name: str, run_name: str, format: Literal["md"]):
    """
    Generate and display detailed information about each item in a dataset run.

    Args:
        dataset_name (str): Name of the dataset to analyze
        run_name (str): Name of the specific run to analyze
        format (Literal["md"]): Output format (currently only supports markdown)
    """
    langfuse = Langfuse()
    try:
        run = langfuse.get_dataset_run(dataset_name, run_name)
    except NotFoundError as e:
        click.echo(f"❌ Run {run_name} not found: {e}")
        return
    summary = calculate_run_summary(langfuse, run)

    click.echo("\n# Items details")
    click.echo(" (items marked with ❌ were not evaluated successfully)\n")
    for item in summary.item_details:
        click.echo("\n".join(item.to_markdown()))
        click.echo("")


class RelevantScorePrefixes(Enum):
    """Enumeration of score prefixes used in evaluation metrics.
    (the actual score name also includes the model used to generate the score)
    """

    NOISE = "noise"
    BUGS_NOT_FOUND = "bugs_not_found"
    CONTENT_MATCH = "content_match"
    LOCATION_MATCH = "location_match"
    BUGS_FOUND_COUNT = "bugs_found_count"

    def get_score_by_prefix(self, scores: list[dict]) -> dict | None:
        return next((s for s in scores if s["name"].startswith(self.value)), None)


def pretty_print_scores(scores: list[dict]) -> list[str]:
    """
    Format scores for display, highlighting important metrics.

    Args:
        scores (list[dict]): List of score dictionaries to format

    Returns:
        list[str]: List of formatted strings representing the scores
    """
    # Filter out error_running_evaluation and sort by highlight order
    filtered_scores = [s for s in scores if s["name"] != "error_running_evaluation"]

    # Print scores
    lines = []
    for score in filtered_scores:
        score_name = score["name"]
        highlight_name = next(
            (name.value for name in RelevantScorePrefixes if name.value in score_name), None
        )
        value = score["value"]
        comment = score["comment"]

        if highlight_name:
            display_name = (
                click.style(highlight_name, bold=True) + score_name[len(highlight_name) :]
            )
        else:
            display_name = score_name

        display_value = click.style(str(value), bold=True)

        lines.append(f"* {display_name}: {display_value}")
        lines.append(f"  {comment}")
    return lines


def pretty_print_expected_bugs(expected_bugs: list[EvalItemOutput]) -> list[str]:
    """
    Format expected bugs for display.

    Args:
        expected_bugs (list[EvalItemOutput]): List of expected bug outputs to format

    Returns:
        list[str]: List of formatted strings representing the expected bugs
    """
    lines = []
    for bug in expected_bugs:
        description_lines = bug.description.strip().split("\n")
        lines.append(f"* {bug.encoded_location}")
        lines.append(f"  {description_lines[0]}")
        lines.extend(map(lambda line: (" " * 2) + line, description_lines[1:]))
    return lines


def pretty_print_generated_bug_predictions(
    generated_bug_predictions: list[BugPrediction],
) -> list[str]:
    """
    Format generated bug predictions for display.

    Args:
        generated_bug_predictions (list[BugPrediction]): List of bug predictions to format

    Returns:
        list[str]: List of formatted strings representing the bug predictions
    """
    lines = []
    for bug_prediction in generated_bug_predictions:
        lines.append(f"* {bug_prediction.encoded_location}")
        lines.append(f"  {bug_prediction.short_description}")
        lines.append(f"  {bug_prediction.description}")
        lines.append(f"  {bug_prediction.suggested_fix}")
        lines.append(
            f"  severity: {bug_prediction.severity}; confidence: {bug_prediction.confidence}"
        )
        lines.append("")
    return lines


@dataclass
class ItemOrigin:
    org_name: str
    repo_name: str
    pr_id: int
    commit_sha: str

    @classmethod
    def from_input(cls, input: EvalItemInput) -> Self:
        return cls(
            org_name=input.repo.owner,
            repo_name=input.repo.name,
            pr_id=input.pr_id,
            commit_sha=input.commit_sha,
        )


@dataclass
class RelevantItemInfo:
    item_id: str
    trace_id: str
    was_evaluated_successfully: bool
    scores: list[dict]
    origin: ItemOrigin
    expected_bugs: list[EvalItemOutput]
    generated_bug_predictions: list[BugPrediction]

    def to_markdown(self) -> list[str]:
        """
        Convert the item information to markdown format.

        Returns:
            list[str]: List of strings representing the item in markdown format
        """
        success_symbol = "✓" if self.was_evaluated_successfully else "❌"
        lines = []
        lines.append(f"{success_symbol} Item {self.item_id} - Trace ID: {self.trace_id}")
        lines.append(
            f"  Origin: {self.origin.org_name}/{self.origin.repo_name} - PR #{self.origin.pr_id} - Commit SHA: {self.origin.commit_sha} - URL: https://github.com/{self.origin.org_name}/{self.origin.repo_name}/pull/{self.origin.pr_id}"
        )
        if self.was_evaluated_successfully:
            padding = 4
            lines.append("    # Scores")
            lines.append("    --------------------------------")
            lines.extend(map(lambda line: " " * padding + line, pretty_print_scores(self.scores)))
            lines.append("")
            lines.append("    # Expected bugs")
            lines.append("    --------------------------------")
            if self.expected_bugs:
                lines.extend(
                    map(
                        lambda line: " " * padding + line,
                        pretty_print_expected_bugs(self.expected_bugs),
                    )
                )
            else:
                lines.append("    No expected bugs")
            lines.append("")
            lines.append("    # Generated bug predictions")
            lines.append("    --------------------------------")
            if self.generated_bug_predictions:
                lines.extend(
                    map(
                        lambda line: " " * padding + line,
                        pretty_print_generated_bug_predictions(self.generated_bug_predictions),
                    )
                )
            else:
                lines.append("    No generated bug predictions")
        return lines


@dataclass
class PositiveItemsSummary:
    items_count: int
    total_bugs_expected: int
    total_bugs_found: int
    total_predicted_bugs: int
    avg_content_match: float
    avg_location_match: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class NegativeItemsSummary:
    items_count: int
    total_predicted_bugs: int
    false_positives_distribution: dict[int, int]  # Maps number of false positives to count of items


@dataclass
class RunSummaryInfo:
    items_count: int
    items_evaluated_successfully: int
    item_details: list[RelevantItemInfo]
    positive_items_summary: PositiveItemsSummary
    negative_items_summary: NegativeItemsSummary
    predicted_bugs_count_distribution: dict[
        int, int
    ]  # Maps number of predicted bugs to count of items


def get_relevant_info_for_item(langfuse: Langfuse, item: DatasetRunItem) -> RelevantItemInfo:
    """
    Extract relevant information from a dataset run item.

    Args:
        langfuse (Langfuse): Langfuse client instance
        item (DatasetRunItem): Dataset run item to analyze

    Returns:
        RelevantItemInfo: Structured information about the item
    """
    trace = langfuse.fetch_trace(item.trace_id)
    dataset_item = langfuse.get_dataset_item(item.dataset_item_id)

    # Load the expected issues from the dataset item.
    if not dataset_item.expected_output:
        # This happens in negative test cases where we don't have any expected issues.
        # The value is `{}` in this case.
        list_of_issues = []
    elif isinstance(dataset_item.expected_output, list):
        list_of_issues = dataset_item.expected_output
    else:
        list_of_issues = [dataset_item.expected_output]
    list_of_issues = [EvalItemOutput.model_validate(issue) for issue in list_of_issues]

    item_origin = ItemOrigin.from_input(EvalItemInput.model_validate(dataset_item.input))

    generated_bug_predictions = (
        [BugPrediction.model_validate(prediction) for prediction in trace.data.output]
        if trace.data.output
        else []
    )
    scores: list[dict] = [
        {"name": score.name, "value": score.value, "comment": score.comment}
        for score in trace.data.scores
    ]
    error_score = next(
        (score for score in scores if score["name"] == "error_running_evaluation"), None
    )
    was_evaluated_successfully = error_score is not None and error_score["value"] == 0
    return RelevantItemInfo(
        item_id=dataset_item.id,
        trace_id=item.trace_id,
        was_evaluated_successfully=was_evaluated_successfully,
        scores=scores,
        expected_bugs=list_of_issues,
        generated_bug_predictions=generated_bug_predictions,
        origin=item_origin,
    )


@dataclass
class ItemDetailedScores:
    bugs_expected: int
    bugs_found: int
    content_match: float
    location_match: float
    noise: float
    predicted_bugs_count: int


def calculate_item_detailed_scores(item: RelevantItemInfo) -> ItemDetailedScores | None:
    bugs_found_score = RelevantScorePrefixes.BUGS_FOUND_COUNT.get_score_by_prefix(item.scores)
    if bugs_found_score is None:
        return None
    match = re.match(r"Expected number of bugs: (\d+);", bugs_found_score["comment"])
    if match is None:
        return None
    bugs_expected = int(match.group(1))
    content_match_score = RelevantScorePrefixes.CONTENT_MATCH.get_score_by_prefix(item.scores)
    location_match_score = RelevantScorePrefixes.LOCATION_MATCH.get_score_by_prefix(item.scores)
    noise_score = RelevantScorePrefixes.NOISE.get_score_by_prefix(item.scores)
    # All bug predictions are the matched (bugs_found) + unmatched (noise) bug predictions
    predicted_bugs_count = noise_score["value"] if noise_score else 0 + bugs_found_score["value"]
    return ItemDetailedScores(
        bugs_expected=bugs_expected,
        bugs_found=bugs_found_score["value"],
        content_match=content_match_score["value"] if content_match_score else 0,
        location_match=location_match_score["value"] if location_match_score else 0,
        noise=noise_score["value"] if noise_score else 0,
        predicted_bugs_count=predicted_bugs_count,
    )


def calculate_run_summary(langfuse: Langfuse, run: DatasetRunWithItems) -> RunSummaryInfo:
    """
    Calculate summary statistics for a dataset run.

    Args:
        langfuse (Langfuse): Langfuse client instance
        run (DatasetRunWithItems): Dataset run to analyze

    Returns:
        RunSummaryInfo: Summary information about the run
    """
    items_in_run = [get_relevant_info_for_item(langfuse, item) for item in run.dataset_run_items]

    # Calculate averages for successful evaluations only
    successful_items = [item for item in items_in_run if item.was_evaluated_successfully]
    positive_items_summary: dict = {
        "items_count": 0,
        "total_bugs_found": 0,
        "total_bugs_expected": 0,
        "total_predicted_bugs": 0,
        "total_content_match": 0.0,
        "total_location_match": 0.0,
    }
    negative_items_summary: dict = {
        "items_count": 0,
        "total_predicted_bugs": 0,
        "false_positives_distribution": {},
    }
    predicted_bugs_count_distribution: dict[int, int] = {}

    for item in successful_items:
        item_detailed_scores = calculate_item_detailed_scores(item)
        if item_detailed_scores is None:
            click.echo(f"! Bugs found score not found for item {item.item_id}. Ignoring item.")
            continue

        # Update overall predicted bugs count distribution
        predicted_bugs_count = item_detailed_scores.bugs_found + item_detailed_scores.noise
        predicted_bugs_count_distribution[predicted_bugs_count] = (
            predicted_bugs_count_distribution.get(predicted_bugs_count, 0) + 1
        )

        if item_detailed_scores.bugs_expected == 0:
            # Negative item (no expected bugs)
            negative_items_summary["items_count"] += 1
            negative_items_summary["total_predicted_bugs"] += predicted_bugs_count

            # Update false positives distribution
            negative_items_summary["false_positives_distribution"][predicted_bugs_count] = (
                negative_items_summary["false_positives_distribution"].get(predicted_bugs_count, 0)
                + 1
            )
        else:
            # Positive item (has expected bugs)
            positive_items_summary["items_count"] += 1
            positive_items_summary["total_bugs_found"] += item_detailed_scores.bugs_found
            positive_items_summary["total_bugs_expected"] += item_detailed_scores.bugs_expected
            positive_items_summary["total_predicted_bugs"] += predicted_bugs_count
            positive_items_summary["total_content_match"] += item_detailed_scores.content_match
            positive_items_summary["total_location_match"] += item_detailed_scores.location_match

    # Calculate precision, recall, and F1 score
    total_true_positives = positive_items_summary["total_bugs_found"]
    total_predicted = (
        positive_items_summary["total_predicted_bugs"]
        + negative_items_summary["total_predicted_bugs"]
    )

    precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
    recall = (
        positive_items_summary["total_bugs_found"] / positive_items_summary["total_bugs_expected"]
        if positive_items_summary["total_bugs_expected"] > 0
        else 0.0
    )
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate averages for content and location match only for bugs that were found
    avg_content_match = (
        positive_items_summary["total_content_match"] / positive_items_summary["total_bugs_found"]
        if positive_items_summary["total_bugs_found"] > 0
        else 0.0
    )
    avg_location_match = (
        positive_items_summary["total_location_match"] / positive_items_summary["total_bugs_found"]
        if positive_items_summary["total_bugs_found"] > 0
        else 0.0
    )

    return RunSummaryInfo(
        items_count=len(items_in_run),
        items_evaluated_successfully=sum(item.was_evaluated_successfully for item in items_in_run),
        item_details=items_in_run,
        positive_items_summary=PositiveItemsSummary(
            items_count=positive_items_summary["items_count"],
            total_bugs_expected=positive_items_summary["total_bugs_expected"],
            total_bugs_found=positive_items_summary["total_bugs_found"],
            total_predicted_bugs=positive_items_summary["total_predicted_bugs"],
            avg_content_match=avg_content_match,
            avg_location_match=avg_location_match,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        ),
        negative_items_summary=NegativeItemsSummary(
            items_count=negative_items_summary["items_count"],
            total_predicted_bugs=negative_items_summary["total_predicted_bugs"],
            false_positives_distribution=negative_items_summary["false_positives_distribution"],
        ),
        predicted_bugs_count_distribution=predicted_bugs_count_distribution,
    )


def build_enclosing_row(
    column_widths: list[int], position: Literal["top", "middle", "bottom"]
) -> str:
    chars_lookup = {
        "top": ["┌", "┬", "┐"],
        "middle": ["├", "┼", "┤"],
        "bottom": ["└", "┴", "┘"],
    }
    left_char, middle_char, right_char = chars_lookup[position]
    row = left_char
    for i in range(len(column_widths)):
        row += "─" * (column_widths[i] + 2) + middle_char
    return row[:-1] + right_char


def create_table(
    title: str | None,
    subtitle: str | None,
    headers: list[str],
    rows: list[list[str]],
    column_widths: list[int] | None = None,
    alignments: list[str] | None = None,
) -> list[str]:
    """
    Creates a formatted table for terminal output.

    Args:
        title (str | None): Optional title for the table
        subtitle (str | None): Optional subtitle for the table
        headers (list[str]): List of column headers
        rows (list[list[str]]): List of rows, where each row is a list of cell values
        column_widths (list[int] | None): Optional list of column widths. If not provided, will be calculated based on content
        alignments (list[str] | None): Optional list of alignments ('left', 'right', 'center'). Defaults to 'left'

    Returns:
        list[str]: List of strings representing the table lines
    """
    # Check that the length of each row matches length of headers
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Length of each row must match length of headers")

    # Calculate column widths if not provided
    if column_widths is None:
        column_widths = [len(header) for header in headers]
        for row in rows:
            for i in range(len(row)):
                column_widths[i] = max(column_widths[i], len(row[i]))

    # Set default alignments if not provided
    if alignments is None:
        alignments = ["left"] * len(headers)

    # Create the table
    lines = []

    # Add title if provided
    if title:
        lines.append(click.style(f"# {title}", bold=True))
    if subtitle:
        lines.append(subtitle)

    # Create the format string for rows
    format_parts = []
    for width, alignment in zip(column_widths, alignments):
        if alignment == "right":
            format_parts.append(f"{{:>{width}}}")
        elif alignment == "center":
            format_parts.append(f"{{:^{width}}}")
        else:  # left
            format_parts.append(f"{{:<{width}}}")
    row_format = " │ ".join(format_parts)

    # Create the table
    first_row = build_enclosing_row(column_widths, "top")
    lines.append(first_row)

    # Headers
    header_cells = [str(h) for h in headers]
    lines.append("│ " + row_format.format(*header_cells) + " │")

    middle_row = build_enclosing_row(column_widths, "middle")
    lines.append(middle_row)

    # Rows
    for row in rows:
        row_cells = [str(cell) for cell in row]
        lines.append("│ " + row_format.format(*row_cells) + " │")

    last_row = build_enclosing_row(column_widths, "bottom")
    lines.append(last_row)

    return lines


def create_distribution_chart(
    title: str,
    metric_label: str,
    distribution: dict[int, int],
    bar_length: int = 30,
) -> list[str]:
    """
    Creates a bar chart visualization of a distribution for terminal output.

    Args:
        title (str): Title of the chart
        metric_label (str): Label for the metric being displayed
        distribution (dict[int, int]): Dictionary mapping values to their counts
        bar_length (int, optional): Maximum length of the bars in characters. Defaults to 30.

    Returns:
        list[str]: List of strings representing the chart lines
    """
    if not distribution:
        return []

    lines = []
    lines.append(click.style(f"# {title}", bold=True))
    lines.append(f"{metric_label} | Count")

    max_count = max(distribution.values())
    for value, count in sorted(distribution.items()):
        bar = "█" * int((count / max_count) * bar_length)
        format_part = f"{{:>{len(metric_label)}.1f}}"
        lines.append(f"{format_part.format(value)} | {bar} {count}")

    lines.append("")  # Add empty line at the end
    return lines


if __name__ == "__main__":
    main()
