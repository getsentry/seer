"""
This module contains functions for getting run reports from Langfuse.
⚠️ It is created to be executed manually.


"""

import re
from dataclasses import dataclass
from enum import Enum

import click
from click import option
from langfuse import Langfuse
from langfuse.api.resources.commons.errors import NotFoundError
from langfuse.api.resources.commons.types.dataset_run_item import DatasetRunItem
from langfuse.api.resources.commons.types.dataset_run_with_items import DatasetRunWithItems


@click.group()
def main():
    pass


@main.command()
@option("--dataset-name", type=str, required=True)
@option("--run-name", type=str, required=True)
@option(
    "--hide-details", is_flag=True, help="Whether to hide the details of the items.", default=False
)
def run_report(dataset_name: str, run_name: str, hide_details: bool):
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
    click.echo("# Summary")
    click.echo("┌─────────────────────────────┬────────┬────────────┐")
    click.echo("│ Metric                      │  Count │ Percentage │")
    click.echo("├─────────────────────────────┼────────┼────────────┤")
    click.echo(f"│ Total items                 │ {summary.items_count:>6} │    100.00% │")
    click.echo(
        f"│ Evaluated successfully      │ {summary.items_evaluated_successfully:>6} │ {evaluated_successfully_percentage:>9.2f}% │"
    )
    click.echo(
        f"│ Evaluated with errors       │ {summary.items_count - summary.items_evaluated_successfully:>6} │ {evaluated_with_errors_percentage:>9.2f}% │"
    )
    click.echo("└─────────────────────────────┴────────┴────────────┘")

    click.echo("\n# Suggestion Count Distribution (All Items)")
    click.echo("Suggestions | Count")
    max_count = max(summary.suggestion_count_distribution.values())
    bar_length = 30  # Maximum length of the bar
    for suggestion_count, item_count in sorted(summary.suggestion_count_distribution.items()):
        bar = "█" * int((item_count / max_count) * bar_length)
        click.echo(f"{suggestion_count:>11.1f} | {bar} {item_count}")

    click.echo("\n# Positive Items Statistics (items with expected bugs)")
    click.echo(f"Total items: {summary.positive_items_summary.items_count}")
    click.echo("┌─────────────────────────────┬──────────┬──────────┐")
    click.echo("│ Metric                      │   Avg    │   Total  │")
    click.echo("├─────────────────────────────┼──────────┼──────────┤")
    click.echo(
        f"│ Total bugs expected         │ {(summary.positive_items_summary.total_bugs_expected / summary.positive_items_summary.items_count):>8.2f} │ {summary.positive_items_summary.total_bugs_expected:>8.2f} │"
    )
    click.echo(
        f"│ Total bugs found            │ {(summary.positive_items_summary.total_bugs_found / summary.positive_items_summary.items_count):>8.2f} │ {summary.positive_items_summary.total_bugs_found:>8.2f} │"
    )
    click.echo(
        f"│ Total suggested bugs        │ {(summary.positive_items_summary.total_suggested_bugs / summary.positive_items_summary.items_count):>8.2f} │ {summary.positive_items_summary.total_suggested_bugs:>8.2f} │"
    )
    click.echo("└─────────────────────────────┴──────────┴──────────┘")

    click.echo("\n# Quality Metrics")
    click.echo("┌─────────────────────────────┬──────────┐")
    click.echo("│ Metric                      │  Score   │")
    click.echo("├─────────────────────────────┼──────────┤")
    click.echo(
        f"│ Precision                   │ {summary.positive_items_summary.precision:>8.2f} │"
    )
    click.echo(f"│ Recall                      │ {summary.positive_items_summary.recall:>8.2f} │")
    click.echo(f"│ F1 Score                    │ {summary.positive_items_summary.f1_score:>8.2f} │")
    click.echo(
        f"│ Avg Content Match           │ {summary.positive_items_summary.avg_content_match:>8.2f} │"
    )
    click.echo(
        f"│ Avg Location Match          │ {summary.positive_items_summary.avg_location_match:>8.2f} │"
    )
    click.echo("└─────────────────────────────┴──────────┘")

    click.echo("\n# Negative Items Statistics (items without expected bugs)")
    click.echo(f"Total items: {summary.negative_items_summary.items_count}")
    click.echo("┌─────────────────────────────┬──────────┬──────────┐")
    click.echo("│ Metric                      │   Avg    │   Total  │")
    click.echo("├─────────────────────────────┼──────────┼──────────┤")
    click.echo(
        f"│ Total suggested bugs        │ {(summary.negative_items_summary.total_suggested_bugs / summary.negative_items_summary.items_count):>8.2f} │ {summary.negative_items_summary.total_suggested_bugs:>8.2f} │"
    )
    click.echo("└─────────────────────────────┴──────────┴──────────┘")

    click.echo("\n# False positives distribution")
    click.echo("False Positives | Count")
    max_count = max(summary.negative_items_summary.false_positives_distribution.values())
    bar_length = 30  # Maximum length of the bar
    for fp_count, item_count in sorted(
        summary.negative_items_summary.false_positives_distribution.items()
    ):
        bar = "█" * int((item_count / max_count) * bar_length)
        click.echo(f"{fp_count:>15.1f} | {bar} {item_count}")

    if not hide_details:
        click.echo("\n# Items details")
        click.echo(" (items marked with ❌ were not evaluated successfully)\n")
        for item in summary.item_details:
            item.pretty_print()


class RelevantScorePrefixes(Enum):
    NOISE = "noise"
    BUGS_NOT_FOUND = "bugs_not_found"
    CONTENT_MATCH = "content_match"
    LOCATION_MATCH = "location_match"
    BUGS_FOUND_COUNT = "bugs_found_count"

    def get_score_by_prefix(self, scores: list[dict]) -> dict | None:
        return next((s for s in scores if s["name"].startswith(self.value)), None)


def pretty_print_scores(scores: list[dict]) -> None:
    """Print scores in a formatted way, highlighting important metrics."""

    # Filter out error_running_evaluation and sort by highlight order
    filtered_scores = [s for s in scores if s["name"] != "error_running_evaluation"]

    # Print scores
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

        click.echo(f"    * {display_name}: {display_value}")
        click.echo(f"      {comment}")


@dataclass
class RelevantItemInfo:
    item_id: str
    trace_id: str
    was_evaluated_successfully: bool
    scores: list[dict]

    def pretty_print(self) -> None:
        success_symbol = "✓" if self.was_evaluated_successfully else "❌"
        click.echo(f"{success_symbol} Item {self.item_id} - Trace ID: {self.trace_id}")
        if self.was_evaluated_successfully:
            pretty_print_scores(self.scores)


@dataclass
class PositiveItemsSummary:
    items_count: int
    total_bugs_expected: int
    total_bugs_found: int
    total_suggested_bugs: int
    avg_content_match: float
    avg_location_match: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class NegativeItemsSummary:
    items_count: int
    total_suggested_bugs: int
    false_positives_distribution: dict[int, int]  # Maps number of false positives to count of items


@dataclass
class RunSummaryInfo:
    items_count: int
    items_evaluated_successfully: int
    item_details: list[RelevantItemInfo]
    positive_items_summary: PositiveItemsSummary
    negative_items_summary: NegativeItemsSummary
    suggestion_count_distribution: dict[int, int]  # Maps number of suggestions to count of items


def get_relevant_info_for_item(langfuse: Langfuse, item: DatasetRunItem) -> RelevantItemInfo:
    trace = langfuse.fetch_trace(item.trace_id)
    scores: list[dict] = [
        {"name": score.name, "value": score.value, "comment": score.comment}
        for score in trace.data.scores
    ]
    error_score = next(
        (score for score in scores if score["name"] == "error_running_evaluation"), None
    )
    was_evaluated_successfully = error_score is not None and error_score["value"] == 0
    return RelevantItemInfo(
        item_id=item.id,
        trace_id=item.trace_id,
        was_evaluated_successfully=was_evaluated_successfully,
        scores=scores,
    )


@dataclass
class ItemDetailedScores:
    bugs_expected: int
    bugs_found: int
    content_match: float
    location_match: float
    noise: float
    suggestions_count: int


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
    # All suggestions are the matched (bugs_found) + unmatched (noise) suggestions
    suggestions_count = noise_score["value"] if noise_score else 0 + bugs_found_score["value"]
    return ItemDetailedScores(
        bugs_expected=bugs_expected,
        bugs_found=bugs_found_score["value"],
        content_match=content_match_score["value"] if content_match_score else 0,
        location_match=location_match_score["value"] if location_match_score else 0,
        noise=noise_score["value"] if noise_score else 0,
        suggestions_count=suggestions_count,
    )


def calculate_run_summary(langfuse: Langfuse, run: DatasetRunWithItems) -> RunSummaryInfo:
    items_in_run = [get_relevant_info_for_item(langfuse, item) for item in run.dataset_run_items]

    # Calculate averages for successful evaluations only
    successful_items = [item for item in items_in_run if item.was_evaluated_successfully]
    positive_items_summary: dict = {
        "items_count": 0,
        "total_bugs_found": 0,
        "total_bugs_expected": 0,
        "total_suggested_bugs": 0,
        "total_content_match": 0.0,
        "total_location_match": 0.0,
    }
    negative_items_summary: dict = {
        "items_count": 0,
        "total_suggested_bugs": 0,
        "false_positives_distribution": {},
    }
    suggestion_count_distribution: dict[int, int] = {}

    for item in successful_items:
        item_detailed_scores = calculate_item_detailed_scores(item)
        if item_detailed_scores is None:
            click.echo(f"! Bugs found score not found for item {item.item_id}. Ignoring item.")
            continue

        # Update overall suggestion count distribution
        suggestion_count = item_detailed_scores.suggestions_count
        current_count = suggestion_count_distribution.get(suggestion_count, 0)
        suggestion_count_distribution[suggestion_count] = current_count + 1

        if item_detailed_scores.bugs_expected == 0:
            # Negative item (no expected bugs)
            negative_items_summary["items_count"] += 1
            negative_items_summary["total_suggested_bugs"] += suggestion_count

            # Update false positives distribution
            current_count = negative_items_summary["false_positives_distribution"].get(
                suggestion_count, 0
            )
            negative_items_summary["false_positives_distribution"][suggestion_count] = (
                current_count + 1
            )
        else:
            # Positive item (has expected bugs)
            positive_items_summary["items_count"] += 1
            positive_items_summary["total_bugs_found"] += item_detailed_scores.bugs_found
            positive_items_summary["total_bugs_expected"] += item_detailed_scores.bugs_expected
            positive_items_summary["total_suggested_bugs"] += suggestion_count
            positive_items_summary["total_content_match"] += item_detailed_scores.content_match
            positive_items_summary["total_location_match"] += item_detailed_scores.location_match

    # Calculate precision, recall, and F1 score
    precision = (
        positive_items_summary["total_bugs_found"] / positive_items_summary["total_suggested_bugs"]
        if positive_items_summary["total_suggested_bugs"] > 0
        else 0.0
    )
    recall = (
        positive_items_summary["total_bugs_found"] / positive_items_summary["total_bugs_expected"]
        if positive_items_summary["total_bugs_expected"] > 0
        else 0.0
    )
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate averages for content and location match
    avg_content_match = (
        positive_items_summary["total_content_match"] / positive_items_summary["items_count"]
        if positive_items_summary["items_count"] > 0
        else 0.0
    )
    avg_location_match = (
        positive_items_summary["total_location_match"] / positive_items_summary["items_count"]
        if positive_items_summary["items_count"] > 0
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
            total_suggested_bugs=positive_items_summary["total_suggested_bugs"],
            avg_content_match=avg_content_match,
            avg_location_match=avg_location_match,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        ),
        negative_items_summary=NegativeItemsSummary(
            items_count=negative_items_summary["items_count"],
            total_suggested_bugs=negative_items_summary["total_suggested_bugs"],
            false_positives_distribution=negative_items_summary["false_positives_distribution"],
        ),
        suggestion_count_distribution=suggestion_count_distribution,
    )


if __name__ == "__main__":
    main()
