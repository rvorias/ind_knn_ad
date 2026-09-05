"""Non-interactive dataset creation and inspection commands."""

import json
from pathlib import Path

import click

from indad.workspace import create_dataset, import_images, inspect_dataset


def _run(action, *args, **kwargs):
    try:
        return action(*args, **kwargs)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group()
def cli():
    """Create datasets, import images, and report readiness as JSON."""


@cli.command("create")
@click.argument("name")
@click.option(
    "--root", type=click.Path(path_type=Path), default="datasets", show_default=True
)
def create(name, root):
    """Create a new named dataset without overwriting data."""
    path = _run(create_dataset, root, name)
    click.echo(json.dumps({"dataset": str(path)}))


@cli.command("import")
@click.argument(
    "dataset", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "images",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "ground_truth"]),
    default="train",
    show_default=True,
)
@click.option("--label", default="good", show_default=True)
def import_command(dataset, images, split, label):
    """Copy explicit image files into a split and label folder."""
    files = _run(lambda: [(path.name, path.read_bytes()) for path in images])
    imported = _run(import_images, dataset, files, split, label)
    click.echo(json.dumps({"imported": imported}))


@cli.command("inspect")
@click.argument(
    "dataset", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--require",
    type=click.Choice(
        ["training", "inspection", "image_evaluation", "pixel_evaluation"]
    ),
    help="Exit 1 if the requested workflow is not ready; still emit the report.",
)
def inspect(dataset, require):
    """Print a deterministic manifest and health report without changing data."""
    report = _run(inspect_dataset, dataset)
    click.echo(json.dumps(report, indent=2))
    if require and not report["readiness"][require]:
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    cli()
