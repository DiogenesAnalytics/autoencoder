"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Autoencoder."""


if __name__ == "__main__":
    main(prog_name="autoencoder")  # pragma: no cover
