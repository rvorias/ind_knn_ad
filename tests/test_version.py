from click.testing import CliRunner

import indad
from indad.cli import cli_interface


def test_runtime_version():
    assert indad.__version__ == "0.4.0"


def test_cli_version():
    result = CliRunner().invoke(cli_interface, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "indad, version 0.4.0"
