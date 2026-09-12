from __future__ import annotations

import pytest

from usctbench.cli import main


def test_cli_help_and_algorithm_list(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0

    assert main(["list-algorithms"]) == 0

    output = capsys.readouterr().out
    assert "USCT benchmark command-line interface" in output
    assert "straight_cgls" in output
    assert "fwi_wust" in output
    assert "fwi_kwave_adapter" not in output
    assert "diffusion_fwi_kwave_adapter" not in output
