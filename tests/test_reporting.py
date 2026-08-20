from pathlib import Path

from arc_deformation.reporting import generate_report


def test_reference_results_generate_complete_submission_assets(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    generate_report(repository / "results" / "reference", tmp_path)

    expected = {
        "adjusted_association_table.tex",
        "association_adjustment.pdf",
        "association_forest.pdf",
        "association_table.tex",
        "hodge_sensitivity.pdf",
        "hodge_sensitivity_table.tex",
        "left_only_comparison_table.tex",
        "model_comparison.pdf",
        "model_table_all.tex",
        "paired_comparison_table.tex",
        "results.tex",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})

    paired_table = (tmp_path / "paired_comparison_table.tex").read_text(encoding="utf-8")
    assert "Comparison model" in paired_table
    assert "Lesion + log-velocity Hodge" in paired_table
    assert "-0.06 (-0.42, 0.29)" in paired_table
