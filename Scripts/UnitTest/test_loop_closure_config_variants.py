from pathlib import Path

from Utility.Config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "Config" / "Experiment" / "MACVO"


def _assert_loop_variant_preserves_vo(
    base_name: str,
    loop_name: str,
    residual_mode: str,
) -> None:
    _, base = load_config(CONFIG_ROOT / base_name)
    _, loop = load_config(CONFIG_ROOT / loop_name)

    loop_odometry = {
        key: value
        for key, value in loop["Odometry"].items()
        if key not in {"loop_closure", "global_pgo"}
    }
    assert loop["Common"] == base["Common"]
    assert loop_odometry == base["Odometry"]
    assert loop["Data"] == base["Data"]
    assert loop["Preprocess"] == base["Preprocess"]

    assert loop["Odometry"]["optimizer"]["args"]["graph_type"] == residual_mode
    assert (
        loop["Odometry"]["loop_closure"]["vins_geometry"]
        ["network_refinement"]["residual_mode"]
        == residual_mode
    )
    assert (
        loop["Odometry"]["global_pgo"]["observation_residual_mode"]
        == residual_mode
    )


def test_performant_disp_loop_variant_preserves_original_vo() -> None:
    _assert_loop_variant_preserves_vo(
        "MACVO_Performant.yaml",
        "MACVO_Performant_Loop_Disp.yaml",
        "disp",
    )


def test_paper_icp_loop_variant_preserves_original_vo() -> None:
    _assert_loop_variant_preserves_vo(
        "Paper_Reproduce.yaml",
        "MACVO_PaperReproduce_Loop_ICP.yaml",
        "icp",
    )
