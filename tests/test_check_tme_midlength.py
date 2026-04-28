import pytest

from tools.debug.check_tme_midlength import summarize_records


def test_midlength_summary_passes_when_all_logged_tme_grads_are_small():
    records = [
        {
            "step": 10,
            "grad_norm_tme": 0.5,
            "grad_health_tme": {
                "max_abs": 0.2,
                "nonfinite_tensors": 0,
                "nonfinite_values": 0,
            },
        },
        {
            "step": 500,
            "grad_norm_tme": 9.5,
            "grad_health_tme": {
                "max_abs": 1.0,
                "nonfinite_tensors": 0,
                "nonfinite_values": 0,
            },
        },
    ]

    summary = summarize_records(records, threshold=10.0, min_step=500)

    assert summary["passed"] is True
    assert summary["max_step"] == 500
    assert summary["max_grad_norm_tme"] == pytest.approx(9.5)


def test_midlength_summary_fails_on_late_grad_growth():
    records = [
        {
            "step": 500,
            "grad_norm_tme": 10.0,
            "grad_health_tme": {
                "max_abs": 0.2,
                "nonfinite_tensors": 0,
                "nonfinite_values": 0,
            },
        },
    ]

    summary = summarize_records(records, threshold=10.0, min_step=500)

    assert summary["passed"] is False
    assert summary["failure_count"] == 1


def test_midlength_summary_fails_before_required_step():
    records = [
        {
            "step": 490,
            "grad_norm_tme": 0.1,
            "grad_health_tme": {
                "max_abs": 0.1,
                "nonfinite_tensors": 0,
                "nonfinite_values": 0,
            },
        },
    ]

    summary = summarize_records(records, threshold=10.0, min_step=500)

    assert summary["passed"] is False
    assert summary["max_step"] == 490
