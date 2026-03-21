from types import SimpleNamespace

from train_scripts.exp_config_utils import (
    resolve_exp_active_channels,
    resolve_exp_dataset_kwargs,
)


def test_resolve_exp_dataset_kwargs_reads_data_section():
    config = SimpleNamespace(
        data=SimpleNamespace(
            active_channels=[
                "cell_masks",
                "cell_type_healthy",
                "cell_type_cancer",
                "cell_type_immune",
                "cell_state_prolif",
                "cell_state_nonprolif",
                "cell_state_dead",
                "vasculature",
                "oxygen",
                "glucose",
            ],
            exp_channels_dir="custom_exp_channels",
            features_dir="custom_features",
            vae_features_dir="custom_vae_features",
            exp_index_h5="custom_metadata/exp_index.hdf5",
            vae_prefix="custom_vae",
            ssl_prefix="custom_ssl",
        )
    )

    dataset_kwargs = resolve_exp_dataset_kwargs(config)

    assert dataset_kwargs["active_channels"][0] == "cell_masks"
    assert dataset_kwargs["exp_channels_dir"] == "custom_exp_channels"
    assert dataset_kwargs["features_dir"] == "custom_features"
    assert dataset_kwargs["vae_features_dir"] == "custom_vae_features"
    assert dataset_kwargs["exp_index_h5"] == "custom_metadata/exp_index.hdf5"
    assert dataset_kwargs["vae_prefix"] == "custom_vae"
    assert dataset_kwargs["ssl_prefix"] == "custom_ssl"


def test_top_level_active_channels_override_data_section():
    config = SimpleNamespace(
        active_channels=[
            "cell_mask",
            "cell_type_healthy",
            "cell_type_cancer",
            "cell_type_immune",
            "cell_state_prolif",
            "cell_state_nonprolif",
            "cell_state_dead",
            "vasculature",
            "oxygen",
            "glucose",
        ],
        data=SimpleNamespace(
            active_channels=[
                "cell_masks",
                "cell_type_healthy",
                "cell_type_cancer",
                "cell_type_immune",
                "cell_state_prolif",
                "cell_state_nonprolif",
                "cell_state_dead",
                "vasculature",
                "oxygen",
                "glucose",
            ],
        ),
    )

    active_channels = resolve_exp_active_channels(config)

    assert active_channels[0] == "cell_masks"
    assert active_channels.count("cell_masks") == 1
