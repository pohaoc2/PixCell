from types import SimpleNamespace

from train_scripts.exp_config_utils import (
    DEFAULT_ACTIVE_CHANNELS,
    _cfg_get,
    canonicalize_exp_channels,
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


def test_cfg_get_with_dict_mapping():
    d = {"key": "value", "other": 42}
    assert _cfg_get(d, "key") == "value"
    assert _cfg_get(d, "missing", "default") == "default"


def test_cfg_get_with_none_container():
    assert _cfg_get(None, "anything", "fallback") == "fallback"


def test_cfg_get_with_namespace_attr():
    ns = SimpleNamespace(foo="bar")
    assert _cfg_get(ns, "foo") == "bar"
    assert _cfg_get(ns, "missing", 99) == 99


def test_canonicalize_exp_channels_deduplication():
    channels = ["cell_masks", "cell_masks", "vasculature"]
    result = canonicalize_exp_channels(channels)
    assert result == ["cell_masks", "vasculature"]
    assert result.count("cell_masks") == 1


def test_canonicalize_exp_channels_resolves_alias():
    channels = ["cell_mask", "vasculature"]
    result = canonicalize_exp_channels(channels)
    assert result[0] == "cell_masks"


def test_resolve_exp_active_channels_with_none_config():
    result = resolve_exp_active_channels(None)
    assert result[0] == "cell_masks"
    assert result == DEFAULT_ACTIVE_CHANNELS
