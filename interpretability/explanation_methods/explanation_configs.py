"""
Configurations for the explanations methods defined in explainers
"""

explainer_configs = {
    "Ours": {
        "default": {}
    },
    "RISE": {
        "default": {"n": 500, "s": 8, "p": .1, "batch_size": 8}
    },
    "GCam": {
        "default": {}
    },
    "Occlusion": {
        "default": {},
        "Occ5": {"ks": 5,
                 "stride":  2},
        "Occ9": {
            "ks": 9,
            "stride": 2
        },
        "Occ9-TI": {
            "ks": 9,
            "stride":  4,
            "batch_size": 1
        },
        "Occ13-TI": {
            "ks": 13,
            "stride": 4,
            "batch_size": 1
        },
    },
    "LIME": {
        "default": {"kernel_size": 4, "num_samples": 500, "num_features": 5, "batch_size": 8},
    },
    "IxG": {
        "default": {}
    },
    "GB": {
        "default": {}
    },
    "Grad": {
        "default": {}
    },
    "IntGrad": {
        "default": {"n_steps": 50, "internal_batch_size": 8}
    },
    "DeepLIFT": {
        "default": {}
    }
}
