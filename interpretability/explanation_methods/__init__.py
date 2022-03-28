from interpretability.explanation_methods.explainers.rise import RISE
from interpretability.explanation_methods.explainers.lime import Lime
from interpretability.explanation_methods.explainers.occlusion import Occlusion
from interpretability.explanation_methods.explainers.captum import GradCam, GB, IxG, Grad, DeepLIFT, IntGrad
from interpretability.explanation_methods.explanation_configs import explainer_configs

explainer_map = {
    "Ours": lambda x: x,
    "RISE": RISE,
    "Occlusion": Occlusion,
    "GCam": GradCam,
    "LIME": Lime,
    "IntGrad": IntGrad,
    "GB": GB,
    "IxG": IxG,
    "Grad": Grad,
    "DeepLIFT": DeepLIFT

}


def get_explainer(trainer, explainer_name, config_name):
    return explainer_map[explainer_name](trainer, **explainer_configs[explainer_name][config_name])
