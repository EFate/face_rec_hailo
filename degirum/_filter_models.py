#
# _filter_models.py - DeGirum Python SDK: common models filtering
# Copyright DeGirum Corp. 2022
#
# Contains DeGirum common models filtering implementation
#

import re
from typing import Optional, List, Union, Dict
from .exceptions import DegirumException
from .aiclient import ModelParams


def check_runtime_device_supported(
    runtime: Optional[str],
    device: Optional[str],
    mparams: ModelParams,
    system_rt_dev_pairs: Optional[List[List[str]]] = None,
) -> bool:
    """Check if model supports given runtime agent type and device type combination.

    Args:
        runtime: runtime agent type or "*"/None to ignore

        device: device type or "*"/None to ignore

        mparams: model info

        system_rt_dev_pairs: optional list of available system devices in format ["RUNTIME","DEVICE"]
        to additionally verify against

    Returns:
        True if model supports given runtime agent type and device type combination, False otherwise
    """

    if runtime == "*":
        runtime = None
    if device == "*":
        device = None

    supported_list = [[mparams.RuntimeAgent, mparams.DeviceType]]
    supported_types = mparams.SupportedDeviceTypes
    if supported_types:
        # convert string to list of [runtime, device] pairs
        supported_list.extend(t.split("/") for t in re.split("[,; ]+", supported_types))

    # check if given runtime/device combination is supported
    match_list = []
    for rt_dev_pair in supported_list:
        if len(rt_dev_pair) != 2:
            raise DegirumException(
                f"Invalid format of supported device list '{supported_types}' for model '{mparams.ModelPath}'"
            )

        runtime_match = (
            runtime is None or rt_dev_pair[0] == "*" or runtime == rt_dev_pair[0]
        )
        device_match = (
            device is None or rt_dev_pair[1] == "*" or device == rt_dev_pair[1]
        )
        if runtime_match and device_match:
            match_list.append(
                [
                    runtime if runtime is not None else rt_dev_pair[0],
                    device if device is not None else rt_dev_pair[1],
                ]
            )

    if not system_rt_dev_pairs:
        return True if match_list else False

    # additionally check against system supported devices
    for sys_pair in system_rt_dev_pairs:
        for match in match_list:
            runtime_match = match[0] == "*" or sys_pair[0] == match[0]
            device_match = match[1] == "*" or sys_pair[1] == match[1]
            if runtime_match and device_match:
                return True

    return False


def _filter_models(
    models: Dict[str, ModelParams],
    *,
    system_supported_device_types: List[str],
    model_family=None,
    runtime: Union[str, List[str], None] = None,
    device: Union[str, List[str], None] = None,
    device_type: Union[str, List[str], None] = None,
    precision: Union[str, List[str], None] = None,
    pruned: Union[str, List[str], None] = None,
    postprocess_type: Union[str, List[str], None] = None,
) -> Dict[str, ModelParams]:
    """List all available model matching to specified filter values.

    Args:
        models: model dictionary containing model parameters
        system_supported_device_types: list of available system devices in format "RUNTIME/DEVICE"
        model_family: model family - a string to match any part of the model name; for example, "yolo", "mobilenet1"
        runtime: type of inference runtime(s): a string or a string list of runtime labels
        device: target inference device(s): a string or a string list of device labels
        device_type: target inference device(s):string or list of strings of full device type names in "RUNTIME/DEVICE" format
        precision: model calculation precision: a string or a string list of precision labels; possible labels: "quant", "float"
        pruned: model density: a string or a string list of density labels; possible labels: "dense", "pruned"
        postprocess_type: model output postprocess type: a string or a string list of postprocess type labels

    Returns:
        part of `models` dictionary, containing only models matching to specified filter values.
    """

    def _to_list(value) -> List[str]:
        """Convert value to the list containing this value if it is not list already.
        Returns empty list if value is None
        """
        if not value:
            return []
        elif not isinstance(value, list):
            return [value]
        else:
            return value

    model_family = (
        [model_family]
        if isinstance(model_family, re.Pattern)
        else _to_list(model_family)
    )
    runtime = _to_list(runtime)
    device = _to_list(device)
    device_type = _to_list(device_type)
    if not device_type:
        device_type = ["*/*"]
    precision = _to_list(precision)
    pruned = _to_list(pruned)
    postprocess_type = _to_list(postprocess_type)

    system_rt_dev_pairs = [d.split("/") for d in system_supported_device_types]

    for value, description, supported_values, check in [
        (
            model_family,
            "model_family",
            ["yolo", "mobilenet", "..."],
            lambda f, lst: not isinstance(f, str) and not isinstance(f, re.Pattern),
        ),
        (
            runtime,
            "runtime",
            ["N2X", "TFLITE", "..."],
            lambda f, lst: not isinstance(f, str),
        ),
        (
            device,
            "device",
            ["ORCA1", "CPU", "..."],
            lambda f, lst: not isinstance(f, str),
        ),
        (
            device_type,
            "device_type",
            ["N2X/ORCA1", "TFLITE/CPU", "..."],
            lambda f, lst: not isinstance(f, str) or len(f.split("/")) != 2,
        ),
        (
            precision,
            "precision",
            ["QUANT", "FLOAT"],
            lambda f, lst: not isinstance(f, str) or f.upper() not in lst,
        ),
        (
            pruned,
            "pruned",
            ["PRUNED", "DENSE"],
            lambda f, lst: not isinstance(f, str) or f.upper() not in lst,
        ),
        (
            postprocess_type,
            "postprocess_type",
            ["Detection", "Classification", "..."],
            lambda f, lst: not isinstance(f, str),
        ),
    ]:
        for f in value:
            if check(f, supported_values):
                raise DegirumException(
                    f"Filter '{description}' has unsupported value '{f}'. Possible values are: {supported_values}"
                )

    def re_filter(model, mparams, lst):
        return lst[0].match(model)

    def str_filter(model, mparams, lst):
        return any(n in model for n in lst)

    def density_filter(model, mparams):
        p = model.split("--")[-1].split("_")[::-1]
        return "PRUNED" if len(p) >= 5 and p[4].upper() == "PRUNED" else "DENSE"

    def quant_filter(model, mparams):
        p = model.split("--")[-1].split("_")[::-1]
        return "QUANT" if len(p) >= 4 and p[3].upper() == "QUANT" else "FLOAT"

    def is_runtime_supported(model, mparams, runtime_list):
        return any(
            check_runtime_device_supported(rt, None, mparams, system_rt_dev_pairs)
            for rt in runtime_list
        )

    def is_device_supported(model, mparams, device_list):
        return any(
            check_runtime_device_supported(None, dev, mparams, system_rt_dev_pairs)
            for dev in device_list
        )

    def is_device_type_supported(model, mparams, device_type_list):
        return any(
            check_runtime_device_supported(rt, dev, mparams, system_rt_dev_pairs)
            for rt, dev in [dt.split("/") for dt in device_type_list]
        )

    def postprocess_type_filter(model, mparams, postprocess_type_list):
        return any(
            postprocess_type == mparams.OutputPostprocessType
            for postprocess_type in postprocess_type_list
        )

    filters_configuration = [
        (
            model_family,
            (
                re_filter
                if model_family and isinstance(model_family[0], re.Pattern)
                else str_filter
            ),
        ),
        (runtime, is_runtime_supported),
        (device, is_device_supported),
        (device_type, is_device_type_supported),
        (
            [s.upper() for s in precision],
            lambda model, mparams, lst: quant_filter(model, mparams) in lst,
        ),
        (
            [s.upper() for s in pruned],
            lambda model, mparams, lst: density_filter(model, mparams) in lst,
        ),
        (postprocess_type, postprocess_type_filter),
    ]

    for filter_to_check, check_function in filters_configuration:
        if filter_to_check:
            models = {
                model: mparams
                for model, mparams in models.items()
                if check_function(model, mparams, filter_to_check)
            }

    return models
