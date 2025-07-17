#
# DeGirum Python SDK: inference results postprocessor package
# Copyright DeGirum Corp. 2025
#
# Implements postprocessor classes to handle different types of specific inference results data.
#


import pathlib
import importlib
from typing import Optional
from ..exceptions import DegirumException
from ..log import log_wrap
from ._InferenceResults import InferenceResults


_supported_types_map: dict = {}  # map of postprocessor types to postprocessor classes

# Discover and load all postprocessor modules:
# iterate over matching Python files in the current package directory
for py_file in pathlib.Path(__file__).parent.glob("_*Results.py"):
    module_name = py_file.stem  # e.g., "_FooResult"
    class_name = module_name.lstrip("_")  # e.g., "FooResult"
    mod = importlib.import_module(f".{module_name}", package=__package__)

    # check if postprocessor class exists in the module
    pp_class = getattr(mod, class_name, None)
    if pp_class is None:
        raise ImportError(
            f"Expected class '{class_name}' not found in module '{module_name}'"
        )

    # add postprocessor class to package globals
    globals()[class_name] = pp_class

    # fill the supported types map
    supported_types = pp_class.supported_types()
    if isinstance(supported_types, str):
        supported_types = [supported_types]
    elif supported_types is None:
        supported_types = []
    for supported_type in supported_types:
        if supported_type in _supported_types_map:
            raise DegirumException(
                f"Postprocessor type '{supported_type}' is already registered by '{_supported_types_map[supported_type].__name__}'"
            )
        _supported_types_map[supported_type] = pp_class


def class_from_type_string(postprocess_type):
    """Get postprocessor class by type string.

    Args:
        postprocess_type (str): postprocessor type.

    Returns:
        postprocessor class
    """

    postprocessor = _supported_types_map.get(postprocess_type, None)
    if postprocessor is None:
        raise DegirumException(f"Postprocessor type '{postprocess_type}' is not known")
    return postprocessor


def _create_overlay_color_dataset(
    postprocess_type: str, num_classes: int, label_dict: dict
):
    """Create and return default color data based on postprocessor type.

    Args:
        postprocess_type (str): postprocessor type.
        num_classes (int): number of class categories.
        label_dict (dict[str, str]): Model labels dictionary.

    Returns:
        result (list[tuple] | tuple):
            overlay color data
    """
    return class_from_type_string(postprocess_type).generate_overlay_color(
        num_classes, label_dict
    )


@log_wrap
def create_postprocessor(postprocess_type: str, *args, **kwargs) -> InferenceResults:
    """Create and return postprocessor object.

    Args:
        postprocess_type (str): postprocessor type.
        args: positional arguments. For the list of arguments see documentation for constructor of [degirum.postprocessor.InferenceResults][] class.
        kwargs: keyword arguments.

    Returns:
        InferenceResults instance corresponding to model results type.
    """
    return class_from_type_string(postprocess_type)(*args, **kwargs)


@log_wrap
def register_postprocessor(
    postprocess_type: str, postprocessor_class: Optional[type], overwrite: bool = False
) -> None:
    """Register a custom postprocessor class for a specific postprocess type.

    Args:
        postprocess_type (str): The postprocess type to register.
        postprocessor_class (type): The custom postprocessor class to register or None to unregister.
        overwrite (bool): If True, overwrite existing registration. Default is False.

    Raises:
        DegirumException: If the postprocess type is already registered and overwrite is False.
    """

    if postprocessor_class is None:
        if postprocess_type in _supported_types_map:
            del _supported_types_map[postprocess_type]
    else:
        if not overwrite and postprocess_type in _supported_types_map:
            raise DegirumException(
                f"Postprocessor type '{postprocess_type}' is already registered by '{_supported_types_map[postprocess_type].__name__}'"
            )
        _supported_types_map[postprocess_type] = postprocessor_class
