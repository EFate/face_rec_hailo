#
# DeGirum AI Inference Software Package
# Copyright DeGirum Corp. 2022
#

from ._version import pysdk_version

__version__ = pysdk_version

import argparse
import logging
from typing import List, Dict, Union

from .zoo_manager import ZooManager
from .aiclient import ModelParams

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __dir__():
    return [
        "connect",
        "load_model",
        "list_models",
        "get_supported_devices",
        "enable_default_logger",
        "aiclient",
        "CoreClient",
        "LOCAL",
        "CLOUD",
    ]


def connect(
    inference_host_address: str,
    zoo_url: str = "",
    token: str = "",
) -> ZooManager:
    """Connect to the AI inference host and model zoo of your choice.

    This is the main PySDK entry point: you start your work with PySDK by calling this function.

    The following use cases are supported:

    1. You want to perform **cloud inferences** and take models from some **cloud model zoo**.
    2. You want to perform inferences on some **AI server** and take models from some **cloud model zoo**.
    3. You want to perform inferences on some **AI server** and take models from its **local model zoo**.
    4. You want to perform inferences on **local AI hardware** and take models from some **cloud model zoo**.
    5. You want to perform inferences on **local AI hardware** and take models from the **local model zoo** directory on your local drive.
    6. You want to perform inferences on **local AI hardware** and use **particular model** from your local drive.

    Args:
        inference_host_address: Inference engine designator; it defines which inference engine to use.

            - For AI Server-based inference it can be either the hostname or IP address of the AI Server host,
            optionally followed by the port number in the form `:port`.

            - For DeGirum Cloud Platform-based inference it is the string `"@cloud"` or [degirum.CLOUD][] constant.

            - For local inference it is the string `"@local"` or [degirum.LOCAL][] constant.

        zoo_url: Model zoo URL string which defines the model zoo to operate with.

            - For a cloud model zoo, it is specified in the following format: `<cloud server prefix>[/<zoo suffix>]`.
            The `<cloud server prefix>` part is the cloud platform root URL, typically `https://hub.degirum.com`.
            The optional `<zoo suffix>` part is the cloud zoo URL suffix in the form `<organization>/<model zoo name>`.
            You can confirm zoo URL suffix by visiting your cloud user account and opening the model zoo management page.
            If `<zoo suffix>` is not specified, then DeGirum public model zoo `degirum/public` is used.

            - For AI Server-based inferences, you may omit both `zoo_url` and `token` parameters.
            In this case locally-deployed model zoo of the AI Server will be used.

            - For local AI hardware inferences you specify `zoo_url` parameter as either a path to a local
            model zoo directory, or a path to model's .json configuration file.
            The `token` parameter is not needed in this case.

        token: Cloud API access token used to access the cloud zoo.

            - To obtain this token you need to open a user account on [DeGirum cloud platform](https://hub.degirum.com).
            Please login to your account and go to the token generation page to generate an API access token.

    Returns:
        An instance of Model Zoo manager object configured to work with AI inference host and model zoo of your choice.

    Once you created Model Zoo manager object, you may use the following methods:

    - [degirum.zoo_manager.ZooManager.list_models][] to list and search models available in the model zoo.
    - [degirum.zoo_manager.ZooManager.load_model][] to create [degirum.model.Model][] model handling object
    to be used for AI inferences.
    - [degirum.zoo_manager.ZooManager.model_info][] to request model parameters.

    """

    return ZooManager(inference_host_address, zoo_url, token)


CLOUD: str = ZooManager._CLOUD
""" Cloud inference designator:
use it as a first argument of [degirum.connect][] function to specify cloud-based inference """

LOCAL: str = ZooManager._LOCAL
""" Local inference designator:
use it as a first argument of [degirum.connect][] function to specify inference on local AI hardware """


def enable_default_logger(
    level: int = logging.DEBUG,
) -> logging.StreamHandler:
    """
    Helper function for adding a StreamHandler to the package logger. Removes any existing handlers.
    Useful for debugging.

    Args:
        level: Logging level as defined in logging python package. defaults to logging.DEBUG.

    Returns:
        Returns an instance of added StreamHandler.
    """
    logger = logging.getLogger(__name__)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s][%(threadName)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


def _command_entrypoint(arg_str=None):
    from .server import _server_args, _download_zoo_args
    from ._zoo_accessor import _system_info_args, _trace_args
    # from ._runtime_installer import _get_platform, _install_runtime_args

    parser = argparse.ArgumentParser(description="DeGirum toolkit")

    subparsers = parser.add_subparsers(
        help="use -h flag to see help on subcommands", required=True
    )

    # version subcommand
    subparser = subparsers.add_parser(
        "version",
        description="Get DeGirum package version",
        help="get DeGirum package version",
    )

    def _get_version(args):
        print(__version__)

    subparser.set_defaults(func=_get_version)

    # server subcommand
    subparser = subparsers.add_parser(
        "server",
        description="Manage DeGirum AI server on local host",
        help="manage DeGirum AI server on local host",
    )
    _server_args(subparser)

    # download-zoo subcommand
    subparser = subparsers.add_parser(
        "download-zoo",
        description="Download models from DeGirum cloud model zoo",
        help="download models from DeGirum cloud model zoo",
    )
    _download_zoo_args(subparser)

    # sys-info subcommand
    subparser = subparsers.add_parser(
        "sys-info",
        description="Print system information",
        help="print system information",
    )
    _system_info_args(subparser)

    # trace subcommand
    subparser = subparsers.add_parser(
        "trace",
        description="Manage tracing",
        help="manage tracing",
    )
    _trace_args(subparser)

    # install-runtime subcommand (only on Debian and Windows)
    # current_platform = _get_platform()
    # if current_platform == "Debian":  # or current_platform == "Windows":
    #     subparser = subparsers.add_parser(
    #         "install-runtime",
    #         description="Install runtimes and drivers or list available plugins",
    #         help="Install plugins or list available ones",
    #         usage="%(prog)s list | plugin_name [plugin_args]",
    #     )
    #     _install_runtime_args(subparser)

    # parse args
    args = parser.parse_args(arg_str.split() if arg_str else None)

    # execute subcommand
    args.func(args)


def load_model(
    model_name: str,
    inference_host_address: str,
    zoo_url: str = "",
    token: str = "",
    **kwargs
):
    """
    Load a model from the model zoo for the inference.

    Args:
        model_name: Model name to load from the model zoo.

        inference_host_address: Inference engine designator; it defines which inference engine to use.

        zoo_url: Model zoo URL string which defines the model zoo to operate with.

        token: Cloud API access token used to access the cloud zoo.

        **kwargs (any):
            you may pass arbitrary model properties to be assigned to the model object in a form
            of property=value

    !!! note

        For detailed description of `zoo_url`, `inference_host_address`, and `token` parameters refer to [degirum.connect][] function.

    Returns (degirum.model.Model):
        An instance of [degirum.model.Model][] model handling object to be used for AI inferences.
    """
    zoo = ZooManager(inference_host_address, zoo_url, token)
    return zoo.load_model(model_name, **kwargs)


def list_models(
    inference_host_address: str, zoo_url: str, token: str = "", **kwargs
) -> Union[List[str], Dict[str, ModelParams]]:
    """
    List models in the model zoo matching to specified filter values.

    Args:
        inference_host_address: Inference engine designator; it defines which inference engine to use.

        zoo_url: Model zoo URL string which defines the model zoo to operate with.

        token: Cloud API access token used to access the cloud zoo.

        **kwargs (any): filter parameters to narrow down the list of models.

    !!! note

        For detailed description of `zoo_url`, `inference_host_address`, and `token` parameters refer to [degirum.connect][] function.
        For detailed description of `kwargs` parameters refer to [degirum.ZooManager.list_models][] method.

    Returns:
        A dictionary with model names as keys and model info as values.

    """

    zoo = ZooManager(inference_host_address, zoo_url, token)
    return zoo.list_models(**kwargs, names_only=False)


def get_supported_devices(
    inference_host_address: str,
    zoo_url: str = "",
    token: str = "",
) -> List[str]:
    """Get runtime/device type names, which are available in the inference engine.

    Args:
        inference_host_address: Inference engine designator; it defines which inference engine to use.

        zoo_url: Optional model zoo URL string which defines the model zoo to operate with.
            Makes sense only for cloud inference engines to specify another base URL.

        token: Cloud API access token used to access the cloud zoo.

    !!! note

        For detailed description of `inference_host_address` and `token` parameters refer to [degirum.connect][] function.

    Returns:
        list of runtime/device type names; each element is a string in a format "RUNTIME/DEVICE"
    """

    zoo = ZooManager(inference_host_address, zoo_url, token)
    return zoo.supported_device_types()
