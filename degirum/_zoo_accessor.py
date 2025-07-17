#
# _zoo_accessor.py - DeGirum Python SDK: zoo accessors
# Copyright DeGirum Corp. 2022
#
# Contains DeGirum zoo accessors implementation
#

import json
import copy
import io
from pathlib import Path
import zipfile
import logging
from abc import ABC, abstractmethod
from urllib.parse import urlparse, quote
import requests
from requests.adapters import HTTPAdapter, Retry
from enum import IntEnum
from typing import Optional, List, Union, Dict
from .exceptions import DegirumException
from .model import Model, _ClientModel, _ServerModel, _CloudServerModel
from ._filter_models import _filter_models, check_runtime_device_supported
from .aiclient import (
    ModelParams,
    get_modelzoo_list,
    system_info as server_system_info,
    trace_manage as server_trace_manage,
)
from .log import log_wrap
from ._version import pysdk_version

logger = logging.getLogger(__name__)


class _CommonZooAccessor(ABC):
    """Zoo Accessor abstract class"""

    def __init__(self, my_url: str):
        """Constructor

        Args:
            my_url: accessor-specific URL
        """
        self._url = my_url
        self._system_info_cached: Optional[dict] = None

    @property
    def url(self):
        return str(self._url)

    @log_wrap
    def list_models(
        self,
        model_family=None,
        *,
        runtime: Union[str, List[str], None] = None,
        device: Union[str, List[str], None] = None,
        device_type: Union[str, List[str], None] = None,
        precision: Union[str, List[str], None] = None,
        pruned: Union[str, List[str], None] = None,
        postprocess_type: Union[str, List[str], None] = None,
        names_only: bool = True,
    ) -> Union[List[str], Dict[str, ModelParams]]:
        """
        Get a list of names of AI models available in the connected model zoo which match specified
        filtering criteria.

        Keyword Args:
            model_family (str): Model family name filter.

                - When you pass a string, it will be used as search substring in the model name.
                For example, `"yolo"`, `"mobilenet"`.

                - You may also pass `re.Pattern` object. In this case it will do regular expression pattern search.

            runtime (str): Runtime agent type -- string or list of strings of runtime agent types.

            device (str): Target inference device -- string or list of strings of device names.

            device_type (str): Target inference device(s) -- string or list of strings of full device type names in "RUNTIME/DEVICE" format.

            precision (str): Model calculation precision - string or list of strings of model precision labels.

                Possible labels: `"quant"`, `"float"`.

            pruned (str): Model density -- string or list of strings of model density labels.

                Possible labels: `"dense"`, `"pruned"`.

            postprocess_type (str): Postprocess type -- string or list of strings of postprocess type labels.

            names_only (bool): True to request only model names, False to demand full model info.

        Returns:
            When names_only is True it is the list of model name strings matching specified filtering criteria,
            otherwise it is the dictionary of model names as keys and model info as values.

            Use a string from that list/dict as a parameter of [degirum.zoo_manager.ZooManager.load_model][] method.
        """

        model_list = _filter_models(
            self._get_model_list(False),
            system_supported_device_types=self._system_supported_device_types(),
            model_family=model_family,
            runtime=runtime,
            device=device,
            device_type=device_type,
            precision=precision,
            pruned=pruned,
            postprocess_type=postprocess_type,
        )
        if names_only:
            return sorted(model_list.keys())
        else:
            return model_list

    @log_wrap
    def model_info(self, model: str) -> ModelParams:
        """Request model parameters for given model name.

        Args:
            model: model name as returned by list_models()

        Returns:
            model parameter object
        """

        model_info = self._get_model_info(model)
        if model_info:
            return copy.deepcopy(model_info)
        else:
            raise DegirumException(
                f"Model '{model}' is not found in model zoo '{self.url}'"
            )

    @abstractmethod
    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model name.

        Args:
            model: model name as returned by list_models()

        Returns:
            model object corresponding to given model name
        """

    @log_wrap
    def system_info(self, update: bool = False) -> dict:
        """Return host system information dictionary

        Args:
            update: force update system information, otherwise take from cache

        Returns:
            host system information dictionary. Format:
                `{"Devices": {"<runtime>/<device>": {<device_info>}, ...}, ["Software Version": "<version>"]}`
        """
        if self._system_info_cached is None or update:
            self._system_info_cached = self._query_system_info()
        return self._system_info_cached

    def _system_supported_device_types(self) -> List[str]:
        """Get runtime/device type names, which are available in the inference system."""

        return self.system_info()["Devices"].keys()

    def _model_supported_device_types(self, model_params: ModelParams) -> List[str]:
        """Get runtime/device type names, which can be used by the given model:
        supported by the model itself and available in the inference system.

        Args:
            model_params: model parameters as returned by `Model.model_info()`.

        Returns:
            List of device names, which can be used for the model inference. Each element of the list is a string in a
            "RUNTIME/DEVICE" format, where RUNTIME is a runtime agent type and DEVICE is a device name.
        """

        sys_devs = [d.split("/") for d in self._system_supported_device_types()]
        ret = [
            f"{agent_device[0]}/{agent_device[1]}"
            for agent_device in sys_devs
            if check_runtime_device_supported(
                agent_device[0], agent_device[1], model_params
            )
        ]
        return ret

    #
    # methods to be implemented in derived classes
    #

    @abstractmethod
    def _query_system_info(self) -> dict:
        """Query host system information dictionary"""

    @abstractmethod
    def _get_model_list(self, names_only: bool) -> Dict[str, ModelParams]:
        """Get list of models available in the zoo

        Args:
            names_only: true to request only model names, false to demand full model info
        """

    @abstractmethod
    def _get_model_info(self, model: str) -> ModelParams:
        """Get model info reference for given model name or None if model is not found"""


class _FixedAssetsZooAccessor(_CommonZooAccessor):
    """Zoo accessor implementation with fixed model assets"""

    def __init__(self, my_url: str):
        """Constructor.

        Args:
            my_url: accessor-specific URL
        """

        self._assets: Dict[str, ModelParams] = {}
        super().__init__(my_url)
        self._rescan_zoo()

    def _get_model_list(self, names_only: bool) -> Dict[str, ModelParams]:
        """Get list of models available in the zoo

        Args:
            names_only: true to request only model names, false to demand full model info
        """
        return self._assets  # since we always have full assets, we can return them all

    def _get_model_info(self, model: str) -> ModelParams:
        """Get model info reference for given model name or None if model is not found"""
        return self._assets.get(model)

    @abstractmethod
    @log_wrap
    def _rescan_zoo(self):
        """Update list of assets. To be implemented in derived classes"""


class _LocalInferenceSingleFileZooAccessor(_FixedAssetsZooAccessor):
    """Local inference, single file zoo implementation"""

    def __init__(self, url):
        """Constructor.

        Args:
            url: path to the model JSON configuration file in the local filesystem.
        """
        super().__init__(url)

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model identifier.

        Args:
            model: model identifier

        Returns model object corresponding to model identifier.
        """

        model_params = self.model_info(model)

        # Check Supported Device Types for this model
        supported_device_types = self._model_supported_device_types(model_params)

        if not supported_device_types:
            raise DegirumException(
                f"Model '{model}' does not have any supported runtime/device combinations that will work on this system."
            )

        return _ClientModel(
            model,
            model_params,
            supported_device_types,
        )

    @log_wrap
    def _rescan_zoo(self):
        """Update list of assets"""

        self._assets = {Path(self.url).stem: _ClientModel.load_config_file(self.url)}

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""
        from .CoreClient import system_info as core_system_info

        return core_system_info()


class _LocalInferenceLocalDirZooAccessor(_FixedAssetsZooAccessor):
    """Local inference, local directory zoo implementation"""

    def __init__(self, url):
        """Constructor.

        Args:
            url: local zoo directory path
        """
        super().__init__(url)

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model identifier.

        Args:
            model: model identifier

        Returns model object corresponding to model identifier.
        """

        model_params = self.model_info(model)

        # Check Supported Device Types for this model
        supported_device_types = self._model_supported_device_types(model_params)
        if not supported_device_types:
            raise DegirumException(
                f"Model '{model}' does not have any supported runtime/device combinations that will work on this system."
            )

        return _ClientModel(
            model,
            model_params,
            supported_device_types,
        )

    @log_wrap
    def _rescan_zoo(self):
        """Update list of assets"""

        # recursively iterate over all JSON files in zoo directory
        json_files = sorted(Path(self.url).rglob("*.json"))
        for f in json_files:
            try:
                mparams = _ClientModel.load_config_file(str(f))

                # accept only valid model configuration files, which have checksum and config version
                if mparams.Checksum and mparams.ConfigVersion > 0:
                    self._assets[f.stem] = mparams
            except Exception:
                pass  # ignore invalid model configuration files

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""
        from .CoreClient import system_info as core_system_info

        return core_system_info()


class _AIServerLocalZooAccessor(_FixedAssetsZooAccessor):
    """AI server inference, local model zoo implementation"""

    def __init__(self, url):
        """Constructor.

        Args:
            url: AI server hostname or IP address
        """
        super().__init__(url)

    @log_wrap
    def _rescan_zoo(self):
        """Update cached list of models according to the current server model zoo contents"""
        self._assets = {a.name: a.extended_params for a in get_modelzoo_list(self.url)}

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model name.

        Args:
            model: model name as returned by list_models()

        Returns model object corresponding to given model name
        """
        model_params = self.model_info(model)
        return _ServerModel(
            self.url,
            model,
            model_params,
            self._model_supported_device_types(model_params),
        )

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""

        return server_system_info(self.url)


class _CloudZooAccessorBase(_CommonZooAccessor):
    """Cloud model zoo access: base implementation"""

    _default_cloud_server = "hub.degirum.com"
    """ DeGirum cloud server hostname """

    _default_cloud_zoo = "/degirum/public"
    """ DeGirum public zoo name. You can freely use all models available in this public model zoo """

    _default_cloud_url = "https://" + _default_cloud_server + _default_cloud_zoo
    """ Full DeGirum cloud public zoo URL. You can freely use all models available in this public model zoo """

    class _AssetsType(IntEnum):
        """Model assets cache type enumeration"""

        Partial = 0  # only some models are stored in the assets cache
        NamesOnly = 1  # only model names are stored in the assets cache
        Full = 2  # all models with their parameters are stored in the assets cache

    class _Assets:
        """Model assets class"""

        def __init__(
            self, model_params: Optional[ModelParams], model_label_dict: Optional[dict]
        ):
            """Constructor.

            Args:
                model_params: model parameters
                model_label_dict: model label dictionary
            """

            self.model_params = model_params
            self.model_label_dict = model_label_dict

    def __init__(self, url: str, token: str):
        """Constructor.

        Args:
            url: path to the cloud zoo in `"https://<cloud server URL>[/<zoo URL>]"` format
            token: cloud zoo access token
        """
        url_parsed = urlparse(url)
        url = f"{url_parsed.scheme}://{url_parsed.hostname}" + (
            f":{url_parsed.port}" if url_parsed.port else ""
        )
        self._zoo_url = (
            quote(url_parsed.path)
            if url_parsed.path
            else _CloudZooAccessorBase._default_cloud_zoo
        )
        self._token = token
        self._timeout = 5
        self._assets: Dict[str, _CloudZooAccessorBase._Assets] = {}
        self._assets_type = self._AssetsType.Partial
        super().__init__(url)

    @log_wrap
    def _cloud_server_request(self, api_url: str, is_octet_stream: bool = False):
        """Perform request to cloud server

        Args:
            api_url: api url request
            is_octet_stream: true to request binary data, false to request JSON

        Returns:
            response binary content
        """

        try:
            retries = Retry(total=3)  # the number of retries for http/https requests
            s = requests.Session()
            for m in ["https://", "http://"]:
                s.mount(m, HTTPAdapter(max_retries=retries))
            headers = {"token": self._token}
            if is_octet_stream:
                headers["accept"] = "application/octet-stream"

            headers["dg_version"] = pysdk_version
            logger.info(f"sending a request to {self.url}{api_url}")
            res = s.get(
                f"{self.url}{api_url}",
                headers=headers,
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise DegirumException(
                f"Unable to access server {self.url.split('://')[-1]}: {e}"
            ) from None
        if res.status_code == 401:
            response = res.json()
            reason = (
                response["detail"]
                if response and isinstance(response, dict) and "detail" in response
                else "invalid token value"
            )
            raise DegirumException(
                f"Unable to connect to server {self.url.split('://')[-1]}: {reason}"
            )

        try:
            res.raise_for_status()
        except requests.RequestException as e:
            details = str(e)
            try:
                j = res.json()
                if "detail" in j:
                    details = f"{j['detail']}. (cloud server response: {str(e)})"
            except json.JSONDecodeError:
                pass
            raise DegirumException(details) from None

        # if we followed a redirect to https, update url
        if (
            len(res.history) == 1
            and res.url.startswith("https://")
            and self._url.startswith("http://")
        ):
            self._url = "https" + self._url[4:]

        if is_octet_stream:
            return res.content
        else:
            try:
                return res.json()
            except json.JSONDecodeError:
                raise DegirumException(
                    f"Unable to parse response from server {self.url.split('://')[-1]}: {res}"
                ) from None

    def ext_model_name(self, simple_model_name: str) -> str:
        """Construct extended cloud model name from simple model name and zoo path"""
        return f"{self._zoo_url[1:]}/{simple_model_name}"

    @log_wrap
    def download_model(self, model: str, dest_root_path: Path):
        """Download model from the cloud server.

        Args:
            model: model name as returned by list_models()
            dest_root_path: root destination directory path
        """

        # download model archive from cloud zoo
        res = self._cloud_server_request(
            f"/zoo/v1/public/models{self._zoo_url}/{model}", True
        )

        # unzip model archive into model directory
        with zipfile.ZipFile(io.BytesIO(res)) as z:
            z.extractall(dest_root_path)

    def _get_model_list(self, names_only: bool) -> Dict[str, ModelParams]:
        """Get dict of models available in the zoo

        Args:
            names_only: true to request only model names, false to demand full model info

        Returns:
            dict of model names as keys and model info as values
        """

        if names_only:
            if self._assets_type < self._AssetsType.NamesOnly:
                # query model names list from cloud server and store it in the cache
                model_list = self._cloud_server_request(
                    f"/zoo/v1/public/models{self._zoo_url}?short=true"
                )
                self._assets = {
                    k: self._Assets(None, None) for k, _ in model_list.items()
                }
                self._assets_type = self._AssetsType.NamesOnly
        else:
            if self._assets_type < self._AssetsType.Full:
                # query full model list from cloud server and store it in the cache
                model_list = self._cloud_server_request(
                    f"/zoo/v1/public/models{self._zoo_url}"
                )
                self._assets = {
                    k: self._Assets(ModelParams(json.dumps(v)), None)
                    for k, v in model_list.items()
                }
                self._assets_type = self._AssetsType.Full

        return {model: assets.model_params for model, assets in self._assets.items()}

    def _get_model_assets(self, model: str, get_labels: bool) -> _Assets:
        """Get model assets for given model name"""

        assets = self._assets.get(model)

        if assets is None or assets.model_params is None:
            model_info = self._cloud_server_request(
                f"/zoo/v1/public/models{self._zoo_url}/{model}/info"
            )

            assets = self._Assets(
                ModelParams(json.dumps(model_info["model_params"])),
                Model._harmonize_label_dictionary(model_info["model_labels"]),
            )
            self._assets[model] = assets

        if get_labels and assets.model_label_dict is None:
            assets.model_label_dict = Model._harmonize_label_dictionary(
                self._cloud_server_request(
                    f"/zoo/v1/public/models{self._zoo_url}/{model}/dictionary"
                )
            )

        return assets

    def _get_model_info(self, model: str) -> ModelParams:
        """Get model info reference for given model name or None if model is not found"""

        assets = self._get_model_assets(model, False)
        return assets.model_params


class _LocalHWCloudZooAccessor(_CloudZooAccessorBase):
    """Local inference, cloud model zoo implementation"""

    def __init__(self, url: str, token: str):
        """Constructor.

        Args:
            url: path to the cloud zoo in `"https://<cloud server URL>[/<zoo URL>]"` format
            token: cloud zoo access token
        """
        super().__init__(url, token)

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model name.

        Args:
            model: model name as returned by list_models()

        Returns:
            model object corresponding to given model name
        """

        assets = self._get_model_assets(model, True)
        model_params: ModelParams = copy.deepcopy(assets.model_params)
        model_params.CloudModelName = self.ext_model_name(model)
        model_params.CloudURL = self._url
        model_params.CloudToken = self._token if self._token else ""

        # Check Supported Device Types for this model
        supported_device_types = self._model_supported_device_types(model_params)
        if not supported_device_types:
            raise DegirumException(
                f"Cloud Model '{model}' does not have any supported runtime/device combinations that will work on this system."
            )

        return _ClientModel(
            model,
            model_params,
            supported_device_types,
            assets.model_label_dict,
        )

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""
        from .CoreClient import system_info as core_system_info

        return core_system_info()


class _AIServerCloudZooAccessor(_CloudZooAccessorBase):
    """AI server inference, cloud model zoo implementation"""

    def __init__(self, host: str, url: str, token: str):
        """Constructor.

        Args:
            host: AI server hostname
            url: path to the cloud zoo in `"https://<cloud server URL>[/<zoo URL>]"` format
            token: cloud zoo access token
        """
        self._host = host
        super().__init__(url, token)

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model name.

        Args:
            model: model name as returned by list_models()

        Returns:
            model object corresponding to given model name
        """

        assets = self._get_model_assets(model, True)
        model_params: ModelParams = copy.deepcopy(assets.model_params)
        model_params.CloudURL = self._url
        model_params.CloudToken = self._token if self._token else ""

        # Check Supported Device Types for this model
        supported_device_types = self._model_supported_device_types(model_params)
        if not supported_device_types:
            raise DegirumException(
                f"Cloud Model '{model}' does not have any supported runtime/device combinations that will work on this system."
            )

        return _ServerModel(
            self._host,
            self.ext_model_name(model),
            model_params,
            supported_device_types,
            assets.model_label_dict,
        )

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""
        return server_system_info(self._host)


class _CloudServerZooAccessor(_CloudZooAccessorBase):
    """Cloud server inference, cloud model zoo implementation"""

    def __init__(self, url: str, token: str):
        """Constructor.

        Args:
            url: path to the cloud zoo in `"https://<cloud server URL>[/<zoo URL>]"` format
            token: cloud zoo access token
        """
        super().__init__(url, token)

    @log_wrap
    def load_model(self, model: str):
        """Create model object for given model name.

        Args:
            model: model name as returned by list_models()

        Returns:
            model object corresponding to given model name
        """

        assets = self._get_model_assets(model, True)
        model_params: ModelParams = copy.deepcopy(assets.model_params)

        # Check Supported Device Types for this model
        supported_device_types = self._model_supported_device_types(model_params)
        if not supported_device_types:
            raise DegirumException(
                f"Cloud Model '{model}' does not have any supported runtime/device combinations that will work on this system."
            )

        return _CloudServerModel(
            self.url,
            self._token,
            self.ext_model_name(model),
            model_params,
            supported_device_types,
            assets.model_label_dict,
        )

    def _query_system_info(self) -> dict:
        """Return host system information dictionary"""
        return self._cloud_server_request("/devices/api/v1/public/system-info")


def _system_info_run(args):
    """
    Execute system_info command

    Args:
        args: argparse command line arguments
    """

    import yaml

    if args.host:
        info = server_system_info(args.host)
    else:
        from .CoreClient import system_info as core_system_info

        info = core_system_info()

    # remove virtual devices
    if "Devices" in info:
        info["Devices"].pop("DUMMY/DUMMY", None)

    print(yaml.dump(info, sort_keys=False))


def _system_info_args(parser):
    """
    Define sys-info subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "--host",
        default="",
        help="remote AI server hostname/IP; omit for local info",
    )
    parser.set_defaults(func=_system_info_run)


def _trace_run(args):
    """
    Execute trace command

    Args:
        args: argparse command line arguments
    """

    import yaml

    if args.host:
        trace_mgr = lambda req: server_trace_manage(args.host, req)
    else:
        from .CoreClient import trace_manage as core_trace_manage

        trace_mgr = lambda req: core_trace_manage(req)

    if args.command == "list":
        ret = trace_mgr({"config_get": 1})["config_get"]
        print(yaml.dump(ret, sort_keys=False))

    elif args.command == "configure":
        groups = {}

        def apply(arg, level):
            if isinstance(arg, list):
                for gr in arg:
                    groups[gr] = level

        apply(args.basic, 1)
        apply(args.detailed, 2)
        apply(args.full, 3)
        trace_mgr({"config_set": groups})

    elif args.command == "read":
        ret = trace_mgr({"trace_read": {"size": args.filesize}})["trace_read"]
        if args.file:
            with open(args.file, "w") as f:
                f.write(ret)
        else:
            print(ret)


def _trace_args(parser):
    """
    Define trace subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """

    parser.add_argument(
        "command",
        nargs="?",
        choices=["list", "configure", "read"],
        default="list",
        help="trace command: list all available trace groups; configure trace groups; read trace to file",
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="[all] remote AI server hostname/IP (default is 'localhost')",
    )

    parser.add_argument(
        "--basic",
        nargs="+",
        metavar="TRACE-GROUP",
        help="[configure] trace groups to trace with Basic trace level",
    )

    parser.add_argument(
        "--detailed",
        nargs="+",
        metavar="TRACE-GROUP",
        help="[configure] trace groups to trace with Detailed trace level",
    )

    parser.add_argument(
        "--full",
        nargs="+",
        metavar="TRACE-GROUP",
        help="[configure] trace groups to trace with Full trace level",
    )

    parser.add_argument(
        "--file",
        default="",
        metavar="FILENAME",
        help="[read] filename to save trace data into (default is '': print to console)",
    )

    parser.add_argument(
        "--filesize",
        type=int,
        default=10000000,
        help="[read] max. trace data size to read (default is 10000000)",
    )

    parser.set_defaults(func=_trace_run)
