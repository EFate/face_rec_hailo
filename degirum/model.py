#
# model.py - DeGirum Python SDK: model implementation
# Copyright DeGirum Corp. 2022
#
# Implements DeGirum model class
#

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
import copy
from pathlib import Path
import json
import logging
import sys
import threading
import time

import asyncio
from typing import Optional, Callable, Union, Iterator, List, Dict

import msgpack
from ._socketio_msgpack import MsgPackNumpyPacket
import socketio

from .exceptions import DegirumException, validate_color_tuple
from ._preprocessor import _PreprocessorAggregator
from .postprocessor import (
    class_from_type_string,
    _create_overlay_color_dataset,
    InferenceResults,
)
from .aiclient import (
    ModelParams,
    AIModelAsync,
    system_info as server_system_info,
    check_error as dg_check_error,
)
from .log import log_wrap, async_log_wrap
from ._model_param_helpers import model_shape_get, model_shape_set
from ._version import pysdk_version

logger = logging.getLogger(__name__)


class Model(ABC):
    """
    Model class. Handles whole inference lifecycle for a single model: input data preprocessing, inference,
    and postprocessing.

    !!! note

        You never construct model objects yourself -- instead you call
        [degirum.zoo_manager.ZooManager.load_model][] method to create [degirum.model.Model][] instances for you.

    """

    class _Stat:
        """Statistics class.
        Keeps min/max, average, and count
        """

        def __init__(self, desc: str):
            """Constructor"""
            self.min = sys.float_info.max
            self.max = -sys.float_info.max
            self.sum = 0.0
            self.cnt = 0
            self.desc = desc

        @property
        def avg(self):
            """Average value"""
            return self.sum / self.cnt

        def add(self, val: float):
            """Add value to statistics"""
            if val < self.min:
                self.min = val
            if val > self.max:
                self.max = val
            self.cnt += 1
            self.sum += val

        def __str__(self):
            """Convert to string"""
            return f"{self.desc:30}, {self.min:9.2f}, {self.avg:9.2f}, {self.max:9.2f}, {self.cnt:6}"

    class _StatsDict(dict):
        """Dictionary of statistics with pretty-printing"""

        def __str__(self):
            """Convert to string"""
            head = f"{'Statistic':30}, {'Min':>9}, {'Average':>9}, {'Max':>9}, {'Count':>6}\n"
            return head + "\n".join(str(v) for _, v in self.items())

    @log_wrap
    def __init__(
        self,
        model_name: str,
        model_params: ModelParams,
        supported_device_types: List[str],
    ):
        """Constructor.

        !!! note

            You never construct model objects yourself -- instead you call
            [degirum.zoo_manager.ZooManager.load_model][] method to create [degirum.model.Model][] instances for you.

        """
        self._model_name = model_name
        self._model_parameters = model_params
        self._supported_device_types = supported_device_types
        self._preprocessor = _PreprocessorAggregator(self._model_parameters)
        self._overlay_color: Union[list, tuple, None] = None
        self._overlay_line_width = 3
        self._overlay_show_labels = True
        self._overlay_show_probabilities = False
        self._overlay_alpha: Union[float, str] = "auto"
        self._overlay_font_scale = 1.0
        self._overlay_blur: Union[str, list, None] = None
        self._save_model_image = False
        self._time_stats = self._StatsDict()
        self._results: list = []  # inference results record list
        self._non_blocking_batch_predict = False
        self._result_ready_cv = threading.Condition()
        self._in_context = False
        # by default, set format to JPEG for all image-type inputs to reduce traffic
        self._patch_img_fmt("JPEG")
        self._frame_queue_depth = 8
        self._inference_timeout_s = self._deduce_inference_timeout_s()
        self._custom_postprocessor: Optional[type] = None
        self._label_dict: Union[None, Callable, dict] = None
        self._output_class_set: set = set()

    def __dir__(self):
        return [
            "__call__",
            "custom_postprocessor",
            "device_type",
            "devices_available",
            "devices_selected",
            "eager_batch_size",
            "frame_queue_depth",
            "image_backend",
            "inference_results_type",
            "inference_timeout_s",
            "input_image_format",
            "input_letterbox_fill_color",
            "input_numpy_colorspace",
            "input_pad_method",
            "input_crop_percentage",
            "input_resize_method",
            "input_shape",
            "label_dictionary",
            "measure_time",
            "model_info",
            "non_blocking_batch_predict",
            "output_confidence_threshold",
            "output_class_set",
            "output_max_detections",
            "output_max_detections_per_class",
            "output_max_classes_per_detection",
            "output_nms_threshold",
            "output_pose_threshold",
            "output_postprocess_type",
            "output_top_k",
            "output_use_regular_nms",
            "overlay_alpha",
            "overlay_blur",
            "overlay_color",
            "overlay_font_scale",
            "overlay_line_width",
            "overlay_show_labels",
            "overlay_show_probabilities",
            "predict",
            "predict_batch",
            "predict_dir",
            "reset_time_stats",
            "save_model_image",
            "supported_device_types",
            "time_stats",
        ]

    @log_wrap
    def __enter__(self):
        """Context manager enter handler."""
        self._in_context = True
        return self

    @log_wrap
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit handler."""
        self._in_context = False
        self._release_runtime()

    @log_wrap
    def predict(self, data) -> InferenceResults:
        """Perform whole inference lifecycle: input data preprocessing, inference, and postprocessing.

        Args:
            data (any): Inference input data. Input data type depends on the model.

                - If the model expects image data, then the input data is either:

                    - Input image path string.
                    - NumPy 3D array of pixels in a form HWC.
                    where color dimension is native to selected graphical backend (RGB for `'pil'` and BGR for `'opencv'` backend)
                    - `PIL.Image` object (only for `'pil'` backend).

                - If the model expects audio data, then the input data is NumPy 1D array with audio data samples.

                - If the model expects raw tensor data, then the input data is NumPy multidimensional array with shape matching model input.

                - In case of multi-input model a list of elements of the supported data type is expected.

        Returns:
            Inference result object, which allows you to access inference results as a dictionary or as
                an overlay image if it is supported by the model. For your convenience, all image coordinates in case
                of detection models are converted from model coordinates to original image coordinates.
        """

        @log_wrap
        def source():
            yield (data if isinstance(data, str) else "", data)

        res = list(self._predict_impl(source))
        return res[0]

    @log_wrap
    def __call__(self, data):
        """Perform whole inference lifecycle: input data preprocessing, inference and postprocessing.

        Same as [degirum.model.Model.predict][].
        """
        return self.predict(data)

    @log_wrap
    def predict_batch(self, data) -> Iterator[InferenceResults]:
        """Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Such iterator object should return the same object types which regular [degirum.model.Model.predict][] method accepts.

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.

                Each element returned by this iterator can be one of the following:

                - A single input data object, in case of single-input model.
                - A `list` of input data objects, in case of multi-input model.
                - A `tuple` containing a pair of input data object or a `list` of input data objects as a first element
                and frame info object as a second element of the `tuple`.

                The input data object type depends on the model.

                - If the model expects image data, then the input data object is either:
                    - Input image path string.
                    - NumPy 3D array of pixels in a form HWC, where color dimension is native to selected graphical backend
                    (RGB for `'pil'` and BGR for `'opencv'` backend).
                    - `PIL.Image` object (only for `'pil'` backend).

                - If the model expects audio data, then the input data object is NumPy 1D array with audio data samples.

                - If the model expects raw tensor data, then the input data object is NumPy multidimensional array with shape
                matching model input.

                The frame info object is passed to the inference result object unchanged and can be accessed via `info`
                property of the inference result object.

        Returns:
            Generator object which iterates over inference result objects. This allows you directly using the
                result of [degirum.model.Model.predict_batch][] in `for` loops.

        Example:
            ```python
                for result in model.predict_batch(['image1.jpg','image2.jpg']):
                    print(result)
            ```
        """

        @log_wrap
        def source():
            for d in data:
                if isinstance(d, tuple):
                    # if data is tuple, we treat first element as frame data and second element as frame info
                    yield d[1], d[0]
                else:
                    # otherwise we treat data as frame data and if it is string, we set frame info equal to frame data
                    # (data is string when it is a filename)
                    yield d if isinstance(d, str) else "", d

        for res in self._predict_impl(source):
            yield res

    @log_wrap
    def predict_dir(
        self,
        path: str,
        *,
        recursive: bool = False,
        extensions=[".jpg", ".jpeg", ".png", ".bmp"],
    ) -> Iterator[InferenceResults]:
        """Perform whole inference lifecycle for all files from specified directory matching given file extensions.

        Supports only single-input models.

        Args:
            path: Directory name containing files to be processed.
            recursive: True to recursively walk through all subdirectories in a directory. Default is `False`.
            extensions (list[str]): Single string or list of strings containing file extension(s) to process.

        Returns:
            Generator object to iterate over inference result objects. This allows you directly using the
                result of [degirum.model.Model.predict_dir][] in `for` loops.

        Example:
            ```python
                for result in model.predict_dir('./some_path'):
                    print(result)
            ```
        """

        if len(self._model_parameters.InputType) > 1:
            raise DegirumException(
                "'predict_dir' method is not supported for models with number of inputs greater than one."
            )

        mask = "**/*.*" if recursive else "*.*"
        ext = extensions if isinstance(extensions, list) else [extensions]

        @log_wrap
        def source():
            for e in Path(path).glob(mask):
                if e.is_file() and e.suffix.lower() in ext:
                    yield str(e), str(e)

        for res in self._predict_impl(source):
            yield res

    @property
    def custom_postprocessor(self) -> Optional[type]:
        """Custom postprocessor class. When not None, the object of this class is returned as inference result.
        Such custom postprocessor classes must be inherited from [degirum.postprocessor.InferenceResults][] class.
        """
        return self._custom_postprocessor

    @custom_postprocessor.setter
    def custom_postprocessor(self, val: type):
        if not issubclass(val, InferenceResults):
            raise DegirumException(
                f"Custom postprocessor must be inherited from {InferenceResults.__name__} class"
            )
        self._custom_postprocessor = val

    @property
    def eager_batch_size(self) -> int:
        """The size of the batch (number of consecutive frames before this model is switched
        to another model during batch predict) to be used by device scheduler when inferencing this model.
        """
        return self._model_parameters.EagerBatchSize

    @eager_batch_size.setter
    def eager_batch_size(self, val: int):
        self._model_parameters.EagerBatchSize = val

    @property
    def frame_queue_depth(self) -> int:
        """The depth of the model prediction queue. When the queue size reaches this value,
        the next prediction call will block until there will be space in the queue."""
        return self._frame_queue_depth

    @frame_queue_depth.setter
    def frame_queue_depth(self, val: int):
        max_depth = 160
        if val < 1 or val > max_depth:
            raise DegirumException(
                f"Frame queue depth {val} is out of range 1..{max_depth}"
            )

        if self._frame_queue_depth != val:
            self._frame_queue_depth = val
            self._model_parameters.dirty = True

    @property
    def supported_device_types(self) -> List[str]:
        """
        The list of supported device types in format `<runtime>/<device>` for this model.
        """
        return self._supported_device_types

    @property
    def device_type(self) -> str:
        """The type of the device to be used for model inference in a format `<runtime>/<device>`

        Setter accepts either a string which specifies single device in a format `<runtime>/<device>`
        or it can be a list of such strings. In this case the first supported device type
        from the list will be selected.

        Supported device types can be obtained by [degirum.model.Model.supported_device_types][] property.

        Getter returns currently selected device type.
        """

        return (
            self._model_parameters.RuntimeAgent
            + "/"
            + self._model_parameters.DeviceType
        )

    @device_type.setter
    def device_type(self, val: Union[str, List[str]]):

        def check_device_type(device_type: str) -> Optional[list]:

            agent_device = device_type.split("/")
            if len(agent_device) != 2:
                raise DegirumException(
                    f"Device type must be in format <runtime>/<device>, not '{val}'"
                )

            if device_type in self._supported_device_types:
                return agent_device

            return None

        val = val if isinstance(val, list) else [val]
        agent_device: Optional[list] = None
        for device_type in val:
            agent_device = check_device_type(device_type)
            if agent_device is not None:
                break  # take first suitable
        if agent_device is None:
            raise DegirumException(
                f"None of the device types in the list {val} is supported by model {self._model_name}. "
                + f"Supported device types are: {self._supported_device_types}."
            )

        self._model_parameters.RuntimeAgent = agent_device[0]
        self._model_parameters.DeviceType = agent_device[1]
        self._inference_timeout_s = self._deduce_inference_timeout_s()
        self._model_parameters.dirty = True

    @property
    @abstractmethod
    def devices_available(self) -> list:
        """The list of inference device indices which can be used for model inference."""

    def _get_device_list(self, sys_info: dict) -> list:
        """Retrieve list of available devices from given system info dictionary"""

        if "Devices" in sys_info:
            devices = sys_info["Devices"]
            dev_key = (
                self._model_parameters.RuntimeAgent
                + "/"
                + self._model_parameters.DeviceType
            )
            if dev_key in devices:
                return list(range(len(devices[dev_key])))
        return []

    @property
    def devices_selected(self) -> list:
        """The list of inference device indices selected for model inference."""

        dev_avail = self.devices_available
        dev_mask = self._model_parameters.DeviceMask
        return [
            bi
            for bi in range(64)
            if ((dev_mask & (1 << bi)) != 0) and (bi in dev_avail)
        ]

    @devices_selected.setter
    def devices_selected(self, val: list):
        dev_avail = self.devices_available
        dev_mask = sum([1 << bi for bi in val if bi in dev_avail])

        if dev_mask == 0:
            raise DegirumException(
                f"The list of devices to select {val} and the list of available devices {dev_avail}"
                + f" for model '{self._model_name}' have zero intersection"
            )

        self._model_parameters.DeviceMask = dev_mask

    @property
    def model_info(self):
        """Return model information object to provide read-only access to model parameters.

        New deep copy is created each time."""
        return copy.deepcopy(self._model_parameters)

    def _deduce_inference_timeout_s(self) -> float:
        """Deduce inference timeout based on model type"""

        default_timeout_s = 40.0
        agent_timeouts_s = {
            "TENSORRT": 180.0,
        }
        device_timeouts_s = {
            "CPU": 100.0,
        }

        ret = agent_timeouts_s.get(self._model_parameters.RuntimeAgent, None)
        if not ret:
            ret = device_timeouts_s.get(
                self._model_parameters.DeviceType, default_timeout_s
            )
        return ret

    @property
    def inference_timeout_s(self) -> float:
        """The maximum time in seconds to wait for inference result from the model."""
        return self._inference_timeout_s

    @inference_timeout_s.setter
    def inference_timeout_s(self, val: float):
        if self._inference_timeout_s != val:
            self._inference_timeout_s = val
            self._model_parameters.dirty = True

    @property
    def measure_time(self) -> bool:
        """Flag to enable measuring and collecting inference time statistics.

        Call [degirum.model.Model.time_stats][] to query accumulated inference time statistics.
        """
        return self._model_parameters.MeasureTime

    @measure_time.setter
    def measure_time(self, val: bool):
        if not self._model_parameters.MeasureTime and val:
            self._time_stats = self._StatsDict()
        self._model_parameters.MeasureTime = val

    @property
    def non_blocking_batch_predict(self):
        """Flag to control the behavior of the generator object returned by `predict_batch()` method.

        - When the flag is set to `True`, the generator accepts `None` from the inference input data
        iterator object (passed as `data` parameter): If `None` is returned, the model predict step is skipped for
        this iteration. Also, when no inference results are available in the result queue at this iteration,
        the generator yields `None` result.

        - When the flag is set to `False` (default value), the generator does not allow `None` to be returned from
        the inference input data iterator object: If `None` is returned, an exception is raised.
        Also, when no inference results are available in the result queue at this iteration,
        the generator continues to the next iteration of the input data iterator.

        - Setting this flag to `True` allows using `predict_batch()` generator in a non-blocking manner, assuming
        the design of input data iterator object is also non-blocking, i.e., returning `None` when no data is
        available instead of waiting for the data. Every next element request from the generator will not block
        the execution waiting for either input data or inference results, returning `None` when no results are
        available.
        """

        return self._non_blocking_batch_predict

    @non_blocking_batch_predict.setter
    def non_blocking_batch_predict(self, val: bool):
        assert isinstance(val, bool)
        self._non_blocking_batch_predict = val

    @property
    @abstractmethod
    def label_dictionary(self) -> Dict[int, str]:
        """Get model class label dictionary.

        Each dictionary element is key-value pair, where the key is the class ID
        and the value is the class label string."""

    @staticmethod
    def _harmonize_label_dictionary(label_dictionary: dict) -> Dict[int, str]:
        """Convert label dictionary to canonic form: integer key and string value
        Args:
            label_dictionary: dictionary to convert

        Returns:
            converted dictionary
        """
        return {
            int(k): str(v)
            for k, v in label_dictionary.items()
            if isinstance(k, (int, float)) or (isinstance(k, str) and k.isdigit())
        }

    @property
    def input_resize_method(self) -> str:
        """Input image resize method -- one of `'nearest'`, `'bilinear'`, `'area'`, `'bicubic'`, or `'lanczos'`."""
        return self._preprocessor.resize_method

    @input_resize_method.setter
    def input_resize_method(self, val: str):
        self._preprocessor.resize_method = val

    @property
    def input_shape(self) -> List[List[int]]:
        """Input tensor shapes. List of tensor shapes per input.

        Each element of that list is another list containing tensor dimensions, slowest dimension first:
            - if InputShape model parameter is specified, its value is used.
            - otherwise, all *defined* InputN/H/W/C model parameters are used as [InputN, InputH, InputW, InputC] list.
        """
        return model_shape_get(self._model_parameters, -1, 0)

    @input_shape.setter
    def input_shape(self, val: List[List[int]]):
        for ii, new_shape in enumerate(val):
            model_shape_set(self._model_parameters, ii, new_shape)

    @property
    def input_pad_method(self) -> str:
        """Input image pad method -- one of `'stretch'`,  `'letterbox'`, `'crop-first'`, or `'crop-last'`.

        - In case of `'stretch'`, the input image will be resized to the model input size **without** preserving aspect ratio.
        - In case of `'letterbox'`, the input image will be resized to the model input size preserving aspect ratio.
        - In case of `'crop-first'`, the input image will be cropped to input_crop_percentage around the center and then resized.
        - In the case of 'crop-last', if the model's input dimensions are square, the image is resized with its smaller side matching the model dimension, preserving the aspect ratio. If the dimensions are rectangular, the image is resized and stretched to fit the model's input dimensions. After resizing, the image is cropped to the model's input dimensions and aspect ratio based on the 'input_crop_percentage' property.

        The voids will be filled with solid color specified by `input_letterbox_fill_color` property.
        In all cases [degirum.model.Model.input_resize_method][] property specifies the algorithm for resizing.
        """
        return self._preprocessor.pad_method

    @input_pad_method.setter
    def input_pad_method(self, val: str):
        self._preprocessor.pad_method = val

    @property
    def input_numpy_colorspace(self) -> str:
        """Input image colorspace -- one of `'RGB'`, `'BGR'`, or `'auto'`.

        This parameter is used **only** to identify colorspace for NumPy arrays.

        `'auto'` translates to `'BGR'` for `opencv` backend, and to `'RGB'` for `pil` backend.
        """
        return self._preprocessor.input_colorspace

    @input_numpy_colorspace.setter
    def input_numpy_colorspace(self, val: str):
        self._preprocessor.input_colorspace = val

    @property
    def input_letterbox_fill_color(self) -> tuple:
        """Image fill color in case of `'letterbox'` padding (see [degirum.model.Model.input_pad_method][]
        property for details).

        3-element RGB tuple."""
        return self._preprocessor.fill_color

    @input_letterbox_fill_color.setter
    def input_letterbox_fill_color(self, val: tuple):
        self._preprocessor.fill_color = validate_color_tuple(val)

    @property
    def input_crop_percentage(self) -> float:
        """Percentage of image to crop around. Valid range: `[0..1]`."""
        return self._preprocessor.crop_percentage

    @input_crop_percentage.setter
    def input_crop_percentage(self, val: float):
        if val < 0.0 or val > 1.0:
            raise DegirumException("Crop Percentage must be a value between 0 and 1")
        self._preprocessor.crop_percentage = val

    @property
    def image_backend(self) -> str:
        """Graphical library (*backend*) to use for graphical tasks -- one of `'pil'`, `'opencv'`, `'auto'`

        `'auto'` means try OpenCV first, if not installed, try PIL."""
        return self._preprocessor.image_backend

    @image_backend.setter
    def image_backend(self, val: str):
        self._preprocessor.image_backend = val

    @property
    def input_image_format(self) -> str:
        """Defines the image format for model inputs of image type -- one of `'JPEG'` or `'RAW'`."""
        return self._preprocessor.image_format

    @input_image_format.setter
    def input_image_format(self, val: str):
        self._preprocessor.image_format = val

    @property
    def save_model_image(self) -> bool:
        """Flag to enable/disable saving of model input image in inference results.

        Model input image is the image converted to AI model input specifications as raw binary array.
        """
        return self._save_model_image

    @save_model_image.setter
    def save_model_image(self, val: bool):
        self._save_model_image = val

    @property
    def output_postprocess_type(self) -> str:
        """Inference result post-processing type.

        You may set it to `'None'` to bypass post-processing."""
        return self._model_parameters.OutputPostprocessType

    @output_postprocess_type.setter
    def output_postprocess_type(self, val: str):
        self._model_parameters.OutputPostprocessType = val

    @property
    def inference_results_type(self) -> str:
        """Inference result type. Specifies the type of inference results to be returned by the model inference.
        When empty, it is deduced from the model output_postprocess_type.
        """
        return self._model_parameters.InferenceResultsType

    @inference_results_type.setter
    def inference_results_type(self, val: str):
        self._model_parameters.InferenceResultsType = val

    @property
    def output_class_set(self) -> set:
        """Labels filter: list of class labels/category IDs to be included in inference results.

        !!! note

            You can use [degirum.model.Model.label_dictionary][] property to obtain a list of model classes.
        """
        return self._output_class_set

    @output_class_set.setter
    def output_class_set(self, val: set):
        if not isinstance(val, set):
            raise DegirumException(
                f"output_class_set must be a set, not `{type(val).__name__}`"
            )
        if val:
            if not all(isinstance(e, int) for e in val) and not all(
                isinstance(e, str) for e in val
            ):
                raise DegirumException(
                    "output_class_set must contain only int or str elements"
                )
        self._output_class_set = val

    @property
    def output_confidence_threshold(self) -> float:
        """Confidence threshold used in inference result post-processing.

        Valid range: `[0..1]`.

        Only objects with scores higher than this threshold are reported.

        !!! note

            For classification models if [degirum.model.Model.output_top_k][] parameter is set to non-zero value,
            then it supersedes this threshold -- [degirum.model.Model.output_top_k][] highest score classes are always reported.
        """
        return self._model_parameters.OutputConfThreshold

    @output_confidence_threshold.setter
    def output_confidence_threshold(self, val: float):
        self._model_parameters.OutputConfThreshold = val

    @property
    def output_top_k(self) -> float:
        """The number of classes with highest scores to report for classification models.

        When set to `0`, then report all classes with scores greater than [degirum.model.Model.output_confidence_threshold][].
        """
        return self._model_parameters.OutputTopK

    @output_top_k.setter
    def output_top_k(self, val: float):
        self._model_parameters.OutputTopK = val

    @property
    def output_nms_threshold(self) -> float:
        """Non-Max Suppression (NMS) threshold used in inference result post-processing.

        Valid range: `[0..1]`.

        Applicable only for models which utilize NMS algorithm."""
        return self._model_parameters.OutputNMSThreshold

    @output_nms_threshold.setter
    def output_nms_threshold(self, val: float):
        self._model_parameters.OutputNMSThreshold = val

    @property
    def output_pose_threshold(self) -> float:
        """Pose detection threshold used in inference result post-processing.

        Valid range: `[0..1]`.

        Applicable only for pose detection models."""
        return self._model_parameters.PoseThreshold

    @output_pose_threshold.setter
    def output_pose_threshold(self, val: float):
        self._model_parameters.PoseThreshold = val

    @property
    def output_max_detections(self) -> int:
        """Max Detection number used in inference result post-processing, and specifies the total maximum objects
        of number to be detected.

        Applicable only for detection models."""
        return self._model_parameters.MaxDetections

    @output_max_detections.setter
    def output_max_detections(self, val: int):
        self._model_parameters.MaxDetections = val

    @property
    def output_max_detections_per_class(self) -> int:
        """Max Detections Per Class number used in inference result post-processing, and specifies the maximum number
        of objects to keep during per class non-max suppression process for regular algorithm.

        Applicable only for detection models."""
        return self._model_parameters.MaxDetectionsPerClass

    @output_max_detections_per_class.setter
    def output_max_detections_per_class(self, val: int):
        self._model_parameters.MaxDetectionsPerClass = val

    @property
    def output_max_classes_per_detection(self) -> int:
        """Max Detections Per Class number used in inference result post-processing, and specifies the maximum number
        of highest probability classes per anchor to be processed during the non-max suppression process for fast algorithm.

        Applicable only for detection models."""
        return self._model_parameters.MaxClassesPerDetection

    @output_max_classes_per_detection.setter
    def output_max_classes_per_detection(self, val: int):
        self._model_parameters.MaxClassesPerDetection = val

    @property
    def output_use_regular_nms(self) -> bool:
        """Use Regular NMS value used in inference result post-processing and specifies the algorithm to use for
        detection postprocessing.

        If value is `True`, regular Non-Max suppression algorithm is used -- NMS is calculated for each class
        separately and after that all results are merged.

        If value is `False`, fast Non-Max suppression algorithm is used -- NMS is calculated for all classes
        simultaneously.
        """
        return self._model_parameters.UseRegularNMS

    @output_use_regular_nms.setter
    def output_use_regular_nms(self, val: bool):
        self._model_parameters.UseRegularNMS = val

    @property
    def overlay_color(self):
        """Color for inference results drawing on overlay image.

        3-element RGB tuple or list of 3-element RGB tuples.

        The `overlay_color` property is used to define the color to draw overlay details. In the case of a single RGB tuple,
        the corresponding color is used to draw all the overlay data: points, boxes, labels, segments, etc.
        In the case of a list of RGB tuples the behavior depends on the model type:

        - For classification models different colors from the list are used to draw labels of different classes.
        - For detection models different colors are used to draw labels *and boxes* of different classes.
        - For pose detection models different colors are used to draw keypoints of different persons.
        - For segmentation models different colors are used to highlight segments of different classes.

        If the list size is less than the number of classes of the model, then `overlay_color` values are used cyclically,
        for example, for three-element list it will be `overlay_color[0]`, then `overlay_color[1]`, `overlay_color[2]`,
        and again `overlay_color[0]`.

        The default value of `overlay_color` is a single RBG tuple of yellow color for all model types except segmentation models.
        For segmentation models it is the list of RGB tuples with the list size equal to the number of model classes.
        You can use [degirum.model.Model.label_dictionary][] property to obtain a list of model classes.
        Each color is automatically assigned to look pretty and different from other colors in the list.
        """
        if self._overlay_color is None:
            self._overlay_color = _create_overlay_color_dataset(
                self._model_parameters.OutputPostprocessType,
                self._model_parameters.OutputNumClasses,
                self.label_dictionary,
            )
        return copy.deepcopy(self._overlay_color)

    @overlay_color.setter
    def overlay_color(self, val):
        if isinstance(val, Iterable) and all(isinstance(e, Iterable) for e in val):
            # sequence of colors
            self._overlay_color = [validate_color_tuple(e) for e in val]
        else:
            # single color
            self._overlay_color = validate_color_tuple(val)

    @property
    def overlay_line_width(self) -> int:
        """Line width for inference results drawing on overlay image.

        See [degirum.postprocessor.InferenceResults.image_overlay][] for more details.
        """
        return self._overlay_line_width

    @overlay_line_width.setter
    def overlay_line_width(self, val):
        self._overlay_line_width = val

    @property
    def overlay_show_labels(self) -> bool:
        """Flag to enable/disable drawing class labels on overlay image.

        See [degirum.postprocessor.InferenceResults.image_overlay][] for more details.
        """
        return self._overlay_show_labels

    @overlay_show_labels.setter
    def overlay_show_labels(self, val):
        self._overlay_show_labels = val

    @property
    def overlay_show_probabilities(self) -> bool:
        """Flag to enable/disable drawing class probabilities on overlay image.

        See [degirum.postprocessor.InferenceResults.image_overlay][] for more details.
        """
        return self._overlay_show_probabilities

    @overlay_show_probabilities.setter
    def overlay_show_probabilities(self, val):
        self._overlay_show_probabilities = val

    @property
    def overlay_alpha(self) -> Union[float, str]:
        """Alpha-blend weight for inference results drawing on overlay image.

        `float` number in range `[0..1]`.

        See [degirum.postprocessor.InferenceResults.image_overlay][] for more details.
        """
        return self._overlay_alpha

    @overlay_alpha.setter
    def overlay_alpha(self, val: float):
        self._overlay_alpha = val

    @property
    def overlay_blur(self) -> Union[str, list, None]:
        """Overlay blur option.

        `None` for no blur, `"all"` to blur all objects, a class label or list of class
        labels to blur specific objects."""
        return self._overlay_blur

    @overlay_blur.setter
    def overlay_blur(self, val: Union[str, list, None]):
        self._overlay_blur = val

    @property
    def overlay_font_scale(self) -> float:
        """Font scale for inference results drawing on overlay image.

        `float` positive number.

        See [degirum.postprocessor.InferenceResults.image_overlay][] for more details.
        """
        return self._overlay_font_scale

    @overlay_font_scale.setter
    def overlay_font_scale(self, val):
        self._overlay_font_scale = val

    @log_wrap
    def _record_time(self, desc: str, time_ms: float):
        """Record execution time in time statistics"""
        if desc in self._time_stats:
            self._time_stats[desc].add(time_ms)
        else:
            stat = self._Stat(desc)
            stat.add(time_ms)
            self._time_stats[desc] = stat

    def time_stats(self) -> dict:
        """Query inference time statistics.

        Returns:
            Dictionary containing time statistic objects.

                - A key in that dictionary is a string description of a particular inference step.
                - Each statistic object keeps min, max, and average values in milliseconds, accumulated over all
                inferences performed on this model since the model creation of last call of statistic reset
                method [degirum.model.Model.reset_time_stats][].
                - Time statistics are accumulated only when [degirum.model.Model.measure_time][] property is set to `True`.
        """
        return self._time_stats

    def reset_time_stats(self):
        """Reset inference time statistics.

        [degirum.model.Model.time_stats][] method will return empty dictionary after this call.
        """
        self._time_stats = self._StatsDict()

    def get_inference_results_type(self) -> str:
        """Get inference results type string, deduced from model parameters"""
        return (
            self._model_parameters.InferenceResultsType
            if self._model_parameters.InferenceResultsType
            else self._model_parameters.OutputPostprocessType
        )

    def get_inference_results_class(self) -> type:
        """Get inference results class, deduced from model parameters"""
        return (
            class_from_type_string(self.get_inference_results_type())
            if self._custom_postprocessor is None
            else self._custom_postprocessor
        )

    @log_wrap
    def _reconnect(self) -> bool:
        """Perform server (re)connect.
        Return True when reconnect really took place, False otherwise."""
        return False

    @abstractmethod
    @contextmanager
    def _predict_handler(self):
        """Context manager to prepare and wrapup the asynchronous inference process."""
        yield

    @log_wrap
    def _create_postprocessor(self, inference_results, inference_info):
        """Create InferenceResults instance for current inference results

        Args:
            inference_results: inference results record
            inference_info: inference info record

        Returns:
            InferenceResults instance
        """

        frame_info = inference_info["frame_info"]
        preprocessed_info = inference_info["preprocessed_info"]
        info = (
            preprocessed_info[0]
            if isinstance(preprocessed_info, list)
            else preprocessed_info
        )
        # info is single-channel preprocessor info

        result_class = self.get_inference_results_class()
        result = result_class(
            inference_results=inference_results,
            conversion=info["converter"] if "converter" in info else None,
            input_image=info["image_input"] if "image_input" in info else None,
            model_image=info["image_result"] if self._save_model_image else None,
            draw_color=self.overlay_color,
            line_width=self.overlay_line_width,
            show_labels=self.overlay_show_labels,
            show_probabilities=self.overlay_show_probabilities,
            alpha=self.overlay_alpha,
            blur=self.overlay_blur,
            font_scale=self.overlay_font_scale,
            fill_color=(
                self.input_letterbox_fill_color if "image_input" in info else None
            ),
            frame_info=frame_info,
            label_dictionary=self.label_dictionary,
            input_shape=self.input_shape,
        )
        return result

    @log_wrap
    def _predict_impl(self, sources):
        """Perform whole inference lifecycle: input data preprocessing, inference and postprocessing.

        - `source`: inference input data generator

        Returns generator object to iterate over inference results.
        """

        key_timing = "Timing"

        @log_wrap
        def prepare_results(result, info):
            # check for errors
            error_msg = dg_check_error(result)
            if error_msg:
                raise DegirumException(
                    f"Model '{self._model_name}' inference failed: {error_msg}"
                )

            # create postprocessor object
            pp = self._create_postprocessor(result, info)

            # record timings
            if self.measure_time:
                r = pp._inference_results
                if (
                    isinstance(r, list)
                    and len(r) > 0
                    and isinstance(r[0], dict)
                    and key_timing in r[0]
                ):
                    pp.timing = r[0][key_timing]  # copy timing data to postprocessor
                    del r[0][key_timing]  # and remove it from results
                    for key, val in pp.timing.items():
                        self._record_time(key, val)

                frame_duration_ns = time.time_ns() - info["start_time_ns"]
                self._record_time("FrameTotalDuration_ms", frame_duration_ns * 1e-6)

            # filter labels
            if self._output_class_set and isinstance(pp._inference_results, list):
                for e in self._output_class_set:
                    # fastest method to get first element of the set is to use for/break
                    key = "label" if isinstance(e, str) else "category_id"
                    pp._inference_results[:] = [
                        r
                        for r in pp._inference_results
                        if isinstance(r, dict)
                        and r.get(key, None) in self._output_class_set
                    ]
                    break

            return pp

        self._results = []
        postprocessor_data = []

        saved_exception: Optional[DegirumException] = None
        try:
            with self._predict_handler():  # __exit__() of this context manager has runtime-specific cleanup code
                assert self._runtime
                if self._preprocessor.has_image_inputs:
                    self._preprocessor.generate_image_result = self._save_model_image

                # iterate frames taken from source iterator
                for frame_info, data in sources():
                    if data is None:
                        if not self.non_blocking_batch_predict:
                            raise DegirumException(
                                f"Model '{self._model_name}' misconfiguration:"
                                + " input data iterator returns None but non-blocking batch predict mode is not enabled"
                            )

                    else:
                        # record frame start time
                        start_time_ns = time.time_ns()

                        # do pre-processing
                        (
                            preprocessed_data,
                            preprocessed_info,
                        ) = self._preprocessor.forward(data)

                        # store input data for post-processor
                        postprocessor_data.append(
                            dict(
                                preprocessed_info=preprocessed_info,
                                start_time_ns=start_time_ns,
                                frame_info=frame_info,
                            )
                        )

                        if self.measure_time:
                            preproc_duration_ns = time.time_ns() - start_time_ns
                            self._record_time(
                                "PythonPreprocessDuration_ms",
                                preproc_duration_ns * 1e-6,
                            )

                        # schedule frame for prediction
                        self._runtime.predict(preprocessed_data, "")

                    # check and process results queue
                    if len(self._results) > 0:
                        yield prepare_results(
                            self._results.pop(0), postprocessor_data.pop(0)
                        )
                    else:
                        if self.non_blocking_batch_predict:
                            yield None  # no results available: yield None in non-blocking mode

                # wait for completion of all scheduled predictions yielding results as they become available
                start_time_s = time.time()
                while len(postprocessor_data) > 0:
                    with self._result_ready_cv:
                        self._result_ready_cv.wait_for(
                            lambda: len(self._results) > 0, timeout=0.01
                        )
                    if self._reconnect():
                        start_time_s = time.time()  # reset start time on reconnect

                    while self._results:
                        yield prepare_results(
                            self._results.pop(0), postprocessor_data.pop(0)
                        )
                        start_time_s = time.time()  # reset start time on each result

                    if (
                        time.time() - start_time_s > self.inference_timeout_s
                        and len(postprocessor_data) > 0
                        and len(self._results) == 0
                    ):
                        raise DegirumException(
                            f"Timeout waiting for model '{self._model_name}' inference completion "
                            + f"(timeout = {self.inference_timeout_s} sec, missing results = {len(postprocessor_data)})"
                        )

            assert len(postprocessor_data) >= len(self._results)

        except Exception as e:
            saved_exception = DegirumException(str(e))
            # NOTE: we stringize exception instead of saving original exception object to avoid cycling references

        try:
            # process all unprocessed results (even in case of error we want to yield them)
            while self._results:
                yield prepare_results(self._results.pop(0), postprocessor_data.pop(0))
        except Exception as e:
            if saved_exception is None:
                saved_exception = DegirumException(str(e))

        if saved_exception is not None:
            self._release_runtime()

            msg = f"Failed to perform model '{self._model_name}' inference: {str(saved_exception)}"
            logger.debug(saved_exception.traceback)
            raise DegirumException(msg) from saved_exception

    @log_wrap
    def _release_runtime(self):
        """Release runtime object"""
        self._runtime = None  # release runtime object

    @log_wrap
    def _patch_img_fmt(self, new_fmt: str):
        """Patch model input image format to given format for all image-type inputs"""
        new_fmt_array = self._model_parameters.InputImgFmt
        for i, input in enumerate(self._model_parameters.InputType):
            if input == "Image":
                new_fmt_array[i] = new_fmt
        self._model_parameters.InputImgFmt = new_fmt_array


class _ClientModel(Model):
    """
    Local Model class. Handles whole inference lifecycle for single model on locally-installed DeGirum hardware:
    input data preprocessing, inference and postprocessing.
    """

    @staticmethod
    @log_wrap
    def load_config_file(config_file: str) -> ModelParams:
        """Load configuration file.

        - `config_file`: model configuration file name
        """
        if not Path(config_file).exists():
            raise DegirumException(
                f"Model configuration file '{config_file}' is not found"
            )
        conf_path = Path(config_file).resolve()
        model_params = ModelParams(conf_path.read_text().replace("\n", ""))

        # fix various paths to be the same as path to JSON
        if model_params.ModelPath:
            model_params.ModelPath = str(conf_path.with_name(model_params.ModelPath))
        if model_params.LabelsPath:
            model_params.LabelsPath = str(conf_path.with_name(model_params.LabelsPath))
        if model_params.PythonFile:
            model_params.PythonFile = str(conf_path.with_name(model_params.PythonFile))

        return model_params

    @log_wrap
    def __init__(
        self,
        model_name: str,
        model_params: ModelParams,
        supported_device_types: List[str],
        label_dict: Union[None, dict, Callable] = None,
    ):
        """Constructor.
        Args:
            model_name: Model name.
            model_params: Model configuration file contents.
            label_dict: Model label dictionary (optional).
        """
        super().__init__(model_name, model_params, supported_device_types)
        self._runtime = None
        self._label_dict = label_dict
        self._patch_img_fmt("RAW")  # set RAW format to reduce computations

    @log_wrap
    @contextmanager
    def _predict_handler(self):
        """Context manager to prepare and wrapup the asynchronous inference process."""
        from .CoreClient import AsyncRuntime as CoreClientAsyncRuntime

        if self._runtime is None or self._model_parameters.dirty:
            self._runtime = CoreClientAsyncRuntime(
                str(self._model_parameters),
                None,
                self.frame_queue_depth,
                None,
                int(self.inference_timeout_s * 1000),
            )
            self._model_parameters.dirty = False

        def callback(result, frame_info, rt_context):
            self._results.append(result)
            with self._result_ready_cv:
                self._result_ready_cv.notify_all()

        self._runtime.set_callback(callback)

        try:
            yield
        finally:
            self._runtime.finish()
            self._runtime.set_callback(None)

    @property
    def label_dictionary(self) -> Dict[int, str]:
        """Get model class label dictionary"""
        if callable(self._label_dict):
            # request label dictionary and cache it
            self._label_dict = Model._harmonize_label_dictionary(self._label_dict())
        elif self._label_dict is None:
            if self._model_parameters.LabelsPath:
                self._label_dict = Model._harmonize_label_dictionary(
                    json.loads(Path(self._model_parameters.LabelsPath).read_text())
                )
            else:
                self._label_dict = {}

        return self._label_dict if isinstance(self._label_dict, dict) else {}

    @property
    def devices_available(self) -> list:
        """The list of inference device indices which can be used for model inference"""
        from .CoreClient import system_info as core_system_info

        return self._get_device_list(core_system_info())


class _ServerModel(Model):
    """
    AI Server Model class. Handles whole inference lifecycle for single model on remote AI server:
    input data preprocessing, inference and postprocessing.
    """

    @log_wrap
    def __init__(
        self,
        host: str,
        model_name: str,
        model_params: ModelParams,
        supported_device_types: List[str],
        label_dict: Union[None, dict, Callable] = None,
    ):
        """Constructor.

        Args:
            host: Degirum AI server hostname
            model_name: extended for cloud model, simple for local model
            model_params: model parameters
            label_dict: model label dictionary (optional)
        """
        super().__init__(model_name, model_params, supported_device_types)
        self._host = host
        self._runtime = None
        self._label_dict = label_dict

    @log_wrap
    def _init_runtime(self):
        """Initialize runtime object"""
        self._runtime = AIModelAsync(
            self._host,
            self._model_name,
            self._model_parameters,
            self.frame_queue_depth,
            10000,
            int(self.inference_timeout_s * 1000),
        )
        self._model_parameters.dirty = False

    @log_wrap
    @contextmanager
    def _predict_handler(self):
        """Context manager to prepare and wrapup the asynchronous inference process."""

        @log_wrap
        def callback(result, frame_info):
            self._results.append(result)
            with self._result_ready_cv:
                self._result_ready_cv.notify_all()

        if self._runtime is None or self._model_parameters.dirty:
            self._init_runtime()
        assert self._runtime

        self._runtime.observe_output_stream(callback)
        self._runtime.start_run()

        try:
            yield
        finally:
            try:
                self._runtime.stop_run(False)
            finally:
                self._runtime.observe_output_stream(None)

    @property
    def label_dictionary(self) -> Dict[int, str]:
        """Get model class label dictionary"""
        if callable(self._label_dict):
            # request label dictionary and cache it
            self._label_dict = Model._harmonize_label_dictionary(self._label_dict())
        elif self._label_dict is None:
            if self._runtime is None:
                self._init_runtime()
            assert self._runtime
            self._label_dict = Model._harmonize_label_dictionary(
                self._runtime.label_dictionary()
            )

        return self._label_dict if isinstance(self._label_dict, dict) else {}

    @property
    def devices_available(self) -> list:
        """The list of inference device indices which can be used for model inference"""
        return self._get_device_list(server_system_info(self._host))


class _CloudServerModel(Model):
    """
    AI Cloud Server Model class. Handles whole inference lifecycle for single model on remote AI cloud server:
    input data preprocessing, inference and postprocessing.
    """

    class _CloudServerRuntime:
        """Cloud server runtime implementation."""

        @log_wrap
        def __init__(
            self,
            *,
            url: str,
            token: str,
            model_name: str,
            model_params: ModelParams,
            inference_timeout_s: float,
            frame_queue_depth: int = 80,
        ):
            """Constructor.
            -`url`: Degirum AI server hostname
            -`token`: Degirum AI server authorization token
            -`model_name`: extended model name
            -`model_params`: model parameters
            -`inference_timeout_s`: inference timeout in seconds
            -`frame_queue_depth`: the depth of the internal frame queue
            """
            self._url = url
            self._token = token
            self._model_name = model_name
            self._model_parameters = model_params
            self._inference_timeout_s = inference_timeout_s
            self._frame_queue_depth = frame_queue_depth

            self._url_suffix = "/api/v2"
            self._callback = None
            self._frames_in_process: list = []
            self._cv = threading.Condition()
            # Using async client due to issues with sync client's websocket package
            self._sio = socketio.AsyncClient(
                reconnection=False, serializer=MsgPackNumpyPacket
            )
            self._input_frame_number = 0
            self._result_frame_number = 0
            self._socketio_connect_retries = 3
            self._socketio_connect_default_timeout = 5
            self._async_timeout = 5
            self._connection_error_message = None
            self._unexpectedly_closed = True
            self._error_result_state = False
            self._do_reconnect = True
            self._results_dict: dict = {}
            self._thread: Optional[threading.Thread] = None
            self._loop: Optional[asyncio.AbstractEventLoop] = None

            @self._sio.event
            @async_log_wrap
            async def connect():
                logger.info("websocket connection successful")

            @self._sio.event
            @async_log_wrap
            async def disconnect():
                logger.info("websocket disconnect")
                if self._unexpectedly_closed:
                    with self._cv:
                        self._do_reconnect = True
                        self._cv.notify()

            @self._sio.event
            @async_log_wrap
            async def connect_error(message):
                if isinstance(message, str):
                    self._connection_error_message = message
                elif isinstance(message, dict):
                    self._connection_error_message = message.get("message", message)

            @self._sio.on("predict_result")
            @async_log_wrap
            async def results(data, frame_no):
                with self._cv:
                    frame_number = int(frame_no)
                    logger.info(f"Received frame #{frame_no} results")

                    error_msg = dg_check_error(data)
                    if error_msg and "[CRITICAL]" in error_msg:
                        logger.warning(
                            f"Error in frame #{frame_no} results: {error_msg}"
                        )
                        if self._sio.connected:
                            self._sio.disconnect()
                        self._do_reconnect = True
                        self._cv.notify()
                        return

                    if self._result_frame_number > frame_number:
                        data = dict(
                            success=False,
                            msg=f"Result frame numbers mismatch: expected {self._result_frame_number}, processed {frame_number}",
                        )
                        self._result_frame_number = frame_number
                        if self._callback is not None:
                            self._callback(data, "")  # type: ignore[unreachable]
                        self._error_result_state = True
                        self._cv.notify()

                    else:  # self._result_frame_number <= frame_number:
                        self._results_dict[frame_number] = data
                        while self._result_frame_number in self._results_dict:
                            current_data = self._results_dict[self._result_frame_number]
                            del self._results_dict[self._result_frame_number]
                            self._result_frame_number += 1
                            self._frames_in_process.pop(0)
                            if self._callback is not None:
                                self._callback(current_data, "")  # type: ignore[unreachable]
                            if dg_check_error(current_data):
                                self._error_result_state = True
                                break  # in case of error all further results are ignored
                        if self._result_frame_number >= frame_number:
                            self._cv.notify()

            self.start_async_thread()
            # connect to cloud server
            self.reconnect(wait=False)

        # every runtime instance gets its own thread with an event loop
        @log_wrap
        def start_async_thread(self):
            """Start the asyncio event loop in a separate thread"""
            self._loop = asyncio.new_event_loop()

            def run_loop(loop: asyncio.AbstractEventLoop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._thread = threading.Thread(
                target=run_loop, args=(self._loop,), daemon=True
            )
            self._thread.start()

            # Wait for loop to be ready
            while not self._loop.is_running():
                threading.Event().wait(0.01)

        @log_wrap
        def stop_async_thread(self):
            """Stop the asyncio event loop thread"""
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop = None
            if self._thread:
                self._thread.join()
                self._thread = None

        @log_wrap
        def run_coro_sync(self, coro):
            """Run a coroutine from sync context and wait for result"""
            if not self._loop:
                raise RuntimeError("Async thread not started")

            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=self._async_timeout)

        @log_wrap
        def sio_connect(self, url, **kwargs):
            """Connect to server (blocking)"""
            return self.run_coro_sync(self._sio.connect(url, **kwargs))

        @log_wrap
        def sio_emit(self, event, data):
            """Emit event (blocking)"""
            return self.run_coro_sync(self._sio.emit(event, data))

        @log_wrap
        def sio_sleep(self, duration):
            """Sleep (blocking)"""
            return self.run_coro_sync(self._sio.sleep(duration))

        @log_wrap
        def sio_disconnect(self):
            """Disconnect (blocking)"""
            return self.run_coro_sync(self._sio.disconnect())

        @log_wrap
        def sio_wait(self):
            """Wait for the socket connection to be established (blocking)"""
            return self.run_coro_sync(self._sio.wait())

        @log_wrap
        def reconnect(self, *, wait: bool = True) -> bool:
            """Perform server (re)connect

            -`wait`: is it necessary to wait for current session to complete before reconnection

            Return True when reconnect really took place, False otherwise.
            """
            if not self._sio.connected and self._do_reconnect:
                for attempt in range(self._socketio_connect_retries):
                    logger.info(
                        f"Attempting websocket connection to {self._url}{self._url_suffix}/socket.io"
                    )
                    try:
                        name_parts = self._model_name.split("/")
                        if wait:
                            self.sio_wait()
                        self.sio_connect(
                            self._url,
                            socketio_path=f"{self._url_suffix}/socket.io",
                            transports="websocket",
                            wait_timeout=self._socketio_connect_default_timeout,
                            auth=dict(
                                token=self._token,
                                organization=name_parts[0],
                                zoo=name_parts[1],
                                model=name_parts[2],
                                model_params=str(self._model_parameters),
                                frame_queue_depth=self._frame_queue_depth,
                                inference_timeout_s=self._inference_timeout_s,
                                dg_version=pysdk_version,
                            ),
                        )
                    except socketio.exceptions.ConnectionError as e:
                        if attempt == self._socketio_connect_retries - 1:
                            reason = (
                                self._connection_error_message
                                if self._connection_error_message
                                else str(e)
                            )
                            self._error_result_state = True
                            raise DegirumException(
                                f"Unable to open connection to cloud server {self._url.split('://')[-1]}: {reason}"
                            ) from None
                    else:
                        break

                self._do_reconnect = False
                self._input_frame_number = 0
                self._result_frame_number = 0
                for data in self._frames_in_process:
                    self._send_data(data)
                return True
            else:
                return False

        @log_wrap
        def _send_data(self, data):
            """Send frame data to cloud server"""

            logger.info(f"sending frame #{str(self._input_frame_number)}")

            # we pack data to msgpack format
            data_packed = msgpack.packb(data if isinstance(data, list) else [data])

            # packet must start with printable symbol to overcome socketio issue with bytes transfer;
            # also this symbol is used as protocol version number
            packet = (b"0" + data_packed, str(self._input_frame_number))

            try:
                self.sio_emit("predict", packet)
            except Exception:
                # we ignore transport errors here, because they will be handled
                # properly with retries in reconnect(), which is called for every frame
                pass

            self._input_frame_number += 1

        @log_wrap
        def set_callback(self, cb):
            """Install runtime callback

            -`callback`: callback to handle inference result.
            """
            self._callback = cb

        @log_wrap
        def predict(self, frame_data, frame_info):
            """Schedule model prediction using input data"""

            while len(self._frames_in_process) >= self._frame_queue_depth:
                with self._cv:
                    if not self._cv.wait_for(
                        predicate=(
                            lambda: len(self._frames_in_process)
                            < self._frame_queue_depth
                            or self._error_result_state
                            or self._do_reconnect
                        ),
                        timeout=self._inference_timeout_s,
                    ):
                        raise DegirumException(
                            f"Timeout waiting for space in frame queue during inference of model '{self._model_name}' "
                            + f"(timeout = {self._inference_timeout_s} s, queue depth = {self._frame_queue_depth})"
                        )

                    # when error is detected elsewhere, all further frames are ignored
                    if self._error_result_state:
                        return
                self.reconnect()

            self._frames_in_process.append(frame_data)
            self._send_data(frame_data)

        @log_wrap
        def start_run(self):
            """Init current prediction cycle"""
            self._frames_in_process = []

        @property
        def connected(self):
            """Check if runtime currently connected to server or not"""
            return self._sio.connected

        @log_wrap
        def disconnect(self):
            """Close active connection"""
            self.set_callback(None)
            if self._sio:
                self._unexpectedly_closed = False
                if self._sio.connected:
                    self.sio_disconnect()
                self._sio.handlers = {}

        @log_wrap
        def __del__(self):
            """Destructor"""
            self.disconnect()
            self.stop_async_thread()

    @log_wrap
    def __init__(
        self,
        host: str,
        token: str,
        model_name: str,
        model_params: ModelParams,
        supported_device_types: List[str],
        label_dict: Union[None, dict, Callable],
    ):
        """Constructor.

        -`host`: Degirum AI server hostname
        -`token`: Degirum AI server authorization token
        -`model_name`: extended model name
        -`model_params`: model parameters
        -`label_dict`: model label dictionary
        """
        if len(model_name.split("/")) != 3:
            raise DegirumException(
                f"Incorrect model name '{model_name}': it should be in `organization/zoo/model` format"
            )

        super().__init__(model_name, model_params, supported_device_types)
        self._runtime = None
        self._host = host
        self._token = token
        self._label_dict = label_dict

    @log_wrap
    def __del__(self):
        """Destructor"""
        self._release_runtime()

    @log_wrap
    def _init_runtime(self):
        """Initialize runtime object"""

        if self._runtime:
            self._runtime.disconnect()

        self._runtime = _CloudServerModel._CloudServerRuntime(
            url=self._host,
            token=self._token,
            model_name=self._model_name,
            model_params=self._model_parameters,
            inference_timeout_s=self.inference_timeout_s,
            frame_queue_depth=self.frame_queue_depth,
        )
        self._model_parameters.dirty = False

    @log_wrap
    def _reconnect(self):
        """Perform server (re)connect"""
        if self._runtime:
            return self._runtime.reconnect()
        return False

    @log_wrap
    def _release_runtime(self):
        """Release runtime object"""
        if self._runtime and self._runtime.connected:
            self._runtime.disconnect()  # close connection
        self._runtime = None  # release runtime object

    @log_wrap
    @contextmanager
    def _predict_handler(self):
        """Context manager to prepare and wrapup the asynchronous inference process."""

        if (
            self._runtime is None
            or not self._runtime.connected
            or self._model_parameters.dirty
        ):
            self._init_runtime()

        assert self._runtime

        # define callback to handle inference results
        @log_wrap
        def callback(result, frame_info):
            self._results.append(result)
            with self._result_ready_cv:
                self._result_ready_cv.notify_all()

        # install callback
        self._runtime.set_callback(callback)
        # init prediction cycle
        self._runtime.start_run()

        try:
            yield
        finally:
            try:
                self._runtime.set_callback(None)  # remove callback
            finally:
                if not self._in_context:
                    self._release_runtime()  # release runtime

    @property
    def label_dictionary(self) -> Dict[int, str]:
        """Get model class label dictionary"""
        if callable(self._label_dict):
            # request label dictionary and cache it
            self._label_dict = Model._harmonize_label_dictionary(self._label_dict())

        return self._label_dict if isinstance(self._label_dict, dict) else {}

    @property
    def devices_available(self) -> list:
        """The list of inference device indices which can be used for model inference"""
        return list(range(64))  # all devices are always available on the cloud
