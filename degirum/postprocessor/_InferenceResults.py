#
# _InferenceResults.py - DeGirum Python SDK: base class for postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements InferenceResults class to be used as a base class for all postprocessors
#

import copy
import numpy
import yaml
from typing import Union, Optional, List
from collections.abc import Iterable
from ..log import log_wrap
from ..exceptions import validate_color_tuple


class _ListFlowTrue(list):
    """list subclass to specify custom yaml style"""


# add custom representer for list type with flow_style=True
yaml.add_representer(
    _ListFlowTrue,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


class InferenceResults:
    """Inference results container class.

    This class is a base class for a set of classes designed to handle
    inference results of particular model types such as classification, detection etc.

    !!! note

        You never construct model objects yourself. Objects of those classes are returned by various predict
        methods of [degirum.model.Model][] class.
    """

    @log_wrap
    def __init__(
        self,
        *,
        inference_results,
        conversion,
        input_image=None,
        model_image=None,
        draw_color=(255, 255, 128),
        line_width: int = 3,
        show_labels: bool = True,
        show_probabilities: bool = False,
        alpha: Union[float, str] = "auto",
        font_scale: float = 1.0,
        fill_color=(0, 0, 0),
        blur: Union[str, list, None] = None,
        frame_info=None,
        label_dictionary: dict = {},
        input_shape: Optional[list] = None,
    ):
        """Constructor.

        !!! note

            You never construct `InferenceResults` objects yourself -- the ancestors of this class are returned
            as results of AI inferences from [degirum.model.Model.predict][], [degirum.model.Model.predict_batch][],
            and [degirum.model.Model.predict_dir][] methods.

        Args:
            inference_results (list): Inference results data.
            conversion (Callable): Coordinate conversion function accepting two arguments `(x,y)` and returning two-element tuple.
                This function should convert model-based coordinates to input image coordinates.
            input_image (any): Original input data.
            model_image (any): Input data converted per AI model input specifications.
            draw_color (tuple): Color for inference results drawing on overlay image.
            line_width: Line width in pixels for inference results drawing on overlay image.
            show_labels: True to draw class labels on overlay image.
            show_probabilities: True to draw class probabilities on overlay image.
            alpha: Alpha-blend weight for overlay details.
            font_scale: Font scale to use for overlay text.
            fill_color (tuple): RGB color tuple to use for filling if any form of padding is used.
            blur: Optional blur parameter to apply to the overlay image. If None, no blur is applied. If "all"
                all objects are blurred. If a class label or a list of class labels is provided, only objects with
                those labels are blurred.
            frame_info (any): Input data frame information object.
            label_dictionary (dict[str, str]): Model label dictionary.
            input_shape(list): Model input shape. Mandatory for image-type postprocessing.

        """
        self.timing: dict = {}
        self._inference_results = copy.deepcopy(inference_results)
        self._conversion = conversion
        self._input_image = input_image
        self._model_image = model_image
        self._overlay_color = draw_color
        self._show_labels = show_labels
        self._show_labels_below = True
        self._show_probabilities = show_probabilities
        self._line_width = line_width
        self._alpha = 1.0 if alpha == "auto" else float(alpha)
        self._font_scale = font_scale
        self._fill_color = fill_color
        self._blur = blur
        self._frame_info = frame_info
        self._label_dictionary = label_dictionary
        self._input_shape = input_shape

    def __str__(self):
        """Conversion to string"""
        return str(self._inference_results)

    def __repr__(self):
        return self.__str__()

    def __dir__(self):
        return [
            "image",
            "image_model",
            "image_overlay",
            "info",
            "results",
            "timing",
        ]

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return ["None", "Base"]

    @property
    def image_overlay(self):
        """Image with AI inference results drawn on a top of original image.

        Drawing details depend on the inference result type:

        - For classification models the list of class labels with probabilities is printed below the original image.
        - For object detection models bounding boxes of detected object are drawn on the original image.
        - For pose detection models detected keypoints and keypoint connections are drawn on the original image.
        - For segmentation models detected segments are drawn on the original image.

        Returned image object type is defined by the selected graphical backend (see [degirum.model.Model.image_backend][]).
        """
        return copy.deepcopy(self._input_image)

    @property
    def image(self):
        """Original image.

        Returned image object type is defined by the selected graphical backend (see [degirum.model.Model.image_backend][]).
        """
        return self._input_image

    @property
    def image_model(self):
        """Model input image data: image converted to AI model input specifications.

        Image type is raw binary array."""
        return self._model_image

    @property
    def results(self) -> list:
        """Inference results list.

        Each element of the list is a dictionary containing information about one inference result.
        The dictionary contents depends on the AI model.


        **For classification models** each inference result dictionary contains the following keys:

        - `category_id`: class numeric ID.
        - `label`: class label string.
        - `score`: class probability.

        Example:
            ```json
            [
                {'category_id': 0, 'label': 'cat', 'score': 0.99},
                {'category_id': 1, 'label': 'dog', 'score': 0.01}
            ]
            ```

        **For multi-label classification models** each inference result dictionary contains the following keys:

        - `classifier`: object class string.
        - `results`: list of class labels and its scores. Scores are optional.

        The `results` list element is a dictionary with the following keys:

        - `label`: class label string.
        - `score`: optional class label probability.

        Example:
            ```json
            [
                {
                    'classifier': 'vehicle color',
                    'results': [
                        {'label': 'red', 'score': 0.99},
                        {'label': 'blue', 'score': 0.01}
                     ]
                },
                {
                    'classifier': 'vehicle type',
                    'results': [
                        {'label': 'car', 'score': 0.99},
                        {'label': 'truck', 'score': 0.01}
                    ]
                }
            ]
            ```


        **For object detection models** each inference result dictionary may contain the following keys:

        - `category_id`: detected object class numeric ID.
        - `label`: detected object class label string.
        - `score`: detected object probability.
        - `bbox`: detected object bounding box list `[xtop, ytop, xbot, ybot]`.
        - `landmarks`: optional list of keypoints or landmarks. It is the list of dictionaries, one per each keypoint/landmark.
        - `mask`: optinal dictionary of run-length encoded (RLE) object segmentation mask array representation.
        - `angle`: optional angle (in radians) for rotating bounding box around its center. This is used in the case of oriented bounding boxes.

        The `landmarks` list is defined for special cases like pose detection of face points detection results.
        Each `landmarks` list element is a dictionary with the following keys:

        - `category_id`: keypoint numeric ID.
        - `label`: keypoint label string.
        - `score`: keypoint detection probability.
        - `landmark`: keypoint coordinate list `[x,y,visibility,a,b,...]`.
        - `connect`: optional list of IDs of connected keypoints.

        The `mask` dictionary is defined for the special case of object segmentation results, with the following keys:

        - `x_min`: x-coordinate in the model input image at which the top-left corner of the box enclosing this mask should be placed.
        - `y_min`: y-coordinate in the model input image at which the top-left corner of the box enclosing this mask should be placed.
        - `height`: height of segmentation mask array
        - `width`: width of segmentation mask array
        - `data`: string representation of a buffer of unsigned 32-bit integers carrying the RLE segmentation mask array.

        The object detection keys (`bbox`, `score`, `label`, and `category_id`) must be either all present or all absent.
        In the former case the result format is suitable to represent pure object detection results.
        In the later case, one of the following keys must be present:

        - the `landmarks` key
        - the `mask` key

        The following statements are then true:

        - If the `landmarks` key is present, the result format is suitable to represent pure landmark detection results, such as pose detection.
        - If the `mask` key is present, the result format is suitable to represent pure segmentation results. If, optionally,
            the `category_id` key is also present, the result format is suitable to represent semantic segmentation results.

        When both object detection keys and the `landmarks` key are present, the result format is suitable to represent mixed model results,
        when the model detects not only object bounding boxes, but also keypoints/landmarks within the bounding box.

        When both object detection keys and the `mask` key are present, the result format is suitable to represent mixed model results,
        when the model detects not only object bounding boxes, but also segmentation masks within the bounding box (i.e. instance segmentation).

        Example of pure object detection results:

        Example:
            ```json
            [
                {'category_id': 0, 'label': 'cat', 'score': 0.99, 'bbox': [10, 20, 100, 200]},
                {'category_id': 1, 'label': 'dog', 'score': 0.01, 'bbox': [200, 100, 300, 400]}
            ]
            ```

        Example of oriented object detection results:

        Example:
            ```json
            [
                {'category_id': 0, 'label': 'car', 'score': 0.99, 'bbox': [10, 20, 100, 200], 'angle': 0.79}
            ]
            ```

        Example of landmark object detection results:

        Example:
            ```json
            [
                {
                    'landmarks': [
                        {'category_id': 0, 'label': 'Nose', 'score': 0.99, 'landmark': [10, 20]},
                        {'category_id': 1, 'label': 'LeftEye', 'score': 0.98, 'landmark': [15, 25]},
                        {'category_id': 2, 'label': 'RightEye', 'score': 0.97, 'landmark': [18, 28]}
                    ]
                }
            ]
            ```

        Example of segmented object detection results:

        Example:
            ```json
            [
                {
                    'mask': {'x_min': 1, 'y_min': 1, 'height': 2, 'width': 2, 'data': 'AAAAAAEAAAAAAAAAAQAAAAIAAAABAAAA'}
                }
            ]
            ```

        **For hand palm detection** models each inference result dictionary contains the following keys:

        - `score`: probability of detected hand.
        - `handedness`: probability of right hand.
        - `landmarks`: list of dictionaries, one per each hand keypoint.

        Each `landmarks` list element is a dictionary with the following keys:

        - `label`: classified object class label.
        - `category_id`: classified object class index.
        - `landmark`: landmark point coordinate list `[x, y, z]`.
        - `world_landmark`: metric world landmark point coordinate list `[x, y, z]`.
        - `connect`: list of adjacent landmarks indexes.

        Example:
            ```json
            [
                {
                    'score': 0.99,
                    'handedness': 0.98,
                    'landmarks': [
                        {
                            'label': 'Wrist',
                            'category_id': 0,
                            'landmark': [10, 20, 30],
                            'world_landmark': [10, 20, 30],
                            'connect': [1]
                        },
                        {
                            'label': 'Thumb',
                            'category_id': 1,
                            'landmark': [15, 25, 35],
                            'world_landmark': [15, 25, 35],
                            'connect': [0]
                        }
                    ]
                }
            ]
            ```

        **For segmentation models** inference result is a single-element list. That single element is a dictionary,
        containing single key `data`. The value of this key is 2D numpy array of integers, where each integer value
        represents a class ID of the corresponding pixel. The class IDs are defined by the model label dictionary.

        Example:
            ```json
            [
                {
                    'data': numpy.array([
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [2, 2, 2, 3, 3, 3],
                        [2, 2, 2, 3, 3, 3],
                        [2, 2, 2, 3, 3, 3],
                    ])
                }
            ]
            ```

        """
        return self._inference_results

    @property
    def overlay_color(self):
        """Color for inference results drawing on overlay image.

        3-element RGB tuple or list of 3-element RGB tuples."""
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
    def overlay_show_labels(self) -> bool:
        """Specifies if class labels should be drawn on overlay image."""
        return self._show_labels

    @overlay_show_labels.setter
    def overlay_show_labels(self, val):
        self._show_labels = val

    @property
    def overlay_show_probabilities(self) -> bool:
        """Specifies if class probabilities should be drawn on overlay image."""
        return self._show_probabilities

    @overlay_show_probabilities.setter
    def overlay_show_probabilities(self, val):
        self._show_probabilities = val

    @property
    def overlay_line_width(self) -> int:
        """Line width in pixels for inference results drawing on overlay image."""
        return self._line_width

    @overlay_line_width.setter
    def overlay_line_width(self, val):
        self._line_width = val

    @property
    def overlay_alpha(self) -> float:
        """Alpha-blend weight for overlay details."""
        return self._alpha

    @overlay_alpha.setter
    def overlay_alpha(self, val: float):
        self._alpha = val
        if hasattr(self, "_segm_alpha"):
            self._segm_alpha = val

    @property
    def overlay_blur(self) -> Union[str, list, None]:
        """Overlay blur option. None for no blur, "all" to blur all objects, a class label or list of class
        labels to blur specific objects."""
        return self._blur

    @overlay_blur.setter
    def overlay_blur(self, val: Union[str, list, None]):
        self._blur = val

    @property
    def overlay_font_scale(self) -> float:
        """Font scale to use for overlay text."""
        return self._font_scale

    @overlay_font_scale.setter
    def overlay_font_scale(self, val):
        self._font_scale = val

    @property
    def overlay_fill_color(self) -> tuple:
        """Image fill color in case of image padding.

        3-element RGB tuple."""
        return self._fill_color

    @overlay_fill_color.setter
    def overlay_fill_color(self, val: tuple):
        self._fill_color = validate_color_tuple(val)

    @property
    def info(self):
        """Input data frame information object."""
        return self._frame_info

    @staticmethod
    def generate_colors():
        """Generate a list of unique RGB color tuples."""
        bits = lambda n, f: numpy.array(
            list(numpy.binary_repr(n, 24)[f::-3]), numpy.uint8
        )
        return [
            (
                int(numpy.packbits(bits(x, -3)).item()),
                int(numpy.packbits(bits(x, -2)).item()),
                int(numpy.packbits(bits(x, -1)).item()),
            )
            for x in range(256)
        ]

    @staticmethod
    def generate_overlay_color(num_classes, label_dict) -> Union[list, tuple]:
        """Overlay colors generator.

        Args:
            num_classes (int): number of class categories.
            label_dict (dict): Model labels dictionary.

        Returns:
            Overlay color tuple or list of tuples.
        """
        return (255, 255, 0)

    @staticmethod
    def _format_num(number, precision=2):
        "Return formatted number based on type of numeric value"
        if isinstance(number, int):
            return str(number)
        else:
            return f"{number:.{precision}f}"
