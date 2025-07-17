#
# _SegmentationResults.py - DeGirum Python SDK: global semantic segmentation results postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements SegmentationResults class: global semantic segmentation results postprocessor
#

import itertools
import yaml
import numpy
from typing import Union, List
from ..log import log_wrap
from ._InferenceResults import InferenceResults
from ..exceptions import DegirumException
from .._draw_primitives import create_draw_primitives


class SegmentationResults(InferenceResults):
    """InferenceResult class implementation for segmentation results type"""

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return "Segmentation"

    max_colors = 256

    @log_wrap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "alpha" in kwargs and kwargs["alpha"] == "auto":
            self._alpha = 0.5

        if not isinstance(self._inference_results, list):
            raise DegirumException(
                "Segmentation Postprocessor: inference results data must be a list"
            )
        if len(self._inference_results) != 1:
            raise DegirumException(
                "Segmentation Postprocessor: inference results data must contain one element"
            )
        if not isinstance(self._inference_results[0], dict):
            raise DegirumException(
                "Segmentation Postprocessor: inference results data element must be a dictionary"
            )
        if "data" not in self._inference_results[0]:
            raise DegirumException(
                "Segmentation Postprocessor: inference results data element dictionary must contain 'data' key"
            )
        if not isinstance(self._inference_results[0]["data"], numpy.ndarray):
            raise DegirumException(
                "Segmentation Postprocessor: inference results 'data' value must be numpy.ndarray"
            )

    __init__.__doc__ = InferenceResults.__init__.__doc__

    @staticmethod
    def generate_overlay_color(num_classes, label_dict) -> list:
        """Overlay colors generator.

        Args:
            num_classes (int): number of class categories.
            label_dict (dict): Model labels dictionary.

        Returns:
            general overlay color data for segmentation results
        """
        colors = InferenceResults.generate_colors()
        if not label_dict:
            if num_classes <= 0:
                raise DegirumException(
                    "Segmentation Postprocessor: either non empty labels dictionary or OutputNumClasses greater than 0 must be specified for Segmentation postprocessor"
                )
            return colors[:num_classes]

        else:
            if any(not isinstance(k, int) for k in label_dict.keys()):
                raise DegirumException(
                    "Segmentation Postprocessor: non integer keys in label dictionary are not supported"
                )
            if any(
                k < 0 or k > SegmentationResults.max_colors for k in label_dict.keys()
            ):
                raise DegirumException(
                    f"Segmentation Postprocessor: label key values must be within [0, {SegmentationResults.max_colors}] range"
                )
            colors = colors[: len(label_dict)]
            for k, v in label_dict.items():
                if v == "background":
                    colors.insert(k, (0, 0, 0))  # default non-mask color for background
                    colors.pop(0)
            return colors

    @property
    def image_overlay(self):
        """Image with AI inference results drawn. Image type is defined by the selected graphical backend."""

        draw = create_draw_primitives(
            self._input_image, self._alpha, self._alpha, self._font_scale
        )
        result = (
            numpy.copy(self._inference_results[0]["data"]).squeeze().astype(numpy.uint8)
        )
        lut = numpy.empty((256, 1, 3), dtype=numpy.uint8)
        lut[:, :, :] = (0, 0, 0)  # default non-mask color
        current_color_set = itertools.cycle(
            self._overlay_color
            if isinstance(self._overlay_color, list)
            else [self._overlay_color]
        )
        for i in range(256):
            lut[i, :, :] = next(current_color_set)

        draw.image_segmentation_overlay(self._conversion, result, lut)
        return draw.image_overlay()

    def __str__(self):
        """
        Convert inference results to string
        """
        res = self._inference_results[0]["data"]
        res_list = {
            "segments": ", ".join(
                self._label_dictionary.get(i, str(i)) for i in numpy.unique(res)
            ),
        }
        return yaml.dump(res_list, sort_keys=False)
