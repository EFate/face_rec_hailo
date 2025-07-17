#
# _ClassificationResults.py - DeGirum Python SDK: classification results postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements ClassificationResults class: classification results postprocessor
#

import itertools
import yaml
from typing import Union, List
from ..log import log_wrap
from ._InferenceResults import InferenceResults
from .._draw_primitives import create_draw_primitives


class ClassificationResults(InferenceResults):
    """InferenceResult class implementation for classification results type"""

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return ["Classification", "DetectionYoloPlates", "DetectionYoloV8Plates"]

    @log_wrap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __init__.__doc__ = InferenceResults.__init__.__doc__

    def __dir__(self):
        return super().__dir__() + ["overlay_show_labels_below"]

    @property
    def overlay_show_labels_below(self):
        """Specifies if overlay labels should be drawn below the image or on image itself"""
        return self._show_labels_below

    @overlay_show_labels_below.setter
    def overlay_show_labels_below(self, val):
        self._show_labels_below = val

    @property
    def image_overlay(self):
        """Image with AI inference results drawn. Image type is defined by the selected graphical backend.
        Each time this property is accessed, new overlay image object is created and all overlay details
        are redrawn according to the current settings of overlay_*** properties.
        """
        prev_bbox = (0, 0, 0, 0)
        spacer = 3

        def get_string():
            for res in self._inference_results:
                if "label" not in res or "score" not in res:
                    continue
                if self._show_labels and self._show_probabilities:
                    str = (
                        f"{res['label']}: {InferenceResults._format_num(res['score'])}"
                    )
                elif self._show_labels:
                    str = res["label"]
                elif self._show_probabilities:
                    str = InferenceResults._format_num(res["score"])
                else:
                    str = ""
                yield str

        draw = create_draw_primitives(
            self._input_image, self._alpha, self._alpha, self._font_scale
        )
        if self._show_labels_below and (self._show_labels or self._show_probabilities):
            w = 0
            h = 0
            for label in get_string():
                lw, lh, _ = draw.text_size(label)
                w = max(w, 2 * spacer + lw)
                h += spacer + lh
            if h > 0:
                h += spacer
            w, h = draw.image_overlay_extend(w, h, self._fill_color)
            prev_bbox = (0, 0, 0, h)

        current_color_set = itertools.cycle(
            self._overlay_color
            if isinstance(self._overlay_color, list)
            else [self._overlay_color]
        )
        if self._show_labels or self._show_probabilities:
            for label in get_string():
                overlay_color = next(current_color_set)
                prev_bbox = draw.draw_text(
                    spacer, prev_bbox[3] + spacer, overlay_color, label
                )

        return draw.image_overlay()

    def __str__(self):
        """
        Convert inference results to string
        """
        res_list = []
        for el in self._inference_results:
            d = {}
            if "label" in el:
                d["label"] = el["label"]
            if "score" in el:
                d["score"] = el["score"]
            if "category_id" in el:
                d["category_id"] = el["category_id"]
            res_list.append(d)
        return yaml.safe_dump(res_list, sort_keys=False)
