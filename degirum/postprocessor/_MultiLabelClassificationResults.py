#
# _MultiLabelClassificationResults.py - DeGirum Python SDK: multi-label classification results postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements MultiLabelClassificationResults class: multi-label classification results postprocessor
#

import itertools
import yaml
from typing import Union, List
from ..log import log_wrap
from ._InferenceResults import InferenceResults
from .._draw_primitives import create_draw_primitives


class MultiLabelClassificationResults(InferenceResults):
    """InferenceResult class implementation for multi-label classification results type"""

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return "MultiLabelClassification"

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
        indentation = 2

        def get_string():
            for result in self._inference_results:
                if "classifier" not in result or "results" not in result:
                    continue
                str = f"{result['classifier']}:"
                for labels in result["results"]:
                    if "label" not in labels:
                        continue
                    if (
                        "score" in labels
                        and self._show_probabilities
                        and self._show_labels
                    ):
                        str += f"\n{' ' * indentation}{labels['label']}: {InferenceResults._format_num(labels['score'])}"
                    elif self._show_labels:
                        str += f"\n{' ' * indentation}{labels['label']}"
                    elif self._show_probabilities:
                        str += f"\n{' ' * indentation}{InferenceResults._format_num(labels['score'])}"
                    else:
                        str = ""
                yield str

        draw = create_draw_primitives(
            self._input_image, self._alpha, self._alpha, self._font_scale
        )
        if self._show_labels_below and (self._show_labels or self._show_probabilities):
            w = 0
            h = 0
            for lbs in get_string():
                labels_list = lbs.split("\n")
                for res_string in labels_list:
                    lw, lh, _ = draw.text_size(res_string)
                    w = max(w, spacer + lw)
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
            for lbs in get_string():
                labels_list = lbs.split("\n")
                for res_string in labels_list:
                    overlay_color = next(current_color_set)
                    prev_bbox = draw.draw_text(
                        spacer, prev_bbox[3] + spacer, overlay_color, res_string
                    )
        return draw.image_overlay()

    def __str__(self):
        """
        Convert inference results to string
        """
        res_list = []
        for el in self._inference_results:
            d = {}
            if "classifier" in el:
                d["classifier"] = el["classifier"]
            if "results" in el:
                d["results"] = el["results"]
                for labels in el:
                    if "label" in labels:
                        d["label"] = labels["label"]
                    if "score" in labels:
                        d["score"] = labels["score"]
            res_list.append(d)
        return yaml.safe_dump(res_list, sort_keys=False)
