#
# _Hand_DetectionResults.py - DeGirum Python SDK: hand detection results postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements Hand_DetectionResults class: hand detection results postprocessor
#

import itertools
import yaml
import math
from typing import Union, List
from ..log import log_wrap
from ._InferenceResults import InferenceResults, _ListFlowTrue
from ..exceptions import DegirumException
from .._draw_primitives import create_draw_primitives


class Hand_DetectionResults(InferenceResults):
    """InferenceResult class implementation for pose detection results type"""

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return "HandDetection"

    @log_wrap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for el in self._inference_results:
            if "landmarks" in el:
                for m in el["landmarks"]:
                    m["landmark"] = [
                        *self._conversion(*m["landmark"][:2]),
                        m["landmark"][2],
                    ]

    __init__.__doc__ = InferenceResults.__init__.__doc__

    def get_distance_color(self, value):
        value = max(-1, min(1, value))
        sigma = 0.5
        offset = 0.6
        red = int(math.exp(-((value - offset) ** 2) / (2 * (sigma) ** 2)) * 256)
        red = max(0, min(255, red))
        green = int(math.exp(-((value) ** 2) / (2 * (sigma) ** 2)) * 256)
        green = max(0, min(255, green))
        blue = int(math.exp(-((value + offset) ** 2) / (2 * (sigma) ** 2)) * 256)
        blue = max(0, min(255, blue))
        return (red, green, blue)

    @property
    def image_overlay(self):
        """Image with AI inference results drawn. Image type is defined by the selected graphical backend."""

        draw = create_draw_primitives(
            self._input_image, self._alpha, self._alpha, self._font_scale
        )
        current_color_set = itertools.cycle(
            self._overlay_color
            if isinstance(self._overlay_color, list)
            else [self._overlay_color]
        )

        if self._input_shape is None:
            raise DegirumException(
                "Hand Detection Postprocessor: input shape is not set. Please set it before using this function."
            )
        _, model_h, _, _ = self._input_shape[0]

        for res in self._inference_results:
            if "landmarks" not in res or "score" not in res or "handedness" not in res:
                continue
            landmarks = res["landmarks"]
            overlay_color = next(current_color_set)
            for landmark in landmarks:
                point = landmark["landmark"]

                # draw lines
                if self._line_width > 0:
                    for neighbor in landmark["connect"]:
                        point2 = landmarks[neighbor]["landmark"]
                        draw.draw_line(
                            point[0],
                            point[1],
                            point2[0],
                            point2[1],
                            self._line_width,
                            overlay_color,
                        )

                    point_color = self.get_distance_color(point[2] / model_h * 3)

                    # then draw point
                    draw.draw_circle(
                        point[0],
                        point[1],
                        2 * self._line_width,
                        self._line_width,
                        point_color,
                        fill=True,
                    )

                str = ""
                # draw probabilities on wrist only
                if self._show_labels:
                    str = landmark["label"]
                    if self._show_probabilities and landmark["label"] == "Wrist":
                        str = f"{str}:{InferenceResults._format_num(res['score'])},"
                        if res["handedness"] > 0.5:
                            str = f"{str} right:{res['handedness']:5.2f}"
                        else:
                            str = f"{str} left:{(1 - res['handedness']):5.2f}"
                elif self._show_probabilities and landmark["label"] == "Wrist":
                    str = f"{InferenceResults._format_num(res['score'])},"
                    if res["handedness"] > 0.5:
                        str = f"{str} right:{res['handedness']:5.2f}"
                    else:
                        str = f"{str} left:{(1 - res['handedness']):5.2f}"

                if str != "":
                    spacer = 3 * self._line_width
                    draw.draw_text_label(
                        point[0] + spacer,
                        point[1] - spacer,
                        point[0] + spacer,
                        point[1] + spacer,
                        overlay_color,
                        str,
                        self._line_width,
                    )

        return draw.image_overlay()

    def __str__(self):
        """
        Convert inference results to string
        """

        def landmarks(marks):
            return [
                dict(
                    label=m["label"],
                    category_id=m["category_id"],
                    landmark=_ListFlowTrue(m["landmark"]),
                    world_landmark=_ListFlowTrue(m["world_landmark"]),
                    connect=_ListFlowTrue([marks[e]["label"] for e in m["connect"]]),
                )
                for m in marks
            ]

        res_list = []
        for el in self._inference_results:
            d = {}
            if "score" in el:
                d["score"] = el["score"]
            if "handedness" in el:
                d["handedness"] = el["handedness"]
            if "landmarks" in el:
                d["landmarks"] = landmarks(el["landmarks"])
            res_list.append(d)

        return yaml.dump(res_list, sort_keys=False)
