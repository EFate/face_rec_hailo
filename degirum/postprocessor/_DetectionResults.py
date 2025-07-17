#
# _DetectionResults.py - DeGirum Python SDK: detection results postprocessor
# Copyright DeGirum Corp. 2025
#
# Implements DetectionResults class: detection results postprocessor
#

import itertools
import yaml
import numpy
import base64
import copy
from typing import Union, List
from ..log import log_wrap
from ._InferenceResults import InferenceResults, _ListFlowTrue
from ..exceptions import DegirumException
from .._draw_primitives import (
    create_draw_primitives,
    _inv_conversion_calc,
    xyxy2xywh,
    xywhr2xy_corners,
)


class DetectionResults(InferenceResults):
    """InferenceResult class implementation for detection results type"""

    @staticmethod
    def supported_types() -> Union[str, List[str]]:
        """List supported result types for this postprocessor."""
        return [
            "Detection",
            "DetectionYolo",
            "DetectionYoloV8",
            "DetectionYoloV10",
            "DetectionYoloV8OBB",
            "DetectionYoloHailo",
            "DetectionDamoYolo",
            "FaceDetection",
            "PoseDetection",
            "PoseDetectionYoloV8",
            "SegmentationYoloV8",
        ]

    max_colors = 255

    @log_wrap
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_result = None
        self.masks_present = False
        if "alpha" in kwargs and kwargs["alpha"] == "auto":
            self._segm_alpha = 0.5
        else:
            self._segm_alpha = self._alpha
        for res in self._inference_results:
            if "bbox" in res:
                box = res["bbox"]
                res["bbox"] = [
                    *self._conversion(*box[:2]),
                    *self._conversion(*box[2:]),
                ]
            if "landmarks" in res:
                for m in res["landmarks"]:
                    m["landmark"] = [
                        *self._conversion(*m["landmark"][:2]),
                    ]
            if "mask" in res:
                if not self.masks_present:
                    self.masks_present = True
                res["mask"] = numpy.array(self._run_length_decode(res["mask"])).astype(
                    numpy.float32
                )
                res["mask"] = self._resize_mask(res["mask"])

    __init__.__doc__ = InferenceResults.__init__.__doc__

    def _run_length_decode(self, rle):
        """
        Returns NumPy array for run length encoded string, reshaped to specified shape

        Args:
            rle (dict): RLE image segmentation mask dictionary

        Returns:
            results (NumPy array): NumPy array, representing an image segmentation mask.
        """

        if self._input_shape is None:
            raise DegirumException(
                "Detection Postprocessor: input shape is not set. Please set it before using this function."
            )
        _, model_h, model_w, _ = self._input_shape[0]

        x_min = rle.get("x_min", 0)
        y_min = rle.get("y_min", 0)
        height = rle.get("height", model_h)
        width = rle.get("width", model_w)
        rle_array = numpy.frombuffer(base64.b64decode(rle["data"]), dtype=numpy.uint32)
        N = len(rle_array) // 2
        mask = numpy.repeat(rle_array[:N], rle_array[N:]).reshape((height, width))
        mask_out = numpy.zeros((model_h, model_w), dtype=mask.dtype)
        mask_out[y_min : y_min + height, x_min : x_min + width] = mask
        return mask_out

    def _resize_mask(self, mask):
        """Return image segmentation mask scaled with respect to the provided image transformation callback

        - `conversion`: coordinate conversion function accepting two arguments (x,y) and returning two-element tuple
        - `overlay_data`: overlay data to blend on top of input image
        - `original_width`: original image width
        - `original_height`: original image height
        """
        import cv2

        # map corners from original image to model output
        original_height, original_width = (
            self._input_image.shape[:2]
            if hasattr(self._input_image, "shape")
            else (self._input_image.height, self._input_image.width)
        )
        height, width = mask.shape
        inv_conversion = _inv_conversion_calc(self._conversion, width, height)
        p1 = [int(round(i)) for i in inv_conversion(0, 0)]
        p2 = [int(round(i)) for i in inv_conversion(original_width, original_height)]

        img = mask[
            max(p1[1], 0) : min(p2[1], height), max(p1[0], 0) : min(p2[0], width)
        ]

        # add padding to cropped image
        if p1[0] < 0 or p1[1] < 0 or p2[0] > width or p2[1] > height:
            background = numpy.zeros((p2[1] - p1[1], p2[0] - p1[0]), img.dtype)
            background[
                abs(p1[1]) : (abs(p1[1]) + img.shape[0]),
                abs(p1[0]) : (abs(p1[0]) + img.shape[1]),
            ] = img
            img = background

        mask = cv2.resize(
            img, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )
        return mask

    # deduce color based on category_id or label
    def _deduce_color(self, id, label, current_color_set):
        if label is None:
            if id is None:
                # both label and id are missing: simply use next color
                return next(current_color_set)
            else:
                # label is missing, but id is there: use id
                pass
        else:
            if id is None:
                # id is missing: use label hash
                id = hash(label)
            else:
                if (
                    self._label_dictionary is not None
                    and self._label_dictionary.get(id, "") != label
                ):
                    # label with this id is not in dictionary
                    id = self._label_dictionary.get(label, None)
                    if id is None:
                        # label is not in dictionary: add it in reverse lookup manner assigning new unique id
                        id = (
                            (
                                1
                                + max(
                                    (
                                        k
                                        if isinstance(k, int)
                                        else (v if isinstance(v, int) else 0)
                                    )
                                    for k, v in self._label_dictionary.items()
                                )
                            )
                            if self._label_dictionary
                            else 0
                        )
                        self._label_dictionary[label] = id
                    else:
                        # id is in dictionary in reverse lookup manner: use it
                        pass
                else:
                    # both label and id are there, and label matches id: use id
                    pass

        return self._overlay_color[id % len(self._overlay_color)]

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
                    "Detection Postprocessor: either non empty labels dictionary or OutputNumClasses greater than 0 must be specified for Detection postprocessor"
                )
            colors = colors[1 : num_classes + 1]

        else:
            if any(not isinstance(k, int) for k in label_dict.keys()):
                raise DegirumException(
                    "Detection Postprocessor: non integer keys in label dictionary are not supported"
                )
            if any(k < 0 for k in label_dict.keys()):
                raise DegirumException(
                    "Detection Postprocessor: label key values must be greater than 0"
                )
            colors = colors[1 : len(label_dict) + 1]
        colors[0] = (255, 255, 0)
        return colors

    @property
    def image_overlay(self):
        """Image with AI inference results drawn. Image type is defined by the selected graphical backend."""
        segm_alpha = self._segm_alpha if self.masks_present else None
        draw = create_draw_primitives(
            self._input_image, self._alpha, segm_alpha, self._font_scale
        )
        many_colors = isinstance(self._overlay_color, list)
        current_color_set = itertools.cycle(
            self._overlay_color if many_colors else [self._overlay_color]
        )

        line_width = self._line_width
        show_labels = self._show_labels
        show_probabilities = self._show_probabilities
        spacer = 3 * line_width

        for res in self._inference_results:
            label = res.get("label", None)
            id = res.get("category_id", None)
            overlay_color = (
                self._deduce_color(id, label, current_color_set)
                if many_colors
                else self._overlay_color
            )

            # draw bounding boxes
            box = res.get("bbox", None)
            if box is not None:
                angle = res.get("angle", None)
                is_rotated_bbox = angle is not None

                # apply blur first (if needed)
                if not is_rotated_bbox and (
                    (
                        isinstance(self._blur, str)
                        and (self._blur == "all" or label == self._blur)
                    )
                    or (isinstance(self._blur, list) and label in self._blur)
                ):
                    draw.blur_box(*box)

                if line_width > 0:
                    if not is_rotated_bbox:
                        draw.draw_box(*box, line_width, overlay_color)
                    else:
                        xywhr = numpy.array(xyxy2xywh(numpy.array(box)).tolist() + [angle])
                        poly = xywhr2xy_corners(xywhr).astype(int)
                        draw.draw_polygon(poly, line_width, overlay_color)
                        box = poly[[0, 2]].flatten().tolist()

                show_labels_and_label_is_not_none = show_labels and label is not None
                capt = label if show_labels_and_label_is_not_none else ""
                if show_probabilities:
                    score = res.get("score", None)
                    if score is not None:
                        capt = (
                            f"{label}: {InferenceResults._format_num(score)}"
                            if show_labels_and_label_is_not_none
                            else InferenceResults._format_num(score)
                        )

                if capt != "":
                    draw.draw_text_label(
                        *box, overlay_color, capt, line_width, is_rotated_bbox
                    )

            # draw landmarks
            landmarks = res.get("landmarks", None)
            if landmarks is not None:
                for landmark in landmarks:
                    point = landmark["landmark"]
                    if line_width > 0:
                        draw.draw_circle(
                            point[0],
                            point[1],
                            2,
                            line_width,
                            overlay_color,
                            True,
                        )

                        lm_connect = landmark.get("connect", None)
                        if lm_connect is not None:
                            for neighbor in lm_connect:
                                point2 = landmarks[neighbor]["landmark"]
                                draw.draw_line(
                                    point[0],
                                    point[1],
                                    point2[0],
                                    point2[1],
                                    line_width,
                                    overlay_color,
                                )

                    lm_label = landmark.get("label", None)
                    show_lm_labels = show_labels and lm_label is not None
                    capt = lm_label if show_lm_labels else ""
                    if show_probabilities:
                        score = landmark.get("score", None)
                        if score is not None:
                            capt = (
                                f"{lm_label}: {InferenceResults._format_num(score)}"
                                if show_lm_labels
                                else InferenceResults._format_num(score)
                            )

                    if capt != "":
                        draw.draw_text_label(
                            point[0] + spacer,
                            point[1] - spacer,
                            point[0] + spacer,
                            point[1] + spacer,
                            overlay_color,
                            capt,
                            line_width,
                        )

            # collect segmentation masks
            mask = res.get("mask", None)
            if mask is not None:
                if self.mask_result is None:
                    self.mask_result = numpy.zeros(mask.shape).astype(numpy.uint8)
                category_id = res.get("category_id", 0)
                render_mask = (mask > 0).astype(numpy.uint8) * (
                    category_id % DetectionResults.max_colors + 1
                )
                merge_condition = render_mask != 0
                self.mask_result = numpy.where(
                    merge_condition, render_mask, self.mask_result
                )

        # draw segmentation masks
        if self.mask_result is not None:
            lut = numpy.empty((256, 1, 3), dtype=numpy.uint8)
            lut[:, :, :] = (0, 0, 0)  # default non-mask color
            for i in range(1, 256):
                lut[i, :, :] = next(current_color_set)

            draw.image_segmentation_overlay(
                self._conversion, self.mask_result, lut, convert=False
            )

        return draw.image_overlay()

    def __str__(self):
        """
        Convert inference results to string
        """
        results = copy.deepcopy(self._inference_results)
        for res in results:
            if "bbox" in res:
                res["bbox"] = _ListFlowTrue(res["bbox"])
            if "landmarks" in res:
                for lm in res["landmarks"]:
                    if "landmark" in lm:
                        lm["landmark"] = _ListFlowTrue(lm["landmark"])
                        if "connect" in lm:
                            if "label" in lm:
                                lm["connect"] = _ListFlowTrue(
                                    [
                                        res["landmarks"][e]["label"]
                                        for e in lm["connect"]
                                    ]
                                )
                            else:
                                lm["connect"] = _ListFlowTrue(lm["connect"])
            if "mask" in res:
                del res["mask"]

        return yaml.dump(results, sort_keys=False)
