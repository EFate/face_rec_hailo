#
# _draw_primitives.py - DeGirum Python SDK: draw primitives for result postprocessing
# Copyright DeGirum Corp. 2022
#
# Implements draw primitive classes to handle different types of image manipulations
#

import importlib
import importlib.util
import numpy
from pathlib import Path
from contextlib import contextmanager
from fractions import Fraction
from .exceptions import DegirumException


class GlobalOptions:
    """Global options class"""

    use_font_anti_aliasing = True  # enable/disable font anti-aliasing
    blur_scale = 20  # blur kernel size scale factor
    blur_min_kernel = 5  # minimum blur kernel size

    @staticmethod
    @contextmanager
    def set(option: str, value):
        """Context manager to set global option value

        Args:
            option: option name
            value: new option value
        """

        if not hasattr(GlobalOptions, option):
            raise DegirumException(f"Unknown option {option}")

        old_value = getattr(GlobalOptions, option)
        setattr(GlobalOptions, option, value)
        try:
            yield old_value
        finally:
            setattr(GlobalOptions, option, old_value)


def _luminance(color: tuple) -> float:
    """Calculate luminance from RGB color"""
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def _inv_conversion_calc(conversion, width, height):
    """Invert conversion function calculation

    - `conversion`: conversion function to invert
    - `width`: max conversion width
    - `height`: max conversion height
    """
    p1 = (width / 2, height / 2)
    p2 = (width / 2 - 1, height / 2 - 1)
    f1 = conversion(*p1)
    f2 = conversion(*p2)
    a = ((f2[0] - f1[0]) / (p2[0] - p1[0]), (f2[1] - f1[1]) / (p2[1] - p1[1]))
    b = (f1[0] - p1[0] * a[0], f1[1] - p1[1] * a[1])
    return lambda x, y: ((x - b[0]) / a[0], (y - b[1]) / a[1])


def _image_segmentation_overlay(
    conversion, overlay_data, original_width, original_height, lut, convert=True
):
    """Return input image scaled with respect to the provided image transformation callback with overlay data added

    - `conversion`: coordinate conversion function accepting two arguments (x,y) and returning two-element tuple
    - `overlay_data`: overlay data to blend on top of input image
    - `original_width`: original image width
    - `original_height`: original image height
    - `lut`: overlay data look up table in RGB format
    - `convert`: apply coordinate conversion function to overlay data if True, otherwise use provided overlay data
    """
    assert isinstance(overlay_data, numpy.ndarray)
    import cv2
    from PIL import Image

    if convert:
        # map corners from original image to model output
        height, width = overlay_data.shape
        inv_conversion = _inv_conversion_calc(conversion, width, height)
        p1 = [int(i) for i in inv_conversion(0, 0)]
        p2 = [int(i) for i in inv_conversion(original_width, original_height)]

        img = overlay_data[
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

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(
            (original_width, original_height), Image.Resampling.NEAREST
        )
        img = numpy.array(pil_img)
    else:
        img = overlay_data
    lut = cv2.cvtColor(lut, cv2.COLOR_RGB2BGR)
    return cv2.LUT(cv2.merge((img, img, img)), lut)


def _create_alpha_channel(img, alpha):
    """Convert BGR to RGBA image where all non-black pixels has specified alpha channel value, and zero otherwise

    -`img`: image to convert
    -`alpha`: alpha channel value to set for non-black pixels
    """
    import cv2

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) != 0
    res = numpy.concatenate(
        (img, numpy.zeros((*mask.shape, 1), dtype=numpy.uint8)), axis=-1
    )
    res[mask] = res[mask] + [0, 0, 0, alpha]
    return cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA)


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    - x: The input bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = numpy.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywhr2xy_corners(x):
    """
     Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
     be in radians from 0 to pi/2.

    - x: Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).
    """

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = numpy.cos(angle), numpy.sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = numpy.concatenate(vec1, -1)
    vec2 = numpy.concatenate(vec2, -1)
    pt1 = ctr - vec1 - vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr + vec1 + vec2
    pt4 = ctr - vec1 + vec2
    poly = numpy.stack([pt1, pt2, pt3, pt4], -2)
    min_y = numpy.min(poly[:, 1])
    candidates = poly[poly[:, 1] == min_y]
    min_x = numpy.min(candidates[:, 0])
    min_index = numpy.where((poly[:, 0] == min_x) & (poly[:, 1] == min_y))[0][0]
    poly = numpy.roll(poly, -min_index, axis=0)
    return poly


class _PrimitivesDrawPIL:
    """Drawing class for PIL backend"""

    def __init__(self, image, alpha, segm_alpha, font_scale):
        """Constructor.

        - image: native image object
        - alpha: alpha-blend weight for overlay details
        - segm_alpha: alpha-blend weight for segmentation overlay details
            (when superimposed over detection overlay details)
        - font_scale: font scale to use for overlay details
        """

        self._pil = importlib.import_module("PIL.Image")
        self._draw = importlib.import_module("PIL.ImageDraw")
        self._font = importlib.import_module("PIL.ImageFont")
        self._pil_filter = importlib.import_module("PIL.ImageFilter")
        self._original_image = image
        self._image = self._pil.new("RGBA", image.size)
        self._segm_image = None
        self._alpha = alpha
        self._segm_alpha = segm_alpha
        self._font_to_draw = self._font.truetype(
            str(Path(__file__).parent.resolve() / "LiberationMono-Regular.ttf"),
            size=int(font_scale * 12),
        )

    def _adj_color(self, color):
        return color + (int(255 * self._alpha),)

    def draw_circle(self, cx, cy, radius, width, color, fill=False):
        """Draw circle.

        - cx: X coordinate of center
        - cy: Y coordinate of center
        - radius: circle radius
        - width: line width
        - color: color to use
        - fill: whether to fill the circle
        """
        draw = self._draw.Draw(self._image)
        adj_color = self._adj_color(color)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=adj_color,
            width=width,
            fill=adj_color if fill else None,
        )

    def text_size(self, text):
        """Calculate text box width, height, and baseline (y-coordinate of the baseline relative to the bottom-most text point).

        - text: text to calculate box size

        Returns tuple containing text width, text height, and baseline
        """
        ascent, descent = self._font_to_draw.getmetrics()
        bbox = self._font_to_draw.getbbox(text)
        text_w = abs(bbox[2] - bbox[0])
        text_h = abs(bbox[3] - bbox[1]) + descent
        return text_w, text_h, 0

    def draw_text(self, px, py, color, text) -> tuple:
        """Draw text string.

        - px: X coordinate of upper left point
        - py: Y coordinate of upper left point
        - color: color to use
        - text: text to draw

        Returns drawn text label bounding box coordinates
        """
        draw = self._draw.Draw(self._image)
        draw.text((px, py), text, font=self._font_to_draw, fill=self._adj_color(color))
        text_w, text_h, _ = self.text_size(text)
        return (px, py, px + text_w, py + text_h)

    def draw_text_label(
        self, x1, y1, x2, y2, color, text, line_width, is_rotated_frame=False
    ) -> tuple:
        """Draw text label near given rectangular frame so it will be always visible

        - x1: X coordinate of top left point of frame
        - y1: Y coordinate of top left point of frame
        - x2: X coordinate of bottom right point of frame
        - y2: Y coordinate of bottom right point of frame
        - color: color to use
        - text: text to draw
        - line_width: line width
        - is_rotated_frame: whether frame is rotated

        Returns drawn text label bounding box coordinates
        """

        draw = self._draw.Draw(self._image)

        margin = 2
        text_w, text_h, _ = self.text_size(text)
        bbox_w = text_w + 2 * margin
        bbox_h = text_h + 2 * margin
        label_top = True

        if y1 >= bbox_h:
            ty = y1 - bbox_h + margin + line_width - 1  # above frame top
        elif y2 + bbox_h < self._image.size[1]:
            ty = y2 + margin - line_width + 1  # below frame bottom
            label_top = False
        else:
            ty = y1 + margin  # below frame top

        x1_check = x1
        x2_check = x2
        if is_rotated_frame:
            if label_top:
                x2_check = x1
            else:
                x1_check = x2

        if x1_check + bbox_w < self._image.size[0]:
            tx = x1_check + margin  # from frame left
        elif x2_check >= bbox_w:
            tx = x2_check - bbox_w + margin
        elif self._image.size[0] >= bbox_w:
            tx = self._image.size[0] - bbox_w + margin  # to image right
        else:
            tx = margin  # from image left

        bbox = (tx - margin, ty - margin, tx + text_w + margin, ty + text_h + margin)
        adj_color = self._adj_color(color)
        draw.rectangle(
            bbox,
            fill=adj_color,
            outline=adj_color,
            width=1,
        )

        text_color = (0, 0, 0) if _luminance(color) > 180 else (255, 255, 255)
        self.draw_text(tx, ty, text_color, text)
        return bbox

    def draw_line(self, x1, y1, x2, y2, width, color):
        """Draw line.

        - x1: X coordinate of beginning point
        - y1: Y coordinate of beginning point
        - x2: X coordinate of ending point
        - y2: Y coordinate of ending point
        - width: line width
        - color: color to use
        """
        draw = self._draw.Draw(self._image)
        draw.line([x1, y1, x2, y2], fill=self._adj_color(color), width=width)

    def blur_box(self, x1, y1, x2, y2):
        """Blur bounding box region

        - x1: X coordinate of top left point
        - y1: Y coordinate of top left point
        - x2: X coordinate of bottom right point
        - y2: Y coordinate of bottom right point
        """
        x1 = int(min(x1, x2))
        x2 = int(max(x1, x2))
        y1 = int(min(y1, y2))
        y2 = int(max(y1, y2))
        radius = max(
            GlobalOptions.blur_min_kernel,
            max(self._original_image.size) // GlobalOptions.blur_scale,
        )
        region = self._original_image.crop((x1, y1, x2, y2))
        region = region.filter(self._pil_filter.BoxBlur(radius))
        self._image.paste(region, (x1, y1))

    def draw_box(self, x1, y1, x2, y2, width, color):
        """Draw rectangle.

        - x1: X coordinate of top left point
        - y1: Y coordinate of top left point
        - x2: X coordinate of bottom right point
        - y2: Y coordinate of bottom right point
        - width: line width
        - color: color to use
        """
        draw = self._draw.Draw(self._image)
        draw.rectangle([x1, y1, x2, y2], outline=self._adj_color(color), width=width)

    def draw_polygon(self, polygon, width, color):
        """Draw polygon.

        - polygon: list of points defining polygon
        - width: line width
        - color: color to use
        """
        draw = self._draw.Draw(self._image)
        poly = polygon.reshape(-1).tolist()
        draw.polygon(poly, outline=self._adj_color(color), width=width)

    def image_overlay(self):
        """Return image overlay with proper blending"""
        # apply segmentation overlay details if they are present
        if self._segm_image is not None:
            ret = self._pil.composite(
                self._segm_image, self._original_image, self._segm_image.getchannel(3)
            )
        else:
            ret = self._original_image
        # apply non-segmentation overlay details
        ret = self._pil.composite(self._image, ret, self._image.getchannel(3))
        return ret

    def image_overlay_extend(self, width, height, fill_color):
        """Update input image by extending its size.

        - width: width to extend up to
        - height: height value to extend
        - fill_color: color to use use if any form of padding is used

        Returns original image width and height
        """
        original_w = self._original_image.width
        original_h = self._original_image.height
        w = max(self._original_image.width, width)
        h = self._original_image.height + height
        image = self._pil.new(self._original_image.mode, (w, h), fill_color)
        image.paste(self._original_image, (0, 0))
        self._original_image = image
        self._image = self._pil.new("RGBA", image.size)
        self._segm_image = None
        return original_w, original_h

    def image_segmentation_overlay(self, conversion, overlay_data, lut, convert=True):
        """Return an image scaled with respect to the provided image transformation callback with overlay data

        - `conversion`: coordinate conversion function accepting two arguments (x,y) and returning two-element tuple
        - `overlay_data`: overlay data to blend
        - `lut`: overlay data look up table in RGB format
        - `convert`: apply coordinate conversion function to overlay data if True, otherwise use provided overlay data
        """
        orig_image = numpy.array(self._original_image)
        orig_height, orig_width = orig_image.shape[:2]
        img = _image_segmentation_overlay(
            conversion, overlay_data, orig_width, orig_height, lut, convert
        )
        img = _create_alpha_channel(img, int(255 * self._segm_alpha))
        self._segm_image = self._pil.fromarray(img)


class _PrimitivesDrawOpenCV:
    """Drawing class for OpenCV backend"""

    def __init__(self, image, alpha, segm_alpha, font_scale):
        """Constructor.

        - image: native image object
        - alpha: alpha-blend weight for overlay details
        - segm_alpha: alpha-blend weight for segmentation overlay details
            (when superimposed over detection overlay details)
        - font_scale: font scale to use for overlay details
        """
        self._original_image = image
        if alpha < 1.0 or segm_alpha is not None:
            self._image = numpy.zeros(image.shape, image.dtype)
        else:
            self._image = image.copy()
        self._segm_image = None
        self._alpha = alpha
        self._segm_alpha = segm_alpha
        self._cv = importlib.import_module("cv2")
        self._font = self._cv.FONT_HERSHEY_PLAIN
        self._font_scale = font_scale

    def _adj_color(self, color):
        return color[::-1]

    def draw_circle(self, cx, cy, radius, width, color, fill=False):
        """Draw circle.

        - px: X coordinate
        - py: Y coordinate
        - radius: circle radius
        - width: line width
        - color: color to use
        - fill: whether to fill the circle
        """

        if fill:
            self._cv.circle(
                self._image,
                (int(cx), int(cy)),
                int(radius + width / 2),  # wider circle to account for 1px thickness
                self._adj_color(color),
                -1,
            )
        else:
            self._cv.circle(
                self._image,
                (int(cx), int(cy)),
                int(radius),
                self._adj_color(color),
                width,
            )

    def text_size(self, text):
        """Calculate text box width, height, and baseline (y-coordinate of the baseline relative to the bottom-most text point).

        - text: text to calculate box size

        Returns tuple containing text width, text height, and baseline
        """
        text_size, baseline = self._cv.getTextSize(
            text, self._font, self._font_scale, 1
        )
        text_w = text_size[0]
        text_h = text_size[1] + baseline
        return (text_w, text_h, baseline)

    def draw_text(self, px, py, color, text):
        """Draw text string.

        - px: X coordinate of upper left point
        - py: Y coordinate of upper left point
        - color: color to use
        - text: text to draw
        """
        px = int(px)
        py = int(py)
        text_w, text_h, baseline = self.text_size(text)
        self._cv.putText(
            self._image,
            text,
            (px, py + text_h - 2),
            self._font,
            self._font_scale,
            self._adj_color(color),
            1,
            (
                self._cv.LINE_AA
                if GlobalOptions.use_font_anti_aliasing
                else self._cv.LINE_8
            ),
        )
        return (px, py, px + text_w, py + text_h)

    def draw_text_label(
        self, x1, y1, x2, y2, color, text, line_width, is_rotated_frame=False
    ) -> tuple:
        """Draw text label near given rectangular frame so it will be always visible

        - x1: X coordinate of top left point of frame
        - y1: Y coordinate of top left point of frame
        - x2: X coordinate of bottom right point of frame
        - y2: Y coordinate of bottom right point of frame
        - color: color to use
        - text: text to draw
        - line_width: line width
        - is_rotated_frame: whether frame is rotated

        Returns drawn text label bounding box coordinates
        """

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        margin = 1
        text_w, text_h, baseline = self.text_size(text)
        bbox_w = text_w + 2 * margin
        bbox_h = text_h + baseline + 2 * margin
        shape = self._image.shape
        label_top = True

        half_lw = line_width // 2 + line_width % 2
        if y1 >= bbox_h:
            ty = y1 - bbox_h + margin - half_lw  # above frame top
        elif y2 + bbox_h < shape[0]:
            ty = y2 + margin + half_lw  # below frame bottom
            label_top = False
        else:
            ty = y1 + margin  # below frame top

        x1_check = x1
        x2_check = x2
        if is_rotated_frame:
            if label_top:
                x2_check = x1
            else:
                x1_check = x2

        if x1_check + bbox_w < shape[1]:
            tx = x1_check + margin - half_lw  # from frame left
        elif x2_check >= bbox_w:
            tx = x2_check - bbox_w + margin + half_lw  # to frame right
        elif shape[1] >= bbox_w:
            tx = shape[1] - bbox_w + margin  # to image right
        else:
            tx = margin  # from image left

        bbox = (
            tx - margin,
            ty - margin,
            tx + text_w + margin,
            ty + text_h + baseline + margin,
        )

        self._cv.rectangle(
            self._image, bbox[:2], bbox[-2:], self._adj_color(color), self._cv.FILLED
        )

        text_color = (1, 1, 1) if _luminance(color) > 180 else (255, 255, 255)
        self._cv.putText(
            self._image,
            text,
            (tx, ty + text_h - 2),
            self._font,
            self._font_scale,
            text_color,
            1,
            self._cv.LINE_AA,
        )

        return bbox

    def draw_line(self, x1, y1, x2, y2, width, color):
        """Draw line.

        - x1: X coordinate of beginning point
        - y1: Y coordinate of beginning point
        - x2: X coordinate of ending point
        - y2: Y coordinate of ending point
        - width: line width
        - color: color to use
        """

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        self._cv.line(self._image, (x1, y1), (x2, y2), self._adj_color(color), width)

    def blur_box(self, x1, y1, x2, y2):
        """Blur bounding box region

        - x1: X coordinate of top left point
        - y1: Y coordinate of top left point
        - x2: X coordinate of bottom right point
        - y2: Y coordinate of bottom right point
        """

        h, w = self._image.shape[:2]
        x1 = int(min(x1, x2))
        x2 = int(max(x1, x2))
        y1 = int(min(y1, y2))
        y2 = int(max(y1, y2))
        kernel_size = (
            max(GlobalOptions.blur_min_kernel, w // GlobalOptions.blur_scale),
            max(GlobalOptions.blur_min_kernel, h // GlobalOptions.blur_scale),
        )
        self._image[y1:y2, x1:x2] = self._cv.blur(
            self._original_image[y1:y2, x1:x2], kernel_size
        )

    def draw_box(self, x1, y1, x2, y2, width, color):
        """Draw rectangle.

        - x1: X coordinate of top left point
        - y1: Y coordinate of top left point
        - x2: X coordinate of bottom right point
        - y2: Y coordinate of bottom right point
        - width: line width
        - color: color to use
        """
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        self._cv.rectangle(
            self._image, (x1, y1), (x2, y2), self._adj_color(color), width
        )

    def draw_polygon(self, polygon, width, color):
        """Draw polygon.

        - polygon: numpy.array with (x, y) point tuples defining polygon
        - width: line width
        - color: color to use
        """
        self._cv.polylines(self._image, [polygon], True, self._adj_color(color), width)

    def _alpha_blend(self, image1, image2, blend_alpha):
        """Return image that merges image2 with image1 using alpha-blending"""
        alpha = Fraction(blend_alpha).limit_denominator(255)
        alpha_complement = Fraction(1.0 - blend_alpha).limit_denominator(255)
        mask = self._cv.cvtColor(image2, self._cv.COLOR_BGR2GRAY) != 0
        ret = image1.copy()
        ret[mask] = (
            (image2[mask].astype(numpy.uint16) * alpha.numerator) // alpha.denominator
            + (image1[mask].astype(numpy.uint16) * alpha_complement.numerator)
            // alpha_complement.denominator
        ).astype(image1.dtype)
        return ret

    def image_overlay(self):
        """Return image overlay with proper blending"""
        if self._alpha < 1.0 or self._segm_alpha is not None:
            # apply segmentation overlay details if they are present
            if self._segm_image is not None:
                ret = self._alpha_blend(
                    self._original_image, self._segm_image, self._segm_alpha
                )
            else:
                ret = self._original_image
            # apply non-segmentation overlay details if they are present
            if numpy.any(self._image):
                ret = self._alpha_blend(ret, self._image, self._alpha)
        else:
            ret = self._image
        return ret

    def image_overlay_extend(self, width, height, fill_color):
        """Update input image by extending its size.

        - width: width to extend up to
        - height: height value to extend
        - fill_color: color to use if any form of padding is used

        Returns original image width and height
        """
        original_w = self._original_image.shape[1]
        original_h = self._original_image.shape[0]
        w = max(original_w, width)
        h = original_h + height
        image = numpy.zeros(
            (h, w, self._original_image.shape[2]), self._original_image.dtype
        )
        image[:] = tuple(reversed(fill_color))
        image[0:original_h, 0:original_w, :] = self._original_image
        self._original_image = image
        if self._alpha < 1.0 or self._segm_alpha is not None:
            self._image = numpy.zeros(image.shape, image.dtype)
        else:
            self._image = image.copy()
        self._segm_image = None
        return original_w, original_h

    def image_segmentation_overlay(self, conversion, overlay_data, lut, convert=True):
        """Return an image scaled with respect to the provided image transformation callback with overlay data

        - `conversion`: coordinate conversion function accepting two arguments (x,y) and returning two-element tuple
        - `overlay_data`: overlay data to blend
        - `lut`: overlay data look up table in RGB format
        - `convert`: apply coordinate conversion function to overlay data if True, otherwise use provided overlay data
        """
        orig_height, orig_width = self._original_image.shape[:2]
        self._segm_image = _image_segmentation_overlay(
            conversion, overlay_data, orig_width, orig_height, lut, convert
        )


def create_draw_primitives(image_data, alpha, segm_alpha, font_scale):
    """Create and return PrimitivesDraw object to use to draw overlays.

    - image-data: inference input image
    - alpha: alpha-blend weight for overlay details
    - segm_alpha: alpha-blend weight for segmentation overlay details
        (when superimposed over detection overlay details)
    - font_scale: font scale to use for overlay details
    """
    if (
        isinstance(image_data, numpy.ndarray)
        and len(image_data.shape) == 3
        and importlib.util.find_spec("cv2")
    ):
        return _PrimitivesDrawOpenCV(image_data, alpha, segm_alpha, font_scale)

    if importlib.util.find_spec("PIL"):
        pillow = importlib.import_module("PIL")
        if pillow and isinstance(image_data, pillow.Image.Image):
            return _PrimitivesDrawPIL(image_data, alpha, segm_alpha, font_scale)

    raise DegirumException("Unknown preprocessed image data format")
