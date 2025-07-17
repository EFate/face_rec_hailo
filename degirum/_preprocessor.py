#
# _preprocessor.py - DeGirum Python SDK: image preprocessing
# Copyright DeGirum Corp. 2022
#
# Implements _Image_Preprocessor class to convert images from various input formats
# to representation compatible with AI model input
#

import io
from urllib.parse import urlparse

import numpy
import requests
from typing import Union
from abc import ABC, abstractmethod
from .exceptions import DegirumException, validate_color_tuple
from .aiclient import ModelParams as ModelParams
from .log import log_wrap
from ._model_param_helpers import model_shape_get

_image_backend_list = ("pil", "opencv", "auto")
""" list of graphical backends """


class _Image_Preprocessor(ABC):
    """
    Image Preprocessor interface class
    """

    _resize_method_list = ("nearest", "bilinear", "area", "bicubic", "lanczos")
    """ list of resize methods """

    _pad_method_list = ("stretch", "letterbox", "crop-first", "crop-last")
    """ list of padding methods """

    _colorspace_list = ("RGB", "BGR", "auto")
    """ list of color spaces """

    _image_formats = ("JPEG", "RAW")
    """ list of image formats """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        model_input=0,
        resize_method="bilinear",
        pad_method="letterbox",
        input_colorspace="auto",
        fill_color=(0, 0, 0),
        crop_percentage=1.0,
        image_backend,
    ):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index
            - `resize_method`: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos';
            - `pad_method`: one of 'stretch', 'letterbox', 'crop-first', 'crop-last';
            - `crop_percentage`: percentage of image dimensions to crop around with cropping pad methods. Value in the range [0..1];
            - `input_colorspace`: one of 'auto', 'RGB' or 'BGR';
            - `fill_color`: fill color in case of 'letterbox' padding; 3-element RGB tuple
            - `image_backend`: graphical image backend, one of 'pil', 'opencv'
        """

        self._model_params = model_params
        self._model_input = model_input
        self._resize_method = resize_method
        self._pad_method = pad_method
        self._input_colorspace = input_colorspace
        self._fill_color = fill_color
        self._crop_percentage = crop_percentage
        self._image_backend = image_backend
        self.generate_image_result = True
        self._resize_options: dict = {}

    @property
    def image_backend(self) -> str:
        """Graphical image backend: one of 'pil', 'opencv'"""
        return self._image_backend

    @property
    def resize_method(self) -> str:
        """Image resize method: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos'; default is 'bilinear'"""
        return self._resize_method

    @resize_method.setter
    def resize_method(self, value: str):
        if value in _Image_Preprocessor._resize_method_list:
            self._resize_method = value
        else:
            raise DegirumException(
                f"Incorrect resize method '{value}'. Must be one of the following: {_Image_Preprocessor._resize_method_list}"
            )

    @property
    def pad_method(self) -> str:
        """Image padding method: one of 'stretch', 'letterbox'; default is 'letterbox'"""
        return self._pad_method

    @pad_method.setter
    def pad_method(self, value: str):
        if value in _Image_Preprocessor._pad_method_list:
            self._pad_method = value
        else:
            raise DegirumException(
                f"Incorrect padding method '{value}'. Must be one of the following: {_Image_Preprocessor._pad_method_list}"
            )

    @property
    def fill_color(self) -> tuple:
        """fill color in case of 'letterbox' padding; 3-element RGB tuple, default is (0,0,0)"""
        return self._fill_color

    @fill_color.setter
    def fill_color(self, value: tuple):
        self._fill_color = validate_color_tuple(value)

    @property
    def input_colorspace(self) -> str:
        """input color space - one of 'auto', 'RGB' or 'BGR'; default is 'auto'"""
        return self._input_colorspace

    @input_colorspace.setter
    def input_colorspace(self, value: str):
        if value in _Image_Preprocessor._colorspace_list:
            self._input_colorspace = value
        else:
            raise DegirumException(
                f"Incorrect color space '{value}'. Must be one of the following: {_Image_Preprocessor._colorspace_list}"
            )

    @property
    def crop_percentage(self) -> float:
        """center crop percentage - float value between 0 and 1 for how much of the image to crop"""
        return self._crop_percentage

    @crop_percentage.setter
    def crop_percentage(self, value: float):
        if value < 0.0 or value > 1.0:
            raise DegirumException(
                f"Invalid center crop percentage value {value}, must be between 0.0 and 1.0"
            )
        self._crop_percentage = value

    @property
    def image_format(self) -> str:
        """image format - one of 'JPEG' or 'RAW'"""
        return self._model_params.InputImgFmt[self._model_input]

    @image_format.setter
    def image_format(self, value: str):
        if value in _Image_Preprocessor._image_formats:
            new_fmt_array = self._model_params.InputImgFmt
            new_fmt_array[self._model_input] = value
            self._model_params.InputImgFmt = new_fmt_array
        else:
            raise DegirumException(
                f"Incorrect image format '{value}'. Must be one of the following: {_Image_Preprocessor._image_formats}"
            )

    @abstractmethod
    def _get_image_size(self, img):
        """Helper method for getting the size of an image
            - `img`: the image to get the size of
        Returns tuple of (width, height)
        """

    @abstractmethod
    def _crop(self, img, left, top, right, bottom):
        """Helper method for cropping an image
            - `img`: the image to crop
            - `left`: left pixel coordinate of crop area
            - `top`: top pixel coordinate of crop area
            - `right`: right pixel coordinate of crop area
            - `bottom`: bottom pixel coordinate of crop area
        Returns cropped image
        """

    @abstractmethod
    def _resize(self, img, width, height, resample_method):
        """Helper method for resizing an image
            - `img`: the image to resize
            - `width`: desired image width
            - `height`: desired image height
            - `resample_method`: resampling method
        Returns resized image
        """

    @abstractmethod
    def _pad(self, img, width, height, fill_color):
        """Helper method for adding padding to an image
            - `img`: the image add padding to
            - `width`: desired image width
            - `height`: desired image height
            - `fill_color`: color tuple for what color to pad with
        Returns tuple with new image, left pixel coordinate of original image inside new image
        and top pixel coordiante of original image inside new image
        """

    @abstractmethod
    def _corrected_fill_color(self, colorspace):
        """helper method for correcting fill color tuple to match colorspace"""

    def _preprocess_input(self, image: str):
        """Perform input image download if image is http or https link

        - image: input image path string,

        Returns BytesIO object containing image data
        """
        assert isinstance(image, str)
        if "http:" == image[:5] or "https:" == image[:6]:
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
                "Dnt": "1",
                "Host": urlparse(image).netloc,
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            }
            return io.BytesIO(requests.get(image, headers=headers, timeout=5).content)
        return None

    def _center_crop(self, image, new_width, new_height):
        """Helper function for cropping an image around the center"""
        width, height = self._get_image_size(image)

        left = int(round((width - new_width) / 2.0))
        right = left + new_width

        top = int(round((height - new_height) / 2.0))
        bottom = top + new_height

        return self._crop(image, left, top, right, bottom), left, top

    def _get_input_shape(self) -> list:
        """Helper function for getting the input shape in the form of a NxHxWxC list"""

        return model_shape_get(self._model_params, self._model_input, 4)

    def _apply_pad_method(self, image, colorspace="RGB"):
        """Applies pad method to the image. If the pad method is 'stretch' then the image will be resized
        to the dimensions of the model. If the pad method is 'letterbox' then the image will be resized to fit
        inside the model while maintaining aspect ratio and then padded with fill_color to match the model dimensions.
        If the pad method is 'crop-first' then the image is first cropped around the center, preserving crop_percentage
        of the image to match the model aspect ratio, then it is resized to fit the model dimensions. If the pad method
        is 'crop-last', then the image is first resized. If the model input is a square then the image is resized so that
        the shortest image side fits inside the model. If the model input is not a square, then the image is stretch resized
        to match the model aspect ratio with respect to crop_percentage. The image is then cropped around the center to
        match the model dimensions with respect to crop_percentage.
            - `image`: image to apply the pad method to
            - `colorspace`: colorspace of the image

        Returns a tuple containing modified image and coordinate conversion function
        """
        resample_method = self._resize_options[self._resize_method]

        _, h, w, _ = self._get_input_shape()

        dx = dy = 0  # offset of left top corner of original image in resized image
        image_obj = image
        iw, ih = self._get_image_size(image)
        coord_conv = lambda x, y: (
            iw * min(1, max(0, x / w)),
            ih * min(1, max(0, y / h)),
        )
        if self._pad_method == "stretch" and (iw != w or ih != h):
            image_obj = self._resize(image_obj, w, h, resample_method)
        elif self._pad_method == "letterbox" and (iw != w or ih != h):
            scale = min(w / iw, h / ih)
            nw = min(round(iw * scale), w)
            nh = min(round(ih * scale), h)
            gain = min(nh / ih, nw / iw)
            # resize preserving aspect ratio
            scaled_image = self._resize(image_obj, nw, nh, resample_method)

            image_obj, dx, dy = self._pad(
                scaled_image, w, h, self._corrected_fill_color(colorspace)
            )
            coord_conv = lambda x, y: (
                min(max(0, (x - dx) / gain), iw),
                min(max(0, (y - dy) / gain), ih),
            )
        elif self._pad_method == "crop-first":
            # scale model dimensions to fit image
            scale = min(
                iw * self._crop_percentage / w,
                ih * self._crop_percentage / h,
            )
            nw = int(w * scale)
            nh = int(h * scale)

            if nw != iw or nh != ih:
                image_obj, dx, dy = self._center_crop(image_obj, nw, nh)

            if nw != w or nh != h:
                image_obj = self._resize(image_obj, w, h, resample_method)

            coord_conv = lambda x, y: (
                nw * min(1, max(0, x / w)) + dx,
                nh * min(1, max(0, y / h)) + dy,
            )
        elif self._pad_method == "crop-last":
            # scale image dimensions to fit model
            if w == h:  # scale image preserving aspect ratio for square model
                if iw >= ih:
                    nh = int(h / self._crop_percentage)
                    nw = int(nh * iw / ih)
                else:
                    nw = int(w / self._crop_percentage)
                    nh = int(nw * ih / iw)
            else:  # stretch to model aspect ratio for non-square model
                nh = int(h / self._crop_percentage)
                nw = int(w / self._crop_percentage)

            if nw != iw or nh != ih:
                image_obj = self._resize(image_obj, nw, nh, resample_method)

            if w != nw or h != nh:
                image_obj, dx, dy = self._center_crop(image_obj, w, h)

            coord_conv = lambda x, y: (
                iw * min(1, max(0, (x + dx) / nw)),
                ih * min(1, max(0, (y + dy) / nh)),
            )
        return image_obj, coord_conv

    @log_wrap
    def forward(self, image):
        """
        Perform image conversion to model requirements
            - image: input image path string,
                or numpy 3D array of pixels in a form HWC,
                    where color dimension is native to selected backend: RGB for 'pil' and BGR for 'opencv' backend
                or PIL.Image object (only for 'pil' backend)
        Returns dictionary:
            `raw_result` is 'bytes' sequence suitable to feed to the AI model with given parameters.
            `image_input` is RGB image object before resize, native to selected graphical backend:
                PIL.Image for 'pil' and numpy.ndarray for 'opencv'.
            `converter` is conversion lambda from resized to original image coordinates.
            `image_result` is the image object corresponding to `raw_result`
        """
        pass


class _Preprocessor_PIL(_Image_Preprocessor):
    """
    Implementation of image preprocessor for 'pil' backend
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index;
            - `resize_method`: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos';
            - `pad_method`: one of 'stretch', 'letterbox';
            - `input_colorspace`: one of 'RGB', 'BGR', or 'auto';
            - `fill_color`: fill color in case of 'letterbox' padding; 3-element RGB tuple
        """
        kwargs["image_backend"] = "pil"
        super().__init__(*args, **kwargs)
        self._backend = __import__("PIL", fromlist=["Image"]).Image
        self._resize_options = dict(
            zip(
                _Image_Preprocessor._resize_method_list,
                [
                    self._backend.Resampling.NEAREST,
                    self._backend.Resampling.BILINEAR,
                    self._backend.Resampling.BOX,
                    self._backend.Resampling.BICUBIC,
                    self._backend.Resampling.LANCZOS,
                ],
            )
        )

    def _get_image_size(self, img):
        """Helper method for getting the size of an image
            - `img`: the image to get the size of
        Returns tuple of (width, height)
        """
        return img.size

    def _crop(self, img, left, top, right, bottom):
        """Helper method for cropping an image
            - `img`: the image to crop
            - `left`: left pixel coordinate of crop area
            - `top`: top pixel coordinate of crop area
            - `right`: right pixel coordinate of crop area
            - `bottom`: bottom pixel coordinate of crop area
        Returns cropped image
        """
        return img.crop((left, top, right, bottom))

    def _resize(self, img, width, height, resample_method):
        """Helper method for resizing an image
            - `img`: the image to resize
            - `width`: desired image width
            - `height`: desired image height
            - `resample_method`: resampling method
        Returns resized image
        """
        return img.resize((width, height), resample=resample_method)

    def _pad(self, img, width, height, fill_color):
        """Helper method for adding padding to an image
            - `img`: the image add padding to
            - `width`: desired image width
            - `height`: desired image height
            - `fill_color`: color tuple for what color to pad with
        Returns tuple with new image, left pixel coordinate of original image inside new image
        and top pixel coordiante of original image inside new image
        """
        iw, ih = self._get_image_size(img)
        left = (width - iw) // 2
        top = (height - ih) // 2

        image_obj = self._backend.new(img.mode, (width, height), fill_color)
        image_obj.paste(img, (left, top))

        return image_obj, left, top

    def _corrected_fill_color(self, colorspace):
        """helper method for correcting fill color tuple to match colorspace"""
        return self._fill_color if colorspace == "RGB" else self._fill_color[::-1]

    @log_wrap
    def forward(self, image):
        """Implementation for 'pil'"""

        if isinstance(image, bytes):
            # pass-through
            return dict(
                raw_result=image,
                image_input=image,
                converter=lambda x, y: (x, y),
                image_result=image if self.generate_image_result else None,
            )

        input_colorspace = self._input_colorspace
        if input_colorspace == "auto":
            input_colorspace = "RGB"

        image_ret = None
        if isinstance(image, str):
            # image is path to image file
            buf = self._preprocess_input(image)
            if buf:
                image_ret = self._backend.open(buf)
            else:
                image_ret = self._backend.open(image)
        else:
            if isinstance(image, self._backend.Image):
                # image is PIL image
                image_ret = image.copy()
            elif isinstance(image, numpy.ndarray):
                if len(image.shape) == 3:
                    # image is numpy array: convert to PIL image
                    image_ret = self._backend.fromarray(
                        image[..., ::-1] if input_colorspace == "BGR" else image
                    )
                else:
                    raise DegirumException(
                        f"Image shape '{image.shape}' is not supported for 'pil' image backend"
                    )
            else:
                raise DegirumException(
                    f"Image type '{type(image)}' is not supported for 'pil' image backend"
                )

        if image_ret.mode != "RGB":
            image_ret = image_ret.convert("RGB")

        #
        # resize + pad/crop
        #
        image_obj, coord_conv = self._apply_pad_method(image_ret, image_ret.mode)

        #
        # convert to what model requires
        #
        if self.image_format == "JPEG":
            # save to byte buffer
            buf = io.BytesIO()
            image_obj.save(buf, format="JPEG")
            return dict(
                raw_result=buf.getvalue(),
                image_input=image_ret,
                converter=coord_conv,
                image_result=image_obj if self.generate_image_result else None,
            )

        elif self.image_format == "RAW":
            # convert to model raw data type
            buf = numpy.asarray(image_obj).astype(
                numpy.uint8
                if self._model_params.InputRawDataType[self._model_input] == "DG_UINT8"
                else numpy.float32
            )

            # convert to model color space
            if self._model_params.InputColorSpace[self._model_input] != image_ret.mode:
                buf = buf[..., ::-1]

            # convert to bytes
            return dict(
                raw_result=buf.tobytes(),
                image_input=image_ret,
                converter=coord_conv,
                image_result=image_obj if self.generate_image_result else None,
            )

        else:
            input_format = self.image_format + (
                ("/" + self._model_params.InputRawDataType[self._model_input])
                if self.image_format == "RAW"
                else ""
            )
            raise DegirumException(
                f"Model image format '{input_format}' is not supported by preprocessor"
            )


class _Preprocessor_CV(_Image_Preprocessor):
    """
    Implementation of image preprocessor for 'opencv' backend
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index;
            - `resize_method`: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos';
            - `pad_method`: one of 'stretch', 'letterbox';
            - `input_colorspace`: one of 'RGB', 'BGR', or 'auto';
            - `fill_color`: fill color in case of 'letterbox' padding; 3-element RGB tuple
        """
        kwargs["image_backend"] = "opencv"
        super().__init__(*args, **kwargs)
        self._backend = __import__("cv2")
        self._resize_options = dict(
            zip(
                _Image_Preprocessor._resize_method_list,
                [
                    self._backend.INTER_NEAREST,
                    self._backend.INTER_LINEAR,
                    self._backend.INTER_AREA,
                    self._backend.INTER_CUBIC,
                    self._backend.INTER_LANCZOS4,
                ],
            )
        )

    def _get_image_size(self, img):
        """Helper method for getting the size of an image
            - `img`: the image to get the size of
        Returns tuple of width, height
        """
        return img.shape[1], img.shape[0]

    def _crop(self, img, left, top, right, bottom):
        """Helper method for cropping an image
            - `img`: the image to crop
            - `left`: left pixel coordinate of crop area
            - `top`: top pixel coordinate of crop area
            - `right`: right pixel coordinate of crop area
            - `bottom`: bottom pixel coordinate of crop area
        Returns cropped image
        """
        if len(img.shape) == 2:
            return img[top:bottom, left:right]
        else:
            return img[top:bottom, left:right, ...]

    def _resize(self, img, width, height, resample_method):
        """Helper method for resizing an image
            - `img`: the image to resize
            - `width`: desired image width
            - `height`: desired image height
            - `resample_method`: resampling method
        Returns resized image
        """
        return self._backend.resize(img, (width, height), interpolation=resample_method)

    def _pad(self, img, width, height, fill_color):
        """Helper method for adding padding to an image
            - `img`: the image add padding to
            - `width`: desired image width
            - `height`: desired image height
            - `fill_color`: color tuple for what color to pad with
        Returns tuple with new image, left pixel coordinate of original image inside new image
        and top pixel coordiante of original image inside new image
        """
        iw, ih = self._get_image_size(img)
        left = (width - iw) // 2
        top = (height - ih) // 2

        image_obj = numpy.zeros((height, width, 3), img.dtype)
        if self._fill_color != (0, 0, 0):
            image_obj[:] = fill_color

        image_obj[top : top + ih, left : left + iw, :] = img
        return image_obj, left, top

    def _corrected_fill_color(self, colorspace):
        """helper method for correcting fill color tuple to match colorspace"""
        return (
            tuple(reversed(self._fill_color))
            if colorspace == "BGR"
            else self._fill_color
        )

    @log_wrap
    def forward(self, image):
        """Implementation for 'opencv'"""

        image_input = None
        inp_color = None

        input_colorspace = self._input_colorspace
        if input_colorspace == "auto":
            input_colorspace = "BGR"

        if isinstance(image, bytes):
            # pass-through
            return dict(
                raw_result=image,
                image_input=image,
                converter=lambda x, y: (x, y),
                image_result=image if self.generate_image_result else None,
            )

        if isinstance(image, str):
            # image is path to image file
            buf = self._preprocess_input(image)
            if buf:
                image_input = self._backend.imdecode(
                    numpy.frombuffer(buf.read(), numpy.uint8),
                    self._backend.IMREAD_COLOR,
                )
                if image_input is None:
                    raise DegirumException(f"Failed to decode image from '{image}'")
            else:
                image_input = self._backend.imread(image)
                if image_input is None:
                    raise DegirumException(f"Failed to read image from '{image}'")
            inp_color = "BGR"
        else:
            if isinstance(image, numpy.ndarray):
                if len(image.shape) == 3:
                    # image is 3D numpy array
                    image_input = image.copy()
                else:
                    raise DegirumException(
                        f"Image array shape '{image.shape}' is not supported for 'opencv' image backend"
                    )
            else:
                raise DegirumException(
                    f"Image type '{type(image)}' is not supported for 'opencv' image backend"
                )

            inp_color = input_colorspace

        #
        # resize + pad/crop
        #
        image_obj, coord_conv = self._apply_pad_method(image_input, inp_color)

        #
        # convert to what model requires
        #
        if self.image_format == "JPEG":
            if inp_color != "BGR":
                # convert to "BGR" color space since imencode() expects BGR
                image_obj_bgr = self._backend.cvtColor(
                    image_obj, self._backend.COLOR_RGB2BGR
                )  # here we assume only BGR and RGB
            else:
                image_obj_bgr = image_obj

            # encode to JPEG
            _, buf = self._backend.imencode(".jpg", image_obj_bgr)

            # convert to model color space
            if self.generate_image_result:
                if self._model_params.InputColorSpace[self._model_input] == "BGR":
                    # model color space is BGR
                    image_result = image_obj_bgr
                else:
                    # model color space is RGB
                    if inp_color == "BGR":
                        image_result = self._backend.cvtColor(
                            image_obj, self._backend.COLOR_BGR2RGB
                        )  # here we assume only BGR and RGB
                    else:
                        image_result = image_obj
            else:
                image_result = None

            return dict(
                raw_result=buf.tobytes(),  # save to byte buffer
                image_input=image_input,
                converter=coord_conv,
                image_result=image_result,
            )

        elif self.image_format == "RAW" and (
            self._model_params.InputRawDataType[self._model_input] == "DG_UINT8"
            or self._model_params.InputRawDataType[self._model_input] == "DG_FLT"
        ):
            # convert to model color space
            if inp_color != self._model_params.InputColorSpace[self._model_input]:
                image_obj = self._backend.cvtColor(
                    image_obj, self._backend.COLOR_BGR2RGB
                )  # here we assume only BGR and RGB

            # convert to model data type
            if (
                self._model_params.InputRawDataType[self._model_input] == "DG_UINT8"
                and image_obj.dtype != numpy.uint8
            ):
                image_obj = image_obj.astype(numpy.uint8)
            elif (
                self._model_params.InputRawDataType[self._model_input] == "DG_FLT"
                and image_obj.dtype != numpy.float32
            ):
                image_obj = image_obj.astype(numpy.float32)

            return dict(
                raw_result=image_obj.tobytes(),
                image_input=image_input,
                converter=coord_conv,
                image_result=image_obj if self.generate_image_result else None,
            )

        else:
            input_format = self.image_format + (
                ("/" + self._model_params.InputRawDataType[self._model_input])
                if self.image_format == "RAW"
                else ""
            )
            raise DegirumException(
                f"Model image format '{input_format}' is not supported by preprocessor"
            )


@log_wrap
def create_image_preprocessor(*args, **kwargs) -> _Image_Preprocessor:
    """
    Create and return preprocessor object
        - `model_params`: model parameters object
        - `model_input`: preprocessor input index;
        - `resize_method`: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos'; default is 'bilinear';
        - `pad_method`: one of 'stretch', 'letterbox'; default is 'letterbox';
        - `input_colorspace`: one of 'RGB', 'BGR', or 'auto'; default is 'auto';
        - `fill_color`: fill color in case of 'letterbox' padding; 3-element RGB tuple, default is (0,0,0);
        - `image_backend`: one of 'pil', 'opencv', 'auto'; default is 'auto': try OpenCV, if not installed, try PIL
    """

    def auto(*args, **kwargs):
        try:
            return _Preprocessor_CV(*args, **kwargs)
        except BaseException:
            return _Preprocessor_PIL(*args, **kwargs)

    variants = {
        "pil": lambda: _Preprocessor_PIL,
        "opencv": lambda: _Preprocessor_CV,
        "auto": lambda: auto,
    }

    preproc = variants.get(kwargs["image_backend"], None)
    if preproc is None:
        raise DegirumException(
            f"Incorrect image backend '{kwargs['image_backend']}'. It must be one of the following: {_image_backend_list}"
        )
    return preproc()(*args, **kwargs)


class _CommonPreprocessor:
    """Implementation of common basic preprocessor"""

    def __init__(self, model_params: ModelParams, *, model_input=0):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index
        """
        if model_input >= len(model_params.InputType):
            raise DegirumException(
                f"Invalid model input '{model_input}' passed. Model total number of inputs is '{len(model_params.InputType)}'"
            )
        self._model_params = model_params
        self._model_input = model_input

    @log_wrap
    def forward(self, data):
        """Perform numpy tensor check according to model requirements
            - `data`: input numpy tensor
        Returns dictionary:
            `raw_result` is 'bytes' sequence suitable to feed to the AI model with given parameters.
        """
        if not isinstance(data, numpy.ndarray):
            raise DegirumException(
                f"Unknown input data passed as the input #{self._model_input}. Numpy tensor is expected."
            )
        return dict(raw_result=data.tobytes())


class _TensorPreprocessor(_CommonPreprocessor):
    """Implementation of tensor preprocessor"""

    def __init__(self, *args, **kwargs):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index
        """
        super().__init__(*args, **kwargs)

    @log_wrap
    def forward(self, data):
        """Perform numpy tensor check according to model requirements
            - `data`: input numpy tensor
        Returns dictionary:
            `raw_result` is 'bytes' sequence suitable to feed to the AI model with given parameters.
        """
        shape_list = []
        if (
            self._model_params.InputShape_defined(self._model_input)
            and self._model_params.InputShape[self._model_input]
        ):
            shape_list = self._model_params.InputShape[self._model_input]
        else:
            if (
                self._model_params.InputN_defined(self._model_input)
                and self._model_params.InputN[self._model_input] > 0
            ):
                shape_list.append(self._model_params.InputN[self._model_input])
            if (
                self._model_params.InputH_defined(self._model_input)
                and self._model_params.InputH[self._model_input] > 0
            ):
                shape_list.append(self._model_params.InputH[self._model_input])
            if (
                self._model_params.InputW_defined(self._model_input)
                and self._model_params.InputW[self._model_input] > 0
            ):
                shape_list.append(self._model_params.InputW[self._model_input])
            if (
                self._model_params.InputC_defined(self._model_input)
                and self._model_params.InputC[self._model_input] > 0
            ):
                shape_list.append(self._model_params.InputC[self._model_input])
        shape = tuple(shape_list)

        if not isinstance(data, numpy.ndarray):
            raise DegirumException(
                f"Unknown input data passed as the input #{self._model_input}. Numpy tensor of ({shape}) expected."
            )
        if numpy.shape(data) != shape:
            raise DegirumException(
                f"Shape of tensor passed as the input #{self._model_input} does not match to model parameters. Expected tensor shape is {shape}."
            )
        return dict(raw_result=data.tobytes())


class _AudioPreprocessor(_CommonPreprocessor):
    """Implementation of audio preprocessor"""

    def __init__(self, *args, **kwargs):
        """
        Constructor
            - `model_params`: model parameters object
            - `model_input`: preprocessor input index
        """
        super().__init__(*args, **kwargs)

    @log_wrap
    def forward(self, data):
        """Perform numpy tensor check according to model requirements
            - `data`: input numpy tensor
        Returns dictionary:
            `raw_result` is 'bytes' sequence suitable to feed to the AI model with given parameters.
        """
        if (
            not isinstance(data, numpy.ndarray)
            or len(data.shape) != 1
            or data.shape[0] != self._model_params.InputWaveformSize[self._model_input]
        ):
            raise DegirumException(
                f"Input data format passed for the audio model input #{self._model_input} is not supported: "
                + f"expected 1-D numpy tensor containing audio waveform of {self._model_params.InputWaveformSize[self._model_input]} samples"
            )

        if (
            (
                self._model_params.InputRawDataType[self._model_input] == "DG_INT16"
                and data.dtype != "int16"
            )
            or (
                self._model_params.InputRawDataType[self._model_input] == "DG_FLT"
                and data.dtype != "float32"
            )
            or (
                self._model_params.InputRawDataType[self._model_input] == "DG_UINT8"
                and data.dtype != "uint8"
            )
        ):
            raise DegirumException(
                f"Tensor element data type for the audio model input #{self._model_input} should be '{self._model_params.InputRawDataType[self._model_input][3:]}'"
            )

        return dict(raw_result=data.tobytes())


class _PreprocessorAggregator:
    """
    Preprocessor aggregator class
    """

    def __init__(self, model_params: ModelParams):
        """
        Constructor
            - `model_params`: model parameters object
        """
        self._model_parameters = model_params
        self._preprocessors: list[Union[_Image_Preprocessor, _CommonPreprocessor]] = []
        for i, input in enumerate(model_params.InputType):
            if input == "Image":
                self._preprocessors.append(
                    create_image_preprocessor(
                        model_params,
                        image_backend=model_params.ImageBackend[i],
                        resize_method=model_params.InputResizeMethod[i],
                        pad_method=model_params.InputPadMethod[i],
                        crop_percentage=model_params.InputCropPercentage[i],
                        model_input=i,
                    )
                )
            elif input == "Tensor":
                self._preprocessors.append(
                    _TensorPreprocessor(model_params, model_input=i)
                )
            elif input == "Audio":
                self._preprocessors.append(
                    _AudioPreprocessor(model_params, model_input=i)
                )
            else:
                raise DegirumException(f"Unknown InputType: {input}")

    def _image_property_iter(self):
        """Iterate over image preprocessors"""
        iterator = iter(
            e for e in self._preprocessors if isinstance(e, _Image_Preprocessor)
        )
        try:
            yield next(iterator)
        except StopIteration:
            raise DegirumException(
                "Cannot access preprocessor image-related properties: the model has no image type inputs"
            )
        yield from iterator

    def _get_image_pp(self):
        """Return first image preprocessor or raise exception if it is not found."""
        return next(self._image_property_iter())

    @property
    def has_image_inputs(self) -> bool:
        """Check if there is at least one image input available"""
        return (
            next(
                filter(
                    lambda pp: isinstance(pp, _Image_Preprocessor), self._preprocessors
                ),
                None,
            )
            is not None
        )

    @property
    def image_backend(self) -> str:
        """Graphical image backend: one of 'pil', 'opencv'"""
        return self._get_image_pp().image_backend

    @image_backend.setter
    def image_backend(self, val: str):
        # check if there is at least one image preprocessor
        self.image_backend

        for i, p in enumerate(self._preprocessors):
            if not isinstance(p, _Image_Preprocessor):
                continue
            if val == p.image_backend:
                continue
            new_preprocessor = create_image_preprocessor(
                self._model_parameters, image_backend=val
            )
            self._preprocessors[i] = new_preprocessor
            # reassign previous property values to new backend
            new_preprocessor.resize_method = p.resize_method
            new_preprocessor.pad_method = p.pad_method
            new_preprocessor.fill_color = p.fill_color
            new_preprocessor.input_colorspace = p.input_colorspace
            new_preprocessor.crop_percentage = p.crop_percentage

    @property
    def resize_method(self) -> str:
        """Image resize method: one of 'nearest', 'bilinear', 'area', 'bicubic', 'lanczos'; default is 'bilinear'"""
        return self._get_image_pp().resize_method

    @resize_method.setter
    def resize_method(self, value: str):
        for p in self._image_property_iter():
            p.resize_method = value

    @property
    def pad_method(self) -> str:
        """Image padding method: one of 'stretch', 'letterbox'; default is 'letterbox'"""
        return self._get_image_pp().pad_method

    @pad_method.setter
    def pad_method(self, value: str):
        for p in self._image_property_iter():
            p.pad_method = value

    @property
    def fill_color(self) -> tuple:
        """fill color in case of 'letterbox' padding; 3-element RGB tuple, default is (0,0,0)"""
        return self._get_image_pp().fill_color

    @fill_color.setter
    def fill_color(self, value: tuple):
        value = validate_color_tuple(value)
        for p in self._image_property_iter():
            p.fill_color = value

    @property
    def input_colorspace(self) -> str:
        """input color space - one of 'auto', 'RGB' or 'BGR'; default is 'RGB'"""
        return self._get_image_pp().input_colorspace

    @input_colorspace.setter
    def input_colorspace(self, value: str):
        for p in self._image_property_iter():
            p.input_colorspace = value

    @property
    def crop_percentage(self) -> float:
        """Center crop percentage - float value between 0 and 1"""
        return self._get_image_pp().crop_percentage

    @crop_percentage.setter
    def crop_percentage(self, value: float):
        for p in self._image_property_iter():
            p.crop_percentage = value

    @property
    def image_format(self) -> str:
        """input color space - one of 'auto', 'RGB' or 'BGR'; default is 'RGB'"""
        return self._get_image_pp().image_format

    @image_format.setter
    def image_format(self, val: str):
        for p in self._image_property_iter():
            p.image_format = val

    @property
    def generate_image_result(self) -> bool:
        """Enable/disable generation of `image_result` key in the preprocessor result dictionary"""
        return self._get_image_pp().generate_image_result

    @generate_image_result.setter
    def generate_image_result(self, val: bool):
        for p in self._image_property_iter():
            p.generate_image_result = val

    @log_wrap
    def forward(self, data):
        """Perform input data preprocessing.
        - `data`: list of input data object

        Returns
        - list of bytes sequence suitable to feed to the AI model with given parameters.
        - list of dictionaries:
            `image_input` is RGB image object before resize, native to selected graphical backend:
                PIL.Image for 'pil' and numpy.ndarray for 'opencv'.
            `converter` is conversion lambda from resized to original image coordinates.
            `image_result` is the image object corresponding to `raw_result`
        """
        data = data if isinstance(data, list) else [data]
        if len(data) != len(self._preprocessors):
            raise DegirumException(
                f"The length of input data list '{len(data)} does not match to the number of model inputs '{len(self._preprocessors)}'"
            )

        results = []
        info = []
        for d, p in zip(data, self._preprocessors):
            result = p.forward(d)
            results.append(result["raw_result"])
            del result["raw_result"]
            info.append(result)

        results = results[0] if len(results) == 1 else results
        info = info[0] if len(info) == 1 else info
        return results, info
