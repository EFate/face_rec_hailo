# PySDK Postprocessors

## Concepts

A **postprocessor complex** in PySDK is responsible for converting the raw output tensors of a model into a more usable format
and drawing graphical AI annotations over the original image.

A postprocessor complex in PySDK consists of two parts:

- core-level postprocessor,
- PySDK-level postprocessor.

For short, we will refer to them as **core postprocessor** and **renderer**.

The core postprocessor is a class that takes the raw output tensors of a model and converts them into a more usable format.
For example, the core postprocessor for classification models takes the raw output tensors and converts them into a 
list of class labels and their corresponding probabilities.

The core postprocessor is executed on the AI server side. It provides data reduction to reduce the 
amount of data that needs to be sent from AI server to clients.

The PySDK-level postprocessor, or renderer, is a class that takes the output of core postprocessor and 
can draw graphical AI annotations over the original image. The renderer is executed on the client side. 


### Core Postprocessors

There can be two types of core postprocessors:

- built-in postprocessors
- Python postprocessors

Built-in postprocessors are implemented as a part of the AI server (compiled into it). They are used for the most common model types.
Adding a new built-in postprocessor requires modifying the AI server code and recompiling it.

To specify the built-in postprocessor type to be used by the model, you need to define the `OutputPostprocessType` model parameter 
in the `POST_PROCESS` section of the model JSON file.
The built-in postprocessor type is a string that specifies the type of postprocessor to use. It should match one of the built-in postprocessor types defined 
in the AI server code. For example, the built-in postprocessor type for classification models is `"Classification"`.

There is a special built-in postprocessor type called `"None"`. It is used when you do not need (or do not want) any postprocessing. When such 
postprocessor type is used, the **bypass postprocessor** is applied: it just returns raw output tensors.
The bypass postprocessor will also be used in the case when you specify the postprocessor type unknown to the AI server core.

Python postprocessors are implemented in Python and are distributed with the model as model artifacts.
They are loaded dynamically by the AI server at runtime. Adding a new Python postprocessor does not require recompiling the AI server.
To specify the Python postprocessor, you need to define the `PythonFile` model parameter in the `POST_PROCESS` section of the model JSON file
to be equal to the name of the Python file that contains the postprocessor class.

When the `PythonFile` model parameter is defined, the AI server core ignores the `OutputPostprocessType` model parameter (but the PySDK client will not, 
more on that later).


### Renderers

Renderers are implemented in PySDK client and executed on a client side. They are used to draw graphical AI annotations over the original image.
Each renderer is implemented as a class that inherits from the `InferenceResults` base class:

- It keeps the original image and the output of the core postprocessor (aka inference results).
- On construction, typical renderer recalculates all image coordinates in the inference results to the original image coordinates.
- It implements `image_overlay` property that makes a copy of original image, draws AI annotation based on the inference results and
returns the annotated image.
- It implements `supported_types` static method which returns the list of postprocessor types (as defined in `OutputPostprocessType` model parameter) this renderer supports.

PySDK comes with a set of built-in renderers for the most common model types.
All renderers are implemented in the `postprocessor` sub-package of `degirum` package. 

The renderer module filename must be constructed as follows: `_<type>Results.py`, where `<type>` is some designator of the postprocessor type 
(can be any reasonable name). The renderer class name must be `<type>Results` (i.e. it must match the module name to be correctly discovered by PySDK).

When PySDK loads postprocessor sub-package, it imports all modules in the sub-package directory matching the pattern `_<type>Results.py`.
Then it gets a renderer class from the module by the name `<type>Results` and calls `supported_types` static method to get a list
of postprocessor types this renderer supports. Then it registers all supported postprocessor types in the `_supported_types_map` dictionary by
adding the mapping `<type> -> <type>Results` to it for each supported postprocessor type `<type>`.

When PySDK instantiates the renderer, it takes the postprocessor type from the `OutputPostprocessType` model parameter and searches for the 
renderer class that supports this postprocessor type in the `_supported_types_map` dictionary. If the renderer class is found, it is instantiated and returned.

You may register your own renderer class by calling `degirum.postprocessor.register_postprocessor` function passing the postprocessor type string 
and the renderer class. A renderer registered this way will be used exactly like a built-in renderer.


## How to Implement New Renderer

1. Invent a name for your postprocessor type. It should be a string that describes the postprocessor type. 
   For this example we will use `Awesome` postprocessor type string.
1. Create a new Python file in the `postprocessor` sub-package of `degirum` package named after your postprocessor type: `_AwesomeResults.py`.
1. In `_AwesomeResults.py` define a new class named `AwesomeResults` that inherits from `InferenceResults` base class imported as `from ._InferenceResults import InferenceResults`.
1. In `AwesomeResults` class implement static method `supported_types` which returns a list of postprocessor types this renderer supports or just a string,
    if only one type is supported. For this example we will return the string `"Awesome"`.
1. In `AwesomeResults` class implement constructor which must call `super().__init__(*args, **kwargs)` first. If you need to 
    recalculate image coordinates in the inference results to the original image coordinates, do it after that. Inference results
    are stored in the `self._inference_results` list. Call `self._conversion(x,y)` to convert (x,y) coordinate from model input coordinate system to original 
    image coordinate system. 
1. If you need to, implement `image_overlay` property which makes a copy of original image taken from `self._input_image`, draws AI annotation based on the 
    inference results and returns the annotated image. Use `create_draw_primitives` function imported as `from .._draw_primitives import create_draw_primitives` 
    to create a drawing object, which supports both PIL and OpenCV drawing. Refer to existing renderers code for examples.
1. If you want some fancy printing of inference results, implement `__str__` method.
1. In the model JSON file, define the `OutputPostprocessType` model parameter in the `POST_PROCESS` section to be equal to `"Awesome"`.

NOTE: if `"Awesome"` postprocessor type is not known to the AI server core, it will be ignored and the bypass postprocessor will be used instead, so
your `AwesomeResults` objects will receive raw output tensors into `self._inference_results`. If you want to do some core-level postprocessing, you either need 
to implement full-fledged core postprocessor in C++ and compile it into the AI server, or implement a Python postprocessor and define the `PythonFile` model 
parameter in the `POST_PROCESS` section (see below).

### Simple Example

The following example shows how to implement a simple renderer for the `Awesome` postprocessor type.
It expects inference results in a format which is similar to detection results having bounding boxes, labels, and scores.

The constructor converts the bounding box coordinates from model input coordinate system to original image coordinate system.

The `image_overlay` property draws the bounding boxes and labels over the original image and returns the annotated image.

The `__str__` method prints the inference results in a human-readable format.

```python

import yaml
from .._InferenceResults import InferenceResults, _ListFlowTrue
from .._draw_primitives import create_draw_primitives

class AwesomeResults(InferenceResults):

    @staticmethod
    def supported_types():
        return "Awesome"  # we can return string if only one type is supported

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # convert image coordinates to original image coordinates for all bounding boxes
        for res in self._inference_results:
            if "bbox" in res:
                box = res["bbox"]
                res["bbox"] = [
                    *self._conversion(*box[:2]),
                    *self._conversion(*box[2:]),
                ]

    @property
    def image_overlay(self):

        # create drawing object
        draw = create_draw_primitives(
            self._input_image, self._alpha, self._alpha, self._font_scale
        )

        colors = (
            self._overlay_color
            if isinstance(self._overlay_color, list)
            else [self._overlay_color]
        )

        for res in self._inference_results:
            # draw bounding boxes
            box = res.get("bbox", None)
            if box is not None:
                id = res.get("category_id", None)
                color = colors[id % len(colors)] if id is not None else colors[0]
                draw.draw_box(*box, self._line_width, color)

                capt = ""
                if self._show_labels:
                    label = res.get("label", None)
                    if label is not None:
                        capt = label

                if self._show_probabilities:
                    score = res.get("score", None)
                    if score is not None:
                        capt = (
                            f"{label}: {InferenceResults._format_num(score)}"
                            if label
                            else InferenceResults._format_num(score)
                        )

                if capt:
                    draw.draw_text_label(*box, color, capt, self._line_width)

        return draw.image_overlay()

    def __str__(self):
        results = copy.deepcopy(self._inference_results)
        for res in results:
            if "bbox" in res:
                # to print bbox in a single line
                res["bbox"] = _ListFlowTrue(res["bbox"])

        return yaml.dump(res, sort_keys=False)

```

## How to Implement New Python Postprocessor

1. Create new Python file in the model artifact directory. 
1. Define a new class that must be named `PostProcessor`
1. In `PostProcessor` class define three methods:
    - `def __init__(self)` constructor which takes no arguments and initializes the class.
    - `def configure(self, json_config)` method which takes a model configuration dictionary as an argument. 
        This method is called by the AI server core to configure the postprocessor.
    - `def forward(self, tensor_list, details_list)` method which takes a list of tensors and a list of details as arguments. 
        This method is called by the AI server core to do post-processing of one frame.
1. In the `configure` method you can take all the model parameters you will need from the `json_config` dictionary and store them in the class instance.
1. In the `forward` method you do all the post-processing. 
    - The `tensor_list` argument is a list of model output tensors. Each tensor is just a numpy array.
    - The `details_list` argument is a list of tensor details. Each element of this list is a dictionary compatible with TFLite tensor details dictionary
        as described in the [TensorFlow Lite documentation](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter#get_input_details). 
    - You should return a list of dictionaries, where each dictionary is a post-processed result; for example, for detection models it can be 
        a list of bounding boxes, labels, and scores.
1. In the model parameters JSON file assign `PythonFile` model parameter in the `POST_PROCESS` section to the name of the Python file you created in the first step.


### Simple Example

The following example shows how to implement simple regression post-processor.

The constructor is empty, but you can use it to initialize any class variables you need.

The `configure` method takes the model configuration dictionary and extracts the `RegressionMaxValue`, `RegressionMinValue`, and `LabelsPath` parameters.
Then it loads the labels from the file specified in `LabelsPath` parameter.

The `forward` method takes the output tensor and the details lists, takes the first tensor, assuming it has only one value, 
dequantizes this value, scales it to the range defined by `RegressionMaxValue` and `RegressionMinValue`,
and returns a single-element list containing a dictionary with the label and the regression value.


```python

import numpy as np
import json


class PostProcessor:
    def __init__(self):
        pass

    def configure(self, json_config):
        self._max_value = json_config["POST_PROCESS"][0]["RegressionMaxValue"]
        self._min_value = json_config["POST_PROCESS"][0]["RegressionMinValue"]
        self._labels = json.load(
            open(json_config["POST_PROCESS"][0]["LabelsPath"], "r")
        )

    def forward(self, tensor_list, details_list):
        qp = details_list[0]["quantization_parameters"]
        scale = qp["scales"][0]
        offset = qp["zero_points"][0]

        value_dequant = (float(np.squeeze(tensor_list[0])) - offset) * scale
        value = (
            value_dequant * (self._max_value - self._min_value) / 6.0 + self._min_value
        )

        results = [{"label": self._labels["0"], "score": value}]
        return json.dumps(results)
```
