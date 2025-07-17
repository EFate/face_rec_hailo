import numpy as np
import json


# Post-processor class, must have fixed name 'PostProcessor'
class PostProcessor:
    def __init__(self, json_config):
        """
        Initialize the post-processor with configuration settings.

        Parameters:
            json_config (str): JSON string containing post-processing configuration.
        """

    def forward(self, tensor_list, details_list):
        """
        Dequantizes the raw tensor output using the provided scale and zero point.

        Parameters:
            tensor_list (list): List of tensors from the model.
            details_list (list): Additional metadata for the tensors.

        Returns:
            dict: Modified inference result with dequantized data and updated metadata.
        """

        ret = []
        for data, tensor_info in zip(tensor_list, details_list):
            # Dequantize the tensor
            quantization = tensor_info["quantization"]
            dequantized_data = (
                data.astype(np.float32) - quantization[1]
            ) * quantization[0]

            # Reshape to (1, x)
            reshaped_data = dequantized_data.flatten().reshape(1, -1)

            tensor = dict(
                id=tensor_info["index"],
                name=tensor_info["name"],
                shape=reshaped_data.shape,
                quantization=dict(axis=-1, scale=[1], zero=[0]),
                type="DG_FLT",
                size=reshaped_data.size,
                data=reshaped_data.tolist(),
            )
            ret.append(tensor)

        return ret
