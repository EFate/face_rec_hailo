#
# _model_param_helpers.py - DeGirum Python SDK: helpers for model parameters handling
# Copyright DeGirum Corp. 2025
#
# Contains implementation of various helpers for model parameters handling
#

from .aiclient import ModelParams
from .exceptions import DegirumException
import copy


def model_shape_get(model_params: ModelParams, inp_idx: int, expected_size) -> list:
    """
    Helper method: deduce model input shape from model parameters.

    Args:
        model_params: model parameters
        inp_idx: input index; or -1 to get all inputs
        expected_size: expected size of the shape list, 0 if not checked
    Returns:
        model input shape list; or list of model input shape lists when inp_idx == -1
    """

    inp_idx_list = range(len(model_params.InputShape)) if inp_idx < 0 else [inp_idx]
    ret = []

    for ii in inp_idx_list:
        if model_params.InputShape_defined(ii) and model_params.InputShape[ii]:
            # InputShape has priority over InputN, InputH, InputW, InputC
            shape = model_params.InputShape[ii]
            if expected_size != 0 and len(shape) != expected_size:
                raise DegirumException(
                    f"The input shape parameter InputShape for input #{ii} must have "
                    f"{expected_size} elements, while it has {len(shape)}"
                )
            ret.append(copy.copy(shape))
        else:
            if expected_size != 0:
                # When expected_size is set, fill the shape list
                # and put N/H/W/C values in the right places
                shape = [1] * expected_size
                if (
                    model_params.InputN_defined(ii)
                    and model_params.InputN[ii] > 0
                    and expected_size >= 1
                ):
                    shape[0] = model_params.InputN[ii]
                if (
                    model_params.InputH_defined(ii)
                    and model_params.InputH[ii] > 0
                    and expected_size >= 2
                ):
                    shape[1] = model_params.InputH[ii]
                if (
                    model_params.InputW_defined(ii)
                    and model_params.InputW[ii] > 0
                    and expected_size >= 3
                ):
                    shape[2] = model_params.InputW[ii]
                if (
                    model_params.InputC_defined(ii)
                    and model_params.InputC[ii] > 0
                    and expected_size >= 4
                ):
                    shape[3] = model_params.InputC[ii]
                ret.append(shape)
            else:
                # When expected_size is not set, fill the shape list
                # with all the values that are defined
                shape = []
                if model_params.InputN_defined(ii) and model_params.InputN[ii] > 0:
                    shape.append(model_params.InputN[ii])
                if model_params.InputH_defined(ii) and model_params.InputH[ii] > 0:
                    shape.append(model_params.InputH[ii])
                if model_params.InputW_defined(ii) and model_params.InputW[ii] > 0:
                    shape.append(model_params.InputW[ii])
                if model_params.InputC_defined(ii) and model_params.InputC[ii] > 0:
                    shape.append(model_params.InputC[ii])
                ret.append(shape)

    return ret if inp_idx < 0 else ret[0]


def model_shape_set(model_params: ModelParams, inp_idx: int, new_shape: list):
    """
    Helper method: set model input shape in model parameters.
    Args:
        model_params: model parameters
        inp_idx: input index
        new_shape: model input shape list to set for given input index
    """

    if model_params.InputShape_defined(inp_idx) and model_params.InputShape[inp_idx]:
        # set InputShape if defined

        shape = model_params.InputShape
        shape[inp_idx] = new_shape
        model_params.InputShape = shape

    else:
        # otherwise set InputN/H/W/C
        shape_len = len(new_shape)

        N = model_params.InputN if model_params.InputN_defined(inp_idx) else None
        H = model_params.InputH if model_params.InputH_defined(inp_idx) else None
        W = model_params.InputW if model_params.InputW_defined(inp_idx) else None
        C = model_params.InputC if model_params.InputC_defined(inp_idx) else None

        idx = 0
        too_short = False
        if N is not None and N[inp_idx] > 0:
            if idx < shape_len:
                N[inp_idx] = new_shape[idx]
                model_params.InputN = N
                idx += 1
            else:
                too_short = True

        if H is not None and H[inp_idx] > 0:
            if idx < shape_len:
                H[inp_idx] = new_shape[idx]
                model_params.InputH = H
                idx += 1
            else:
                too_short = True

        if W is not None and W[inp_idx] > 0:
            if idx < shape_len:
                W[inp_idx] = new_shape[idx]
                model_params.InputW = W
                idx += 1
            else:
                too_short = True

        if C is not None and C[inp_idx] > 0:
            if idx < shape_len:
                C[inp_idx] = new_shape[idx]
                model_params.InputC = C
                idx += 1
            else:
                too_short = True

        if idx < shape_len:
            raise DegirumException(
                f"Cannot set input shape for input #{inp_idx}: too many dimensions {shape_len}; only {idx} are required"
            )

        if too_short:
            raise DegirumException(
                f"Cannot set input shape for input #{inp_idx}: not enough dimensions {shape_len}"
            )
