"""
Some of the functions here are taken from the Modifier class we had before
"""

from typing import Dict

import torch
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn
import numpy as np

from .quantized_funcs import quantized_func_map
from .quantized_modules import quantized_module_map

from torch import nn

from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn


type_to_name_map = {
    nn.Linear: "linear",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
}

###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file is used for modify layers/modules in the maze graph
# suitable path is : mase/machop/chop/passes/graph/transforms/quantize/modify_tensorRT.py
############################################################


def create_new_module_tensorRT(
    mase_op: str,
    original_module: nn.Module,
    config: dict,
    node_meta: dict,
    baseline_module: nn.Module = None,
    successor_module: nn.Module = None,
    input_layers=None,
    output_layers=None,
    fake = True,
):
    print("fake:",fake)
    original_module_cls = type(original_module)
    quant_name = config.get("name")

    if mase_op == "linear":
        import pdb; pdb.set_trace()
        new_module_cls = quantized_module_map[f"linear_{quant_name}"]
        use_bias = original_module.bias is not None
        # NOTE: We don't support training with pruning on base module. Only quantized modules for now.
        use_pruning = any(
            isinstance(original_module, quantized_module)
            for quantized_module in quantized_module_map.values()
        ) and (original_module.pruning_masks is not None)
        if fake == True:
            new_module = quant_nn.Linear(
                in_features = original_module.in_features,
                out_features = original_module.out_features,
                bias=True,
                quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW)
        # else:
        #     new_module = quant_nn.Linear(
        #     in_features = original_module.in_features,
        #     out_features = original_module.out_features,
        #     bias=True,
        #     quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR_Not_Fake,
        #     quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW_Not_Fake)

    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module


def create_new_fn(node, config: Dict):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"]
    quant_name = config.get("name")
    func_name = f"{mase_op}_{quant_name}"
    new_func = quantized_func_map[func_name]
    args, kwargs = node.args, node.kwargs | {"config": config}
    return new_func, args, kwargs
