from copy import copy, deepcopy
import logging
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from .modify import create_new_fn, create_new_module
from .modify_tensorRT import create_new_module_tensorRT
from .quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param
from .summary import graph_iterator_compare_nodes, graph_iterator_node_histogram

logger = logging.getLogger(__name__)


###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file have Quantization pass and calibration pass
# suitable path is: mase/machop/chop/passes/graph/transforms/quantize/quantize_tensorRT.py
############################################################

QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]

def graph_iterator_quantize_by_type_tensorRT(graph, config: dict,fake=False):
    # Some modules might need information from two graphs to be initilized
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        bl_graph = None
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            new_module = create_new_module_tensorRT(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                bl_module,
                successor_module,
                fake,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            update_quant_meta_param(node, node_config, get_mase_op(node))
            import pdb; pdb.set_trace()
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # new_node.meta["mase"].node -> new_node
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def tensorRT_quantize_pass(graph, pass_args=None,fake = False):
    print("hello world")
    import pdb; pdb.set_trace()
    by = pass_args["by"]
    match by:
        case "type":
            graph = graph_iterator_quantize_by_type_tensorRT(graph, pass_args,fake)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

def calibration_pass(graph, pass_args=None,data_module=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph.model.to(device)
    for name in graph.modules.keys():
        if name.endswith('_quantizer'):
            graph.modules[name].enable_calib()
            graph.modules[name].disable_quant()  # Use full precision data to calibrate
    
    count = 0
    if count <= 1:
        for inputs in data_module.train_dataloader():
            xs, ys = inputs
            xs, ys = xs.to(device), ys.to(device)
            graph.model(xs)
            count += 1

    for name in graph.modules.keys():
        if name.endswith('_quantizer'):
            graph.modules[name].load_calib_amax()
            graph.modules[name].disable_calib()
            graph.modules[name].enable_quant()
            import pdb; pdb.set_trace()
            print(f"Max absolute value for {name}: {graph.modules[name].amax}")
    graph.model.to(device)

    return graph, {}
