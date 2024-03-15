import torch
import operator
import copy

def if_tensor_is_resized_to_full_then_resize_it_to_0_at_end_of_graph(mod):
    # FSDP graph has this invariant that if a tensor needs to be resized to full during execution of the graph, it *will* be resized to 0 again before exit of graph.
    tensors_resized = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] > 0:
            tensors_resized.add(n.args[0])
    for tensor in list(tensors_resized):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (tensor, 0), {})
        mod.graph.lint()
        mod.recompile()


def move_resize_to_0_to_end_of_graph(mod):
    # This pass is always a good idea to do so to avoid any use-after-free issues.
    resize_to_0_nodes = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] == 0:
            resize_to_0_nodes.add(n)
    for resize_to_0_node in list(resize_to_0_nodes):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (resize_to_0_node.args[0], 0), {})
        mod.graph.erase_node(resize_to_0_node)
        mod.graph.lint()
        mod.recompile()


def replace_primal_clone_at_beginning_of_graph_with_primal(mod):
    # Replace `clone(primal)` at beginning of graph with `primal`.
    # This is only safe if the graph does not have any autograd-affecting mutations and not explicitly cloning the primal through user code.
    # (i.e. only `with no_grad(): foreach_copy_` and `resize_storage_bytes_` is supported now).
    # TODO add checks to make sure the above invariant is maintained.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if n.op != "placeholder" and n.target is not torch.ops.inductor.resize_storage_bytes_.default:
            if n.target is torch.ops.aten.clone.default:
                if n.args[0] in primal_inputs_tensor_only:
                    n.replace_all_uses_with(n.args[0])
                    mod.graph.erase_node(n)
                    mod.graph.lint()
                    mod.recompile()
            else:
                break


def replace_primal_noop_as_strided_with_primal(mod):
    # Replace `as_strided(primal, ...)` with `primal`, if the as_strided is a no-op based on size and stride info. Should be always safe to do.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if (
            n.target is torch.ops.aten.as_strided.default \
            and n.args[0] in primal_inputs_tensor_only \
            and n.meta['val'].shape == n.args[0].meta['val'].shape \
            and n.meta['val'].stride() == n.args[0].meta['val'].stride()
        ):
            n.replace_all_uses_with(n.args[0])
            mod.graph.erase_node(n)
            mod.graph.lint()
            mod.recompile()


def input_is_used_in_other_ops(ops, inp_n, allowlist_callback=lambda n: False):
    for n in ops:
        if (not allowlist_callback(n)) and inp_n in flatten_arg_list(n.args):
            return True
    return False


def reinplace_foreach_copy_if_input_has_no_other_use_in_graph(mod):
    """
    _foreach_copy_1 = torch.ops.aten._foreach_copy.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    getitem_44: "f32[2, 76137800]" = _foreach_copy_1[0]
    getitem_45: "f32[2, 6170]" = _foreach_copy_1[1]
    getitem_46: "f32[2, 76137800]" = _foreach_copy_1[2]
    getitem_47: "f32[2, 6170]" = _foreach_copy_1[3];  _foreach_copy_1 = None

    ->

    # _foreach_copy__1 = torch.ops.aten._foreach_copy_.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    ... = torch.ops.aten.copy_.default(primal_1, getitem_28)
    ... = torch.ops.aten.copy_.default(primal_2, getitem_33)
    ... = torch.ops.aten.copy_.default(primal_3, getitem_38)
    ... = torch.ops.aten.copy_.default(primal_4, getitem_43)
    """
    # Use schema.is_mutable to detect inplace ops
    # TODO: this pass is maybe super slow, need optimization
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten._foreach_copy.default:
            _foreach_copy_outplace_node = n
            if all(not input_is_used_in_other_ops(
                node_list,
                inp_n,
                allowlist_callback=lambda n_: (
                    n_ == _foreach_copy_outplace_node \
                    or (n_.target is torch.ops.inductor.resize_storage_bytes_.default) \
                    or (n_.target is torch.ops.aten.copy_.default and n_.args[0] == inp_n and n_.args[1].target is operator.getitem and n_.args[1].args[0] == _foreach_copy_outplace_node and n_.args[1].args[1] == inp_ind)
                )  # ignore 1) this op, 2) resize_storage_bytes_ ops, 3) copy_ op that just copies the getitem back into the foreach_copy input
            ) for inp_ind, inp_n in enumerate(_foreach_copy_outplace_node.args[0])):
                with mod.graph.inserting_before(_foreach_copy_outplace_node):
                    for i, arg in enumerate(_foreach_copy_outplace_node.args[0]):
                        copy_to = arg
                        copy_from = _foreach_copy_outplace_node.args[1][i]
                        # _foreach_copy_inplace_node = mod.graph.call_function(torch.ops.aten._foreach_copy_.default, _foreach_copy_outplace_node.args, {})
                        # NOTE: Inductor seems to fail when encountering `_foreach_copy_` op. Need more investigation.
                        mod.graph.call_function(torch.ops.aten.copy_.default, (copy_to, copy_from), {})
                getitem_nodes = set()
                for node in node_list[i+1:]:
                    if node.target is operator.getitem and node.args[0] == _foreach_copy_outplace_node:
                        getitem_nodes.add(node)
                for node in getitem_nodes:
                    node.replace_all_uses_with(_foreach_copy_outplace_node.args[0][node.args[1]])
                    mod.graph.erase_node(node)
                mod.graph.erase_node(_foreach_copy_outplace_node)
                mod.graph.lint()
                mod.recompile()


def remove_inplace_copy_into_the_same_buffer(mod):
    """
    copy__1: "f32[12340]" = torch.ops.aten.copy_.default(arg4_1, arg4_1)

    ->

    deleted
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.copy_.default:
            inplace_copy_node = n
            inplace_copy_to = n.args[0]
            inplace_copy_from = n.args[1]
            if inplace_copy_to == inplace_copy_from:
                mod.graph.erase_node(inplace_copy_node)
                mod.graph.lint()
                mod.recompile()
                continue


def if_inplace_copy_output_has_no_other_use_and_is_then_resized_to_0_then_use_this_output_buffer_to_replace_all_occurrence_of_the_input(mod):
    """
    ... (no use of `arg2_1`)
    copy__default_6: "f32[12340, 12340]" = torch.ops.aten.copy_.default(as_strided, as_strided_3);  as_strided_3 = None
    ... (uses `as_strided`)
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(arg2_1, as_strided)
    ... (no use of `arg2_1`)
    resize_storage_bytes__default_8 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 0);  arg2_1 = None

    ->

    ... (no use of `arg2_1`)
    copy__default_6: "f32[12340, 12340]" = torch.ops.aten.copy_.default(arg2_1, as_strided_3);  as_strided_3 = None
    ... (uses `arg2_1`)
    resize_storage_bytes__default_8 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 0);  arg2_1 = None
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.copy_.default:
            inplace_copy_node = n
            inplace_copy_to = n.args[0]
            inplace_copy_from = n.args[1]
            for j, node in enumerate(node_list[i+1:]):
                if node.target is torch.ops.inductor.resize_storage_bytes_.default and node.args[0] == inplace_copy_to and node.args[1] == 0:
                    mod.graph.erase_node(inplace_copy_node)
                    inplace_copy_from.replace_all_uses_with(inplace_copy_to)
                    mod.graph.erase_node(inplace_copy_from)
                    mod.graph.lint()
                    mod.recompile()
                    break
                elif inplace_copy_to in flatten_arg_list(node.args):
                    # `inplace_copy_to` is used in a subsequent (non-`resize_`) op, so we don't do the replacement in this case
                    break


def reinplace_copy_if_input_has_no_other_use_in_graph(mod):
    """
    copy: "f32[304575880]" = torch.ops.aten.copy.default(slice_scatter_7, wait_tensor);  slice_scatter_7 = wait_tensor = None

    ->

    copy = torch.ops.aten.copy_.default(slice_scatter_7, wait_tensor);
    """
    for n in list(mod.graph.nodes):
        if n.target is torch.ops.aten.copy.default:
            copy_outplace_node = n
            inp_n = copy_outplace_node.args[0]
            if not input_is_used_in_other_ops(
                list(mod.graph.nodes),
                inp_n,
                allowlist_callback=lambda n: (n == copy_outplace_node or (n.target is torch.ops.inductor.resize_storage_bytes_.default))  # ignore this op, and ignore resize_storage_bytes_ ops
            ):
                with mod.graph.inserting_before(copy_outplace_node):
                    mod.graph.call_function(torch.ops.aten.copy_.default, (inp_n, copy_outplace_node.args[1]), {})
                copy_outplace_node.replace_all_uses_with(inp_n)
                mod.graph.erase_node(copy_outplace_node)
                mod.graph.lint()
                mod.recompile()


def remove_clone_if_input_has_no_other_use_in_graph(mod):
    """
    clone: "f32[2, 12340]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None

    ->

    expand
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.clone.default:
            clone_inp = n.args[0]
            if not input_is_used_in_other_ops(node_list, clone_inp, allowlist_callback=lambda n_: n_ == n):
                n.replace_all_uses_with(clone_inp)
                mod.graph.erase_node(n)
                mod.graph.lint()
                mod.recompile()

def replace_noop_consecutive_permutes_with_original_input_if_first_permute_out_has_no_other_use(mod):
    """
    # NOTE: we only handle len(permute_dims) = 2 case for now.

    permute_3: "f32[12340, 12340]" = torch.ops.aten.permute.default(getitem_106, [1, 0])
    permute_4: "f32[12340, 12340]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

    ->

    getitem_106
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.permute.default and len(n.args[1]) == 2:
            permute_dims = n.args[1]
            first_permute_node = n
            second_permute_node = None
            first_permute_output_has_other_use = False
            # First check that the first permute output has no other use
            for j, node in enumerate(node_list[i+1:]):
                if first_permute_node in flatten_arg_list(node.args):
                    if node.target is not torch.ops.aten.permute.default:
                        first_permute_output_has_other_use = True
                    else:
                        if node.args[1] == permute_dims:
                            # if permute_dims also match, we know these two consecutive permutes lead to a no-op.
                            second_permute_node = node
                        else:
                            first_permute_output_has_other_use = True
            if second_permute_node is not None and not first_permute_output_has_other_use:
                second_permute_node.replace_all_uses_with(first_permute_node.args[0])
                mod.graph.erase_node(second_permute_node)
                mod.graph.erase_node(first_permute_node)
                mod.graph.lint()
                mod.recompile()


def replace_as_strided_scatter_with_primal_if_primal_has_no_other_use_after_this_op(mod):
    """
    as_strided_scatter_3: "f32[12340]" = torch.ops.aten.as_strided_scatter.default(primals_4, view_15, [12340], [1], 0);

    ->

    primals_4
    """
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for i, n in enumerate(list(mod.graph.nodes)):
        if n.target is torch.ops.aten.as_strided_scatter.default:
            as_strided_scatter_node = n
            if as_strided_scatter_node.args[0] in primal_inputs_tensor_only:
                primal = as_strided_scatter_node.args[0]
                if (
                    primal.meta['val'].shape == as_strided_scatter_node.meta['val'].shape \
                    and primal.meta['val'].stride() == as_strided_scatter_node.meta['val'].stride() \
                    and not input_is_used_in_other_ops(list(mod.graph.nodes)[i+1:], primal, allowlist_callback=lambda n: n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] == 0)
                ):
                    as_strided_scatter_node.replace_all_uses_with(primal)
                    mod.graph.erase_node(as_strided_scatter_node)
                    mod.graph.lint()
                    mod.recompile()


def use_input_as_output_for_inplace_copy_ops(mod):
    """
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(getitem_2, as_strided)
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(copy_, add_1)
    ->
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(getitem_2, as_strided)
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(getitem_2, add_1)
    """
    for n in list(mod.graph.nodes):
        if n.target is torch.ops.aten.copy_.default:
            left_inp_n = n.args[0]
            n.replace_all_uses_with(left_inp_n)
            mod.graph.lint()
            mod.recompile()


def flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def if_tensor_is_resized_to_0_immediately_after_inplace_copy_then_delete_the_copy(mod):
    """
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(primals_6, getitem_44)
    resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  primals_6 = None
    ->
    resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  primals_6 = None
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.copy_.default:
            inplace_copy_inp = n.args[0]
            for j, node in enumerate(node_list[i+1:]):
                if inplace_copy_inp in flatten_arg_list(node.args):
                    if node.target is torch.ops.inductor.resize_storage_bytes_.default and node.args[1] == 0:
                        mod.graph.erase_node(n)
                        mod.graph.lint()
                        mod.recompile()
                        break
                    else:
                        break


def dedup_resize_to_same_size(mod):
    """
    NOTE: dedup by keeping the last one (i.e. the first one in reversed node order)

    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 609102400)
    resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 609102400)

    ->

    resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 609102400)
    """
    node_list = list(mod.graph.nodes)
    resize_input_to_node = {}
    for n in reversed(node_list):
        if n.target is torch.ops.inductor.resize_storage_bytes_.default:
            resize_input = n.args[0]
            resize_to_size = n.args[1]
            if resize_input in resize_input_to_node and resize_input_to_node[resize_input].args[1] == resize_to_size:
                mod.graph.erase_node(n)
            else:
                resize_input_to_node[resize_input] = n
    mod.graph.lint()
    mod.recompile()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def remove_all_slice_and_slice_scatter_following_split_with_sizes_and_inplace_copy(mod):
    """
    NOTE: essentially undoing functionalization for the inplace copy

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:52 in foreach_all_gather, code: foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(slice_6, [76137800, 6170, 76137800, 6170]);  slice_6 = None
    getitem_8: "f32[76137800]" = split_with_sizes_2[0]
    getitem_9: "f32[6170]" = split_with_sizes_2[1]
    getitem_10: "f32[76137800]" = split_with_sizes_2[2]
    getitem_11: "f32[6170]" = split_with_sizes_2[3];  split_with_sizes_2 = None

    # No stacktrace found for following nodes
    copy__default: "f32[76137800]" = torch.ops.aten.copy_.default(getitem_8, arg7_1);  arg7_1 = None
    copy__default_1: "f32[6170]" = torch.ops.aten.copy_.default(getitem_9, arg8_1);  arg8_1 = None
    copy__default_2: "f32[76137800]" = torch.ops.aten.copy_.default(getitem_10, arg9_1);  arg9_1 = None
    copy__default_3: "f32[6170]" = torch.ops.aten.copy_.default(getitem_11, arg10_1);  arg10_1 = None

    slice_tensor_4: "f32[152287940]" = torch.ops.aten.slice.Tensor(empty_1, 0, 0, 152287940)
    slice_scatter_default_8: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, getitem_8, 0, 0, 76137800);  slice_tensor_4 = getitem_8 = None
    slice_scatter_default_9: "f32[304575880]" = torch.ops.aten.slice_scatter.default(empty_1, slice_scatter_default_8, 0, 0, 152287940);  empty_1 = slice_scatter_default_8 = None
    slice_tensor_5: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_default_9, 0, 0, 152287940)
    slice_scatter_default_10: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_tensor_5, getitem_9, 0, 76137800, 76143970);  slice_tensor_5 = getitem_9 = None
    slice_scatter_default_11: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_9, slice_scatter_default_10, 0, 0, 152287940);  slice_scatter_default_9 = slice_scatter_default_10 = None
    slice_tensor_6: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 0, 152287940)
    slice_scatter_default_12: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, getitem_10, 0, 76143970, 152281770);  slice_tensor_6 = getitem_10 = None
    slice_scatter_default_13: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_11, slice_scatter_default_12, 0, 0, 152287940);  slice_scatter_default_11 = slice_scatter_default_12 = None
    slice_tensor_7: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_default_13, 0, 0, 152287940)
    slice_scatter_default_14: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_tensor_7, getitem_11, 0, 152281770, 152287940);  slice_tensor_7 = getitem_11 = None
    slice_scatter_default_15: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_13, slice_scatter_default_14, 0, 0, 152287940);  slice_scatter_default_13 = slice_scatter_default_14 = None

    ... = some_op(slice_scatter_default_15)

    ->

    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(slice_6, [76137800, 6170, 76137800, 6170]);  slice_6 = None
    getitem_8: "f32[76137800]" = split_with_sizes_2[0]
    getitem_9: "f32[6170]" = split_with_sizes_2[1]
    getitem_10: "f32[76137800]" = split_with_sizes_2[2]
    getitem_11: "f32[6170]" = split_with_sizes_2[3];  split_with_sizes_2 = None

    # No stacktrace found for following nodes
    copy__default: "f32[76137800]" = torch.ops.aten.copy_.default(getitem_8, arg7_1);  arg7_1 = None
    copy__default_1: "f32[6170]" = torch.ops.aten.copy_.default(getitem_9, arg8_1);  arg8_1 = None
    copy__default_2: "f32[76137800]" = torch.ops.aten.copy_.default(getitem_10, arg9_1);  arg9_1 = None
    copy__default_3: "f32[6170]" = torch.ops.aten.copy_.default(getitem_11, arg10_1);  arg10_1 = None

    ... = some_op(slice_6)
    """
    node_list = list(mod.graph.nodes)
    for i in range(len(node_list)):
        n = node_list[i]
        if n.target is torch.ops.aten.split_with_sizes.default:
            split_with_sizes_input = n.args[0]
            chunk_sizes = n.args[1]
            num_chunks = len(chunk_sizes)
            if i + num_chunks * 5 >= len(node_list):  # each chunk incurs 5 additional ops: getitem + copy_ + slice + slice_scatter + slice_scatter
                break
            if not all(n_.target is operator.getitem for n_ in node_list[i+1:i+num_chunks+1]):
                break
            if not all(n_.target is torch.ops.aten.copy_.default for n_ in node_list[i+num_chunks+1:i+num_chunks*2+1]):
                break
            if not all(
                oc[0].target is torch.ops.aten.slice.Tensor \
                and oc[1].target is torch.ops.aten.slice_scatter.default \
                and oc[2].target is torch.ops.aten.slice_scatter.default \
                for oc in chunks(node_list[i+num_chunks*2+1:i+num_chunks*5+1], 3)
            ):
                break
            last_slice_scatter_node = node_list[i + num_chunks * 5]
            assert last_slice_scatter_node.target is torch.ops.aten.slice_scatter.default
            last_slice_scatter_node.replace_all_uses_with(split_with_sizes_input)
            mod.graph.erase_node(last_slice_scatter_node)
            mod.graph.lint()
            mod.recompile()
