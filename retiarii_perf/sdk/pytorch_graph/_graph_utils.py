# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import queue
import re
from collections import defaultdict
import torch
from torch.utils.tensorboard._pytorch_graph import NodePy, NodePyIO, NodePyOP, GraphPy
CLASSTYPE_KIND = 'ClassType'
GETATTR_KIND = 'prim::GetAttr'
CAT_KIND = 'aten::cat'

_logger = logging.getLogger(__name__)


def build_module_graph(model, dummy_input):
    return TorchModuleGraph(model, dummy_input)


def build_graph(model, dummy_input, verbose=False):
    g = TorchProtoGraph(model, dummy_input, verbose)
    return g.graph_def, g.stepstats


def parse_traced_name(module_name):
    prefix = 'TracedModule['
    suffix = ']'
    if module_name.startswith(prefix) and module_name.endswith(suffix):
        module_name = module_name[len(prefix):-len(suffix)]
    return module_name


class TorchGraph:
    """
    This class is to extract pytorch model topology graph by tracing
    """

    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        traced_model : torch._C.torch.jit.TopLevelTracedModule
            An alredy traced model, if traced_model is not None, then TorchGraph will build the graph
            based on this traced model and won't trace the model again.
        """
        assert torch.__version__ >= '1.3.1'
        # check if the input is legal
        if traced_model is not None:
            assert isinstance(traced_model, torch.jit.TopLevelTracedModule)
            self.trace = traced_model
            # it's ok if the graph is already unpacked
            torch._C._jit_pass_inline(self.trace.graph)
        elif model is not None and dummy_input is not None:
            self.bound_model = model
            self._trace(model, dummy_input)
        else:
            raise Exception(
                'Please provide model & dummy_input or the traced_model as inputs')

    def _trace(self, model, dummy_input):
        with torch.onnx.set_training(model, False):
            self.trace = torch.jit.trace(model, dummy_input)
            torch._C._jit_pass_inline(self.trace.graph)


class TorchProtoGraph(TorchGraph):
    """
    Generates model graph for pytorch models in protobuf, this implementation
    is borrowed from pytorch v1.4.0, and fixed following issues:
    https://github.com/pytorch/pytorch/issues/33691
    https://github.com/pytorch/pytorch/issues/33670

    """

    def __init__(self, model, dummy_input, verbose=False):
        super().__init__(model, dummy_input)

        from tensorboard.compat.proto.config_pb2 import RunMetadata
        from tensorboard.compat.proto.graph_pb2 import GraphDef
        from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
        from tensorboard.compat.proto.versions_pb2 import VersionDef

        list_of_nodes = self.parse(self.trace.graph, self.trace, dummy_input)
        if verbose:
            print(self.trace.graph)
        self.stepstats = RunMetadata(step_stats=StepStats(
            dev_stats=[DeviceStepStats(device="/device:CPU:0")]))
        self.graph_def = GraphDef(
            node=list_of_nodes, versions=VersionDef(producer=22))

    def parse(self, graph, trace, args=None, omit_useless_nodes=True):
        """This method parses an optimized PyTorch model graph and produces
        a list of nodes and node stats for eventual conversion to TensorBoard
        protobuf format.

        Args:
        graph (PyTorch module): The model graph to be parsed.
        trace (PyTorch JIT TracedModule): The model trace to be parsed.
        args (tuple): input tensor[s] for the model.
        omit_useless_nodes (boolean): Whether to remove nodes from the graph.
        """
        nodes_py = GraphPy()
        for node in graph.inputs():
            if omit_useless_nodes:
                if not node.uses():  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != CLASSTYPE_KIND:
                nodes_py.append(NodePyIO(node, 'input'))

        attr_to_scope = dict()

        def node_to_name(d):
            return str(d).split(":")[0].strip()
        for node in graph.nodes():
            if node.kind() == GETATTR_KIND:
                attr_name = node.s('name')
                node_name = node_to_name(node)
                parent = node.input().node()
                # If the parent node is not the top-level "self" node
                if parent.kind() == GETATTR_KIND:
                    parent_scope = attr_to_scope[node_to_name(parent)]
                    attr_scope = parent_scope.split('/')[-1]
                    attr_to_scope[node_name] = '{}/{}.{}'.format(
                        parent_scope, attr_scope, attr_name)
                else:
                    attr_to_scope[node_name] = '__module.{}'.format(attr_name)
                # We don't need classtype nodes; scope will provide this information
                if node.output().type().kind() != CLASSTYPE_KIND:
                    node_py = NodePyOP(node)
                    node_py.scopeName = attr_to_scope[node_name]
                    nodes_py.append(node_py)
            else:
                nodes_py.append(NodePyOP(node))

        # Create sink nodes for output ops
        for i, node in enumerate(graph.outputs()):
            node_py = NodePyIO(node, 'output')
            node_py.debugName = "output.{}".format(i + 1)
            node_py.inputs = [node.debugName()]
            nodes_py.append(node_py)

        alias_to_name = dict()
        base_name = parse_traced_name(trace._name)
        for name, module in trace.named_modules(prefix='__module'):
            mod_name = parse_traced_name(module._name)
            attr_name = name.split('.')[-1]
            alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)

        for node in nodes_py.nodes_op:
            module_aliases = node.scopeName.split('/')[-1].split('.')
            module_name = ''
            for i, alias in enumerate(module_aliases):
                if i == 0:
                    module_name = alias
                    node.scopeName = base_name
                else:
                    module_name += '.' + alias
                    node.scopeName += '/' + \
                        (alias_to_name[module_name]
                         if module_name in alias_to_name else alias)

        nodes_py.populate_namespace_from_OP_to_IO()
        return nodes_py.to_proto()


class NodePyGroup(NodePy):
    """
    This class is used to represent a graph node which consists of multiple jit traced nodes. In a pytorch trace graph,
    there are multiple nodes are traced for one torch.nn.Module object, we group them together to form a single node to
    represent the torch.nn.Module object. We also group some functional call trace nodes together to form a new node.
    """

    def __init__(self, name, unique_name, node_type, op_type, node_cpps, inputs=None, outputs=None):
        """
        Parameters:
        -----------
        name: str
            node name, such as `conv1`, `backbone.classifier`
        unique_name: str
            A global unique name for current node. Due to some modules,
            such as relu, may be reused several times, so the scopename
            is not suitable as the global unique identifier, so we add a
            unique_name for each node as the global unique identifier.
            We should use the unique_name to traverset the module graph.
        node_type: str
            `module` or `func`
        op_type: str
            operation type, such as `Conv2d`, `aten::view`
        node_cpps: list of torch._C.Node
            jit trace nodes which are included in this new node
        inputs: list of str
            All the inputs of this node, each element is debugName of one input
        outputs: list of str
            All the outputs of this node, each element is debugName of one output
        """
        super(NodePyGroup, self).__init__(name, [])
        self.node_cpps = node_cpps
        self.name = name
        self.unique_name = unique_name
        self.op_type = op_type
        self.type = node_type
        self.nodes = []
        self.auxiliary = None
        self.add_nodes(node_cpps)
        self.inputs = inputs
        self.outputs = outputs

    def add_nodes(self, node_cpps):
        for node_cpp in node_cpps:
            nodepy = NodePyOP(node_cpp)
            nodepy.name = node_cpp.scopeName() + '_' + node_cpp.kind()
            self.nodes.append(nodepy)

    def sub_node_names(self):
        return [x.name for x in self.nodes]

    def __repr__(self):
        return 'name: {}, type: {}, op_type: {}, sub_nodes: {}, inputs: {}, outputs: {}, aux: {}'.format(
            self.name, self.type, self.op_type, self.sub_node_names(),
            self.inputs, self.outputs, self.auxiliary
        )


class TorchModuleGraph(TorchGraph):
    """
    Generates model graph, each node is created from single or multiple jit trace nodes.
    """

    def __init__(self, model=None, dummy_input=None, traced_model=None):
        super().__init__(model, dummy_input, traced_model)
        self.global_count = 0
        self.name_to_node, self.input_to_node, self.output_to_node = self._build_graph()
        self._extract_auxiliary_info()

    def _expand_non_prim_node(self, node, nodes, input_to_node, output_to_node,
                              module_type):
        """
        For trace graph nodes, some nodes are not in modules, these nodes are usually generated by
        the functions directly called in module ```forward```. For such nodes, some of them are
        trivial op which are label by ```prim::```, some of them are not such ops which is call
        non-prim ops. This function is to merge neighbor prim ops to a non-prim op, to construct
        a node.

        Parameters
        ----------
        node : trace graph node
            The non-prim node to expand
        nodes : list of trace graph node
            All the trace graph nodes within the same scope as the non-prim node
        input_to_node : dict
            key: input name, value: a node that uses this input
        output_to_node : dict
            key: output name, value: a node that generates this output
        module_type : str
            can be 'module' or 'func'

        Returns
        -------
        node
            the expanded non-prim node
        """
        # TODO: scope name could be empty
        node_name = '.'.join([self._get_module_name(
            node.scopeName()), node.kind(), str(self.global_count)])
        unique_name = node_name
        _logger.debug("expand non-prim node, node name: %s", node_name)
        self.global_count += 1
        op_type = node.kind()
        node_group = [node]
        inputs = list()
        outputs = list()
        node_queue = queue.Queue()
        node_queue.put(node)
        while not node_queue.empty():
            curr_node = node_queue.get()
            for _input in curr_node.inputs():
                input_name = _input.debugName()
                if input_name in output_to_node and output_to_node[input_name] in nodes:
                    predecessor_node = output_to_node[input_name]
                    if predecessor_node.kind().startswith('prim::'):
                        node_group.append(predecessor_node)
                        node_queue.put(predecessor_node)
                    else:
                        inputs.append(input_name)
                else:
                    inputs.append(input_name)
        for output in node.outputs():
            outputs.append(output.debugName())
        if not unique_name.isidentifier():
            unique_name = re.sub('[^0-9a-zA-Z_]', '_', unique_name)
        nodepy = NodePyGroup(node_name, unique_name, module_type, op_type,
                             node_group, inputs=inputs, outputs=outputs)

        # use DFS to retrieve input arguments
        def _visit_node(curr_node, arg_list, node_group):
            """
            Parameters
            ----------
            """
            for _input in curr_node.inputs():
                input_name = _input.debugName()
                if input_name in output_to_node and output_to_node[input_name] in node_group:
                    predecessor_node = output_to_node[input_name]
                    if predecessor_node.kind() == 'prim::ListConstruct':
                        arg_list.append([])
                        _visit_node(predecessor_node, arg_list[-1], node_group)
                    elif predecessor_node.kind() == 'prim::Constant':
                        args = [ o.toIValue() for o in predecessor_node.outputs() ]
                        assert len(args) == 1
                        if args[0] == None:
                            if curr_node.kind() not in ['aten::mean', 'aten::avg_pool2d', 'aten::ones_like', 'aten::to', 'aten::softmax', 'aten::sum', 'aten::arange']:
                                print('zql: ', curr_node)
                            assert curr_node.kind() in ['aten::mean', 'aten::avg_pool2d', 'aten::ones_like', 'aten::to', 'aten::softmax', 'aten::sum', 'aten::arange']
                            continue # to deal with: aten::mean has additional input None at the end
                        # TODO: this is urgly
                        # if it is tensor, use string "Tensor" instead
                        for i, arg in enumerate(args):
                            if type(arg) == torch.Tensor:
                                args[i] = 'Tensor'
                        arg_list.extend(args)
                    elif predecessor_node.kind() == 'prim::NumToTensor':
                        args = [ i.debugName() for i in predecessor_node.inputs() ]
                        assert len(args) == 1
                        arg_list.extend(args)
                    elif predecessor_node.kind() == 'prim::ListUnpack':
                        args = [ i.debugName() for i in predecessor_node.inputs() ]
                        assert len(args) == 1
                        arg_list.extend(args)
                    elif predecessor_node.kind().startswith('prim::'):
                        raise RuntimeError('Node type {} is not correctly handled during retrieving input arguments'.format(predecessor_node.kind()))
                    else:
                        raise RuntimeError('Got non prim node {} after prim node'.format(predecessor_node.kind()))
                else:
                    arg_list.append(input_name)
        input_args = list()
        _visit_node(node, input_args, node_group)
        if nodepy.auxiliary is not None:
            nodepy.auxiliary['input_args'] = input_args
        else:
            nodepy.auxiliary = { 'input_args': input_args }

        return nodepy

    def build_module_hierarchy(self):
        """
        After the graph is constructed from trace,
        module hierarchy can also be reconstructed from the trace.
        First reconstruct the scope name tree, then attach nodes to the tree.
        """
        class TreeNode:
            def __init__(self, local_name, full_name):
                self.local_name = local_name
                self.full_name = full_name
                self.graph_nodes = []
                self.child_tnodes = {}
            def add_child(self, child_tnode):
                assert child_tnode.local_name not in self.child_tnodes
                self.child_tnodes[child_tnode.local_name] = child_tnode
            def attach_node(self, graph_node):
                self.graph_nodes.append(graph_node)
        def insert_scope_name(root, name_slices):
            curr_node = root
            for i, name in enumerate(name_slices):
                if name not in curr_node.child_tnodes:
                    curr_node.child_tnodes[name] = TreeNode(name, '.'.join(name_slices[:i+1]))
                curr_node = curr_node.child_tnodes[name]
        def visualize_module_hierarchy(root):
            import graphviz
            vgraph = graphviz.Digraph(format='jpg')
            render_cfg = {'shape': 'ellipse', 'style': 'solid'}
            q = []
            q.append(root)
            vgraph.node(root.full_name, **render_cfg)
            while q:
                tnode = q.pop(0)
                for _, child in tnode.child_tnodes.items():
                    vgraph.node(child.full_name, **render_cfg)
                    vgraph.edge(tnode.full_name, child.full_name)
                    q.append(child)
            vgraph.render('tree_graph')
        def attach_node_to_tree(root, name_slices, node_pygroup):
            curr_node = root
            for name in name_slices:
                if name in curr_node.child_tnodes:
                    curr_node = curr_node.child_tnodes[name]
                else:
                    curr_node.graph_nodes.append(node_pygroup)
                    break

        # reconstruct the scope name tree
        assert torch.__version__ >= '1.4.0', 'Only support PyTorch >= 1.4.0'
        root_tnode = TreeNode('root', 'root')
        graph = self.trace.graph
        for node in graph.nodes():
            scope_name = node.scopeName().split('/')[-1].replace('__module.', '')
            name_slices = scope_name.split('.')
            insert_scope_name(root_tnode, name_slices)
        # visualize_module_hierarchy(root_tnode)

        # attach nodes to the tree
        if not hasattr(self, 'name_to_node'):
            print('do not have attr name_to_node')
            return
        for _, node_pygroup in self.name_to_node.items():
            name = node_pygroup.name
            #print(name)
            name_slices = name.split('.')
            attach_node_to_tree(root_tnode, name_slices, node_pygroup)
        return root_tnode

    def _expand_module_node(self, node, node_name, unique_name, op_type, nodes,
                            input_to_node, output_to_node, module_type):
        """
        merge the adjacent nodes of the module. The difference between the
        _expand_module_node and _expand_non_prim_node is that, the _expand_non_prim_node
        only merge the prim:: nodes into the aten:: node, in contrast,the _expand_module_node
        will merge all adjacent nodes into a same nodepy group.

        Parameters
        ----------
        node : trace graph node
            The non-prim node to expand
        node_name : str
            specify the node_name for NodePyGroup
        unique_name : str
            unique_name for the NodePyGroup
        op_type : str
            specify the op_type for the NodePyGroup
        nodes : list of trace graph node
            All the trace graph nodes within the same scope as the non-prim node
        input_to_node : dict
            key: input name, value: a node that uses this input
        output_to_node : dict
            key: output name, value: a node that generates this output
        module_type : str
            can be 'module' or 'func'
        Returns
        -------
        node
            the expanded non-prim node

        """
        _logger.debug("expand module node, node name: %s", node_name)
        self.global_count += 1
        if not op_type:
            op_type = node.kind()
        node_group = [node]
        inputs = list()
        outputs = list()
        node_queue = queue.Queue()
        node_queue.put(node)
        visited = {node}
        while not node_queue.empty():
            curr_node = node_queue.get()
            for _input in curr_node.inputs():
                input_name = _input.debugName()
                if input_name in output_to_node and output_to_node[input_name] in nodes:
                    predecessor_node = output_to_node[input_name]
                    if predecessor_node not in visited:
                        node_group.append(predecessor_node)
                        node_queue.put(predecessor_node)
                        visited.add(predecessor_node)
                else:
                    inputs.append(input_name)
            for _output in curr_node.outputs():
                output_name = _output.debugName()
                if output_name in input_to_node and input_to_node[output_name] in nodes:
                    successor_node = input_to_node[output_name]
                    if successor_node not in visited:
                        node_group.append(successor_node)
                        node_queue.put(successor_node)
                        visited.add(successor_node)
                else:
                    outputs.append(output_name)

        nodepy = NodePyGroup(node_name, unique_name, module_type, op_type,
                             node_group, inputs=inputs, outputs=outputs)
        return nodepy

    def _extract_cat_info(self, node_group, cpp_node):
        """
        Extract the detail information of the cat operation,
        such the order of the input tensor, the shape of each
        input tensor, the output shape, and the cat dimension.

        Parameters
        ----------
        node_group : NodePyGroup
        cpp_node: torch._C.Node
            It should be ```aten::cat``` node

        Returns
        -------
        dict
            Include auxiliary information for the cat operation.
            This dict objec has four keys: 'cat_dim', 'out_shape',
            'in_order' and 'in_shape'. cat_dim is the dimension of
            the cat operation to concat the input tensors. out_shape
            is the shape of the output tensor of the cat operation.
            in_order is an ordered list which contains the corresponding
            parent operaion nodes of the input tensors. in_shape is also
            an ordered list that contains the input shapes of the input
            tensor.
        """
        return {} # TODO: hack
        # only suport the cat operation
        assert cpp_node.kind() == CAT_KIND
        cat_info = {}
        # get the shape of the output tensor
        t_output = cpp_node.output()
        out_shape = t_output.type().sizes()
        cat_info['out_shape'] = out_shape
        # get the cat dimension
        inputs = cpp_node.inputs()
        cat_dim = list(inputs)[1].toIValue()
        cat_info['cat_dim'] = cat_dim
        # get the order of the input tensors
        # To get the order of the input tensors, we need
        # to be aware of the topology of the model, which
        # means we should extract the auxiliary information
        # after the build_index function.
        input_order = []
        list_construct_cpp = list(cpp_node.inputs())[0].node()
        input_tensors = list(list_construct_cpp.inputs())
        for _tensor in input_tensors:
            debug_name = _tensor.debugName()
            input_order.append(self.output_to_node[debug_name].unique_name)
        cat_info['in_order'] = input_order
        input_shapes = [t.type().sizes() for t in input_tensors]
        cat_info['in_shape'] = input_shapes
        return cat_info

    def _extract_shape_info(self, node):
        """
        Extract the shape information of ```aten::view``` node

        Parameters
        ----------
        node : trace graph node
            It should be ```aten::view``` node

        Returns
        -------
        dict
            Include shape of input tensor and shape of output tensor
        """
        t_input = None
        for _input in node.inputs():
            t_input = _input
            break
        t_output = node.output()
        assert isinstance(t_input.type(), torch._C.TensorType)
        assert isinstance(t_output.type(), torch._C.TensorType)
        in_shape = t_input.type().sizes()
        out_shape = t_output.type().sizes()
        return {'in_shape': in_shape, 'out_shape': out_shape}

    def _extract_leaf_modules(self):
        """
        Extract leaf modules from the given graph. Leaf module means it does not have submodules.
        To extract leaf modules because only leaf module can be replaced. And shape inference can
        be done in leaf module level. Other shape inference is done in lower level i.e.,
        operation level.

        Returns
        -------
        list
            a list of scope name of all the leaf modules
        """
        def is_parent(name1, name2):
            """
            check if name1 is parent node of name2, for example:
            name1: aa.bb,  name2: aa.bb.cc,  return True
            name1: aa.b,  name2: aa.bb, return False
            """
            parts1, parts2 = name1.split('.'), name2.split('.')
            if len(parts1) >= len(parts2):
                return False
            for i, _ in enumerate(parts1):
                if parts2[i] != parts1[i]:
                    return False
            return True
        module_names = sorted([x[0]
                               for x in self.trace.named_modules() if x[0]])
        leaf_nodes = []
        for i, name in enumerate(module_names):
            if i + 1 >= len(module_names) or not is_parent(name, module_names[i + 1]):
                leaf_nodes.append(name)
        return leaf_nodes

    def _get_module_name(self, scope_name):
        """
        Retrieve module name from scope name.
        Parameters:
        -----------
        scope_name: str
            scope_name of a graph node, for example:
            for pytorch 1.3.1: MyModel/BackboneModel[backbone]/Conv2d[conv2]
            for pytorch 1.4.0: __module.backbone/__module.backbone.conv2

        Returns:
        -------
        str
            module name, such as backbone.conv2
        """
        if torch.__version__ >= '1.4.0':
            return scope_name.split('/')[-1].replace('__module.', '')
        else:
            return '.'.join(re.findall(r'\[(.*?)\]', scope_name))

    def _build_index(self, nodes_op):
        name_to_node = dict()
        input_to_node = defaultdict(list)
        output_to_node = dict()
        for node in nodes_op:
            name_to_node[node.unique_name] = node
            for _input in node.inputs:
                input_to_node[_input].append(node)
            for output in node.outputs:
                assert not output in output_to_node, \
                    "One output cannot be generated by multiple nodes"
                output_to_node[output] = node
        return name_to_node, input_to_node, output_to_node

    def _build_graph(self):
        """
        Build graph using our defined format from jit trace.
        There are basically three steps: first, construct necessary information (data structures),
        second, extract all the modules to convert to node, Third, extract all functions to convert
        to node.

        Returns
        -------
        dict
            use name to index nodes, key: node name, value: node
        dict
            use input (its name) to index nodes,
            key: input, value: list of nodes that take this input
        dict
            use output (its name) to index nodes,
            key: output, value: node that generates this output
        """
        omit_useless_nodes = True
        graph = self.trace.graph
        _logger.debug(graph)
        print(graph)
        # build output mapping, from output debugName to its node
        output_to_node = {x.debugName(): n for n in graph.nodes()
                          for x in n.outputs()}
        # build input mapping, from input debugName to its node
        input_to_node = {x.debugName(): n for n in graph.nodes()
                         for x in n.inputs()}
        # build module mapping, from module name to all nodes (as list) under this module scope
        module_to_nodes = defaultdict(list)
        # the mapping of function (non-module in forward) to nodes, key is scope name
        func_to_nodes = defaultdict(list)

        nodes_py = GraphPy()
        for node in graph.inputs():
            if omit_useless_nodes:
                if not node.uses():  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != 'ClassType':
                nodes_py.append(NodePyIO(node, 'input'))

        self.leaf_modules = self._extract_leaf_modules()
        module_to_type = {name: parse_traced_name(
            module._name) for name, module in self.trace.named_modules()}

        # associate module name with their trace graph nodes
        for node in graph.nodes():
            module_name = self._get_module_name(node.scopeName())
            if module_name in self.leaf_modules:
                module_to_nodes[module_name].append(node)
            else:
                func_to_nodes[node.scopeName()].append(node)
        # build node group for module
        for module_name, node_cpps in module_to_nodes.items():
            use_count = 0
            merged = set()
            for node in node_cpps:
                if node not in merged:
                    # modules that have same scope name may have different locations in the
                    # graph. Futhermore, there are also lots of prim:: nodes that in node_cpps,
                    # so we also need to call the expand_module_node.
                    unique_name = module_name
                    if use_count > 0:
                        unique_name = module_name + '.%d' % use_count
                    node_group = self._expand_module_node(
                        node, module_name, unique_name, module_to_type[module_name],
                        node_cpps, input_to_node, output_to_node, 'module')
                    nodes_py.nodes_op.append(node_group)
                    use_count += 1
                    #if module_name == 'layers.0.op.bn.mask_opt':
                    #    print('node group node cpps: ', node_group.node_cpps)
                    merged.update(node_group.node_cpps)

        # each scope_name may have multiple funcs, we split them and create node for each of them
        # build node group for torch.nn.functional
        for _, nodes in func_to_nodes.items():
            # extract non prim:: nodes
            non_prim_nodes = list()
            for node in nodes:
                if not node.kind().startswith('prim::'):
                    non_prim_nodes.append(node)
            # for each non prim node, expand it
            for node in non_prim_nodes:
                node_group = self._expand_non_prim_node(
                    node, nodes, input_to_node, output_to_node, 'func')
                nodes_py.nodes_op.append(node_group)
                # get shape infor for view (aten::view) func
                # if node_group.op_type in ['aten::view', 'aten::flatten']:
                #     node_group.auxiliary = self._extract_shape_info(node)

        for node in graph.outputs():  # Create sink nodes for output ops
            node_py = NodePyIO(node, 'output')
            nodes_py.append(node_py)

        self.nodes_py = nodes_py
        # build index
        return self._build_index(self.nodes_py.nodes_op)

    def _extract_auxiliary_info(self):
        """
        Extract the auxiliary information for the nodegroups
        if necessary. For example, view/flatten operations may
        need the shape of the input tensor and output tensor.
        """
        # extract the input & output shape for the view and flatten
        for node_group in self.nodes_py.nodes_op:
            if node_group.op_type in ['aten::view', 'aten::flatten', 'aten::mean', 'aten::reshape']:
                # get shape infor for view (aten::view) func
                cpp_node = list(filter(lambda x: x.kind() == node_group.op_type,
                                       node_group.node_cpps))[0]
                if node_group.auxiliary is None:
                    node_group.auxiliary = self._extract_shape_info(cpp_node)
                else:
                    node_group.auxiliary.update(self._extract_shape_info(cpp_node))
            elif node_group.op_type == CAT_KIND:
                # get the detail information for cat func
                cpp_node = list(filter(lambda x: x.kind() == node_group.op_type,
                                       node_group.node_cpps))[0]
                if node_group.auxiliary is None:
                    node_group.auxiliary = self._extract_cat_info(node_group, cpp_node)
                else:
                    node_group.auxiliary.update(self._extract_cat_info(node_group, cpp_node))

    def find_predecessors(self, unique_name):
        """
        Find predecessor node of the given node

        Parameters
        ----------
        unique_name : str
            The unique name of the node

        Returns
        -------
        list
            a list of nodes who are the given node's predecessor
        """
        predecessors = []
        for _input in self.name_to_node[unique_name].inputs:
            if not _input in self.output_to_node:
                _logger.debug("cannot find node with %s as its output", _input)
            else:
                node_py = self.output_to_node[_input]
                predecessors.append(node_py.unique_name)
        return predecessors

    def find_successors(self, unique_name):
        """
        Find successor nodes of the given node

        Parameters
        ----------
        unique_name : str
            The unique name of the node

        Returns
        -------
        list
            a list of nodes who are the given node's successor
        """
        successors = []
        for output in self.name_to_node[unique_name].outputs:
            if output not in self.input_to_node:
                # may reach the output of the whole graph
                continue
            nodes_py = self.input_to_node[output]
            for node_py in nodes_py:
                successors.append(node_py.unique_name)
        return successors

    def get_input_nodes(self):
        """
        Returns
        -------
        dict
            key is input name, value is input node
        """
        omit_useless_nodes = True
        graph = self.trace.graph
        #nodes_py = GraphPy()
        inputs = {}
        for node in graph.inputs():
            if omit_useless_nodes:
                if not node.uses():  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != 'ClassType':
                #nodes_py.append()
                input_node = NodePyIO(node, 'input')
                input_name = node.debugName()
                inputs[input_name] = input_node
        return inputs

    def get_output_nodes(self):
        """
        Returns
        -------
        dict
            key is output name, value is output node
        """
        graph = self.trace.graph
        outputs = {}
        for node in graph.outputs():  # Create sink nodes for output ops
            output_node = NodePyIO(node, 'output')
            output_name = node.debugName()
            outputs[output_name] = output_node
        return outputs