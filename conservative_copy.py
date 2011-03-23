from copy import copy
import theano as th
import theano.tensor as T

def isshared(node):
    "Is there a better way to do this?"
    return hasattr(node, 'update')
    
def unpack_nodes(expr):
    """
    Destructures the graph down to constants or random variables and returns 
    all as a flat set.
    """
    nodes = set([expr])
    for i in expr.owner.inputs:
        nodes.add(i)
        if i.owner:
            nodes |= unpack_nodes(i)
    return nodes

def isroot(node, nodes):
    "Does the node not have any parents amongst the other nodes?"
    if not node.owner:
        return True
    if len(set(node.owner.inputs) & nodes) == 0:
        return True
    return False

def maybe_replace_all(nodes, d):
    """
    Returns nodes again, but nodes that are in the keys of d are replaced 
    with the corresponding values.
    """
    return [d.get(val, val) for val in nodes]
    
def inputs_replaced(node, replaced_nodes):
    return not isroot(node, replaced_nodes)

def clone_with_inputs(expr, inputs):
    inps, outs, other_stuff = rebuild_collect_shared( expr, inputs)
    return outs

def clone_with_inputs(node, inputs):
    return node.owner.clone_with_new_inputs(inputs).out

def conservative_clone(expr, replace, reuse_shared=False):
    """
    Clones the expression graph with the requested replacements, making as few
    total replacements as possible.
    
    If the reuse_shared argument is set, all shared nodes are reused in the new
    expression even if some of their ancestors have been replaced.
    """
    
    cur_nodes = unpack_nodes(expr)
    cur_nodes.remove(expr)
    replace = copy(replace)
    remaining_nodes = copy(cur_nodes)
    new_nodes = set()
    replaced_nodes = set(replace.keys())
    
    while len(remaining_nodes)>0:
        for c in cur_nodes:
            if reuse_shared and isshared(c):
                remaining_nodes.discard(c)
                new_nodes.add(c)
            elif isroot(c, remaining_nodes):
                remaining_nodes.discard(c)                
                c_ = replace.get(c,c)
                if inputs_replaced(c_, replaced_nodes):
                    new_node = clone_with_inputs(c_, maybe_replace_all(c.owner.inputs, replace))
                    new_nodes.add(new_node)
                    replace[c] = new_node
                    replaced_nodes.add(c)
                else:
                    new_nodes.add(c_)
          
    if inputs_replaced(expr, replaced_nodes):        
        return clone_with_inputs(expr, maybe_replace_all(expr.owner.inputs, replace))
    else:
        return expr