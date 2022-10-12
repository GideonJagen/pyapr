from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
from _pyaprwrapper.graph import place_aggregates as _place_aggregates
from ..filter import gradient_magnitude
from .._common import _check_input
from typing import Union, Optional, Tuple
import numpy as np

__allowed_input_types__ = (ShortParticles, FloatParticles)


def place_aggregates(nodes: np.ndarray,
             edges: np.ndarray):
    """
    Construct a graph by linking each particle to its face-side neighbours in each dimension.
    The graph is represented by two numpy arrays. One containing all nodes and nodefeatures and the other containing edges between the nodes.
    Node features in the graph consists of particle intensity, particle layer, particle x-coord, particle y-coord and particle z-coord.

    Parameters
    ----------
    nodes: ndarray
        Input nodes of the apr-graph
    edges: ndarray
        Input edges of the apr-graph
    output: tuple(ndarray)
        Graph represented as two numpy arrays, one for nodes and one for the edges.
        Containes the inputed graph as well as the added aggregate nodes and edges


    Returns
    -------
    output: tuple of dataframes
        Graph represented as two numpy arrays, one for nodes and one for the edges.
        Containes the inputed graph as well as the added aggregate nodes and edges
    """
    # _check_input(apr, parts, __allowed_input_types__)
    
    edges = edges[edges[:, 0].argsort()] # Tmp fix while im figuring out pybind datastructures
    aggregate_nodes, aggregate_edges = _place_aggregates(nodes, edges)

    return (aggregate_nodes, aggregate_edges)