from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
from _pyaprwrapper.graph import construct_graph as _construct_graph
from ..filter import gradient_magnitude
from .._common import _check_input
from typing import Union, Optional, Tuple
import numpy as np

__allowed_input_types__ = (ShortParticles, FloatParticles)


def construct_graph(apr: APR,
             parts: Union[ShortParticles, FloatParticles],
             avg_num_neighbors: float = 3.3) -> tuple:
    """
    Construct a graph by linking each particle to its face-side neighbours in each dimension.
    The graph is represented by two numpy arrays. One containing all nodes and nodefeatures and the other containing edges between the nodes.
    Node features in the graph consists of particle intensity, particle layer, particle x-coord, particle y-coord and particle z-coord.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values.
    avg_num_neighbors: float
        Controls the amount of memory initially allocated for edges. If memory is being reallocated,
        consider increasing this value. (default: 3.3)
    output: ByteParticles or ShortParticles, optional
        Particle object to which the resulting mask is written. If None, a new ByteParticles object
        is generated. (default: None)

    Returns
    -------
    output: tuple of dataframes
        Graph represented as two numpy arrays, one for nodes and one for the edges
    """
    _check_input(apr, parts, __allowed_input_types__)
    
    output_nodes = np.zeros((apr.total_number_particles(), 5))
    output_edges = np.zeros((2*int(apr.total_number_particles()*avg_num_neighbors), 2)).astype('int32')
    _construct_graph(apr, parts, output_nodes, output_edges)

    return (output_nodes, output_edges[~np.all(output_edges == 0, axis=1)])