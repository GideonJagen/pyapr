

#ifndef PYLIBAPR_BINDGRAPH_HPP
#define PYLIBAPR_BINDGRAPH_HPP

#include <pybind11/pybind11.h>

#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/ImagePatch.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRNumerics.hpp"
#include "algorithm/LocalIntensityScale.hpp"

#include "data_containers/src/BindParticleData.hpp"
#include "maxflow-v3.04.src/graph.h"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

namespace py = pybind11;
using namespace py::literals;


template<typename inputType, typename S>
void graph_apr(APR& apr, PyParticleData<inputType>& input_parts, py::array_t<double, py::array::c_style>& node_array, py::array_t<int, py::array::c_style>& edges_array) {

    auto resNodes = node_array.mutable_unchecked<2>();
    auto resEdges = edges_array.mutable_unchecked<2>();

    //APRTimer timer(true);

    // Loop over particles and edd edges
    //timer.start_timer("populate graph");
    auto it = apr.random_iterator();
    auto neighbour_iterator = apr.random_iterator();

    uint64_t edge_counter = 0;

    for(int level = it.level_min(); level <= it.level_max(); ++level) {

        //const float base_dist = it.level_size(level);

        for(int z = 0; z < apr.z_num(level); ++z) {
            for(int x = 0; x < apr.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    const size_t ct_id = it;
                    const float val = input_parts[ct_id];

                    resNodes(ct_id, 0) = x;
                    resNodes(ct_id, 1) = it.y();
                    resNodes(ct_id, 2) = z;
                    resNodes(ct_id, 3) = level; 
                    resNodes(ct_id, 4) = val;

                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    // Edges are bidirectional, so we only need the positive directions
                    for (int direction = 0; direction < 6; direction+=2) {
                        it.find_neighbours_in_direction(direction);

                        // For each face, there can be 0-4 neighbours
                        for (int index = 0; index < it.number_neighbours_in_direction(direction); ++index) {
                            if (neighbour_iterator.set_neighbour_iterator(it, direction, index)) {
                                //will return true if there is a neighbour defined
                                const size_t neigh_id = neighbour_iterator;

                                // in some cases the neighbour iterator and iterator are simply swapped
                                if(neigh_id == ct_id) {
                                    continue;
                                }

                                // Add the edge twice since it is bidirectional
                                resEdges(edge_counter, 0) = ct_id;
                                resEdges(edge_counter, 1) = neigh_id;

                                resEdges(edge_counter+1, 0) = neigh_id;
                                resEdges(edge_counter+1, 1) = ct_id;
                                edge_counter+=2;
                            }
                        }
                    }
                }
            }
        }
    }
    //timer.stop_timer();

    //std::cout << "number of edges = " << edge_counter << " (" << (float)edge_counter/(float)apr.total_number_particles() << " x nparts) | Number of nodes: " << node_array.size() << std::endl;

    //node_array = nodes;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t idx = 0; idx < 2; ++idx) {
        continue;
    }
    //timer.stop_timer();
}


template<typename inputType>
void bindConvertGraph(py::module& m) {
    m.def("construct_graph", &graph_apr<inputType, uint8_t>, "Construct graph from the particle representation",
          "apr"_a, "input_parts"_a, "node_array"_a, "edge_array"_a);

    m.def("construct_graph", &graph_apr<inputType, uint16_t>, "Construct graph from the particle representation",
          "apr"_a, "input_parts"_a, "node_array"_a, "edge_array"_a);
}



void AddConstructGraph(py::module &m) {

    bindConvertGraph<uint16_t>(m);
    bindConvertGraph<float>(m);
}

#endif //PYLIBAPR_BINDGRAPHCUT_HPP
