

#ifndef PYLIBAPR_BINDAGGREGATES_HPP
#define PYLIBAPR_BINDAGGREGATES_HPP

#include <pybind11/pybind11.h>

#include "data_structures/APR/APR.hpp"

#include <math.h>
#include <stdlib.h>
#include <algorithm>


namespace py = pybind11;
using namespace py::literals;

bool sortcol(const std::vector<double>& v1, const std::vector<double>& v2) {
    return v1[0] > v2[0];
}

void _get_neighbours(std::vector<std::vector<int>>& edges, std::vector<int>& allVisitedNodes, std::vector<int>& visited, int currNode, int currDepth, int maxDepth) {
    visited.push_back(currNode);
    allVisitedNodes[currNode] = 1;
    if (currDepth == maxDepth) {
        return;
    }
    auto it = std::lower_bound(edges[0].cbegin(), edges[0].cend(), currNode);
    int index;
    for (; *it == currNode; it++) {
        index = std::distance(edges[0].cbegin(), it);
        if (!allVisitedNodes[edges[1][index]]) {
            _get_neighbours(edges, allVisitedNodes, visited, edges[1][index], currDepth+1, maxDepth);
        }
    }
    return;
} 

std::vector<int> get_neighbours(std::vector<std::vector<int>>& edges, std::vector<int>& allVisitedNodes, int startNode, int maxDepth) {
    std::vector<int> visited = {};
    _get_neighbours(edges, allVisitedNodes, visited, startNode, 0, maxDepth);
    return visited;
}

int place_node(std::vector<std::vector<double>>& nodes, std::vector<int>& allNodesVisited) {
    // Take the highest intensity node that is not in occupied
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!allNodesVisited[nodes[i][1]]) {
            return nodes[i][1];
        }
    }
    return -1;
}

std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<int>>> place_aggregates(py::array_t<double, py::array::c_style>& node_array, py::array_t<int, py::array::c_style>& edges_array) {

    auto nodes = node_array.mutable_unchecked<2>();
    auto edges = edges_array.mutable_unchecked<2>();

    APRTimer timer(true);
    timer.start_timer("make aggregate nodes");

    // Create array of nodes only including intensity and index
    std::vector<std::vector<double>> nodesV(nodes.shape(0), std::vector<double>(2));
    for(size_t i = 0; i < (size_t) nodes.shape(0); i++){
        nodesV[i][0] = nodes(i, 4);
        nodesV[i][1] = i;
    }
    sort(nodesV.begin(), nodesV.end(), sortcol);

    // Deep copy edges to get regular 2d-array
    std::vector<std::vector<int>> edgesV(edges.shape(1), std::vector<int>(edges.shape(0)));
    for(size_t i = 0; i < (size_t) edges.shape(0); i++){
        edgesV[0][i] = edges(i, 0);
        edgesV[1][i] = edges(i, 1);
    }

    const int nNodes = nodes.shape(0);
    const int nAggregateNodes = 0.004*nNodes;

    int id;
    std::vector<int> allNodesVisited(nNodes, 0);
    std::vector<int> neighbours;
    std::vector<std::vector<int>> aggregateEdges;
    std::vector<std::vector<double>> aggregateNodes(nAggregateNodes, std::vector<double>(5));
    for(int i = 0; i < nAggregateNodes; i++) {
        // Place random aggregate node and find neighbours
        id = place_node(nodesV, allNodesVisited); //Works for now but might want to rewrite to only include last layer
        if (id == -1)
            break;

        neighbours = get_neighbours(edgesV, allNodesVisited, id, nodes(id, 1)*2);

        // Avrage all neighbours to get value for aggregateNode
        aggregateNodes[i][0] = nodes(id, 0);
        aggregateNodes[i][1] = nodes(id, 1);
        aggregateNodes[i][2] = nodes(id, 2);
        aggregateNodes[i][3] = nodes(id, 3);

        for (size_t j = 0; j < neighbours.size(); j++){
            aggregateEdges.push_back({i, neighbours[j]});
            //for (int k = 0; k < nodes.shape(1); k++) {
            aggregateNodes[i][4] += nodes(neighbours[j], 4);
            //}
        }
        aggregateNodes[i][4] /= neighbours.size();
        //for (int k = 2; k < nodes.shape(1); k++) {
        //    aggregateNodes[i][k] = std::lround(aggregateNodes[i][k]/neighbours.size());
        //}

    }  

    timer.stop_timer();
    
    return std::make_tuple( aggregateNodes, aggregateEdges );
}


//template<typename inputType>
void bindPlaceAggregates(py::module& m) {
    m.def("place_aggregates", &place_aggregates, "Place aggregate nodes into the apr-graph",
          "nodes"_a, "edges"_a);
}



void AddPlaceAggregates(py::module &m) {

    bindPlaceAggregates(m);
}

#endif //PYLIBAPR_BINDGRAPHCUT_HPP