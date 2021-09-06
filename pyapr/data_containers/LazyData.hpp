//
// Created by joel on 03.09.21.
//

#ifndef PYLIBAPR_LAZYDATA_HPP
#define PYLIBAPR_LAZYDATA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <data_structures/APR/particles/LazyData.hpp>

namespace py = pybind11;

template<typename DataType>
void AddLazyData(pybind11::module &m, const std::string &aTypeString) {

    using namespace pybind11::literals;
    using LazyDataType = LazyData<DataType>;
    std::string typeStr = "LazyData" + aTypeString;

    py::class_<LazyDataType>(m, typeStr.c_str())
            .def(py::init())
            .def("init_file", &LazyDataType::init_file, "initialize dataset from an open APRFile",
                 "aprFile"_a, "name"_a, "apr_or_tree"_a)
            .def("open", &LazyDataType::open, "open dataset")
            .def("close", &LazyDataType::close, "close dataset")
            .def("dataset_size", &LazyDataType::dataset_size, "return size of open dataset");
}

#endif //PYLIBAPR_LAZYDATA_HPP
