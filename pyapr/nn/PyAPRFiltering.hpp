//
// Created by Joel Jonsson on 09.05.19.
//

#ifndef PYLIBAPR_PYAPRFILTERING_HPP
#define PYLIBAPR_PYAPRFILTERING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/iterators/APRTreeIterator.hpp"
#include "data_structures/APR/iterators/APRIterator.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"

#include "data_containers/PyAPR.hpp"
#include "data_containers/PyPixelData.hpp"


namespace py = pybind11;


namespace PyAPRFiltering {

    template<typename T>
    void update_dense_array(const uint64_t level, const uint64_t z, APR &apr, ParticleData<T> &tree_data, PixelData<T> &temp_vec,
                            T *part_int, const std::vector<int> &stencil_shape, const std::vector<int> &stencil_half,
                            uint64_t in_offset);


    template<typename T>
    void update_dense_array_backward(const uint64_t level, const uint64_t z, APR &apr, ParticleData<T> &grad_tree_data,
                                     PixelData<T> &temp_vec_di, T *grad_input_ptr,
                                     const std::vector<int> &stencil_shape, const std::vector<int> &stencil_half,
                                     const uint64_t in_offset);

    template<typename T>
    void convolve_generic(APR &apr, T *input_intensities,
                          const std::vector<PixelData<T>> &stencil_vec, T bias, int out_channel,
                          int in_channel, int current_max_level, const uint64_t in_offset,
                          ParticleData<T> &tree_data, const size_t number_in_channels,
                          const uint64_t out_offset, T *output_ptr);


    template<typename T>
    void convolve_generic_backward(APR &apr, T *input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                   const std::vector<PixelData<T>> &stencil_vec,
                                   T *grad_output_ptr, T *grad_input_ptr, std::vector<float> &temp_vec_db,
                                   std::vector<T> &temp_vec_dw, const uint64_t grad_weight_offset,
                                   const size_t db_offset, ParticleData<T> &grad_tree_temp, ParticleData<T> &tree_data,
                                   const int out_channel, const int in_channel, const unsigned int current_max_level);


    template<typename T>
    void convolve_3x3(APR &apr, T *input_intensities,
                      const std::vector<PixelData<T>> &stencil_vec, T bias, int out_channel,
                      int in_channel, int current_max_level, const uint64_t in_offset,
                      ParticleData<T> &tree_data, const size_t number_in_channels,
                      const uint64_t out_offset, T *output_ptr);


    template<typename T>
    void convolve_3x3_backward(APR &apr, T *input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                               const std::vector<PixelData<T>> &stencil_vec,
                               T *grad_output_ptr, T *grad_input_ptr, std::vector<float> &temp_vec_db,
                               std::vector<T> &temp_vec_dw, const uint64_t grad_weight_offset,
                               const size_t db_offset,
                               ParticleData<T> &grad_tree_temp, ParticleData<T> &tree_data,
                               const int out_channel, const int in_channel, const unsigned int current_max_level);

    template<typename T>
    void convolve_1x1(APR &apr, T* input_ptr, const std::vector<float> &stencil_vec, float bias, int out_channel,
                      int in_channel, int current_max_level, const uint64_t in_offset, const size_t number_in_channels,
                      const uint64_t out_offset, T* output_ptr);


    template<typename T>
    void convolve_1x1_backward(APR &apr, T* input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                               const std::vector<float> &stencil_vec, T* grad_output_ptr, T* grad_input_ptr,
                               std::vector<float> &temp_vec_db, std::vector<float> &temp_vec_dw,
                               const uint64_t grad_weight_offset, const size_t db_offset, const int out_channel,
                               const int in_channel, const unsigned int current_max_level);


    template<typename T>
    void max_pool_2x2(APR &apr, T *input_ptr, T *output_ptr, uint64_t *idx_ptr,
                      const uint64_t in_offset, const uint64_t out_offset, const int64_t tree_offset_in,
                      const int64_t tree_offset_out, const unsigned int current_max_level);


    template<typename T>
    void max_pool_backward_stored_indices(T* grad_input_ptr, T* grad_output_ptr, uint64_t* index_ptr,
                                          uint64_t in_size, uint64_t out_size);


    template<typename T>
    void fill_tree_mean(APR &apr, T *input_ptr, ParticleData<T> &tree_data, uint64_t in_offset, unsigned int current_max_level);


    template<typename T>
    void fill_tree_mean_backward(APR &apr, T *grad_input_ptr, ParticleData<T> &grad_tree_temp, uint64_t in_offset,
                                 unsigned int current_max_level);

    template<typename APRIteratorType, typename APRTreeIteratorType>
    int64_t compute_tree_offset(APRIteratorType &apr_iterator, APRTreeIteratorType &tree_iterator, unsigned int level);

    template<typename APRIteratorType, typename APRTreeIteratorType>
    uint64_t number_parts_at_level(APRIteratorType &apr_iterator, APRTreeIteratorType &tree_iterator, unsigned int max_level);


    template<typename T>
    void transposed_conv_2x2(APR &apr, T *input_ptr, T *output_ptr, PixelData<T>& stencil_2x2, T weight_1x1, T bias,
                             uint64_t in_offset, uint64_t out_offset, int64_t tree_offset_in,
                             int64_t tree_offset_out, int current_max_level);

    template<typename T>
    void transposed_conv_2x2_backward(APR &apr, T *input_ptr, PixelData<T>& stencil_2x2, T weight_1x1,
                                      T* grad_input_ptr, std::vector<float>& temp_dw_2x2, std::vector<float>& temp_dw_1x1,
                                      std::vector<float>& temp_db, T* grad_output_ptr, size_t dw_2x2_offset, size_t dw_1x1_offset,
                                      size_t db_offset, uint64_t in_offset, uint64_t out_offset,
                                      int64_t tree_offset_in, int64_t tree_offset_out, int current_max_level);
}

//*****************************************************************************************************************
//                              Forward convolution arbitrary filter size
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::convolve_generic(APR &apr, T *input_intensities,
                                      const std::vector<PixelData<T>> &stencil_vec, T bias, int out_channel,
                                      int in_channel, int current_max_level, const uint64_t in_offset,
                                      ParticleData<T> &tree_data, const size_t number_in_channels,
                                      const uint64_t out_offset, T *output_ptr) {

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();

    APRTimer timer(false);

    int stencil_counter = 0;

    const T b = in_channel == (number_in_channels - 1) ? bias : 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                (int) stencil_vec[stencil_counter].x_num,
                                                (int) stencil_vec[stencil_counter].z_num};

        const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                               (stencil_shape[1] - 1) / 2,
                                               (stencil_shape[2] - 1) / 2};

        unsigned int z = 0;
        unsigned int x = 0;

        const unsigned int y_num_m = (apr.org_dims(0) > 1) ? apr_iterator.y_num(level) + stencil_shape[0] - 1 : 1;
        const unsigned int x_num_m = (apr.org_dims(1) > 1) ? apr_iterator.x_num(level) + stencil_shape[1] - 1 : 1;

        PixelData<T> temp_vec;
        temp_vec.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

        //initial condition
        update_dense_array(level, 0, apr, tree_data, temp_vec,
                           input_intensities, stencil_shape, stencil_half, in_offset);

        /// Compute convolution output at apr particles
        timer.start_timer("convolve apr particles");

        for (x = 0; x < apr.x_num(level); ++x) {
            for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                T neigh_sum = 0;
                int counter = 0;

                for (size_t ix = 0; ix < stencil_shape[1]; ++ix) {
                    for (size_t iy = 0; iy < stencil_shape[0]; ++iy) {
                        neigh_sum += temp_vec.at(iy + apr_iterator.y(), ix + x, 0) *
                                     stencil_vec[stencil_counter].mesh[counter++];
                    }
                }

                output_ptr[out_offset + apr_iterator] += neigh_sum + b;

            }//y, pixels/columns (apr)
        }//x , rows (apr)

        timer.stop_timer();

        /// if there are downsampled values, we need to use the tree iterator for those outputs
        if (level == current_max_level && current_max_level < apr.level_max()) {

            timer.start_timer("convolve tree particles");

            const int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

            for (x = 0; x < tree_iterator.x_num(level); ++x) {
                for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                    T neigh_sum = 0;
                    int counter = 0;

                    for (size_t ix = 0; ix < stencil_shape[1]; ++ix) {
                        for (size_t iy = 0; iy < stencil_shape[0]; ++iy) {
                            neigh_sum += temp_vec.at(iy + tree_iterator.y(), ix + x, 0) *
                                         stencil_vec[stencil_counter].mesh[counter++];
                        }
                    }

                    output_ptr[out_offset + tree_iterator + tree_offset] += neigh_sum + b;

                }//y, pixels/columns (tree)
            }//x, rows (tree)
            timer.stop_timer();
        } //if

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
    }//levels
}


//*****************************************************************************************************************
//                              Forward convolution 3x3 filter
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::convolve_3x3(APR &apr, T *input_intensities,
                                  const std::vector<PixelData<T>> &stencil_vec, T bias, int out_channel,
                                  int in_channel, int current_max_level, const uint64_t in_offset,
                                  ParticleData<T> &tree_data, const size_t number_in_channels,
                                  const uint64_t out_offset, T *output_ptr){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();

    int stencil_counter = 0;

    const T b = in_channel==(number_in_channels-1) ? bias : 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        const std::vector<int> stencil_shape = {3, 3, 1};
        const std::vector<int> stencil_half = {1, 1, 0};

        unsigned int z = 0;
        unsigned int x = 0;

        const int y_num_m = (apr.org_dims(0) > 1) ? apr_iterator.y_num(level) + stencil_shape[0] - 1 : 1;
        const int x_num_m = (apr.org_dims(1) > 1) ? apr_iterator.x_num(level) + stencil_shape[1] - 1 : 1;

        // TODO: allocate this memory outside and simply change dimension sizes and pad boundary in each iteration
        PixelData<T> temp_vec;
        temp_vec.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

        // fill temporary image with values (isotropic neighborhoods)
        update_dense_array(level, 0, apr, tree_data, temp_vec,
                               input_intensities, stencil_shape, stencil_half, in_offset);

        /// Compute convolution output at apr particles

        for (x = 0; x < apr.x_num(level); ++x) {
            for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                T neigh_sum = temp_vec.at(apr_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                              temp_vec.at(apr_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                              temp_vec.at(apr_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                              temp_vec.at(apr_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                              temp_vec.at(apr_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                              temp_vec.at(apr_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                              temp_vec.at(apr_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                              temp_vec.at(apr_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                              temp_vec.at(apr_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                output_ptr[out_offset + apr_iterator] += neigh_sum + b;

            }//y, pixels/columns (apr)
        }//x , rows (apr)

        /// if there are downsampled values, we need to use the tree iterator for those outputs
        if(level == current_max_level && current_max_level < apr.level_max()) {

            const int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

            for (x = 0; x < tree_iterator.x_num(level); ++x) {
                for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                    T neigh_sum = temp_vec.at(tree_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                  temp_vec.at(tree_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                  temp_vec.at(tree_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                  temp_vec.at(tree_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                  temp_vec.at(tree_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                  temp_vec.at(tree_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                  temp_vec.at(tree_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                  temp_vec.at(tree_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                  temp_vec.at(tree_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                    output_ptr[out_offset + tree_iterator + tree_offset] += neigh_sum + b;

                }//y, pixels/columns (tree)
            }//x, rows (tree)
        } //if

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
    }//levels
}


//*****************************************************************************************************************
//                              Forward convolution 1x1 filter
//*****************************************************************************************************************

template<typename T>
void PyAPRFiltering::convolve_1x1(APR &apr, T* input_ptr, const std::vector<float> &stencil_vec, float bias,
                                  int out_channel, int in_channel, int current_max_level, const uint64_t in_offset,
                                  const size_t number_in_channels, const uint64_t out_offset, T* output_ptr){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();

    APRTimer timer(false);

    const T b = in_channel==(number_in_channels-1) ? bias : 0.0f;

    unsigned int z;
    unsigned int x;

    int stencil_counter = 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        T w = (T) stencil_vec[stencil_counter];

        for (z = 0; z < apr.z_num(level); ++z) {

            /// Compute convolution output at apr particles
            timer.start_timer("convolve 1x1 apr particles");

            for (x = 0; x < apr.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    output_ptr[out_offset + apr_iterator] +=
                            w * input_ptr[in_offset + apr_iterator] + b;

                }//y, pixels/columns (apr)
            }//x , rows (apr)

            timer.stop_timer();

            /// if there are downsampled values, we need to use the tree iterator for those outputs
            if(level == current_max_level && current_max_level < apr.level_max()) {

                timer.start_timer("convolve 1x1 tree particles");

                const int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

                for (x = 0; x < tree_iterator.x_num(level); ++x) {
                    for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                        const uint64_t idx = tree_iterator + tree_offset;

                        output_ptr[out_offset + idx] += w * input_ptr[in_offset + idx] + b;

                    }//y, pixels/columns (tree)
                }//x, rows (tree)
                timer.stop_timer();
            } //if

        }//z

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
    }//levels

}//convolve1x1_batchparallel


//*****************************************************************************************************************
//                              Temporary isotropic patch reconstruction
//*****************************************************************************************************************

template<typename T>
void PyAPRFiltering::update_dense_array(const uint64_t level, const uint64_t z, APR &apr, ParticleData<T> &tree_data,
                                        PixelData<T> &temp_vec, T *part_int, const std::vector<int> &stencil_shape,
                                        const std::vector<int> &stencil_half, uint64_t in_offset){

    auto apr_iterator = apr.iterator();
    auto treeIterator = apr.tree_iterator();
    uint64_t x;

    const uint64_t x_num_m = temp_vec.x_num;
    const uint64_t y_num_m = temp_vec.y_num;

    for (x = 0; x < apr_iterator.x_num(level); ++x) {

        //
        //  This loop recreates particles at the current level, using a simple copy
        //

        uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);

        for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

            temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator + in_offset];

        }
    }

    if (level > apr_iterator.level_min()) {
        const int y_num = apr_iterator.y_num(level);

        //
        //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
        //

        for (x = 0; x < apr.x_num(level); ++x) {

            for (apr_iterator.begin(level-1, z/2, x/2); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                            z % stencil_shape[2]) = part_int[apr_iterator + in_offset];
                temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                            z % stencil_shape[2]) = part_int[apr_iterator + in_offset];

            }
        }
    }

    /******** start of using the tree iterator for downsampling ************/

    if (level < apr_iterator.level_max()) {

        for (x = 0; x < apr.x_num(level); ++x) {
            for (treeIterator.begin(level, z, x); treeIterator < treeIterator.end(); ++treeIterator) {

                temp_vec.at(treeIterator.y() + stencil_half[0], x + stencil_half[1],
                            z % stencil_shape[2]) = tree_data[treeIterator];

            }
        }
    }
}


//*****************************************************************************************************************
//                              Backward convolution arbitrary filter size
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::convolve_generic_backward(APR &apr, T *input_ptr, const uint64_t in_offset,
                                               const uint64_t out_offset,
                                               const std::vector<PixelData<T>> &stencil_vec,
                                               T *grad_output_ptr, T *grad_input_ptr,
                                               std::vector<float> &temp_vec_db, std::vector<T> &temp_vec_dw,
                                               const uint64_t grad_weight_offset, const size_t db_offset,
                                               ParticleData<T> &grad_tree_temp, ParticleData<T> &tree_data,
                                               const int out_channel, const int in_channel,
                                               const unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();
    int stencil_counter = 0;
    T d_bias = 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        //PixelData<float> stencil(stencil_vec[stencil_counter], true);

        const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                (int) stencil_vec[stencil_counter].x_num,
                                                (int) stencil_vec[stencil_counter].z_num};

        const std::vector<int> stencil_half = {(stencil_shape[0]-1)/2,
                                               (stencil_shape[1]-1)/2,
                                               (stencil_shape[2]-1)/2};

        const uint64_t dw_offset = grad_weight_offset + stencil_counter * stencil_shape[0] * stencil_shape[1];

        // assert stencil_shape compatible with apr org_dims?

        unsigned int z = 0;
        unsigned int x = 0;

        const int z_num = 1; //apr_iterator.z_num(level);

        const int y_num_m = (apr.org_dims(0) > 1) ? apr_iterator.y_num(level) +
                                                               stencil_shape[0] - 1 : 1;
        const int x_num_m = (apr.org_dims(1) > 1) ? apr_iterator.x_num(level) +
                                                               stencil_shape[1] - 1 : 1;


        PixelData<T> temp_vec;
        temp_vec.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

        PixelData<T> temp_vec_di;
        temp_vec_di.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0);

        /*
        for (int padd = 0; padd < stencil_half[2]; ++padd) {
            update_dense_array2(level,
                                padd,
                                apr,
                                apr_iterator,
                                tree_iterator,
                                tree_data,
                                temp_vec,
                                input_intensities,
                                stencil_shape,
                                stencil_half,
                                in_offset);
        }
        */

        for (z = 0; z < apr.z_num(level); ++z) {

            if (z < (z_num - stencil_half[2])) {
                //update the next z plane for the access
                update_dense_array(level, z + stencil_half[2], apr, tree_data,
                                   temp_vec, input_ptr, stencil_shape, stencil_half, in_offset);
            } else {
                //padding
                uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);

                for (x = 0; x < temp_vec.x_num; ++x) {
                    std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                              temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                }
            }

            for(x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    const T dO = grad_output_ptr[out_offset + apr_iterator];
                    d_bias += dO;

                    for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                        for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                            temp_vec_di.at(apr_iterator.y() + iy, x + ix, 0) += dO * stencil_vec[stencil_counter].mesh[ix*stencil_shape[0] + iy];
                        }
                    }

                    for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                        for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                            temp_vec_dw[dw_offset + ix*stencil_shape[0] + iy] += dO * temp_vec.at(apr_iterator.y() + iy, x + ix, 0);
                        }
                    }

                } // y pixels/columns (apr)
            } // x

            /// if there are downsampled values, we need to use the tree iterator for those outputs
            if(level == current_max_level && current_max_level < apr.level_max()) {

                int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

                for(x = 0; x<tree_iterator.x_num(level); ++x) {
                    for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                        //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        //const int i = x + stencil_half[1];

                        T dO = grad_output_ptr[out_offset + tree_iterator + tree_offset];
                        d_bias += dO;

                        for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                            for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                temp_vec_di.at(tree_iterator.y() + iy, x + ix, 0) += dO * stencil_vec[stencil_counter].mesh[ix*stencil_shape[0] + iy];
                            }
                        }

                        for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                            for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                temp_vec_dw[dw_offset + ix*stencil_shape[0] + iy] += dO * temp_vec.at(tree_iterator.y() + iy, x + ix, 0);
                            }
                        }

                    }//y, pixels/columns (tree)
                }//x
            } //if

            //TODO: this works for 2D images, but for 3D the updating needs to change
            /// push temp_vec_di to grad_input and grad_tree_temp
            update_dense_array_backward(level, z, apr, grad_tree_temp, temp_vec_di,
                                        grad_input_ptr, stencil_shape, stencil_half, in_offset);

        }//z

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

    }//levels

    /// push d_bias to temp_vec_db
    temp_vec_db[out_channel + db_offset] += (float) d_bias;
}


//*****************************************************************************************************************
//                              Backward convolution 3x3 filter
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::convolve_3x3_backward(APR &apr, T *input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                           const std::vector<PixelData<T>> &stencil_vec, T *grad_output_ptr,
                                           T *grad_input_ptr, std::vector<float> &temp_vec_db,
                                           std::vector<T> &temp_vec_dw, const uint64_t grad_weight_offset,
                                           const size_t db_offset, ParticleData<T> &grad_tree_temp,
                                           ParticleData<T> &tree_data, const int out_channel,
                                           const int in_channel, const unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();
    int stencil_counter = 0;
    T d_bias = 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        //PixelData<float> stencil(stencil_vec[stencil_counter], true);

        const uint64_t dw_offset = grad_weight_offset + stencil_counter * 9;

        const std::vector<int> stencil_shape = {3, 3, 1};
        const std::vector<int> stencil_half = {1, 1, 0};

        // assert stencil_shape compatible with apr org_dims?

        unsigned int z = 0;
        unsigned int x = 0;

        const int z_num = 1; //apr_iterator.z_num(level);

        const int y_num_m = (apr.org_dims(0) > 1) ? apr_iterator.y_num(level) +
                                                               stencil_shape[0] - 1 : 1;
        const int x_num_m = (apr.org_dims(1) > 1) ? apr_iterator.x_num(level) +
                                                               stencil_shape[1] - 1 : 1;


        PixelData<T> temp_vec;
        temp_vec.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

        PixelData<T> temp_vec_di;
        temp_vec_di.initWithValue(y_num_m, x_num_m, stencil_shape[2], 0);

        for (z = 0; z < apr.z_num(level); ++z) {

            if (z < (z_num - stencil_half[2])) {
                //update the next z plane for the access
                update_dense_array(level, z + stencil_half[2], apr, tree_data,
                                       temp_vec, input_ptr, stencil_shape, stencil_half, in_offset);
            } else {
                //padding
                uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);

                for (x = 0; x < temp_vec.x_num; ++x) {
                    std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                              temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                }
            }

            for(x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    const T dO = grad_output_ptr[out_offset + apr_iterator];
                    d_bias += dO;

                    const size_t y_num = temp_vec.y_num;
                    const uint64_t idx0 = x * y_num + apr_iterator.y();
                    const uint64_t idx1 = idx0 + y_num;
                    const uint64_t idx2 = idx1 + y_num;

                    temp_vec_di.mesh[idx0]   += dO * stencil_vec[stencil_counter].mesh[0];
                    temp_vec_di.mesh[idx0+1] += dO * stencil_vec[stencil_counter].mesh[1];
                    temp_vec_di.mesh[idx0+2] += dO * stencil_vec[stencil_counter].mesh[2];

                    temp_vec_di.mesh[idx1]   += dO * stencil_vec[stencil_counter].mesh[3];
                    temp_vec_di.mesh[idx1+1] += dO * stencil_vec[stencil_counter].mesh[4];
                    temp_vec_di.mesh[idx1+2] += dO * stencil_vec[stencil_counter].mesh[5];

                    temp_vec_di.mesh[idx2]   += dO * stencil_vec[stencil_counter].mesh[6];
                    temp_vec_di.mesh[idx2+1] += dO * stencil_vec[stencil_counter].mesh[7];
                    temp_vec_di.mesh[idx2+2] += dO * stencil_vec[stencil_counter].mesh[8];


                    temp_vec_dw[dw_offset]   += dO * temp_vec.mesh[idx0];
                    temp_vec_dw[dw_offset+1] += dO * temp_vec.mesh[idx0+1];
                    temp_vec_dw[dw_offset+2] += dO * temp_vec.mesh[idx0+2];

                    temp_vec_dw[dw_offset+3] += dO * temp_vec.mesh[idx1];
                    temp_vec_dw[dw_offset+4] += dO * temp_vec.mesh[idx1+1];
                    temp_vec_dw[dw_offset+5] += dO * temp_vec.mesh[idx1+2];

                    temp_vec_dw[dw_offset+6] += dO * temp_vec.mesh[idx2];
                    temp_vec_dw[dw_offset+7] += dO * temp_vec.mesh[idx2+1];
                    temp_vec_dw[dw_offset+8] += dO * temp_vec.mesh[idx2+2];

                }
            }

            /// if there are downsampled values, we need to use the tree iterator for those outputs
            if(level == current_max_level && current_max_level < apr.level_max()) {

                int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

                for(x = 0; x<tree_iterator.x_num(level); ++x) {
                    for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                        T dO = grad_output_ptr[out_offset + tree_iterator + tree_offset];
                        d_bias += dO;

                        const size_t y_num = temp_vec.y_num;
                        const uint64_t idx0 = x * y_num + tree_iterator.y();
                        const uint64_t idx1 = idx0 + y_num;
                        const uint64_t idx2 = idx1 + y_num;

                        temp_vec_di.mesh[idx0]   += dO * stencil_vec[stencil_counter].mesh[0];
                        temp_vec_di.mesh[idx0+1] += dO * stencil_vec[stencil_counter].mesh[1];
                        temp_vec_di.mesh[idx0+2] += dO * stencil_vec[stencil_counter].mesh[2];

                        temp_vec_di.mesh[idx1]   += dO * stencil_vec[stencil_counter].mesh[3];
                        temp_vec_di.mesh[idx1+1] += dO * stencil_vec[stencil_counter].mesh[4];
                        temp_vec_di.mesh[idx1+2] += dO * stencil_vec[stencil_counter].mesh[5];

                        temp_vec_di.mesh[idx2]   += dO * stencil_vec[stencil_counter].mesh[6];
                        temp_vec_di.mesh[idx2+1] += dO * stencil_vec[stencil_counter].mesh[7];
                        temp_vec_di.mesh[idx2+2] += dO * stencil_vec[stencil_counter].mesh[8];


                        temp_vec_dw[dw_offset]   += dO * temp_vec.mesh[idx0];
                        temp_vec_dw[dw_offset+1] += dO * temp_vec.mesh[idx0+1];
                        temp_vec_dw[dw_offset+2] += dO * temp_vec.mesh[idx0+2];

                        temp_vec_dw[dw_offset+3] += dO * temp_vec.mesh[idx1];
                        temp_vec_dw[dw_offset+4] += dO * temp_vec.mesh[idx1+1];
                        temp_vec_dw[dw_offset+5] += dO * temp_vec.mesh[idx1+2];

                        temp_vec_dw[dw_offset+6] += dO * temp_vec.mesh[idx2];
                        temp_vec_dw[dw_offset+7] += dO * temp_vec.mesh[idx2+1];
                        temp_vec_dw[dw_offset+8] += dO * temp_vec.mesh[idx2+2];
                    }//y, pixels/columns (tree)
                }//x
            } //if

            //TODO: this works for 2D images, but for 3D the updating needs to change
            /// push temp_vec_di to grad_input and grad_tree_temp
            update_dense_array_backward(level, z, apr, grad_tree_temp, temp_vec_di,
                                        grad_input_ptr, stencil_shape, stencil_half, in_offset);

        }//z

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

    }//levels

    /// push d_bias to temp_vec_db
    temp_vec_db[out_channel + db_offset] += (float) d_bias;
}



//*****************************************************************************************************************
//                              Backward convolution 1x1 filter
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::convolve_1x1_backward(APR &apr, T* input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                           const std::vector<float> &stencil_vec, T* grad_output_ptr,
                                           T* grad_input_ptr, std::vector<float> &temp_vec_db,
                                           std::vector<float> &temp_vec_dw, const uint64_t grad_weight_offset,
                                           const size_t db_offset, const int out_channel, const int in_channel,
                                           const unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();

    float d_bias = 0;
    int stencil_counter = 0;

    for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

        float d_weight = 0;
        const float w = stencil_vec[stencil_counter];

        unsigned int z = 0;
        unsigned int x = 0;

        for (z = 0; z < apr.z_num(level); ++z) {

            for(x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    const T dO = grad_output_ptr[out_offset + apr_iterator];

                    d_bias += dO;
                    d_weight += dO * input_ptr[in_offset + apr_iterator];
                    grad_input_ptr[in_offset + apr_iterator] += dO * w;

                }//y, pixels/columns
            } //x

            /// if there are downsampled values, we need to use the tree iterator for those outputs
            if(level == current_max_level && current_max_level < apr.level_max()) {

                const int64_t tree_offset = compute_tree_offset(apr_iterator, tree_iterator, level);

                for(x = 0; x < tree_iterator.x_num(level); ++x) {
                    for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                        const uint64_t idx = tree_iterator + tree_offset;
                        const T dO = grad_output_ptr[out_offset + idx];

                        d_bias += dO;
                        d_weight += dO * input_ptr[in_offset + idx];
                        grad_input_ptr[in_offset + idx] += dO * w;

                    }//y, pixels/columns (tree)
                } //x
            } //if
        }//z

        temp_vec_dw[grad_weight_offset + stencil_counter] += d_weight;

        // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
        stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

    }//levels

    temp_vec_db[db_offset + out_channel] += d_bias;

}


//*****************************************************************************************************************
//                              Backward temporary isotropic patch reconstruction
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::update_dense_array_backward(const uint64_t level, const uint64_t z, APR &apr,
                                                 ParticleData<T> &grad_tree_data, PixelData<T> &temp_vec_di,
                                                 T *grad_input_ptr, const std::vector<int> &stencil_shape,
                                                 const std::vector<int> &stencil_half, const uint64_t in_offset){

    auto apr_iterator = apr.iterator();
    auto treeIterator = apr.tree_iterator();

    size_t x;

    const size_t x_num_m = temp_vec_di.x_num;
    const size_t y_num_m = temp_vec_di.y_num;


    /// insert values at matching apr particle locations

    for (x = 0; x < apr_iterator.x_num(level); ++x) {

        uint64_t mesh_offset = x_num_m * y_num_m * (z % stencil_shape[2]) + (x + stencil_half[1]) * y_num_m + stencil_half[0];

        for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

            grad_input_ptr[in_offset + apr_iterator] += temp_vec_di.mesh[apr_iterator.y() + mesh_offset];
        }
    }

    if (level > apr_iterator.level_min()) {
        const int y_num = apr_iterator.y_num(level);

        /// insert values at lower level particles

        for (x = 0; x < apr.x_num(level); ++x) {

            for (apr_iterator.begin(level-1, z/2, x/2); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);

                grad_input_ptr[in_offset + apr_iterator] +=
                        temp_vec_di.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]) +
                        temp_vec_di.at(y_m + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

            }
        }
    }

    /******** insert values at tree particle locations ************/

    if (level < apr_iterator.level_max()) {

        for (x = 0; x < apr.x_num(level); ++x) {
            for (treeIterator.begin(level, z, x); treeIterator < treeIterator.end(); ++treeIterator) {

                grad_tree_data[treeIterator] += temp_vec_di.at(treeIterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

            }
        }
    }
}


//*****************************************************************************************************************
//                              Fill APRTree by average-downsampling APR particles
//*****************************************************************************************************************

template<typename T>
void PyAPRFiltering::fill_tree_mean(APR &apr, T *input_ptr, ParticleData<T> &tree_data, uint64_t in_offset,
                                    unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();

    if( current_max_level < apr.level_max() ) {
        tree_data.init(tree_iterator.particles_level_end(current_max_level));
    } else {
        tree_data.init(apr.total_number_parent_cells());
    }

    auto parent_iterator = apr.tree_iterator();

    //int z_d;
    int x_d;

    //py::buffer_info input_buf = particle_data.request();
    //auto input_ptr = (float *) input_buf.ptr;

    /// if downsampling has been performed, insert the downsampled values directly
    if(current_max_level < apr.level_max()) {

        int z=0;
        int x;

        const int64_t tree_offset_in  = compute_tree_offset(apr_iterator, tree_iterator, current_max_level);

        //for (z = 0; z < treeIterator.z_num(current_max_level); z++) {
        for (x = 0; x < tree_iterator.x_num(current_max_level); ++x) {
            for (tree_iterator.begin(current_max_level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                tree_data[tree_iterator] = input_ptr[in_offset + tree_iterator + tree_offset_in];

            }
        }//x
        //}//z
    }//if

    /// now fill in parent nodes of APR particles
    for (unsigned int level = current_max_level; level > apr_iterator.level_min(); --level) {
        //z_d = 0;
        int z = 0;

        //for (z_d = 0; z_d < parentIterator.z_num(level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.z_num(level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < parent_iterator.x_num(level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.x_num(level)-1); ++x) {

                parent_iterator.begin(level - 1, z / 2, x / 2);

                //dealing with boundary conditions
                T scale_factor_xz =
                        (((2 * parent_iterator.x_num(level - 1) != apr.x_num(level)) &&
                          ((x / 2) == (parent_iterator.x_num(level - 1) - 1))) +
                         ((2 * parent_iterator.z_num(level - 1) != apr.z_num(level)) &&
                          (z / 2) == (parent_iterator.z_num(level - 1) - 1))) * 2;

                if (scale_factor_xz == 0) {
                    scale_factor_xz = 1;
                }

                T scale_factor_yxz = scale_factor_xz;

                if ((2 * parent_iterator.y_num(level - 1) != apr.y_num(level))) {
                    scale_factor_yxz = scale_factor_xz * 2;
                }


                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                        parent_iterator++;
                    }


                    if (parent_iterator.y() == (parent_iterator.y_num(level - 1) - 1)) {
                        tree_data[parent_iterator] +=
                                scale_factor_yxz * input_ptr[in_offset + apr_iterator] / 8.0f;
                    } else {

                        tree_data[parent_iterator] +=
                                scale_factor_xz * input_ptr[in_offset + apr_iterator] / 8.0f;
                    }

                }
            }
        }
        //}
        //}
    }

    ///then do the rest of the tree where order matters
    for (unsigned int level = std::min(current_max_level, (unsigned int)tree_iterator.level_max()); level > tree_iterator.level_min(); --level) {
        //z_d = 0;
        int z = 0;

        //for (z_d = 0; z_d < treeIterator.z_num(level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.z_num(level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < tree_iterator.x_num(level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_iterator.x_num(level)-1); ++x) {

                parent_iterator.begin(level - 1, z/2, x/2);

                T scale_factor_xz =
                        (((2 * parent_iterator.x_num(level - 1) != parent_iterator.x_num(level)) &&
                          ((x / 2) == (parent_iterator.x_num(level - 1) - 1))) +
                         ((2 * parent_iterator.z_num(level - 1) != parent_iterator.z_num(level)) &&
                          ((z / 2) == (parent_iterator.z_num(level - 1) - 1)))) * 2;

                if (scale_factor_xz == 0) {
                    scale_factor_xz = 1;
                }

                T scale_factor_yxz = scale_factor_xz;

                if ((2 * parent_iterator.y_num(level - 1) != parent_iterator.y_num(level))) {
                    scale_factor_yxz = scale_factor_xz * 2;
                }

                for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                    while (parent_iterator.y() != tree_iterator.y() / 2) {
                        parent_iterator++;
                    }

                    if (parent_iterator.y() == (parent_iterator.y_num(level - 1) - 1)) {
                        tree_data[parent_iterator] += scale_factor_yxz * tree_data[tree_iterator] / 8.0f;
                    } else {
                        tree_data[parent_iterator] += scale_factor_xz * tree_data[tree_iterator] / 8.0f;
                    }

                }
            }
        }
        //}
        //}
    }
}


//*****************************************************************************************************************
//                              Backward fill APRTree (push APRTree gradient values to APR particles)
//*****************************************************************************************************************

template<typename T>
void PyAPRFiltering::fill_tree_mean_backward(APR &apr, T *grad_input_ptr, ParticleData<T> &grad_tree_temp,
                                             uint64_t in_offset, unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();
    auto parent_iterator = apr.tree_iterator();

    //int z_d;
    int x_d;

    /// go through the tree from top (low level) to bottom (high level) and push values downwards
    for (unsigned int level = tree_iterator.level_min()+1;
         level <= std::min(current_max_level, (unsigned int)tree_iterator.level_max());
         ++level) {

        //z_d = 0;
        int z = 0;

        //for (z_d = 0; z_d < treeIterator.z_num(level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.z_num(level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents

        for (x_d = 0; x_d < tree_iterator.x_num(level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_iterator.x_num(level)-1); ++x) {

                parent_iterator.begin(level - 1, z/2, x/2);

                T scale_factor_xz =
                        (((2 * parent_iterator.x_num(level - 1) != parent_iterator.x_num(level)) &&
                          ((x / 2) == (parent_iterator.x_num(level - 1) - 1))) +
                         ((2 * parent_iterator.z_num(level - 1) != parent_iterator.z_num(level)) &&
                          ((z / 2) == (parent_iterator.z_num(level - 1) - 1)))) * 2;

                if (scale_factor_xz == 0) {
                    scale_factor_xz = 1;
                }

                T scale_factor_yxz = scale_factor_xz;

                if ((2 * parent_iterator.y_num(level - 1) != parent_iterator.y_num(level))) {
                    scale_factor_yxz = scale_factor_xz * 2;
                }

                for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                    while (parent_iterator.y() != tree_iterator.y() / 2) {
                        parent_iterator++;
                    }

                    if (parent_iterator.y() == (parent_iterator.y_num(level - 1) - 1)) {
                        //tree_data[parentIterator] = scale_factor_yxz * tree_data[treeIterator] / 8.0f +
                        //                            tree_data[parentIterator];
                        grad_tree_temp[tree_iterator] += grad_tree_temp[parent_iterator] * scale_factor_yxz / 8.0f;

                    } else {
                        //tree_data[parentIterator] = scale_factor_xz * tree_data[treeIterator] / 8.0f +
                        //                            tree_data[parentIterator];
                        grad_tree_temp[tree_iterator] += grad_tree_temp[parent_iterator] * scale_factor_xz / 8.0f;
                    }

                }
            }
        }
        //}
        //}
    }

    /// go through apr particles and transfer values from grad_tree_temp to grad_input
    for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
        //int z_d = 0;
        int z = 0;

        //for (z_d = 0; z_d < parentIterator.z_num(level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.z_num(level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents

        for (x_d = 0; x_d < parent_iterator.x_num(level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.x_num(level)-1); ++x) {

                parent_iterator.begin(level - 1, z / 2, x / 2);

                //dealing with boundary conditions
                T scale_factor_xz =
                        (((2 * parent_iterator.x_num(level - 1) != apr.x_num(level)) &&
                          ((x / 2) == (parent_iterator.x_num(level - 1) - 1))) +
                         ((2 * parent_iterator.z_num(level - 1) != apr.z_num(level)) &&
                          (z / 2) == (parent_iterator.z_num(level - 1) - 1))) * 2;

                if (scale_factor_xz == 0) {
                    scale_factor_xz = 1;
                }

                T scale_factor_yxz = scale_factor_xz;

                if ((2 * parent_iterator.y_num(level - 1) != apr.y_num(level))) {
                    scale_factor_yxz = scale_factor_xz * 2;
                }


                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                    while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                        parent_iterator++;
                    }


                    if (parent_iterator.y() == (parent_iterator.y_num(level - 1) - 1)) {
                        //tree_data[parentIterator] =
                        //        scale_factor_yxz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                        //        tree_data[parentIterator];

                        grad_input_ptr[in_offset + apr_iterator] +=
                                scale_factor_yxz * grad_tree_temp[parent_iterator] / 8.0f;
                    } else {

                        //tree_data[parentIterator] =
                        //        scale_factor_xz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                        //        tree_data[parentIterator];

                        grad_input_ptr[in_offset + apr_iterator] +=
                                scale_factor_xz * grad_tree_temp[parent_iterator] / 8.0f;
                    }

                }
            }
        }
        //}
        //}
    }

    /// if downsampling has been performed, the downsampled values have to be accessed via the tree iterator
    if(current_max_level < apr.level_max()) {

        int z=0;
        int x;

        int64_t tree_offset_in  = compute_tree_offset(apr_iterator, tree_iterator, current_max_level);

        //for (z = 0; z < treeIterator.z_num(current_max_level); z++) {

        for (x = 0; x < tree_iterator.x_num(current_max_level); ++x) {
            for (tree_iterator.begin(current_max_level, z, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                //tree_data[treeIterator] = input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                grad_input_ptr[in_offset + tree_iterator + tree_offset_in] += grad_tree_temp[tree_iterator];

            }
        }//x
        //}//z
    }//if
}


//*****************************************************************************************************************
//                              Forward 2x2 maxpooling
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::max_pool_2x2(APR &apr, T *input_ptr, T *output_ptr, uint64_t *idx_ptr,
                                  const uint64_t in_offset, const uint64_t out_offset, const int64_t tree_offset_in,
                                  const int64_t tree_offset_out, const unsigned int current_max_level){

    auto apr_iterator = apr.iterator();
    auto treeIterator = apr.tree_iterator();
    auto parentIterator = apr.tree_iterator();

    /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged
    for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
        int z = 0;
        int x = 0;

        for (x = 0; x < apr_iterator.x_num(level); ++x) {
            for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                output_ptr[apr_iterator + out_offset] = input_ptr[apr_iterator + in_offset];
                idx_ptr[apr_iterator + out_offset] = apr_iterator + in_offset;
            }
        }
    }

    /// Downsample the APR particles
    //int z_d = 0;
    int z = 0;
    unsigned int x;

    for(x = 0; x < apr_iterator.x_num(current_max_level); ++x) {

        parentIterator.begin(current_max_level - 1, z / 2, x / 2);

        for (apr_iterator.begin(current_max_level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

            while (parentIterator.y() != (apr_iterator.y() / 2)) {
                parentIterator++;
            }

            uint64_t out_idx = (parentIterator + out_offset) + tree_offset_out;
            uint64_t in_idx = apr_iterator + in_offset;

            T tmp = input_ptr[in_idx];

            if(tmp > output_ptr[out_idx]) {
                idx_ptr[out_idx] = in_idx;
                output_ptr[out_idx] = tmp;
            }
        }
    }

    /// Downsample tree particles
    if( current_max_level < apr.level_max()) {
        //int z_d = 0;
        int z = 0;

        for(x = 0; x < treeIterator.x_num(current_max_level); ++x) {

            parentIterator.begin(current_max_level - 1, z / 2, x / 2);

            for (treeIterator.begin(current_max_level, z, x); treeIterator < treeIterator.end(); ++treeIterator) {

                while (parentIterator.y() != (treeIterator.y() / 2)) {
                    parentIterator++;
                }

                uint64_t out_idx = (parentIterator + out_offset) + tree_offset_out;
                uint64_t in_idx = (treeIterator + in_offset) + tree_offset_in;

                T tmp = input_ptr[in_idx];

                if(tmp > output_ptr[out_idx]) {
                    idx_ptr[out_idx] = in_idx;
                    output_ptr[out_idx] = tmp;
                }
            }
        }
    }
}


//*****************************************************************************************************************
//                              Compute APRTree index offset
//*****************************************************************************************************************
template<typename APRIteratorType, typename APRTreeIteratorType>
int64_t PyAPRFiltering::compute_tree_offset(APRIteratorType &apr_iterator, APRTreeIteratorType &tree_iterator, unsigned int level){

    int64_t number_parts;
    int64_t tree_start;

    if(level >= apr_iterator.level_min()) {
        number_parts = (int64_t) apr_iterator.particles_level_end(level);//apr.apr_access.global_index_by_level_and_zx_end[level].back();
    } else {
        number_parts = 0;
    }

    if(level > tree_iterator.level_min()) {
        tree_start = (int64_t) tree_iterator.particles_level_begin(level); //apr.apr_tree.tree_access.global_index_by_level_and_zx_end[level - 1].back();
    } else {
        tree_start = 0;
    }

    return number_parts - tree_start;
}


template<typename T>
void PyAPRFiltering::max_pool_backward_stored_indices(T* grad_input_ptr, T* grad_output_ptr, uint64_t* index_ptr, uint64_t in_size, uint64_t out_size) {

    size_t i;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
    for (i = 0; i < out_size; ++i) {
        grad_input_ptr[index_ptr[i]] = grad_output_ptr[i];
    }
}



/**
* Computes the required number of intensity values required to represent the image with a given maximum level.
*/
template<typename APRIteratorType, typename APRTreeIteratorType>
uint64_t PyAPRFiltering::number_parts_at_level(APRIteratorType &apr_iterator, APRTreeIteratorType &tree_iterator, unsigned int max_level){

    unsigned int number_parts;
    unsigned int tree_start;
    unsigned int tree_end;

    if(max_level >= apr_iterator.level_min()) {
        number_parts = apr_iterator.particles_level_end(max_level); //apr.apr_access.global_index_by_level_and_zx_end[max_level].back();
    } else {
        number_parts = 0;
    }

    if(max_level > tree_iterator.level_min()) {
        tree_start = tree_iterator.particles_level_begin(max_level); //apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level - 1].back();
    } else {
        tree_start = 0;
    }

    if(max_level >= tree_iterator.level_min()) {
        tree_end = tree_iterator.particles_level_end(max_level); //apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level].back();
    } else {
        tree_end = 0;
    }

    unsigned int number_graduated_parts = tree_end - tree_start;

    uint64_t number_parts_out = number_parts + number_graduated_parts;

    return number_parts_out;
}


//*****************************************************************************************************************
//                              Forward 2x2 transposed convolution
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::transposed_conv_2x2(APR &apr, T *input_ptr, T *output_ptr, PixelData<T>& stencil_2x2, T weight_1x1,
                                         const T bias, const uint64_t in_offset, const uint64_t out_offset, const int64_t tree_offset_in,
                                         const int64_t tree_offset_out, const int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();
    auto parent_iterator = apr.tree_iterator();

    /// APR particles at or below current_max_level undergo a 1x1 convolution

    for (int level = apr_iterator.level_min(); level <= current_max_level; ++level) {
        for (int x = 0; x < apr_iterator.x_num(level); ++x) {
            for (apr_iterator.begin(level, 0, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                output_ptr[apr_iterator + out_offset] += input_ptr[apr_iterator + in_offset] * weight_1x1 + bias;

            }
        }
    }

    /// Upsampling is performed for all tree (!) particles at current_max_level.
    /// The output particles can belong to the APR or the tree.

    //loop over output APR particles
    for(int x = 0; x < apr_iterator.x_num(current_max_level + 1); ++x) {

        parent_iterator.begin(current_max_level, 0, x / 2);

        for (apr_iterator.begin(current_max_level + 1, 0, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

            while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                parent_iterator++;
            }

            output_ptr[out_offset + apr_iterator] += stencil_2x2.mesh[(x & 1) * 2 + (apr_iterator.y() & 1)] *
                                                     input_ptr[in_offset + tree_offset_in + parent_iterator] +
                                                     bias;
        }
    }

    //loop over output tree particles
    if(current_max_level < tree_iterator.level_max()) { //tree_iterator.max_level() = apr_iterator.max_level() - 1

        for(int x = 0; x < tree_iterator.x_num(current_max_level + 1); ++x) {

        parent_iterator.begin(current_max_level, 0, x / 2);

            for (tree_iterator.begin(current_max_level + 1, 0, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                while (parent_iterator.y() != (tree_iterator.y() / 2)) {
                    parent_iterator++;
                }

                output_ptr[out_offset + tree_offset_out + tree_iterator] += stencil_2x2.mesh[(x & 1) * 2 + (tree_iterator.y() & 1)] *
                                                                            input_ptr[in_offset + tree_offset_in + parent_iterator] +
                                                                            bias;
            }
        }
    }
}


//*****************************************************************************************************************
//                              Backward 2x2 transposed convolution
//*****************************************************************************************************************
template<typename T>
void PyAPRFiltering::transposed_conv_2x2_backward(APR &apr, T *input_ptr, PixelData<T>& stencil_2x2, T weight_1x1,
                        T* grad_input_ptr, std::vector<float>& temp_dw_2x2, std::vector<float>& temp_dw_1x1,
                        std::vector<float>& temp_db, T* grad_output_ptr, const size_t dw_2x2_offset, const size_t dw_1x1_offset,
                        const size_t db_offset, const uint64_t in_offset, const uint64_t out_offset,
                        const int64_t tree_offset_in, const int64_t tree_offset_out, const int current_max_level){

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.tree_iterator();
    auto parent_iterator = apr.tree_iterator();

    T d_bias = 0;

    /// APR particles at or below current_max_level undergo a 1x1 convolution

    for (int level = apr_iterator.level_min(); level <= current_max_level; ++level) {
        for (int x = 0; x < apr_iterator.x_num(level); ++x) {
            for (apr_iterator.begin(level, 0, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

                const T dO = grad_output_ptr[apr_iterator + out_offset];

                d_bias += dO;
                temp_dw_1x1[dw_1x1_offset] += dO * input_ptr[apr_iterator + in_offset];
                grad_input_ptr[apr_iterator + in_offset] += dO * weight_1x1;
            }
        }
    }

    /// Upsampling is performed for all tree (!) particles at current_max_level.
    /// The output particles can belong to the APR or the tree.

    //loop over output APR particles
    for(int x = 0; x < apr_iterator.x_num(current_max_level + 1); ++x) {

        parent_iterator.begin(current_max_level, 0, x / 2);

        for (apr_iterator.begin(current_max_level + 1, 0, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {

            while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                parent_iterator++;
            }

            const T dO = grad_output_ptr[out_offset + apr_iterator];

            d_bias += dO;
            grad_input_ptr[in_offset + tree_offset_in + parent_iterator] += dO * stencil_2x2.mesh[(x & 1) * 2 + (apr_iterator.y() & 1)];
            temp_dw_2x2[dw_2x2_offset + (x & 1) * 2 + (apr_iterator.y() & 1)] += dO * input_ptr[in_offset + tree_offset_in + parent_iterator];
        }
    }

    //loop over output tree particles
    if(current_max_level < tree_iterator.level_max()) { //tree_iterator.max_level() = apr_iterator.max_level() - 1

        for(int x = 0; x < tree_iterator.x_num(current_max_level + 1); ++x) {

            parent_iterator.begin(current_max_level, 0, x / 2);

            for (tree_iterator.begin(current_max_level + 1, 0, x); tree_iterator < tree_iterator.end(); ++tree_iterator) {

                while (parent_iterator.y() != (tree_iterator.y() / 2)) {
                    parent_iterator++;
                }

                const T dO = grad_output_ptr[out_offset + tree_offset_out + tree_iterator];

                d_bias += dO;
                grad_input_ptr[in_offset + tree_offset_in + parent_iterator] += dO * stencil_2x2.mesh[(x & 1) * 2 + (tree_iterator.y() & 1)];
                temp_dw_2x2[dw_2x2_offset + (x & 1) * 2 + (tree_iterator.y() & 1)] += dO * input_ptr[in_offset + tree_offset_in + parent_iterator];
            }
        }
    }

    temp_db[db_offset] += d_bias;
}

#endif //PYLIBAPR_PYAPRFILTERING_HPP
