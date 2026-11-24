/**
 * @file h5read.hpp
 * @brief Reads typed arrays and lists the contents of groups in HDF5 files.
 *
 * These are the read-side counterparts of h5write.hpp and know nothing
 * about what the data means: a dataset is an array with a shape and an
 * element type, a group is a list of links. Readers that attach meaning,
 * such as the reflection table or the image sources, build on these. A
 * dataset is read whole into a vector whose element type the caller
 * names; the dataset's element size must match it, which is the only
 * type check performed, so a float dataset read as int32_t is accepted
 * and misread. Every function that takes a filename opens and closes the
 * file itself, which suits one-off reads of small arrays and not a loop
 * over many datasets in one file; list_link_names takes an open group for
 * that reason.
 */

#pragma once

#include "dx2/h5/h5utils.hpp"
#include "dx2/logging.hpp"
#include <chrono>
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/// A dataset read whole: its elements in row-major order and its shape.
template <typename T> struct H5ArrayData {
  std::vector<T> data;
  std::vector<size_t> shape;
};

/**
 * @brief Names of the links directly under an open group, in name order.
 *
 * Every link is listed whatever it points at, including soft and external
 * links whose target does not exist, so the result can be resolved with
 * h5link before anything is opened.
 *
 * @param group An open group or file.
 * @throws std::runtime_error if the group cannot be iterated.
 */
std::vector<std::string> list_link_names(hid_t group);

/**
 * @brief Full paths of the datasets directly under a group.
 *
 * Links to anything other than a dataset, and links that cannot be
 * followed, are left out. A missing group is logged and gives an empty
 * list.
 *
 * @param filename The HDF5 file.
 * @param group_name Absolute path of the group, such as
 * "/dials/processing/group_0".
 * @throws std::runtime_error if the file cannot be opened.
 */
std::vector<std::string> get_datasets_in_group(std::string_view filename,
                                               std::string_view group_name);

/**
 * @brief Full paths of every dataset under a group and its subgroups.
 *
 * Groups reached by more than one path are visited once.
 */
std::vector<std::string>
get_datasets_in_group_recursive(std::string_view filename,
                                std::string_view group_name);

/**
 * @brief Reads a dataset whole, with its shape.
 *
 * @tparam T The element type to read as; its size must equal the stored
 * element size.
 * @param filename The HDF5 file.
 * @param dataset_name Absolute path of the dataset.
 * @throws std::runtime_error if the file or dataset cannot be opened, the
 * element size differs from sizeof(T), or the read fails.
 */
template <typename T>
H5ArrayData<T>
read_array_with_shape_from_h5_file(std::string_view filename,
                                   std::string_view dataset_name) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string fname(filename);
  std::string dset_name(dataset_name);

  h5utils::H5File file(H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!file) {
    throw std::runtime_error("Error: Unable to open file: " + fname);
  }
  h5utils::H5Dataset dataset(H5Dopen(file, dset_name.c_str(), H5P_DEFAULT));
  if (!dataset) {
    throw std::runtime_error("Error: Unable to open dataset: " + dset_name);
  }

  // The stored element size is the only check on T.
  h5utils::H5Type datatype(H5Dget_type(dataset));
  if (H5Tget_size(datatype) != sizeof(T)) {
    throw std::runtime_error(
        "Error: Dataset type size does not match expected type size.");
  }

  h5utils::H5Space dataspace(H5Dget_space(dataset));
  int ndims = H5Sget_simple_extent_ndims(dataspace);
  if (ndims <= 0) {
    throw std::runtime_error("Error: Dataset has invalid dimensionality.");
  }
  std::vector<hsize_t> dims(ndims);
  H5Sget_simple_extent_dims(dataspace, dims.data(), nullptr);
  size_t num_elements = 1;
  for (auto d : dims) {
    num_elements *= d;
  }

  std::vector<T> data_out(num_elements);
  herr_t status = H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          data_out.data());
  if (status < 0) {
    throw std::runtime_error("Error: Unable to read dataset: " + dset_name);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed_time =
      std::chrono::duration<double>(end_time - start_time).count();
  dx2_log::debug("READ TIME for {} : {:.4f}s", dset_name, elapsed_time);

  return {std::move(data_out), std::vector<size_t>(dims.begin(), dims.end())};
}

/// Reads a dataset whole, discarding its shape.
template <typename T>
std::vector<T> read_array_from_h5_file(std::string_view filename,
                                       std::string_view dataset_name) {
  return read_array_with_shape_from_h5_file<T>(filename, dataset_name).data;
}

/// The last component of an object path: "c" for "/a/b/c".
std::string get_dataset_name(std::string_view path);
