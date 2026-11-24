/**
 * @file dials_format.hpp
 * @brief The attributes DIALS requires on a processed-data HDF5 file.
 *
 * DIALS reads reflection tables from HDF5 only when the file root carries
 * a `file_type` attribute equal to "dials_processed_data" and a
 * `file_version` of 1, and it recovers the experiments each row belongs to
 * from `experiment_ids` and `identifiers` attributes on the table's group.
 * The check compares Python objects, so the attribute types matter as much
 * as their values: `file_type` must be a scalar variable-length UTF-8
 * string, which h5py reads back as `str`, and `file_version` a scalar
 * 64-bit integer. A fixed-length string reads back as `bytes` and a
 * one-element array is not a scalar; both fail the check. The format is
 * defined in `dials/util/table_as_hdf5_file.py` (dials/dials@02bb48d).
 */

#pragma once

#include <cstdint>
#include <hdf5.h>
#include <string>
#include <string_view>
#include <vector>

/// Value of the root `file_type` attribute DIALS (>= 3.31) requires.
constexpr std::string_view DIALS_FILE_TYPE = "dials_processed_data";

/// Version of the DIALS processed-data layout dx2 writes. Version 1 keeps
/// each table at `/dials/<process>/<group>`.
constexpr int64_t DIALS_FILE_VERSION = 1;

/**
 * @brief Reads the `experiment_ids` and `identifiers` attributes of a
 * table group.
 *
 * A missing attribute leaves its output untouched; one that cannot be read
 * is logged and left empty.
 */
void read_experiment_metadata(hid_t group_id,
                              std::vector<uint64_t> &experiment_ids,
                              std::vector<std::string> &identifiers);

/**
 * @brief Writes the `experiment_ids` and `identifiers` attributes of a
 * table group.
 *
 * @throws std::runtime_error if either vector is empty or an attribute
 * cannot be written.
 */
void write_experiment_metadata(hid_t group_id,
                               const std::vector<uint64_t> &experiment_ids,
                               const std::vector<std::string> &identifiers);

/**
 * @brief Marks an HDF5 file as DIALS processed data.
 *
 * @param file_id An open, writable HDF5 file without these attributes.
 * @throws std::runtime_error if either attribute cannot be written.
 */
void write_file_format_attributes(hid_t file_id);
