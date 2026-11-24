#include "dx2/dials_format.hpp"
#include "dx2/h5/h5utils.hpp"
#include "dx2/logging.hpp"
#include <fmt/core.h>
#include <stdexcept>

namespace {

/// A variable-length UTF-8 string type, the type h5py gives a Python str.
h5utils::H5Type make_utf8_string_type(const char *purpose) {
  h5utils::H5Type str_type(H5Tcopy(H5T_C_S1));
  if (!str_type || H5Tset_size(str_type, H5T_VARIABLE) < 0 ||
      H5Tset_cset(str_type, H5T_CSET_UTF8) < 0 ||
      H5Tset_strpad(str_type, H5T_STR_NULLTERM) < 0) {
    throw std::runtime_error(
        fmt::format("Failed to create {} string type", purpose));
  }
  return str_type;
}

} // namespace

void read_experiment_metadata(hid_t group_id,
                              std::vector<uint64_t> &experiment_ids,
                              std::vector<std::string> &identifiers) {
  if (H5Aexists(group_id, "experiment_ids") > 0) {
    h5utils::H5Attr attr(H5Aopen(group_id, "experiment_ids", H5P_DEFAULT));
    h5utils::H5Space space(H5Aget_space(attr));
    hssize_t num_elements = H5Sget_simple_extent_npoints(space);

    experiment_ids.resize(num_elements);
    if (H5Aread(attr, H5T_NATIVE_ULLONG, experiment_ids.data()) < 0) {
      dx2_log::warning(fmt::format(
          "Failed to read experiment_ids attribute ({} entries), skipping.",
          num_elements));
      experiment_ids.clear();
    }
  }

  if (H5Aexists(group_id, "identifiers") > 0) {
    h5utils::H5Attr attr(H5Aopen(group_id, "identifiers", H5P_DEFAULT));
    h5utils::H5Type type(H5Aget_type(attr));
    h5utils::H5Space space(H5Aget_space(attr));
    hssize_t num_elements = H5Sget_simple_extent_npoints(space);

    std::vector<char *> raw_strings(num_elements);
    if (H5Aread(attr, type, raw_strings.data()) < 0) {
      dx2_log::warning(fmt::format(
          "Failed to read identifiers attribute ({} entries), skipping.",
          num_elements));
      return;
    }

    // A null entry is a valid HDF5 variable-length string and reads as empty
    identifiers.clear();
    identifiers.reserve(num_elements);
    for (const char *raw : raw_strings) {
      identifiers.emplace_back(raw ? raw : "");
    }

    // Variable-length strings are allocated by HDF5 during the read
    H5Treclaim(type, space, H5P_DEFAULT, raw_strings.data());
  }
}

void write_experiment_metadata(hid_t group_id,
                               const std::vector<uint64_t> &experiment_ids,
                               const std::vector<std::string> &identifiers) {
  if (experiment_ids.empty() || identifiers.empty()) {
    throw std::runtime_error(
        "Experiment IDs and identifiers must not be empty.");
  }

  {
    hsize_t dims = experiment_ids.size();
    h5utils::H5Space space(H5Screate_simple(1, &dims, nullptr));
    h5utils::H5Attr attr(H5Acreate2(group_id, "experiment_ids",
                                    H5T_NATIVE_ULLONG, space, H5P_DEFAULT,
                                    H5P_DEFAULT));
    if (!space || !attr ||
        H5Awrite(attr, H5T_NATIVE_ULLONG, experiment_ids.data()) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to write experiment_ids attribute ({} entries)",
                      experiment_ids.size()));
    }
  }

  {
    hsize_t dims = identifiers.size();
    h5utils::H5Space space(H5Screate_simple(1, &dims, nullptr));
    h5utils::H5Type str_type = make_utf8_string_type("identifiers");

    std::vector<const char *> c_strs;
    for (const auto &s : identifiers) {
      c_strs.push_back(s.c_str());
    }

    h5utils::H5Attr attr(H5Acreate2(group_id, "identifiers", str_type, space,
                                    H5P_DEFAULT, H5P_DEFAULT));
    if (!space || !attr || H5Awrite(attr, str_type, c_strs.data()) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to write identifiers attribute ({} entries)",
                      identifiers.size()));
    }
  }
}

void write_file_format_attributes(hid_t file_id) {
  h5utils::H5Space scalar(H5Screate(H5S_SCALAR));
  if (!scalar) {
    throw std::runtime_error("Failed to create scalar dataspace");
  }

  {
    h5utils::H5Type str_type = make_utf8_string_type("file_type");
    // A variable-length string is written as a pointer to a C string
    const std::string value(DIALS_FILE_TYPE);
    const char *c_str = value.c_str();
    h5utils::H5Attr attr(H5Acreate2(file_id, "file_type", str_type, scalar,
                                    H5P_DEFAULT, H5P_DEFAULT));
    if (!attr || H5Awrite(attr, str_type, &c_str) < 0) {
      throw std::runtime_error("Failed to write file_type attribute");
    }
  }

  {
    const int64_t value = DIALS_FILE_VERSION;
    h5utils::H5Attr attr(H5Acreate2(file_id, "file_version", H5T_NATIVE_INT64,
                                    scalar, H5P_DEFAULT, H5P_DEFAULT));
    if (!attr || H5Awrite(attr, H5T_NATIVE_INT64, &value) < 0) {
      throw std::runtime_error("Failed to write file_version attribute");
    }
  }
}
