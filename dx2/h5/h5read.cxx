#include "dx2/h5/h5read.hpp"
#include <fmt/format.h>
#include <unordered_set>

namespace {

/// Collects link names for H5Literate2.
herr_t collect_name(hid_t, const char *name, const H5L_info2_t *,
                    void *op_data) {
  static_cast<std::vector<std::string> *>(op_data)->emplace_back(name);
  return 0;
}

/// Appends every dataset under loc_id to datasets, recursing into groups
/// and visiting each group path once.
void traverse_datasets(hid_t loc_id, const std::string &path,
                       std::vector<std::string> &datasets,
                       std::unordered_set<std::string> &visited_groups) {
  for (const std::string &name : list_link_names(loc_id)) {
    const std::string full_path = path + "/" + name;
    H5O_info2_t obj_info;
    if (H5Oget_info_by_name3(loc_id, name.c_str(), &obj_info, H5O_INFO_BASIC,
                             H5P_DEFAULT) < 0) {
      dx2_log::error(
          fmt::format("Unable to get object info for: {}", full_path));
      continue;
    }
    if (obj_info.type == H5O_TYPE_DATASET) {
      datasets.push_back(full_path);
    } else if (obj_info.type == H5O_TYPE_GROUP &&
               visited_groups.insert(full_path).second) {
      h5utils::H5Group group(H5Gopen2(loc_id, name.c_str(), H5P_DEFAULT));
      if (!group) {
        dx2_log::error(fmt::format("Unable to open group: {}", full_path));
        continue;
      }
      traverse_datasets(group, full_path, datasets, visited_groups);
    }
  }
}

} // namespace

std::vector<std::string> list_link_names(hid_t group) {
  std::vector<std::string> names;
  if (H5Literate2(group, H5_INDEX_NAME, H5_ITER_INC, nullptr, collect_name,
                  &names) < 0) {
    throw std::runtime_error("Error: Unable to iterate group links.");
  }
  return names;
}

std::vector<std::string> get_datasets_in_group(std::string_view filename,
                                               std::string_view group_name) {
  std::string fname(filename);
  std::string gpath(group_name);
  h5utils::H5File file(H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!file) {
    throw std::runtime_error("Error: Unable to open file: " + fname);
  }
  h5utils::H5Group group(H5Gopen2(file, gpath.c_str(), H5P_DEFAULT));
  if (!group) {
    dx2_log::warning(fmt::format("Missing group '{}', skipping.", gpath));
    return {};
  }

  // Only links that lead to a dataset are kept.
  std::vector<std::string> datasets;
  for (const std::string &name : list_link_names(group)) {
    H5O_info2_t obj_info;
    if (H5Oget_info_by_name3(group, name.c_str(), &obj_info, H5O_INFO_BASIC,
                             H5P_DEFAULT) < 0) {
      continue;
    }
    if (obj_info.type == H5O_TYPE_DATASET) {
      datasets.push_back(gpath + "/" + name);
    }
  }
  return datasets;
}

std::vector<std::string>
get_datasets_in_group_recursive(std::string_view filename,
                                std::string_view group_name) {
  std::string fname(filename);
  std::string gpath(group_name);
  h5utils::H5File file(H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!file) {
    throw std::runtime_error("Error: Unable to open file: " + fname);
  }
  h5utils::H5Group group(H5Gopen2(file, gpath.c_str(), H5P_DEFAULT));
  if (!group) {
    dx2_log::warning(fmt::format("Missing group '{}', skipping.", gpath));
    return {};
  }
  std::vector<std::string> datasets;
  std::unordered_set<std::string> visited_groups;
  traverse_datasets(group, gpath, datasets, visited_groups);
  return datasets;
}

std::string get_dataset_name(std::string_view path) {
  size_t pos = path.find_last_of('/');
  if (pos == std::string_view::npos) {
    return std::string(path);
  }
  return std::string(path.substr(pos + 1));
}
