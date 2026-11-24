/**
 * @file h5write.cxx
 * @brief Implementation file for HDF5 writing utilities.
 */

#include "dx2/h5/h5write.hpp"
#include <fmt/core.h>
#include <stdexcept>

#pragma region Raw writer
h5utils::H5Group traverse_or_create_groups(hid_t parent,
                                           const std::string &path) {
  // Strip leading '/' characters, if any, to prevent empty group names
  size_t start_pos = path.find_first_not_of('/');
  if (start_pos == std::string::npos) {
    return h5utils::H5Group(parent); // Path is just "/", return parent as-is
  }

  std::string cleaned_path = path.substr(start_pos);

  /*
   * This is the base case for recursion. When the path is empty, we
   * have reached the final group in the path and we return the parent
   * group.
   */
  if (cleaned_path.empty()) {
    return h5utils::H5Group(parent);
  }

  // Split the path into the current group name and the remaining path
  size_t pos = cleaned_path.find('/');
  std::string group_name =
      (pos == std::string::npos) ? cleaned_path : cleaned_path.substr(0, pos);
  std::string remaining_path =
      (pos == std::string::npos) ? "" : cleaned_path.substr(pos + 1);

  // Try to open group, suppress errors if not found
  H5ErrorSilencer silencer;
  h5utils::H5Group next_group(H5Gopen(parent, group_name.c_str(), H5P_DEFAULT));

  // If the group does not exist, create it
  if (!next_group) {
    next_group = h5utils::H5Group(H5Gcreate(
        parent, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (!next_group) {
      throw std::runtime_error("Error: Unable to create or open group: " +
                               group_name);
    }
  }

  // If there are no remaining path components, return the current group
  if (remaining_path.empty()) {
    return next_group;
  }

  // Recursively traverse or create the next group in the path
  return traverse_or_create_groups(next_group, remaining_path);
}
#pragma endregion
