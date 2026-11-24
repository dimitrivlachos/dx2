#include "dx2/h5/h5link.hpp"
#include <cstdlib>
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdexcept>

namespace dx2::h5link {

namespace fs = std::filesystem;

namespace {

/// Links followed before a path is declared a cycle.
constexpr int MAX_LINK_HOPS = 32;

/// Splits an object path on "/" into its non-empty components.
std::vector<std::string> split_path(std::string_view path) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start < path.size()) {
    const size_t end = path.find('/', start);
    const std::string_view part =
        path.substr(start, end == std::string_view::npos ? end : end - start);
    if (!part.empty()) {
      parts.emplace_back(part);
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return parts;
}

/// Joins components back into an absolute object path, "/" when empty.
std::string join_path(std::span<const std::string> parts) {
  std::string path;
  for (const std::string &part : parts) {
    path += '/';
    path += part;
  }
  return path.empty() ? "/" : path;
}

/// Resolves a soft link target, relative to the group holding the link,
/// into an absolute component list with "." and ".." applied.
std::vector<std::string> apply_soft_target(std::span<const std::string> group,
                                           std::string_view target) {
  std::vector<std::string> parts;
  if (target.empty() || target.front() != '/') {
    parts.assign(group.begin(), group.end());
  }
  for (const std::string &part : split_path(target)) {
    if (part == ".") {
      continue;
    }
    if (part == "..") {
      if (!parts.empty()) {
        parts.pop_back();
      }
      continue;
    }
    parts.push_back(part);
  }
  return parts;
}

fs::path absolute_or_as_given(const fs::path &path) {
  std::error_code ec;
  const fs::path absolute = fs::absolute(path, ec);
  return ec ? path : absolute.lexically_normal();
}

} // namespace

std::vector<fs::path>
link_search_prefixes(LinkKind kind, const std::optional<fs::path> &prefix,
                     const fs::path &link_file) {
  std::vector<fs::path> prefixes;
  // The environment variable holds a colon separated list and comes first.
  const char *env_name =
      kind == LinkKind::VirtualSource ? "HDF5_VDS_PREFIX" : "HDF5_EXT_PREFIX";
  if (const char *env = std::getenv(env_name); env && *env != '\0') {
    for (const std::string &part : [&] {
           std::vector<std::string> parts;
           std::string_view rest(env);
           while (!rest.empty()) {
             const size_t colon = rest.find(':');
             parts.emplace_back(rest.substr(0, colon));
             rest = colon == std::string_view::npos ? std::string_view{}
                                                    : rest.substr(colon + 1);
           }
           return parts;
         }()) {
      if (!part.empty()) {
        prefixes.emplace_back(part);
      }
    }
  }
  // Then the caller's prefix, then the directory of the linking file, then
  // the name as given.
  if (prefix) {
    prefixes.push_back(*prefix);
  }
  prefixes.push_back(absolute_or_as_given(link_file).parent_path());
  prefixes.emplace_back();
  return prefixes;
}

ResolvedLink find_linked_file(std::string_view name, const fs::path &link_file,
                              std::span<const fs::path> prefixes) {
  if (name == ".") {
    return {absolute_or_as_given(link_file), "", fs::exists(link_file)};
  }
  const fs::path given(name);
  if (given.is_absolute()) {
    return {given.lexically_normal(), "", fs::exists(given)};
  }
  // The first existing candidate wins; failing that, the first candidate
  // names where the file is expected to appear.
  std::optional<fs::path> first;
  for (const fs::path &prefix : prefixes) {
    const fs::path candidate = absolute_or_as_given(prefix / given);
    if (!first) {
      first = candidate;
    }
    if (fs::exists(candidate)) {
      return {candidate, "", true};
    }
  }
  return {first.value_or(absolute_or_as_given(given)), "", false};
}

ResolvedLink resolve_link(hid_t file, const fs::path &file_path,
                          std::string_view object_path,
                          std::span<const fs::path> prefixes) {
  // The walk owns any file it opens along the way; the first file belongs
  // to the caller.
  h5utils::H5File owned;
  hid_t current = file;
  fs::path current_path = absolute_or_as_given(file_path);
  std::vector<std::string> remaining = split_path(object_path);
  std::vector<std::string> walked;
  int hops = 0;

  while (!remaining.empty()) {
    walked.push_back(remaining.front());
    remaining.erase(remaining.begin());
    const std::string here = join_path(walked);

    // A missing component is reported as an exception, not an HDF5 trace.
    H5L_info2_t info{};
    bool found = false;
    {
      h5utils::H5ErrorSilencer silence;
      found = H5Lget_info2(current, here.c_str(), &info, H5P_DEFAULT) >= 0;
    }
    if (!found) {
      throw std::runtime_error(
          fmt::format("no link {} in {}", here, current_path));
    }
    if (info.type == H5L_TYPE_HARD) {
      continue;
    }
    if (++hops > MAX_LINK_HOPS) {
      throw std::runtime_error(fmt::format("more than {} links on {} in {}",
                                           MAX_LINK_HOPS, here, current_path));
    }

    // Symbolic links carry their target as a value whose size the info
    // reports.
    std::string value(info.u.val_size, '\0');
    if (H5Lget_val(current, here.c_str(), value.data(), value.size(),
                   H5P_DEFAULT) < 0) {
      throw std::runtime_error(
          fmt::format("cannot read link {} in {}", here, current_path));
    }

    if (info.type == H5L_TYPE_SOFT) {
      // The target replaces the component just walked; the rest of the
      // path continues under it.
      walked.pop_back();
      std::vector<std::string> target =
          apply_soft_target(walked, std::string_view(value.c_str()));
      target.insert(target.end(), remaining.begin(), remaining.end());
      remaining = std::move(target);
      walked.clear();
      continue;
    }

    if (info.type == H5L_TYPE_EXTERNAL) {
      const char *target_file = nullptr;
      const char *target_object = nullptr;
      unsigned flags = 0;
      if (H5Lunpack_elink_val(value.data(), value.size(), &flags, &target_file,
                              &target_object) < 0) {
        throw std::runtime_error(fmt::format(
            "cannot unpack external link {} in {}", here, current_path));
      }
      ResolvedLink found =
          find_linked_file(target_file, current_path, prefixes);
      std::vector<std::string> target = split_path(target_object);
      target.insert(target.end(), remaining.begin(), remaining.end());
      // A missing file ends the walk with the path still to be followed.
      if (!found.file_exists) {
        return {found.file, join_path(target), false};
      }
      owned = h5utils::H5File(
          H5Fopen(found.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
      if (!owned) {
        throw std::runtime_error(
            fmt::format("cannot open {} linked from {} in {}", found.file, here,
                        current_path));
      }
      current = owned;
      current_path = found.file;
      remaining = std::move(target);
      walked.clear();
      continue;
    }

    throw std::runtime_error(fmt::format(
        "link {} in {} has an unsupported type", here, current_path));
  }
  return {current_path, join_path(walked), true};
}

} // namespace dx2::h5link
