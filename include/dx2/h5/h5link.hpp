/**
 * @file h5link.hpp
 * @brief Resolves an HDF5 object path to the file and object it names.
 *
 * NeXus master files point at their data through links: soft links within
 * the master, external links into per-module data files, and the source
 * file names recorded in a virtual dataset. H5Dopen2 follows all of these
 * on its own, but opens the target file quietly inside the library, which
 * leaves the caller with neither the file's path nor its own handle. A
 * reader that wants to open the data file itself, say what is missing when
 * a link dangles, or wait for a file that does not exist yet needs the
 * target as a name rather than an opened object. That is what resolve_link
 * gives: the path is walked one component at a time, soft links are
 * replaced by their target and external links move the walk into the named
 * file, until the remaining components are hard links or the file they
 * live in does not exist.
 *
 * Relative file names are searched the way HDF5 does for a virtual
 * dataset's sources and for external links: each prefix in the matching
 * environment variable (HDF5_VDS_PREFIX or HDF5_EXT_PREFIX, colon
 * separated), then the prefix set on the property list, then the
 * directory of the file holding the link, then the name as given. The
 * first candidate that exists wins. A VDS source name of "." means the
 * file holding the virtual dataset. The same candidate list, built by
 * link_search_prefixes, is applied to every link so the raw-chunk path and
 * HDF5's own H5Dread agree on which file is read.
 *
 * A chain of more than 32 links is treated as a cycle and rejected.
 */

#pragma once

#include "dx2/h5/h5utils.hpp"
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dx2::h5link {

/// Which HDF5 search rule a relative file name follows.
enum class LinkKind : uint8_t { External, VirtualSource };

/**
 * @brief Where a path leads once every link on it has been followed.
 *
 * `file` is the file holding the object, absolute when it exists. When it
 * does not exist, `file` is the first candidate the search would have
 * tried and `object_path` is the part of the path still to be walked
 * inside it. The object itself is not opened or checked; a hard link whose
 * target is missing is reported by H5Dopen2 on the result.
 */
struct ResolvedLink {
  std::filesystem::path file;
  std::string object_path;
  bool file_exists = true;
};

/**
 * @brief Builds the candidate directory list for relative file names.
 *
 * @param kind Selects HDF5_VDS_PREFIX or HDF5_EXT_PREFIX.
 * @param prefix The prefix a caller would set on the property list, if any.
 * @param link_file The file holding the link; its directory is a candidate.
 * @return Prefixes in HDF5's search order. An empty path means the name as
 * given, relative to the working directory.
 */
std::vector<std::filesystem::path>
link_search_prefixes(LinkKind kind,
                     const std::optional<std::filesystem::path> &prefix,
                     const std::filesystem::path &link_file);

/**
 * @brief Finds the file a relative or absolute file name refers to.
 *
 * @param name The file name as stored in the link or VDS mapping.
 * @param link_file The file holding the link, returned for a "." name.
 * @param prefixes From link_search_prefixes.
 * @return The first existing candidate, or if none exists the first
 * candidate, with file_exists telling which. The returned path is absolute
 * whenever the candidate can be made so.
 */
ResolvedLink find_linked_file(std::string_view name,
                              const std::filesystem::path &link_file,
                              std::span<const std::filesystem::path> prefixes);

/**
 * @brief Follows every link on a path and names the object it reaches.
 *
 * @param file An open handle to the file holding the start of the path.
 * @param file_path The path that file was opened from.
 * @param object_path Absolute object path within the file.
 * @param prefixes Candidate directories for relative external link names,
 * from link_search_prefixes with LinkKind::External.
 * @return The file and object path at the end of the chain.
 * @throws std::runtime_error if a link on the path does not exist, cannot
 * be read, or the chain exceeds the hop limit.
 */
ResolvedLink resolve_link(hid_t file, const std::filesystem::path &file_path,
                          std::string_view object_path,
                          std::span<const std::filesystem::path> prefixes);

} // namespace dx2::h5link
