/**
 * @file test_h5link.cxx
 * @brief Tests for following HDF5 soft and external links to a file name.
 *
 * External links name real files, so the fixtures are written to a
 * per-process directory under the system temporary directory and removed
 * when the test binary exits. The link environment variables are left
 * unset; the prefix order is exercised through the explicit prefix.
 */

#include "dx2/h5/h5link.hpp"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <unistd.h>

using namespace dx2::h5link;
namespace fs = std::filesystem;

namespace {

fs::path scratch_dir() {
  static const fs::path dir =
      fs::temp_directory_path() / ("dx2_h5link_" + std::to_string(getpid()));
  fs::create_directories(dir);
  return dir;
}

class ScratchCleanup : public testing::Environment {
public:
  void TearDown() override { fs::remove_all(scratch_dir()); }
};
const auto *const scratch_cleanup =
    testing::AddGlobalTestEnvironment(new ScratchCleanup);

/// Creates a file with an empty scalar dataset at /entry/data/data.
void write_data_file(const fs::path &path) {
  h5utils::H5File file(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group entry(
      H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group data(
      H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Space space(H5Screate(H5S_SCALAR));
  h5utils::H5Dataset dataset(H5Dcreate2(data, "data", H5T_NATIVE_INT, space,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
}

/// The master holds every kind of link the resolver must follow. The data
/// file sits in a subdirectory so that relative names only resolve through
/// an explicit prefix, never through the master's own directory.
struct Fixture {
  fs::path master = scratch_dir() / "master.h5";
  fs::path data_dir = scratch_dir() / "data";
  fs::path data = data_dir / "data_000001.h5";
  h5utils::H5File file;

  Fixture() {
    fs::create_directories(data_dir);
    write_data_file(data);
    file = h5utils::H5File(
        H5Fcreate(master.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group group(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Space space(H5Screate(H5S_SCALAR));
    h5utils::H5Dataset local(H5Dcreate2(group, "local", H5T_NATIVE_INT, space,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5Lcreate_soft("local", group, "soft_relative", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_soft("/entry/data/local", group, "soft_absolute", H5P_DEFAULT,
                   H5P_DEFAULT);
    H5Lcreate_soft("../data/./soft_relative", group, "soft_dotted", H5P_DEFAULT,
                   H5P_DEFAULT);
    H5Lcreate_soft("loop_b", group, "loop_a", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_soft("loop_a", group, "loop_b", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_external(data.c_str(), "/entry/data/data", group,
                       "external_absolute", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_external("data_000001.h5", "/entry/data/data", group,
                       "external_relative", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_external("missing.h5", "/entry/data/data", group,
                       "external_missing", H5P_DEFAULT, H5P_DEFAULT);
    // An external link into a group, followed by more path inside the
    // target file.
    H5Lcreate_external("data_000001.h5", "/entry", group, "external_group",
                       H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_soft("external_relative", group, "soft_to_external", H5P_DEFAULT,
                   H5P_DEFAULT);
  }

  std::vector<fs::path> prefixes() const {
    return link_search_prefixes(LinkKind::External, data_dir, master);
  }
  ResolvedLink resolve(std::string_view path) const {
    return resolve_link(file, master, path, prefixes());
  }
};

} // namespace

// A path of hard links resolves to itself in the starting file.
TEST(H5LinkTest, HardLinksStayInTheStartingFile) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/local");
  EXPECT_EQ(r.file, fs::absolute(f.master)) << "wrong file for a hard path";
  EXPECT_EQ(r.object_path, "/entry/data/local") << "hard path was rewritten";
  EXPECT_TRUE(r.file_exists) << "the master exists";
}

// Soft links are replaced by their target, whether that target is given
// relative to the link's group, absolutely, or with "." and ".." in it.
TEST(H5LinkTest, SoftLinksResolveToTheirTarget) {
  Fixture f;
  for (const char *name : {"soft_relative", "soft_absolute", "soft_dotted"}) {
    const ResolvedLink r = f.resolve(std::string("/entry/data/") + name);
    EXPECT_EQ(r.object_path, "/entry/data/local")
        << name << " did not resolve to the dataset";
    EXPECT_EQ(r.file, fs::absolute(f.master)) << name << " left the master";
  }
}

// Two soft links pointing at each other never reach a hard link; the hop
// limit turns the cycle into an error naming the link.
TEST(H5LinkTest, ASoftLinkCycleIsRejected) {
  Fixture f;
  EXPECT_THROW(f.resolve("/entry/data/loop_a"), std::runtime_error)
      << "a soft link cycle resolved";
}

// A link that does not exist is an error, not a dangling result.
TEST(H5LinkTest, AMissingLinkThrows) {
  Fixture f;
  EXPECT_THROW(f.resolve("/entry/data/absent"), std::runtime_error)
      << "a missing link resolved";
}

// An external link with an absolute file name moves the walk into that
// file and reports the object path there.
TEST(H5LinkTest, AnAbsoluteExternalLinkNamesTheTargetFile) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/external_absolute");
  EXPECT_EQ(r.file, fs::absolute(f.data)) << "wrong target file";
  EXPECT_EQ(r.object_path, "/entry/data/data") << "wrong target object";
  EXPECT_TRUE(r.file_exists) << "the data file exists";
}

// A relative external link name is found through the prefix list; the
// data file is not beside the master, so only the explicit prefix finds it.
TEST(H5LinkTest, ARelativeExternalLinkIsFoundThroughThePrefix) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/external_relative");
  EXPECT_EQ(r.file, fs::absolute(f.data)) << "prefix search missed the file";
  EXPECT_TRUE(r.file_exists) << "the data file exists";

  // Without the prefix the search falls through to the master's directory
  // and reports where the file was expected.
  const std::vector<fs::path> no_prefix =
      link_search_prefixes(LinkKind::External, std::nullopt, f.master);
  const ResolvedLink miss = resolve_link(
      f.file, f.master, "/entry/data/external_relative", no_prefix);
  EXPECT_FALSE(miss.file_exists)
      << "found a file that is not beside the master";
  EXPECT_EQ(miss.file, fs::absolute(f.master).parent_path() / "data_000001.h5")
      << "expected location is not the master's directory";
}

// An external link to a file that does not exist is not an error: the
// result names the expected file and the path left to follow in it.
TEST(H5LinkTest, AMissingExternalFileIsReportedNotThrown) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/external_missing");
  EXPECT_FALSE(r.file_exists) << "a missing file was reported as existing";
  EXPECT_EQ(r.file, fs::absolute(f.data_dir / "missing.h5"))
      << "expected location is not the first prefix";
  EXPECT_EQ(r.object_path, "/entry/data/data") << "remaining path lost";
}

// Path components after an external link continue inside the target file.
TEST(H5LinkTest, ComponentsAfterAnExternalLinkContinueInTheTarget) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/external_group/data/data");
  EXPECT_EQ(r.file, fs::absolute(f.data)) << "wrong file after the link";
  EXPECT_EQ(r.object_path, "/entry/data/data") << "trailing path not joined";
}

// A soft link to an external link follows both.
TEST(H5LinkTest, ASoftLinkToAnExternalLinkFollowsBoth) {
  Fixture f;
  const ResolvedLink r = f.resolve("/entry/data/soft_to_external");
  EXPECT_EQ(r.file, fs::absolute(f.data)) << "chain stopped at the soft link";
  EXPECT_EQ(r.object_path, "/entry/data/data") << "wrong final object";
}

// The search order is the environment variable, the caller's prefix, the
// linking file's directory, then the name as given.
TEST(H5LinkTest, PrefixesFollowTheHdf5SearchOrder) {
  const fs::path master = scratch_dir() / "master.h5";
  const std::vector<fs::path> p = link_search_prefixes(
      LinkKind::VirtualSource, fs::path("/opt/data"), master);
  ASSERT_EQ(p.size(), 3u) << "unexpected prefix count with no environment";
  EXPECT_EQ(p[0], "/opt/data") << "caller prefix is not first";
  EXPECT_EQ(p[1], fs::absolute(master).parent_path())
      << "master directory is not second";
  EXPECT_TRUE(p[2].empty()) << "bare name is not last";
}

// A VDS source name of "." is the file holding the virtual dataset.
TEST(H5LinkTest, ADotSourceNameIsTheLinkingFile) {
  Fixture f;
  const ResolvedLink r = find_linked_file(".", f.master, f.prefixes());
  EXPECT_EQ(r.file, fs::absolute(f.master)) << "\".\" is not the master";
  EXPECT_TRUE(r.file_exists) << "the master exists";
}
