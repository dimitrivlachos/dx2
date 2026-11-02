#include <dx2/image/pixel_type.hpp>
#include <gtest/gtest.h>
#include <hdf5.h>

namespace {

template <typename List> struct to_gtest_types;
template <typename... Ts> struct to_gtest_types<h5dispatch::type_list<Ts...>> {
  using type = ::testing::Types<Ts...>;
};

/// An HDF5 file held in memory by the core driver, never written to disk.
struct InMemoryFile {
  h5utils::H5File file;
  InMemoryFile() {
    h5utils::H5Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_core(fapl, 1 << 20, false);
    file = h5utils::H5File(
        H5Fcreate("in-memory.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
  }
};

/// A 2x3 dataset of the given HDF5 type, named after the test.
h5utils::H5Dataset make_dataset(hid_t file, const char *name, hid_t type) {
  hsize_t dims[2] = {2, 3};
  h5utils::H5Space space(H5Screate_simple(2, dims, nullptr));
  return h5utils::H5Dataset(H5Dcreate2(file, name, type, space, H5P_DEFAULT,
                                       H5P_DEFAULT, H5P_DEFAULT));
}

} // namespace

/*
 * Typed over every entry in dx2::PixelTypes, so a newly added pixel type is
 * covered without editing this file.
 */
template <typename T> class PixelTypeTest : public ::testing::Test {};
TYPED_TEST_SUITE(PixelTypeTest, to_gtest_types<dx2::PixelTypes>::type);

// The enum and the type list are two spellings of one list; a type must map
// to an enum value that maps back to the same type.
TYPED_TEST(PixelTypeTest, EnumAndTypeListRoundTrip) {
  constexpr dx2::PixelType type = dx2::pixel_type_of_v<TypeParam>;
  bool same = dx2::visit_pixel_type(type, [](auto tag) {
    return std::is_same_v<typename decltype(tag)::type, TypeParam>;
  });
  EXPECT_TRUE(same) << "visit_pixel_type gave a different type than "
                       "pixel_type_of_v mapped from";
}

// pixel_size is the only place a byte count is derived from the enum, so it
// must agree with sizeof.
TYPED_TEST(PixelTypeTest, PixelSizeMatchesSizeof) {
  EXPECT_EQ(dx2::pixel_size(dx2::pixel_type_of_v<TypeParam>), sizeof(TypeParam))
      << "pixel_size disagrees with sizeof for this type";
}

// Every type has a name for error messages.
TYPED_TEST(PixelTypeTest, EveryTypeHasAName) {
  EXPECT_FALSE(dx2::pixel_type_name(dx2::pixel_type_of_v<TypeParam>).empty())
      << "pixel_type_name returned an empty name";
}

// A dataset created with the type's own HDF5 datatype reads back as that
// PixelType, which is how a reader learns the type from a file.
TYPED_TEST(PixelTypeTest, PixelTypeFromH5RecognisesTheDataset) {
  InMemoryFile f;
  h5utils::H5Type type = h5dispatch::H5TypeTraits<TypeParam>::make();
  h5utils::H5Dataset dataset = make_dataset(f.file, "pixels", type);
  ASSERT_TRUE(dataset) << "Could not create the in-memory dataset";

  EXPECT_EQ(dx2::pixel_type_from_h5(dataset), dx2::pixel_type_of_v<TypeParam>)
      << "pixel_type_from_h5 returned a different PixelType";
}

// An enum value outside the list, which can only come from a cast of bad
// data, must fail loudly rather than index past the tables.
TEST(PixelTypeTest, OutOfRangeValueThrows) {
  const auto bad = static_cast<dx2::PixelType>(dx2::kNumPixelTypes);
  EXPECT_THROW(dx2::pixel_size(bad), std::invalid_argument)
      << "pixel_size accepted an out-of-range value";
  EXPECT_THROW(dx2::pixel_type_name(bad), std::invalid_argument)
      << "pixel_type_name accepted an out-of-range value";
  EXPECT_THROW(dx2::visit_pixel_type(bad, [](auto) {}), std::invalid_argument)
      << "visit_pixel_type accepted an out-of-range value";
}

// A 64-bit integer dataset is a valid column type but not a pixel type, so
// the pixel dispatch must reject it rather than fall through to a column
// type.
TEST(PixelTypeTest, PixelTypeFromH5RejectsANonPixelType) {
  InMemoryFile f;
  h5utils::H5Type type = h5dispatch::make_h5_type<int64_t>();
  h5utils::H5Dataset dataset = make_dataset(f.file, "wide", type);
  ASSERT_TRUE(dataset) << "Could not create the in-memory dataset";

  EXPECT_THROW(dx2::pixel_type_from_h5(dataset), std::runtime_error)
      << "pixel_type_from_h5 accepted a 64-bit integer dataset";
}
