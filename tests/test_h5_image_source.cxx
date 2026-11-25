#include <algorithm>
#include <atomic>
#include <chrono>
#include <dx2/h5/h5utils.hpp>
#include <dx2/image/chunk_decoder.hpp>
#include <dx2/image/h5_image_source.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <vector>

using dx2::ChunkDecoder;
using dx2::Compression;
using dx2::Frame;
using dx2::FrameShape;
using dx2::H5ImageSource;
using dx2::PixelType;
using dx2::RawChunk;

namespace {

constexpr size_t SLOW = 32;
constexpr size_t FAST = 48;
constexpr size_t NUM_FRAMES = 3;

/// A per-process scratch directory; files are removed with it.
std::filesystem::path scratch_dir() {
  static const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("dx2_h5_image_source_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  return dir;
}

// Deterministic pixel values that differ between frames and across the
// frame, small enough for every pixel type under test.
template <typename T> std::vector<T> expected_pixels(size_t frame) {
  std::vector<T> pixels(SLOW * FAST);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<T>((frame * 131 + i * 7) % 251);
  }
  return pixels;
}

struct Fixture {
  std::filesystem::path path;
  std::string dataset = "/entry/data/data";
};

// Writes a {NUM_FRAMES, SLOW, FAST} dataset with the filter declared as a
// writer declares it, chunks encoded by dx2 and stored with
// H5Dwrite_chunk. Only the first written_frames chunks exist. With
// chunked false the dataset is contiguous and written through H5Dwrite.
template <typename T>
Fixture write_fixture(const std::string &name, Compression compression,
                      size_t written_frames = NUM_FRAMES, bool chunked = true) {
  Fixture fixture{scratch_dir() / (name + ".h5")};
  h5utils::H5File file(
      H5Fcreate(fixture.path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group entry(
      H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group data(
      H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  const hsize_t dims[3] = {NUM_FRAMES, SLOW, FAST};
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  if (chunked) {
    const hsize_t chunk[3] = {1, SLOW, FAST};
    H5Pset_chunk(dcpl, 3, chunk);
    if (compression == Compression::BitshuffleLZ4) {
      const unsigned user_values[2] = {0, 2};
      H5Pset_filter(dcpl, dx2::bitshuffle_filter_id, H5Z_FLAG_OPTIONAL, 2,
                    user_values);
    }
  }
  h5utils::H5Space space(H5Screate_simple(3, dims, nullptr));
  h5utils::H5Type type = h5dispatch::H5TypeTraits<T>::make();
  h5utils::H5Dataset dataset(
      H5Dcreate2(data, "data", type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT));

  for (size_t frame = 0; frame < written_frames; ++frame) {
    const auto pixels = expected_pixels<T>(frame);
    const hsize_t offset[3] = {frame, 0, 0};
    if (chunked) {
      const auto chunk = dx2::encode_chunk(std::as_bytes(std::span(pixels)),
                                           compression, sizeof(T));
      H5Dwrite_chunk(dataset, H5P_DEFAULT, 0, offset, chunk.size(),
                     chunk.data());
    } else {
      const hsize_t count[3] = {1, SLOW, FAST};
      h5utils::H5Space file_space(H5Dget_space(dataset));
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr, count,
                          nullptr);
      const hsize_t mem_dims[2] = {SLOW, FAST};
      h5utils::H5Space mem_space(H5Screate_simple(2, mem_dims, nullptr));
      H5Dwrite(dataset, type, mem_space, file_space, H5P_DEFAULT,
               pixels.data());
    }
  }
  return fixture;
}

template <typename T> std::span<const T> as_pixels(const Frame &frame) {
  return frame.view().as<T>();
}

template <typename T> std::vector<T> to_vector(std::span<const T> s) {
  return {s.begin(), s.end()};
}

} // namespace

class H5ImageSourceEnvironment : public testing::Environment {
public:
  void TearDown() override { std::filesystem::remove_all(scratch_dir()); }
};

const auto *const environment =
    testing::AddGlobalTestEnvironment(new H5ImageSourceEnvironment);

template <typename T> class H5ImageSourceTypedTest : public testing::Test {};
using TestedPixelTypes = testing::Types<uint16_t, uint32_t>;
TYPED_TEST_SUITE(H5ImageSourceTypedTest, TestedPixelTypes);

// Opening reports the dataset's extent, element type and pipeline, and
// recognises the one-chunk-per-frame layout as raw-chunk capable.
TYPED_TEST(H5ImageSourceTypedTest, OpenReportsShapeTypeAndCompression) {
  const auto fixture = write_fixture<TypeParam>(
      "open_" +
          std::string(dx2::pixel_type_name(dx2::pixel_type_of_v<TypeParam>)),
      Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  EXPECT_EQ(source.num_frames(), NUM_FRAMES) << "Frame count differs";
  EXPECT_EQ(source.shape(), (FrameShape{SLOW, FAST})) << "Shape differs";
  EXPECT_EQ(source.pixel_type(), dx2::pixel_type_of_v<TypeParam>)
      << "Pixel type differs";
  EXPECT_EQ(source.compression(), Compression::BitshuffleLZ4)
      << "Compression differs";
  EXPECT_TRUE(source.supports_raw_chunks())
      << "One chunk per frame with a codec must offer raw chunks";
  EXPECT_EQ(source.frame_bytes(), SLOW * FAST * sizeof(TypeParam))
      << "frame_bytes differs";
  EXPECT_GE(source.max_raw_chunk_bytes(), source.frame_bytes())
      << "The raw bound must cover an incompressible frame";
}

// Every frame decodes to the pixels written, for both element sizes; the
// legacy reader could only ever get one of these right per build.
TYPED_TEST(H5ImageSourceTypedTest, ReadFrameMatchesTheWrittenPixels) {
  const auto fixture = write_fixture<TypeParam>(
      "read_" +
          std::string(dx2::pixel_type_name(dx2::pixel_type_of_v<TypeParam>)),
      Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  for (size_t frame = 0; frame < NUM_FRAMES; ++frame) {
    Frame decoded = source.read_frame(frame);
    EXPECT_EQ(to_vector(as_pixels<TypeParam>(decoded)),
              expected_pixels<TypeParam>(frame))
        << "Frame " << frame << " differs from what was written";
  }
}

// A raw chunk read into a caller's buffer carries the filter header and
// decodes, through a decoder built from the source's pipeline, to the
// same pixels as read_frame. This is the GPU path.
TEST(H5ImageSourceTest, RawChunkDecodesToTheSameFrame) {
  const auto fixture =
      write_fixture<uint16_t>("raw", Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
  RawChunk raw = source.read_raw_chunk(1, buffer);
  EXPECT_EQ(raw.info.frame, 1u) << "ChunkInfo names the wrong frame";
  EXPECT_EQ(raw.info.filter_mask, 0u) << "A filtered chunk has mask 0";
  EXPECT_EQ(raw.bytes.size(), raw.info.stored_bytes)
      << "bytes is not the stored chunk";
  EXPECT_EQ(raw.header_bytes, 12u) << "bitshuffle+lz4 chunks have a header";
  ASSERT_TRUE(raw.container_header()) << "No container header parsed";
  EXPECT_EQ(raw.container_header()->uncompressed_bytes, source.frame_bytes())
      << "Header does not record the frame size";
  EXPECT_LT(raw.info.stored_bytes, source.frame_bytes())
      << "Smooth test data did not compress";

  ChunkDecoder decoder(source.filters(), source.pixel_type(), SLOW * FAST);
  std::vector<uint16_t> decoded(SLOW * FAST);
  decoder.decode(raw.bytes, raw.info.filter_mask,
                 std::as_writable_bytes(std::span(decoded)));
  EXPECT_EQ(decoded, expected_pixels<uint16_t>(1))
      << "Raw chunk did not decode to the written pixels";
}

// HDF5 reading the same file through the filter plugin agrees with the
// dx2 decoder, so the fixture is a file any NeXus consumer could read.
TEST(H5ImageSourceTest, HdfReadThroughThePluginAgreesWithTheDecoder) {
  if (H5Zfilter_avail(dx2::bitshuffle_filter_id) <= 0) {
    GTEST_SKIP() << "bitshuffle plugin not loadable";
  }
  const auto fixture =
      write_fixture<uint16_t>("plugin", Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  Frame decoded = source.read_frame(2);

  h5utils::H5File file(
      H5Fopen(fixture.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  h5utils::H5Dataset dataset(
      H5Dopen2(file, fixture.dataset.c_str(), H5P_DEFAULT));
  std::vector<uint16_t> all(NUM_FRAMES * SLOW * FAST);
  ASSERT_GE(H5Dread(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    all.data()),
            0)
      << "H5Dread through the plugin failed";
  std::vector<uint16_t> frame2(all.begin() + 2 * SLOW * FAST, all.end());
  EXPECT_EQ(to_vector(as_pixels<uint16_t>(decoded)), frame2)
      << "HDF5 and dx2 disagree on frame 2";
}

// Chunks that were never written have no ChunkInfo, as do frames beyond
// the extent, and reading them is an error rather than fill values.
TEST(H5ImageSourceTest, UnwrittenFrameHasNoChunkInfo) {
  const auto fixture =
      write_fixture<uint16_t>("partial", Compression::BitshuffleLZ4, 2);
  H5ImageSource source(fixture.path);
  EXPECT_TRUE(source.chunk_info(1)) << "Written frame 1 has no chunk";
  EXPECT_FALSE(source.chunk_info(2)) << "Unwritten frame 2 has a chunk";
  EXPECT_FALSE(source.chunk_info(NUM_FRAMES))
      << "A frame beyond the extent has a chunk";
  std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
  EXPECT_THROW(source.read_raw_chunk(2, buffer), std::runtime_error)
      << "Reading an unwritten chunk did not throw";
  EXPECT_THROW(source.read_frame(2), std::runtime_error)
      << "Decoding an unwritten frame did not throw";
}

// wait_for_frame returns at once for a written frame and gives up at the
// deadline for one that never arrives.
TEST(H5ImageSourceTest, WaitForFrameHonoursTheDeadline) {
  using namespace std::chrono;
  const auto fixture =
      write_fixture<uint16_t>("wait", Compression::BitshuffleLZ4, 1);
  H5ImageSource source(fixture.path);
  EXPECT_TRUE(source.wait_for_frame(0, 1s)) << "Written frame was waited on";

  const auto start = steady_clock::now();
  EXPECT_FALSE(source.wait_for_frame(1, 60ms, {}, 10ms))
      << "Unwritten frame was reported readable";
  const auto waited = steady_clock::now() - start;
  EXPECT_GE(waited, 60ms) << "Gave up before the deadline";
  EXPECT_LT(waited, 1s) << "Waited far past the deadline";

  std::stop_source stop;
  stop.request_stop();
  EXPECT_FALSE(source.wait_for_frame(1, 10s, stop.get_token()))
      << "A requested stop did not end the wait";
}

// A view of the wrong type or shape is refused before any read.
TEST(H5ImageSourceTest, MismatchedViewIsRefused) {
  const auto fixture =
      write_fixture<uint16_t>("view", Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  Frame wrong_type(PixelType::UInt32, FrameShape{SLOW, FAST});
  EXPECT_THROW(source.read_frame_into(0, wrong_type.view()), std::runtime_error)
      << "A uint32 view was accepted for uint16 data";
  Frame wrong_shape(PixelType::UInt16, FrameShape{FAST, SLOW});
  EXPECT_THROW(source.read_frame_into(0, wrong_shape.view()),
               std::runtime_error)
      << "A transposed view was accepted";
}

// An unfiltered chunked dataset still offers raw chunks, with no header,
// and decodes by copy.
TEST(H5ImageSourceTest, UnfilteredChunksAreOfferedRawWithoutAHeader) {
  const auto fixture = write_fixture<uint16_t>("plain", Compression::None);
  H5ImageSource source(fixture.path);
  EXPECT_EQ(source.compression(), Compression::None) << "Compression differs";
  EXPECT_TRUE(source.supports_raw_chunks()) << "Unfiltered chunks are raw";
  std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
  RawChunk raw = source.read_raw_chunk(0, buffer);
  EXPECT_EQ(raw.header_bytes, 0u) << "Unfiltered chunks have no header";
  EXPECT_EQ(raw.info.stored_bytes, source.frame_bytes())
      << "Unfiltered chunk is not frame sized";
  EXPECT_EQ(to_vector(as_pixels<uint16_t>(source.read_frame(0))),
            expected_pixels<uint16_t>(0))
      << "Plain frame differs";
}

// A contiguous dataset has no chunks, so frames come through H5Dread.
TEST(H5ImageSourceTest, ContiguousDatasetFallsBackToHyperslabReads) {
  const auto fixture = write_fixture<uint16_t>("contiguous", Compression::None,
                                               NUM_FRAMES, false);
  H5ImageSource source(fixture.path);
  EXPECT_FALSE(source.supports_raw_chunks())
      << "A contiguous dataset cannot offer raw chunks";
  EXPECT_FALSE(source.chunk_info(0)) << "A contiguous dataset has no chunks";
  std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
  EXPECT_THROW(source.read_raw_chunk(0, buffer), std::runtime_error)
      << "read_raw_chunk on a contiguous dataset did not throw";
  for (size_t frame = 0; frame < NUM_FRAMES; ++frame) {
    EXPECT_EQ(to_vector(as_pixels<uint16_t>(source.read_frame(frame))),
              expected_pixels<uint16_t>(frame))
        << "Frame " << frame << " differs through the hyperslab path";
  }
  EXPECT_THROW(source.read_frame(NUM_FRAMES), std::runtime_error)
      << "A frame beyond the extent did not throw";
}

// Many threads reading every frame of one source at once all see the
// written pixels; the HDF5 lock is the only serialisation.
TEST(H5ImageSourceTest, ConcurrentReadersAgree) {
  const auto fixture =
      write_fixture<uint16_t>("threads", Compression::BitshuffleLZ4);
  H5ImageSource source(fixture.path);
  constexpr int NUM_THREADS = 8;
  constexpr int ROUNDS = 20;
  std::atomic<int> mismatches{0};
  std::vector<std::jthread> threads;
  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t] {
      std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
      for (int round = 0; round < ROUNDS; ++round) {
        const size_t frame = (t + round) % NUM_FRAMES;
        if (to_vector(as_pixels<uint16_t>(source.read_frame(frame))) !=
            expected_pixels<uint16_t>(frame)) {
          ++mismatches;
        }
        RawChunk raw = source.read_raw_chunk(frame, buffer);
        if (raw.info.frame != frame) {
          ++mismatches;
        }
      }
    });
  }
  threads.clear();
  EXPECT_EQ(mismatches.load(), 0) << "Concurrent reads returned wrong data";
}

// Errors name what was missing.
TEST(H5ImageSourceTest, MissingFileOrDatasetThrowsWithTheName) {
  const auto fixture =
      write_fixture<uint16_t>("missing", Compression::BitshuffleLZ4);
  h5utils::H5ErrorSilencer quiet;
  try {
    H5ImageSource source(fixture.path, {.dataset_path = "/entry/nothing"});
    FAIL() << "A missing dataset did not throw";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("/entry/nothing"), std::string::npos)
        << "Message does not name the dataset: " << e.what();
  }
  EXPECT_THROW(H5ImageSource(scratch_dir() / "absent.h5"), std::runtime_error)
      << "A missing file did not throw";
}

// ---------------------------------------------------------------------
// Layouts: virtual datasets, "." sources, numbered external links.
// ---------------------------------------------------------------------

namespace {

using Pixel = uint16_t;

/// Creates a {frames, SLOW, fast} uint16 dataset chunked one per frame.
h5utils::H5Dataset create_frames(hid_t group, const std::string &name,
                                 size_t frames, Compression compression,
                                 size_t fast = FAST) {
  const hsize_t dims[3] = {frames, SLOW, fast};
  const hsize_t chunk[3] = {1, SLOW, fast};
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  H5Pset_chunk(dcpl, 3, chunk);
  if (compression == Compression::BitshuffleLZ4) {
    const unsigned user_values[2] = {0, 2};
    H5Pset_filter(dcpl, dx2::bitshuffle_filter_id, H5Z_FLAG_OPTIONAL, 2,
                  user_values);
  }
  h5utils::H5Space space(H5Screate_simple(3, dims, nullptr));
  h5utils::H5Type type = h5dispatch::H5TypeTraits<Pixel>::make();
  return h5utils::H5Dataset(H5Dcreate2(group, name.c_str(), type, space,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT));
}

/// Stores expected_pixels(pattern) as chunk `local` of a dataset.
void write_frame(hid_t dataset, size_t local, size_t pattern,
                 Compression compression) {
  const auto pixels = expected_pixels<Pixel>(pattern);
  const hsize_t offset[3] = {local, 0, 0};
  const auto chunk = dx2::encode_chunk(std::as_bytes(std::span(pixels)),
                                       compression, sizeof(Pixel));
  H5Dwrite_chunk(dataset, H5P_DEFAULT, 0, offset, chunk.size(), chunk.data());
}

/// Writes a data file whose /entry/data/data holds `patterns.size()`
/// frames, frame i showing expected_pixels(patterns[i]).
std::filesystem::path write_data_file(const std::filesystem::path &path,
                                      const std::vector<size_t> &patterns,
                                      Compression compression) {
  std::filesystem::create_directories(path.parent_path());
  h5utils::H5File file(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group entry(
      H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group data(
      H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Dataset dataset =
      create_frames(data, "data", patterns.size(), compression);
  for (size_t i = 0; i < patterns.size(); ++i) {
    write_frame(dataset, i, patterns[i], compression);
  }
  return path;
}

/// One virtual mapping: `count` whole frames of `dataset` in `file`
/// starting at source frame `source_first`, placed at virtual frame
/// `virtual_first`. With `unlimited` the run is open-ended on both sides.
struct Mapping {
  std::string file;
  std::string dataset;
  hsize_t virtual_first;
  hsize_t source_first;
  hsize_t count;
  bool unlimited = false;
};

/// Creates /entry/data/data in `file` as a virtual dataset of
/// `virtual_frames` frames, SLOW by `fast`, from the mappings. The source
/// selections use the same `fast` extent, so a value below FAST selects
/// the left part of each source frame.
void add_vds(hid_t file, const std::vector<Mapping> &mappings,
             hsize_t virtual_frames, size_t fast = FAST) {
  h5utils::H5Group entry;
  {
    h5utils::H5ErrorSilencer quiet;
    entry = h5utils::H5Group(H5Gopen2(file, "entry", H5P_DEFAULT));
  }
  if (!entry) {
    entry = h5utils::H5Group(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  }
  h5utils::H5Group data;
  {
    h5utils::H5ErrorSilencer quiet;
    data = h5utils::H5Group(H5Gopen2(entry, "data", H5P_DEFAULT));
  }
  if (!data) {
    data = h5utils::H5Group(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  }

  const bool unlimited =
      std::any_of(mappings.begin(), mappings.end(),
                  [](const Mapping &m) { return m.unlimited; });
  const hsize_t vdims[3] = {unlimited ? 0 : virtual_frames, SLOW, fast};
  const hsize_t vmax[3] = {unlimited ? H5S_UNLIMITED : virtual_frames, SLOW,
                           fast};
  h5utils::H5Space vspace(H5Screate_simple(3, vdims, vmax));
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  for (const Mapping &m : mappings) {
    const hsize_t stride[3] = {1, 1, 1};
    const hsize_t count[3] = {m.unlimited ? H5S_UNLIMITED : m.count, 1, 1};
    const hsize_t block[3] = {m.unlimited ? 1 : 1, SLOW, fast};
    // Virtual side: count frames of one block each at virtual_first.
    const hsize_t vstart[3] = {m.virtual_first, 0, 0};
    H5Sselect_hyperslab(vspace, H5S_SELECT_SET, vstart, stride, count, block);
    // Source side: the same run at source_first in a source space whose
    // frame extent is open, so the mapping does not fix the source size.
    const hsize_t sdims[3] = {0, SLOW, fast};
    const hsize_t smax[3] = {H5S_UNLIMITED, SLOW, fast};
    h5utils::H5Space sspace(H5Screate_simple(3, sdims, smax));
    const hsize_t sstart[3] = {m.source_first, 0, 0};
    H5Sselect_hyperslab(sspace, H5S_SELECT_SET, sstart, stride, count, block);
    ASSERT_GE(
        H5Pset_virtual(dcpl, vspace, m.file.c_str(), m.dataset.c_str(), sspace),
        0)
        << "H5Pset_virtual failed for " << m.file << ":" << m.dataset;
  }
  h5utils::H5Type type = h5dispatch::H5TypeTraits<Pixel>::make();
  h5utils::H5Dataset vds(
      H5Dcreate2(data, "data", type, vspace, H5P_DEFAULT, dcpl, H5P_DEFAULT));
  ASSERT_TRUE(vds) << "could not create the virtual dataset";
}

h5utils::H5File create_master(const std::filesystem::path &path) {
  return h5utils::H5File(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
}

/// Reads every frame and compares it with its pattern.
void expect_frames(H5ImageSource &source, const std::vector<size_t> &patterns,
                   const char *what) {
  ASSERT_EQ(source.num_frames(), patterns.size())
      << what << ": frame count differs";
  for (size_t v = 0; v < patterns.size(); ++v) {
    const Frame frame = source.read_frame(v);
    EXPECT_EQ(to_vector(as_pixels<Pixel>(frame)),
              expected_pixels<Pixel>(patterns[v]))
        << what << ": frame " << v << " differs";
  }
}

} // namespace

// A virtual dataset over two files, the second mapped from a nonzero
// source offset, flattens to two sources and reads through the raw path.
TEST(H5ImageLayoutTest, VirtualDatasetOverTwoFilesReadsEveryFrame) {
  const auto dir = scratch_dir() / "vds_two";
  write_data_file(dir / "data_a.h5", {0, 1, 2}, Compression::BitshuffleLZ4);
  write_data_file(dir / "data_b.h5", {100, 101, 102, 103, 3, 4, 5},
                  Compression::BitshuffleLZ4);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    // Relative names, found through the master's directory.
    add_vds(file,
            {{"data_a.h5", "/entry/data/data", 0, 0, 3},
             {"data_b.h5", "/entry/data/data", 3, 4, 3}},
            6);
  }
  H5ImageSource source(master);
  const auto sources = source.sources();
  ASSERT_EQ(sources.size(), 2u) << "expected two sources";
  EXPECT_EQ(sources[0].file, dir / "data_a.h5") << "first source differs";
  EXPECT_EQ(sources[1].num_frames, 7u) << "second source extent differs";
  EXPECT_TRUE(source.supports_raw_chunks()) << "raw path should be offered";
  EXPECT_EQ(source.compression(), Compression::BitshuffleLZ4)
      << "compression differs";
  expect_frames(source, {0, 1, 2, 3, 4, 5}, "two-file VDS");

  // The raw chunk of a frame in the second file decodes to that frame
  // and reports the virtual index.
  std::vector<std::byte> buffer(source.max_raw_chunk_bytes());
  const RawChunk raw = source.read_raw_chunk(4, buffer);
  EXPECT_EQ(raw.info.frame, 4u) << "raw chunk reports the source index";
  ChunkDecoder decoder(source.filters(), source.pixel_type(), SLOW * FAST);
  std::vector<Pixel> pixels(SLOW * FAST);
  decoder.decode(raw.bytes, raw.info.filter_mask,
                 std::as_writable_bytes(std::span(pixels)));
  EXPECT_EQ(pixels, expected_pixels<Pixel>(4)) << "raw chunk 4 differs";
}

// A "." source is the master itself; its dataset name may be a local
// dataset or an external link the legacy reader special-cased.
TEST(H5ImageLayoutTest, DotSourcesResolveInsideTheMaster) {
  const auto dir = scratch_dir() / "vds_dot";
  const auto external =
      write_data_file(dir / "data_000002.h5", {3, 4, 5}, Compression::None);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Dataset local =
        create_frames(data, "local", 3, Compression::None);
    for (size_t i = 0; i < 3; ++i) {
      write_frame(local, i, i, Compression::None);
    }
    H5Lcreate_external(external.c_str(), "/entry/data/data", data,
                       "data_000002", H5P_DEFAULT, H5P_DEFAULT);
    add_vds(file,
            {{".", "/entry/data/local", 0, 0, 3},
             {".", "/entry/data/data_000002", 3, 0, 3}},
            6);
  }
  H5ImageSource source(master);
  const auto sources = source.sources();
  ASSERT_EQ(sources.size(), 2u) << "expected two sources";
  EXPECT_EQ(sources[0].file, master) << "local source is not the master";
  EXPECT_EQ(sources[0].dataset, "/entry/data/local") << "local dataset name";
  EXPECT_EQ(sources[1].file, external) << "external link was not followed";
  expect_frames(source, {0, 1, 2, 3, 4, 5}, "dot VDS");
}

// A mapping with an unlimited frame count takes every frame the source
// holds at open.
TEST(H5ImageLayoutTest, UnlimitedMappingTakesTheSourceExtent) {
  const auto dir = scratch_dir() / "vds_unlimited";
  write_data_file(dir / "data.h5", {0, 1, 2, 3}, Compression::BitshuffleLZ4);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    add_vds(file, {{"data.h5", "/entry/data/data", 0, 0, 0, true}}, 0);
  }
  H5ImageSource source(master);
  expect_frames(source, {0, 1, 2, 3}, "unlimited VDS");
}

// Without /entry/data/data the numbered siblings are concatenated in name
// order, with relative file names found only through the prefix option.
TEST(H5ImageLayoutTest, NumberedExternalLinksConcatenateInNameOrder) {
  const auto dir = scratch_dir() / "numbered";
  write_data_file(dir / "files" / "data_000001.h5", {0, 1, 2},
                  Compression::BitshuffleLZ4);
  write_data_file(dir / "files" / "data_000002.h5", {3, 4, 5},
                  Compression::BitshuffleLZ4);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    // Created out of order so that name order, not creation order, wins.
    H5Lcreate_external("data_000002.h5", "/entry/data/data", data,
                       "data_000002", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_external("data_000001.h5", "/entry/data/data", data,
                       "data_000001", H5P_DEFAULT, H5P_DEFAULT);
    // A sibling that does not match the pattern is ignored.
    H5Lcreate_external("unrelated.h5", "/x", data, "mask", H5P_DEFAULT,
                       H5P_DEFAULT);
  }
  H5ImageSource::Options options;
  options.vds_prefix = dir / "files";
  H5ImageSource source(master, options);
  const auto sources = source.sources();
  ASSERT_EQ(sources.size(), 2u) << "expected two numbered sources";
  EXPECT_EQ(sources[0].file, dir / "files" / "data_000001.h5")
      << "name order not respected";
  EXPECT_TRUE(source.supports_raw_chunks()) << "raw path should be offered";
  expect_frames(source, {0, 1, 2, 3, 4, 5}, "numbered datasets");

  // Without the prefix the files are not beside the master, and the
  // error names where the first was looked for.
  try {
    H5ImageSource unresolved(master);
    FAIL() << "opened with unresolvable relative links";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("data_000001.h5"), std::string::npos)
        << "error does not name the missing file: " << e.what();
  }
}

// A mapping that takes part of each frame cannot be flattened; the
// virtual dataset is read through HDF5 instead, with no raw path.
TEST(H5ImageLayoutTest, SubFrameMappingFallsBackToReadingTheVds) {
  const auto dir = scratch_dir() / "vds_half";
  write_data_file(dir / "data.h5", {0, 1, 2}, Compression::None);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    add_vds(file, {{"data.h5", "/entry/data/data", 0, 0, 3}}, 3, FAST / 2);
  }
  H5ImageSource source(master);
  EXPECT_FALSE(source.supports_raw_chunks())
      << "a sub-frame VDS has no raw path";
  ASSERT_EQ(source.sources().size(), 1u) << "the VDS itself is the one source";
  EXPECT_EQ(source.sources()[0].file, master) << "source is not the VDS";
  EXPECT_EQ(source.shape(), (FrameShape{SLOW, FAST / 2})) << "shape differs";
  const Frame frame = source.read_frame(1);
  const auto full = expected_pixels<Pixel>(1);
  const auto half = as_pixels<Pixel>(frame);
  for (size_t row = 0; row < SLOW; ++row) {
    for (size_t col = 0; col < FAST / 2; ++col) {
      ASSERT_EQ(half[row * (FAST / 2) + col], full[row * FAST + col])
          << "pixel (" << row << ", " << col << ") differs";
    }
  }
}

// An external link to a file that is not there is an error naming the
// file looked for, not a quiet empty source.
TEST(H5ImageLayoutTest, MissingExternalFileIsNamedInTheError) {
  const auto dir = scratch_dir() / "missing";
  std::filesystem::create_directories(dir);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5Lcreate_external("nowhere.h5", "/entry/data/data", data, "data",
                       H5P_DEFAULT, H5P_DEFAULT);
  }
  try {
    H5ImageSource source(master);
    FAIL() << "opened through a dangling external link";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("nowhere.h5"), std::string::npos)
        << "error does not name the missing file: " << e.what();
  }
}

// Sources must agree on shape; a mismatched one is refused by name.
TEST(H5ImageLayoutTest, SourcesWithDifferentShapesAreRefused) {
  const auto dir = scratch_dir() / "vds_mismatch";
  write_data_file(dir / "data_a.h5", {0, 1}, Compression::None);
  {
    h5utils::H5File file(H5Fcreate((dir / "data_b.h5").c_str(), H5F_ACC_TRUNC,
                                   H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    create_frames(data, "data", 2, Compression::None, FAST / 2);
  }
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5Lcreate_external("data_000001.h5", "/entry/data/data", data,
                       "data_000001", H5P_DEFAULT, H5P_DEFAULT);
    H5Lcreate_external("data_000002.h5", "/entry/data/data", data,
                       "data_000002", H5P_DEFAULT, H5P_DEFAULT);
  }
  std::filesystem::rename(dir / "data_a.h5", dir / "data_000001.h5");
  std::filesystem::rename(dir / "data_b.h5", dir / "data_000002.h5");
  EXPECT_THROW(H5ImageSource source(master), std::runtime_error)
      << "sources of different shape were accepted";
}

namespace {

/// A file access list for the latest file format, which SWMR writing
/// requires.
h5utils::H5Plist latest_fapl() {
  h5utils::H5Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
  H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
  return fapl;
}

/// Creates a bitshuffle-LZ4 /entry/data/data of no frames yet with an
/// unlimited frame extent, in a file a SWMR writer can open.
void create_growing_file(const std::filesystem::path &path) {
  std::filesystem::create_directories(path.parent_path());
  h5utils::H5File file(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, latest_fapl()));
  h5utils::H5Group entry(
      H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group data(
      H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  const hsize_t dims[3] = {0, SLOW, FAST};
  const hsize_t max_dims[3] = {H5S_UNLIMITED, SLOW, FAST};
  const hsize_t chunk[3] = {1, SLOW, FAST};
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  H5Pset_chunk(dcpl, 3, chunk);
  const unsigned user_values[2] = {0, 2};
  H5Pset_filter(dcpl, dx2::bitshuffle_filter_id, H5Z_FLAG_OPTIONAL, 2,
                user_values);
  h5utils::H5Space space(H5Screate_simple(3, dims, max_dims));
  h5utils::H5Type type = h5dispatch::H5TypeTraits<Pixel>::make();
  h5utils::H5Dataset dataset(
      H5Dcreate2(data, "data", type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT));
  ASSERT_TRUE(dataset) << "could not create the growing dataset";
}

} // namespace

// A SWMR writer extends the dataset one frame at a time and flushes while
// four readers each wait for the next unclaimed frame. Every frame must be
// delivered with its own pattern, the count must follow the extent, and a
// frame past the end must still time out. The writer shares the readers'
// process, so this exercises the refresh path rather than cross-process
// consistency, and the writer must hold the file before a SWMR reader
// opens it.
TEST(H5ImageSwmrTest, GrowingDatasetIsFollowedByWaitingReaders) {
  using namespace std::chrono_literals;
  constexpr size_t FRAMES = 6;
  constexpr size_t READERS = 4;
  const auto path = scratch_dir() / "swmr_grow" / "data.h5";
  create_growing_file(path);

  // The writer opens first and waits for the go signal, so the reader can
  // observe the empty dataset before anything is written.
  std::atomic<bool> writer_open{false};
  std::atomic<bool> go{false};
  std::thread writer([&] {
    h5utils::H5File file(H5Fopen(
        path.c_str(), H5F_ACC_RDWR | H5F_ACC_SWMR_WRITE, latest_fapl()));
    h5utils::H5Dataset dataset(H5Dopen2(file, "/entry/data/data", H5P_DEFAULT));
    writer_open = true;
    while (!go) {
      std::this_thread::sleep_for(1ms);
    }
    for (size_t k = 0; k < FRAMES; ++k) {
      const hsize_t extent[3] = {k + 1, SLOW, FAST};
      H5Dset_extent(dataset, extent);
      write_frame(dataset, k, k, Compression::BitshuffleLZ4);
      H5Dflush(dataset);
      std::this_thread::sleep_for(30ms);
    }
  });
  while (!writer_open) {
    std::this_thread::sleep_for(1ms);
  }

  H5ImageSource::Options options;
  options.swmr = true;
  options.refresh_min_interval = 5ms;
  H5ImageSource source(path, options);
  EXPECT_EQ(source.num_frames(), 0u) << "no frame has been written yet";
  EXPECT_FALSE(source.chunk_info(0)) << "frame 0 reported before it exists";
  go = true;

  // Readers claim frames in turn and wait for each one.
  std::atomic<size_t> next{0};
  std::atomic<size_t> delivered{0};
  std::vector<std::thread> readers;
  for (size_t r = 0; r < READERS; ++r) {
    readers.emplace_back([&] {
      for (size_t f = next++; f < FRAMES; f = next++) {
        ASSERT_TRUE(source.wait_for_frame(f, 10s, {}, 5ms))
            << "frame " << f << " did not arrive";
        const Frame frame = source.read_frame(f);
        EXPECT_EQ(to_vector(as_pixels<Pixel>(frame)), expected_pixels<Pixel>(f))
            << "frame " << f << " differs";
        ++delivered;
      }
    });
  }
  for (std::thread &reader : readers) {
    reader.join();
  }
  writer.join();
  EXPECT_EQ(delivered.load(), FRAMES) << "not every frame was delivered";
  EXPECT_EQ(source.num_frames(), FRAMES) << "count did not follow the extent";
  EXPECT_FALSE(source.wait_for_frame(FRAMES, 50ms, {}, 5ms))
      << "a frame past the end was reported";
}

// A virtual dataset names a second file that is not there when the source
// opens. In SWMR mode its frames report no chunk, quietly, until the file
// appears; without SWMR the open is an error naming the file.
TEST(H5ImageSwmrTest, AwaitedSourceFileGivesNoChunkUntilItAppears) {
  const auto dir = scratch_dir() / "swmr_await";
  write_data_file(dir / "data_a.h5", {0, 1}, Compression::BitshuffleLZ4);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    add_vds(file,
            {{"data_a.h5", "/entry/data/data", 0, 0, 2},
             {"data_b.h5", "/entry/data/data", 2, 0, 2}},
            4);
  }
  EXPECT_THROW(
      {
        try {
          H5ImageSource plain(master);
        } catch (const std::runtime_error &e) {
          EXPECT_NE(std::string(e.what()).find("data_b.h5"), std::string::npos)
              << "error does not name the file: " << e.what();
          throw;
        }
      },
      std::runtime_error)
      << "a missing source without SWMR must be an error";

  H5ImageSource::Options options;
  options.swmr = true;
  H5ImageSource source(master, options);
  ASSERT_EQ(source.sources().size(), 2u) << "both sources should be listed";
  EXPECT_FALSE(source.sources()[1].open) << "the missing file counts as open";
  EXPECT_EQ(source.num_frames(), 4u) << "a fixed mapping keeps its count";
  EXPECT_TRUE(source.chunk_info(1)) << "frame 1 is in the file that exists";
  EXPECT_FALSE(source.chunk_info(2)) << "frame 2 reported before its file";
  EXPECT_THROW(source.read_frame(2), std::runtime_error)
      << "reading an awaited frame must throw";

  write_data_file(dir / "data_b.h5", {2, 3}, Compression::BitshuffleLZ4);
  EXPECT_TRUE(source.chunk_info(2))
      << "frame 2 missing after its file appeared";
  EXPECT_TRUE(source.sources()[1].open) << "the late file was not opened";
  expect_frames(source, {0, 1, 2, 3}, "after the second file appeared");
}

// A source that appears late with a different image size cannot be
// checked at open, so the request that opens it throws instead.
TEST(H5ImageSwmrTest, LateSourceWithADifferentShapeIsRefusedWhenItAppears) {
  const auto dir = scratch_dir() / "swmr_late_shape";
  write_data_file(dir / "data_a.h5", {0}, Compression::BitshuffleLZ4);
  const auto master = dir / "master.h5";
  {
    h5utils::H5File file = create_master(master);
    add_vds(file,
            {{"data_a.h5", "/entry/data/data", 0, 0, 1},
             {"data_b.h5", "/entry/data/data", 1, 0, 1}},
            2);
  }
  H5ImageSource::Options options;
  options.swmr = true;
  H5ImageSource source(master, options);
  EXPECT_FALSE(source.chunk_info(1)) << "frame 1 reported before its file";

  // The late file is half as wide as the first.
  {
    std::filesystem::create_directories(dir);
    h5utils::H5File file(H5Fcreate((dir / "data_b.h5").c_str(), H5F_ACC_TRUNC,
                                   H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group entry(
        H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Group data(
        H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5utils::H5Dataset narrow =
        create_frames(data, "data", 1, Compression::BitshuffleLZ4, FAST / 2);
  }
  EXPECT_THROW(source.chunk_info(1), std::runtime_error)
      << "a late source of another shape must be refused";
}
