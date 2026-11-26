#include <algorithm>
#include <chrono>
#include <dx2/h5/h5utils.hpp>
#include <dx2/image/chunk_decoder.hpp>
#include <dx2/image/frame_prefetcher.hpp>
#include <dx2/image/h5_image_source.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <set>
#include <unistd.h>
#include <vector>

using dx2::Compression;
using dx2::FramePrefetcher;
using dx2::H5ImageSource;

namespace {

using Pixel = uint16_t;
constexpr size_t SLOW = 24;
constexpr size_t FAST = 40;
constexpr size_t NUM_FRAMES = 12;

std::filesystem::path scratch_dir() {
  static const auto dir = [] {
    const auto path = std::filesystem::temp_directory_path() /
                      ("dx2_frame_prefetcher_" + std::to_string(getpid()));
    std::filesystem::create_directories(path);
    return path;
  }();
  return dir;
}

/// The pixels of frame `index`: distinct per frame and per pixel.
std::vector<Pixel> expected_pixels(size_t index) {
  std::vector<Pixel> pixels(SLOW * FAST);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<Pixel>(index * 7919 + i);
  }
  return pixels;
}

/// Writes a bitshuffle-LZ4 file of NUM_FRAMES frames, the first `written`
/// of them stored, and returns its path.
std::filesystem::path write_fixture(const std::string &name, size_t written) {
  const auto path = scratch_dir() / (name + ".h5");
  h5utils::H5File file(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group entry(
      H5Gcreate2(file, "entry", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  h5utils::H5Group data(
      H5Gcreate2(entry, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  const hsize_t dims[3] = {NUM_FRAMES, SLOW, FAST};
  const hsize_t chunk[3] = {1, SLOW, FAST};
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  H5Pset_chunk(dcpl, 3, chunk);
  const unsigned user_values[2] = {0, 2};
  H5Pset_filter(dcpl, dx2::bitshuffle_filter_id, H5Z_FLAG_OPTIONAL, 2,
                user_values);
  h5utils::H5Space space(H5Screate_simple(3, dims, nullptr));
  h5utils::H5Type type = h5dispatch::H5TypeTraits<Pixel>::make();
  h5utils::H5Dataset dataset(
      H5Dcreate2(data, "data", type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT));
  for (size_t f = 0; f < written; ++f) {
    const auto pixels = expected_pixels(f);
    const hsize_t offset[3] = {f, 0, 0};
    const auto encoded =
        dx2::encode_chunk(std::as_bytes(std::span(pixels)),
                          Compression::BitshuffleLZ4, sizeof(Pixel));
    H5Dwrite_chunk(dataset, H5P_DEFAULT, 0, offset, encoded.size(),
                   encoded.data());
  }
  return path;
}

std::vector<Pixel> pixels_of(const FramePrefetcher::Item &item) {
  const auto view = item.frame.view().as<Pixel>();
  return {view.begin(), view.end()};
}

} // namespace

// In-order delivery hands out every frame of the range ascending with the
// pixels that were written, whichever worker read it first.
TEST(FramePrefetcherTest, InOrderDeliveryReturnsEveryFrameAscending) {
  H5ImageSource source(write_fixture("in_order", NUM_FRAMES));
  FramePrefetcher::Options options;
  options.workers = 3;
  options.window = 4;
  FramePrefetcher prefetcher(source, 2, NUM_FRAMES, options);
  size_t expected = 2;
  while (const auto item = prefetcher.next()) {
    EXPECT_EQ(item->index, expected) << "frames arrived out of order";
    EXPECT_EQ(item->info.frame, expected) << "chunk info names another frame";
    EXPECT_EQ(pixels_of(*item), expected_pixels(expected))
        << "frame " << expected << " differs";
    ++expected;
  }
  EXPECT_EQ(expected, NUM_FRAMES) << "the range was cut short";
  EXPECT_FALSE(prefetcher.next()) << "a finished run must stay finished";
}

// Any-order delivery still hands out each frame exactly once, each with
// its own pixels.
TEST(FramePrefetcherTest, AnyOrderDeliveryReturnsEachFrameOnce) {
  H5ImageSource source(write_fixture("any_order", NUM_FRAMES));
  FramePrefetcher::Options options;
  options.delivery = FramePrefetcher::Delivery::AnyOrder;
  FramePrefetcher prefetcher(source, 0, NUM_FRAMES, options);
  std::set<size_t> seen;
  while (const auto item = prefetcher.next()) {
    EXPECT_TRUE(seen.insert(item->index).second)
        << "frame " << item->index << " delivered twice";
    EXPECT_EQ(pixels_of(*item), expected_pixels(item->index))
        << "frame " << item->index << " differs";
  }
  EXPECT_EQ(seen.size(), NUM_FRAMES) << "not every frame was delivered";
}

// Raw payload items carry the stored chunk, which decodes to the same
// frame the decoded path would have delivered.
TEST(FramePrefetcherTest, RawPayloadDecodesToTheWrittenFrame) {
  H5ImageSource source(write_fixture("raw", NUM_FRAMES));
  FramePrefetcher::Options options;
  options.payload = FramePrefetcher::Payload::Raw;
  FramePrefetcher prefetcher(source, 0, NUM_FRAMES, options);
  const dx2::ChunkDecoder decoder(source.filters(), source.pixel_type(),
                                  SLOW * FAST);
  size_t count = 0;
  while (const auto item = prefetcher.next()) {
    ASSERT_EQ(item->frame.size_bytes(), item->info.stored_bytes)
        << "the chunk does not span the stored bytes";
    EXPECT_EQ(item->header_bytes, 12u) << "bitshuffle header not reported";
    std::vector<Pixel> pixels(SLOW * FAST);
    decoder.decode(item->frame.bytes(), item->info.filter_mask,
                   std::as_writable_bytes(std::span(pixels)));
    EXPECT_EQ(pixels, expected_pixels(item->index))
        << "frame " << item->index << " differs after decoding";
    ++count;
  }
  EXPECT_EQ(count, NUM_FRAMES) << "not every chunk was delivered";
}

// Buffers given back with recycle are written into again by later frames,
// so every delivery after the pool is primed must still carry its own
// pixels and nothing left from the frame that had the buffer before.
TEST(FramePrefetcherTest, RecycledBuffersCarryTheRightFrame) {
  H5ImageSource source(write_fixture("recycle", NUM_FRAMES));
  FramePrefetcher::Options options;
  options.workers = 2;
  options.window = 2;
  FramePrefetcher prefetcher(source, 0, NUM_FRAMES, options);
  std::set<const std::byte *> buffers;
  size_t count = 0;
  while (auto item = prefetcher.next()) {
    EXPECT_EQ(pixels_of(*item), expected_pixels(item->index))
        << "frame " << item->index << " differs";
    buffers.insert(item->frame.bytes().data());
    prefetcher.recycle(std::move(*item));
    ++count;
  }
  EXPECT_EQ(count, NUM_FRAMES) << "not every frame was delivered";
  EXPECT_LT(buffers.size(), NUM_FRAMES)
      << "no buffer was reused although every item was recycled";
}

// Stopping while workers wait on frames that will never be written makes
// next() report the end and lets the destructor join without waiting out
// the frame timeout.
TEST(FramePrefetcherTest, StopEndsTheRunWhileWorkersWait) {
  using namespace std::chrono;
  H5ImageSource source(write_fixture("stop", 2));
  FramePrefetcher::Options options;
  options.frame_timeout = 60s;
  options.poll = 10ms;
  const auto start = steady_clock::now();
  {
    FramePrefetcher prefetcher(source, 0, NUM_FRAMES, options);
    const auto first = prefetcher.next();
    ASSERT_TRUE(first) << "the first written frame should arrive";
    EXPECT_EQ(first->index, 0u) << "first delivery is not frame 0";
    prefetcher.stop();
    // Frame 1 may or may not be taken; nothing beyond it can be.
    while (const auto item = prefetcher.next()) {
      EXPECT_LT(item->index, 2u) << "an unwritten frame was delivered";
    }
  }
  EXPECT_LT(steady_clock::now() - start, 5s)
      << "stop did not release the waiting workers";
}

// A frame that never arrives surfaces from next() as an exception naming
// it, once; the run then reads as ended.
TEST(FramePrefetcherTest, MissingFrameTimesOutThroughNext) {
  using namespace std::chrono;
  H5ImageSource source(write_fixture("timeout", 3));
  FramePrefetcher::Options options;
  options.workers = 1;
  options.window = 1;
  options.frame_timeout = 100ms;
  options.poll = 10ms;
  FramePrefetcher prefetcher(source, 0, NUM_FRAMES, options);
  for (size_t f = 0; f < 3; ++f) {
    const auto item = prefetcher.next();
    ASSERT_TRUE(item) << "written frame " << f << " did not arrive";
    EXPECT_EQ(item->index, f) << "frames out of order";
  }
  EXPECT_THROW(
      {
        try {
          prefetcher.next();
        } catch (const std::runtime_error &e) {
          EXPECT_NE(std::string(e.what()).find("frame 3"), std::string::npos)
              << "error does not name the frame: " << e.what();
          throw;
        }
      },
      std::runtime_error)
      << "the missing frame must surface as an error";
  EXPECT_FALSE(prefetcher.next()) << "a failed run must read as ended";
}

// A window too small to give every worker a frame is refused, as is a raw
// payload from a source without raw chunks.
TEST(FramePrefetcherTest, ImpossibleOptionsAreRefused) {
  H5ImageSource source(write_fixture("options", NUM_FRAMES));
  FramePrefetcher::Options small;
  small.workers = 4;
  small.window = 2;
  EXPECT_THROW(FramePrefetcher(source, 0, NUM_FRAMES, small),
               std::invalid_argument)
      << "a window smaller than the workers must be refused";
  FramePrefetcher::Options none;
  none.workers = 0;
  EXPECT_THROW(FramePrefetcher(source, 0, NUM_FRAMES, none),
               std::invalid_argument)
      << "zero workers must be refused";
}
