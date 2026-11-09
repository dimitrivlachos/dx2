#include <algorithm>
#include <array>
#include <dx2/h5/h5utils.hpp>
#include <dx2/image/chunk_decoder.hpp>
#include <dx2/image/compression.hpp>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

using dx2::ChunkDecoder;
using dx2::Codec;
using dx2::Compression;
using dx2::FilterPipeline;
using dx2::FilterStage;
using dx2::Lz4ContainerHeader;
using dx2::PixelType;

namespace {

// cd_values as the bitshuffle filter's set_local records them:
// {major, minor, elem_size, block_size, compressor}.
FilterPipeline bitshuffle_pipeline(unsigned compressor, unsigned elem_size = 2,
                                   unsigned block_size = 0) {
  FilterStage stage;
  stage.id = dx2::bitshuffle_filter_id;
  stage.cd_values = {0, 4, elem_size, block_size, compressor};
  stage.name = "bitshuffle";
  return FilterPipeline{{stage}};
}

FilterPipeline lz4_pipeline() {
  FilterStage stage;
  stage.id = dx2::lz4_filter_id;
  stage.name = "lz4";
  return FilterPipeline{{stage}};
}

std::span<const std::byte> as_bytes(const auto &vec) {
  return std::as_bytes(std::span(vec));
}

std::span<std::byte> as_writable_bytes(auto &vec) {
  return std::as_writable_bytes(std::span(vec));
}

// Pixels with a Poisson-like spread of small counts and a few hot ones, so
// that bitshuffle has something to do and LZ4 something to compress.
std::vector<uint16_t> test_pixels(size_t n, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::poisson_distribution<int> background(3);
  std::vector<uint16_t> pixels(n);
  for (size_t i = 0; i < n; ++i) {
    pixels[i] = static_cast<uint16_t>(background(rng));
    if (i % 97 == 0) {
      pixels[i] = static_cast<uint16_t>(1000 + i % 5000);
    }
  }
  return pixels;
}

// An in-memory HDF5 file; the name only has to be unique while open.
h5utils::H5File in_memory_file(const char *name) {
  h5utils::H5Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
  H5Pset_fapl_core(fapl, 1 << 20, false);
  return h5utils::H5File(H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
}

// A one-chunk 3-D uint16 dataset {1, slow, fast} with one filter set as a
// writer sets it, with the filter's user parameters.
h5utils::H5Dataset
make_filtered_dataset(hid_t file, size_t slow, size_t fast, H5Z_filter_t filter,
                      std::span<const unsigned> user_values) {
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  const hsize_t dims[3] = {1, slow, fast};
  H5Pset_chunk(dcpl, 3, dims);
  H5Pset_filter(dcpl, filter, H5Z_FLAG_OPTIONAL, user_values.size(),
                user_values.data());
  h5utils::H5Space space(H5Screate_simple(3, dims, nullptr));
  return h5utils::H5Dataset(H5Dcreate2(file, "data", H5T_NATIVE_UINT16, space,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT));
}

// Writes pixels through the HDF5 filter plugin and returns the chunk as
// the plugin stored it, or nothing if the plugin did not run.
std::optional<std::vector<std::byte>>
chunk_stored_by_plugin(hid_t dataset, const std::vector<uint16_t> &pixels) {
  if (H5Dwrite(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               pixels.data()) < 0) {
    return std::nullopt;
  }
  // The chunk reaches the file only once it leaves the chunk cache.
  H5Dflush(dataset);
  const hsize_t offset[3] = {0, 0, 0};
  hsize_t stored_bytes = 0;
  if (H5Dget_chunk_storage_size(dataset, offset, &stored_bytes) < 0) {
    return std::nullopt;
  }
  std::vector<std::byte> chunk(stored_bytes);
  uint32_t filter_mask = 0;
  if (H5Dread_chunk(dataset, H5P_DEFAULT, offset, &filter_mask, chunk.data()) <
          0 ||
      filter_mask != 0) {
    return std::nullopt;
  }
  return chunk;
}

// Writes a stored chunk straight into the dataset and reads it back as
// pixels through the HDF5 filter plugin.
std::vector<uint16_t> pixels_read_by_plugin(hid_t dataset,
                                            std::span<const std::byte> chunk,
                                            size_t num_pixels) {
  const hsize_t offset[3] = {0, 0, 0};
  std::vector<uint16_t> pixels(num_pixels);
  if (H5Dwrite_chunk(dataset, H5P_DEFAULT, 0, offset, chunk.size(),
                     chunk.data()) < 0 ||
      H5Dread(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              pixels.data()) < 0) {
    return {};
  }
  return pixels;
}

} // namespace

// An empty pipeline is the uncompressed case.
TEST(FilterPipelineTest, EmptyPipelineSummarisesAsNone) {
  EXPECT_EQ(FilterPipeline{}.summary(), Compression::None)
      << "An empty pipeline did not summarise as None";
}

// The bitshuffle filter's compressor parameter selects the variant.
TEST(FilterPipelineTest, BitshuffleCompressorParameterSelectsTheVariant) {
  EXPECT_EQ(bitshuffle_pipeline(0).summary(), Compression::Bitshuffle)
      << "compressor 0 is bitshuffle alone";
  EXPECT_EQ(bitshuffle_pipeline(2).summary(), Compression::BitshuffleLZ4)
      << "compressor 2 is bitshuffle+lz4";
  EXPECT_EQ(bitshuffle_pipeline(3).summary(), Compression::BitshuffleZstd)
      << "compressor 3 is bitshuffle+zstd";
  EXPECT_EQ(bitshuffle_pipeline(7).summary(), Compression::Unsupported)
      << "An unknown compressor value must not be guessed";
}

// Single-stage LZ4 and deflate are named; a second stage makes the pipeline
// unsupported however familiar the stages are.
TEST(FilterPipelineTest, OtherSingleStagesAreNamedAndTwoStagesAreUnsupported) {
  EXPECT_EQ(lz4_pipeline().summary(), Compression::LZ4)
      << "Filter 32004 is lz4";
  FilterStage deflate;
  deflate.id = H5Z_FILTER_DEFLATE;
  EXPECT_EQ(FilterPipeline{{deflate}}.summary(), Compression::Deflate)
      << "Filter 1 is deflate";
  FilterStage shuffle;
  shuffle.id = H5Z_FILTER_SHUFFLE;
  EXPECT_EQ((FilterPipeline{{shuffle, deflate}}.summary()),
            Compression::Unsupported)
      << "shuffle then deflate must be unsupported";
}

// The pipeline read back from a property list matches what was set,
// including the parameter values a codec depends on.
TEST(FilterPipelineTest, PipelineReadsBackFromAPropertyList) {
  h5utils::H5Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
  const hsize_t chunk[3] = {1, 4, 8};
  ASSERT_GE(H5Pset_chunk(dcpl, 3, chunk), 0) << "H5Pset_chunk failed";
  const unsigned cd_values[5] = {0, 4, 2, 256, 2};
  ASSERT_GE(H5Pset_filter(dcpl, dx2::bitshuffle_filter_id, H5Z_FLAG_OPTIONAL, 5,
                          cd_values),
            0)
      << "H5Pset_filter failed";

  FilterPipeline pipeline = FilterPipeline::from_plist(dcpl);
  ASSERT_EQ(pipeline.stages.size(), 1u) << "Expected one stage";
  EXPECT_EQ(pipeline.stages[0].id, dx2::bitshuffle_filter_id)
      << "Stage id differs";
  EXPECT_EQ(pipeline.stages[0].flags, H5Z_FLAG_OPTIONAL)
      << "Stage flags differ";
  EXPECT_EQ(pipeline.stages[0].cd_values,
            (std::vector<unsigned>{0, 4, 2, 256, 2}))
      << "cd_values differ";
  EXPECT_EQ(pipeline.summary(), Compression::BitshuffleLZ4)
      << "Summary differs";
}

// A writer passes the filter its two user parameters, {block_size,
// compressor}. When the plugin is loadable, dataset creation runs its
// set_local, which stores {major, minor, elem_size, block_size,
// compressor}; that stored form is what every real file holds and what
// summary() reads.
TEST(FilterPipelineTest, DatasetCreationExpandsTheBitshuffleParameters) {
  if (H5Zfilter_avail(dx2::bitshuffle_filter_id) <= 0) {
    GTEST_SKIP() << "bitshuffle plugin not loadable; set_local cannot run";
  }
  h5utils::H5File file = in_memory_file("expand.h5");
  ASSERT_TRUE(file) << "Could not create the in-memory file";
  const unsigned user_values[2] = {0, 2};
  h5utils::H5Dataset dataset =
      make_filtered_dataset(file, 4, 8, dx2::bitshuffle_filter_id, user_values);
  ASSERT_TRUE(dataset) << "Could not create the dataset";

  FilterPipeline pipeline = FilterPipeline::from_dataset(dataset);
  ASSERT_EQ(pipeline.stages.size(), 1u) << "Expected one stage";
  ASSERT_EQ(pipeline.stages[0].cd_values.size(), 5u)
      << "set_local did not expand the parameters: " << pipeline.describe();
  EXPECT_EQ(pipeline.stages[0].cd_values[2], 2u)
      << "Slot 2 is not the element size";
  EXPECT_EQ(pipeline.stages[0].cd_values[4], 2u)
      << "Slot 4 is not the compressor";
  EXPECT_EQ(pipeline.summary(), Compression::BitshuffleLZ4)
      << "Summary differs";
}

// The header is big-endian: 8 bytes of uncompressed size then 4 of block
// size in bytes. The literal bytes here are the filter's own layout.
TEST(Lz4ContainerHeaderTest, HeaderParsesBigEndianFields) {
  const std::array<std::byte, 12> bytes = {
      std::byte{0}, std::byte{0}, std::byte{0},    std::byte{0},
      std::byte{0}, std::byte{1}, std::byte{0x00}, std::byte{0x02},
      std::byte{0}, std::byte{0}, std::byte{0x20}, std::byte{0x00}};
  Lz4ContainerHeader header = Lz4ContainerHeader::parse(bytes);
  EXPECT_EQ(header.uncompressed_bytes, 0x10002u)
      << "uncompressed_bytes was not read big-endian";
  EXPECT_EQ(header.block_bytes, 0x2000u)
      << "block_bytes was not read big-endian";
  EXPECT_EQ(header.block_elements(2), 4096u)
      << "block_elements is block_bytes / elem_size";

  std::array<std::byte, 12> rewritten{};
  header.write(rewritten);
  EXPECT_EQ(rewritten, bytes) << "write did not reproduce the header bytes";
}

// Fewer than 12 bytes cannot be a compressed chunk.
TEST(Lz4ContainerHeaderTest, ShortChunkIsRejected) {
  std::array<std::byte, 11> bytes{};
  EXPECT_THROW(Lz4ContainerHeader::parse(bytes), std::runtime_error)
      << "An 11-byte chunk was accepted as a header";
}

// The codec alternatives are in Compression order, so compression_of is
// the variant index; make_codec builds the matching alternative.
TEST(CodecTest, MakeCodecBuildsTheAlternativeForEachCompression) {
  EXPECT_TRUE(std::holds_alternative<dx2::IdentityCodec>(
      dx2::make_codec(Compression::None, 2)))
      << "None is not the identity codec";
  Codec bshuf = dx2::make_codec(Compression::BitshuffleLZ4, 4, 256);
  ASSERT_TRUE(std::holds_alternative<dx2::BitshuffleLz4Codec>(bshuf))
      << "BitshuffleLZ4 built the wrong codec";
  EXPECT_EQ(std::get<dx2::BitshuffleLz4Codec>(bshuf).elem_size, 4u)
      << "Element size was not carried into the codec";
  EXPECT_EQ(dx2::compression_of(bshuf), Compression::BitshuffleLZ4)
      << "compression_of does not invert make_codec";
  Codec lz4 = dx2::make_codec(Compression::LZ4, 2, 512);
  EXPECT_EQ(std::get<dx2::Lz4Codec>(lz4).block_bytes, 1024u)
      << "LZ4 block size in bytes is block_elements * elem_size";
  EXPECT_THROW(dx2::make_codec(Compression::Deflate, 2), std::runtime_error)
      << "Deflate has no codec";
}

// From a pipeline, deflate and unknown filters yield no codec so a reader
// can hand them to HDF5; the bitshuffle block size comes from cd_values[3].
TEST(CodecTest, PipelineWithoutACodecYieldsNothing) {
  FilterStage deflate;
  deflate.id = H5Z_FILTER_DEFLATE;
  EXPECT_FALSE(dx2::make_codec(FilterPipeline{{deflate}}, PixelType::UInt16))
      << "Deflate must be left to HDF5";
  EXPECT_FALSE(
      ChunkDecoder::can_decode(FilterPipeline{{deflate}}, PixelType::UInt16))
      << "can_decode disagrees with make_codec";
  auto bshuf =
      dx2::make_codec(bitshuffle_pipeline(0, 2, 1024), PixelType::UInt16);
  ASSERT_TRUE(bshuf) << "Bitshuffle alone has a codec";
  EXPECT_EQ(std::get<dx2::BitshuffleCodec>(*bshuf).block_elements, 1024u)
      << "Block size was not read from cd_values[3]";
}

// A compressed chunk from encode_chunk starts with the header the
// reference filter writes: the pixel byte count and the default block
// size (8192 bytes for a 2-byte element, which is 4096 elements) when no
// block size was requested.
TEST(EncodeChunkTest, EncodedChunkCarriesTheFilterHeader) {
  const auto pixels = test_pixels(32 * 64);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::BitshuffleLZ4, 2);
  ASSERT_GT(chunk.size(), Lz4ContainerHeader::SIZE_BYTES)
      << "Encoded chunk is too short to hold a header";
  Lz4ContainerHeader header = Lz4ContainerHeader::parse(chunk);
  EXPECT_EQ(header.uncompressed_bytes, pixels.size() * 2)
      << "Header does not record the pixel byte count";
  EXPECT_EQ(header.block_bytes, 8192u)
      << "Header does not record the default block size in bytes";
  EXPECT_LT(chunk.size(), pixels.size() * 2)
      << "Sparse count data did not compress";
}

// Pixels survive a bitshuffle+lz4 round trip through the decoder.
TEST(ChunkDecoderTest, BitshuffleLz4RoundTripRestoresThePixels) {
  const auto pixels = test_pixels(128 * 256);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::BitshuffleLZ4, 2);

  ChunkDecoder decoder(bitshuffle_pipeline(2), PixelType::UInt16,
                       pixels.size());
  EXPECT_EQ(decoder.compression(), Compression::BitshuffleLZ4)
      << "Decoder reports the wrong compression";
  EXPECT_EQ(decoder.header_bytes(), 12u) << "Compressed chunks have headers";
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Decoded pixels differ from the originals";
}

// A writer may pick its own block size. The header records it and the
// decoder must use that, not the library default, or every block after the
// first is unshuffled with the wrong stride.
TEST(ChunkDecoderTest, DecoderHonoursTheBlockSizeInTheHeader) {
  const auto pixels = test_pixels(128 * 256, 7);
  const size_t block_elements = 256;
  const auto chunk = dx2::encode_chunk(
      as_bytes(pixels), Compression::BitshuffleLZ4, 2, block_elements);
  EXPECT_EQ(Lz4ContainerHeader::parse(chunk).block_bytes, block_elements * 2)
      << "Header does not carry the requested block size";

  // The pipeline still says block size 0, as a writer that set it at
  // compress time but not in cd_values would leave it.
  ChunkDecoder decoder(bitshuffle_pipeline(2), PixelType::UInt16,
                       pixels.size());
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels)
      << "Decoding with a non-default block size lost pixels";
}

// Bitshuffle without compression has no header and takes its block size
// from the filter parameters.
TEST(ChunkDecoderTest, BitshuffleAloneRoundTripsWithoutAHeader) {
  const auto pixels = test_pixels(64 * 64, 3);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::Bitshuffle, 2);
  EXPECT_EQ(chunk.size(), pixels.size() * 2)
      << "Bitshuffle alone must not change the size";

  ChunkDecoder decoder(bitshuffle_pipeline(0), PixelType::UInt16,
                       pixels.size());
  EXPECT_EQ(decoder.header_bytes(), 0u) << "Bitshuffle alone has no header";
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Unshuffled pixels differ from the originals";
}

// 32-bit pixels take the same path with a different element size; this is
// the case the legacy reader got wrong by fixing the size at build time.
TEST(ChunkDecoderTest, ThirtyTwoBitPixelsRoundTrip) {
  std::vector<uint32_t> pixels(64 * 128);
  std::iota(pixels.begin(), pixels.end(), 70000u);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::BitshuffleLZ4, 4);

  ChunkDecoder decoder(bitshuffle_pipeline(2, 4), PixelType::UInt32,
                       pixels.size());
  EXPECT_EQ(decoder.decoded_bytes(), pixels.size() * 4)
      << "decoded_bytes ignores the element size";
  std::vector<uint32_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Decoded uint32 pixels differ";
}

// Plain LZ4 round-trips through its own codec, in several blocks.
TEST(ChunkDecoderTest, Lz4RoundTripsInBlocks) {
  const auto pixels = test_pixels(100 * 100, 5);
  const size_t block_elements = 3000;
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::LZ4, 2, block_elements);
  EXPECT_EQ(Lz4ContainerHeader::parse(chunk).block_bytes, block_elements * 2)
      << "Header does not carry the block size in bytes";
  EXPECT_LT(chunk.size(), pixels.size() * 2) << "Count data did not compress";

  ChunkDecoder decoder(lz4_pipeline(), PixelType::UInt16, pixels.size());
  EXPECT_EQ(decoder.header_bytes(), 12u) << "LZ4 chunks have headers";
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Decoded lz4 pixels differ";
}

// The LZ4 container, built by hand from the documented format: header, then
// per block a big-endian length and the bytes, held raw when the length
// equals the block's decoded size. 24 bytes in blocks of 16 gives a full
// block and a partial last one.
TEST(ChunkDecoderTest, Lz4DecodesAHandBuiltContainerWithRawBlocks) {
  std::vector<uint16_t> pixels(12);
  std::iota(pixels.begin(), pixels.end(), 1u);
  std::vector<std::byte> chunk(12 + 4 + 16 + 4 + 8);
  Lz4ContainerHeader{24, 16}.write(chunk);
  const auto bytes = as_bytes(pixels);
  chunk[15] = std::byte{16};
  std::copy_n(bytes.begin(), 16, chunk.begin() + 16);
  chunk[35] = std::byte{8};
  std::copy_n(bytes.begin() + 16, 8, chunk.begin() + 36);

  ChunkDecoder decoder(lz4_pipeline(), PixelType::UInt16, pixels.size());
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Raw blocks were not copied through";

  // A block length running past the chunk is caught before any read.
  chunk[35] = std::byte{9};
  EXPECT_THROW(decoder.decode(chunk, 0, as_writable_bytes(decoded)),
               std::runtime_error)
      << "A block length past the end of the chunk was accepted";
}

// With filter mask bit 0 set the stage was skipped and the chunk holds
// plain pixels, whatever the dataset's pipeline says.
TEST(ChunkDecoderTest, MaskedChunkIsCopiedAsPlainPixels) {
  const auto pixels = test_pixels(16 * 16);
  ChunkDecoder decoder(bitshuffle_pipeline(2), PixelType::UInt16,
                       pixels.size());
  EXPECT_EQ(decoder.header_bytes(1), 0u)
      << "A masked chunk has no filter header";
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(as_bytes(pixels), 1, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Masked chunk was not copied through";
}

// The decoder refuses pipelines without a codec at construction, so a
// reader can choose the HDF5 fallback before reading any chunk.
TEST(ChunkDecoderTest, PipelineWithoutACodecIsRefusedAtConstruction) {
  FilterStage deflate;
  deflate.id = H5Z_FILTER_DEFLATE;
  EXPECT_THROW(ChunkDecoder(FilterPipeline{{deflate}}, PixelType::UInt16, 16),
               std::runtime_error)
      << "Constructing a decoder for deflate did not throw";
}

// Size mismatches are caught before any library call touches memory.
TEST(ChunkDecoderTest, WrongSizedBuffersThrow) {
  const auto pixels = test_pixels(64 * 64);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::BitshuffleLZ4, 2);
  ChunkDecoder decoder(bitshuffle_pipeline(2), PixelType::UInt16,
                       pixels.size());

  std::vector<uint16_t> short_dst(pixels.size() - 1);
  EXPECT_THROW(decoder.decode(chunk, 0, as_writable_bytes(short_dst)),
               std::runtime_error)
      << "A short destination was accepted";

  ChunkDecoder other(bitshuffle_pipeline(2), PixelType::UInt16,
                     pixels.size() / 2);
  std::vector<uint16_t> half(pixels.size() / 2);
  EXPECT_THROW(other.decode(chunk, 0, as_writable_bytes(half)),
               std::runtime_error)
      << "A header claiming more bytes than the chunk holds was accepted";
}

// zstd follows the same container; the test runs only in a build with it.
TEST(ChunkDecoderTest, BitshuffleZstdRoundTripsWhenCompiledIn) {
  if (!dx2::zstd_supported()) {
    GTEST_SKIP() << "zstd support is not compiled in";
  }
  const auto pixels = test_pixels(64 * 128, 11);
  const auto chunk =
      dx2::encode_chunk(as_bytes(pixels), Compression::BitshuffleZstd, 2);
  ChunkDecoder decoder(bitshuffle_pipeline(3), PixelType::UInt16,
                       pixels.size());
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(chunk, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Decoded zstd pixels differ";
}

// Cross-checks against the HDF5 filter plugins: a chunk encoded here reads
// back through the plugin, and a chunk the plugin stored decodes here.
// This pins the container format to the writers in the field rather than
// to this implementation. Skipped where the plugin cannot load; the second
// half is skipped where the plugin's own compressor fails and HDF5 stores
// the chunk raw, which the LZ4 plugin in hdf5-external-filter-plugins
// 0.1.0 does.
struct PluginCase {
  const char *label;
  H5Z_filter_t filter;
  std::vector<unsigned> user_values;
  Compression compression;
  size_t block_elements;
};

class PluginCrossCheck : public testing::TestWithParam<PluginCase> {};

TEST_P(PluginCrossCheck, PluginAndCodecAgreeOnTheStoredChunk) {
  const PluginCase &c = GetParam();
  if (H5Zfilter_avail(c.filter) <= 0) {
    GTEST_SKIP() << c.label << " plugin not loadable";
  }
  const size_t slow = 60;
  const size_t fast = 100;
  const auto pixels = test_pixels(slow * fast, 13);

  // Codec encodes, plugin reads.
  h5utils::H5File read = in_memory_file("plugin_reads.h5");
  h5utils::H5Dataset plugin_read =
      make_filtered_dataset(read, slow, fast, c.filter, c.user_values);
  ASSERT_TRUE(plugin_read) << "Could not create the dataset";
  const auto encoded =
      dx2::encode_chunk(as_bytes(pixels), c.compression, 2, c.block_elements);
  EXPECT_EQ(pixels_read_by_plugin(plugin_read, encoded, pixels.size()), pixels)
      << "Plugin could not read the codec's chunk";

  // Plugin writes, codec decodes.
  h5utils::H5File written = in_memory_file("plugin_writes.h5");
  h5utils::H5Dataset plugin_written =
      make_filtered_dataset(written, slow, fast, c.filter, c.user_values);
  ASSERT_TRUE(plugin_written) << "Could not create the dataset";
  auto stored = chunk_stored_by_plugin(plugin_written, pixels);
  if (!stored) {
    GTEST_SKIP() << c.label
                 << " plugin did not compress the chunk; only its decompress "
                    "side was checked";
  }
  ChunkDecoder decoder(FilterPipeline::from_dataset(plugin_written),
                       PixelType::UInt16, pixels.size());
  ASSERT_EQ(decoder.compression(), c.compression)
      << "The dataset's pipeline did not summarise as expected";
  std::vector<uint16_t> decoded(pixels.size());
  decoder.decode(*stored, 0, as_writable_bytes(decoded));
  EXPECT_EQ(decoded, pixels) << "Codec could not decode the plugin's chunk";
}

INSTANTIATE_TEST_SUITE_P(
    Plugins, PluginCrossCheck,
    testing::Values(PluginCase{"bitshuffle",
                               dx2::bitshuffle_filter_id,
                               {0, 0},
                               Compression::Bitshuffle,
                               0},
                    PluginCase{"bitshuffle_lz4",
                               dx2::bitshuffle_filter_id,
                               {0, 2},
                               Compression::BitshuffleLZ4,
                               0},
                    PluginCase{"bitshuffle_lz4_small_blocks",
                               dx2::bitshuffle_filter_id,
                               {256, 2},
                               Compression::BitshuffleLZ4,
                               256},
                    PluginCase{
                        "lz4", dx2::lz4_filter_id, {}, Compression::LZ4, 0},
                    PluginCase{"lz4_partial_last_block",
                               dx2::lz4_filter_id,
                               {},
                               Compression::LZ4,
                               1024}),
    [](const testing::TestParamInfo<PluginCase> &info) {
      return info.param.label;
    });
