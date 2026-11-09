/**
 * @file compression.hpp
 * @brief The filter pipeline of an image dataset and what it means.
 *
 * HDF5 stores each chunk through the dataset's filter pipeline, recorded
 * in the dataset creation property list as an ordered list of filter ids
 * with per-filter parameters. FilterPipeline is that list read back
 * verbatim, and its summary() names the one case the pipeline amounts
 * to, so the rest of the reader can switch on a Compression value rather
 * than re-inspect filter ids. Detector files use the bitshuffle filter
 * (id 32008) with LZ4 or zstd selected in its parameters; the LZ4 filter
 * (32004) and deflate (1) are recognised so they can be named, and every
 * other pipeline is Unsupported, which leaves decoding to HDF5 itself.
 *
 * The LZ4 filter stores a chunk as a 12-byte big-endian header, the
 * uncompressed byte count then the block size in bytes, followed by
 * blocks that each carry their own 4-byte compressed length. The
 * compressing bitshuffle variants reuse that container, so
 * Lz4ContainerHeader reads the header of both. The bitshuffle-only
 * variant has no header.
 *
 * A pipeline describes the dataset; each chunk also carries a filter
 * mask whose bit i set means stage i was skipped for that chunk, which
 * happens when an optional filter fails or would enlarge the data. The
 * decoder applies the mask; this header only describes.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dx2 {

/// Registered id of the bitshuffle HDF5 filter.
inline constexpr H5Z_filter_t bitshuffle_filter_id = 32008;
/// Registered id of the LZ4 HDF5 filter.
inline constexpr H5Z_filter_t lz4_filter_id = 32004;

/// The bitshuffle filter's compressor selector, its cd_values[4].
enum class BitshuffleCompressor : unsigned { None = 0, LZ4 = 2, Zstd = 3 };

/// One filter in a dataset's pipeline, as stored in the file.
struct FilterStage {
  H5Z_filter_t id = H5Z_FILTER_NONE;
  unsigned flags = 0;
  std::vector<unsigned> cd_values;
  std::string name;
};

/// What a filter pipeline amounts to for decoding.
enum class Compression : uint8_t {
  None,
  Bitshuffle,
  BitshuffleLZ4,
  BitshuffleZstd,
  LZ4,
  Deflate,
  Unsupported,
};

/// Short name of a Compression value, for messages.
std::string_view compression_name(Compression compression);

/// A dataset's filter pipeline, in application order.
struct FilterPipeline {
  std::vector<FilterStage> stages;

  /// Reads the pipeline of a dataset creation property list.
  static FilterPipeline from_plist(hid_t dcpl);
  /// Reads the pipeline of an open dataset.
  static FilterPipeline from_dataset(hid_t dataset);

  /**
   * @brief The one case this pipeline amounts to.
   *
   * An empty pipeline is None. A single bitshuffle stage maps to
   * Bitshuffle, BitshuffleLZ4 or BitshuffleZstd by its compressor
   * parameter, a single LZ4 or deflate stage to LZ4 or Deflate. Anything
   * else, including any two-stage pipeline, is Unsupported.
   */
  Compression summary() const;

  /// Filter ids and parameters as text, for messages.
  std::string describe() const;
};

/// The 12-byte header in front of an LZ4, bitshuffle-LZ4 or bitshuffle-zstd
/// chunk.
struct Lz4ContainerHeader {
  static constexpr size_t SIZE_BYTES = 12;

  /// Byte count of the chunk once decoded.
  uint64_t uncompressed_bytes = 0;
  /// Block size in bytes; for bitshuffle, the block size in elements times
  /// the element size.
  uint32_t block_bytes = 0;

  /**
   * @brief Reads the header from the start of a stored chunk.
   * @throws std::runtime_error if fewer than SIZE_BYTES bytes are given.
   */
  static Lz4ContainerHeader parse(std::span<const std::byte> chunk);

  /// Writes the header into the first SIZE_BYTES bytes of dst.
  void write(std::span<std::byte> dst) const;

  /// The block size in elements of the given size, for bitshuffle.
  size_t block_elements(size_t elem_size) const {
    return block_bytes / elem_size;
  }
};

} // namespace dx2
