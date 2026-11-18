/**
 * @file chunk_decoder.hpp
 * @brief Turns a stored chunk back into pixels, and pixels into a chunk.
 *
 * Each compression that dx2 handles itself is one codec: a small struct
 * holding the parameters that compression needs and nothing else, with
 * decode, encode and header_bytes members. Codec is the variant over
 * them, so a chunk is handled by one std::visit and adding a compression
 * means adding one struct and one alternative; a codec that lacks a
 * member fails to compile at the visit rather than falling through a
 * forgotten branch. make_codec is the one place that maps a filter
 * pipeline onto a codec, and returns nothing for pipelines that HDF5
 * must decode, deflate among them. The codecs hold no scratch state and
 * the bitshuffle and LZ4 routines they call work only on the buffers
 * they are given, so one ChunkDecoder serves every thread.
 *
 * Bitshuffle-LZ4 and bitshuffle-zstd chunks are unshuffled with the
 * block size recorded in their own Lz4ContainerHeader, which is what the
 * reference filter does and what a writer that chose a non-default block
 * size requires; the block size in the filter parameters is only the
 * writer's request. The bitshuffle library reads each block's compressed
 * length from the stream without knowing the input length, so a
 * corrupted chunk can be read past its end before the length check here
 * catches it. The LZ4 codec checks every length against the input
 * before use.
 *
 * encode produces the stored form of a chunk, header included, matching
 * the HDF5 filters byte for byte. It exists for writing test fixtures
 * and small files, not for a detector-rate writer. encoded_bound is the
 * largest stored chunk a given decoded size can produce, for sizing the
 * buffers raw chunks are read into.
 */

#pragma once

#include "dx2/image/compression.hpp"
#include "dx2/image/pixel_type.hpp"
#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace dx2 {

/// No filter: the stored chunk is the pixels. Also every masked chunk.
struct IdentityCodec {
  size_t header_bytes() const { return 0; }
  size_t encoded_bound(size_t decoded_bytes) const { return decoded_bytes; }
  void decode(std::span<const std::byte> raw, std::span<std::byte> dst) const;
  std::vector<std::byte> encode(std::span<const std::byte> pixels) const;
};

/// Bitshuffle alone: bits transposed within blocks of elements, no header.
struct BitshuffleCodec {
  size_t elem_size;
  /// Block size in elements; 0 is the library default.
  size_t block_elements;
  size_t header_bytes() const { return 0; }
  size_t encoded_bound(size_t decoded_bytes) const { return decoded_bytes; }
  void decode(std::span<const std::byte> raw, std::span<std::byte> dst) const;
  std::vector<std::byte> encode(std::span<const std::byte> pixels) const;
};

/// Bitshuffle then LZ4 per block, behind an Lz4ContainerHeader.
struct BitshuffleLz4Codec {
  size_t elem_size;
  /// Block size in elements used by encode; 0 is the library default.
  size_t block_elements;
  size_t header_bytes() const { return Lz4ContainerHeader::SIZE_BYTES; }
  size_t encoded_bound(size_t decoded_bytes) const;
  void decode(std::span<const std::byte> raw, std::span<std::byte> dst) const;
  std::vector<std::byte> encode(std::span<const std::byte> pixels) const;
};

/// Bitshuffle then zstd per block, behind an Lz4ContainerHeader. Decoding
/// throws in a build without zstd; make_codec never produces it there.
struct BitshuffleZstdCodec {
  size_t elem_size;
  /// Block size in elements used by encode; 0 is the library default.
  size_t block_elements;
  size_t header_bytes() const { return Lz4ContainerHeader::SIZE_BYTES; }
  size_t encoded_bound(size_t decoded_bytes) const;
  void decode(std::span<const std::byte> raw, std::span<std::byte> dst) const;
  std::vector<std::byte> encode(std::span<const std::byte> pixels) const;
};

/// The LZ4 filter: LZ4 per block of bytes, behind an Lz4ContainerHeader.
/// A block whose stored length equals its decoded length is held raw.
struct Lz4Codec {
  /// Block size in bytes used by encode; 0 means one block per chunk.
  size_t block_bytes;
  size_t header_bytes() const { return Lz4ContainerHeader::SIZE_BYTES; }
  size_t encoded_bound(size_t decoded_bytes) const;
  void decode(std::span<const std::byte> raw, std::span<std::byte> dst) const;
  std::vector<std::byte> encode(std::span<const std::byte> pixels) const;
};

/// Every compression dx2 decodes itself, in Compression order.
using Codec = std::variant<IdentityCodec, BitshuffleCodec, BitshuffleLz4Codec,
                           BitshuffleZstdCodec, Lz4Codec>;

static_assert(Codec(IdentityCodec{}).index() ==
              static_cast<size_t>(Compression::None));
static_assert(Codec(BitshuffleCodec{}).index() ==
              static_cast<size_t>(Compression::Bitshuffle));
static_assert(Codec(BitshuffleLz4Codec{}).index() ==
              static_cast<size_t>(Compression::BitshuffleLZ4));
static_assert(Codec(BitshuffleZstdCodec{}).index() ==
              static_cast<size_t>(Compression::BitshuffleZstd));
static_assert(Codec(Lz4Codec{}).index() ==
              static_cast<size_t>(Compression::LZ4));

/// The Compression a codec handles.
inline Compression compression_of(const Codec &codec) {
  return static_cast<Compression>(codec.index());
}

/// Whether this build decodes bitshuffle-zstd chunks.
bool zstd_supported();

/**
 * @brief The codec for a compression, with the parameters a writer picks.
 *
 * @param elem_size Element size in bytes.
 * @param block_elements Block size in elements for the bitshuffle codecs,
 * 0 for the library default; for LZ4, multiplied by elem_size to give the
 * block size in bytes, 0 for one block per chunk.
 * @throws std::runtime_error for a compression without a codec, or for
 * bitshuffle-zstd in a build without zstd.
 */
Codec make_codec(Compression compression, size_t elem_size,
                 size_t block_elements = 0);

/**
 * @brief The codec for a dataset's pipeline, or nothing if HDF5 must
 * decode it.
 *
 * Block sizes come from the filter parameters: cd_values[3] in elements
 * for bitshuffle, cd_values[0] in bytes for LZ4.
 */
std::optional<Codec> make_codec(const FilterPipeline &pipeline, PixelType type);

/**
 * @brief Decodes stored chunks of one dataset into pixels.
 *
 * Immutable after construction and safe to share between threads.
 */
class ChunkDecoder {
public:
  /// Whether a ChunkDecoder can be built for this pipeline.
  static bool can_decode(const FilterPipeline &pipeline, PixelType type);

  /**
   * @param pipeline The dataset's filter pipeline.
   * @param type The dataset's pixel type.
   * @param num_pixels Pixels per chunk.
   * @throws std::runtime_error if the pipeline has no codec.
   */
  ChunkDecoder(const FilterPipeline &pipeline, PixelType type,
               size_t num_pixels);

  const Codec &codec() const { return codec_; }
  Compression compression() const { return compression_of(codec_); }
  PixelType type() const { return type_; }
  size_t num_pixels() const { return num_pixels_; }
  /// Size of a decoded chunk.
  size_t decoded_bytes() const { return decoded_bytes_; }

  /**
   * @brief Bytes the stored chunk carries before its compressed payload.
   *
   * The codec's header size, or 0 when the chunk's filter mask skipped
   * the stage.
   */
  size_t header_bytes(unsigned filter_mask = 0) const;

  /**
   * @brief Decodes one stored chunk.
   *
   * @param raw The chunk exactly as stored.
   * @param filter_mask The chunk's filter mask from HDF5; bit 0 set means
   * the single stage was skipped and raw holds plain pixels.
   * @param dst Exactly decoded_bytes() bytes.
   * @throws std::runtime_error on a size mismatch or a decoding error.
   */
  void decode(std::span<const std::byte> raw, unsigned filter_mask,
              std::span<std::byte> dst) const;

private:
  Codec codec_;
  PixelType type_;
  size_t num_pixels_;
  size_t decoded_bytes_;
};

/**
 * @brief Produces the stored form of a chunk.
 *
 * @param pixels The chunk's pixels; a whole number of elements.
 * @param compression A compression with a codec.
 * @param elem_size Size of one element in bytes.
 * @param block_elements As for make_codec.
 * @return The chunk as HDF5 would store it, header included.
 * @throws std::runtime_error as for make_codec, or on an encoding error.
 */
std::vector<std::byte> encode_chunk(std::span<const std::byte> pixels,
                                    Compression compression, size_t elem_size,
                                    size_t block_elements = 0);

} // namespace dx2
