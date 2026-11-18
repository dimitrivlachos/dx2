/**
 * @file image_source.hpp
 * @brief A sequence of detector frames, read by index from any thread.
 *
 * An ImageSource is a fixed number of frames of one shape and PixelType,
 * each stored as one chunk. Callers either decode frames into their own
 * memory with read_frame_into, or take the stored chunk with
 * read_raw_chunk and decode it themselves, which is how a GPU pipeline
 * uploads compressed bytes and unshuffles on the device. A RawChunk
 * reports where in the chunk the compressed payload starts, so no caller
 * hard-codes a header size. Every method is safe to call from many
 * threads at once after construction; an implementation that reads HDF5
 * relies on the library's own lock and adds none.
 *
 * Frames can be absent: a file still being written has frames whose
 * chunk does not exist yet. chunk_info says whether a frame is readable
 * now without blocking, and wait_for_frame polls it with a deadline.
 * Policy for how long to wait, and for giving up when nothing new has
 * arrived from anywhere, belongs to the caller.
 */

#pragma once

#include "dx2/image/chunk_decoder.hpp"
#include "dx2/image/compression.hpp"
#include "dx2/image/frame.hpp"
#include "dx2/image/pixel_type.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stop_token>

namespace dx2 {

/// Where and how one frame's chunk is stored.
struct ChunkInfo {
  size_t frame = 0;
  /// HDF5 filter mask; bit i set means pipeline stage i was skipped.
  unsigned filter_mask = 0;
  /// Byte offset of the chunk in its file, or 0 when not addressable.
  uint64_t address = 0;
  /// Size of the stored chunk in bytes.
  size_t stored_bytes = 0;
};

/// A stored chunk in a caller's buffer, with what is known about it.
struct RawChunk {
  /// The stored chunk, exactly stored_bytes long.
  std::span<std::byte> bytes;
  ChunkInfo info;
  /// Bytes before the compressed payload; 0 when there is no header.
  size_t header_bytes = 0;

  /// The chunk without its header.
  std::span<std::byte> payload() const { return bytes.subspan(header_bytes); }
  /// The container header, when the chunk has one.
  std::optional<Lz4ContainerHeader> container_header() const {
    if (header_bytes == 0) {
      return std::nullopt;
    }
    return Lz4ContainerHeader::parse(bytes);
  }
};

/**
 * @brief Frames of one shape and type, readable by index from any thread.
 */
class ImageSource {
public:
  virtual ~ImageSource() = default;

  virtual size_t num_frames() const = 0;
  virtual FrameShape shape() const = 0;
  virtual PixelType pixel_type() const = 0;
  virtual const FilterPipeline &filters() const = 0;
  virtual Compression compression() const = 0;

  /// Whether read_raw_chunk works: one chunk per frame, with a codec.
  virtual bool supports_raw_chunks() const = 0;
  /// Upper bound on stored_bytes for any chunk, for sizing buffers.
  virtual size_t max_raw_chunk_bytes() const = 0;

  /**
   * @brief Where a frame's chunk is, or nothing if it is not readable
   * yet. One probe; never blocks waiting for data.
   */
  virtual std::optional<ChunkInfo> chunk_info(size_t frame) = 0;

  /**
   * @brief Reads a frame's stored chunk into dst.
   * @param dst At least the chunk's stored_bytes; max_raw_chunk_bytes()
   * always suffices.
   * @throws std::runtime_error if the frame is not readable, raw chunks
   * are not supported, or dst is too small.
   */
  virtual RawChunk read_raw_chunk(size_t frame, std::span<std::byte> dst) = 0;

  /**
   * @brief Decodes a frame into dst.
   * @param dst A view of this source's pixel type and shape.
   * @throws std::runtime_error if the frame is not readable or dst does
   * not match.
   */
  virtual void read_frame_into(size_t frame, FrameView dst) = 0;

  /// Bytes in one decoded frame.
  size_t frame_bytes() const {
    return shape()[0] * shape()[1] * pixel_size(pixel_type());
  }

  /// A newly allocated, decoded frame.
  Frame read_frame(size_t frame) {
    Frame result(pixel_type(), shape());
    read_frame_into(frame, result.view());
    return result;
  }

  /**
   * @brief Polls chunk_info until the frame is readable.
   *
   * @param timeout Total time to wait before giving up.
   * @param stop Ends the wait early.
   * @param poll Interval between probes.
   * @return true once the frame is readable; false on timeout or stop.
   */
  bool wait_for_frame(
      size_t frame, std::chrono::milliseconds timeout,
      std::stop_token stop = {},
      std::chrono::milliseconds poll = std::chrono::milliseconds(100));
};

} // namespace dx2
