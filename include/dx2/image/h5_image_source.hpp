/**
 * @file h5_image_source.hpp
 * @brief ImageSource over a three-dimensional HDF5 dataset.
 *
 * The dataset is {frames, slow, fast}, the layout of NeXus detector data.
 * When it is chunked with one chunk per frame, {1, slow, fast}, each
 * frame is one stored chunk and the source offers it raw through
 * H5Dread_chunk, bypassing HDF5's filter pipeline; its own ChunkDecoder
 * then decodes, or the caller does. A dataset chunked any other way, or
 * stored contiguously, or filtered with a pipeline dx2 has no codec for,
 * is read frame by frame with a hyperslab H5Dread and HDF5's own filters,
 * which means the plugin for that filter must be loadable.
 *
 * One file and one dataset handle are opened at construction and shared
 * by every thread. The HDF5 library used here is built thread-safe, so
 * each call is serialised by its global lock; nothing here adds a second
 * lock, and the decode step of read_frame_into runs outside the library
 * entirely. Frames whose chunk has not been written, as in a file still
 * being filled, report no chunk_info. The frame count is read once at
 * open and does not follow a growing dataset yet.
 */

#pragma once

#include "dx2/h5/h5utils.hpp"
#include "dx2/image/chunk_decoder.hpp"
#include "dx2/image/image_source.hpp"
#include <filesystem>
#include <optional>
#include <string>

namespace dx2 {

class H5ImageSource : public ImageSource {
public:
  struct Options {
    /// Path of the image dataset within the file.
    std::string dataset_path = "/entry/data/data";
    /// Open for single-writer multiple-reader access.
    bool swmr = false;
  };

  /**
   * @param path The HDF5 file holding the dataset.
   * @throws std::runtime_error if the file or dataset cannot be opened or
   * the dataset is not {frames, slow, fast}.
   */
  H5ImageSource(const std::filesystem::path &path, Options options);
  explicit H5ImageSource(const std::filesystem::path &path);

  size_t num_frames() const override { return num_frames_; }
  FrameShape shape() const override { return shape_; }
  PixelType pixel_type() const override { return pixel_type_; }
  const FilterPipeline &filters() const override { return filters_; }
  Compression compression() const override { return filters_.summary(); }
  bool supports_raw_chunks() const override {
    return one_chunk_per_frame_ && decoder_.has_value();
  }
  size_t max_raw_chunk_bytes() const override;

  std::optional<ChunkInfo> chunk_info(size_t frame) override;
  RawChunk read_raw_chunk(size_t frame, std::span<std::byte> dst) override;
  void read_frame_into(size_t frame, FrameView dst) override;

  const std::filesystem::path &path() const { return path_; }
  const Options &options() const { return options_; }

private:
  /// Reads the frame through HDF5's own filter pipeline.
  void read_frame_hyperslab(size_t frame, FrameView dst);

  std::filesystem::path path_;
  Options options_;
  h5utils::H5File file_;
  h5utils::H5Dataset dataset_;
  size_t num_frames_ = 0;
  FrameShape shape_{};
  PixelType pixel_type_{};
  FilterPipeline filters_;
  bool one_chunk_per_frame_ = false;
  std::optional<ChunkDecoder> decoder_;
};

inline H5ImageSource::H5ImageSource(const std::filesystem::path &path)
    : H5ImageSource(path, Options{}) {}

} // namespace dx2
