/**
 * @file h5_image_source.hpp
 * @brief ImageSource over the three-dimensional image data of a NeXus file.
 *
 * The frames of a NeXus master file are rarely stored in the master
 * itself. The dataset named by the caller, /entry/data/data by default,
 * may be a plain {frames, slow, fast} dataset, possibly reached through a
 * soft or external link; a virtual dataset whose mappings stitch frame
 * ranges from several source files into one virtual extent; or absent,
 * with the frames spread over sibling datasets data_000001, data_000002
 * and so on, each an external link into a data file. All three are
 * flattened at open into one list of sources, the real datasets dx2 opens
 * itself, and one sorted list of segments, each saying which run of
 * virtual frames comes from which source and at what offset. Every read
 * starts by locating its frame in that table and then works on the plain
 * dataset it lands in; nothing downstream knows which layout the master
 * used.
 *
 * A source chunked with one chunk per frame, {1, slow, fast}, offers each
 * frame as its stored chunk through H5Dread_chunk, bypassing HDF5's filter
 * pipeline, with dx2's own ChunkDecoder to decode or the caller doing so.
 * Raw chunks are offered only when every source qualifies and shares one
 * filter pipeline summary. Anything else is read frame by frame with a
 * hyperslab H5Dread and HDF5's own filters, which needs the filter plugin
 * to be loadable. A virtual dataset whose mappings are not whole frames
 * from a three-dimensional source, such as a sub-frame slab or a printf
 * pattern, is read through HDF5 as a single source with no raw path.
 *
 * Source files named relative to the master are found the way HDF5 finds
 * them, so the raw path and the fallback agree: the HDF5_VDS_PREFIX or
 * HDF5_EXT_PREFIX environment variable, then the prefix in Options, then
 * the master's directory, then the name as given. Every source file must
 * exist at open; a missing one is an error naming the path looked for.
 *
 * One handle per file and dataset is opened at construction and shared by
 * every thread. The HDF5 library used here is built thread-safe, so each
 * call is serialised by its global lock; nothing here adds a second lock,
 * and the decode step of read_frame_into runs outside the library
 * entirely. Frames whose chunk has not been written report no chunk_info,
 * as do frames a segment promises beyond its source's current extent. The
 * extents are read once at open and do not follow a growing dataset yet.
 */

#pragma once

#include "dx2/h5/h5utils.hpp"
#include "dx2/image/chunk_decoder.hpp"
#include "dx2/image/image_source.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dx2 {

class H5ImageSource : public ImageSource {
public:
  struct Options {
    /// Path of the image dataset within the master file.
    std::string dataset_path = "/entry/data/data";
    /// Open for single-writer multiple-reader access.
    bool swmr = false;
    /// Directory tried for relative source file names after the
    /// environment variables and before the master's directory.
    std::optional<std::filesystem::path> vds_prefix;
  };

  /// A real dataset the frames are read from, as the caller sees it.
  struct SourceInfo {
    std::filesystem::path file;
    std::string dataset;
    size_t num_frames;
  };

  /**
   * @param path The NeXus master file.
   * @throws std::runtime_error if the master, a source file or a source
   * dataset cannot be opened, or a source is not {frames, slow, fast} of
   * the same shape and type as the others.
   */
  H5ImageSource(const std::filesystem::path &path, Options options);
  explicit H5ImageSource(const std::filesystem::path &path);

  size_t num_frames() const override { return num_frames_; }
  FrameShape shape() const override { return shape_; }
  PixelType pixel_type() const override { return pixel_type_; }
  const FilterPipeline &filters() const override {
    return sources_.front().filters;
  }
  Compression compression() const override { return filters().summary(); }
  bool supports_raw_chunks() const override { return raw_chunks_; }
  size_t max_raw_chunk_bytes() const override;

  std::optional<ChunkInfo> chunk_info(size_t frame) override;
  RawChunk read_raw_chunk(size_t frame, std::span<std::byte> dst) override;
  void read_frame_into(size_t frame, FrameView dst) override;

  const std::filesystem::path &path() const { return path_; }
  const Options &options() const { return options_; }
  /// The datasets frames are read from, in segment order of first use.
  std::vector<SourceInfo> sources() const;

private:
  struct Source {
    std::filesystem::path file;
    std::string dataset;
    h5utils::H5File file_handle;
    h5utils::H5Dataset dataset_handle;
    size_t num_frames = 0;
    bool one_chunk_per_frame = false;
    FilterPipeline filters;
    std::optional<ChunkDecoder> decoder;
  };
  /// A run of virtual frames [first, first + count) stored as frames
  /// [source_first, source_first + count) of sources_[source].
  struct Segment {
    size_t first;
    size_t count;
    size_t source;
    size_t source_first;
  };
  /// A frame located in its source.
  struct Located {
    Source &source;
    size_t local;
  };

  void discover_layout();
  void add_virtual_segments(hid_t vds, const std::filesystem::path &vds_file);
  void add_numbered_datasets();
  size_t add_source(const std::filesystem::path &file,
                    const std::string &dataset);
  Located locate(size_t frame);
  void read_frame_hyperslab(Located at, FrameView dst);

  std::filesystem::path path_;
  Options options_;
  h5utils::H5File master_;
  std::vector<Source> sources_;
  std::vector<Segment> segments_;
  size_t num_frames_ = 0;
  FrameShape shape_{};
  PixelType pixel_type_{};
  bool raw_chunks_ = false;
};

inline H5ImageSource::H5ImageSource(const std::filesystem::path &path)
    : H5ImageSource(path, Options{}) {}

} // namespace dx2
