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
 * exist at open unless the master is opened for SWMR reading; then a
 * source whose file is not there yet is left closed and opened on the
 * first request for one of its frames, which reports no chunk until the
 * file can be opened. The numbered layout needs every file at open, since
 * each file's frames start where the extents of the files before it add
 * up to. At least one source must exist at open, because the shape, pixel
 * type and filter pipeline are taken from it; a source opened later must
 * match it, and must offer raw chunks if the others do, or the request
 * that opened it throws.
 *
 * One handle per file and dataset is shared by every thread. The HDF5
 * library used here is built thread-safe, so each call is serialised by
 * its global lock; the only lock added here guards a source's lazy open
 * and refresh, and the decode step of read_frame_into runs outside the
 * library entirely. Frames whose chunk has not been written report no
 * chunk_info, as do frames a segment promises beyond its source's current
 * extent. In SWMR mode a frame that is not there triggers H5Drefresh on
 * its source, at most once per refresh_min_interval however many threads
 * ask, before the chunk is queried again, so a growing dataset is
 * followed and N pollers cost one refresh per interval. num_frames
 * follows the source extents as last refreshed: a plain dataset and an
 * unlimited virtual mapping take their count from the source, while a
 * fixed mapping and a numbered file keep the count found at open. HDF5
 * shares one file object between every open of a file within a process,
 * so two readers of the same files in one process must agree on the SWMR
 * flag, or the second open fails.
 */

#pragma once

#include "dx2/h5/h5utils.hpp"
#include "dx2/image/chunk_decoder.hpp"
#include "dx2/image/image_source.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dx2 {

class H5ImageSource : public ImageSource {
public:
  struct Options {
    /// Path of the image dataset within the master file.
    std::string dataset_path = "/entry/data/data";
    /// Open for single-writer multiple-reader access: sources may appear
    /// and grow after open.
    bool swmr = false;
    /// Shortest gap between two H5Drefresh calls on one source.
    std::chrono::milliseconds refresh_min_interval{20};
    /// Directory tried for relative source file names after the
    /// environment variables and before the master's directory.
    std::optional<std::filesystem::path> vds_prefix;
  };

  /// A real dataset the frames are read from, as the caller sees it.
  struct SourceInfo {
    std::filesystem::path file;
    std::string dataset;
    /// Frames the dataset held when last opened or refreshed; 0 while the
    /// file is still awaited.
    size_t num_frames;
    bool open;
  };

  /**
   * @param path The NeXus master file.
   * @throws std::runtime_error if the master, a source file or a source
   * dataset cannot be opened, no source file exists yet, or a source is
   * not {frames, slow, fast} of the same shape and type as the others.
   */
  H5ImageSource(const std::filesystem::path &path, Options options);
  explicit H5ImageSource(const std::filesystem::path &path);

  size_t num_frames() const override;
  FrameShape shape() const override { return shape_; }
  PixelType pixel_type() const override { return pixel_type_; }
  const FilterPipeline &filters() const override {
    return sources_[reference_]->filters;
  }
  Compression compression() const override { return filters().summary(); }
  bool supports_raw_chunks() const override { return raw_chunks_; }
  size_t max_raw_chunk_bytes() const override;

  std::optional<ChunkInfo> chunk_info(size_t frame) override;
  RawChunk read_raw_chunk(size_t frame, std::span<std::byte> dst) override;
  void read_frame_into(size_t frame, FrameView dst) override;

  const std::filesystem::path &path() const { return path_; }
  const Options &options() const { return options_; }
  /// The datasets frames are read from, in order of first use.
  std::vector<SourceInfo> sources() const;

private:
  /// A real dataset. The handles are set once, under the mutex, and read
  /// without it after `open` is seen true; the frame count follows the
  /// extent as last read.
  struct Source {
    std::filesystem::path file;
    std::string dataset;
    std::mutex mutex;
    std::atomic<bool> open{false};
    h5utils::H5File file_handle;
    h5utils::H5Dataset dataset_handle;
    std::atomic<size_t> num_frames{0};
    bool one_chunk_per_frame = false;
    FilterPipeline filters;
    std::optional<ChunkDecoder> decoder;
    std::chrono::steady_clock::time_point last_refresh{};
  };
  /// A run of virtual frames starting at `first`, stored from frame
  /// `source_first` of sources_[source]. A fixed run holds `count`
  /// frames; an unlimited one holds whatever the source has from
  /// `source_first` on.
  struct Segment {
    size_t first;
    size_t count;
    size_t source;
    size_t source_first;
    bool unlimited;
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
                    const std::string &dataset, bool file_exists);
  void open_dataset(Source &source);
  void check_matches_reference(const Source &source) const;
  bool ensure_open(Source &source);
  void refresh(Source &source);
  size_t segment_frames(const Segment &segment) const;
  std::optional<Located> locate(size_t frame);
  Located locate_open(size_t frame);
  std::optional<ChunkInfo> query_chunk(Located at, size_t frame);
  void read_frame_hyperslab(Located at, FrameView dst);

  std::filesystem::path path_;
  Options options_;
  h5utils::H5File master_;
  std::vector<std::unique_ptr<Source>> sources_;
  std::vector<Segment> segments_;
  size_t reference_ = 0;
  FrameShape shape_{};
  PixelType pixel_type_{};
  bool raw_chunks_ = false;
};

inline H5ImageSource::H5ImageSource(const std::filesystem::path &path)
    : H5ImageSource(path, Options{}) {}

} // namespace dx2
