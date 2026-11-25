#include "dx2/image/h5_image_source.hpp"
#include "dx2/h5/h5link.hpp"
#include "dx2/h5/h5read.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <fmt/std.h>
#include <regex>
#include <stdexcept>
#include <vector>

namespace dx2 {

using h5utils::H5Plist;
using h5utils::H5Space;
using h5utils::H5Type;
namespace fs = std::filesystem;

namespace {

/// Reads a virtual mapping's file or dataset name, which HDF5 returns by
/// size first.
template <typename Getter>
std::string virtual_name(Getter getter, hid_t dcpl, size_t index) {
  const ssize_t size = getter(dcpl, index, nullptr, 0);
  if (size < 0) {
    throw std::runtime_error("cannot read a virtual dataset mapping");
  }
  std::string name(static_cast<size_t>(size), '\0');
  getter(dcpl, index, name.data(), name.size() + 1);
  return name;
}

/// The frames a regular 3-D hyperslab selects, if it covers whole frames:
/// start 0 and the full extent in the image dimensions, and a contiguous
/// run in the frame dimension. count is nullopt for an unlimited run.
struct FrameRun {
  hsize_t first;
  std::optional<hsize_t> count;
};
std::optional<FrameRun> whole_frame_run(hid_t space, const hsize_t *dims) {
  if (H5Sget_simple_extent_ndims(space) != 3 ||
      H5Sget_select_type(space) != H5S_SEL_HYPERSLABS ||
      H5Sis_regular_hyperslab(space) <= 0) {
    return std::nullopt;
  }
  hsize_t start[3], stride[3], count[3], block[3];
  if (H5Sget_regular_hyperslab(space, start, stride, count, block) < 0) {
    return std::nullopt;
  }
  // Image dimensions: one block of the full extent, or extent blocks of 1.
  for (int d = 1; d < 3; ++d) {
    const bool full = start[d] == 0 && count[d] * block[d] == dims[d] &&
                      (count[d] == 1 || stride[d] == block[d]);
    if (!full) {
      return std::nullopt;
    }
  }
  // Frame dimension: a contiguous run, possibly unlimited.
  if (count[0] == H5S_UNLIMITED) {
    return block[0] == 1 && stride[0] == 1
               ? std::optional<FrameRun>{{start[0], std::nullopt}}
               : std::nullopt;
  }
  if (count[0] != 1 && stride[0] != block[0]) {
    return std::nullopt;
  }
  return FrameRun{start[0], count[0] * block[0]};
}

/// A virtual mapping dx2 cannot flatten; the dataset is then read through
/// HDF5 as it stands.
struct UnsupportedMapping : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// The frame extent of an open dataset.
size_t frame_extent(hid_t dataset) {
  H5Space space(H5Dget_space(dataset));
  hsize_t dims[3] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  return dims[0];
}

} // namespace

H5ImageSource::H5ImageSource(const fs::path &path, Options options)
    : path_(path), options_(std::move(options)) {
  unsigned flags = H5F_ACC_RDONLY;
  if (options_.swmr) {
    flags |= H5F_ACC_SWMR_READ;
  }
  master_ = h5utils::H5File(H5Fopen(path_.c_str(), flags, H5P_DEFAULT));
  if (!master_) {
    throw std::runtime_error(fmt::format("cannot open {}", path_));
  }
  discover_layout();

  // The first source that exists sets the shape, type and pipeline every
  // other source, now or later, has to match.
  const auto reference = std::find_if(
      sources_.begin(), sources_.end(),
      [](const std::unique_ptr<Source> &s) { return s->open.load(); });
  if (reference == sources_.end()) {
    throw std::runtime_error(fmt::format(
        "no source file of {} in {} exists yet", options_.dataset_path, path_));
  }
  reference_ = static_cast<size_t>(reference - sources_.begin());
  const Source &first = **reference;
  {
    H5Space space(H5Dget_space(first.dataset_handle));
    hsize_t dims[3] = {};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    shape_ = {dims[1], dims[2]};
  }
  pixel_type_ = pixel_type_from_h5(first.dataset_handle);

  // Raw chunks need every source open now to offer them; one opened later
  // is held to the same standard when it opens.
  raw_chunks_ = true;
  for (const std::unique_ptr<Source> &source : sources_) {
    if (!source->open.load()) {
      continue;
    }
    check_matches_reference(*source);
    raw_chunks_ = raw_chunks_ && source->one_chunk_per_frame &&
                  source->decoder.has_value();
  }
}

void H5ImageSource::discover_layout() {
  // A dataset that is not there at all means the numbered-dataset layout.
  bool exists = false;
  {
    h5utils::H5ErrorSilencer silence;
    exists = H5Lexists(master_, options_.dataset_path.c_str(), H5P_DEFAULT) > 0;
  }
  if (!exists) {
    add_numbered_datasets();
    return;
  }

  // Otherwise follow the links to the real dataset and look at its layout.
  const std::vector<fs::path> prefixes = h5link::link_search_prefixes(
      h5link::LinkKind::External, options_.vds_prefix, path_);
  const h5link::ResolvedLink target =
      h5link::resolve_link(master_, path_, options_.dataset_path, prefixes);
  if (!target.file_exists && !options_.swmr) {
    throw std::runtime_error(fmt::format("{} in {} points at {}, which does "
                                         "not exist",
                                         options_.dataset_path, path_,
                                         target.file));
  }
  const size_t index =
      add_source(target.file, target.object_path, target.file_exists);
  if (!target.file_exists) {
    // A plain dataset awaited in SWMR mode; its layout is checked when it
    // appears.
    segments_.push_back({0, 0, index, 0, true});
    return;
  }
  const hid_t dataset = sources_[index]->dataset_handle;
  const fs::path dataset_file = sources_[index]->file;
  H5Plist dcpl(H5Dget_create_plist(dataset));
  if (H5Pget_layout(dcpl) == H5D_VIRTUAL) {
    // The virtual dataset is replaced by its sources when every mapping
    // is whole frames; otherwise it stays as the one source, read through
    // HDF5.
    const size_t before = sources_.size();
    try {
      add_virtual_segments(dataset, dataset_file);
    } catch (const UnsupportedMapping &) {
      sources_.resize(before);
      segments_.clear();
      segments_.push_back({0, 0, index, 0, true});
      return;
    }
    sources_.erase(sources_.begin() + static_cast<ptrdiff_t>(index));
    for (Segment &segment : segments_) {
      if (segment.source > index) {
        --segment.source;
      }
    }
    return;
  }
  // A plain dataset is one open-ended run so a growing extent is followed.
  segments_.push_back({0, 0, index, 0, true});
}

void H5ImageSource::add_virtual_segments(hid_t vds, const fs::path &vds_file) {
  H5Plist dcpl(H5Dget_create_plist(vds));
  H5Space vspace_all(H5Dget_space(vds));
  hsize_t vdims[3] = {};
  if (H5Sget_simple_extent_ndims(vspace_all) != 3) {
    throw UnsupportedMapping("virtual dataset is not 3-D");
  }
  H5Sget_simple_extent_dims(vspace_all, vdims, nullptr);

  size_t count = 0;
  if (H5Pget_virtual_count(dcpl, &count) < 0) {
    throw std::runtime_error("cannot count virtual mappings");
  }
  const std::vector<fs::path> vds_prefixes = h5link::link_search_prefixes(
      h5link::LinkKind::VirtualSource, options_.vds_prefix, vds_file);
  const std::vector<fs::path> elink_prefixes = h5link::link_search_prefixes(
      h5link::LinkKind::External, options_.vds_prefix, vds_file);

  for (size_t i = 0; i < count; ++i) {
    // The virtual selection must be whole frames: it says which frames.
    H5Space vspace(H5Pget_virtual_vspace(dcpl, i));
    const std::optional<FrameRun> virtual_run = whole_frame_run(vspace, vdims);
    if (!virtual_run) {
      throw UnsupportedMapping("virtual mapping is not whole frames");
    }
    const std::string file_name =
        virtual_name(H5Pget_virtual_filename, dcpl, i);
    const std::string dataset_name =
        virtual_name(H5Pget_virtual_dsetname, dcpl, i);
    if (file_name.find('%') != std::string::npos ||
        dataset_name.find('%') != std::string::npos) {
      throw UnsupportedMapping("printf-style virtual mapping");
    }

    // The source name may itself be a link chain inside the source file,
    // which can only be followed once the file is there; until then the
    // name is taken as given.
    const h5link::ResolvedLink found =
        h5link::find_linked_file(file_name, vds_file, vds_prefixes);
    h5link::ResolvedLink target{found.file, dataset_name, false};
    if (found.file_exists) {
      h5utils::H5File source_file(
          H5Fopen(found.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
      if (!source_file) {
        throw std::runtime_error(fmt::format("cannot open {}", found.file));
      }
      target = h5link::resolve_link(source_file, found.file, dataset_name,
                                    elink_prefixes);
    }
    if (!target.file_exists && !options_.swmr) {
      throw std::runtime_error(fmt::format(
          "virtual source {} of {} does not exist", target.file, vds_file));
    }
    const size_t index =
        add_source(target.file, target.object_path, target.file_exists);

    // The source selection must be whole frames of the source's own image
    // size, which is only known once the source is open; an awaited
    // source is held to the virtual size and checked when it appears.
    hsize_t sdims[3] = {vdims[0], vdims[1], vdims[2]};
    if (sources_[index]->open.load()) {
      H5Space space(H5Dget_space(sources_[index]->dataset_handle));
      H5Sget_simple_extent_dims(space, sdims, nullptr);
    }
    H5Space sspace(H5Pget_virtual_srcspace(dcpl, i));
    const std::optional<FrameRun> source_run = whole_frame_run(sspace, sdims);
    if (!source_run) {
      throw UnsupportedMapping("virtual source selection is not whole frames");
    }
    segments_.push_back({static_cast<size_t>(virtual_run->first),
                         static_cast<size_t>(virtual_run->count.value_or(0)),
                         index, static_cast<size_t>(source_run->first),
                         !virtual_run->count.has_value()});
  }
  if (segments_.empty()) {
    throw UnsupportedMapping("virtual dataset maps no frames");
  }
  std::sort(
      segments_.begin(), segments_.end(),
      [](const Segment &a, const Segment &b) { return a.first < b.first; });
}

void H5ImageSource::add_numbered_datasets() {
  // The siblings data_000001, data_000002, ... of the missing dataset,
  // concatenated in name order.
  const std::string &path = options_.dataset_path;
  const size_t slash = path.find_last_of('/');
  const std::string group_path =
      slash == 0 || slash == std::string::npos ? "/" : path.substr(0, slash);
  h5utils::H5Group group(H5Gopen2(master_, group_path.c_str(), H5P_DEFAULT));
  if (!group) {
    throw std::runtime_error(fmt::format("no dataset {} and no group {} in {}",
                                         path, group_path, path_));
  }
  static const std::regex numbered("^data_[0-9]{6}$");
  std::vector<std::string> names;
  for (const std::string &name : list_link_names(group)) {
    if (std::regex_match(name, numbered)) {
      names.push_back(name);
    }
  }
  if (names.empty()) {
    throw std::runtime_error(
        fmt::format("no dataset {} and no data_NNNNNN datasets in {} of {}",
                    path, group_path, path_));
  }
  std::sort(names.begin(), names.end());

  // Each file's frames start where the extents before it add up to, so
  // every file has to be there to be measured.
  const std::vector<fs::path> prefixes = h5link::link_search_prefixes(
      h5link::LinkKind::External, options_.vds_prefix, path_);
  size_t next_frame = 0;
  for (const std::string &name : names) {
    const std::string link_path =
        group_path == "/" ? "/" + name : group_path + "/" + name;
    const h5link::ResolvedLink target =
        h5link::resolve_link(master_, path_, link_path, prefixes);
    if (!target.file_exists) {
      throw std::runtime_error(
          fmt::format("{} in {} points at {}, which does not exist", link_path,
                      path_, target.file));
    }
    const size_t index = add_source(target.file, target.object_path, true);
    const size_t frames = sources_[index]->num_frames.load();
    segments_.push_back({next_frame, frames, index, 0, false});
    next_frame += frames;
  }
}

size_t H5ImageSource::add_source(const fs::path &file,
                                 const std::string &dataset, bool file_exists) {
  // The same dataset reached twice is one source.
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sources_[i]->file == file && sources_[i]->dataset == dataset) {
      return i;
    }
  }
  auto source = std::make_unique<Source>();
  source->file = file;
  source->dataset = dataset;
  if (file_exists) {
    unsigned flags = H5F_ACC_RDONLY;
    if (options_.swmr) {
      flags |= H5F_ACC_SWMR_READ;
    }
    source->file_handle =
        h5utils::H5File(H5Fopen(file.c_str(), flags, H5P_DEFAULT));
    if (!source->file_handle) {
      throw std::runtime_error(fmt::format("cannot open {}", file));
    }
    open_dataset(*source);
    source->open.store(true);
  }
  sources_.push_back(std::move(source));
  return sources_.size() - 1;
}

void H5ImageSource::open_dataset(Source &source) {
  source.dataset_handle = h5utils::H5Dataset(
      H5Dopen2(source.file_handle, source.dataset.c_str(), H5P_DEFAULT));
  if (!source.dataset_handle) {
    throw std::runtime_error(
        fmt::format("no dataset {} in {}", source.dataset, source.file));
  }

  // The dataset must be {frames, slow, fast}; raw chunks need one chunk
  // per frame and a codec for the pipeline.
  H5Space space(H5Dget_space(source.dataset_handle));
  if (!space || H5Sget_simple_extent_ndims(space) != 3) {
    throw std::runtime_error(fmt::format("{} in {} is not a 3-D dataset",
                                         source.dataset, source.file));
  }
  hsize_t dims[3] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  source.num_frames.store(dims[0]);
  H5Plist dcpl(H5Dget_create_plist(source.dataset_handle));
  source.filters = FilterPipeline::from_plist(dcpl);
  if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
    hsize_t chunk[3] = {};
    if (H5Pget_chunk(dcpl, 3, chunk) == 3) {
      source.one_chunk_per_frame =
          chunk[0] == 1 && chunk[1] == dims[1] && chunk[2] == dims[2];
    }
  }
  const PixelType type = pixel_type_from_h5(source.dataset_handle);
  if (source.one_chunk_per_frame &&
      ChunkDecoder::can_decode(source.filters, type)) {
    source.decoder.emplace(source.filters, type, dims[1] * dims[2]);
  }
}

void H5ImageSource::check_matches_reference(const Source &source) const {
  const Source &first = *sources_[reference_];
  H5Space space(H5Dget_space(source.dataset_handle));
  hsize_t dims[3] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  if (dims[1] != shape_[0] || dims[2] != shape_[1]) {
    throw std::runtime_error(fmt::format(
        "{} in {} is {}x{}, {} in {} is {}x{}", source.dataset, source.file,
        dims[1], dims[2], first.dataset, first.file, shape_[0], shape_[1]));
  }
  if (pixel_type_from_h5(source.dataset_handle) != pixel_type_) {
    throw std::runtime_error(
        fmt::format("{} in {} has a different pixel type from {} in {}",
                    source.dataset, source.file, first.dataset, first.file));
  }
  if (source.filters.summary() != first.filters.summary()) {
    throw std::runtime_error(fmt::format(
        "{} in {} is {} compressed, {} in {} is {}", source.dataset,
        source.file, compression_name(source.filters.summary()), first.dataset,
        first.file, compression_name(first.filters.summary())));
  }
}

bool H5ImageSource::ensure_open(Source &source) {
  if (source.open.load(std::memory_order_acquire)) {
    return true;
  }
  std::lock_guard lock(source.mutex);
  if (source.open.load(std::memory_order_relaxed)) {
    return true;
  }
  // A file that cannot be opened yet is one still being written; the
  // open is tried again on the next request.
  {
    h5utils::H5ErrorSilencer silence;
    source.file_handle = h5utils::H5File(H5Fopen(
        source.file.c_str(), H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT));
  }
  if (!source.file_handle) {
    return false;
  }
  open_dataset(source);
  check_matches_reference(source);
  if (raw_chunks_ && !(source.one_chunk_per_frame && source.decoder)) {
    throw std::runtime_error(
        fmt::format("{} in {} does not offer raw chunks like the other "
                    "sources of {}",
                    source.dataset, source.file, path_));
  }
  source.open.store(true, std::memory_order_release);
  return true;
}

void H5ImageSource::refresh(Source &source) {
  std::lock_guard lock(source.mutex);
  const auto now = std::chrono::steady_clock::now();
  if (now - source.last_refresh < options_.refresh_min_interval) {
    return;
  }
  // The refresh re-reads the dataset's metadata, including its chunk
  // index and extent, from the file as the writer last flushed it.
  if (H5Drefresh(source.dataset_handle) < 0) {
    throw std::runtime_error(
        fmt::format("cannot refresh {} in {}", source.dataset, source.file));
  }
  source.num_frames.store(frame_extent(source.dataset_handle));
  source.last_refresh = now;
}

size_t H5ImageSource::segment_frames(const Segment &segment) const {
  if (!segment.unlimited) {
    return segment.count;
  }
  const size_t extent = sources_[segment.source]->num_frames.load();
  return extent > segment.source_first ? extent - segment.source_first : 0;
}

size_t H5ImageSource::num_frames() const {
  size_t frames = 0;
  for (const Segment &segment : segments_) {
    frames = std::max(frames, segment.first + segment_frames(segment));
  }
  return frames;
}

std::vector<H5ImageSource::SourceInfo> H5ImageSource::sources() const {
  std::vector<SourceInfo> info;
  info.reserve(sources_.size());
  for (const std::unique_ptr<Source> &source : sources_) {
    info.push_back({source->file, source->dataset, source->num_frames.load(),
                    source->open.load()});
  }
  return info;
}

std::optional<H5ImageSource::Located> H5ImageSource::locate(size_t frame) {
  // Segments are sorted by first frame; the last one starting at or
  // before the frame holds it, unless the frame falls in a gap or past a
  // fixed run. An unlimited run takes every later frame, written or not.
  auto after =
      std::upper_bound(segments_.begin(), segments_.end(), frame,
                       [](size_t f, const Segment &s) { return f < s.first; });
  if (after == segments_.begin()) {
    return std::nullopt;
  }
  const Segment &segment = *(after - 1);
  if (!segment.unlimited && frame >= segment.first + segment.count) {
    return std::nullopt;
  }
  return Located{*sources_[segment.source],
                 segment.source_first + (frame - segment.first)};
}

H5ImageSource::Located H5ImageSource::locate_open(size_t frame) {
  std::optional<Located> at = locate(frame);
  if (!at) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not mapped", frame, path_));
  }
  if (!ensure_open(at->source)) {
    throw std::runtime_error(fmt::format("frame {} of {} is in {}, which "
                                         "does not exist yet",
                                         frame, path_, at->source.file));
  }
  return *at;
}

size_t H5ImageSource::max_raw_chunk_bytes() const {
  size_t bound = 0;
  for (const std::unique_ptr<Source> &source : sources_) {
    bound = std::max(bound, source->decoder ? std::visit(
                                                  [&](const auto &codec) {
                                                    return codec.encoded_bound(
                                                        frame_bytes());
                                                  },
                                                  source->decoder->codec())
                                            : frame_bytes());
  }
  return bound;
}

std::optional<ChunkInfo> H5ImageSource::query_chunk(Located at, size_t frame) {
  if (at.local >= at.source.num_frames.load()) {
    return std::nullopt;
  }
  // An allocated chunk has an address and a size; an unwritten one
  // reports HADDR_UNDEF and 0 without error.
  const hsize_t offset[3] = {at.local, 0, 0};
  unsigned filter_mask = 0;
  haddr_t address = HADDR_UNDEF;
  hsize_t size = 0;
  if (H5Dget_chunk_info_by_coord(at.source.dataset_handle, offset, &filter_mask,
                                 &address, &size) < 0) {
    throw std::runtime_error(fmt::format("cannot query chunk {} of {} in {}",
                                         at.local, at.source.dataset,
                                         at.source.file));
  }
  if (address == HADDR_UNDEF || size == 0) {
    return std::nullopt;
  }
  return ChunkInfo{frame, filter_mask, static_cast<uint64_t>(address),
                   static_cast<size_t>(size)};
}

std::optional<ChunkInfo> H5ImageSource::chunk_info(size_t frame) {
  const std::optional<Located> at = locate(frame);
  if (!at || !ensure_open(at->source)) {
    return std::nullopt;
  }
  // A layout without one chunk per frame has no chunk to report.
  if (!at->source.one_chunk_per_frame) {
    return std::nullopt;
  }
  // A frame the source has not grown to, or not written, may have arrived
  // since the metadata was last read; in SWMR mode one refresh settles it.
  std::optional<ChunkInfo> info = query_chunk(*at, frame);
  if (!info && options_.swmr) {
    refresh(at->source);
    info = query_chunk(*at, frame);
  }
  return info;
}

RawChunk H5ImageSource::read_raw_chunk(size_t frame, std::span<std::byte> dst) {
  if (!raw_chunks_) {
    throw std::runtime_error(fmt::format(
        "{} in {} does not offer raw chunks ({} compression)",
        options_.dataset_path, path_, compression_name(compression())));
  }
  const std::optional<ChunkInfo> info = chunk_info(frame);
  if (!info) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not written", frame, path_));
  }
  if (dst.size() < info->stored_bytes) {
    throw std::runtime_error(
        fmt::format("chunk {} is {} bytes, buffer holds {}", frame,
                    info->stored_bytes, dst.size()));
  }
  const Located at = locate_open(frame);
  const hsize_t offset[3] = {at.local, 0, 0};
  uint32_t filter_mask = 0;
  if (H5Dread_chunk(at.source.dataset_handle, H5P_DEFAULT, offset, &filter_mask,
                    dst.data()) < 0) {
    throw std::runtime_error(fmt::format("cannot read chunk {} of {} in {}",
                                         at.local, at.source.dataset,
                                         at.source.file));
  }
  return RawChunk{dst.first(info->stored_bytes), *info,
                  at.source.decoder->header_bytes(filter_mask)};
}

void H5ImageSource::read_frame_into(size_t frame, FrameView dst) {
  if (dst.type() != pixel_type_ || dst.shape() != shape_) {
    throw std::runtime_error(
        fmt::format("frame view is {}x{} {}, source is {}x{} {}", dst.slow(),
                    dst.fast(), pixel_type_name(dst.type()), shape_[0],
                    shape_[1], pixel_type_name(pixel_type_)));
  }
  if (!raw_chunks_) {
    read_frame_hyperslab(locate_open(frame), dst);
    return;
  }
  // The stored chunk lands in a per-thread buffer that grows to the
  // largest chunk seen, then decodes outside the HDF5 lock.
  thread_local std::vector<std::byte> scratch;
  const std::optional<ChunkInfo> info = chunk_info(frame);
  if (!info) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not written", frame, path_));
  }
  if (scratch.size() < info->stored_bytes) {
    scratch.resize(info->stored_bytes);
  }
  const RawChunk raw = read_raw_chunk(frame, scratch);
  locate_open(frame).source.decoder->decode(raw.bytes, raw.info.filter_mask,
                                            dst.bytes());
}

void H5ImageSource::read_frame_hyperslab(Located at, FrameView dst) {
  if (at.local >= at.source.num_frames.load() && options_.swmr) {
    refresh(at.source);
  }
  if (at.local >= at.source.num_frames.load()) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not written", at.local, at.source.file));
  }
  if (at.source.one_chunk_per_frame) {
    const hsize_t offset[3] = {at.local, 0, 0};
    haddr_t address = HADDR_UNDEF;
    hsize_t size = 0;
    unsigned mask = 0;
    if (H5Dget_chunk_info_by_coord(at.source.dataset_handle, offset, &mask,
                                   &address, &size) >= 0 &&
        (address == HADDR_UNDEF || size == 0)) {
      throw std::runtime_error(fmt::format("frame {} of {} is not written",
                                           at.local, at.source.file));
    }
  }
  // Select the frame in the file and read it into a {slow, fast} memory
  // space of the matching native type.
  H5Space file_space(H5Dget_space(at.source.dataset_handle));
  const hsize_t start[3] = {at.local, 0, 0};
  const hsize_t count[3] = {1, shape_[0], shape_[1]};
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count,
                      nullptr);
  const hsize_t mem_dims[2] = {shape_[0], shape_[1]};
  H5Space mem_space(H5Screate_simple(2, mem_dims, nullptr));
  H5Type mem_type = visit_pixel_type(pixel_type_, [](auto tag) {
    return h5dispatch::H5TypeTraits<typename decltype(tag)::type>::make();
  });
  if (H5Dread(at.source.dataset_handle, mem_type, mem_space, file_space,
              H5P_DEFAULT, dst.bytes().data()) < 0) {
    throw std::runtime_error(fmt::format("cannot read frame {} of {} in {}",
                                         at.local, at.source.dataset,
                                         at.source.file));
  }
}

} // namespace dx2
