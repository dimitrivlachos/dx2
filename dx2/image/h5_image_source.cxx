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

  // Every source must agree with the first on shape, type and pipeline;
  // raw chunks need every source to offer them.
  const Source &first = sources_.front();
  {
    H5Space space(H5Dget_space(first.dataset_handle));
    hsize_t dims[3] = {};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    shape_ = {dims[1], dims[2]};
  }
  pixel_type_ = pixel_type_from_h5(first.dataset_handle);
  raw_chunks_ = true;
  for (const Source &source : sources_) {
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
      throw std::runtime_error(
          fmt::format("{} in {} is {} compressed, {} in {} is {}",
                      source.dataset, source.file,
                      compression_name(source.filters.summary()), first.dataset,
                      first.file, compression_name(first.filters.summary())));
    }
    raw_chunks_ =
        raw_chunks_ && source.one_chunk_per_frame && source.decoder.has_value();
  }
  for (const Segment &segment : segments_) {
    num_frames_ = std::max(num_frames_, segment.first + segment.count);
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
  if (!target.file_exists) {
    throw std::runtime_error(fmt::format("{} in {} points at {}, which does "
                                         "not exist",
                                         options_.dataset_path, path_,
                                         target.file));
  }
  // Adding sources reallocates sources_, so the dataset is referred to by
  // index and its handle and path are copied out first.
  const size_t index = add_source(target.file, target.object_path);
  const hid_t dataset = sources_[index].dataset_handle;
  const fs::path dataset_file = sources_[index].file;
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
      segments_.push_back({0, sources_[index].num_frames, index, 0});
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
  segments_.push_back({0, sources_[index].num_frames, index, 0});
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
    // Both selections must be whole frames: the virtual side says which
    // frames, the source side says where they come from.
    H5Space vspace(H5Pget_virtual_vspace(dcpl, i));
    H5Space sspace(H5Pget_virtual_srcspace(dcpl, i));
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

    // The source name may itself be a link chain inside the source file.
    const h5link::ResolvedLink found =
        h5link::find_linked_file(file_name, vds_file, vds_prefixes);
    if (!found.file_exists) {
      throw std::runtime_error(fmt::format(
          "virtual source {} of {} does not exist", found.file, vds_file));
    }
    h5utils::H5File source_file(
        H5Fopen(found.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!source_file) {
      throw std::runtime_error(fmt::format("cannot open {}", found.file));
    }
    const h5link::ResolvedLink target = h5link::resolve_link(
        source_file, found.file, dataset_name, elink_prefixes);
    if (!target.file_exists) {
      throw std::runtime_error(
          fmt::format("{} in {} points at {}, which does not exist",
                      dataset_name, found.file, target.file));
    }
    const size_t index = add_source(target.file, target.object_path);
    const size_t source_frames = sources_[index].num_frames;

    hsize_t sdims[3] = {};
    {
      H5Space space(H5Dget_space(sources_[index].dataset_handle));
      H5Sget_simple_extent_dims(space, sdims, nullptr);
    }
    const std::optional<FrameRun> source_run = whole_frame_run(sspace, sdims);
    if (!source_run) {
      throw UnsupportedMapping("virtual source selection is not whole frames");
    }
    // An unlimited run takes every frame the source has from its start.
    const size_t frames =
        virtual_run->count
            ? static_cast<size_t>(*virtual_run->count)
            : source_frames -
                  std::min<size_t>(source_frames, source_run->first);
    if (frames == 0) {
      continue;
    }
    segments_.push_back({static_cast<size_t>(virtual_run->first), frames, index,
                         static_cast<size_t>(source_run->first)});
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
    const size_t index = add_source(target.file, target.object_path);
    const size_t frames = sources_[index].num_frames;
    segments_.push_back({next_frame, frames, index, 0});
    next_frame += frames;
  }
}

size_t H5ImageSource::add_source(const fs::path &file,
                                 const std::string &dataset) {
  // The same dataset reached twice is one source.
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sources_[i].file == file && sources_[i].dataset == dataset) {
      return i;
    }
  }
  Source source;
  source.file = file;
  source.dataset = dataset;
  unsigned flags = H5F_ACC_RDONLY;
  if (options_.swmr) {
    flags |= H5F_ACC_SWMR_READ;
  }
  source.file_handle =
      h5utils::H5File(H5Fopen(file.c_str(), flags, H5P_DEFAULT));
  if (!source.file_handle) {
    throw std::runtime_error(fmt::format("cannot open {}", file));
  }
  source.dataset_handle = h5utils::H5Dataset(
      H5Dopen2(source.file_handle, dataset.c_str(), H5P_DEFAULT));
  if (!source.dataset_handle) {
    throw std::runtime_error(fmt::format("no dataset {} in {}", dataset, file));
  }

  // The dataset must be {frames, slow, fast}; raw chunks need one chunk
  // per frame and a codec for the pipeline.
  H5Space space(H5Dget_space(source.dataset_handle));
  if (!space || H5Sget_simple_extent_ndims(space) != 3) {
    throw std::runtime_error(
        fmt::format("{} in {} is not a 3-D dataset", dataset, file));
  }
  hsize_t dims[3] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  source.num_frames = dims[0];
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
  sources_.push_back(std::move(source));
  return sources_.size() - 1;
}

std::vector<H5ImageSource::SourceInfo> H5ImageSource::sources() const {
  std::vector<SourceInfo> info;
  info.reserve(sources_.size());
  for (const Source &source : sources_) {
    info.push_back({source.file, source.dataset, source.num_frames});
  }
  return info;
}

H5ImageSource::Located H5ImageSource::locate(size_t frame) {
  if (frame >= num_frames_) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is beyond the {} in {}", frame,
                    options_.dataset_path, num_frames_, path_));
  }
  // Segments are sorted by first frame; the last one starting at or
  // before the frame holds it, unless the frame falls in a gap.
  auto after =
      std::upper_bound(segments_.begin(), segments_.end(), frame,
                       [](size_t f, const Segment &s) { return f < s.first; });
  if (after == segments_.begin()) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not mapped", frame, path_));
  }
  const Segment &segment = *(after - 1);
  if (frame >= segment.first + segment.count) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not mapped", frame, path_));
  }
  return {sources_[segment.source],
          segment.source_first + (frame - segment.first)};
}

size_t H5ImageSource::max_raw_chunk_bytes() const {
  size_t bound = 0;
  for (const Source &source : sources_) {
    bound = std::max(bound, source.decoder ? std::visit(
                                                 [&](const auto &codec) {
                                                   return codec.encoded_bound(
                                                       frame_bytes());
                                                 },
                                                 source.decoder->codec())
                                           : frame_bytes());
  }
  return bound;
}

std::optional<ChunkInfo> H5ImageSource::chunk_info(size_t frame) {
  if (frame >= num_frames_) {
    return std::nullopt;
  }
  const Located at = locate(frame);
  // A frame the source has not grown to yet, or a layout without one
  // chunk per frame, has no chunk to report.
  if (!at.source.one_chunk_per_frame || at.local >= at.source.num_frames) {
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
  const Located at = locate(frame);
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
    read_frame_hyperslab(locate(frame), dst);
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
  locate(frame).source.decoder->decode(raw.bytes, raw.info.filter_mask,
                                       dst.bytes());
}

void H5ImageSource::read_frame_hyperslab(Located at, FrameView dst) {
  if (at.local >= at.source.num_frames) {
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
