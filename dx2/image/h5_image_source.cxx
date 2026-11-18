#include "dx2/image/h5_image_source.hpp"
#include <fmt/format.h>
#include <fmt/std.h>
#include <stdexcept>
#include <vector>

namespace dx2 {

using h5utils::H5Plist;
using h5utils::H5Space;
using h5utils::H5Type;

H5ImageSource::H5ImageSource(const std::filesystem::path &path, Options options)
    : path_(path), options_(std::move(options)) {
  // Open the file read-only, in SWMR mode when asked, and the dataset.
  unsigned flags = H5F_ACC_RDONLY;
  if (options_.swmr) {
    flags |= H5F_ACC_SWMR_READ;
  }
  file_ = h5utils::H5File(H5Fopen(path_.c_str(), flags, H5P_DEFAULT));
  if (!file_) {
    throw std::runtime_error(fmt::format("cannot open {}", path_));
  }
  dataset_ = h5utils::H5Dataset(
      H5Dopen2(file_, options_.dataset_path.c_str(), H5P_DEFAULT));
  if (!dataset_) {
    throw std::runtime_error(
        fmt::format("no dataset {} in {}", options_.dataset_path, path_));
  }

  // The dataset must be {frames, slow, fast}.
  H5Space space(H5Dget_space(dataset_));
  if (!space || H5Sget_simple_extent_ndims(space) != 3) {
    throw std::runtime_error(fmt::format("{} in {} is not a 3-D dataset",
                                         options_.dataset_path, path_));
  }
  hsize_t dims[3] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  num_frames_ = dims[0];
  shape_ = {dims[1], dims[2]};
  pixel_type_ = pixel_type_from_h5(dataset_);

  // Raw chunks need one chunk per frame; anything else goes through
  // H5Dread.
  H5Plist dcpl(H5Dget_create_plist(dataset_));
  filters_ = FilterPipeline::from_plist(dcpl);
  if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
    hsize_t chunk[3] = {};
    if (H5Pget_chunk(dcpl, 3, chunk) == 3) {
      one_chunk_per_frame_ =
          chunk[0] == 1 && chunk[1] == dims[1] && chunk[2] == dims[2];
    }
  }
  if (one_chunk_per_frame_ && ChunkDecoder::can_decode(filters_, pixel_type_)) {
    decoder_.emplace(filters_, pixel_type_, shape_[0] * shape_[1]);
  }
}

size_t H5ImageSource::max_raw_chunk_bytes() const {
  if (!decoder_) {
    return frame_bytes();
  }
  return std::visit(
      [&](const auto &codec) { return codec.encoded_bound(frame_bytes()); },
      decoder_->codec());
}

std::optional<ChunkInfo> H5ImageSource::chunk_info(size_t frame) {
  if (!one_chunk_per_frame_ || frame >= num_frames_) {
    return std::nullopt;
  }
  // An allocated chunk has an address and a size; an unwritten one
  // reports HADDR_UNDEF and 0 without error.
  const hsize_t offset[3] = {frame, 0, 0};
  unsigned filter_mask = 0;
  haddr_t address = HADDR_UNDEF;
  hsize_t size = 0;
  if (H5Dget_chunk_info_by_coord(dataset_, offset, &filter_mask, &address,
                                 &size) < 0) {
    throw std::runtime_error(fmt::format("cannot query chunk {} of {} in {}",
                                         frame, options_.dataset_path, path_));
  }
  if (address == HADDR_UNDEF || size == 0) {
    return std::nullopt;
  }
  return ChunkInfo{frame, filter_mask, static_cast<uint64_t>(address),
                   static_cast<size_t>(size)};
}

RawChunk H5ImageSource::read_raw_chunk(size_t frame, std::span<std::byte> dst) {
  if (!supports_raw_chunks()) {
    throw std::runtime_error(fmt::format(
        "{} in {} does not offer raw chunks ({} layout, {} filters)",
        options_.dataset_path, path_,
        one_chunk_per_frame_ ? "one chunk per frame" : "other",
        compression_name(compression())));
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
  const hsize_t offset[3] = {frame, 0, 0};
  uint32_t filter_mask = 0;
  if (H5Dread_chunk(dataset_, H5P_DEFAULT, offset, &filter_mask, dst.data()) <
      0) {
    throw std::runtime_error(
        fmt::format("cannot read chunk {} of {}", frame, path_));
  }
  return RawChunk{dst.first(info->stored_bytes), *info,
                  decoder_->header_bytes(filter_mask)};
}

void H5ImageSource::read_frame_into(size_t frame, FrameView dst) {
  if (dst.type() != pixel_type_ || dst.shape() != shape_) {
    throw std::runtime_error(
        fmt::format("frame view is {}x{} {}, source is {}x{} {}", dst.slow(),
                    dst.fast(), pixel_type_name(dst.type()), shape_[0],
                    shape_[1], pixel_type_name(pixel_type_)));
  }
  if (!supports_raw_chunks()) {
    read_frame_hyperslab(frame, dst);
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
  decoder_->decode(raw.bytes, raw.info.filter_mask, dst.bytes());
}

void H5ImageSource::read_frame_hyperslab(size_t frame, FrameView dst) {
  if (frame >= num_frames_) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is beyond the {} in {}", frame,
                    options_.dataset_path, num_frames_, path_));
  }
  if (one_chunk_per_frame_ && !chunk_info(frame)) {
    throw std::runtime_error(
        fmt::format("frame {} of {} is not written", frame, path_));
  }
  // Select the frame in the file and read it into a {slow, fast} memory
  // space of the matching native type.
  H5Space file_space(H5Dget_space(dataset_));
  const hsize_t start[3] = {frame, 0, 0};
  const hsize_t count[3] = {1, shape_[0], shape_[1]};
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count,
                      nullptr);
  const hsize_t mem_dims[2] = {shape_[0], shape_[1]};
  H5Space mem_space(H5Screate_simple(2, mem_dims, nullptr));
  H5Type mem_type = visit_pixel_type(pixel_type_, [](auto tag) {
    return h5dispatch::H5TypeTraits<typename decltype(tag)::type>::make();
  });
  if (H5Dread(dataset_, mem_type, mem_space, file_space, H5P_DEFAULT,
              dst.bytes().data()) < 0) {
    throw std::runtime_error(
        fmt::format("cannot read frame {} of {}", frame, path_));
  }
}

} // namespace dx2
