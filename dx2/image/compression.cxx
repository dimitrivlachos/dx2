#include "dx2/image/compression.hpp"
#include "dx2/h5/h5utils.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <stdexcept>

namespace dx2 {

std::string_view compression_name(Compression compression) {
  switch (compression) {
  case Compression::None:
    return "none";
  case Compression::Bitshuffle:
    return "bitshuffle";
  case Compression::BitshuffleLZ4:
    return "bitshuffle+lz4";
  case Compression::BitshuffleZstd:
    return "bitshuffle+zstd";
  case Compression::LZ4:
    return "lz4";
  case Compression::Deflate:
    return "deflate";
  case Compression::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

FilterPipeline FilterPipeline::from_plist(hid_t dcpl) {
  FilterPipeline pipeline;
  const int count = H5Pget_nfilters(dcpl);
  if (count < 0) {
    throw std::runtime_error("Failed to read the filter pipeline");
  }
  for (int i = 0; i < count; ++i) {
    // First call with no room for parameters reports how many there are;
    // the second call fills them in.
    FilterStage stage;
    size_t cd_nelmts = 0;
    char name[256] = {};
    unsigned config = 0;
    stage.id = H5Pget_filter2(dcpl, static_cast<unsigned>(i), &stage.flags,
                              &cd_nelmts, nullptr, 0, nullptr, &config);
    if (stage.id < 0) {
      throw std::runtime_error(
          fmt::format("Failed to read filter {} of the pipeline", i));
    }
    stage.cd_values.resize(cd_nelmts);
    if (H5Pget_filter2(dcpl, static_cast<unsigned>(i), &stage.flags, &cd_nelmts,
                       stage.cd_values.data(), sizeof(name), name,
                       &config) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to read filter {} of the pipeline", i));
    }
    stage.name = name;
    pipeline.stages.push_back(std::move(stage));
  }
  return pipeline;
}

FilterPipeline FilterPipeline::from_dataset(hid_t dataset) {
  h5utils::H5Plist dcpl(H5Dget_create_plist(dataset));
  if (!dcpl) {
    throw std::runtime_error(
        "Failed to get the dataset's creation property list");
  }
  return from_plist(dcpl);
}

Compression FilterPipeline::summary() const {
  if (stages.empty()) {
    return Compression::None;
  }
  if (stages.size() != 1) {
    return Compression::Unsupported;
  }
  const FilterStage &stage = stages.front();
  if (stage.id == bitshuffle_filter_id) {
    // cd_values are {major, minor, elem_size, block_size, compressor,
    // [level]}; a short list means the filter's defaults, no compressor.
    const unsigned compressor =
        stage.cd_values.size() > 4 ? stage.cd_values[4] : 0;
    switch (static_cast<BitshuffleCompressor>(compressor)) {
    case BitshuffleCompressor::None:
      return Compression::Bitshuffle;
    case BitshuffleCompressor::LZ4:
      return Compression::BitshuffleLZ4;
    case BitshuffleCompressor::Zstd:
      return Compression::BitshuffleZstd;
    }
    return Compression::Unsupported;
  }
  if (stage.id == lz4_filter_id) {
    return Compression::LZ4;
  }
  if (stage.id == H5Z_FILTER_DEFLATE) {
    return Compression::Deflate;
  }
  return Compression::Unsupported;
}

std::string FilterPipeline::describe() const {
  if (stages.empty()) {
    return "no filters";
  }
  std::string text;
  for (const FilterStage &stage : stages) {
    if (!text.empty()) {
      text += " -> ";
    }
    text += fmt::format("{}({}){}", stage.name, stage.id,
                        fmt::format("[{}]", fmt::join(stage.cd_values, ",")));
  }
  return text;
}

Lz4ContainerHeader Lz4ContainerHeader::parse(std::span<const std::byte> chunk) {
  if (chunk.size() < SIZE_BYTES) {
    throw std::runtime_error(
        fmt::format("LZ4 container header needs {} bytes, got {}", SIZE_BYTES,
                    chunk.size()));
  }
  Lz4ContainerHeader header;
  for (size_t i = 0; i < 8; ++i) {
    header.uncompressed_bytes =
        (header.uncompressed_bytes << 8) | std::to_integer<uint64_t>(chunk[i]);
  }
  for (size_t i = 8; i < SIZE_BYTES; ++i) {
    header.block_bytes =
        (header.block_bytes << 8) | std::to_integer<uint32_t>(chunk[i]);
  }
  return header;
}

void Lz4ContainerHeader::write(std::span<std::byte> dst) const {
  if (dst.size() < SIZE_BYTES) {
    throw std::runtime_error(fmt::format(
        "LZ4 container header needs {} bytes, got {}", SIZE_BYTES, dst.size()));
  }
  for (size_t i = 0; i < 8; ++i) {
    dst[i] = static_cast<std::byte>(uncompressed_bytes >> (8 * (7 - i)));
  }
  for (size_t i = 0; i < 4; ++i) {
    dst[8 + i] = static_cast<std::byte>(block_bytes >> (8 * (3 - i)));
  }
}

} // namespace dx2
