#include "dx2/image/chunk_decoder.hpp"
#include <algorithm>
#include <bitshuffle.h>
#include <fmt/format.h>
#include <lz4.h>
#include <stdexcept>

namespace dx2 {

namespace {

/// Text for a negative return code of the bitshuffle library.
std::string bshuf_error(int64_t code) {
  switch (code) {
  case -1:
    return "failed to allocate memory";
  case -11:
    return "missing SSE";
  case -12:
    return "missing AVX";
  case -80:
    return "input size is not a multiple of 8";
  case -81:
    return "block size is not a multiple of 8";
  case -91:
    return "decompression error";
  default:
    return fmt::format("error code {}", code);
  }
}

void check_raw_size(std::span<const std::byte> raw, size_t expected) {
  if (raw.size() != expected) {
    throw std::runtime_error(fmt::format(
        "stored chunk should be {} bytes, got {}", expected, raw.size()));
  }
}

void check_whole_elements(std::span<const std::byte> bytes, size_t elem_size) {
  if (bytes.size() % elem_size != 0) {
    throw std::runtime_error(
        fmt::format("{} bytes is not a whole number of {}-byte elements",
                    bytes.size(), elem_size));
  }
}

/// Reads the header of a compressed chunk and checks it against dst.
std::span<const std::byte> checked_payload(std::span<const std::byte> raw,
                                           std::span<std::byte> dst,
                                           Lz4ContainerHeader &header) {
  header = Lz4ContainerHeader::parse(raw);
  if (header.uncompressed_bytes != dst.size()) {
    throw std::runtime_error(
        fmt::format("chunk header says {} bytes, dataset chunk is {}",
                    header.uncompressed_bytes, dst.size()));
  }
  return raw.subspan(Lz4ContainerHeader::SIZE_BYTES);
}

/// Checks the bytes a bitshuffle decompressor reported against the payload.
void check_consumed(int64_t consumed, size_t payload_bytes,
                    std::string_view what) {
  if (consumed < 0) {
    throw std::runtime_error(
        fmt::format("{} decode failed: {}", what, bshuf_error(consumed)));
  }
  if (static_cast<size_t>(consumed) != payload_bytes) {
    throw std::runtime_error(
        fmt::format("{} decode consumed {} of {} payload bytes", what, consumed,
                    payload_bytes));
  }
}

/// Shared shape of the two bitshuffle compressors: header, then the
/// library's output, trimmed to what it wrote.
template <typename Bound, typename Compress>
std::vector<std::byte>
encode_in_container(std::span<const std::byte> pixels, size_t elem_size,
                    size_t block_elements, Bound bound, Compress compress,
                    std::string_view what) {
  check_whole_elements(pixels, elem_size);
  const size_t num_elements = pixels.size() / elem_size;
  // The filter resolves a zero block size before recording it, so the
  // header always states the size actually used.
  const size_t block = block_elements == 0 ? bshuf_default_block_size(elem_size)
                                           : block_elements;
  const Lz4ContainerHeader header{pixels.size(),
                                  static_cast<uint32_t>(block * elem_size)};
  std::vector<std::byte> out(Lz4ContainerHeader::SIZE_BYTES +
                             bound(num_elements, elem_size, block));
  const int64_t written =
      compress(pixels.data(), out.data() + Lz4ContainerHeader::SIZE_BYTES,
               num_elements, elem_size, block);
  if (written < 0) {
    throw std::runtime_error(
        fmt::format("{} encode failed: {}", what, bshuf_error(written)));
  }
  header.write(out);
  out.resize(Lz4ContainerHeader::SIZE_BYTES + static_cast<size_t>(written));
  return out;
}

uint32_t read_be32(std::span<const std::byte> bytes) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; ++i) {
    value = (value << 8) | std::to_integer<uint32_t>(bytes[i]);
  }
  return value;
}

void write_be32(std::span<std::byte> dst, uint32_t value) {
  for (size_t i = 0; i < 4; ++i) {
    dst[i] = static_cast<std::byte>(value >> (8 * (3 - i)));
  }
}

const char *as_chars(const std::byte *p) {
  return reinterpret_cast<const char *>(p);
}
char *as_chars(std::byte *p) { return reinterpret_cast<char *>(p); }

} // namespace

bool zstd_supported() {
#ifdef ZSTD_SUPPORT
  return true;
#else
  return false;
#endif
}

void IdentityCodec::decode(std::span<const std::byte> raw,
                           std::span<std::byte> dst) const {
  check_raw_size(raw, dst.size());
  std::copy(raw.begin(), raw.end(), dst.begin());
}

std::vector<std::byte>
IdentityCodec::encode(std::span<const std::byte> pixels) const {
  return {pixels.begin(), pixels.end()};
}

void BitshuffleCodec::decode(std::span<const std::byte> raw,
                             std::span<std::byte> dst) const {
  check_raw_size(raw, dst.size());
  check_whole_elements(dst, elem_size);
  const int64_t done =
      bshuf_bitunshuffle(raw.data(), dst.data(), dst.size() / elem_size,
                         elem_size, block_elements);
  if (done < 0) {
    throw std::runtime_error(
        fmt::format("bitshuffle failed: {}", bshuf_error(done)));
  }
}

std::vector<std::byte>
BitshuffleCodec::encode(std::span<const std::byte> pixels) const {
  check_whole_elements(pixels, elem_size);
  std::vector<std::byte> out(pixels.size());
  const int64_t done =
      bshuf_bitshuffle(pixels.data(), out.data(), pixels.size() / elem_size,
                       elem_size, block_elements);
  if (done < 0) {
    throw std::runtime_error(
        fmt::format("bitshuffle failed: {}", bshuf_error(done)));
  }
  return out;
}

void BitshuffleLz4Codec::decode(std::span<const std::byte> raw,
                                std::span<std::byte> dst) const {
  check_whole_elements(dst, elem_size);
  Lz4ContainerHeader header;
  const auto payload = checked_payload(raw, dst, header);
  const int64_t consumed =
      bshuf_decompress_lz4(payload.data(), dst.data(), dst.size() / elem_size,
                           elem_size, header.block_elements(elem_size));
  check_consumed(consumed, payload.size(), "bitshuffle+lz4");
}

std::vector<std::byte>
BitshuffleLz4Codec::encode(std::span<const std::byte> pixels) const {
  return encode_in_container(pixels, elem_size, block_elements,
                             bshuf_compress_lz4_bound, bshuf_compress_lz4,
                             "bitshuffle+lz4");
}

void BitshuffleZstdCodec::decode(std::span<const std::byte> raw,
                                 std::span<std::byte> dst) const {
#ifdef ZSTD_SUPPORT
  check_whole_elements(dst, elem_size);
  Lz4ContainerHeader header;
  const auto payload = checked_payload(raw, dst, header);
  const int64_t consumed =
      bshuf_decompress_zstd(payload.data(), dst.data(), dst.size() / elem_size,
                            elem_size, header.block_elements(elem_size));
  check_consumed(consumed, payload.size(), "bitshuffle+zstd");
#else
  (void)raw;
  (void)dst;
  throw std::runtime_error("bitshuffle+zstd support is not compiled in");
#endif
}

std::vector<std::byte>
BitshuffleZstdCodec::encode(std::span<const std::byte> pixels) const {
#ifdef ZSTD_SUPPORT
  // Level 0 is the filter's default compression level.
  return encode_in_container(
      pixels, elem_size, block_elements, bshuf_compress_zstd_bound,
      [](const void *in, void *out, size_t size, size_t elem, size_t block) {
        return bshuf_compress_zstd(in, out, size, elem, block, 0);
      },
      "bitshuffle+zstd");
#else
  (void)pixels;
  throw std::runtime_error("bitshuffle+zstd support is not compiled in");
#endif
}

void Lz4Codec::decode(std::span<const std::byte> raw,
                      std::span<std::byte> dst) const {
  Lz4ContainerHeader header;
  const auto payload = checked_payload(raw, dst, header);
  // The filter clamps the block size to the chunk, so a header value of 0
  // or larger than the chunk means one block.
  size_t block = header.block_bytes;
  if (block == 0 || block > dst.size()) {
    block = dst.size();
  }

  // Each block is its big-endian stored length then its bytes, compressed
  // unless the length equals the block's decoded size.
  size_t in = 0;
  size_t out = 0;
  while (out < dst.size()) {
    const size_t this_block = std::min(block, dst.size() - out);
    if (payload.size() - in < 4) {
      throw std::runtime_error("lz4 chunk ends inside a block length");
    }
    const uint32_t stored = read_be32(payload.subspan(in));
    in += 4;
    if (stored > payload.size() - in) {
      throw std::runtime_error(fmt::format(
          "lz4 block claims {} bytes, {} remain", stored, payload.size() - in));
    }
    if (stored == this_block) {
      std::copy_n(payload.begin() + in, this_block, dst.begin() + out);
    } else {
      const int decoded = LZ4_decompress_safe(
          as_chars(payload.data() + in), as_chars(dst.data() + out),
          static_cast<int>(stored), static_cast<int>(this_block));
      if (decoded != static_cast<int>(this_block)) {
        throw std::runtime_error(fmt::format(
            "lz4 block decoded to {} bytes, expected {}", decoded, this_block));
      }
    }
    in += stored;
    out += this_block;
  }
  if (in != payload.size()) {
    throw std::runtime_error(fmt::format(
        "lz4 decode consumed {} of {} payload bytes", in, payload.size()));
  }
}

std::vector<std::byte>
Lz4Codec::encode(std::span<const std::byte> pixels) const {
  size_t block = block_bytes;
  if (block == 0 || block > pixels.size()) {
    block = pixels.size();
  }
  const Lz4ContainerHeader header{pixels.size(), static_cast<uint32_t>(block)};
  std::vector<std::byte> out(Lz4ContainerHeader::SIZE_BYTES);
  header.write(out);

  // A block that does not shrink is stored raw, with its length equal to
  // its size, which is how the decoder tells the two apart.
  for (size_t in = 0; in < pixels.size(); in += block) {
    const size_t this_block = std::min(block, pixels.size() - in);
    const size_t start = out.size();
    out.resize(start + 4 + this_block);
    const int compressed = LZ4_compress_default(
        as_chars(pixels.data() + in), as_chars(out.data() + start + 4),
        static_cast<int>(this_block), static_cast<int>(this_block - 1));
    if (compressed > 0) {
      out.resize(start + 4 + static_cast<size_t>(compressed));
      write_be32(std::span(out).subspan(start),
                 static_cast<uint32_t>(compressed));
    } else {
      std::copy_n(pixels.begin() + in, this_block, out.begin() + start + 4);
      write_be32(std::span(out).subspan(start),
                 static_cast<uint32_t>(this_block));
    }
  }
  return out;
}

Codec make_codec(Compression compression, size_t elem_size,
                 size_t block_elements) {
  switch (compression) {
  case Compression::None:
    return IdentityCodec{};
  case Compression::Bitshuffle:
    return BitshuffleCodec{elem_size, block_elements};
  case Compression::BitshuffleLZ4:
    return BitshuffleLz4Codec{elem_size, block_elements};
  case Compression::BitshuffleZstd:
    if (!zstd_supported()) {
      throw std::runtime_error("bitshuffle+zstd support is not compiled in");
    }
    return BitshuffleZstdCodec{elem_size, block_elements};
  case Compression::LZ4:
    return Lz4Codec{block_elements * elem_size};
  case Compression::Deflate:
  case Compression::Unsupported:
    break;
  }
  throw std::runtime_error(
      fmt::format("no codec for {} chunks", compression_name(compression)));
}

std::optional<Codec> make_codec(const FilterPipeline &pipeline,
                                PixelType type) {
  const Compression compression = pipeline.summary();
  const size_t elem_size = pixel_size(type);
  size_t block_elements = 0;
  // Block sizes are the writer's request as the filter recorded it:
  // cd_values[3] in elements for bitshuffle, cd_values[0] in bytes for LZ4.
  if (!pipeline.stages.empty()) {
    const auto &cd = pipeline.stages.front().cd_values;
    if (compression == Compression::LZ4 && !cd.empty()) {
      block_elements = cd[0] / elem_size;
    } else if (cd.size() > 3) {
      block_elements = cd[3];
    }
  }
  // Deflate and unknown pipelines are left to HDF5, as is zstd in a build
  // without it.
  if (compression == Compression::Deflate ||
      compression == Compression::Unsupported ||
      (compression == Compression::BitshuffleZstd && !zstd_supported())) {
    return std::nullopt;
  }
  return make_codec(compression, elem_size, block_elements);
}

bool ChunkDecoder::can_decode(const FilterPipeline &pipeline, PixelType type) {
  return make_codec(pipeline, type).has_value();
}

ChunkDecoder::ChunkDecoder(const FilterPipeline &pipeline, PixelType type,
                           size_t num_pixels)
    : codec_(IdentityCodec{}), type_(type), num_pixels_(num_pixels),
      decoded_bytes_(num_pixels * pixel_size(type)) {
  std::optional<Codec> codec = make_codec(pipeline, type);
  if (!codec) {
    throw std::runtime_error(fmt::format("cannot decode {} chunks ({})",
                                         compression_name(pipeline.summary()),
                                         pipeline.describe()));
  }
  codec_ = *codec;
}

size_t ChunkDecoder::header_bytes(unsigned filter_mask) const {
  if (filter_mask & 1u) {
    return 0;
  }
  return std::visit([](const auto &codec) { return codec.header_bytes(); },
                    codec_);
}

void ChunkDecoder::decode(std::span<const std::byte> raw, unsigned filter_mask,
                          std::span<std::byte> dst) const {
  if (dst.size() != decoded_bytes_) {
    throw std::runtime_error(
        fmt::format("chunk decodes to {} bytes but {} were given",
                    decoded_bytes_, dst.size()));
  }
  // A skipped stage leaves plain pixels in the chunk.
  if (filter_mask & 1u) {
    IdentityCodec{}.decode(raw, dst);
    return;
  }
  std::visit([&](const auto &codec) { codec.decode(raw, dst); }, codec_);
}

std::vector<std::byte> encode_chunk(std::span<const std::byte> pixels,
                                    Compression compression, size_t elem_size,
                                    size_t block_elements) {
  check_whole_elements(pixels, elem_size);
  return std::visit([&](const auto &codec) { return codec.encode(pixels); },
                    make_codec(compression, elem_size, block_elements));
}

} // namespace dx2
