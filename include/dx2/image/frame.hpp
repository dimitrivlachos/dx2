/**
 * @file frame.hpp
 * @brief Typed views over a block of pixel bytes.
 *
 * A frame is a 2-D image whose element type is known only at run time. It
 * travels as bytes plus a PixelType and a {slow, fast} shape, so the same
 * view can describe a buffer the caller owns, such as pinned host memory
 * for a GPU upload or a slot in a queue, or one allocated here as a Frame.
 * Typed access is explicit and checked: as<T>() and as_mdspan<T>() verify
 * the pixel type and the alignment before reinterpreting, and visit()
 * dispatches on the PixelType so generic code is written once as a
 * template over the element type.
 *
 * A view holds exactly one frame's worth of bytes; the constructor rejects
 * any other size, so every view that exists is consistent. The bytes are
 * reinterpreted as T rather than created as T, which is how decoders and
 * HDF5 produce them. Storage from operator new is aligned for every
 * PixelType; a caller-supplied buffer must be too, and as<T>() rejects one
 * that is not.
 */

#pragma once

#include "dx2/image/pixel_type.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <fmt/core.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace dx2 {

/// Row-major 2-D view with run-time extents {slow, fast}.
template <typename T>
using mdspan_2d =
    std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

/// Image extents as {slow, fast}: slow is the row count, fast the row length.
using FrameShape = std::array<size_t, 2>;

/**
 * @brief Non-owning view of one frame as bytes with a PixelType and shape.
 *
 * @tparam Byte std::byte for a writable view, const std::byte for a
 * read-only one. A writable view converts implicitly to a read-only one.
 */
template <typename Byte> class BasicFrameView {
  static_assert(std::is_same_v<std::remove_const_t<Byte>, std::byte>,
                "Byte must be std::byte or const std::byte");

public:
  /// T with the view's constness applied.
  template <typename T>
  using element_t = std::conditional_t<std::is_const_v<Byte>, const T, T>;

  BasicFrameView() = default;

  /**
   * @param bytes Exactly shape[0] * shape[1] * pixel_size(type) bytes.
   * @throws std::invalid_argument if the byte count does not match.
   */
  BasicFrameView(std::span<Byte> bytes, PixelType type, FrameShape shape)
      : bytes_(bytes), type_(type), shape_(shape) {
    const size_t expected = shape[0] * shape[1] * pixel_size(type);
    if (bytes.size() != expected) {
      throw std::invalid_argument(fmt::format(
          "frame view of {}x{} {} pixels needs {} bytes, got {}", shape[0],
          shape[1], pixel_type_name(type), expected, bytes.size()));
    }
  }

  /// A read-only view of a writable one. A template, so that it is never
  /// the copy constructor and the implicit one stays available.
  template <typename OtherByte>
    requires(std::is_const_v<Byte> && std::is_same_v<OtherByte, std::byte>)
  BasicFrameView(const BasicFrameView<OtherByte> &other)
      : bytes_(other.bytes()), type_(other.type()), shape_(other.shape()) {}

  std::span<Byte> bytes() const { return bytes_; }
  PixelType type() const { return type_; }
  FrameShape shape() const { return shape_; }
  size_t slow() const { return shape_[0]; }
  size_t fast() const { return shape_[1]; }
  size_t num_pixels() const { return shape_[0] * shape_[1]; }

  /**
   * @brief The pixels as a flat span of T, row-major.
   * @throws std::runtime_error if T is not the view's pixel type or the
   * buffer is not aligned for T.
   */
  template <typename T> std::span<element_t<T>> as() const {
    if (pixel_type_of_v<T> != type_) {
      throw std::runtime_error(
          fmt::format("frame holds {} pixels, not {}", pixel_type_name(type_),
                      pixel_type_name(pixel_type_of_v<T>)));
    }
    if (reinterpret_cast<std::uintptr_t>(bytes_.data()) % alignof(T) != 0) {
      throw std::runtime_error(fmt::format(
          "frame buffer is not aligned for {} pixels", pixel_type_name(type_)));
    }
    return {reinterpret_cast<element_t<T> *>(bytes_.data()), num_pixels()};
  }

  /// The pixels as a 2-D mdspan indexed (slow, fast). Checks as for as<T>().
  template <typename T> mdspan_2d<element_t<T>> as_mdspan() const {
    auto flat = as<T>();
    return mdspan_2d<element_t<T>>(flat.data(), shape_[0], shape_[1]);
  }

private:
  std::span<Byte> bytes_{};
  PixelType type_{};
  FrameShape shape_{};
};

using FrameView = BasicFrameView<std::byte>;
using ConstFrameView = BasicFrameView<const std::byte>;

/**
 * @brief Calls f(mdspan_2d<T>) with the view's pixels typed.
 *
 * f must return the same type for every T in PixelTypes, or void.
 */
template <typename Byte, typename F>
decltype(auto) visit(BasicFrameView<Byte> view, F &&f) {
  return visit_pixel_type(view.type(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    return f(view.template as_mdspan<T>());
  });
}

/// std::variant over mdspan_2d<T> for every pixel type.
using FrameVariant = h5dispatch::variant_of_t<mdspan_2d, PixelTypes>;

namespace detail {
template <typename T> using const_mdspan_2d = mdspan_2d<const T>;
}

/// std::variant over mdspan_2d<const T> for every pixel type.
using ConstFrameVariant =
    h5dispatch::variant_of_t<detail::const_mdspan_2d, PixelTypes>;

/// The view as a variant, for callers who prefer std::visit.
inline FrameVariant as_variant(FrameView view) {
  return visit(view, [](auto md) -> FrameVariant { return md; });
}

/// The view as a variant, for callers who prefer std::visit.
inline ConstFrameVariant as_variant(ConstFrameView view) {
  return visit(view, [](auto md) -> ConstFrameVariant { return md; });
}

/**
 * @brief An owning frame: storage sized for its type and shape.
 *
 * Storage is left uninitialised, since every use fills it by decoding or
 * reading into it. Move-only.
 */
class Frame {
public:
  Frame() = default;
  Frame(PixelType type, FrameShape shape)
      : type_(type), shape_(shape),
        size_(shape[0] * shape[1] * pixel_size(type)),
        storage_(std::make_unique_for_overwrite<std::byte[]>(size_)) {}
  /// Adopts storage of at least the frame's size, so a buffer can be
  /// reused across frames without being zeroed or faulted in again.
  Frame(std::unique_ptr<std::byte[]> storage, PixelType type, FrameShape shape)
      : type_(type), shape_(shape),
        size_(shape[0] * shape[1] * pixel_size(type)),
        storage_(std::move(storage)) {}

  PixelType type() const { return type_; }
  FrameShape shape() const { return shape_; }
  size_t size_bytes() const { return size_; }

  std::span<std::byte> bytes() { return {storage_.get(), size_}; }
  std::span<const std::byte> bytes() const { return {storage_.get(), size_}; }

  FrameView view() { return FrameView(bytes(), type_, shape_); }
  ConstFrameView view() const { return ConstFrameView(bytes(), type_, shape_); }

  /// Gives up the storage, leaving an empty frame.
  std::unique_ptr<std::byte[]> release() {
    size_ = 0;
    shape_ = {};
    return std::move(storage_);
  }

private:
  PixelType type_{};
  FrameShape shape_{};
  size_t size_{0};
  std::unique_ptr<std::byte[]> storage_{};
};

} // namespace dx2
