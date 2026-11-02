/**
 * @file pixel_type.hpp
 * @brief Runtime description of a detector image's element type.
 *
 * Detector files store pixels as unsigned 16- or 32-bit counts, and other
 * sources use signed integers or floating point. A reader learns the type
 * from the file at run time, so it travels as the PixelType enum rather
 * than as a template parameter. PixelTypes is the same set as a type_list
 * in the same order, and pixel_type_of_v and visit_pixel_type convert
 * between a C++ type and the enum in both directions. Code that needs a
 * concrete type, such as a decoder or a kernel launch, dispatches once
 * through visit_pixel_type and is otherwise an ordinary template.
 *
 * Adding a type means appending it to the enum, to PixelTypes and to the
 * name table at the same position, and giving it an
 * h5dispatch::H5TypeTraits specialisation.
 */

#pragma once

#include "dx2/h5/h5dispatch.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace dx2 {

/// Element type of an image, as stored in the file.
enum class PixelType : uint8_t {
  UInt8,
  UInt16,
  UInt32,
  Int8,
  Int16,
  Int32,
  Float32,
  Float64,
};

/// The C++ type behind each PixelType, in enum order.
using PixelTypes = h5dispatch::type_list<uint8_t, uint16_t, uint32_t, int8_t,
                                         int16_t, int32_t, float, double>;

namespace detail {

template <typename List> struct type_list_size;
template <typename... Ts>
struct type_list_size<h5dispatch::type_list<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)> {};

/// Position of T in List. A type that is absent fails to compile.
template <typename T, typename List> struct type_list_index;
template <typename T, typename... Ts>
struct type_list_index<T, h5dispatch::type_list<T, Ts...>>
    : std::integral_constant<size_t, 0> {};
template <typename T, typename U, typename... Ts>
struct type_list_index<T, h5dispatch::type_list<U, Ts...>>
    : std::integral_constant<
          size_t, 1 + type_list_index<T, h5dispatch::type_list<Ts...>>::value> {
};

template <typename... Ts>
constexpr std::array<size_t, sizeof...(Ts)>
sizes_of(h5dispatch::type_list<Ts...>) {
  return {sizeof(Ts)...};
}

/// Calls f(type_tag<T>) for the index-th T of the list.
template <typename F, typename T, typename... Rest>
decltype(auto) visit_list(size_t index, F &&f,
                          h5dispatch::type_list<T, Rest...>) {
  if constexpr (sizeof...(Rest) == 0) {
    return f(h5dispatch::type_tag<T>);
  } else {
    if (index == 0) {
      return f(h5dispatch::type_tag<T>);
    }
    return visit_list(index - 1, std::forward<F>(f),
                      h5dispatch::type_list<Rest...>{});
  }
}

} // namespace detail

/// Number of supported pixel types.
inline constexpr size_t kNumPixelTypes =
    detail::type_list_size<PixelTypes>::value;

/// The PixelType for a C++ type in PixelTypes.
template <typename T>
inline constexpr PixelType pixel_type_of_v =
    static_cast<PixelType>(detail::type_list_index<T, PixelTypes>::value);

static_assert(pixel_type_of_v<uint8_t> == PixelType::UInt8);
static_assert(pixel_type_of_v<uint16_t> == PixelType::UInt16);
static_assert(pixel_type_of_v<uint32_t> == PixelType::UInt32);
static_assert(pixel_type_of_v<int8_t> == PixelType::Int8);
static_assert(pixel_type_of_v<int16_t> == PixelType::Int16);
static_assert(pixel_type_of_v<int32_t> == PixelType::Int32);
static_assert(pixel_type_of_v<float> == PixelType::Float32);
static_assert(pixel_type_of_v<double> == PixelType::Float64);

namespace detail {

inline constexpr auto kPixelSizes = sizes_of(PixelTypes{});

inline constexpr std::array<std::string_view, kNumPixelTypes> kPixelTypeNames{
    "uint8", "uint16", "uint32",  "int8",
    "int16", "int32",  "float32", "float64"};

inline size_t checked_index(PixelType type) {
  const auto index = static_cast<size_t>(type);
  if (index >= kNumPixelTypes) {
    throw std::invalid_argument("PixelType value " + std::to_string(index) +
                                " is out of range");
  }
  return index;
}

} // namespace detail

/**
 * @brief Size in bytes of one pixel of the given type.
 * @throws std::invalid_argument for a value outside the enum.
 */
inline size_t pixel_size(PixelType type) {
  return detail::kPixelSizes[detail::checked_index(type)];
}

/**
 * @brief Short lower-case name of the type, for messages.
 * @throws std::invalid_argument for a value outside the enum.
 */
inline std::string_view pixel_type_name(PixelType type) {
  return detail::kPixelTypeNames[detail::checked_index(type)];
}

/**
 * @brief Calls f(type_tag<T>) with the C++ type behind a PixelType.
 *
 * f must return the same type for every T in PixelTypes, or void.
 *
 * @throws std::invalid_argument for a value outside the enum.
 */
template <typename F> decltype(auto) visit_pixel_type(PixelType type, F &&f) {
  return detail::visit_list(detail::checked_index(type), std::forward<F>(f),
                            PixelTypes{});
}

/**
 * @brief The PixelType of an open HDF5 dataset.
 * @throws std::runtime_error if the dataset's type is not a pixel type.
 */
inline PixelType pixel_type_from_h5(hid_t dataset_id) {
  PixelType result{};
  h5dispatch::dispatch_h5_dataset_type<PixelTypes>(dataset_id, [&](auto tag) {
    result = pixel_type_of_v<typename decltype(tag)::type>;
  });
  return result;
}

} // namespace dx2
