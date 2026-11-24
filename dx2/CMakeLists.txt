add_library(
    dx2
    utils.cxx
    reflection.cxx
    h5/h5read_processed.cxx
    h5/h5write.cxx
    beam.cxx
    crystal.cxx
    detector.cxx
    goniometer.cxx
    scan.cxx
    detector_attenuations.cxx
    imagesequence.cxx
    experiment.cxx
    experiment_list.cxx
    beam_ops.cxx
    beam_io.cxx
    image/compression.cxx
    image/chunk_decoder.cxx
    image/image_source.cxx
    image/h5_image_source.cxx
    h5/h5link.cxx
)
add_library(dx2::dx2 ALIAS dx2)

# A static dx2 is linked into Python extension modules downstream.
set_target_properties(dx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(
    dx2
    PUBLIC
    Eigen3::Eigen
    fmt::fmt
    nlohmann_json::nlohmann_json
    hdf5::hdf5
    gemmi::gemmi_cpp
    std::mdspan
    PRIVATE
    PkgConfig::LZ4
)

target_include_directories(
    dx2 PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Bitshuffle core, compiled into dx2 from the fetched upstream sources. It
# selects its SIMD kernels from the compiler's predefined macros (__AVX2__,
# __SSE2__, __ARM_NEON); empty flags mean the target's ISA baseline.
set(DX2_BITSHUFFLE_SIMD_FLAGS "" CACHE STRING
    "Extra compile flags for the bitshuffle core, e.g. -mavx2 or -march=native")
set(DX2_BITSHUFFLE_SOURCES
    ${bitshuffle_SOURCE_DIR}/src/bitshuffle.c
    ${bitshuffle_SOURCE_DIR}/src/bitshuffle_core.c
    ${bitshuffle_SOURCE_DIR}/src/iochain.c
)
target_sources(dx2 PRIVATE ${DX2_BITSHUFFLE_SOURCES})
set_source_files_properties(
    ${DX2_BITSHUFFLE_SOURCES}
    PROPERTIES COMPILE_OPTIONS "${DX2_BITSHUFFLE_SIMD_FLAGS}"
)
target_include_directories(dx2 PRIVATE ${bitshuffle_SOURCE_DIR}/src)

# bitshuffle.h gates the zstd entry points on ZSTD_SUPPORT.
if(zstd_FOUND)
    target_compile_definitions(dx2 PRIVATE ZSTD_SUPPORT)
    if(TARGET zstd::libzstd)
        target_link_libraries(dx2 PRIVATE zstd::libzstd)
    else()
        target_link_libraries(dx2 PRIVATE zstd::libzstd_shared)
    endif()
endif()
