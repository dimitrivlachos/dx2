cmake_minimum_required(VERSION 3.28...3.30)

project(dx2 LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global build-configuration features apply only when dx2 is the root project.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
    include(SetDefaultBuildRelWithDebInfo)
    include(AlwaysColourCompilation)
    include(CheckCondaPins)
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(DX2_BUILD_TESTS "Build the dx2 test suite" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)

# #######################################################################
# External Dependencies
find_package(HDF5 REQUIRED)
find_package(gemmi CONFIG REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(fmt 11.0 REQUIRED)
find_package(Eigen3 REQUIRED)

# zstd is optional and enables the bitshuffle+zstd variant.
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)
find_package(zstd CONFIG QUIET)

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# A fetched mdspan installs its headers and package alongside dx2, which the
# installed dx2 headers include.
find_package(mdspan QUIET)
if(NOT mdspan_FOUND AND NOT TARGET std::mdspan)
    FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan
        GIT_TAG mdspan-0.6.0
    )
    FetchContent_MakeAvailable(mdspan)
endif()

if(DX2_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND AND NOT TARGET GTest::gtest_main)
        FetchContent_Declare(
            GTest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
            EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(GTest)
    endif()
endif()

FetchContent_Declare(
    bitshuffle
    GIT_REPOSITORY https://github.com/kiyo-masui/bitshuffle
    GIT_TAG 52aec3b80d05606c090956aecfe868489d96b95c # 0.5.2
    GIT_SUBMODULES ""
    SOURCE_SUBDIR no-cmake-project
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(bitshuffle)

# #######################################################################
# Targets
add_subdirectory(dx2)

if(DX2_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# #######################################################################
# Installation
set(DX2_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/DX2)

install(
    TARGETS dx2
    EXPORT DX2Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)

install(
    DIRECTORY include/dx2/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dx2
    COMPONENT Runtime
    FILES_MATCHING PATTERN "*.hpp"
)

install(
    EXPORT DX2Targets
    FILE DX2Targets.cmake
    NAMESPACE dx2::
    DESTINATION ${DX2_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/DX2Config.cmake"
    INSTALL_DESTINATION ${DX2_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DX2ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DX2Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/DX2ConfigVersion.cmake
    DESTINATION ${DX2_INSTALL_CMAKEDIR}
)
