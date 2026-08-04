cmake_minimum_required(VERSION 3.20)
project(tm30 VERSION 0.1.0 LANGUAGES CXX)

# ── C++20 ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Catch2 v3 via FetchContent ──────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ── Data directory path (for test file loading) ─────────────────────────
set(TM30_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(TM30_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")

# ── Threads (std::thread in tm30.cpp batch path) ─────────────────────────
find_package(Threads REQUIRED)

# ── Core library ─────────────────────────────────────────────────────────
add_library(tm30-core STATIC
  src/tm30/version.cpp
  src/tm30/spd.cpp
  src/tm30/csv_loader.cpp
  src/tm30/resample.cpp
  src/tm30/integrate.cpp
  src/tm30/xyz.cpp
  src/tm30/chromaticity.cpp
  src/tm30/cct.cpp
  src/tm30/reference.cpp
  src/tm30/pipeline.cpp
  src/tm30/ciecam02.cpp
  src/tm30/metrics.cpp
  src/tm30/hue_bins.cpp
  src/tm30/gamut.cpp
  src/tm30/task_pool.cpp
  src/tm30/tm30.cpp
)
target_include_directories(tm30-core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(tm30-core PUBLIC cxx_std_20)
target_link_libraries(tm30-core PUBLIC Threads::Threads)

# ── Tests ────────────────────────────────────────────────────────────────
add_executable(tm30-tests
  tests/slice_00_harness_test.cpp
  tests/slice_01_resample_test.cpp
  tests/slice_02_xyz_test.cpp
  tests/slice_03_cct_test.cpp
  tests/slice_04_reference_test.cpp
  tests/slice_05_ces_colorimetry_test.cpp
  tests/slice_06_ciecam02_test.cpp
  tests/slice_07_deltae_rf_test.cpp
  tests/slice_08_hue_bins_test.cpp
  tests/slice_09_rg_local_cvg_test.cpp
  tests/slice_10_sample_test.cpp
  tests/slice_11_tm30_test.cpp
  tests/slice_12_spd_utils_test.cpp
  tests/slice_13_parallel_test.cpp
)
target_link_libraries(tm30-tests PRIVATE
  tm30-core
  Catch2::Catch2WithMain
)
target_compile_features(tm30-tests PRIVATE cxx_std_20)
target_compile_definitions(tm30-tests PRIVATE
  TM30_DATA_DIR="${TM30_DATA_DIR}"
  TM30_FIXTURE_DIR="${TM30_FIXTURE_DIR}"
)

# ── CTest integration ────────────────────────────────────────────────────
include(CTest)
# Option so sanitizer builds (whose binaries cannot run test discovery -
# e.g. GCC TSan crashes on Ubuntu 24.04's ASLR entropy) can skip it.
option(TM30_ENABLE_CTEST_DISCOVERY
       "Run catch_discover_tests for the tm30-tests target" ON)
if(TM30_ENABLE_CTEST_DISCOVERY)
  catch_discover_tests(tm30-tests)
endif()
