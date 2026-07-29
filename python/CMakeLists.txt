cmake_minimum_required(VERSION 3.20)
project(tm30-python LANGUAGES CXX)

# ── C++20 ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Paths ────────────────────────────────────────────────────────────────
set(TM30_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ── Find Python + nanobind ──────────────────────────────────────────────
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── Core library (same sources as parent CMakeLists) ────────────────────
add_library(tm30-core STATIC
  ${TM30_ROOT}/src/tm30/version.cpp
  ${TM30_ROOT}/src/tm30/spd.cpp
  ${TM30_ROOT}/src/tm30/csv_loader.cpp
  ${TM30_ROOT}/src/tm30/resample.cpp
  ${TM30_ROOT}/src/tm30/integrate.cpp
  ${TM30_ROOT}/src/tm30/xyz.cpp
  ${TM30_ROOT}/src/tm30/chromaticity.cpp
  ${TM30_ROOT}/src/tm30/cct.cpp
  ${TM30_ROOT}/src/tm30/reference.cpp
  ${TM30_ROOT}/src/tm30/pipeline.cpp
  ${TM30_ROOT}/src/tm30/ciecam02.cpp
  ${TM30_ROOT}/src/tm30/metrics.cpp
  ${TM30_ROOT}/src/tm30/hue_bins.cpp
  ${TM30_ROOT}/src/tm30/gamut.cpp
  ${TM30_ROOT}/src/tm30/tm30.cpp
)
target_include_directories(tm30-core
  PUBLIC ${TM30_ROOT}/include
)
target_compile_features(tm30-core PUBLIC cxx_std_20)
set_target_properties(tm30-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Data directory path ─────────────────────────────────────────────────
set(TM30_DATA_DIR "${TM30_ROOT}/data")

# ── nanobind extension module ───────────────────────────────────────────
nanobind_add_module(tm30_core
  tm30_bindings.cpp
)
target_link_libraries(tm30_core PRIVATE tm30-core)
target_compile_definitions(tm30_core PRIVATE
  TM30_DATA_DIR="${TM30_DATA_DIR}"
)

# ── Install into the wheel ───────────────────────────────────────────────
install(TARGETS tm30_core LIBRARY DESTINATION .)
install(FILES tm30_calc.py DESTINATION .)
