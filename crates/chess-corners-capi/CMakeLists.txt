cmake_minimum_required(VERSION 3.20)

# ----------------------------------------------------------------------------
# Version: track the workspace crate version from the root Cargo.toml so the
# CMake package version, the version file, and the .pc never drift.
# ----------------------------------------------------------------------------
get_filename_component(_cc_workspace_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
file(READ "${_cc_workspace_root}/Cargo.toml" _cc_root_cargo)
string(REGEX MATCH "version[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _cc_ver_match "${_cc_root_cargo}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Could not parse version from ${_cc_workspace_root}/Cargo.toml")
endif()
set(_cc_version "${CMAKE_MATCH_1}")

project(chess-corners
  VERSION ${_cc_version}
  DESCRIPTION "Stable C ABI for the chess-corners detector"
  HOMEPAGE_URL "https://github.com/vitalyvorobyev/chess-corners-rs"
  LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_cc_top_level TRUE)
else()
  set(_cc_top_level FALSE)
endif()
option(CHESS_CORNERS_BUILD_TESTS "Build the chess-corners C/C++ smoke tests" ${_cc_top_level})

# ----------------------------------------------------------------------------
# Linkage selection. VCPKG_LIBRARY_LINKAGE (set by vcpkg) wins; otherwise the
# standard BUILD_SHARED_LIBS switch. Default is a static library.
# ----------------------------------------------------------------------------
if(DEFINED VCPKG_LIBRARY_LINKAGE)
  if(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    set(_cc_shared TRUE)
  else()
    set(_cc_shared FALSE)
  endif()
else()
  if(BUILD_SHARED_LIBS)
    set(_cc_shared TRUE)
  else()
    set(_cc_shared FALSE)
  endif()
endif()

if(_cc_shared)
  set(_cc_lib_filename
      "${CMAKE_SHARED_LIBRARY_PREFIX}chess_corners_capi${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(_cc_imported_type SHARED)
else()
  set(_cc_lib_filename
      "${CMAKE_STATIC_LIBRARY_PREFIX}chess_corners_capi${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(_cc_imported_type STATIC)
endif()

# ----------------------------------------------------------------------------
# Locate cargo and the workspace target directory (honours CARGO_TARGET_DIR).
# ----------------------------------------------------------------------------
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR "cargo not found on PATH; install the Rust toolchain")
endif()

execute_process(
  COMMAND "${CARGO_EXECUTABLE}" metadata --format-version 1 --no-deps
  WORKING_DIRECTORY "${_cc_workspace_root}"
  OUTPUT_VARIABLE _cc_metadata
  RESULT_VARIABLE _cc_meta_res
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(_cc_target_dir "")
if(_cc_meta_res EQUAL 0)
  string(JSON _cc_target_dir ERROR_VARIABLE _cc_json_err GET "${_cc_metadata}" "target_directory")
endif()
if(NOT _cc_target_dir)
  set(_cc_target_dir "${_cc_workspace_root}/target")
endif()
set(_cc_artifact_dir "${_cc_target_dir}/release")
set(_cc_lib_path "${_cc_artifact_dir}/${_cc_lib_filename}")

# ----------------------------------------------------------------------------
# Build the Rust library now so the imported target's location exists at
# configure time (needed for install() and for the in-tree tests). A custom
# target re-runs cargo on `cmake --build` to pick up Rust source changes.
# ----------------------------------------------------------------------------
execute_process(
  COMMAND "${CARGO_EXECUTABLE}" build --release -p chess-corners-capi
  WORKING_DIRECTORY "${_cc_workspace_root}"
  RESULT_VARIABLE _cc_build_res)
if(NOT _cc_build_res EQUAL 0)
  message(FATAL_ERROR "cargo build of chess-corners-capi failed")
endif()
if(NOT EXISTS "${_cc_lib_path}")
  message(FATAL_ERROR "expected library not found: ${_cc_lib_path}")
endif()

add_custom_target(chess_corners_cargo ALL
  COMMAND "${CARGO_EXECUTABLE}" build --release -p chess-corners-capi
  WORKING_DIRECTORY "${_cc_workspace_root}"
  BYPRODUCTS "${_cc_lib_path}"
  COMMENT "Building chess-corners-capi (cargo)"
  VERBATIM)

# ----------------------------------------------------------------------------
# Static linking pulls in the Rust standard library's native dependencies.
# Ask the toolchain for them empirically rather than hardcoding per platform.
# (`-framework X` arrives as two tokens; recombine into one item.)
# ----------------------------------------------------------------------------
set(_cc_native_libs "")
if(NOT _cc_shared)
  execute_process(
    COMMAND "${CARGO_EXECUTABLE}" rustc --release -q -p chess-corners-capi
            --crate-type staticlib -- --print native-static-libs
    WORKING_DIRECTORY "${_cc_workspace_root}"
    OUTPUT_VARIABLE _cc_nsl_out
    ERROR_VARIABLE _cc_nsl_err
    RESULT_VARIABLE _cc_nsl_res)
  if(_cc_nsl_res EQUAL 0 AND _cc_nsl_err MATCHES "native-static-libs:([^\r\n]*)")
    separate_arguments(_cc_nsl_args NATIVE_COMMAND "${CMAKE_MATCH_1}")
    set(_cc_prev "")
    foreach(_tok IN LISTS _cc_nsl_args)
      if(_cc_prev STREQUAL "-framework")
        list(APPEND _cc_native_libs "-framework ${_tok}")
        set(_cc_prev "")
      elseif(_tok STREQUAL "-framework")
        set(_cc_prev "-framework")
      else()
        list(APPEND _cc_native_libs "${_tok}")
        set(_cc_prev "")
      endif()
    endforeach()
  else()
    # Fallback if the toolchain note could not be parsed.
    if(APPLE)
      set(_cc_native_libs "-lm")
    elseif(UNIX)
      set(_cc_native_libs "-lpthread;-ldl;-lm")
    endif()
  endif()
  # On macOS -lSystem / -lc are always supplied by the compiler driver; passing
  # them again only produces a "duplicate libraries" linker warning.
  if(APPLE)
    list(REMOVE_ITEM _cc_native_libs "-lSystem" "-lc")
  endif()
endif()

# ----------------------------------------------------------------------------
# Imported target consumed in-tree (tests) and re-created by the installed
# package config for downstream `find_package` consumers.
# ----------------------------------------------------------------------------
add_library(chess-corners::chess-corners ${_cc_imported_type} IMPORTED GLOBAL)
set_target_properties(chess-corners::chess-corners PROPERTIES
  IMPORTED_LOCATION "${_cc_lib_path}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(_cc_native_libs)
  set_property(TARGET chess-corners::chess-corners PROPERTY
    INTERFACE_LINK_LIBRARIES "${_cc_native_libs}")
endif()

# ----------------------------------------------------------------------------
# Install: headers, the selected library, the CMake package config + version
# file, and a pkg-config file.
# ----------------------------------------------------------------------------
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/chess_corners.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/chess_corners.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(FILES "${_cc_lib_path}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# A macOS shared library is built with an absolute install name pointing at the
# build tree; rewrite it to @rpath so the installed copy is relocatable.
if(_cc_shared AND APPLE)
  install(CODE "execute_process(COMMAND install_name_tool -id \"@rpath/${_cc_lib_filename}\" \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${_cc_lib_filename}\")")
endif()

set(_cc_config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/chess-corners")

# Variables substituted into chess-corners-config.cmake.in.
set(CC_IMPORTED_TYPE "${_cc_imported_type}")
set(CC_LIB_FILENAME "${_cc_lib_filename}")
set(CC_INTERFACE_LINK_LIBRARIES "${_cc_native_libs}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/chess-corners-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/chess-corners-config.cmake"
  INSTALL_DESTINATION "${_cc_config_install_dir}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/chess-corners-config-version.cmake"
  VERSION "${_cc_version}"
  COMPATIBILITY SameMinorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/chess-corners-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/chess-corners-config-version.cmake"
  DESTINATION "${_cc_config_install_dir}")

# pkg-config: ${pcfiledir}-relative prefix keeps the .pc relocatable across
# install prefixes.
file(RELATIVE_PATH CC_PC_RELDIR
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  "${CMAKE_INSTALL_PREFIX}")
string(REGEX REPLACE "/+$" "" CC_PC_RELDIR "${CC_PC_RELDIR}")
set(CC_VERSION "${_cc_version}")
if(_cc_shared)
  set(CC_PC_PRIVATE_LIBS "")
else()
  string(REPLACE ";" " " CC_PC_PRIVATE_LIBS "${_cc_native_libs}")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/chess-corners.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/chess-corners.pc"
  @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/chess-corners.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# ----------------------------------------------------------------------------
# In-tree smoke tests against the build-tree library.
# ----------------------------------------------------------------------------
if(CHESS_CORNERS_BUILD_TESTS)
  enable_testing()

  add_executable(cc_c_smoke "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.c")
  target_link_libraries(cc_c_smoke PRIVATE chess-corners::chess-corners)
  add_dependencies(cc_c_smoke chess_corners_cargo)
  add_test(NAME cc_c_smoke COMMAND cc_c_smoke)

  add_executable(cc_cpp_smoke "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/detect.cpp")
  target_compile_features(cc_cpp_smoke PRIVATE cxx_std_17)
  target_link_libraries(cc_cpp_smoke PRIVATE chess-corners::chess-corners)
  add_dependencies(cc_cpp_smoke chess_corners_cargo)
  add_test(NAME cc_cpp_smoke COMMAND cc_cpp_smoke)
  set_tests_properties(cc_cpp_smoke PROPERTIES PASS_REGULAR_EXPRESSION "detected 49 corners")

  # A shared build-tree library may not sit on the default loader path.
  if(_cc_shared)
    if(APPLE)
      set(_cc_loader_var "DYLD_LIBRARY_PATH")
    else()
      set(_cc_loader_var "LD_LIBRARY_PATH")
    endif()
    set_tests_properties(cc_c_smoke cc_cpp_smoke PROPERTIES
      ENVIRONMENT "${_cc_loader_var}=${_cc_artifact_dir}")
  endif()
endif()
