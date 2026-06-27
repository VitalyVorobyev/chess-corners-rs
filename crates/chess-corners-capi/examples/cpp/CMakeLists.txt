cmake_minimum_required(VERSION 3.20)

# Standalone consumer project: builds against an installed chess-corners
# package via find_package. Configure with
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<install-prefix>
project(chess-corners-cpp-example LANGUAGES CXX)

find_package(chess-corners CONFIG REQUIRED)

add_executable(detect detect.cpp)
target_compile_features(detect PRIVATE cxx_std_17)
target_link_libraries(detect PRIVATE chess-corners::chess-corners)

enable_testing()
add_test(NAME detect COMMAND detect)
set_tests_properties(detect PROPERTIES PASS_REGULAR_EXPRESSION "detected 49 corners")

# Ensure a shared library is found at runtime regardless of rpath support.
get_target_property(_cc_loc chess-corners::chess-corners IMPORTED_LOCATION)
if(_cc_loc)
  get_filename_component(_cc_libdir "${_cc_loc}" DIRECTORY)
  if(APPLE)
    set(_cc_loader_var "DYLD_LIBRARY_PATH")
  else()
    set(_cc_loader_var "LD_LIBRARY_PATH")
  endif()
  set_tests_properties(detect PROPERTIES ENVIRONMENT "${_cc_loader_var}=${_cc_libdir}")
endif()
