cmake_minimum_required(VERSION 3.10)
project(InspectBmadAttributes Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ACC_BUILD_DIR "production")
else()
  set(ACC_BUILD_DIR "debug")
endif()

include_directories($ENV{ACC_ROOT_DIR}/${ACC_BUILD_DIR}/modules)
link_directories($ENV{ACC_ROOT_DIR}/${ACC_BUILD_DIR}/lib)

add_executable(gen_attrs attrs.f90)

target_compile_options(gen_attrs PRIVATE -fdollar-ok -fbackslash)

target_link_libraries(gen_attrs
    bmad
    sim_utils
)

message(STATUS "Bmad found at: ${BMAD_LIB}")
