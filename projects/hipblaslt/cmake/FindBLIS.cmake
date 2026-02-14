# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(PkgConfig REQUIRED)
pkg_check_modules(BLIS_MT REQUIRED IMPORTED_TARGET blis-mt)

set(BLIS_FOUND TRUE)
set(BLIS_INCLUDE_DIR ${BLIS_MT_INCLUDE_DIRS})
set(BLIS_LIB ${BLIS_MT_LINK_LIBRARIES})
set(BLIS_LIBRARIES ${BLIS_LIB})
set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})
message(STATUS "BLIS library: ${BLIS_LIBRARIES}")
message(STATUS "BLIS include: ${BLIS_INCLUDE_DIRS}")

# Create an alias target for BLIS
if(NOT TARGET BLIS::BLIS)
    add_library(BLIS::BLIS ALIAS PkgConfig::BLIS_MT)
endif()

message(STATUS "Found BLIS: ${BLIS_LIB}")
message(STATUS "Found BLIS: ${BLIS_INCLUDE_DIR}")
