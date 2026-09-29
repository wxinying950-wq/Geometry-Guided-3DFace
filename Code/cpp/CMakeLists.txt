cmake_minimum_required(VERSION 3.16)
project(FaceStitcheCppDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The repository carries the OpenCV files needed by this demo in Env/opencv.
# The cache variables remain overridable for users who have another OpenCV
# installation, but a fresh configure starts from the bundled copy.
get_filename_component(FACESTITCHE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(FACESTITCHE_OPENCV_ROOT "${FACESTITCHE_ROOT}/Env/opencv")
set(_bundled_include_dir "${FACESTITCHE_OPENCV_ROOT}/include")
set(_bundled_library_dir "${FACESTITCHE_OPENCV_ROOT}/lib")
set(_bundled_runtime_dir "${FACESTITCHE_OPENCV_ROOT}/bin")
set(_bundled_runtime_zip "${_bundled_runtime_dir}/opencv_world450.dll.zip")

# The runtime DLL is stored as a ZIP so the repository can be uploaded through
# GitHub's file APIs. Extract it during configure when the DLL is not already
# present. CMake's bundled tar command supports ZIP archives on Windows.
if(EXISTS "${_bundled_runtime_zip}" AND NOT EXISTS "${_bundled_runtime_dir}/opencv_world450.dll")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E chdir "${_bundled_runtime_dir}" tar xf "${_bundled_runtime_zip}"
        RESULT_VARIABLE _opencv_extract_result)
    if(NOT _opencv_extract_result EQUAL 0 OR NOT EXISTS "${_bundled_runtime_dir}/opencv_world450.dll")
        message(FATAL_ERROR "Could not extract ${_bundled_runtime_zip}")
    endif()
endif()

set(OPENCV_INCLUDE_DIR "${_bundled_include_dir}" CACHE PATH "OpenCV include directory")
set(OPENCV_LIBRARY_DIR "${_bundled_library_dir}" CACHE PATH "OpenCV library directory")
set(OPENCV_LIBRARY "opencv_world450" CACHE STRING "OpenCV library name or full path")

# Replace stale cache values whose directories are no longer available while
# preserving an explicitly selected non-bundled installation.
if(NOT EXISTS "${OPENCV_INCLUDE_DIR}")
    set(OPENCV_INCLUDE_DIR "${_bundled_include_dir}" CACHE PATH "OpenCV include directory" FORCE)
endif()
if(NOT EXISTS "${OPENCV_LIBRARY_DIR}")
    set(OPENCV_LIBRARY_DIR "${_bundled_library_dir}" CACHE PATH "OpenCV library directory" FORCE)
endif()

set(OPENCV_RUNTIME_DIR "")
if(EXISTS "${OPENCV_INCLUDE_DIR}" AND EXISTS "${OPENCV_LIBRARY_DIR}")
    # Do not retain a library discovered by an earlier configure after the
    # include/library directories have been changed.
    unset(OPENCV_LIBRARY_FILE CACHE)
    if(IS_ABSOLUTE "${OPENCV_LIBRARY}" AND EXISTS "${OPENCV_LIBRARY}")
        set(OPENCV_LIBRARY_FILE "${OPENCV_LIBRARY}")
    else()
        find_library(OPENCV_LIBRARY_FILE
            NAMES "${OPENCV_LIBRARY}" opencv_world450
            PATHS "${OPENCV_LIBRARY_DIR}"
            NO_DEFAULT_PATH)
    endif()
    if(NOT OPENCV_LIBRARY_FILE)
        message(FATAL_ERROR "OpenCV library not found in ${OPENCV_LIBRARY_DIR}")
    endif()
    set(OPENCV_INCLUDE_DIRS "${OPENCV_INCLUDE_DIR}")
    set(OPENCV_LIBS "${OPENCV_LIBRARY_FILE}")
    if(OPENCV_INCLUDE_DIR STREQUAL "${_bundled_include_dir}" AND
       OPENCV_LIBRARY_DIR STREQUAL "${_bundled_library_dir}" AND
       IS_DIRECTORY "${_bundled_runtime_dir}")
        set(OPENCV_RUNTIME_DIR "${_bundled_runtime_dir}")
    endif()
else()
    find_package(OpenCV REQUIRED)
    set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(OPENCV_LIBS ${OpenCV_LIBS})
endif()

add_executable(facestitche_demo demo.cpp)
target_link_libraries(facestitche_demo PRIVATE ${OPENCV_LIBS})
target_include_directories(facestitche_demo PRIVATE ${OPENCV_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(facestitche_demo PRIVATE /utf-8)
endif()

# Keep the executable self-contained after building. This removes the need
# to add Env/bin to PATH when the bundled OpenCV copy is used.
if(OPENCV_RUNTIME_DIR)
    add_custom_command(TARGET facestitche_demo POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${OPENCV_RUNTIME_DIR}"
            "$<TARGET_FILE_DIR:facestitche_demo>"
        COMMENT "Copying bundled OpenCV runtime DLLs")
endif()
