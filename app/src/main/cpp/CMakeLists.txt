# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("OakDroidSLAM")

# configure import libs
set(distribution_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../distribution)

add_library(lib_gmath STATIC IMPORTED)
set_target_properties(lib_gmath PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/gmath/lib/${ANDROID_ABI}/libgmath.a)

# ${ANDROID_ABI} is handy for our purpose here. Probably this ${ANDROID_ABI} is
# the most valuable thing of this sample, the rest are pretty much normal cmake
add_library(lib_gperf SHARED IMPORTED)
set_target_properties(lib_gperf PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/gperf/lib/${ANDROID_ABI}/libgperf.so)

######         libusb         ######
add_library(usb-1.0 SHARED IMPORTED )
set_target_properties(usb-1.0 PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/libusb/lib/${ANDROID_ABI}/libusb-1.0.so)

######         opencv         ######
add_library( opencv-lib SHARED IMPORTED )
set_target_properties(opencv-lib PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/opencv/lib/${ANDROID_ABI}/libopencv_java4.so)

######         depthai         ######
add_library( depthai-lib SHARED IMPORTED )
set_target_properties(depthai-lib PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/depthai/lib/${ANDROID_ABI}/libdepthai-core.so)


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        OakDroidSLAM

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp
        utils.cpp)

target_include_directories(OakDroidSLAM PRIVATE
        ${distribution_DIR}/gmath/include
        ${distribution_DIR}/gperf/include
        ${distribution_DIR}/libusb/include
        ${distribution_DIR}/opencv/include
        ${distribution_DIR}/depthai/include)


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        OakDroidSLAM
        android
        lib_gmath
        lib_gperf
        depthai-lib
        opencv-lib
        usb-1.0
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})