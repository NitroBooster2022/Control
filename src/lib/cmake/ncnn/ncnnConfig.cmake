set(NCNN_OPENMP ON)
set(NCNN_THREADS ON)
set(NCNN_VULKAN ON)
set(NCNN_SHARED_LIB OFF)
set(NCNN_SYSTEM_GLSLANG ON)
set(NCNN_SIMPLEVK ON)

if(NCNN_OPENMP)
    find_package(OpenMP)
endif()

if(NCNN_THREADS)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
endif()

if(NCNN_VULKAN)
    if(NOT NCNN_SIMPLEVK)
        find_package(Vulkan REQUIRED)
    endif()

    if(NOT NCNN_SHARED_LIB)
        if(NCNN_SYSTEM_GLSLANG)
            find_package(glslang QUIET)
            if(NOT glslang_FOUND)
                set(GLSLANG_TARGET_DIR "")
                include(${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake)
                include(${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake)
                if(EXISTS "${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
                    # hlsl support can be optional
                    include("${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
                endif()
                include(${GLSLANG_TARGET_DIR}/glslangTargets.cmake)
                include(${GLSLANG_TARGET_DIR}/SPIRVTargets.cmake)
            endif()
        else()
            set(glslang_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../lib/cmake/glslang")
            find_package(glslang QUIET)
        endif()
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/ncnn.cmake)
