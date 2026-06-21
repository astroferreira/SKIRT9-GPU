# //////////////////////////////////////////////////////////////////
# ///     The SKIRT project -- advanced radiative transfer       ///
# ///       © Astronomical Observatory, Ghent University         ///
# //////////////////////////////////////////////////////////////////

# ------------------------------------------------------------------
# Builds the SKIRT executable
# ------------------------------------------------------------------

# list the source files in this directory
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.hpp")

function(add_skirt_executable TARGET)
    # create the executable target
    add_executable(${TARGET} ${SOURCES} ${HEADERS})

    # enable multi-threading
    find_package(Threads REQUIRED)
    target_link_libraries(${TARGET} Threads::Threads)

    # add SMILE library dependencies
    target_link_libraries(${TARGET} serialize schema fundamentals build)
    target_include_directories(${TARGET} PRIVATE ../../SMILE/serialize ../../SMILE/schema ../../SMILE/fundamentals
                                                 ../../SMILE/build)

    # add SKIRT library dependencies
    target_link_libraries(${TARGET} skirtcore)
    target_include_directories(${TARGET} PRIVATE ../core ../mpi ../utils)

    # adjust C++ compiler flags to our needs
    include("../../SMILE/build/CompilerFlags.cmake")
endfunction()

add_skirt_executable(skirt)

add_skirt_executable(skirtgpu)
target_compile_definitions(skirtgpu PRIVATE SKIRTGPU_BINARY)
