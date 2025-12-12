# CMakeLists.txt for scratch/vanet-sumo

# Find ZMQ library explicitly
find_library(ZMQ_LIB NAMES zmq libzmq REQUIRED)
message(STATUS "Found ZMQ Library: ${ZMQ_LIB}")

# Explicitly list sources
set(SOURCES
    vanet-sumo.cc
    rl-interface.cc
)

# Create executable
add_executable(vanet-sumo ${SOURCES})

# Link
target_link_libraries(vanet-sumo
    PRIVATE
      ${ns3-libs}
      ${ns3-contrib-libs}
      ${ZMQ_LIB}
)

set_target_properties(vanet-sumo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_OUTPUT_DIRECTORY}/scratch/vanet-sumo"
)
