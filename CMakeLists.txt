cmake_minimum_required(VERSION 3.8)
project(rio_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

# rio-lib submodule
set(RIO_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/external/rio-lib")
add_subdirectory(${RIO_LIB_DIR} ${CMAKE_BINARY_DIR}/rio-lib-build)

add_executable(rio_node src/rio_node.cpp)

ament_target_dependencies(rio_node
  rclcpp
  std_msgs
  sensor_msgs
  px4_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  Eigen3
)

target_link_libraries(rio_node rio::rio_eskf_cpp)

# Ensure rio headers visible
target_include_directories(rio_node PRIVATE ${RIO_LIB_DIR}/include)

install(TARGETS rio_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()