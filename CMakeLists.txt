cmake_minimum_required(VERSION 3.2)

project(realty VERSION 1.0.$ENV{TRAVIS_BUILD_NUMBER})

if($ENV{TRAVIS_BUILD_NUMBER})
  add_subdirectory(dlib)
endif()


add_executable(rclst rclst.cpp)
add_executable(rclss rclss.cpp)

find_package(Boost COMPONENTS system  REQUIRED)

target_include_directories(rclst PRIVATE ./include)
target_include_directories(rclss PRIVATE ./include)

set_target_properties(rclst rclss PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_OPTIONS -Wall -Wextra -Wpedantic 
)

target_link_libraries(rclst ${Boost_LIBRARIES} openblas dlib )
target_link_libraries(rclss ${Boost_LIBRARIES} openblas dlib)

install(TARGETS rclst RUNTIME DESTINATION bin)
install(TARGETS rclss RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT lisstic.tt@gmail.com)

include (CPack)