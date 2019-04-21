set(FILES_SRC
        test_main.cc
        test_topography.cc
        )

add_executable(
        unit_tests
        ${FILES_SRC}
)

target_link_libraries(
        unit_tests
        PRIVATE
        tgo
        fmt-header-only
)

target_include_directories(
        unit_tests
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)