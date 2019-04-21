set(FILES_SRC
        topography.cc
        tgo.cc
        )

add_library(tgo
        SHARED
        ${FILES_SRC}
        )

target_link_libraries(tgo
        PRIVATE
        fmt-header-only
        ${NLOPT_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        )
