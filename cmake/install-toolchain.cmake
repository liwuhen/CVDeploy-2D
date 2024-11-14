file(MAKE_DIRECTORY ${LIBS})
file(MAKE_DIRECTORY ${WORKSPACE_PATH})
file(MAKE_DIRECTORY ${WORKSPACE_PATH}/log)
file(MAKE_DIRECTORY ${WORKSPACE_PATH}/image)
file(MAKE_DIRECTORY ${CONFIG_PATH})
file(MAKE_DIRECTORY ${THIRD_PARTY}/opencv)
file(MAKE_DIRECTORY ${PROCESSOR_OUTPUT_DIR})
file(MAKE_DIRECTORY ${POSTPRECESSOR_OUTPUT_DIR})


file(GLOB_RECURSE OPENCV_LIBS 
    "${OPENCV_DIR}/lib/libopencv_core.so.3.4"
    "${OPENCV_DIR}/lib/libopencv_imgcodecs.so.3.4"
    "${OPENCV_DIR}/lib/libopencv_imgproc.so.3.4"
    "${OPENCV_DIR}/lib/libopencv_highgui.so.3.4")

# copy opencv libs
foreach(lib ${OPENCV_LIBS})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${lib} ${THIRD_PARTY}/opencv
        COMMENT "Copying Opencv library ${lib} to ${THIRD_PARTY}/opencv"
    )
endforeach()
