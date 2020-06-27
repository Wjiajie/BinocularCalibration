aux_source_directory("../src" SRC_LIST)
SET(LIBCALIBRATION_SRC ${SRC_LIST})
ADD_LIBRARY(calibration STATIC ${LIBCALIBRATION_SRC})
include_directories( "../include")


