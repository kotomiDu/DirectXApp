# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
#
#
# FindNGraph
# ------
#
# This will define the following variables:
#
#   ngraph_FOUND               - True if the system has the nGraph library
#   NGRAPH_LIBRARIES           - nGraph libraries
#   ngraph::ngraph             - nGraph core target
#   ngraph_onnx_importer_FOUND - True if the system has onnx_importer library
#   ONNX_IMPORTER_LIBRARIES    - ONNX importer libraries
#   ngraph::onnx_importer      - ONNX importer target
#
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ngraphConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET ngraph)
    include("${CMAKE_CURRENT_LIST_DIR}/ngraphTargets.cmake")
endif()

set(NGRAPH_LIBRARIES ngraph::ngraph)
set(ngraph_onnx_importer_FOUND ON)
if(ngraph_onnx_importer_FOUND)
    set(ONNX_IMPORTER_LIBRARIES ngraph::onnx_importer)
endif()
check_required_components(ngraph)
