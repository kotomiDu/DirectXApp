#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ngraph::ngraph" for configuration "Release"
set_property(TARGET ngraph::ngraph APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngraph::ngraph PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/ngraph.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/ngraph.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS ngraph::ngraph )
list(APPEND _IMPORT_CHECK_FILES_FOR_ngraph::ngraph "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/ngraph.lib" "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/ngraph.dll" )

# Import target "ngraph::onnx_importer" for configuration "Release"
set_property(TARGET ngraph::onnx_importer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngraph::onnx_importer PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/onnx_importer.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/onnx_importer.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS ngraph::onnx_importer )
list(APPEND _IMPORT_CHECK_FILES_FOR_ngraph::onnx_importer "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/onnx_importer.lib" "${_IMPORT_PREFIX}/deployment_tools/ngraph/lib/onnx_importer.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
