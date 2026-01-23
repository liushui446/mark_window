#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pylon::PylonBase" for configuration "Release"
set_property(TARGET pylon::PylonBase APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylon::PylonBase PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Basler/lib/x64/PylonBase_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonBase_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylon::PylonBase )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylon::PylonBase "${_IMPORT_PREFIX}/Basler/lib/x64/PylonBase_v10.lib" "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonBase_v10.dll" )

# Import target "pylon::PylonUtility" for configuration "Release"
set_property(TARGET pylon::PylonUtility APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylon::PylonUtility PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Basler/lib/x64/PylonUtility_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonUtility_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylon::PylonUtility )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylon::PylonUtility "${_IMPORT_PREFIX}/Basler/lib/x64/PylonUtility_v10.lib" "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonUtility_v10.dll" )

# Import target "pylon::PylonGUI" for configuration "Release"
set_property(TARGET pylon::PylonGUI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylon::PylonGUI PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Basler/lib/x64/PylonGUI_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonGUI_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylon::PylonGUI )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylon::PylonGUI "${_IMPORT_PREFIX}/Basler/lib/x64/PylonGUI_v10.lib" "${_IMPORT_PREFIX}/Basler/Runtime/x64/PylonGUI_v10.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
