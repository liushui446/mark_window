#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pylonC::PylonBase" for configuration "Release"
set_property(TARGET pylonC::PylonBase APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylonC::PylonBase PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Development/lib/x64/PylonBase_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Runtime/x64/PylonBase_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylonC::PylonBase )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylonC::PylonBase "${_IMPORT_PREFIX}/Development/lib/x64/PylonBase_v10.lib" "${_IMPORT_PREFIX}/Runtime/x64/PylonBase_v10.dll" )

# Import target "pylonC::PylonUtility" for configuration "Release"
set_property(TARGET pylonC::PylonUtility APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylonC::PylonUtility PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Development/lib/x64/PylonUtility_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Runtime/x64/PylonUtility_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylonC::PylonUtility )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylonC::PylonUtility "${_IMPORT_PREFIX}/Development/lib/x64/PylonUtility_v10.lib" "${_IMPORT_PREFIX}/Runtime/x64/PylonUtility_v10.dll" )

# Import target "pylonC::PylonGUI" for configuration "Release"
set_property(TARGET pylonC::PylonGUI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylonC::PylonGUI PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Development/lib/x64/PylonGUI_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Runtime/x64/PylonGUI_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylonC::PylonGUI )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylonC::PylonGUI "${_IMPORT_PREFIX}/Development/lib/x64/PylonGUI_v10.lib" "${_IMPORT_PREFIX}/Runtime/x64/PylonGUI_v10.dll" )

# Import target "pylonC::PylonC" for configuration "Release"
set_property(TARGET pylonC::PylonC APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pylonC::PylonC PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/Development/lib/x64/PylonC_v10.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/Runtime/x64/PylonC_v10.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS pylonC::PylonC )
list(APPEND _IMPORT_CHECK_FILES_FOR_pylonC::PylonC "${_IMPORT_PREFIX}/Development/lib/x64/PylonC_v10.lib" "${_IMPORT_PREFIX}/Runtime/x64/PylonC_v10.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
