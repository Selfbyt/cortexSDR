#ifndef CORTEXSDR_SDK_EXPORT_H
#define CORTEXSDR_SDK_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef CORTEXSDR_BUILDING_DLL
    #ifdef __GNUC__
      #define CORTEXSDR_API __attribute__((dllexport))
    #else
      #define CORTEXSDR_API __declspec(dllexport)
    #endif
  #else
    #ifdef __GNUC__
      #define CORTEXSDR_API __attribute__((dllimport))
    #else
      #define CORTEXSDR_API __declspec(dllimport)
    #endif
  #endif
#else
  #if __GNUC__ >= 4
    #define CORTEXSDR_API __attribute__((visibility("default")))
  #else
    #define CORTEXSDR_API
  #endif
#endif

#endif // CORTEXSDR_SDK_EXPORT_H
