#ifndef CORTEX_SDK_EXPORT_H
#define CORTEX_SDK_EXPORT_H

// Define export macros for different platforms
#if defined(_WIN32) || defined(_WIN64)
    #ifdef CORTEXSDR_BUILDING_SHARED_LIBRARY
        #define CORTEXSDR_API __declspec(dllexport)
    #else
        #define CORTEXSDR_API __declspec(dllimport)
    #endif
#else
    #ifdef CORTEXSDR_BUILDING_SHARED_LIBRARY
        #define CORTEXSDR_API __attribute__((visibility("default")))
    #else
        #define CORTEXSDR_API
    #endif
#endif

#endif // CORTEX_SDK_EXPORT_H
