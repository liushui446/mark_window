#ifndef __STD_1_H__
#define __STD_1_H__

# if (defined Windows)
#   define SM_EXPORTS __declspec(dllexport)
# elif defined Linux
#   define SM_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif
