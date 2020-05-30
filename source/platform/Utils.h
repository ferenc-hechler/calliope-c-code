#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stddef.h>

class Vect;
class Mat;

size_t float2char(char* buf, float number, uint8_t digits);
float char2float(const char* buf);

void throwError(const char *errMsg);

void log(const char *msg);
void logFloat(float f);
void logInt(int i);


#endif // UTILS_H
