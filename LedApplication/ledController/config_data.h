#ifndef CONFIG_H // include guard
#define CONFIG_H

#include <stdbool.h>

// 100 microseconds
int timerInitValue = 100;

// TODO move init somewhere else!!!!!

bool dim_enabled_red = 0; 
bool dim_enabled_blue = 0; 
bool dim_enabled_green = 0; 
bool dim_enabled_white = 0; 
bool dim_enabled_master = 0; 

bool dim_keepColors = 0;

int dim_speed_red = 0; 
int dim_speed_blue = 0; 
int dim_speed_green = 0; 
int dim_speed_white = 0; 
int dim_speed_master = 0; 

#endif
