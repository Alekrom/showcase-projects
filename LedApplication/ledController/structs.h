#ifndef STRUCTS_H 
#define STRUCTS_H

//Colors
struct color_t {
  int red;
  int blue;
  int green;
  int white;
} colors;


struct stroboColor {
  bool on = true;
  unsigned long  timer = 0;
  unsigned long  offset = 0;
  unsigned long  onInterval = 0;
  unsigned long  offInterval = 0;
  bool enabled = false;
  bool isOffset = true;
};

struct strobo {
  stroboColor red;
  stroboColor green;
  stroboColor blue;
  stroboColor white;
  stroboColor master;
} strobo;


struct dim {
  bool redUp = false;
  bool blueUp = false;
  bool greenUp = false;
  bool whiteUp = false;
  bool masterUp = false;
} dim_up;

#endif
