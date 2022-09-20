// BT module: HC-05 20:18:07:13:70:88

#include <avr/eeprom.h>
#include <Adafruit_NeoPixel.h>
#include <TimerOne.h>
//#include <StreamUtils.h>
#include "parser.h"
#include "structs.h"
#include "config_data.h" 
#include "MemoryFree.h"
#include "Gamma.h"

#define LED_PIN    7
#define LED_COUNT 120
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_RGBW + NEO_KHZ800);

// TODO move dim and strobo specific actions to new files

bool didReadNewCmd = false;
bool waitsForCmd = false;

char startHandshakeChar = 'x';
char endHandshakeChar = 'y';
char terminator = '!';
String command ="";

// due to color correction everything below is considered off
int lowestColorValue = 28;

// TODO are overflows a problem?
// if timer is set to long increasing it costs too much time for period of 100
// Overflow is a problem, when it overflows while handshake is in progress
unsigned int timerCount = 0;

unsigned int handshakeStart = 0;

// only display colors if necessary
bool colorChanged = true;


void setup() {
  Serial.begin(9600);
  eeprom_read_block((void*)&colors, (void*)0, sizeof(colors));
  strip.begin();
  uint32_t color = strip.Color(colors.green, colors.red, colors.blue, colors.white);
  strip.fill(color);
  strip.show(); // Initialize all pixels to 'off'
  Timer1.initialize(100);
  Timer1.attachInterrupt(timer_isr);
}

void loop() {
  // TODO should be better if done only after color change?
   eeprom_write_block((const void*)&colors, (void*)0, sizeof(colors));
  if (didReadNewCmd) {
      //Serial.println(command);
      parseCommand(command);
      displayColor();
      didReadNewCmd = false;
      command = "";
    }

  if (colorChanged) {
    displayColor();
    colorChanged = false;
  }
}

// check for received bluetooth commands
void serialEvent() {
  if (Serial.available() && waitsForCmd == false) {
    char c = Serial.read();
    if (c == startHandshakeChar) {
      Timer1.detachInterrupt();
      handshakeStart = timerCount;
      waitsForCmd = true;
      Serial.println(startHandshakeChar);
    } 
  }
  
  while (Serial.available() && waitsForCmd == true) {
    char c = Serial.read();
    if (c == terminator) {
      didReadNewCmd = true;
      waitsForCmd = false;
      Timer1.attachInterrupt(timer_isr);
      Serial.println(endHandshakeChar);    
      break;
      }
    command += c;  
    }
}

void timer_isr(void) {
  timerCount += 1;

  // Consider reenabling dependent on timer attach/detach while waiting for command
  //if (waitsForCmd) {handleHandshakeTimeout();}
  handleColorChange();
  handleStrobo();
}

void handleStrobo() {
  handleStroboForColor(&strobo.red);
  handleStroboForColor(&strobo.green);
  handleStroboForColor(&strobo.blue);
  handleStroboForColor(&strobo.white);
  handleStroboForColor(&strobo.master);
}

void handleStroboForColor(stroboColor *color) {
  if (color->enabled) {
    handleOnOff(color);
    color->timer += 1L;
  }
}

void handleOnOff(stroboColor *color) {
  // handle offset first
  if (color->timer == color->offset) {color->isOffset = false;}
  if (color->isOffset) {return;}
  if (color->on) {
    if (color->timer == color->onInterval) {
      switchOnOff(color);
      colorChanged = true;
    }
  } else {
    if (color->timer == color->offInterval) {
      switchOnOff(color);
      colorChanged = true;
    }
  }
}

void switchOnOff(stroboColor *color) {
  color->timer = 0;
  color->on = !color->on;
}

// TODO refactor dim to work like strobo (structs with indiv timers)
void handleColorChange() {
  // handle dim
  // 101 - dim_speed: high speed value -> high dim, 101 because 100 - 100 would result in % 0
  if (dim_enabled_master) {
    if((timerCount % (101 - dim_speed_master) == 0) && dim_speed_master != 0) {
      dim("m");
      colorChanged = true;
    }
  } else {
      if (((timerCount % (101 -  dim_speed_red)) == 0) && dim_enabled_red && dim_speed_red != 0) {
        dim("r");
        colorChanged = true;
      }
      if (((timerCount % (101 -  dim_speed_blue)) == 0) && dim_enabled_blue && dim_speed_blue != 0) {
        dim("b");
        colorChanged = true;
      }
      if (((timerCount % (101 -  dim_speed_green)) == 0) && dim_enabled_green && dim_speed_green != 0) {
        dim("g");
        colorChanged = true;
      }
      if (((timerCount % (101 -  dim_speed_white)) == 0) && dim_enabled_white && dim_speed_white != 0) {
        dim("w");
        colorChanged = true;
      }
  }
}

void handleHandshakeTimeout() {
  // timeout after 2 seconds
  // TODO handle overflow
  if (timerCount > handshakeStart + 2000000L) {
    waitsForCmd = false;
    Serial.println("timeout");
  }
}

 void displayColor() {
  if (waitsForCmd == true) {return;}
  //red and green switched (some strips seem to do that)
  int red = 0;
  int blue = 0;
  int green = 0;
  int white = 0; 

  if (!strobo.master.on && strobo.master.enabled) {
    strip.fill(strip.Color(0,0,0,0));
    return;
  }

  if (strobo.red.on) {
    red = gammaCorrection(colors.red);
  }
  if (strobo.green.on) {
    green = gammaCorrection(colors.green);
  }
  if (strobo.blue.on) {
    blue = gammaCorrection(colors.blue);
  }  
  if (strobo.white.on) {
    white = gammaCorrection(colors.white);
  }
  
  uint32_t color = strip.Color(green, red, blue, white);
  strip.fill(color);
  strip.show();
 }

 void dim(String color) {
  if (color == "m") {
      adjustColorValue(dim_up.redUp, colors.red);
      adjustColorValue(dim_up.blueUp, colors.blue);
      adjustColorValue(dim_up.greenUp, colors.green);
      adjustColorValue(dim_up.whiteUp, colors.white);
  } else if (color == "r") {
      adjustColorValue(dim_up.redUp, colors.red);
  } else if (color == "b") {
      adjustColorValue(dim_up.blueUp, colors.blue);
  } else if (color == "g") {
      adjustColorValue(dim_up.greenUp, colors.green);
  } else if (color == "w") {
      adjustColorValue(dim_up.whiteUp, colors.white);
  }
 }

 int gammaCorrection(int color) {
  if (color < lowestColorValue && color > 0) {
    color = lowestColorValue;
  }
  int adjustedColor = pgm_read_byte(&gamma8[color]);
  return adjustedColor;
 }
 
 void adjustColorValue(bool &isUp, int &value) {
    if (isUp == true) {
      value+=1;
      if (value >= 255) {
        isUp = false;
      } 
    } else {
      value-=1;
      // 28 because everything below is off due to gamme correction and looks bad)
      if (value<=lowestColorValue) {
        isUp = true;
      }
    }
  }
