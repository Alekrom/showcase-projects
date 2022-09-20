
#ifndef STRUCTS_HEADER
#define STRUCTS_HEADER

#include "structs.h"

#endif

#ifndef CONFIG_HEADER
#define CONFIG_HEADER

#include "config_data.h" 

#endif

#ifndef EEPROM_HEADER
#define EEPROM_HEADER

#include <avr/eeprom.h>

#endif

void parseColorCmd(String command) {
    String red = command.substring(2, 5);
    colors.red = red.toInt();
      
    String green = command.substring(6, 9);
    colors.green = green.toInt();
      
    String blue = command.substring(10, 13);
    colors.blue = blue.toInt();

    String white = command.substring(14, 17);
    colors.white = white.toInt();
}

//TODO refactor
void parseDimCmd(String command) {
    if (command[1] == 's') {
      if (command[2] == 'r') {
        dim_speed_red = command.substring(3, 6).toInt();
      } else if (command[2] == 'g') {
        dim_speed_green = command.substring(3, 6).toInt();
      } else if (command[2] == 'b') {
        dim_speed_blue = command.substring(3, 6).toInt();
      } else if (command[2] == 'w') {
        dim_speed_white = command.substring(3, 6).toInt();
      } else if (command[2] == 'm') {
        dim_speed_master = command.substring(3, 6).toInt();
      }
    } else if (command[1] == 'c') {
      if (command[2] == 'r') {
        dim_enabled_red = mapCharToBool(command[3]);
      } else if (command[2] == 'g') {
        dim_enabled_green = mapCharToBool(command[3]);
      } else if (command[2] == 'b') {
        dim_enabled_blue = mapCharToBool(command[3]);
      } else if (command[2] == 'w') {
        dim_enabled_white = mapCharToBool(command[3]);
      } else if (command[2] == 'm') {
        dim_enabled_master = mapCharToBool(command[3]);
      }
    }
}

void parseStroboCmd(String command) {
  if (command[1] == 'c') {
      if (command[2] == 'r') {
        strobo.red.enabled = mapCharToBool(command[3]);
      } else if (command[2] == 'g') {
        strobo.green.enabled = mapCharToBool(command[3]);
      } else if (command[2] == 'b') {
        strobo.blue.enabled = mapCharToBool(command[3]);
      } else if (command[2] == 'w') {
        strobo.white.enabled = mapCharToBool(command[3]);
      } else if (command[2] == 'm') {
        strobo.master.enabled = mapCharToBool(command[3]);
      }
   } else if (command[1] == 'v') {
    if (command[2] == 'r') {
      setStroboValue(&strobo.red, command[3], command.substring(4,7).toInt());
    } else if (command[2] == 'g') {
      setStroboValue(&strobo.green, command[3], command.substring(4,7).toInt());
   } else if (command[2] == 'b') {
      setStroboValue(&strobo.blue, command[3], command.substring(4,7).toInt());
   } else if (command[2] == 'w') {
      setStroboValue(&strobo.white, command[3], command.substring(4,7).toInt());
   } else if (command[2] == 'm') {
      setStroboValue(&strobo.master, command[3], command.substring(4,7).toInt());
   }
 }
}

bool mapCharToBool(char c) {
  if (c == '0') {
    return false;
  } else if (c == '1') {
    return true;
  }
}

void parseCommand(String command) {
     if (command[0] == 'c') {
        parseColorCmd(command);
     } else if (command[0] == 'd') {
      parseDimCmd(command);
     } else if (command[0] == 's') {
      parseStroboCmd(command);
      resetStrobos();
     }
  }

void setStroboValue(stroboColor *color, char type, int value) {
    if (type == '+') {
      color->onInterval= toTimerTicks(value);
      Serial.println("*******");
      Serial.println(color->onInterval);
      Serial.println(value);
      Serial.println(toTimerTicks(value));
    } else if (type == '-') {
      color->offInterval = toTimerTicks(value);
    } else if (type == 'o') {
      color->offset = toTimerTicks(value);
    }
}

//value is in hundreth of seconds
unsigned long toTimerTicks(int value) {
  if (timerInitValue != 100) {
    Serial.println("expected timer value of 100, set value differs");
  }
  // convert to timerticks (=100 microseconds)
  unsigned long timerTicks  = (unsigned long) value * (unsigned long)100;
  return timerTicks;
}

// otherwise signals would not start at same time
void resetStrobos() {
  strobo.red.on = true;
  strobo.red.timer = 0;
  strobo.green.on = true;
  strobo.green.timer = 0;
  strobo.blue.on = true;
  strobo.blue.timer = 0;
  strobo.white.on = true;
  strobo.white.timer = 0;
  strobo.master.on = true;
  strobo.master.timer = 0;
}
  
