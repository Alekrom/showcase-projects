package com.example.bop.ledapp.Util

import java.io.Serializable
import java.lang.Long.parseLong

class RGBWColor : Serializable {

    var white: Int = 0
    var red: Int = 0
    var green: Int = 0
    var blue: Int = 0

    // builds rgbw from  hexstring, first two digits are w
    fun fromHexString(hexString: String) {
        white = parseLong(hexString.substring(0, 2), 16).toInt()
        red = parseLong(hexString.substring(2, 4), 16).toInt()
        green = parseLong(hexString.substring(4, 6), 16).toInt()
        blue = parseLong(hexString.substring(6, 8), 16).toInt()
    }
}