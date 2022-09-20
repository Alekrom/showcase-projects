package com.example.bop.ledapp.Util

import java.io.Serializable

data class DimConfig(val enabled: MutableMap<String,Boolean> = mutableMapOf("r" to false, "b" to false, "g" to false, "w" to false, "m" to false),
                     val speed: MutableMap<String,Int> = mutableMapOf("r" to 0, "b" to 0, "g" to 0, "w" to 0, "m" to 0), var keepColors: Boolean = false): Serializable