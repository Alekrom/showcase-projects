package com.example.bop.ledapp.Util

import java.io.Serializable

data class StroboConfig(val enabled: MutableMap<String, Boolean> = mutableMapOf("r" to false, "b" to false, "g" to false, "w" to false, "m" to false),
                        val onDur: MutableMap<String, Int> = mutableMapOf("r" to 0, "b" to 0, "g" to 0, "w" to 0, "m" to 0),
                        val offDur: MutableMap<String, Int> = mutableMapOf("r" to 0, "b" to 0, "g" to 0, "w" to 0, "m" to 0),
                        val offsets: MutableMap<String, Int> = mutableMapOf("r" to 0, "b" to 0, "g" to 0, "w" to 0, "m" to 0)) : Serializable