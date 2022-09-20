package com.example.bop.ledapp.Util

import java.io.Serializable

data class ColorConfig(var color: RGBWColor = RGBWColor(), var dim: DimConfig = DimConfig(), var strobo: StroboConfig = StroboConfig()): Serializable