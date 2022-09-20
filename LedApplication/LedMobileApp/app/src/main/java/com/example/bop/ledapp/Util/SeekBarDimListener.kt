package com.example.bop.ledapp.Util

import android.widget.SeekBar

class SeekBarDimListener (val callback: (SeekBar, Int) -> Unit) : SeekBar.OnSeekBarChangeListener {
    override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {
        callback(seekBar, i)
    }

    override fun onStartTrackingTouch(seekBar: SeekBar) {
        // Not needed
    }

    override fun onStopTrackingTouch(seekBar: SeekBar) {
        // Not needed
    }
}