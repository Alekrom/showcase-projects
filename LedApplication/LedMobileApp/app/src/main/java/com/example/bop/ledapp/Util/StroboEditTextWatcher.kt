package com.example.bop.ledapp.Util

import android.text.Editable
import android.text.TextWatcher

class StroboEditTextWatcher(private val color: String, private val type: String, val callback: (String, String, String) -> Unit) : TextWatcher {

    override fun afterTextChanged(p0: Editable?) {
        callback(color, type, p0.toString())
    }

    override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {
    }

    override fun onTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {
    }

}