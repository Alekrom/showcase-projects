package com.example.bop.ledapp.Util

import android.widget.EditText


fun getEditTextAsInt(edit: EditText): Int {
    val editString = edit.text.toString()
    if (editString != "") {
        return editString.toInt()
    } else {
        return 0
    }
}
