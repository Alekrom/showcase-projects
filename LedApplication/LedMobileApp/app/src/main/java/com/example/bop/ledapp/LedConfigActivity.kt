package com.example.bop.ledapp

import android.app.Activity
import android.content.DialogInterface
import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import androidx.appcompat.app.AlertDialog
import com.example.bop.ledapp.Util.ColorConfig
import com.example.bop.ledapp.Util.RGBWColor
import com.google.gson.Gson
import com.skydoves.colorpickerview.ColorEnvelope
import com.skydoves.colorpickerview.ColorPickerDialog
import com.skydoves.colorpickerview.listeners.ColorEnvelopeListener
import kotlinx.android.synthetic.main.activity_led_config.*
import java.lang.NumberFormatException

// TODO limit values to 255

class LedConfigActivity : Activity() {

    private var commandQueueManager = CommandQueueManager()
    private var colorConfig: ColorConfig = ColorConfig()
    private val gson = Gson()

    var selectedColor = RGBWColor()
        set(value) {
            if (lockWhiteSwitch.isChecked) {
                value.white = field.white
            }
            field = value
            setColorTxtFields()
            updateColorConfig()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_led_config)

        val extras = intent.extras
        extras?.let {
            if (it.containsKey("config")) {
                colorConfig = it.getSerializable("config") as ColorConfig
            }
        }

        restoreUIValuesFromConfig()

        backBtn.setOnClickListener {
            var intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        colorPickBtn.setOnClickListener {
            displayColorPickerDialog()
        }

        dimFeatureBtn.setOnClickListener {
            var intent = Intent(this, DimConfigActivity::class.java)
            intent.putExtra("config", colorConfig)
            startActivity(intent)
        }

        stroboFeatureBtn.setOnClickListener {
            var intent = Intent(this, StroboFeatureActivity::class.java)
            intent.putExtra("config", colorConfig)
            startActivity(intent)
        }

        initListener()
        setColorTxtFields()
    }

    //TODO customize color picker
    fun displayColorPickerDialog() {
        ColorPickerDialog.Builder(this, AlertDialog.BUTTON_NEUTRAL).apply {
            setTitle("Color Picker Dialog")
            setPreferenceName("MyColorPickerDialog")
            setPositiveButton(getString(R.string.confirm),
                (object: ColorEnvelopeListener {
                    override fun onColorSelected(envelope: ColorEnvelope, fromUser: Boolean) {
                        selectedColor = RGBWColor().apply {
                            fromHexString(envelope.hexCode)
                        }
                        sendColor()
                        Log.v("ColorTag", envelope.hexCode)
                    }
                }))

            setNegativeButton(getString(R.string.cancel),
                (object: DialogInterface.OnClickListener {
                    override fun onClick(dialogInterface: DialogInterface, i: Int) {
                        dialogInterface.dismiss()
                    }
                }))
            attachAlphaSlideBar(true) // default is true. If false, do not show the AlphaSlideBar.
            attachBrightnessSlideBar(true)
            show()
        }
    }

    private fun setColorTxtFields() {
        txtInputRed.setText(selectedColor.red.toString())
        txtInputGreen.setText(selectedColor.green.toString())
        txtInputBlue.setText(selectedColor.blue.toString())
        txtInputWhite.setText(selectedColor.white.toString())
    }

    private fun updateColorConfig() {
        colorConfig.color = selectedColor
    }

    private fun initListener() {

        //TODO refactor to one watcher??
        //TODO error handling if text
        txtInputRed.addTextChangedListener(object: TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                // pass
            }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                // pass
            }

            override fun afterTextChanged(s: Editable?) {
                try {
                    selectedColor.red = Integer.parseInt(s.toString())
                } catch (e: NumberFormatException) {
                    selectedColor.red = 0
                }
                sendColor()
            }
        })

        txtInputGreen.addTextChangedListener(object: TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                // pass
            }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                // pass
            }

            override fun afterTextChanged(s: Editable?) {
                try {
                    selectedColor.green = Integer.parseInt(s.toString())
                } catch (e: NumberFormatException) {
                    selectedColor.green = 0
                }
                sendColor()
            }
        })

        txtInputBlue.addTextChangedListener(object: TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                // pass
            }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                // pass
            }

            override fun afterTextChanged(s: Editable?) {
                try {
                    selectedColor.blue = Integer.parseInt(s.toString())
                } catch (e: NumberFormatException) {
                    selectedColor.blue = 0
                }
                sendColor()
            }
        })

        txtInputWhite.addTextChangedListener(object: TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                // pass
            }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                // pass
            }

            override fun afterTextChanged(s: Editable?) {
                try {
                    selectedColor.white = Integer.parseInt(s.toString())
                } catch (e: NumberFormatException) {
                    selectedColor.white = 0
                }
                sendColor()
            }
        })
    }

    fun restoreUIValuesFromConfig() {
        selectedColor = colorConfig.color
        setColorTxtFields()
    }

    fun sendColor() {
        val strRed = selectedColor.red.toString().padStart(3, '0')
        val strGreen = selectedColor.green.toString().padStart(3, '0')
        val strBlue = selectedColor.blue.toString().padStart(3, '0')
        val strWhite = selectedColor.white.toString().padStart(3, '0')
        val btCommand = "cr" + strRed + "g" + strGreen + "b" + strBlue + "w" + strWhite
        commandQueueManager.enqueue(btCommand)
    }
}
