package com.example.bop.ledapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.CompoundButton
import android.widget.EditText
import android.widget.Switch
import com.example.bop.ledapp.Util.ColorConfig
import com.example.bop.ledapp.Util.getEditTextAsInt
import com.example.bop.ledapp.Util.StroboEditTextWatcher
import kotlinx.android.synthetic.main.activity_strobo.*

class StroboFeatureActivity : Activity() {

    private val commandQueueManager = CommandQueueManager()

    var config: ColorConfig = ColorConfig()
    private val switches = mutableMapOf<String, Switch>()
    private val offsets = mutableMapOf<String, EditText>()
    private val onIntervals = mutableMapOf<String, EditText>()
    private val offIntervals = mutableMapOf<String, EditText>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_strobo)

        val extras = intent.extras
        extras?.let {
            if (it.containsKey("config")) {
                config = it.getSerializable("config") as ColorConfig
            }
        }

        stroboBackBtn.setOnClickListener {
            var intent = Intent(this, LedConfigActivity::class.java)
            intent.putExtra("config", config)
            startActivity(intent)
        }
        createUIElementLists()
        restoreUIValuesFromConfig()
        setListeners()
    }

    fun updateStroboConfig() {
        // TODO rework

        for ((color, switch) in switches) {
            config.strobo.enabled[color] = switch.isChecked
        }
        for ((color, interval) in onIntervals) {
            config.strobo.onDur[color] = getEditTextAsInt(interval)
        }
        for ((color, interval) in offIntervals) {
            config.strobo.offDur[color] = getEditTextAsInt(interval)
        }
        for ((color, offset) in offsets) {
            config.strobo.offsets[color] =  getEditTextAsInt(offset)
        }
    }

    fun updateUI() {
        // Handle Master Switch
        if (stroboMasterSwitch.isChecked) {
            for ((_, switch) in switches) {
                switch.isEnabled = switch == stroboMasterSwitch
            }
            for ((_, interval) in onIntervals) {
                interval.isEnabled = interval == stroboMasterOn
            }

            for ((_, interval) in offIntervals) {
                interval.isEnabled = interval == stroboMasterOff
            }

            for ((_, offset) in offsets) {
                offset.isEnabled = offset == stroboMasterOffset
            }
        } else {
            for ((_, switch) in switches) {
                switch.isEnabled = true
            }
            for ((_, interval) in onIntervals) {
                interval.isEnabled = interval != stroboMasterOn
            }
            for ((_, interval) in offIntervals) {
                interval.isEnabled = interval != stroboMasterOff
            }
            for ((_, offset) in offsets) {
                offset.isEnabled = offset != stroboMasterOffset
            }
        }

        for ((color, switch) in switches) {
            onIntervals[color]?.isEnabled = (switch.isChecked and switch.isEnabled)
            offIntervals[color]?.isEnabled = (switch.isChecked and switch.isEnabled)
            offsets[color]?.isEnabled = (switch.isChecked and switch.isEnabled)
        }
    }

    fun restoreUIValuesFromConfig() {
        val stroboConfig = config.strobo
        for ((color, switch) in switches) {
            switch.isChecked = stroboConfig.enabled[color]!!
        }
        for ((color, interval) in onIntervals) {
            interval.setText(stroboConfig.onDur[color].toString())
        }
        for ((color, interval) in offIntervals) {
            interval.setText(stroboConfig.offDur[color].toString())
        }
        for ((color, interval) in offsets) {
            interval.setText(stroboConfig.offsets[color].toString())
        }
        updateUI()
    }

    fun setListeners() {
        for ((color, switch) in switches) {
            switch.setOnCheckedChangeListener { switchObj, isChecked ->
                sendSwitchBtCommand(switchObj, isChecked)
                updateUI()
                updateStroboConfig()
            }
        }

        for ((color, edit) in onIntervals) {
            edit.addTextChangedListener(StroboEditTextWatcher(color, "on", ::updateEditTextChange))
        }

        for ((color, edit) in offIntervals) {
            edit.addTextChangedListener(StroboEditTextWatcher(color, "off", ::updateEditTextChange))
        }

        for ((color, edit) in offsets) {
            edit.addTextChangedListener(StroboEditTextWatcher(color, "offset", ::updateEditTextChange))
        }
    }

    private fun sendSwitchBtCommand(switch: CompoundButton, isChecked: Boolean) {
        val filteredSwitched = switches.filterValues { it == switch as Switch }
        val color = filteredSwitched.toList()[0].first
        var checkedString = ""
        if (isChecked) {
            checkedString = "1"
        } else {
            checkedString = "0"
        }
        val btCommand = "sc$color$checkedString"
        commandQueueManager.enqueue(btCommand)
    }

    private fun updateEditTextChange(color: String, type: String, value: String) {
        sendEditBtCommand(color, type, value)
        updateStroboConfig()
    }

    private fun sendEditBtCommand(color: String, type: String, value: String) {
        val valueString = value.padStart(3, '0')
        val typeChar = when (type) {
            "on" -> '+'
            "off" -> '-'
            else -> 'o'
        }
        val btCommand = "sv$color$typeChar$valueString"
        commandQueueManager.enqueue(btCommand)
    }

    fun createUIElementLists() {
        switches["r"] = stroboRedSwitch
        switches["b"] = stroboBlueSwitch
        switches["g"] = stroboGreenSwitch
        switches["w"] = stroboWhiteSwitch
        switches["m"] = stroboMasterSwitch

        onIntervals["r"] = stroboRedOn
        onIntervals["b"] = stroboBlueOn
        onIntervals["g"] = stroboGreenOn
        onIntervals["w"] = stroboWhiteOn
        onIntervals["m"] = stroboMasterOn

        offIntervals["r"] = stroboRedOff
        offIntervals["b"] = stroboBlueOff
        offIntervals["g"] = stroboGreenOff
        offIntervals["w"] = stroboWhiteOff
        offIntervals["m"] = stroboMasterOff

        offsets["r"] = stroboRedOffset
        offsets["b"] = stroboBlueOffset
        offsets["g"] = stroboGreenOffset
        offsets["w"] = stroboWhiteOffset
        offsets["m"] = stroboMasterOffset
    }
}