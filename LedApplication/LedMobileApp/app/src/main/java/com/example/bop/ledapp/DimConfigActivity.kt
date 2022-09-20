package com.example.bop.ledapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.CompoundButton
import android.widget.SeekBar
import android.widget.Switch
import com.example.bop.ledapp.Util.ColorConfig
import com.example.bop.ledapp.Util.SeekBarDimListener
import kotlinx.android.synthetic.main.activity_dim.*

class DimConfigActivity: Activity() {

    private val commandQueueManager = CommandQueueManager()

    private var config = ColorConfig()
    private val bars = mutableMapOf<String, SeekBar>()
    private val switches = mutableMapOf<String, Switch>()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dim)
        val extras = intent.extras
        extras?.let {
            if (it.containsKey("config")) {
                config = it.getSerializable("config") as ColorConfig
            }
        }
        dimBackBtn.setOnClickListener {
            var intent = Intent(this, LedConfigActivity::class.java)
            intent.putExtra("config", config)
            startActivity(intent)
        }

        bars["r"] = dimRedBar
        bars["b"] = dimBlueBar
        bars["g"] = dimGreenBar
        bars["w"] = dimWhiteBar
        bars["m"] = dimMasterBar

        switches["r"] = dimSwitchRed
        switches["b"] = dimSwitchBlue
        switches["g"] = dimSwitchGreen
        switches["w"] = dimSwitchWhite
        switches["m"] = dimMasterSpeedSwitch

        restoreUIValuesFromConfig()

        for ((color, bar) in bars) {
            bar.setOnSeekBarChangeListener(SeekBarDimListener(::updateChangeFromBar))
        }

        for ((color, switch) in switches) {
            switch.setOnCheckedChangeListener { switchObj, isChecked ->
                sendSwitchBtCommand(switchObj, isChecked)
                updateUI()
                updateDimConfig()
            }
        }

        dimKeepColorsSwitch.setOnCheckedChangeListener { _, isChecked ->
            var checkedString = "0"
            if (isChecked) checkedString = "1"
            val btCommand = "dk" + checkedString
            commandQueueManager.enqueue(btCommand)
            updateUI()
            updateDimConfig()
        }
    }

    private fun updateUI() {
        // Handle master switch
        if (dimMasterSpeedSwitch.isChecked) {
            for ((color, bar) in bars) {
                bar.isEnabled = bar == dimMasterBar
            }
            for ((color, switch) in switches) {
                switch.isEnabled = switch == dimMasterSpeedSwitch
            }
            return
        }

        if (dimKeepColorsSwitch.isChecked) {
            for ((color, bar) in bars) {
                bar.isEnabled = bar == dimMasterBar
            }
            for ((_, switch) in switches) {
                switch.isEnabled = switch == dimMasterSpeedSwitch
            }
        } else {
            for ((color, switch) in switches) {
                bars[color]?.isEnabled = switch.isChecked
                switch.isEnabled = !dimKeepColorsSwitch.isChecked
            }
        }
    }

    private fun updateChangeFromBar(seekBar: SeekBar, value: Int) {
        sendBarBtCommand(seekBar, value)
        updateDimConfig()
    }

    private fun updateDimConfig() {
        //TODO rework
        for ((color, switch) in switches) {
            config.dim.enabled[color] = switch.isChecked
        }
        for ((color, bar) in bars) {
            config.dim.speed[color] = bar.progress
        }
        config.dim.keepColors = dimKeepColorsSwitch.isChecked
    }

    private fun restoreUIValuesFromConfig() {
        var dimConfig = config.dim
        for ((color, bar) in bars) {
            bar.progress = dimConfig.speed[color]!!
        }
        for ((color, switch) in switches) {
            switch.isChecked = dimConfig.enabled[color]!!
        }
        dimKeepColorsSwitch.isChecked = dimConfig.keepColors
        updateUI()
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
        val btCommand = "dc$color$checkedString"
        commandQueueManager.enqueue(btCommand)
    }

    private fun sendBarBtCommand(seekBar: SeekBar, value: Int) {
        val filteredBars = bars.filterValues { it == seekBar }
        val color = filteredBars.toList()[0].first
        val valueString = value.toString().padStart(3, '0')
        val btCommand = "ds$color$valueString"
        commandQueueManager.enqueue(btCommand)
    }
}