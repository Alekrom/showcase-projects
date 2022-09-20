package com.example.bop.ledapp

import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle

import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*


class MainActivity : AppCompatActivity() {

    //TODO: device pairing (aktuell manuell via bluetooth terminal hc-05

    private val btManager = BtManager()

    private val REQUEST_ENABLE_BT = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        activateBtn.setOnClickListener {
            activate()
        }

    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            connect()
        } else {
            //TODO,unable to activate
            activateBtn.text = "error"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        btManager.close()
        //TODO
        //unregisterReceiver(receiver)
    }

    private fun activate() {
        if (!btManager.isEnabled) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT)
        } else {
            connect()
        }
    }

    private fun connect() {
        val isConnected = btManager.connect()
        if (isConnected) {
            btManager.configureConnection()
            openLedConfig()
        } else {
            //TODO
            //unable to connect to socket
        }

    }

    private fun openLedConfig() {
        val intent = Intent(this, LedConfigActivity::class.java)
        startActivity(intent)
    }

}
