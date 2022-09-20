package com.example.bop.ledapp

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothSocket
import android.util.Log
import java.io.*
import java.util.*
import kotlinx.coroutines.*

// TODO handle handle multiple command while working for handshake?

class BtManager {

    private var connected = false

    val isEnabled: Boolean
        get() = btAdapter.isEnabled

    companion object {
        private var btAdapter = BluetoothAdapter.getDefaultAdapter()
        private var btSocket: BluetoothSocket? = null
        private var outputWriter: BufferedWriter? = null
        private var btOutputStream: OutputStream? = null
        private var btInputStream : InputStream? = null
        private var btInputReader: BufferedReader? = null
        private val hc05Adress = "20:18:07:13:70:88"
        private val uuid = UUID.randomUUID()

        private var command = ""
        private const val beginHandshakeChar = 'x'
        private const val endHandshakeChar = 'y'

        private var callback: (() -> Unit)? = null
    }

    fun connect(): Boolean {
        resetConnection()
        val device = btAdapter.getRemoteDevice(hc05Adress)
        try {
            btSocket = device.createInsecureRfcommSocketToServiceRecord(uuid)
            btSocket!!.connect()
            return true
        } catch (e: IOException) {
            // fallback option due to bug in android bluetooth
            Log.v("BTTAG", "trying fallback...")
            btSocket = device.javaClass.getMethod("createRfcommSocket", (Int::class
                .javaPrimitiveType)).invoke(device, 1) as BluetoothSocket
            var sucess = false
            try {
                btSocket!!.connect()
                connected = true
                Log.v("BTTAG", "connected")
                sucess = true

            } catch (e: IOException) {
                Log.v("BTTAG","fallback option failed")
            }
            return sucess
        }
    }

    fun configureConnection() {
        btInputStream = btSocket!!.inputStream
        btOutputStream = btSocket!!.outputStream
        outputWriter = BufferedWriter(OutputStreamWriter(btOutputStream))
        btInputReader = BufferedReader(InputStreamReader(btInputStream))
    }

    fun sendCommand(s: String) {
        // attach termination symbol
        command = "$s!"
        handshake()
    }

    fun close() {
        resetConnection()
    }

    fun setCallback(callbackFunction: () -> Unit) {
        callback = callbackFunction
    }

    private fun sendCmdString() {
        send(command)
        //TODO ensure callback is always set
        waitForChar(endHandshakeChar, callback!!)
    }

    private fun sendHandshakeChar(hsChar: Char) {
        send(hsChar.toString())
    }

    private fun send(s: String) {
        if (outputWriter != null) {
            try {
                outputWriter!!.write(s)
                outputWriter!!.flush()
            } catch (e: IOException) {
                Log.v("BTTAG", "error sending")
                return
            }
            Log.v("BTTAG", "send succesfull!")
        }
    }

    private fun resetConnection() {
        btOutputStream?.let {
            close()
        }
        btOutputStream = null
        btInputStream?.let {
            close()
        }
        btInputStream = null
        btSocket?.let {
            close()
        }
        btSocket = null
    }

    // make sure strip is ready to receive data
    private fun handshake() {
        sendHandshakeChar(beginHandshakeChar)
        waitForChar(beginHandshakeChar, ::sendCmdString)
    }

    private fun waitForChar(char :Char, callback: () -> Unit) {
        GlobalScope.launch {
            // wait up to two seconds for response
            repeat(10) {
                delay(200)
                // read everything in buffer in case other chars are for some reason there
                while (btInputStream!!.available() > 0) {
                    val receivedC = btInputStream!!.read().toChar()
                    if (receivedC == char) {
                        callback()
                        return@launch
                    }
                }
            }
        }
    }
}