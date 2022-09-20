package com.example.bop.ledapp

import java.util.*

class CommandQueueManager {

    init {
        btManager.setCallback(::commandFinished)
    }

    companion object {
        private val commandQueue: Queue<String> = LinkedList<String>()
        private val btManager = BtManager()
        private var isProcessingCommands = false
    }

    fun enqueue(command: String) {
        commandQueue.add(command)
        if (!isProcessingCommands) {
            sendCommand()
        }
    }

    private fun sendCommand() {
        if (commandQueue.isEmpty()) {
            println("trying to send command with queue empty")
            isProcessingCommands = false
            return
        }
        isProcessingCommands = true
        btManager.sendCommand(commandQueue.remove())
    }

    private fun commandFinished() {
        if (!commandQueue.isEmpty()) {
            sendCommand()
        } else {
            isProcessingCommands = false
        }
    }
}