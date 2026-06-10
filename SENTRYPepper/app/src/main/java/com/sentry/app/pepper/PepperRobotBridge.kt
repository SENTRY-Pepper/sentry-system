package com.sentry.app.pepper

import com.aldebaran.qi.sdk.QiContext
import com.aldebaran.qi.sdk.builder.SayBuilder
import com.aldebaran.qi.Future
import timber.log.Timber
import java.util.concurrent.Executors

object PepperRobotBridge {
    private val speechExecutor = Executors.newSingleThreadExecutor()
    private val speechLock = Any()

    @Volatile
    private var qiContext: QiContext? = null
    private var currentSpeech: Future<Void>? = null
    private var speechSequence: Long = 0

    fun attach(context: QiContext) {
        qiContext = context
        Timber.i("Pepper robot focus attached")
    }

    fun detach() {
        stopSpeaking()
        qiContext = null
        Timber.i("Pepper robot focus detached")
    }

    fun say(text: String): Boolean {
        val context = qiContext ?: return false
        val safeText = text.trim().take(700)
        if (safeText.isBlank()) return false
        val requestId = synchronized(speechLock) {
            cancelCurrentSpeechLocked()
            speechSequence += 1
            speechSequence
        }

        speechExecutor.execute {
            runCatching {
                val speech = SayBuilder.with(context)
                    .withText(safeText)
                    .build()
                    .async()
                    .run()
                synchronized(speechLock) {
                    if (requestId == speechSequence) {
                        currentSpeech = speech
                    } else {
                        speech.requestCancellation()
                    }
                }
            }.onFailure {
                Timber.w(it, "Pepper QiSDK speech failed")
            }
        }
        return true
    }

    fun stopSpeaking() {
        synchronized(speechLock) {
            speechSequence += 1
            cancelCurrentSpeechLocked()
        }
    }

    private fun cancelCurrentSpeechLocked() {
        currentSpeech?.requestCancellation()
        currentSpeech = null
    }
}
