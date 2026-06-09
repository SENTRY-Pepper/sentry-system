package com.sentry.app.pepper

import com.aldebaran.qi.sdk.QiContext
import com.aldebaran.qi.sdk.builder.SayBuilder
import timber.log.Timber
import java.util.concurrent.Executors

object PepperRobotBridge {
    private val speechExecutor = Executors.newSingleThreadExecutor()

    @Volatile
    private var qiContext: QiContext? = null

    fun attach(context: QiContext) {
        qiContext = context
        Timber.i("Pepper robot focus attached")
    }

    fun detach() {
        qiContext = null
        Timber.i("Pepper robot focus detached")
    }

    fun say(text: String): Boolean {
        val context = qiContext ?: return false
        val safeText = text.trim().take(700)
        if (safeText.isBlank()) return false

        speechExecutor.execute {
            runCatching {
                SayBuilder.with(context)
                    .withText(safeText)
                    .build()
                    .run()
            }.onFailure {
                Timber.w(it, "Pepper QiSDK speech failed")
            }
        }
        return true
    }
}
