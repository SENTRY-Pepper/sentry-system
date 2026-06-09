package com.sentry.pepper

import android.os.Bundle
import androidx.activity.ComponentActivity
import com.aldebaran.qi.sdk.QiContext
import com.aldebaran.qi.sdk.QiSDK
import com.aldebaran.qi.sdk.RobotLifecycleCallbacks
import com.aldebaran.qi.sdk.builder.SayBuilder

class MainActivity : ComponentActivity(), RobotLifecycleCallbacks {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        QiSDK.register(this, this)
    }

    override fun onDestroy() {
        QiSDK.unregister(this, this)
        super.onDestroy()
    }

    override fun onRobotFocusGained(qiContext: QiContext?) {
        val say = SayBuilder.with(qiContext)
            .withText("Hello human! How can I help you today?")
            .build()
        say.run()
    }

    override fun onRobotFocusLost() {}
    override fun onRobotFocusRefused(reason: String?) {}
}