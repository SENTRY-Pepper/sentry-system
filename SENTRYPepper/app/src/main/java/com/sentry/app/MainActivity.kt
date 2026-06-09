package com.sentry.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.compose.rememberNavController
import com.aldebaran.qi.sdk.QiContext
import com.aldebaran.qi.sdk.QiSDK
import com.aldebaran.qi.sdk.RobotLifecycleCallbacks
import com.sentry.app.core.navigation.AppNavGraph
import com.sentry.app.core.ui.theme.SentryTheme
import com.sentry.app.pepper.PepperRobotBridge
import dagger.hilt.android.AndroidEntryPoint
import timber.log.Timber

@AndroidEntryPoint
class MainActivity : ComponentActivity(), RobotLifecycleCallbacks {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        QiSDK.register(this, this)
        setContent {
            SentryTheme {
                val navController = rememberNavController()
                AppNavGraph(navController = navController)
            }
        }
    }

    override fun onDestroy() {
        QiSDK.unregister(this, this)
        PepperRobotBridge.detach()
        super.onDestroy()
    }

    override fun onRobotFocusGained(qiContext: QiContext) {
        PepperRobotBridge.attach(qiContext)
        PepperRobotBridge.say(
            "Hello. I am Pepper, your SENTRY cybersecurity training assistant."
        )
    }

    override fun onRobotFocusLost() {
        PepperRobotBridge.detach()
    }

    override fun onRobotFocusRefused(reason: String) {
        Timber.w("Pepper robot focus refused: %s", reason)
    }
}
