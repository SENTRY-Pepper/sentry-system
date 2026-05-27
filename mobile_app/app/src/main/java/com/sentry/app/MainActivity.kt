package com.sentry.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.compose.rememberNavController
import com.sentry.app.core.navigation.AppNavGraph
import com.sentry.app.ui.theme.SentryTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SentryTheme {
                val navController = rememberNavController()
                AppNavGraph(navController = navController)
            }
        }
    }
}