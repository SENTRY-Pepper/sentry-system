package com.sentry.app.core.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.sentry.app.features.admin.analytics.AnalyticsScreen
import com.sentry.app.features.admin.home.AdminHomeScreen
import com.sentry.app.features.auth.AuthScreen
import com.sentry.app.features.chat.ChatScreen
import com.sentry.app.features.manager.home.ManagerHomeScreen
import com.sentry.app.features.settings.SettingsScreen
import com.sentry.app.features.splash.SplashScreen
import com.sentry.app.features.trainee.home.TraineeHomeScreen
import com.sentry.app.features.trainee.results.ResultsScreen
import com.sentry.app.features.trainee.session.SessionScreen

@Composable
fun AppNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = "splash",
    ) {
        composable("splash") {
            SplashScreen(navController = navController)
        }

        composable("auth") {
            AuthScreen(navController = navController)
        }

        composable("traineeHome") {
            TraineeHomeScreen(navController = navController)
        }

        composable("adminHome") {
            AdminHomeScreen(navController = navController)
        }

        composable("managerHome") {
            ManagerHomeScreen(navController = navController)
        }

        composable("analytics") {
            AnalyticsScreen(navController = navController)
        }

        composable("chat") {
            ChatScreen(navController = navController)
        }

        composable("settings") {
            SettingsScreen(navController = navController)
        }

        // routes with args
        composable("session/{sessionId}") { backStackEntry ->
            val sessionId = backStackEntry.arguments?.getString("sessionId") ?: ""
            SessionScreen(navController = navController, sessionId = sessionId)
        }

        composable("results/{sessionId}") { backStackEntry ->
            val sessionId = backStackEntry.arguments?.getString("sessionId") ?: ""
            ResultsScreen(navController = navController, sessionId = sessionId)
        }
    }
}
