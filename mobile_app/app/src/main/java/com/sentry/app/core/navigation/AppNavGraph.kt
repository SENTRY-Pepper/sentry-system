package com.sentry.app.core.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.sentry.app.features.admin.analytics.AnalyticsScreen
import com.sentry.app.features.admin.home.AdminHomeScreen
import com.sentry.app.features.auth.AuthScreen
import com.sentry.app.features.auth.AuthViewModel
import com.sentry.app.features.chat.ChatScreen
import com.sentry.app.features.settings.SettingsScreen
import com.sentry.app.features.splash.SplashScreen
import com.sentry.app.features.trainee.home.TraineeHomeScreen
import com.sentry.app.features.trainee.results.ResultsScreen
import com.sentry.app.features.trainee.session.SessionScreen

@Composable
fun AppNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Routes.SPLASH,
    ) {
        composable(Routes.SPLASH) {
            SplashScreen(
                onAuthenticated = { role ->
                    val dest = if (role == UserRole.ADMIN) Routes.ADMIN_HOME
                               else Routes.TRAINEE_HOME
                    navController.navigate(dest) {
                        popUpTo(Routes.SPLASH) { inclusive = true }
                    }
                },
                onUnauthenticated = {
                    navController.navigate(Routes.AUTH) {
                        popUpTo(Routes.SPLASH) { inclusive = true }
                    }
                },
            )
        }

        composable(Routes.AUTH) {
            val vm: AuthViewModel = hiltViewModel()
            val state by vm.uiState.collectAsStateWithLifecycle()
            AuthScreen(
                state = state,
                onLoginClick = { id, pin, role, org -> vm.login(id, pin, role, org) },
                onLoginSuccess = { role ->
                    val dest = if (role == UserRole.ADMIN) Routes.ADMIN_HOME
                               else Routes.TRAINEE_HOME
                    navController.navigate(dest) {
                        popUpTo(Routes.AUTH) { inclusive = true }
                    }
                },
            )
        }

        composable(Routes.TRAINEE_HOME) {
            TraineeHomeScreen(
                onStartSession = { sessionId ->
                    navController.navigate(Routes.session(sessionId))
                },
                onOpenChat = { navController.navigate(Routes.CHAT) },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }

        composable(
            route     = Routes.SESSION,
            arguments = listOf(navArgument("sessionId") { type = NavType.StringType }),
        ) {
            SessionScreen(
                onSessionComplete = { sessionId ->
                    navController.navigate(Routes.results(sessionId)) {
                        popUpTo(Routes.TRAINEE_HOME)
                    }
                },
                onBack = { navController.popBackStack() },
            )
        }

        composable(
            route     = Routes.RESULTS,
            arguments = listOf(navArgument("sessionId") { type = NavType.StringType }),
        ) {
            ResultsScreen(
                onDone     = {
                    navController.navigate(Routes.TRAINEE_HOME) {
                        popUpTo(Routes.TRAINEE_HOME) { inclusive = true }
                    }
                },
                onOpenChat = { navController.navigate(Routes.CHAT) },
            )
        }

        composable(Routes.ADMIN_HOME) {
            AdminHomeScreen(
                onOpenAnalytics = { navController.navigate(Routes.ANALYTICS) },
                onOpenChat      = { navController.navigate(Routes.CHAT) },
                onOpenSettings  = { navController.navigate(Routes.SETTINGS) },
            )
        }

        composable(Routes.ANALYTICS) {
            AnalyticsScreen(onBack = { navController.popBackStack() })
        }

        composable(Routes.CHAT) {
            ChatScreen(onBack = { navController.popBackStack() })
        }

        composable(Routes.SETTINGS) {
            SettingsScreen(
                onBack   = { navController.popBackStack() },
                onLogout = {
                    navController.navigate(Routes.AUTH) {
                        popUpTo(0) { inclusive = true }
                    }
                },
            )
        }
    }
}