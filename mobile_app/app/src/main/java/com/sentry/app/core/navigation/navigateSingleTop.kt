package com.sentry.app.core.navigation

import androidx.navigation.NavHostController

/**
 * Navigate to a route launching it single-top.
 * Does NOT pop the back stack — use popUpTo at the call site when needed.
 */
fun NavHostController.navigateSingleTop(route: String) {
    navigate(route) {
        launchSingleTop = true
    }
}

/**
 * Navigate to a route and clear everything in the back stack up to
 * and including [popUpToRoute]. Use this for auth → home transitions
 * and logout where you don't want the user pressing back into a previous screen.
 */
fun NavHostController.navigateAndClear(
    route: String,
    popUpToRoute: String,
    inclusive: Boolean = true,
) {
    navigate(route) {
        launchSingleTop = true
        popUpTo(popUpToRoute) { this.inclusive = inclusive }
    }
}