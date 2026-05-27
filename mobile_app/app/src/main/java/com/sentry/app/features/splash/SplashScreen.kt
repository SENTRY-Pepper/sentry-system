package com.sentry.app.features.splash

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import com.sentry.app.core.navigation.UserRole

// ViewModel injected to check stored auth state on launch
@Composable
fun SplashScreen(
    onAuthenticated: (UserRole) -> Unit,
    onUnauthenticated: () -> Unit,
    vm: SplashViewModel = hiltViewModel(),
) {
    LaunchedEffect(Unit) {
        if (vm.isAuthenticated()) {
            onAuthenticated(vm.getRole())
        } else {
            onUnauthenticated()
        }
    }

    Box(
        modifier         = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.primary),
        contentAlignment = Alignment.Center,
    ) {
        // Logo placeholder — replaced when we design this screen
    }
}