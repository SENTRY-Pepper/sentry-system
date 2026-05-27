package com.sentry.app.features.trainee.home

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.ui.components.*

@Composable
fun TraineeHomeScreen(
    onStartSession: (String) -> Unit,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
    vm: TraineeHomeViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(state.sessionStarted) {
        state.sessionStarted?.let { onStartSession(it) }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Welcome back", style = MaterialTheme.typography.headlineMedium)
        Text("Ready for your cybersecurity training?",
             style = MaterialTheme.typography.bodyMedium)

        Spacer(Modifier.height(16.dp))

        ErrorBanner(message = state.error)

        SentryButton(
            text    = "Start training session",
            loading = state.loading,
            onClick = { vm.startSession() },
        )

        OutlinedButton(
            onClick  = onOpenChat,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            Text("Chat with Pepper")
        }
    }

    LoadingOverlay(visible = state.loading)
}