package com.sentry.app.features.trainee.session

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun SessionScreen(
    onSessionComplete: (String) -> Unit,
    vm: SessionViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(state.isComplete) {
        if (state.isComplete) onSessionComplete(state.sessionId)
    }

    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Session screen — coming in sprint 2",
             style = MaterialTheme.typography.bodyLarge)
    }
}