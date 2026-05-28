package com.sentry.app.features.trainee.results

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun ResultsScreen(
    onDone: () -> Unit,
    onOpenChat: () -> Unit,
    vm: ResultsViewModel = hiltViewModel(),
) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Results screen — coming in sprint 3",
             style = MaterialTheme.typography.bodyLarge)
    }
}