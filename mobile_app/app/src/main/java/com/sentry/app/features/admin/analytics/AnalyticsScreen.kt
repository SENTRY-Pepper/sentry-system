package com.sentry.app.features.admin.analytics

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun AnalyticsScreen(
    onBack: () -> Unit,
    vm: AnalyticsViewModel = hiltViewModel(),
) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Analytics — coming in sprint 4",
             style = MaterialTheme.typography.bodyLarge)
    }
}