package com.sentry.app.features.admin.home

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun AdminHomeScreen(
    onOpenAnalytics: () -> Unit,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
    vm: AdminHomeViewModel = hiltViewModel(),
) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Admin home — coming in sprint 4",
             style = MaterialTheme.typography.bodyLarge)
    }
}