package com.sentry.app.features.chat

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun ChatScreen(
    onBack: () -> Unit,
    vm: ChatViewModel = hiltViewModel(),
) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Chat with Pepper — coming in sprint 3",
             style = MaterialTheme.typography.bodyLarge)
    }
}