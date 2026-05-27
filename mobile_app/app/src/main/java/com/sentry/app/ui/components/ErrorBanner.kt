package com.sentry.app.ui.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ErrorBanner(message: String, modifier: Modifier = Modifier) {
    if (message.isBlank()) return
    Surface(
        color    = MaterialTheme.colorScheme.errorContainer,
        shape    = MaterialTheme.shapes.medium,
        modifier = modifier.fillMaxWidth(),
    ) {
        Text(
            text     = message,
            color    = MaterialTheme.colorScheme.onErrorContainer,
            style    = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
        )
    }
}