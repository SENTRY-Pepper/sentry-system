package com.sentry.app.features.auth

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.ui.components.*

@Composable
fun AuthScreen(
    state: AuthUiState,
    onLoginClick: (String, String, String, String) -> Unit,
    onLoginSuccess: (UserRole) -> Unit,
) {
    var participantId by remember { mutableStateOf("") }
    var pin           by remember { mutableStateOf("") }
    var organisation  by remember { mutableStateOf("") }
    var selectedRole  by remember { mutableStateOf(UserRole.TRAINEE) }

    LaunchedEffect(state.success) {
        state.success?.let { onLoginSuccess(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("SENTRY", style = MaterialTheme.typography.headlineLarge)
        Text("Cybersecurity Awareness Training",
             style = MaterialTheme.typography.bodyMedium,
             color = MaterialTheme.colorScheme.onSurfaceVariant)

        Spacer(Modifier.height(32.dp))

        // Role toggle
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            UserRole.entries.forEach { role ->
                FilterChip(
                    selected = selectedRole == role,
                    onClick  = { selectedRole = role },
                    label    = { Text(role.name.lowercase().replaceFirstChar { it.uppercase() }) },
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        SentryTextField(
            value         = participantId,
            onValueChange = { participantId = it },
            label         = "Participant ID",
            placeholder   = "e.g. EMP_001",
        )

        Spacer(Modifier.height(12.dp))

        SentryTextField(
            value                = pin,
            onValueChange        = { pin = it },
            label                = "PIN",
            visualTransformation = PasswordVisualTransformation(),
        )

        Spacer(Modifier.height(12.dp))

        SentryTextField(
            value         = organisation,
            onValueChange = { organisation = it },
            label         = "Organisation",
        )

        Spacer(Modifier.height(8.dp))

        ErrorBanner(message = state.error)

        Spacer(Modifier.height(24.dp))

        SentryButton(
            text    = "Sign in",
            loading = state.loading,
            onClick = {
                onLoginClick(
                    participantId,
                    pin,
                    selectedRole.name.lowercase(),
                    organisation,
                )
            },
        )
    }

    LoadingOverlay(visible = state.loading)
}