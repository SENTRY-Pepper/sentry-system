package com.sentry.app.features.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.R
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors

// component-specific tokens
private val CardBorder = Color(0xFFE0E0E0)
private val DangerRedBg = Color(0xFFFFCDD2)
private val ConnectedGreen = Color(0xFF4CAF50)

@Composable
fun SettingsScreen(
    navController: NavHostController,
    vm: SettingsViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current
    val focusManager = LocalFocusManager.current

    // one-shot logout event — navigate to auth and clear back stack
    LaunchedEffect(Unit) {
        focusManager.clearFocus(force = true)
        vm.events.collect { event ->
            when (event) {
                is SettingsEvent.LoggedOut -> {
                    navController.navigate("auth") {
                        popUpTo(0) { inclusive = true }
                    }
                }
            }
        }
    }

    // avatar initials from participantId
    val initials = state.participantId
        .filter { it.isLetterOrDigit() }
        .takeLast(2)
        .uppercase()
        .ifEmpty { "?" }

    Row(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        // ── Left column — profile + server config ────────────────────
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight(),
        ) {
            SettingsTopBar(
                title = "Settings",
                onBack = { navController.popBackStack() },
            )

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                // profile card
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(16.dp))
                            .background(scheme.surface)
                            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                    ) {
                        Column {
                            // avatar row
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(bottom = 16.dp),
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(48.dp)
                                        .clip(CircleShape)
                                        .background(scheme.primary.copy(alpha = 0.12f)),
                                    contentAlignment = Alignment.Center,
                                ) {
                                    SentryText(
                                        text = initials,
                                        size = SentryTextSize.Lg,
                                        weight = FontWeight.Bold,
                                        color = scheme.primary,
                                    )
                                }
                                Spacer(Modifier.width(14.dp))
                                Column {
                                    SentryText(
                                        text = state.participantId.ifEmpty { "—" },
                                        size = SentryTextSize.Lg,
                                        weight = FontWeight.Bold,
                                        color = scheme.onBackground,
                                    )
                                    SentryText(
                                        text = "${state.organisation} · ${state.role}",
                                        size = SentryTextSize.Sm,
                                        color = scheme.outline,
                                    )
                                }
                            }

                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(0.5.dp)
                                    .background(CardBorder),
                            )

                            ProfileRow(
                                label = "Participant ID",
                                value = state.participantId.ifEmpty { "—" })
                            ProfileRow(
                                label = "Organisation",
                                value = state.organisation.ifEmpty { "—" })
                            ProfileRow(label = "Role", value = state.role.ifEmpty { "—" })
                        }
                    }
                }

                // server configuration
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(16.dp))
                            .background(scheme.surface)
                            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                    ) {
                        Column {
                            SentryText(
                                text = "Server configuration",
                                size = SentryTextSize.Md,
                                weight = FontWeight.Bold,
                                color = scheme.onBackground,
                            )
                            Spacer(Modifier.height(4.dp))
                            SentryText(
                                text = "SENTRY middleware connection",
                                size = SentryTextSize.Sm,
                                color = scheme.outline,
                            )
                            Spacer(Modifier.height(14.dp))
                            OutlinedTextField(
                                value = state.serverUrl,
                                onValueChange = { vm.setServerUrl(it) },
                                label = {
                                    SentryText(
                                        text = "Middleware URL",
                                        size = SentryTextSize.Sm,
                                        color = scheme.outline,
                                    )
                                },
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true,
                                shape = RoundedCornerShape(10.dp),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = scheme.primary,
                                    unfocusedBorderColor = CardBorder,
                                ),
                            )
                            Spacer(Modifier.height(10.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .clip(CircleShape)
                                        .background(ConnectedGreen),
                                )
                                Spacer(Modifier.width(8.dp))
                                SentryText(
                                    text = "Connected · Response time ~142ms",
                                    size = SentryTextSize.Sm,
                                    color = scheme.outline,
                                )
                            }
                        }
                    }
                }

                item { Spacer(Modifier.height(80.dp)) }
            }
        }

        // ── Right column — preferences + sign out ────────────────────
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(scheme.surface),
        ) {
            // right header bar
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .background(scheme.primary),
                contentAlignment = Alignment.CenterStart,
            ) {
                SentryText(
                    text = "Preferences",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = scheme.background,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(16.dp))
                            .background(scheme.background)
                            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                    ) {
                        Column {
                            ToggleRow(
                                label = "Grounded AI responses",
                                subtitle = "Use RAG pipeline for verified answers",
                                enabled = state.groundedAi,
                                onToggle = { vm.setGroundedAi(!state.groundedAi) },
                            )
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(0.5.dp)
                                    .background(CardBorder),
                            )
                            ToggleRow(
                                label = "Show source citations",
                                subtitle = "Display OWASP and legal references",
                                enabled = state.showSources,
                                onToggle = { vm.setShowSources(!state.showSources) },
                            )
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(0.5.dp)
                                    .background(CardBorder),
                            )
                            ToggleRow(
                                label = "Session notifications",
                                subtitle = "Remind me to complete training",
                                enabled = state.notifications,
                                onToggle = { vm.setNotifications(!state.notifications) },
                            )
                        }
                    }
                }

                // sign out button
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(14.dp))
                            .background(scheme.surface)
                            .border(1.5.dp, DangerRedBg, RoundedCornerShape(14.dp))
                            .clickable { vm.logout() }
                            .padding(vertical = 16.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        SentryText(
                            text = "Sign out",
                            size = SentryTextSize.Md,
                            weight = FontWeight.Bold,
                            color = brand.red,
                        )
                    }
                }

                item { Spacer(Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun SettingsTopBar(
    title: String,
    onBack: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .background(scheme.primary)
            .padding(start = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                painter = painterResource(R.drawable.arrow_left_circle),
                contentDescription = "Back",
                tint = Color.White,
                modifier = Modifier.size(30.dp).clickable{onBack()}
            )
            Spacer(Modifier.width(14.dp))
            SentryText(
                text = "Back",
                size = SentryTextSize.Xl,
                weight = FontWeight.Bold,
                color = Color.White,
            )
        }
        SentryText(
            text = title,
            size = SentryTextSize.Xl,
            weight = FontWeight.Bold,
            color = Color.White,
            align = SentryTextAlign.Center,
            //modifier = Modifier.align(Alignment.Center),
        )
        Spacer(Modifier.width(5.dp))

    }
}

@Composable
private fun ProfileRow(label: String, value: String) {
    val scheme = MaterialTheme.colorScheme

    Column {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            SentryText(
                text = label,
                size = SentryTextSize.Md,
                color = scheme.outline,
            )
            SentryText(
                text = value,
                size = SentryTextSize.Md,
                weight = FontWeight.Medium,
                color = scheme.onBackground,
            )
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(0.5.dp)
                .background(CardBorder),
        )
    }
}

@Composable
private fun ToggleRow(
    label: String,
    subtitle: String,
    enabled: Boolean,
    onToggle: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            SentryText(
                text = label,
                size = SentryTextSize.Md,
                weight = FontWeight.Medium,
                color = scheme.onBackground,
            )
            SentryText(
                text = subtitle,
                size = SentryTextSize.Xs,
                color = scheme.outline,
            )
        }
        Spacer(Modifier.width(12.dp))
        // custom toggle — no Material Switch API to avoid API 23 compat issues
        Box(
            modifier = Modifier
                .width(44.dp)
                .height(24.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(if (enabled) scheme.primary else scheme.outline)
                .clickable { onToggle() },
            contentAlignment = if (enabled) Alignment.CenterEnd else Alignment.CenterStart,
        ) {
            Box(
                modifier = Modifier
                    .padding(3.dp)
                    .size(18.dp)
                    .clip(CircleShape)
                    .background(Color.White),
            )
        }
    }
}
