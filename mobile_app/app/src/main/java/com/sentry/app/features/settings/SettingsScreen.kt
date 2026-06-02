package com.sentry.app.features.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.PhilosopherFont

private val BackgroundGray = Color(0xFFF5F5F5)
private val CardBorder     = Color(0xFFE0E0E0)
private val TextPrimary    = Color(0xFF212121)
private val TextSecondary  = Color(0xFF757575)
private val DangerRed      = Color(0xFFF44336)

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onLogout: () -> Unit,
    vm: SettingsViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    var serverUrl     by remember { mutableStateOf("http://10.0.2.2:8000") }
    var groundedAi    by remember { mutableStateOf(true) }
    var showSources   by remember { mutableStateOf(true) }
    var notifications by remember { mutableStateOf(false) }

    LaunchedEffect(state.loggedOut) {
        if (state.loggedOut) onLogout()
    }

    // Outer column — top bar + scrollable body, fills the screen
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundGray),
    ) {

        // ── Top bar ──────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .background(SentryCyan),
        ) {
            IconButton(
                onClick  = onBack,
                modifier = Modifier.align(Alignment.CenterStart),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint               = Color.White,
                    )
                    Text(
                        text       = "Back",
                        color      = Color.White,
                        fontSize   = 13.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
            Text(
                text       = "Settings",
                fontFamily = PhilosopherFont,
                fontSize   = 17.sp,
                fontWeight = FontWeight.Bold,
                color      = Color.White,
                modifier   = Modifier.align(Alignment.Center),
            )
        }

        // ── Scrollable content ───────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {

            // ── Profile card ─────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                    .padding(20.dp),
            ) {
                Column {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier          = Modifier.padding(bottom = 16.dp),
                    ) {
                        Box(
                            modifier         = Modifier
                                .size(48.dp)
                                .clip(CircleShape)
                                .background(Color(0xFFE0F7FA)),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text       = "E4",
                                fontSize   = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color      = SentryCyan,
                            )
                        }
                        Spacer(Modifier.width(14.dp))
                        Column {
                            Text(
                                text       = "EMP_042",
                                fontSize   = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color      = TextPrimary,
                            )
                            Text(
                                text     = "Heritage Insurance · Trainee",
                                fontSize = 12.sp,
                                color    = TextSecondary,
                            )
                        }
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(0.5.dp)
                            .background(CardBorder),
                    )

                    // No isLast parameter — divider always drawn inside ProfileRow
                    ProfileRow(label = "Participant ID", value = "EMP_042")
                    ProfileRow(label = "Organisation",   value = "Heritage Insurance")
                    ProfileRow(label = "Role",           value = "Trainee")
                }
            }

            // ── Server configuration ──────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                    .padding(20.dp),
            ) {
                Column {
                    Text(
                        text       = "Server configuration",
                        fontFamily = PhilosopherFont,
                        fontSize   = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                        modifier   = Modifier.padding(bottom = 4.dp),
                    )
                    Text(
                        text     = "SENTRY middleware connection",
                        fontSize = 12.sp,
                        color    = TextSecondary,
                        modifier = Modifier.padding(bottom = 14.dp),
                    )
                    OutlinedTextField(
                        value         = serverUrl,
                        onValueChange = { serverUrl = it },
                        label         = { Text("Middleware URL", fontSize = 12.sp) },
                        modifier      = Modifier.fillMaxWidth(),
                        singleLine    = true,
                        shape         = RoundedCornerShape(10.dp),
                        colors        = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor   = SentryCyan,
                            unfocusedBorderColor = CardBorder,
                        ),
                    )
                    Spacer(Modifier.height(10.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF4CAF50)),
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text     = "Connected · Response time ~142ms",
                            fontSize = 12.sp,
                            color    = TextSecondary,
                        )
                    }
                }
            }

            // ── Preferences ───────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                    .padding(20.dp),
            ) {
                Column {
                    Text(
                        text       = "Preferences",
                        fontFamily = PhilosopherFont,
                        fontSize   = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                        modifier   = Modifier.padding(bottom = 4.dp),
                    )
                    ToggleRow(
                        label    = "Grounded AI responses",
                        subtitle = "Use RAG pipeline for verified answers",
                        enabled  = groundedAi,
                        onToggle = { groundedAi = !groundedAi },
                    )
                    Box(modifier = Modifier.fillMaxWidth().height(0.5.dp).background(CardBorder))
                    ToggleRow(
                        label    = "Show source citations",
                        subtitle = "Display OWASP and legal references",
                        enabled  = showSources,
                        onToggle = { showSources = !showSources },
                    )
                    Box(modifier = Modifier.fillMaxWidth().height(0.5.dp).background(CardBorder))
                    ToggleRow(
                        label    = "Session notifications",
                        subtitle = "Remind me to complete training",
                        enabled  = notifications,
                        onToggle = { notifications = !notifications },
                    )
                }
            }

            // ── Sign out ──────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White)
                    .border(1.5.dp, Color(0xFFFFCDD2), RoundedCornerShape(14.dp))
                    .clickable { vm.logout() }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text       = "Sign out",
                    fontFamily = PhilosopherFont,
                    fontSize   = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color      = DangerRed,
                )
            }

            Spacer(Modifier.height(8.dp))
        }
    }
}

// isLast removed — divider is always drawn after every row
@Composable
private fun ProfileRow(label: String, value: String) {
    Column {
        Row(
            modifier              = Modifier
                .fillMaxWidth()
                .padding(vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            Text(text = label, fontSize = 13.sp, color = TextSecondary)
            Text(text = value, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = TextPrimary)
        }
        Box(modifier = Modifier.fillMaxWidth().height(0.5.dp).background(CardBorder))
    }
}

@Composable
private fun ToggleRow(
    label: String,
    subtitle: String,
    enabled: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier              = Modifier
            .fillMaxWidth()
            .padding(vertical = 14.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = label,    fontSize = 13.sp, fontWeight = FontWeight.Medium, color = TextPrimary)
            Text(text = subtitle, fontSize = 11.sp, color = TextSecondary)
        }
        Spacer(Modifier.width(12.dp))
        Box(
            modifier = Modifier
                .width(44.dp)
                .height(24.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(if (enabled) SentryCyan else Color(0xFFBDBDBD))
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