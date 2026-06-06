package com.sentry.app.features.trainee.home

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
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.core.navigation.navigateSingleTop
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors

@Composable
fun TraineeHomeScreen(
    navController: NavHostController,
    vm: TraineeHomeViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    LaunchedEffect(Unit) {
        vm.events.collect { event ->
            when (event) {
                is TraineeHomeEvent.NavigateToSession ->
                    navController.navigateSingleTop("session/${event.sessionId}")
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.surface),
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        TraineeTopBar(
            participantId = state.participantId,
            onOpenSettings = { navController.navigateSingleTop("settings") },
        )

        // ── "Tap to ASK" pill — floats just below the top bar ───────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(scheme.background),
            contentAlignment = Alignment.Center,
        ) {
            Box(
                modifier = Modifier
                    .padding(vertical = 10.dp)
                    .clip(RoundedCornerShape(13.dp))
                    .background(scheme.primary)
                    .clickable { navController.navigateSingleTop("chat") }
                    .padding(horizontal = 20.dp, vertical = 17.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    listOf(TrafficGreen, TrafficAmber, TrafficRed).forEach { colour ->
                        Box(
                            modifier = Modifier
                                .size(12.dp)
                                .clip(CircleShape)
                                .background(colour),
                        )
                        Spacer(Modifier.width(5.dp))
                    }
                    SentryText(
                        text = "Tap to ASK",
                        size = SentryTextSize.Sm,
                        weight = FontWeight.Medium,
                        color = Color.White,
                    )
                }
            }
        }

        // ── Welcome + CTA row ────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(scheme.background)
                .padding(horizontal = 20.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            WelcomeHeader(
                participantId = state.participantId,
                organisation = state.organisation,
            )

            CtaCard(
                loading = state.loading,
                error = state.error,
                onStart = { vm.startSession() },
            )
        }

        // ── Main content: progress (left) + stats (right) ────────────
        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(scheme.background)
                .padding(horizontal = 20.dp)
                .padding(bottom = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            // left — progress card
            LazyColumn(
                modifier = Modifier
                    .weight(1.4f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                item {
                    ProgressCard(
                        modules = state.modules,
                        completedCount = state.modules.count { it.isComplete },
                        totalCount = state.modules.size,
                    )
                }
            }

            // right — stat cards only
            LazyColumn(
                modifier = Modifier
                    .weight(0.8f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                item {
                    StatCard(
                        value = state.sessionsCompleted.toString(),
                        label = "Sessions Completed",
                    )
                }
                item { StatCard(value = state.avgAccuracy, label = "Avg Accuracy") }
                item { StatCard(value = state.modulesLeft.toString(), label = "Modules Left") }
            }
        }
    }
}

// ── component-specific tokens ────────────────────────────────────────────────
private val CardBorder    = Color(0xFFE0E0E0)
private val PurpleBadge   = Color(0xFF9C27B0)
private val TrafficGreen  = Color(0xFF4CAF50)
private val TrafficAmber  = Color(0xFFFFC107)
private val TrafficRed    = Color(0xFFF44336)
private val ProgressTrack = Color(0xFFE0E0E0)

// ── Top bar (no "Tap to ASK" inside) ─────────────────────────────────────────
@Composable
private fun TraineeTopBar(
    participantId: String,
    onOpenSettings: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val initials =
        participantId.filter { it.isLetterOrDigit() }.takeLast(2).uppercase().ifEmpty { "?" }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(68.dp)
            .background(scheme.primary),
    ) {
        // left — avatar + participant id
        Row(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .padding(start = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.25f)),
                contentAlignment = Alignment.Center,
            ) {
                SentryText(
                    text = initials,
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
            Spacer(Modifier.width(10.dp))
            Column {
                SentryText(
                    text = participantId.ifEmpty { "Trainee" },
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
                SentryText(
                    text = "Trainee",
                    size = SentryTextSize.Xs,
                    color = Color.White.copy(alpha = 0.8f),
                )
            }
        }

        // centre — SENTRY wordmark
        SentryText(
            text = "SENTRY",
            size = SentryTextSize.Display,
            weight = FontWeight.Bold,
            color = Color.White,
            modifier = Modifier.align(Alignment.Center),
        )

        // right — settings
        IconButton(
            onClick = onOpenSettings,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 8.dp),
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = "Settings",
                tint = Color.White,
                modifier = Modifier.size(26.dp),
            )
        }
    }
}

// ── Welcome header ────────────────────────────────────────────────────────────
@Composable
private fun WelcomeHeader(
    participantId: String,
    organisation: String,
) {
    val scheme = MaterialTheme.colorScheme

    Column {
        SentryText(
            text = "Welcome back, $participantId",
            size = SentryTextSize.Display,
            weight = FontWeight.Bold,
            color = scheme.onBackground,
        )
        SentryText(
            text = "${organisation.ifEmpty { "SENTRY Study" }}  ·  ${currentDate()}",
            size = SentryTextSize.Sm,
            color = scheme.outline,
        )
    }
}

// ── CTA (top-right, beside welcome) ──────────────────────────────────────────
@Composable
private fun CtaCard(
    loading: Boolean,
    error: String,
    onStart: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    Column(horizontalAlignment = Alignment.End) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(14.dp))
                .background(if (loading) scheme.outline else scheme.primary)
                .clickable { if (!loading) onStart() }
                .padding(vertical = 14.dp, horizontal = 20.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                SentryText(
                    text = if (loading) "Starting…" else "Continue Your Training",
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = Color.White,
                    align = SentryTextAlign.Center,
                )
            }
        }
        if (!loading) {
            Spacer(Modifier.height(4.dp))
            SentryText(
                text = "Next: Password Hygiene",
                size = SentryTextSize.Xs,
                color = scheme.outline,
                align = SentryTextAlign.End,
            )
        }
        if (error.isNotBlank()) {
            Spacer(Modifier.height(4.dp))
            SentryText(
                text = error,
                size = SentryTextSize.Xs,
                color = brand.red,
                align = SentryTextAlign.End,
            )
        }
    }
}

// ── Progress card ─────────────────────────────────────────────────────────────
@Composable
private fun ProgressCard(
    modules: List<ModuleProgress>,
    completedCount: Int,
    totalCount: Int,
) {
    val scheme = MaterialTheme.colorScheme
    val pct = if (totalCount > 0) (completedCount * 100 / totalCount) else 0

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(20.dp),
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SentryText(
                    text = "Your Progress",
                    size = SentryTextSize.Lg,
                    weight = FontWeight.Bold,
                    color = scheme.onBackground,
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(PurpleBadge)
                        .padding(horizontal = 14.dp, vertical = 5.dp),
                ) {
                    SentryText(
                        text = "$pct% Complete",
                        size = SentryTextSize.Sm,
                        weight = FontWeight.Bold,
                        color = Color.White,
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            modules.forEach { module ->
                ModuleRow(module = module)
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

// ── Module row ────────────────────────────────────────────────────────────────
@Composable
private fun ModuleRow(module: ModuleProgress) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    val barColor    = if (module.progress > 0f) brand.green else ProgressTrack
    val statusColor = if (module.isComplete) brand.green else scheme.outline

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            SentryText(
                text = module.name,
                size = SentryTextSize.Md,
                weight = FontWeight.Medium,
                color = scheme.onBackground,
            )
            SentryText(
                text = module.status,
                size = SentryTextSize.Sm,
                color = statusColor,
            )
        }
        Spacer(Modifier.height(5.dp))
        LinearProgressIndicator(
            progress = module.progress,
            modifier = Modifier
                .fillMaxWidth()
                .height(14.dp)
                .clip(RoundedCornerShape(50)),
            color = barColor,
            trackColor = ProgressTrack,
        )
    }
}

// ── Stat card ─────────────────────────────────────────────────────────────────
@Composable
private fun StatCard(value: String, label: String) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(vertical = 20.dp, horizontal = 16.dp),
    ) {
        Column(horizontalAlignment = Alignment.Start) {
            SentryText(
                text = value,
                size = SentryTextSize.Hero,
                weight = FontWeight.Bold,
                color = scheme.onBackground,
            )
            SentryText(
                text = label,
                size = SentryTextSize.Md,
                color = scheme.outline,
            )
        }
    }
}

// ── Date helper ───────────────────────────────────────────────────────────────
private fun currentDate(): String {
    val cal = java.util.Calendar.getInstance()
    val days   = arrayOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    val months = arrayOf(
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    return "${days[cal.get(java.util.Calendar.DAY_OF_WEEK) - 1]} " +
            "${cal.get(java.util.Calendar.DAY_OF_MONTH)} " +
            "${months[cal.get(java.util.Calendar.MONTH)]} " +
            "${cal.get(java.util.Calendar.YEAR)}"
}