package com.sentry.app.features.trainee.home

import androidx.compose.foundation.Image
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.R
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

    // one-shot nav event — navigate to session when started
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
        TraineeTopBar(
            participantId = state.participantId,
            onOpenChat = { navController.navigateSingleTop("chat") },
            onOpenSettings = { navController.navigateSingleTop("settings") },
        )

        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(scheme.background)
                .padding(20.dp),
            horizontalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            // ── Left column — welcome + progress ─────────────────────
            LazyColumn(
                modifier = Modifier
                    .weight(1.4f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                item {
                    WelcomeHeader(
                        participantId = state.participantId,
                        organisation = state.organisation,
                    )
                }
                item {
                    ProgressCard(
                        modules = state.modules,
                        completedCount = state.modules.count { it.isComplete },
                        totalCount = state.modules.size,
                    )
                }
            }

            // ── Right column — stats + CTA ───────────────────────────
            LazyColumn(
                modifier = Modifier
                    .weight(0.8f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                item {
                    StatCard(
                        value = state.sessionsCompleted.toString(),
                        label = "Sessions Completed"
                    )
                }
                item { StatCard(value = state.avgAccuracy, label = "Avg Accuracy") }
                item { StatCard(value = state.modulesLeft.toString(), label = "Modules Left") }
                item {
                    CtaCard(
                        loading = state.loading,
                        error = state.error,
                        onStart = { vm.startSession() },
                    )
                }
            }
        }
    }
}

// component-specific tokens
private val CardBorder = Color(0xFFE0E0E0)
private val PurpleBadge = Color(0xFF9C27B0)
private val TrafficGreen = Color(0xFF4CAF50)
private val TrafficAmber = Color(0xFFFFC107)
private val TrafficRed = Color(0xFFF44336)
private val ProgressTrack = Color(0xFFE0E0E0)

@Composable
private fun TraineeTopBar(
    participantId: String,
    onOpenChat: () -> Unit,
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

        // centre — pepper logo + tap to ask
        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painter = painterResource(R.drawable.pepper_robot),
                    contentDescription = null,
                    modifier = Modifier
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(Color.White),
                    contentScale = ContentScale.Crop,
                )
                Spacer(Modifier.width(8.dp))
                SentryText(
                    text = "SENTRY",
                    size = SentryTextSize.Display,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
            // tap to ask button
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .clickable { onOpenChat() }
                    .padding(horizontal = 12.dp, vertical = 3.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    listOf(TrafficGreen, TrafficAmber, TrafficRed).forEach { colour ->
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(colour),
                        )
                        Spacer(Modifier.width(4.dp))
                    }
                    SentryText(
                        text = "Tap to ASK",
                        size = SentryTextSize.Xs,
                        weight = FontWeight.Medium,
                        color = Color.White,
                    )
                }
            }
        }

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

@Composable
private fun ProgressCard(
    modules: List<ModuleProgress>,
    completedCount: Int,
    totalCount: Int,
) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current
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
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun ModuleRow(module: ModuleProgress) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    val barColor = if (module.progress > 0f) brand.green else ProgressTrack
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

        // API 23 safe — non-lambda progress overload
        LinearProgressIndicator(
            progress = module.progress,
            modifier = Modifier
                .fillMaxWidth()
                .height(7.dp)
                .clip(RoundedCornerShape(4.dp)),
            color = barColor,
            trackColor = ProgressTrack,
        )
    }
}

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

@Composable
private fun CtaCard(
    loading: Boolean,
    error: String,
    onStart: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    Column {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(if (loading) scheme.outline else scheme.primary)
                .clickable { if (!loading) onStart() }
                .padding(vertical = 16.dp, horizontal = 12.dp),
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
                if (!loading) {
                    SentryText(
                        text = "Next: Password Hygiene",
                        size = SentryTextSize.Xs,
                        color = Color.White.copy(alpha = 0.85f),
                    )
                }
            }
        }

        if (error.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            SentryText(
                text = error,
                size = SentryTextSize.Xs,
                color = brand.red,
                align = SentryTextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

private fun currentDate(): String {
    val cal = java.util.Calendar.getInstance()
    val days = arrayOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    val months = arrayOf(
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    )
    return "${days[cal.get(java.util.Calendar.DAY_OF_WEEK) - 1]} " +
            "${cal.get(java.util.Calendar.DAY_OF_MONTH)} " +
            "${months[cal.get(java.util.Calendar.MONTH)]} " +
            "${cal.get(java.util.Calendar.YEAR)}"
}