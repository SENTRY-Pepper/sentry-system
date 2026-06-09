package com.sentry.app.features.trainee.home

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.speech.RecognizerIntent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.TextButton
import androidx.compose.ui.window.Dialog
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
    val speechLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@rememberLauncherForActivityResult
        val spoken = result.data
            ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()
            .orEmpty()
        vm.submitQuickAsk(spoken)
    }

    LaunchedEffect(Unit) {
        vm.refreshProfile()
        vm.events.collect { event ->
            when (event) {
                is TraineeHomeEvent.NavigateToSession ->
                    navController.navigateSingleTop(
                        "session/${event.sessionId}/${event.moduleId ?: ""}"
                    )
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
                    .clickable {
                        vm.openQuickAsk()
                        try {
                            speechLauncher.launch(quickAskSpeechIntent())
                        } catch (_: ActivityNotFoundException) {
                            vm.quickAskSpeechUnavailable()
                        }
                    }
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

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .background(scheme.background)
                .padding(horizontal = 20.dp)
                .padding(bottom = 20.dp),
            contentPadding = PaddingValues(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            item {
                StatsRow(
                    sessionsCompleted = state.sessionsCompleted.toString(),
                    averageAccuracy = state.avgAccuracy,
                    modulesLeft = state.modulesLeft.toString(),
                )
            }
            item {
                ModuleCarousel(
                    modules = state.modules,
                    completedCount = state.modules.count { it.isComplete },
                    totalCount = state.modules.size,
                    onModuleClick = { vm.startSession(moduleId = it) },
                )
            }
        }

        if (state.quickAskOpen) {
            QuickAskDialog(
                state = state,
                onDismiss = vm::closeQuickAsk,
                onTalk = {
                    try {
                        speechLauncher.launch(quickAskSpeechIntent())
                    } catch (_: ActivityNotFoundException) {
                        vm.quickAskSpeechUnavailable()
                    }
                },
                onOpenChat = {
                    vm.closeQuickAsk()
                    navController.navigateSingleTop("chat")
                },
            )
        }
    }
}

private val CardBorder = Color(0xFFE0E0E0)
private val PurpleBadge = Color(0xFF9C27B0)
private val TrafficGreen = Color(0xFF4CAF50)
private val TrafficAmber = Color(0xFFFFC107)
private val TrafficRed = Color(0xFFF44336)
private val ProgressTrack = Color(0xFFE0E0E0)

private fun quickAskSpeechIntent(): Intent =
    Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
        )
        putExtra(RecognizerIntent.EXTRA_PROMPT, "Ask Pepper your cybersecurity question")
    }

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
                    .background(Color.White.copy(alpha = 0.25f))
                    .clickable { onOpenChat() },
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

        SentryText(
            text = "SENTRY",
            size = SentryTextSize.Display,
            weight = FontWeight.Bold,
            color = Color.White,
            modifier = Modifier.align(Alignment.Center),
        )

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
private fun QuickAskDialog(
    state: TraineeHomeUiState,
    onDismiss: () -> Unit,
    onTalk: () -> Unit,
    onOpenChat: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    Dialog(onDismissRequest = onDismiss) {
        Card(
            colors = CardDefaults.cardColors(containerColor = scheme.surface),
            shape = RoundedCornerShape(18.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(92.dp)
                        .clip(CircleShape)
                        .background(scheme.primary)
                        .clickable(enabled = !state.quickAskLoading) { onTalk() },
                    contentAlignment = Alignment.Center,
                ) {
                    SentryText(
                        text = "Talk",
                        size = SentryTextSize.Md,
                        weight = FontWeight.Bold,
                        color = Color.White,
                        align = SentryTextAlign.Center,
                    )
                }

                if (state.quickAskTranscript.isNotBlank()) {
                    SentryText(
                        text = state.quickAskTranscript,
                        size = SentryTextSize.Md,
                        color = scheme.onBackground,
                        align = SentryTextAlign.Center,
                        maxLines = 4,
                    )
                } else {
                    SentryText(
                        text = "Ask Pepper a cybersecurity question",
                        size = SentryTextSize.Md,
                        color = scheme.outline,
                        align = SentryTextAlign.Center,
                    )
                }

                when {
                    state.quickAskLoading -> CircularProgressIndicator(color = scheme.primary)
                    state.quickAskError.isNotBlank() -> SentryText(
                        text = state.quickAskError,
                        size = SentryTextSize.Sm,
                        color = TrafficRed,
                        align = SentryTextAlign.Center,
                    )
                    state.quickAskResponse.isNotBlank() -> SentryText(
                        text = state.quickAskResponse,
                        size = SentryTextSize.Sm,
                        color = scheme.onBackground,
                        align = SentryTextAlign.Center,
                        maxLines = 8,
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = onDismiss) {
                        SentryText(
                            text = "Close",
                            size = SentryTextSize.Sm,
                            color = scheme.primary,
                        )
                    }
                    Button(onClick = onOpenChat) {
                        SentryText(
                            text = "Open Chat",
                            size = SentryTextSize.Sm,
                            color = Color.White,
                        )
                    }
                }
            }
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
            text = "${organisation.ifEmpty { "SENTRY Study" }}  -  ${currentDate()}",
            size = SentryTextSize.Sm,
            color = scheme.outline,
        )
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

    Column(horizontalAlignment = Alignment.End) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(14.dp))
                .background(if (loading) scheme.outline else scheme.primary)
                .clickable { if (!loading) onStart() }
                .padding(vertical = 14.dp, horizontal = 20.dp),
            contentAlignment = Alignment.Center,
        ) {
            SentryText(
                text = if (loading) "Starting..." else "Continue Your Training",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = Color.White,
                align = SentryTextAlign.Center,
            )
        }
        if (!loading) {
            Spacer(Modifier.height(4.dp))
            SentryText(
                text = "Next: OWASP Top 10 module",
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

@Composable
private fun StatsRow(
    sessionsCompleted: String,
    averageAccuracy: String,
    modulesLeft: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        StatCard(
            value = sessionsCompleted,
            label = "Sessions Completed",
            modifier = Modifier.weight(1f),
        )
        Spacer(Modifier.width(12.dp))
        StatCard(
            value = averageAccuracy,
            label = "Avg Accuracy",
            modifier = Modifier.weight(1f),
        )
        Spacer(Modifier.width(12.dp))
        StatCard(
            value = modulesLeft,
            label = "Modules Left",
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun ModuleCarousel(
    modules: List<ModuleProgress>,
    completedCount: Int,
    totalCount: Int,
    onModuleClick: (String) -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val listState = rememberLazyListState()
    val pct = if (totalCount > 0) (completedCount * 100 / totalCount) else 0
    val currentIndex = modules.indexOfFirst { it.status == "In Progress" }
        .takeIf { it >= 0 }
        ?: modules.indexOfFirst { !it.isComplete }.takeIf { it >= 0 }
        ?: modules.lastIndex.coerceAtLeast(0)

    LaunchedEffect(currentIndex, modules.size) {
        if (modules.isNotEmpty()) {
            listState.animateScrollToItem(currentIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(vertical = 18.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                SentryText(
                    text = "OWASP Top 10 Modules",
                    size = SentryTextSize.Lg,
                    weight = FontWeight.Bold,
                    color = scheme.onBackground,
                )
                SentryText(
                    text = "Swipe through your learning path",
                    size = SentryTextSize.Sm,
                    color = scheme.outline,
                )
            }
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

        LazyRow(
            state = listState,
            modifier = Modifier.fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 28.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            itemsIndexed(modules) { index, module ->
                ModuleProgressCard(
                    module = module,
                    isCurrent = index == currentIndex,
                    onClick = { onModuleClick(module.id) },
                )
            }
        }
    }
}

@Composable
private fun ModuleProgressCard(
    module: ModuleProgress,
    isCurrent: Boolean,
    onClick: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current
    val progressPct = (module.progress * 100).toInt().coerceIn(0, 100)
    val borderColor = when {
        isCurrent -> scheme.primary
        module.isComplete -> brand.green
        else -> CardBorder
    }
    val badgeBg = when {
        module.isComplete -> brand.green
        isCurrent -> scheme.primary
        else -> scheme.outline
    }

    Column(
        modifier = Modifier
            .width(if (isCurrent) 280.dp else 248.dp)
            .height(178.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(
                if (isCurrent) scheme.primary.copy(alpha = 0.06f) else scheme.background
            )
            .border(1.5.dp, borderColor, RoundedCornerShape(16.dp))
            .clickable { onClick() }
            .padding(16.dp),
        verticalArrangement = Arrangement.SpaceBetween,
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(badgeBg)
                        .padding(horizontal = 10.dp, vertical = 4.dp),
                ) {
                    SentryText(
                        text = module.status,
                        size = SentryTextSize.Xs,
                        weight = FontWeight.Bold,
                        color = Color.White,
                    )
                }
                SentryText(
                    text = "${module.scorePercent?.let { "$it%" } ?: "--"} score",
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = scheme.primary,
                )
            }
            Spacer(Modifier.height(12.dp))
            SentryText(
                text = module.name,
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = scheme.onBackground,
                maxLines = 2,
            )
        }

        Column {
            LinearProgressIndicator(
                progress = { module.progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(12.dp)
                    .clip(RoundedCornerShape(50)),
                color = if (module.progress > 0f) brand.green else ProgressTrack,
                trackColor = ProgressTrack,
            )
            Spacer(Modifier.height(8.dp))
            SentryText(
                text = if (isCurrent) "Current module - $progressPct% complete"
                    else "$progressPct% complete",
                size = SentryTextSize.Sm,
                color = if (isCurrent) scheme.primary else scheme.outline,
            )
        }
    }
}

@Composable
private fun StatCard(
    value: String,
    label: String,
    modifier: Modifier = Modifier,
) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = modifier
            .height(124.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(vertical = 18.dp, horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            SentryText(
                text = value,
                size = SentryTextSize.Hero,
                weight = FontWeight.Bold,
                color = scheme.onBackground,
                align = SentryTextAlign.Center,
            )
            SentryText(
                text = label,
                size = SentryTextSize.Sm,
                color = scheme.outline,
                align = SentryTextAlign.Center,
                maxLines = 2,
            )
        }
    }
}

private fun currentDate(): String {
    val cal = java.util.Calendar.getInstance()
    val days = arrayOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    val months = arrayOf(
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    return "${days[cal.get(java.util.Calendar.DAY_OF_WEEK) - 1]} " +
        "${cal.get(java.util.Calendar.DAY_OF_MONTH)} " +
        "${months[cal.get(java.util.Calendar.MONTH)]} " +
        "${cal.get(java.util.Calendar.YEAR)}"
}
