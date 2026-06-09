package com.sentry.app.features.manager.home

import androidx.compose.foundation.background
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import com.sentry.app.core.navigation.Routes
import com.sentry.app.core.navigation.navigateSingleTop
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors
import com.sentry.app.data.models.response.DepartmentAnalytics
import com.sentry.app.data.models.response.TraineeAnalytics
import com.sentry.app.data.models.response.WeaknessAnalytics

@Composable
fun ManagerHomeScreen(
    navController: NavHostController,
    vm: ManagerHomeViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsState()
    val focusManager = LocalFocusManager.current

    LaunchedEffect(Unit) {
        focusManager.clearFocus(force = true)
    }

    Scaffold(containerColor = MaterialTheme.colorScheme.background) { padding ->
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item { ManagerTopBar(navController, vm::refresh) }

                if (state.loading) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(180.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }

                if (state.error.isNotBlank()) {
                    item { StatusCard(text = state.error, isError = true) }
                }
                if (state.message.isNotBlank()) {
                    item { StatusCard(text = state.message, isError = false) }
                }

                item { OrganisationSummaryCard(state) }
                item { WeaknessCard(state.weaknesses) }
                item { DepartmentCard(state.departments) }
                item { Spacer(modifier = Modifier.height(80.dp)) }
            }

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                item {
                    SentryText(
                        text = "Trainee Performance",
                        size = SentryTextSize.Md,
                        color = MaterialTheme.colorScheme.onBackground,
                    )
                }

                if (state.trainees.isEmpty() && !state.loading) {
                    item {
                        SentryText(
                            text = "No trainee accounts yet.",
                            size = SentryTextSize.Sm,
                            color = MaterialTheme.colorScheme.outline,
                        )
                    }
                } else {
                    items(state.trainees) { trainee ->
                        TraineeCard(trainee = trainee)
                    }
                }

                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun ManagerTopBar(navController: NavHostController, onRefresh: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.primary)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column {
            SentryText(
                text = "Manager Dashboard",
                size = SentryTextSize.Lg,
                color = Color.White,
            )
            SentryText(
                text = "Organisation training performance",
                size = SentryTextSize.Xs,
                color = Color.White.copy(alpha = 0.82f),
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onRefresh,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White.copy(alpha = 0.18f)
                ),
            ) {
                SentryText(
                    text = "Refresh",
                    size = SentryTextSize.Sm,
                    color = Color.White,
                )
            }
            Button(
                onClick = { navController.navigateSingleTop(Routes.Settings.toString()) },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White.copy(alpha = 0.18f)
                ),
            ) {
                SentryText(
                    text = "Settings",
                    size = SentryTextSize.Sm,
                    color = Color.White,
                )
            }
        }
    }
}

@Composable
private fun OrganisationSummaryCard(state: ManagerHomeUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            SentryText(
                text = state.organisationId.ifBlank { "Organisation" },
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricCell("Trainees", state.activeTrainees.toString(), Modifier.weight(1f))
                MetricCell("Sessions", state.totalSessions.toString(), Modifier.weight(1f))
                MetricCell("Completed", state.completedSessions.toString(), Modifier.weight(1f))
            }
            SentryText(
                text = "Completion - ${"%.1f".format(state.completionRate)}%",
                size = SentryTextSize.Xs,
                color = MaterialTheme.colorScheme.outline,
            )
            LinearProgressIndicator(
                progress = (state.completionRate / 100f).coerceIn(0f, 1f),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceVariant,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricCell(
                    "Average Score",
                    "${"%.1f".format(state.averageScore)}%",
                    Modifier.weight(1f),
                )
                MetricCell("Risky Answers", state.riskyAnswers.toString(), Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun WeaknessCard(weaknesses: List<WeaknessAnalytics>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SentryText(
                text = "Risk Areas",
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground,
            )
            if (weaknesses.isEmpty()) {
                SentryText(
                    text = "No risky answers recorded.",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline,
                )
            } else {
                weaknesses.forEach { weakness ->
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            SentryText(
                                text = "${weakness.scenarioType}: ${weakness.scenarioId}",
                                size = SentryTextSize.Xs,
                                color = MaterialTheme.colorScheme.onBackground,
                            )
                            SentryText(
                                text = "${"%.0f".format(weakness.riskRate)}%",
                                size = SentryTextSize.Xs,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                        LinearProgressIndicator(
                            progress = (weakness.riskRate / 100f).coerceIn(0f, 1f),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp)),
                            color = LocalBrandColors.current.red,
                            trackColor = MaterialTheme.colorScheme.surfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun DepartmentCard(departments: List<DepartmentAnalytics>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SentryText(
                text = "Departments",
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground,
            )
            if (departments.isEmpty()) {
                SentryText(
                    text = "No department data yet.",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline,
                )
            } else {
                departments.forEach { department ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            SentryText(
                                text = department.department,
                                size = SentryTextSize.Xs,
                                color = MaterialTheme.colorScheme.onBackground,
                            )
                            SentryText(
                                text = "${department.traineeCount} trainees",
                                size = SentryTextSize.Xs,
                                color = MaterialTheme.colorScheme.outline,
                            )
                        }
                        SentryText(
                            text = "${"%.0f".format(department.averageScore ?: 0f)}%",
                            size = SentryTextSize.Xs,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TraineeCard(trainee: TraineeAnalytics) {
    val brandColors = LocalBrandColors.current
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    SentryText(
                        text = trainee.participantId,
                        size = SentryTextSize.Sm,
                        color = MaterialTheme.colorScheme.onBackground,
                    )
                    SentryText(
                        text = trainee.department ?: "Unassigned department",
                        size = SentryTextSize.Xs,
                        color = MaterialTheme.colorScheme.outline,
                    )
                }
                SentryText(
                    text = if (trainee.isActive) "active" else "inactive",
                    size = SentryTextSize.Xs,
                    color = if (trainee.isActive) brandColors.green else brandColors.red,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricCell(
                    "Score",
                    "${"%.0f".format(trainee.averageScore ?: 0f)}%",
                    Modifier.weight(1f),
                )
                MetricCell("Sessions", trainee.completedSessions.toString(), Modifier.weight(1f))
                MetricCell("Risks", trainee.riskyAnswers.toString(), Modifier.weight(1f))
            }
            if (trainee.weakestCategories.isNotEmpty()) {
                SentryText(
                    text = "Focus: ${trainee.weakestCategories.joinToString(", ")}",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline,
                )
            }
        }
    }
}

@Composable
private fun MetricCell(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        SentryText(
            text = value,
            size = SentryTextSize.Md,
            color = MaterialTheme.colorScheme.primary,
        )
        SentryText(
            text = label,
            size = SentryTextSize.Xs,
            color = MaterialTheme.colorScheme.outline,
        )
    }
}

@Composable
private fun StatusCard(text: String, isError: Boolean) {
    val colors = LocalBrandColors.current
    val tone = if (isError) colors.red else colors.green
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = tone.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(8.dp),
    ) {
        SentryText(
            text = text,
            size = SentryTextSize.Sm,
            color = tone,
            modifier = Modifier.padding(12.dp),
        )
    }
}
