package com.sentry.app.features.admin.analytics

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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.core.ui.theme.LocalBrandColors

@Composable
fun AnalyticsScreen(
    navController: NavHostController,
    vm: AnalyticsViewModel = hiltViewModel()
) {
    val state by vm.uiState.collectAsState()

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // left column — aggregate metrics + condition comparison
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    AnalyticsTopBar(
                        navController = navController,
                        onRefresh = { vm.refresh() })
                }

                if (state.loading) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }

                if (state.error.isNotEmpty()) {
                    item { AnalyticsErrorCard(message = state.error, onRetry = { vm.refresh() }) }
                }

                if (!state.loading && state.error.isEmpty()) {
                    item { AggregateMetricsCard(state = state) }
                    item { ConditionComparisonCard(state = state) }
                    item { Spacer(modifier = Modifier.height(80.dp)) }
                }
            }

            // right column — session list with condition filter
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                item {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        SentryText(
                            text = "Sessions",
                            size = SentryTextSize.Md,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        SentryText(
                            text = "${state.sessions.size} total",
                            size = SentryTextSize.Xs,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                }

                item {
                    ConditionFilterRow(
                        selectedCondition = state.selectedCondition,
                        onSelect = { vm.filterByCondition(it) }
                    )
                }

                val visible = if (state.selectedCondition == null) state.sessions
                else state.sessions.filter { it.condition == state.selectedCondition }

                if (visible.isEmpty() && !state.loading) {
                    item {
                        SentryText(
                            text = "No sessions found.",
                            size = SentryTextSize.Sm,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                } else {
                    items(visible) { session -> SessionRowCard(session = session) }
                }

                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun AnalyticsTopBar(navController: NavHostController, onRefresh: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        SentryText(
            text = "Analytics",
            size = SentryTextSize.Lg,
            color = MaterialTheme.colorScheme.primary
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onRefresh,
                colors = ButtonDefaults.outlinedButtonColors()
            ) {
                SentryText(
                    text = "Refresh",
                    size = SentryTextSize.Sm,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Button(
                onClick = { navController.popBackStack() },
                colors = ButtonDefaults.outlinedButtonColors()
            ) {
                SentryText(
                    text = "Back",
                    size = SentryTextSize.Sm,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
private fun AggregateMetricsCard(state: AnalyticsUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SentryText(
                text = "Overall Metrics",
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricCell(
                    modifier = Modifier.weight(1f),
                    label = "Avg Post Score",
                    value = "${"%.1f".format(state.avgAccuracy * 100)}%"
                )
                MetricCell(
                    modifier = Modifier.weight(1f),
                    label = "Avg Knowledge Gain",
                    value = "${"%.1f".format(state.avgKnowledgeGain * 100)}%"
                )
                MetricCell(
                    modifier = Modifier.weight(1f),
                    label = "Total Sessions",
                    value = state.sessions.size.toString()
                )
            }
        }
    }
}

@Composable
private fun MetricCell(modifier: Modifier = Modifier, label: String, value: String) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        SentryText(
            text = value,
            size = SentryTextSize.Md,
            color = MaterialTheme.colorScheme.primary
        )
        SentryText(
            text = label,
            size = SentryTextSize.Xs,
            color = MaterialTheme.colorScheme.outline
        )
    }
}

@Composable
private fun ConditionComparisonCard(state: AnalyticsUiState) {
    val brandColors = LocalBrandColors.current
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SentryText(
                text = "Grounded vs Baseline Accuracy",
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground
            )
            SentryText(
                text = "Grounded (${state.ragSessionCount} sessions) — ${"%.1f".format(state.ragAvgAccuracy * 100)}%",
                size = SentryTextSize.Xs,
                color = MaterialTheme.colorScheme.outline
            )
            // API 23 safe: non-lambda progress overload
            LinearProgressIndicator(
                progress = state.ragAvgAccuracy,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = brandColors.green,
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
            SentryText(
                text = "Baseline (${state.baselineSessionCount} sessions) — ${"%.1f".format(state.baselineAvgAccuracy * 100)}%",
                size = SentryTextSize.Xs,
                color = MaterialTheme.colorScheme.outline
            )
            LinearProgressIndicator(
                progress = state.baselineAvgAccuracy,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
        }
    }
}

@Composable
private fun ConditionFilterRow(selectedCondition: String?, onSelect: (String?) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(
            label = "All",
            selected = selectedCondition == null,
            onClick = { onSelect(null) })
        FilterChip(
            label = "Grounded",
            selected = selectedCondition == "grounded",
            onClick = { onSelect("grounded") })
        FilterChip(
            label = "Baseline",
            selected = selectedCondition == "baseline",
            onClick = { onSelect("baseline") })
    }
}

@Composable
private fun FilterChip(label: String, selected: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.surface
        ),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
    ) {
        SentryText(
            text = label,
            size = SentryTextSize.Xs,
            color = if (selected) MaterialTheme.colorScheme.onPrimary
            else MaterialTheme.colorScheme.onBackground
        )
    }
}

@Composable
private fun SessionRowCard(session: SessionSummary) {
    val brandColors = LocalBrandColors.current
    val postScore = session.postAssessmentScore ?: 0f
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                SentryText(
                    text = session.participantId,
                    size = SentryTextSize.Sm,
                    color = MaterialTheme.colorScheme.onBackground
                )
                SentryText(
                    text = session.condition,
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline
                )
            }
            Column(horizontalAlignment = Alignment.End) {
                SentryText(
                    text = "${"%.0f".format(postScore * 100)}%",
                    size = SentryTextSize.Sm,
                    color = if (postScore >= 0.7f) brandColors.green else brandColors.red
                )
                val mins = (session.durationSeconds ?: 0) / 60
                SentryText(
                    text = "${mins}m",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}

@Composable
private fun AnalyticsErrorCard(message: String, onRetry: () -> Unit) {
    val brandColors = LocalBrandColors.current
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = brandColors.red.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            SentryText(
                text = message,
                size = SentryTextSize.Sm,
                color = brandColors.red,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = onRetry,
                colors = ButtonDefaults.buttonColors(containerColor = brandColors.red)
            ) {
                SentryText(
                    text = "Retry",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.onPrimary
                )
            }
        }
    }
}

