package com.sentry.app.features.admin.home

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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import com.sentry.app.core.navigation.Routes
import com.sentry.app.core.navigation.navigateSingleTop
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.core.ui.theme.LocalBrandColors

@Composable
fun AdminHomeScreen(
    navController: NavHostController,
    vm: AdminHomeViewModel = hiltViewModel()
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
            // left column: research metrics
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(),
                contentPadding = PaddingValues(bottom = 80.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item { AdminTopBar(navController = navController) }

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
                    item {
                        ErrorCard(message = state.error, onRetry = { vm.loadDashboard() })
                    }
                }

                if (!state.loading && state.error.isEmpty()) {
                    item { ResearchCountCard(state = state) }
                    item { GroundingComparisonCard(state = state) }
                    item { HallucinationCard(state = state) }
                    item { Spacer(modifier = Modifier.height(80.dp)) }
                }
            }

            // right column — recent sessions
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
                            text = "Research Sessions",
                            size = SentryTextSize.Md,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Button(
                            onClick = { navController.navigateSingleTop(Routes.Analytics.toString()) },
                            colors = ButtonDefaults.outlinedButtonColors()
                        ) {
                            SentryText(
                                text = "View All",
                                size = SentryTextSize.Xs,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }

                if (state.recentSessions.isEmpty() && !state.loading) {
                    item {
                        SentryText(
                            text = "No sessions recorded yet.",
                            size = SentryTextSize.Sm,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                } else {
                    items(state.recentSessions) { session ->
                        RecentSessionCard(session = session)
                    }
                }

                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun AdminTopBar(navController: NavHostController) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.primary)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        SentryText(
            text = "Admin Research",
            size = SentryTextSize.Lg,
            color = Color.White
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { navController.navigateSingleTop(Routes.Analytics.toString()) },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White.copy(alpha = 0.18f)
                )
            ) {
                SentryText(
                    text = "Analytics",
                    size = SentryTextSize.Sm,
                    color = Color.White
                )
            }
            Button(
                onClick = { navController.navigateSingleTop(Routes.Settings.toString()) },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White.copy(alpha = 0.18f)
                )
            ) {
                SentryText(
                    text = "Settings",
                    size = SentryTextSize.Sm,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
private fun ResearchCountCard(state: AdminHomeUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier
                .padding(12.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MetricCell(
                modifier = Modifier.weight(1f),
                label = "Total",
                value = state.totalSessions.toString()
            )
            MetricCell(
                modifier = Modifier.weight(1f),
                label = "Grounded",
                value = state.groundedSessions.toString()
            )
            MetricCell(
                modifier = Modifier.weight(1f),
                label = "Baseline",
                value = state.baselineSessions.toString()
            )
        }
    }
}

@Composable
private fun GroundingComparisonCard(state: AdminHomeUiState) {
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
                text = "Grounded - ${"%.1f".format(state.groundedAccuracy * 100)}%",
                size = SentryTextSize.Xs,
                color = MaterialTheme.colorScheme.outline
            )
            // API 23 safe: non-lambda progress overload
            LinearProgressIndicator(
                progress = state.groundedAccuracy.coerceIn(0f, 1f),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = brandColors.green,
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
            SentryText(
                text = "Baseline - ${"%.1f".format(state.baselineAccuracy * 100)}%",
                size = SentryTextSize.Xs,
                color = MaterialTheme.colorScheme.outline
            )
            LinearProgressIndicator(
                progress = state.baselineAccuracy.coerceIn(0f, 1f),
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
private fun HallucinationCard(state: AdminHomeUiState) {
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
                text = "Hallucination Rate",
                size = SentryTextSize.Sm,
                color = MaterialTheme.colorScheme.onBackground
            )
            SentryText(
                text = "Grounded ${"%.1f".format(state.groundedHallucination * 100)}% vs Baseline ${"%.1f".format(state.baselineHallucination * 100)}%",
                size = SentryTextSize.Md,
                color = MaterialTheme.colorScheme.primary
            )
            // API 23 safe: non-lambda progress overload
            LinearProgressIndicator(
                progress = state.groundedHallucination.coerceIn(0f, 1f),
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
private fun RecentSessionCard(session: SessionSummary) {
    val brandColors = LocalBrandColors.current
    val postScore = session.postAssessmentScore ?: 0f
    val postScoreProgress = percentToProgress(postScore)
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
                    text = "${"%.0f".format(postScore)}%",
                    size = SentryTextSize.Sm,
                    color = if (postScoreProgress >= 0.7f) brandColors.green else brandColors.red
                )
                SentryText(
                    text = if (session.isComplete) "complete" else "in progress",
                    size = SentryTextSize.Xs,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}

@Composable
private fun ErrorCard(message: String, onRetry: () -> Unit) {
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

private fun percentToProgress(value: Float): Float =
    (value / 100f).coerceIn(0f, 1f)