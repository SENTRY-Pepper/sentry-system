package com.sentry.app.features.trainee.results

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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
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

private val CardBorder = Color(0xFFE0E0E0)

@Composable
fun ResultsScreen(
    navController: NavHostController,
    sessionId: String,
    vm: ResultsViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current
    val accuracy = if (state.totalCount > 0) {
        (state.correctCount.toFloat() / state.totalCount * 100).toInt()
    } else {
        state.postScore.toInt()
    }
    val summary = state.summary

    Row(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight(),
        ) {
            ResultsTopBar(
                participantId = summary?.participantId ?: state.sessionId,
                onBack = { navController.popBackStack() },
            )

            if (state.loading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center,
                ) {
                    CircularProgressIndicator(color = scheme.primary)
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(20.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    item {
                        ScoreCard(
                            accuracy = accuracy,
                            correctCount = state.correctCount,
                            totalCount = state.totalCount,
                            postScore = state.postScore,
                            durationSeconds = state.durationSeconds,
                            vm = vm,
                        )
                    }

                    item { Spacer(Modifier.height(80.dp)) }
                }
            }
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(scheme.surface),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .background(scheme.primary.copy(alpha = 0.08f)),
                contentAlignment = Alignment.CenterStart,
            ) {
                SentryText(
                    text = "Next steps",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = scheme.primary,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                if (state.error.isNotEmpty()) {
                    item {
                        SentryText(
                            text = state.error,
                            size = SentryTextSize.Sm,
                            color = brand.red,
                        )
                    }
                }

                if (state.missedModuleIds.isNotEmpty()) {
                    item {
                        RecommendationCard(
                            missedModules = state.missedModuleIds.map { vm.moduleTitle(it) },
                            onBackHome = { navController.navigateSingleTop("traineeHome") },
                            onOpenChat = { navController.navigateSingleTop("chat") },
                        )
                    }
                } else if (!state.loading) {
                    item {
                        PerfectScoreCard(
                            onBackHome = { navController.navigateSingleTop("traineeHome") },
                        )
                    }
                }

                item { Spacer(Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun ResultsTopBar(
    participantId: String,
    onBack: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .background(scheme.primary),
    ) {
        IconButton(
            onClick = onBack,
            modifier = Modifier.align(Alignment.CenterStart),
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back",
                tint = Color.White,
            )
        }

        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            SentryText(
                text = "OWASP Training Complete",
                size = SentryTextSize.Xl,
                weight = FontWeight.Bold,
                color = Color.White,
            )
            if (participantId.isNotEmpty()) {
                SentryText(
                    text = participantId,
                    size = SentryTextSize.Xs,
                    color = Color.White.copy(alpha = 0.85f),
                )
            }
        }
    }
}

@Composable
private fun ScoreCard(
    accuracy: Int,
    correctCount: Int,
    totalCount: Int,
    postScore: Float,
    durationSeconds: Int,
    vm: ResultsViewModel,
) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(24.dp),
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            SentryText(
                text = "OWASP Top 10 assessment score",
                size = SentryTextSize.Sm,
                color = scheme.outline,
                align = SentryTextAlign.Center,
            )
            Spacer(Modifier.height(4.dp))
            SentryText(
                text = "$accuracy%",
                size = SentryTextSize.Hero,
                weight = FontWeight.Bold,
                color = scheme.primary,
                align = SentryTextAlign.Center,
            )
            SentryText(
                text = "$correctCount of $totalCount modules correct",
                size = SentryTextSize.Sm,
                color = scheme.outline,
                align = SentryTextAlign.Center,
            )

            Spacer(Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                StatItem(
                    value = "${postScore.toInt()}%",
                    label = "Stored score",
                )
                Divider()
                StatItem(
                    value = vm.formatDuration(durationSeconds),
                    label = "Duration",
                )
                Divider()
                StatItem(
                    value = totalCount.toString(),
                    label = "Modules",
                )
            }
        }
    }
}

@Composable
private fun Divider() {
    Box(
        modifier = Modifier
            .width(0.5.dp)
            .height(40.dp)
            .background(CardBorder),
    )
}

@Composable
private fun StatItem(value: String, label: String) {
    val scheme = MaterialTheme.colorScheme

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        SentryText(
            text = value,
            size = SentryTextSize.Lg,
            weight = FontWeight.Bold,
            color = scheme.onBackground,
        )
        SentryText(
            text = label,
            size = SentryTextSize.Xs,
            color = scheme.outline,
        )
    }
}

@Composable
private fun RecommendationCard(
    missedModules: List<String>,
    onBackHome: () -> Unit,
    onOpenChat: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

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
                text = "Review recommended",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = scheme.onBackground,
            )
            Spacer(Modifier.height(6.dp))
            SentryText(
                text = "Revisit these OWASP areas with Pepper:",
                size = SentryTextSize.Sm,
                color = scheme.outline,
            )
            Spacer(Modifier.height(10.dp))
            missedModules.take(4).forEach { title ->
                SentryText(
                    text = "- $title",
                    size = SentryTextSize.Sm,
                    color = scheme.onBackground,
                    maxLines = 2,
                )
                Spacer(Modifier.height(4.dp))
            }
            Spacer(Modifier.height(12.dp))

            ActionButton(
                text = "Back to home",
                filled = true,
                onClick = onBackHome,
            )
            Spacer(Modifier.height(10.dp))
            ActionButton(
                text = "Ask Pepper to explain further",
                filled = false,
                onClick = onOpenChat,
            )
        }
    }
}

@Composable
private fun PerfectScoreCard(
    onBackHome: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    Column {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(brand.green.copy(alpha = 0.10f))
                .border(1.dp, brand.green, RoundedCornerShape(16.dp))
                .padding(20.dp),
        ) {
            SentryText(
                text = "Perfect score. You handled all OWASP Top 10 scenarios correctly.",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = brand.green,
                align = SentryTextAlign.Center,
            )
        }
        Spacer(Modifier.height(12.dp))
        ActionButton(
            text = "Back to home",
            filled = true,
            onClick = onBackHome,
        )
    }
}

@Composable
private fun ActionButton(
    text: String,
    filled: Boolean,
    onClick: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (filled) scheme.primary else scheme.surface)
            .border(1.5.dp, scheme.primary, RoundedCornerShape(12.dp))
            .clickable { onClick() }
            .padding(vertical = 14.dp),
        contentAlignment = Alignment.Center,
    ) {
        SentryText(
            text = text,
            size = SentryTextSize.Sm,
            weight = FontWeight.Bold,
            color = if (filled) Color.White else scheme.primary,
            align = SentryTextAlign.Center,
        )
    }
}
