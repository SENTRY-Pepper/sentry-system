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

// component-specific tokens
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

    val accuracy = if (state.totalCount > 0)
        (state.correctCount.toFloat() / state.totalCount * 100).toInt() else 0
    val gain = (state.postScore - state.preScore).toInt()
    val improvement = if (state.preScore > 0f)
        ((gain.toFloat() / state.preScore) * 100).toInt() else 0

    // summary data — from real state, falls back gracefully while loading
    val summary = state.summary

    Row(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        // ── Left column — score + breakdown ─────────────────────────
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
                    // score card
                    item {
                        ScoreCard(
                            accuracy = accuracy,
                            correctCount = state.correctCount,
                            totalCount = state.totalCount,
                            preScore = state.preScore,
                            postScore = state.postScore,
                            improvement = improvement,
                            durationSeconds = state.durationSeconds,
                            vm = vm,
                        )
                    }

                    item { Spacer(Modifier.height(80.dp)) }
                }
            }
        }

        // ── Right column — recommendations + actions ─────────────────
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
                // error state
                if (state.error.isNotEmpty()) {
                    item {
                        SentryText(
                            text = state.error,
                            size = SentryTextSize.Sm,
                            color = brand.red,
                        )
                    }
                }

                // recommendation card — show if any scenarios were wrong
                val hasWeakArea = state.correctCount < state.totalCount

                if (hasWeakArea) {
                    item {
                        RecommendationCard(
                            onBackHome = { navController.navigateSingleTop("traineeHome") },
                            onOpenChat = { navController.navigateSingleTop("chat") },
                        )
                    }
                } else if (!state.loading) {
                    item {
                        // all correct — celebrate
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp))
                                .background(brand.green.copy(alpha = 0.10f))
                                .border(1.dp, brand.green, RoundedCornerShape(16.dp))
                                .padding(20.dp),
                        ) {
                            SentryText(
                                text = "Perfect score — excellent security awareness!",
                                size = SentryTextSize.Md,
                                weight = FontWeight.Bold,
                                color = brand.green,
                                align = SentryTextAlign.Center,
                            )
                        }

                        Spacer(Modifier.height(12.dp))

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(scheme.primary)
                                .clickable {
                                    navController.navigateSingleTop("traineeHome")
                                }
                                .padding(vertical = 14.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            SentryText(
                                text = "Back to home",
                                size = SentryTextSize.Md,
                                weight = FontWeight.Bold,
                                color = Color.White,
                            )
                        }
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
                text = "Session Complete",
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
    preScore: Float,
    postScore: Float,
    improvement: Int,
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
                text = "Your accuracy score",
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
                text = "$correctCount of $totalCount correct",
                size = SentryTextSize.Sm,
                color = scheme.outline,
                align = SentryTextAlign.Center,
            )

            Spacer(Modifier.height(20.dp))

            // stats row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                StatItem(
                    value = "${preScore.toInt()}%→${postScore.toInt()}%",
                    label = "Knowledge gain",
                )
                Box(
                    modifier = Modifier
                        .width(0.5.dp)
                        .height(40.dp)
                        .background(CardBorder),
                )
                StatItem(
                    value = vm.formatDuration(durationSeconds),
                    label = "Duration",
                )
                Box(
                    modifier = Modifier
                        .width(0.5.dp)
                        .height(40.dp)
                        .background(CardBorder),
                )
                StatItem(
                    value = "+$improvement%",
                    label = "Improvement",
                )
            }
        }
    }
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
                text = "Recommended next step",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = scheme.onBackground,
            )
            Spacer(Modifier.height(4.dp))
            SentryText(
                text = "Review the scenarios you found difficult with Pepper",
                size = SentryTextSize.Sm,
                color = scheme.outline,
            )
            Spacer(Modifier.height(14.dp))

            // back to home
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(scheme.primary)
                    .clickable { onBackHome() }
                    .padding(vertical = 14.dp),
                contentAlignment = Alignment.Center,
            ) {
                SentryText(
                    text = "Back to home",
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }

            Spacer(Modifier.height(10.dp))

            // ask pepper
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(scheme.surface)
                    .border(1.5.dp, scheme.primary, RoundedCornerShape(12.dp))
                    .clickable { onOpenChat() }
                    .padding(vertical = 14.dp),
                contentAlignment = Alignment.Center,
            ) {
                SentryText(
                    text = "Ask Pepper to explain further",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = scheme.primary,
                    align = SentryTextAlign.Center,
                )
            }
        }
    }
}