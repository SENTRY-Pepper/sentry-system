package com.sentry.app.features.trainee.results

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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.PhilosopherFont

private val CorrectGreen    = Color(0xFF4CAF50)
private val CorrectGreenBg  = Color(0xFFE8F5E9)
private val WrongRed        = Color(0xFFF44336)
private val WrongRedBg      = Color(0xFFFFEBEE)
private val TextPrimary     = Color(0xFF212121)
private val TextSecondary   = Color(0xFF757575)
private val CardBorder      = Color(0xFFE0E0E0)
private val BackgroundGray  = Color(0xFFF5F5F5)

private data class ScenarioResult(
    val name: String,
    val detail: String,
    val correct: Boolean,
)

private val RESULTS = listOf(
    ScenarioResult("Phishing Detection",       "Reported suspicious email to IT",                      true),
    ScenarioResult("USB Drop Simulation",      "Handed unknown drive to IT without plugging in",       true),
    ScenarioResult("Password Hygiene",         "Used password manager for unique credentials",         true),
    ScenarioResult("Voice Social Engineering", "Provided information before verifying caller identity", false),
    ScenarioResult("Network Hygiene",          "Connected via VPN before accessing company systems",   true),
)

@Composable
fun ResultsScreen(
    onDone: () -> Unit,
    onOpenChat: () -> Unit,
    vm: ResultsViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    val correct   = RESULTS.count { it.correct }
    val total     = RESULTS.size
    val accuracy  = (correct.toFloat() / total * 100).toInt()
    val gain      = (state.postScore - state.preScore).toInt()
    val improvement = if (state.preScore > 0)
        ((gain / state.preScore) * 100).toInt() else 0

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
                onClick  = onDone,
                modifier = Modifier.align(Alignment.CenterStart),
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White,
                )
            }
            Column(
                modifier            = Modifier.align(Alignment.Center),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text       = "Session Complete",
                    fontFamily = PhilosopherFont,
                    fontSize   = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
                Text(
                    text     = "EMP_042 · Heritage Insurance",
                    fontSize = 11.sp,
                    color    = Color.White.copy(alpha = 0.85f),
                )
            }
        }

        // ── Scrollable content ───────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {

            // Score card
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
                    .padding(24.dp),
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text     = "Your accuracy score",
                        fontSize = 13.sp,
                        color    = TextSecondary,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text       = "$accuracy%",
                        fontFamily = PhilosopherFont,
                        fontSize   = 56.sp,
                        fontWeight = FontWeight.Bold,
                        color      = SentryCyan,
                    )
                    Text(
                        text     = "$correct of $total correct",
                        fontSize = 13.sp,
                        color    = TextSecondary,
                    )

                    Spacer(Modifier.height(20.dp))

                    // Stats row
                    Row(
                        modifier              = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        StatItem(
                            value = "${state.preScore.toInt()}%→${state.postScore.toInt()}%",
                            label = "Knowledge gain",
                        )
                        Box(
                            modifier = Modifier
                                .width(0.5.dp)
                                .height(40.dp)
                                .background(CardBorder),
                        )
                        StatItem(
                            value = vm.formatDuration(state.durationSeconds),
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

            // Scenario breakdown
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
                        text       = "Scenario breakdown",
                        fontFamily = PhilosopherFont,
                        fontSize   = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                        modifier   = Modifier.padding(bottom = 12.dp),
                    )
                    RESULTS.forEachIndexed { i, result ->
                        ResultRow(result = result)
                        if (i < RESULTS.lastIndex) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(0.5.dp)
                                    .background(CardBorder),
                            )
                        }
                    }
                }
            }

            // Recommendation card
            val weakScenario = RESULTS.firstOrNull { !it.correct }
            if (weakScenario != null) {
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
                            text       = "Recommended next step",
                            fontFamily = PhilosopherFont,
                            fontSize   = 15.sp,
                            fontWeight = FontWeight.Bold,
                            color      = TextPrimary,
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text     = "Focus area: ${weakScenario.name}",
                            fontSize = 13.sp,
                            color    = TextSecondary,
                        )
                        Spacer(Modifier.height(14.dp))

                        // Back to home
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(SentryCyan)
                                .clickable { onDone() }
                                .padding(vertical = 14.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text       = "Back to home",
                                fontFamily = PhilosopherFont,
                                fontSize   = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color      = Color.White,
                            )
                        }

                        Spacer(Modifier.height(10.dp))

                        // Ask Pepper
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.White)
                                .border(1.5.dp, SentryCyan, RoundedCornerShape(12.dp))
                                .clickable { onOpenChat() }
                                .padding(vertical = 14.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text       = "Ask Pepper to explain ${weakScenario.name}",
                                fontFamily = PhilosopherFont,
                                fontSize   = 13.sp,
                                fontWeight = FontWeight.Bold,
                                color      = SentryCyan,
                                textAlign  = TextAlign.Center,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StatItem(value: String, label: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text       = value,
            fontFamily = PhilosopherFont,
            fontSize   = 16.sp,
            fontWeight = FontWeight.Bold,
            color      = TextPrimary,
        )
        Text(
            text     = label,
            fontSize = 11.sp,
            color    = TextSecondary,
        )
    }
}

@Composable
private fun ResultRow(result: ScenarioResult) {
    Row(
        modifier          = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier         = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(if (result.correct) CorrectGreenBg else WrongRedBg),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text       = if (result.correct) "✓" else "✗",
                fontSize   = 14.sp,
                fontWeight = FontWeight.Bold,
                color      = if (result.correct) CorrectGreen else WrongRed,
            )
        }

        Spacer(Modifier.width(12.dp))

        Column(modifier = Modifier.weight(1f)) {
            Text(
                text       = result.name,
                fontSize   = 13.sp,
                fontWeight = FontWeight.Medium,
                color      = TextPrimary,
            )
            Text(
                text     = result.detail,
                fontSize = 11.sp,
                color    = TextSecondary,
            )
        }

        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .background(if (result.correct) CorrectGreenBg else WrongRedBg)
                .padding(horizontal = 10.dp, vertical = 4.dp),
        ) {
            Text(
                text     = if (result.correct) "Correct" else "Risky",
                fontSize = 11.sp,
                color    = if (result.correct) CorrectGreen else WrongRed,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}