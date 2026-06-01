package com.sentry.app.features.admin.analytics

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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
private val CorrectGreen   = Color(0xFF4CAF50)
private val DangerRed      = Color(0xFFF44336)
private val TealDark       = Color(0xFF0097A7)

private data class AnalyticsSession(
    val id: String,
    val condition: String,
    val pre: String,
    val post: String,
    val gain: String,
    val grounding: String,
    val isGrounded: Boolean,
)

private val SESSIONS = listOf(
    AnalyticsSession("EMP_042", "Grounded", "45%", "72%", "+27%", "0.82", true),
    AnalyticsSession("EMP_031", "Grounded", "50%", "65%", "+15%", "0.74", true),
    AnalyticsSession("EMP_019", "Baseline", "48%", "50%", "+2%",  "0.08", false),
    AnalyticsSession("EMP_007", "Grounded", "40%", "68%", "+28%", "0.79", true),
    AnalyticsSession("EMP_012", "Baseline", "52%", "55%", "+3%",  "0.11", false),
    AnalyticsSession("EMP_024", "Grounded", "38%", "70%", "+32%", "0.88", true),
    AnalyticsSession("EMP_036", "Baseline", "55%", "57%", "+2%",  "0.09", false),
)

@Composable
fun AnalyticsScreen(
    onBack: () -> Unit,
    vm: AnalyticsViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    var filterCondition by remember { mutableStateOf("All") }

    val filtered = when (filterCondition) {
        "Grounded" -> SESSIONS.filter { it.isGrounded }
        "Baseline" -> SESSIONS.filter { !it.isGrounded }
        else       -> SESSIONS
    }

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
                        tint = Color.White,
                    )
                    Text(
                        text       = "Overview",
                        color      = Color.White,
                        fontSize   = 13.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
            Text(
                text       = "Analytics dashboard",
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
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {

            // Filter chips
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("All", "Grounded", "Baseline").forEach { label ->
                    FilterChip(
                        label    = label,
                        selected = filterCondition == label,
                        onClick  = { filterCondition = label },
                    )
                }
            }

            // Key metrics
            data class MetricItem(val value: String, val label: String, val sub: String, val color: Color)

            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                listOf(
                    MetricItem("0.73", "Mean grounding accuracy", "Grounded condition",   SentryCyan),
                    MetricItem("0.91", "Hallucination rate",      "Baseline condition",   DangerRed),
                    MetricItem("+27%", "Knowledge gain",          "Grounded vs baseline", CorrectGreen),
                ).forEach { metric ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(14.dp))
                            .background(Color.White)
                            .border(1.dp, CardBorder, RoundedCornerShape(14.dp))
                            .padding(16.dp),
                    ) {
                        Column {
                            Text(
                                text       = metric.value,
                                fontFamily = PhilosopherFont,
                                fontSize   = 26.sp,
                                fontWeight = FontWeight.Bold,
                                color      = metric.color,
                            )
                            Text(text = metric.label, fontSize = 12.sp, color = TextPrimary)
                            Text(text = metric.sub,   fontSize = 11.sp, color = TextSecondary)
                        }
                    }
                }
            }

            // Bar chart card
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
                        text       = "Knowledge gain by condition",
                        fontFamily = PhilosopherFont,
                        fontSize   = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                    )
                    Text(
                        text     = "Grounded (RAG) vs Baseline (LLM only)",
                        fontSize = 12.sp,
                        color    = TextSecondary,
                    )
                    Spacer(Modifier.height(16.dp))
                    SimpleBarChart()
                }
            }

            // Sessions table
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp)),
            ) {
                Column {
                    Text(
                        text       = "All sessions",
                        fontFamily = PhilosopherFont,
                        fontSize   = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                        modifier   = Modifier.padding(16.dp),
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(0.5.dp)
                            .background(CardBorder),
                    )

                    // Table header
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFF9F9F9))
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                    ) {
                        listOf("ID" to 1f, "Condition" to 1.2f, "Pre" to 0.8f,
                            "Post" to 0.8f, "Gain" to 0.8f, "Grounding" to 1f)
                            .forEach { (header, weight) ->
                                Text(
                                    text       = header,
                                    fontSize   = 11.sp,
                                    color      = TextSecondary,
                                    fontWeight = FontWeight.Bold,
                                    modifier   = Modifier.weight(weight),
                                )
                            }
                    }

                    filtered.forEach { session ->
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(0.5.dp)
                                .background(CardBorder),
                        )
                        Row(
                            modifier          = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(session.id,        fontSize = 12.sp, color = TextPrimary,   modifier = Modifier.weight(1f))
                            Box(modifier = Modifier.weight(1.2f)) {
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(
                                            if (session.isGrounded) Color(0xFFE3F2FD)
                                            else Color(0xFFF5F5F5)
                                        )
                                        .padding(horizontal = 8.dp, vertical = 3.dp),
                                ) {
                                    Text(
                                        text     = session.condition,
                                        fontSize = 11.sp,
                                        color    = if (session.isGrounded) TealDark
                                        else TextSecondary,
                                        fontWeight = FontWeight.Bold,
                                    )
                                }
                            }
                            Text(session.pre,       fontSize = 12.sp, color = TextSecondary, modifier = Modifier.weight(0.8f))
                            Text(session.post,      fontSize = 12.sp, color = TextPrimary,   modifier = Modifier.weight(0.8f))
                            Text(
                                text     = session.gain,
                                fontSize = 12.sp,
                                color    = if (session.gain.startsWith("+") &&
                                    session.gain != "+2%" &&
                                    session.gain != "+3%") CorrectGreen
                                else TextSecondary,
                                modifier = Modifier.weight(0.8f),
                            )
                            Text(
                                text     = session.grounding,
                                fontSize = 12.sp,
                                color    = if (session.grounding.toDouble() > 0.5) CorrectGreen
                                else DangerRed,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }

                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun FilterChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(if (selected) SentryCyan else Color.White)
            .border(
                1.dp,
                if (selected) SentryCyan else CardBorder,
                RoundedCornerShape(20.dp),
            )
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 7.dp),
    ) {
        Text(
            text       = label,
            fontSize   = 13.sp,
            color      = if (selected) Color.White else TextSecondary,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        )
    }
}

@Composable
private fun SimpleBarChart() {
    val bars = listOf(
        "Grounded" to 0.90f to SentryCyan,
        "Baseline" to 0.35f to Color(0xFFBDBDBD),
        "Phishing" to 0.72f to SentryCyan,
        "USB"      to 0.58f to SentryCyan,
        "Password" to 0.44f to SentryCyan,
        "Vishing"  to 0.30f to DangerRed,
        "Network"  to 0.52f to SentryCyan,
    )

    val maxHeight = 100.dp

    Row(
        modifier              = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment     = Alignment.Bottom,
    ) {
        bars.forEach { (pair, color) ->
            val (label, value) = pair
            Column(
                modifier            = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Bottom,
            ) {
                Text(
                    text     = "${(value * 100).toInt()}%",
                    fontSize = 9.sp,
                    color    = TextSecondary,
                )
                Spacer(Modifier.height(2.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.6f)
                        .height(maxHeight * value)
                        .clip(RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp))
                        .background(color),
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text     = label,
                    fontSize = 9.sp,
                    color    = TextSecondary,
                )
            }
        }
    }
}