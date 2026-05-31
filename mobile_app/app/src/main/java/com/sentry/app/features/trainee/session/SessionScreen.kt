package com.sentry.app.features.trainee.session

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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

private val ScenarioHeaderPurple = Color(0xFF5C6BC0)
private val CorrectGreen         = Color(0xFF4CAF50)
private val CorrectGreenBg       = Color(0xFFE8F5E9)
private val WrongRed             = Color(0xFFF44336)
private val WrongRedBg           = Color(0xFFFFEBEE)
private val NeutralBorder        = Color(0xFFE0E0E0)
private val TextPrimary          = Color(0xFF212121)
private val TextSecondary        = Color(0xFF757575)
private val AiBubbleBg           = Color(0xFFE3F2FD)
private val AiBubbleBorder       = Color(0xFFBBDEFB)
private val SourceTagBg          = Color(0xFFE8EAF6)
private val SourceTagText        = Color(0xFF3949AB)

@Composable
fun SessionScreen(
    onSessionComplete: (String) -> Unit,
    onBack: () -> Unit,
    vm: SessionViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(state.isComplete) {
        if (state.isComplete) onSessionComplete(state.sessionId)
    }

    val scenario = vm.getCurrentScenario()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF5F5F5)),
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        SessionTopBar(
            current  = state.currentIndex + 1,
            total    = state.totalScenarios,
            type     = scenario.type,
            onBack   = onBack,
        )

        // ── Scrollable body ──────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Scenario card
            ScenarioCard(
                scenario         = scenario,
                selectedChoiceId = state.selectedChoiceId,
                isAnswered       = state.isAnswered,
                onChoiceSelected = { vm.selectChoice(it) },
            )

            // AI feedback — appears after answering
            AnimatedVisibility(
                visible = state.isAnswered,
                enter   = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
            ) {
                ResultFeedback(
                    isCorrect  = state.isCorrect,
                    aiResponse = state.aiResponse,
                    aiSources  = state.aiSources,
                    aiLoading  = state.aiLoading,
                    isLast     = state.currentIndex == state.totalScenarios - 1,
                    onNext     = { vm.nextScenario() },
                )
            }
        }
    }
}

@Composable
private fun SessionTopBar(
    current: Int,
    total: Int,
    type: String,
    onBack: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .background(SentryCyan),
    ) {
        // Back button
        IconButton(
            onClick  = onBack,
            modifier = Modifier.align(Alignment.CenterStart),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector        = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint               = Color.White,
                )
                Text(
                    text     = "BACK",
                    color    = Color.White,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
        }

        // Centre — scenario number + type
        Column(
            modifier            = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text       = "Scenario $current of $total",
                fontFamily = PhilosopherFont,
                fontSize   = 16.sp,
                fontWeight = FontWeight.Bold,
                color      = Color.White,
            )
            Text(
                text     = type,
                fontSize = 12.sp,
                color    = Color.White.copy(alpha = 0.85f),
            )
        }

        // Right — progress pills
        Row(
            modifier          = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            repeat(5) { index ->
                Box(
                    modifier = Modifier
                        .width(28.dp)
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(
                            if (index < current) Color.White
                            else Color.White.copy(alpha = 0.35f)
                        ),
                )
            }
        }
    }
}

@Composable
private fun ScenarioCard(
    scenario: Scenario,
    selectedChoiceId: String?,
    isAnswered: Boolean,
    onChoiceSelected: (String) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .border(1.dp, NeutralBorder, RoundedCornerShape(16.dp)),
    ) {
        // Purple header — matches your Figma exactly
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(ScenarioHeaderPurple)
                .padding(vertical = 18.dp, horizontal = 20.dp),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text       = scenario.title,
                fontFamily = PhilosopherFont,
                fontSize   = 18.sp,
                fontWeight = FontWeight.Bold,
                color      = Color.White,
                textAlign  = TextAlign.Center,
            )
        }

        // Prompt text
        Text(
            text     = scenario.prompt,
            fontSize = 14.sp,
            color    = TextPrimary,
            lineHeight = 22.sp,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
        )

        // Choices
        Column(
            modifier            = Modifier.padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            scenario.choices.forEach { choice ->
                ChoiceRow(
                    choice           = choice,
                    isSelected       = selectedChoiceId == choice.id,
                    isAnswered       = isAnswered,
                    onSelect         = { if (!isAnswered) onChoiceSelected(choice.id) },
                )
            }
        }

        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun ChoiceRow(
    choice: ScenarioChoice,
    isSelected: Boolean,
    isAnswered: Boolean,
    onSelect: () -> Unit,
) {
    val (bgColor, borderColor) = when {
        isAnswered && choice.isCorrect -> CorrectGreenBg to CorrectGreen
        isAnswered && isSelected && !choice.isCorrect -> WrongRedBg to WrongRed
        isSelected -> Color(0xFFE3F2FD) to SentryCyan
        else -> Color.White to NeutralBorder
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(bgColor)
            .border(1.5.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable { onSelect() }
            .padding(horizontal = 14.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        // Radio circle
        Box(
            modifier = Modifier
                .size(26.dp)
                .clip(CircleShape)
                .background(
                    when {
                        isAnswered && choice.isCorrect             -> CorrectGreen
                        isAnswered && isSelected && !choice.isCorrect -> WrongRed
                        isSelected                                 -> SentryCyan
                        else                                       -> Color(0xFFBDBDBD)
                    }
                ),
            contentAlignment = Alignment.Center,
        ) {
            if ((isAnswered && choice.isCorrect) ||
                (isAnswered && isSelected && !choice.isCorrect)) {
                Text(
                    text       = if (choice.isCorrect) "✓" else "✗",
                    color      = Color.White,
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
        }

        Spacer(Modifier.width(12.dp))

        Text(
            text      = choice.text,
            fontSize  = 13.sp,
            color     = TextPrimary,
            lineHeight = 20.sp,
            modifier  = Modifier.weight(1f),
        )
    }
}

@Composable
private fun ResultFeedback(
    isCorrect: Boolean,
    aiResponse: String,
    aiSources: List<String>,
    aiLoading: Boolean,
    isLast: Boolean,
    onNext: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {

        // Result banner
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(if (isCorrect) CorrectGreenBg else WrongRedBg)
                .border(
                    1.dp,
                    if (isCorrect) CorrectGreen else WrongRed,
                    RoundedCornerShape(12.dp),
                )
                .padding(14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text     = if (isCorrect) "✓" else "✗",
                    fontSize = 20.sp,
                    color    = if (isCorrect) CorrectGreen else WrongRed,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text       = if (isCorrect)
                        "Correct decision — well done!"
                    else
                        "Risky decision — let's learn from this",
                    fontFamily = PhilosopherFont,
                    fontSize   = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color      = if (isCorrect) Color(0xFF1B5E20) else Color(0xFFB71C1C),
                )
            }
        }

        // AI grounded response bubble
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(AiBubbleBg)
                .border(1.dp, AiBubbleBorder, RoundedCornerShape(12.dp))
                .padding(16.dp),
        ) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier         = Modifier
                            .size(30.dp)
                            .clip(CircleShape)
                            .background(SentryCyan),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text("P", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text       = "Pepper · Grounded response",
                        fontSize   = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color      = Color(0xFF1565C0),
                    )
                }

                Spacer(Modifier.height(10.dp))

                if (aiLoading) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(
                            modifier  = Modifier.size(18.dp),
                            color     = SentryCyan,
                            strokeWidth = 2.dp,
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text     = "Pepper is thinking…",
                            fontSize = 13.sp,
                            color    = TextSecondary,
                        )
                    }
                } else {
                    Text(
                        text      = aiResponse,
                        fontSize  = 13.sp,
                        color     = TextPrimary,
                        lineHeight = 21.sp,
                    )

                    if (aiSources.isNotEmpty()) {
                        Spacer(Modifier.height(10.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            aiSources.take(3).forEach { source ->
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(20.dp))
                                        .background(SourceTagBg)
                                        .padding(horizontal = 10.dp, vertical = 4.dp),
                                ) {
                                    Text(
                                        text     = source,
                                        fontSize = 10.sp,
                                        color    = SourceTagText,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // Next / Finish button
        if (!aiLoading) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(SentryCyan)
                    .clickable { onNext() }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text       = if (isLast) "View my results" else "Next scenario →",
                    fontFamily = PhilosopherFont,
                    fontSize   = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
            }
        }
    }
}