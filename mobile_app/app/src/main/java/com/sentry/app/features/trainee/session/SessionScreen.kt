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
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.core.navigation.navigateSingleTop
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors

@Composable
fun SessionScreen(
    navController: NavHostController,
    sessionId: String,
    vm: SessionViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val brand = LocalBrandColors.current
    val scheme = MaterialTheme.colorScheme
    val scenario = vm.getCurrentScenario()

    // navigate to results when session ends
    LaunchedEffect(state.isComplete) {
        if (state.isComplete) {
            navController.navigateSingleTop("results/${state.sessionId}")
        }
    }

    Row(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        // Left column — scenario card
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight(),
        ) {
            SessionTopBar(
                current = state.currentIndex + 1,
                total = state.totalScenarios,
                type = scenario.type,
                onBack = { navController.popBackStack() },
            )

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item {
                    ScenarioCard(
                        scenario = scenario,
                        selectedChoiceId = state.selectedChoiceId,
                        isAnswered = state.isAnswered,
                        onChoiceSelected = { vm.selectChoice(it) },
                    )
                }
                item { Spacer(Modifier.height(80.dp)) }
            }
        }

        // ── Right column — AI feedback
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(scheme.surface),
        ) {
            // right column header bar — same height as top bar
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .background(scheme.primary.copy(alpha = 0.08f)),
                contentAlignment = Alignment.CenterStart,
            ) {
                SentryText(
                    text = "Pepper's analysis",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = scheme.primary,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item {
                    AnimatedVisibility(
                        visible = state.isAnswered,
                        enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
                    ) {
                        ResultFeedback(
                            isCorrect = state.isCorrect,
                            aiResponse = state.aiResponse,
                            aiSources = state.aiSources,
                            aiLoading = state.aiLoading,
                            isLast = state.currentIndex == state.totalScenarios - 1,
                            onNext = { vm.nextScenario() },
                        )
                    }

                    // placeholder when not yet answered
                    if (!state.isAnswered) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            SentryText(
                                text = "Select an answer to see Pepper's response",
                                size = SentryTextSize.Sm,
                                color = scheme.outline,
                                align = SentryTextAlign.Center,
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
private fun SessionTopBar(
    current: Int,
    total: Int,
    type: String,
    onBack: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .background(scheme.primary),
    ) {
        // back button
        IconButton(
            onClick = onBack,
            modifier = Modifier.align(Alignment.CenterStart),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White,
                )
                SentryText(
                    text = "BACK",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
        }

        // centre — scenario counter + type
        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            SentryText(
                text = "Scenario $current of $total",
                size = SentryTextSize.Lg,
                weight = FontWeight.Bold,
                color = Color.White,
            )
            SentryText(
                text = type,
                size = SentryTextSize.Sm,
                color = Color.White.copy(alpha = 0.85f),
            )
        }

        // right — progress pills
        Row(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            repeat(total.coerceAtMost(5)) { index ->
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
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, NeutralBorder, RoundedCornerShape(16.dp)),
    ) {
        // purple scenario header
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(ScenarioHeaderPurple)
                .padding(vertical = 18.dp, horizontal = 20.dp),
            contentAlignment = Alignment.Center,
        ) {
            SentryText(
                text = scenario.title,
                size = SentryTextSize.Xl,
                weight = FontWeight.Bold,
                color = Color.White,
                align = SentryTextAlign.Center,
            )
        }

        // prompt
        SentryText(
            text = scenario.prompt,
            size = SentryTextSize.Md,
            color = MaterialTheme.colorScheme.onBackground,
            lineHeight = 22.dp.value.sp,
            maxLines = Int.MAX_VALUE,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
        )

        // choices
        Column(
            modifier = Modifier.padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            scenario.choices.forEach { choice ->
                ChoiceRow(
                    choice = choice,
                    isSelected = selectedChoiceId == choice.id,
                    isAnswered = isAnswered,
                    onSelect = { if (!isAnswered) onChoiceSelected(choice.id) },
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
    val brand = LocalBrandColors.current
    val scheme = MaterialTheme.colorScheme

    val correctBg = brand.green.copy(alpha = 0.12f)
    val wrongBg = brand.red.copy(alpha = 0.10f)

    val bgColor = when {
        isAnswered && choice.isCorrect -> correctBg
        isAnswered && isSelected -> wrongBg
        isSelected -> scheme.primary.copy(alpha = 0.10f)
        else -> scheme.surface
    }

    val borderColor = when {
        isAnswered && choice.isCorrect -> brand.green
        isAnswered && isSelected -> brand.red
        isSelected -> scheme.primary
        else -> NeutralBorder
    }

    val circleColor = when {
        isAnswered && choice.isCorrect -> brand.green
        isAnswered && isSelected -> brand.red
        isSelected -> scheme.primary
        else -> scheme.outline
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
        Box(
            modifier = Modifier
                .size(26.dp)
                .clip(CircleShape)
                .background(circleColor),
            contentAlignment = Alignment.Center,
        ) {
            if (isAnswered && (choice.isCorrect || isSelected)) {
                SentryText(
                    text = if (choice.isCorrect) "✓" else "✗",
                    size = SentryTextSize.Sm,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
        }

        Spacer(Modifier.width(12.dp))

        SentryText(
            text = choice.text,
            size = SentryTextSize.Md,
            color = scheme.onBackground,
            maxLines = Int.MAX_VALUE,
            modifier = Modifier.weight(1f),
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
    val brand = LocalBrandColors.current
    val scheme = MaterialTheme.colorScheme

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {

        // result banner
        val bannerBg =
            if (isCorrect) brand.green.copy(alpha = 0.12f) else brand.red.copy(alpha = 0.10f)
        val bannerBorder = if (isCorrect) brand.green else brand.red
        val bannerColor = if (isCorrect) brand.green else brand.red

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(bannerBg)
                .border(1.dp, bannerBorder, RoundedCornerShape(12.dp))
                .padding(14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                SentryText(
                    text = if (isCorrect) "✓" else "✗",
                    size = SentryTextSize.Xl,
                    weight = FontWeight.Bold,
                    color = bannerColor,
                )
                Spacer(Modifier.width(10.dp))
                SentryText(
                    text = if (isCorrect)
                        "Correct decision — well done!"
                    else
                        "Risky decision — let's learn from this",
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = bannerColor,
                    maxLines = 2,
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
                        modifier = Modifier
                            .size(30.dp)
                            .clip(CircleShape)
                            .background(scheme.primary),
                        contentAlignment = Alignment.Center,
                    ) {
                        SentryText(
                            text = "P",
                            size = SentryTextSize.Sm,
                            weight = FontWeight.Bold,
                            color = Color.White,
                        )
                    }
                    Spacer(Modifier.width(8.dp))
                    SentryText(
                        text = "Pepper · Grounded response",
                        size = SentryTextSize.Sm,
                        weight = FontWeight.Bold,
                        color = AiNameColor,
                    )
                }

                Spacer(Modifier.height(10.dp))

                if (aiLoading) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = scheme.primary,
                            strokeWidth = 2.dp,
                        )
                        Spacer(Modifier.width(8.dp))
                        SentryText(
                            text = "Pepper is thinking…",
                            size = SentryTextSize.Md,
                            color = scheme.outline,
                        )
                    }
                } else {
                    SentryText(
                        text = aiResponse,
                        size = SentryTextSize.Md,
                        color = scheme.onBackground,
                        maxLines = Int.MAX_VALUE,
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
                                    SentryText(
                                        text = source,
                                        size = SentryTextSize.Xs,
                                        color = SourceTagText,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // next / finish button
        if (!aiLoading) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(scheme.primary)
                    .clickable { onNext() }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center,
            ) {
                SentryText(
                    text = if (isLast) "View my results" else "Next scenario →",
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
        }
    }
}


// component-specific design tokens — not in global theme
private val ScenarioHeaderPurple = Color(0xFF5C6BC0)
private val AiBubbleBg = Color(0xFFE3F2FD)
private val AiBubbleBorder = Color(0xFFBBDEFB)
private val SourceTagBg = Color(0xFFE8EAF6)
private val SourceTagText = Color(0xFF3949AB)
private val NeutralBorder = Color(0xFFE0E0E0)
private val AiNameColor = Color(0xFF1565C0)
