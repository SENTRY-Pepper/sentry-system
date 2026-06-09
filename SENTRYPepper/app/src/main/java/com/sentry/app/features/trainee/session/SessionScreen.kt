package com.sentry.app.features.trainee.session

import android.content.Intent
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
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
import com.sentry.app.features.trainee.curriculum.OwaspAnswerOption
import com.sentry.app.features.trainee.curriculum.OwaspQuestion
import com.sentry.app.features.trainee.curriculum.OwaspTrainingModule
import com.sentry.app.pepper.PepperRobotBridge

@Composable
fun SessionScreen(
    navController: NavHostController,
    sessionId: String,
    startModuleId: String? = null,
    vm: SessionViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val scenario = vm.getCurrentScenario()
    val question = vm.getCurrentQuestion()
    val context = LocalContext.current
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }
    val speechLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val spoken = result.data
            ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()
            ?.lowercase()
            .orEmpty()
        spokenAnswerToLabel(spoken)?.let { label ->
            question.options.firstOrNull { it.label.lowercase() == label }
                ?.let { vm.selectChoice(it.id) }
        }
    }

    DisposableEffect(Unit) {
        val engine = TextToSpeech(context) { }
        tts = engine
        onDispose {
            engine.stop()
            engine.shutdown()
        }
    }

    LaunchedEffect(state.isComplete) {
        if (state.isComplete) {
            navController.navigateSingleTop("results/${state.sessionId}")
        }
    }

    LaunchedEffect(
        state.currentModuleIndex,
        state.currentQuestionIndex,
        state.isModuleBreak,
    ) {
        val engine = tts ?: return@LaunchedEffect
        if (state.isModuleBreak) {
            speakWithRobotFallback(
                engine = engine,
                text = "${scenario.owaspId} ${scenario.title} complete. " +
                    "Do you want to proceed to the next module?",
                utteranceId = "module-break-${scenario.id}",
            )
        } else {
            speakWithRobotFallback(
                engine = engine,
                text = buildQuestionSpeech(scenario, question),
                utteranceId = question.id,
            )
        }
    }

    LaunchedEffect(state.isAnswered, state.aiResponse) {
        val engine = tts ?: return@LaunchedEffect
        if (state.isAnswered && state.aiResponse.isNotBlank()) {
            val result = if (state.isCorrect) {
                "That is correct."
            } else {
                "That is not the safest answer."
            }
            speakWithRobotFallback(
                engine = engine,
                text = "$result ${state.aiResponse}",
                utteranceId = "feedback-${state.currentModuleIndex}-${state.currentQuestionIndex}",
            )
        }
    }

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
            SessionTopBar(
                currentModule = state.currentModuleIndex + 1,
                totalModules = state.totalModules,
                currentQuestion = state.currentQuestionIndex + 1,
                totalQuestionsInModule = scenario.questions.size,
                type = "${scenario.owaspId} - ${scenario.difficulty}",
                onBack = { navController.popBackStack() },
            )

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item {
                    if (state.isModuleBreak) {
                        ModuleBreakCard(
                            scenario = scenario,
                            isLastModule = state.currentModuleIndex == state.totalModules - 1,
                            onContinue = { vm.nextScenario() },
                        )
                    } else {
                        ScenarioCard(
                            scenario = scenario,
                            question = question,
                            selectedChoiceId = state.selectedChoiceId,
                            isAnswered = state.isAnswered,
                            onChoiceSelected = { vm.selectChoice(it) },
                        )
                    }
                }
                item { Spacer(Modifier.height(80.dp)) }
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
                    if (!state.isAnswered && !state.isModuleBreak) {
                        VoiceAnswerButton(
                            onListen = {
                                speechLauncher.launch(answerSpeechIntent())
                            },
                        )
                    }

                    AnimatedVisibility(
                        visible = state.isAnswered && !state.isModuleBreak,
                        enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
                    ) {
                        ResultFeedback(
                            isCorrect = state.isCorrect,
                            aiResponse = state.aiResponse,
                            aiSources = state.aiSources,
                            aiLoading = state.aiLoading || state.isFinishing,
                            isLast = state.currentModuleIndex == state.totalModules - 1 &&
                                state.currentQuestionIndex == scenario.questions.size - 1,
                            isLastQuestionInModule = state.currentQuestionIndex ==
                                scenario.questions.size - 1,
                            onNext = { vm.nextScenario() },
                        )
                    }

                    if (!state.isAnswered && !state.isModuleBreak) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            SentryText(
                                text = "Choose A, B, C, or D to see Pepper's feedback",
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

private fun speakWithRobotFallback(
    engine: TextToSpeech,
    text: String,
    utteranceId: String,
) {
    if (!PepperRobotBridge.say(text)) {
        engine.speak(
            text,
            TextToSpeech.QUEUE_FLUSH,
            null,
            utteranceId,
        )
    }
}

private fun buildQuestionSpeech(
    scenario: OwaspTrainingModule,
    question: OwaspQuestion,
): String {
    val answers = question.options.joinToString(". ") {
        "Option ${it.label}. ${it.text}"
    }
    return "${scenario.owaspId}. ${scenario.title}. ${question.scenario}. $answers"
}

private fun answerSpeechIntent(): Intent =
    Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
        )
        putExtra(RecognizerIntent.EXTRA_PROMPT, "Say option A, B, C, or D")
    }

private fun spokenAnswerToLabel(spoken: String): String? {
    val cleaned = spoken.trim().lowercase()
    return when {
        cleaned == "a" || "option a" in cleaned || "answer a" in cleaned -> "a"
        cleaned == "b" || "option b" in cleaned || "answer b" in cleaned -> "b"
        cleaned == "c" || "option c" in cleaned || "answer c" in cleaned -> "c"
        cleaned == "d" || "option d" in cleaned || "answer d" in cleaned -> "d"
        else -> null
    }
}

@Composable
private fun SessionTopBar(
    currentModule: Int,
    totalModules: Int,
    currentQuestion: Int,
    totalQuestionsInModule: Int,
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

        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            SentryText(
                text = "Module $currentModule of $totalModules",
                size = SentryTextSize.Lg,
                weight = FontWeight.Bold,
                color = Color.White,
            )
            SentryText(
                text = "$type - Question $currentQuestion of $totalQuestionsInModule",
                size = SentryTextSize.Sm,
                color = Color.White.copy(alpha = 0.85f),
            )
        }

        Row(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            repeat(totalModules.coerceAtMost(10)) { index ->
                Box(
                    modifier = Modifier
                        .width(18.dp)
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(
                            if (index < currentModule) Color.White
                            else Color.White.copy(alpha = 0.35f)
                        ),
                )
            }
        }
    }
}

@Composable
private fun ScenarioCard(
    scenario: OwaspTrainingModule,
    question: OwaspQuestion,
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
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(ScenarioHeaderPurple)
                .padding(vertical = 18.dp, horizontal = 20.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                SentryText(
                    text = "${scenario.owaspId}: ${scenario.title}",
                    size = SentryTextSize.Xl,
                    weight = FontWeight.Bold,
                    color = Color.White,
                    align = SentryTextAlign.Center,
                )
                Spacer(Modifier.height(4.dp))
                SentryText(
                    text = scenario.difficulty,
                    size = SentryTextSize.Sm,
                    color = Color.White.copy(alpha = 0.86f),
                    align = SentryTextAlign.Center,
                )
            }
        }

        Column(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            LessonBlock(
                title = "What this means",
                text = scenario.summary,
            )
            LessonBlock(
                title = "At work",
                text = scenario.workplaceTakeaway,
            )
            SentryText(
                text = question.scenario,
                size = SentryTextSize.Md,
                color = MaterialTheme.colorScheme.onBackground,
                lineHeight = 22.dp.value.sp,
                maxLines = Int.MAX_VALUE,
            )
        }

        Column(
            modifier = Modifier.padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            question.options.forEach { choice ->
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
private fun ModuleBreakCard(
    scenario: OwaspTrainingModule,
    isLastModule: Boolean,
    onContinue: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(scheme.surface)
            .border(1.dp, NeutralBorder, RoundedCornerShape(16.dp))
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SentryText(
            text = "${scenario.owaspId}: ${scenario.title} complete",
            size = SentryTextSize.Xl,
            weight = FontWeight.Bold,
            color = scheme.primary,
            align = SentryTextAlign.Center,
        )
        SentryText(
            text = "Take a short pause before moving to the next OWASP topic.",
            size = SentryTextSize.Md,
            color = scheme.onBackground,
            align = SentryTextAlign.Center,
            maxLines = Int.MAX_VALUE,
        )
        SentryText(
            text = scenario.workplaceTakeaway,
            size = SentryTextSize.Sm,
            color = scheme.outline,
            align = SentryTextAlign.Center,
            maxLines = Int.MAX_VALUE,
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(scheme.primary)
                .clickable { onContinue() }
                .padding(vertical = 16.dp),
            contentAlignment = Alignment.Center,
        ) {
            SentryText(
                text = if (isLastModule) "Finish training" else "Start next module",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = Color.White,
            )
        }
    }
}

@Composable
private fun VoiceAnswerButton(onListen: () -> Unit) {
    val scheme = MaterialTheme.colorScheme

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(scheme.primary)
            .clickable { onListen() }
            .padding(vertical = 14.dp),
        contentAlignment = Alignment.Center,
    ) {
        SentryText(
            text = "Speak answer: A, B, C, or D",
            size = SentryTextSize.Md,
            weight = FontWeight.Bold,
            color = Color.White,
            align = SentryTextAlign.Center,
        )
    }
}

@Composable
private fun LessonBlock(
    title: String,
    text: String,
) {
    val scheme = MaterialTheme.colorScheme

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(scheme.primary.copy(alpha = 0.06f))
            .padding(12.dp),
    ) {
        SentryText(
            text = title,
            size = SentryTextSize.Xs,
            weight = FontWeight.Bold,
            color = scheme.primary,
        )
        Spacer(Modifier.height(4.dp))
        SentryText(
            text = text,
            size = SentryTextSize.Sm,
            color = scheme.onBackground,
            maxLines = Int.MAX_VALUE,
        )
    }
}

@Composable
private fun ChoiceRow(
    choice: OwaspAnswerOption,
    isSelected: Boolean,
    isAnswered: Boolean,
    onSelect: () -> Unit,
) {
    val brand = LocalBrandColors.current
    val scheme = MaterialTheme.colorScheme

    val bgColor = when {
        isAnswered && choice.isCorrect -> brand.green.copy(alpha = 0.12f)
        isAnswered && isSelected -> brand.red.copy(alpha = 0.10f)
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
                .size(30.dp)
                .clip(CircleShape)
                .background(circleColor),
            contentAlignment = Alignment.Center,
        ) {
            SentryText(
                text = choice.label,
                size = SentryTextSize.Sm,
                weight = FontWeight.Bold,
                color = Color.White,
            )
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
    isLastQuestionInModule: Boolean,
    onNext: () -> Unit,
) {
    val brand = LocalBrandColors.current
    val scheme = MaterialTheme.colorScheme

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
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
            SentryText(
                text = if (isCorrect) "Correct decision" else "Risky decision",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = bannerColor,
                maxLines = 2,
            )
        }

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
                        text = "Pepper - saved grounded feedback",
                        size = SentryTextSize.Sm,
                        weight = FontWeight.Bold,
                        color = AiNameColor,
                    )
                }

                Spacer(Modifier.height(10.dp))

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

                if (aiLoading) {
                    Spacer(Modifier.height(12.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = scheme.primary,
                            strokeWidth = 2.dp,
                        )
                        Spacer(Modifier.width(8.dp))
                        SentryText(
                            text = "Saving your session...",
                            size = SentryTextSize.Sm,
                            color = scheme.outline,
                        )
                    }
                }
            }
        }

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
                    text = when {
                        isLast -> "Finish module"
                        isLastQuestionInModule -> "End module"
                        else -> "Next question"
                    },
                    size = SentryTextSize.Md,
                    weight = FontWeight.Bold,
                    color = Color.White,
                )
            }
        }
    }
}

private val ScenarioHeaderPurple = Color(0xFF5C6BC0)
private val AiBubbleBg = Color(0xFFE3F2FD)
private val AiBubbleBorder = Color(0xFFBBDEFB)
private val SourceTagBg = Color(0xFFE8EAF6)
private val SourceTagText = Color(0xFF3949AB)
private val NeutralBorder = Color(0xFFE0E0E0)
private val AiNameColor = Color(0xFF1565C0)
