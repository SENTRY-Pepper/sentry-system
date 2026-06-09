package com.sentry.app.features.chat

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.speech.RecognizerIntent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.R
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize

@Composable
fun ChatScreen(
    navController: NavHostController,
    vm: ChatViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    var speechError by remember { mutableStateOf("") }
    val speechLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@rememberLauncherForActivityResult
        val spoken = result.data
            ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()
            .orEmpty()
        if (spoken.isNotBlank()) {
            speechError = ""
            vm.sendText(spoken)
        }
    }

    LaunchedEffect(Unit) {
        focusManager.clearFocus(force = true)
    }

    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.lastIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background)
            .imePadding(),
    ) {
        ChatTopBar(
            onBack = { navController.navigateUp() },
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(scheme.surface)
                .padding(horizontal = 18.dp, vertical = 12.dp),
            contentAlignment = Alignment.Center,
        ) {
            PrimaryTalkButton(
                enabled = !state.loading,
                onClick = {
                    focusManager.clearFocus(force = true)
                    try {
                        speechLauncher.launch(chatSpeechIntent())
                    } catch (_: ActivityNotFoundException) {
                        speechError = "Speech recognition is not available on this device."
                    }
                },
            )
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            contentPadding = PaddingValues(vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            items(state.messages) { message ->
                if (message.isUser) {
                    UserMessage(text = message.text)
                } else {
                    PepperMessage(
                        text = message.text,
                        sources = message.sources,
                    )
                }
            }

            if (state.loading) {
                item { PepperTypingIndicator() }
            }
        }

        if (speechError.isNotBlank()) {
            SentryText(
                text = speechError,
                size = SentryTextSize.Xs,
                color = ErrorRed,
                modifier = Modifier.padding(horizontal = 18.dp, vertical = 4.dp),
            )
        }

        ChatInputBar(
            inputText = state.inputText,
            loading = state.loading,
            onInputChanged = vm::onInputChanged,
            onSend = { vm.sendMessage() },
            onVoice = {
                focusManager.clearFocus(force = true)
                try {
                    speechLauncher.launch(chatSpeechIntent())
                } catch (_: ActivityNotFoundException) {
                    speechError = "Speech recognition is not available on this device."
                }
            },
        )
    }
}

@Composable
private fun ChatTopBar(
    onBack: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(68.dp)
            .background(scheme.primary)
            .padding(horizontal = 14.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .clickable { onBack() }
                .padding(end = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                painter = painterResource(R.drawable.arrow_left_circle),
                contentDescription = "Back",
                tint = Color.White,
                modifier = Modifier.size(30.dp),
            )
            Spacer(Modifier.width(8.dp))
            SentryText(
                text = "Back",
                size = SentryTextSize.Md,
                weight = FontWeight.Bold,
                color = Color.White,
            )
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            SentryText(
                text = "Ask Pepper",
                size = SentryTextSize.Xl,
                weight = FontWeight.Bold,
                color = Color.White,
            )
            SentryText(
                text = "OWASP + Kenyan cyber law",
                size = SentryTextSize.Xs,
                color = Color.White.copy(alpha = 0.84f),
            )
        }

        Spacer(Modifier.width(64.dp))
    }
}

@Composable
private fun ChatInputBar(
    inputText: String,
    loading: Boolean,
    onInputChanged: (String) -> Unit,
    onSend: () -> Unit,
    onVoice: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val canSend = inputText.isNotBlank() && !loading

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(scheme.surface)
            .border(0.5.dp, InputBorder, RoundedCornerShape(0.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        OutlinedTextField(
            value = inputText,
            onValueChange = onInputChanged,
            placeholder = {
                SentryText(
                    text = "Ask about cybersecurity...",
                    size = SentryTextSize.Md,
                    color = scheme.outline,
                )
            },
            modifier = Modifier.weight(1f),
            singleLine = true,
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = scheme.primary,
                unfocusedBorderColor = InputBorder,
            ),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { if (canSend) onSend() }),
        )

        Spacer(Modifier.width(8.dp))

        Box(
            modifier = Modifier
                .height(44.dp)
                .clip(RoundedCornerShape(22.dp))
                .background(scheme.primary.copy(alpha = 0.12f))
                .clickable(enabled = !loading) { onVoice() }
                .padding(horizontal = 12.dp),
            contentAlignment = Alignment.Center,
        ) {
            SentryText(
                text = "Talk",
                size = SentryTextSize.Sm,
                weight = FontWeight.Bold,
                color = scheme.primary,
                align = SentryTextAlign.Center,
            )
        }

        Spacer(Modifier.width(8.dp))

        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(if (canSend) scheme.primary else scheme.outline.copy(alpha = 0.4f)),
            contentAlignment = Alignment.Center,
        ) {
            IconButton(onClick = { if (canSend) onSend() }, enabled = canSend) {
                Icon(
                    Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                    tint = Color.White,
                    modifier = Modifier.size(20.dp),
                )
            }
        }
    }
}

@Composable
private fun PrimaryTalkButton(
    enabled: Boolean,
    onClick: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    Box(
        modifier = Modifier
            .size(86.dp)
            .clip(CircleShape)
            .background(if (enabled) scheme.primary else scheme.outline.copy(alpha = 0.45f))
            .clickable(enabled = enabled) { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        SentryText(
            text = "Talk",
            size = SentryTextSize.Md,
            weight = FontWeight.Bold,
            color = Color.White,
            align = SentryTextAlign.Center,
        )
    }
}

@Composable
private fun UserMessage(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End,
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 480.dp)
                .clip(RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp))
                .background(UserBubble)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            SentryText(
                text = text,
                size = SentryTextSize.Md,
                color = Color.White,
                maxLines = Int.MAX_VALUE,
            )
        }
    }
}

@Composable
private fun PepperMessage(text: String, sources: List<String>) {
    val scheme = MaterialTheme.colorScheme

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.Start,
    ) {
        PepperAvatar()

        Spacer(Modifier.width(8.dp))

        Column(modifier = Modifier.widthIn(max = 540.dp)) {
            SentryText(
                text = "Pepper",
                size = SentryTextSize.Xs,
                weight = FontWeight.Bold,
                color = scheme.primary,
                modifier = Modifier.padding(start = 6.dp, bottom = 4.dp),
            )
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp))
                    .background(PepperBubble)
                    .border(
                        0.5.dp,
                        PepperBorder,
                        RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp),
                    )
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            ) {
                SentryText(
                    text = text,
                    size = SentryTextSize.Md,
                    color = scheme.onBackground,
                    maxLines = Int.MAX_VALUE,
                )
            }

            if (sources.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    sources.take(3).forEach { source ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(SourceTagBg)
                                .padding(horizontal = 8.dp, vertical = 3.dp),
                        ) {
                            SentryText(
                                text = source,
                                size = SentryTextSize.Xs,
                                color = SourceTagText,
                                maxLines = 1,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PepperTypingIndicator() {
    val scheme = MaterialTheme.colorScheme

    Row(verticalAlignment = Alignment.CenterVertically) {
        PepperAvatar()
        Spacer(Modifier.width(8.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(18.dp))
                .background(PepperBubble)
                .border(0.5.dp, PepperBorder, RoundedCornerShape(18.dp))
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    color = scheme.primary,
                    strokeWidth = 2.dp,
                )
                SentryText(
                    text = "Pepper is thinking...",
                    size = SentryTextSize.Md,
                    color = scheme.outline,
                )
            }
        }
    }
}

@Composable
private fun PepperAvatar() {
    Box(
        modifier = Modifier
            .size(40.dp)
            .clip(CircleShape)
            .background(Color.White)
            .border(1.dp, InputBorder, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Image(
            painter = painterResource(R.drawable.pepper_robot),
            contentDescription = "Pepper",
            modifier = Modifier
                .size(34.dp)
                .clip(CircleShape),
            contentScale = ContentScale.Crop,
        )
    }
}

private fun chatSpeechIntent(): Intent =
    Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
        )
        putExtra(RecognizerIntent.EXTRA_PROMPT, "Ask Pepper your cybersecurity question")
    }

private val UserBubble = Color(0xFF039BE5)
private val PepperBubble = Color(0xFFEAF7FF)
private val PepperBorder = Color(0xFFB7E3F8)
private val SourceTagBg = Color(0xFFE8F5E9)
private val SourceTagText = Color(0xFF2E7D32)
private val InputBorder = Color(0xFFE0E0E0)
private val ErrorRed = Color(0xFFC62828)
