package com.sentry.app.features.trainee.session

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.QueryResponse
import com.sentry.app.data.repository.QueryRepository
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ScenarioChoice(
    val id: String,
    val text: String,
    val isCorrect: Boolean,
)

data class Scenario(
    val id: String,
    val type: String,
    val title: String,
    val prompt: String,
    val choices: List<ScenarioChoice>,
    val correctExplanation: String,
)

data class SessionUiState(
    val sessionId: String            = "",
    val currentIndex: Int            = 0,
    val totalScenarios: Int          = 5,
    val selectedChoiceId: String?    = null,
    val isAnswered: Boolean          = false,
    val isCorrect: Boolean           = false,
    val aiResponse: String           = "",
    val aiSources: List<String>      = emptyList(),
    val aiLoading: Boolean           = false,
    val isComplete: Boolean          = false,
    val error: String                = "",
)

private val SCENARIOS = listOf(
    Scenario(
        id     = "phishing-01",
        type   = "Phishing Detection",
        title  = "Suspicious email from IT Support",
        prompt = "You receive this email at 8:43 AM from IT Support at support@heriitage-bank.com — notice the spelling. Subject: \"Urgent — verify your account immediately\". The email says your account will be suspended in 24 hours unless you click a link to verify your credentials. What do you do?",
        choices = listOf(
            ScenarioChoice("click",   "Click the link and verify your credentials immediately", false),
            ScenarioChoice("ignore",  "Ignore the email and delete it",                          false),
            ScenarioChoice("report",  "Report it to IT security without clicking any links",     true),
            ScenarioChoice("forward", "Forward it to colleagues to warn them",                   false),
        ),
        correctExplanation = "Reporting to IT without clicking is the correct action.",
    ),
    Scenario(
        id     = "usb-drop-01",
        type   = "USB Drop Simulation",
        title  = "Unknown USB drive found at your desk",
        prompt = "You arrive at work on Monday morning and find a USB drive on your desk. There is no label on it and you do not know who left it there. It could contain important files — or it could be something else entirely. What do you do?",
        choices = listOf(
            ScenarioChoice("plug_in",   "Plug it into your computer to check what is on it",       false),
            ScenarioChoice("take_home", "Take it home to check on your personal computer",          false),
            ScenarioChoice("hand_it",   "Hand it to IT security without plugging it in",            true),
            ScenarioChoice("bin_it",    "Throw it in the bin so no one else picks it up",           false),
        ),
        correctExplanation = "Handing to IT without plugging in is the safest action.",
    ),
    Scenario(
        id     = "password-01",
        type   = "Password Hygiene",
        title  = "Creating a secure password for a new system",
        prompt = "Your company is rolling out a new internal finance portal and you need to create your account password. IT policy requires a strong password. Which approach do you take?",
        choices = listOf(
            ScenarioChoice("name_year",   "Use your name and birth year — easy to remember",               false),
            ScenarioChoice("reuse",       "Reuse your email password since it is already strong",           false),
            ScenarioChoice("manager",     "Use a password manager to generate a unique 16-character password", true),
            ScenarioChoice("passphrase",  "Write a memorable passphrase of four or more random words",     true),
        ),
        correctExplanation = "A password manager or strong passphrase are both correct approaches.",
    ),
    Scenario(
        id     = "social-engineering-01",
        type   = "Voice Social Engineering",
        title  = "Suspicious phone call from IT helpdesk",
        prompt = "You receive a phone call. The caller says they are from your company's IT helpdesk and that there is a critical security issue with your account. They say they need your username and password immediately to fix it before your data is lost. They sound professional and know your name. What do you do?",
        choices = listOf(
            ScenarioChoice("give_creds",    "Give them your credentials since they know your name",                false),
            ScenarioChoice("partial",       "Give them your username but not your password",                       false),
            ScenarioChoice("hang_up_report","Hang up and call IT back on the verified company number to confirm", true),
            ScenarioChoice("ask_email",     "Ask them to send you an email instead and wait",                      false),
        ),
        correctExplanation = "Legitimate IT support will never ask for your password over the phone.",
    ),
    Scenario(
        id     = "network-01",
        type   = "Network Hygiene",
        title  = "Working from a coffee shop on public WiFi",
        prompt = "You are working from a coffee shop and need to access your company's internal system to submit an urgent report. The coffee shop has free public WiFi. Your mobile data is running low. What do you do?",
        choices = listOf(
            ScenarioChoice("public_wifi",  "Connect directly to the public WiFi and access the system",           false),
            ScenarioChoice("wait",         "Wait until you get back to the office",                                false),
            ScenarioChoice("connect_vpn",  "Connect to public WiFi but use your company VPN before accessing systems", true),
            ScenarioChoice("hotspot",      "Use your phone as a personal hotspot instead of public WiFi",          true),
        ),
        correctExplanation = "VPN or personal hotspot protects your data from interception on public networks.",
    ),
)

@HiltViewModel
class SessionViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
    private val queryRepository: QueryRepository,
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])

    private val _uiState = MutableStateFlow(
        SessionUiState(
            sessionId      = sessionId,
            totalScenarios = SCENARIOS.size,
        )
    )
    val uiState = _uiState.asStateFlow()

    fun getCurrentScenario(): Scenario = SCENARIOS[_uiState.value.currentIndex]

    fun selectChoice(choiceId: String) {
        val state = _uiState.value
        if (state.isAnswered) return

        val scenario = getCurrentScenario()
        val choice   = scenario.choices.first { it.id == choiceId }

        _uiState.value = state.copy(
            selectedChoiceId = choiceId,
            isAnswered       = true,
            isCorrect        = choice.isCorrect,
            aiLoading        = true,
        )

        // Fire AI query for grounded explanation
        viewModelScope.launch {
            val query = "Explain why the correct answer to this cybersecurity scenario is correct: ${scenario.title}. ${scenario.correctExplanation}"
            when (val result = queryRepository.groundedQuery(
                query      = query,
                scenarioId = scenario.id,
            )) {
                is NetworkResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        aiResponse = result.data.response,
                        aiSources  = result.data.sources,
                        aiLoading  = false,
                    )
                }
                else -> {
                    _uiState.value = _uiState.value.copy(
                        aiResponse = scenario.correctExplanation,
                        aiSources  = emptyList(),
                        aiLoading  = false,
                    )
                }
            }

            // Log the interaction fire-and-forget
            sessionRepository.logInteraction(
                sessionId        = sessionId,
                scenarioId       = scenario.id,
                scenarioType     = scenario.type,
                decision         = if (choice.isCorrect) "correct" else "risky",
                employeeResponse = choice.text,
                responseTimeMs   = null,
                correctionLoops  = 0,
                aiLatencyMs      = null,
                aiSources        = _uiState.value.aiSources.joinToString(","),
            )
        }
    }

    fun nextScenario() {
        val state = _uiState.value
        val nextIndex = state.currentIndex + 1
        if (nextIndex >= SCENARIOS.size) {
            _uiState.value = state.copy(isComplete = true)
        } else {
            _uiState.value = state.copy(
                currentIndex     = nextIndex,
                selectedChoiceId = null,
                isAnswered       = false,
                isCorrect        = false,
                aiResponse       = "",
                aiSources        = emptyList(),
                aiLoading        = false,
            )
        }
    }
}

