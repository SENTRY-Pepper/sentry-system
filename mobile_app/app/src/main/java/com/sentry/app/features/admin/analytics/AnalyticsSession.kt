package com.sentry.app.features.admin.analytics

data class AnalyticsSession(
    val id: String,
    val condition: String,
    val pre: String,
    val post: String,
    val gain: String,
    val grounding: String,
    val isGrounded: Boolean,
) {
    companion object {
        val SESSIONS = listOf(
            AnalyticsSession("EMP_042", "Grounded", "45%", "72%", "+27%", "0.82", true),
            AnalyticsSession("EMP_031", "Grounded", "50%", "65%", "+15%", "0.74", true),
            AnalyticsSession("EMP_019", "Baseline", "48%", "50%", "+2%", "0.08", false),
            AnalyticsSession("EMP_007", "Grounded", "40%", "68%", "+28%", "0.79", true),
            AnalyticsSession("EMP_012", "Baseline", "52%", "55%", "+3%", "0.11", false),
            AnalyticsSession("EMP_024", "Grounded", "38%", "70%", "+32%", "0.88", true),
            AnalyticsSession("EMP_036", "Baseline", "55%", "57%", "+2%", "0.09", false),
        )
    }
}
