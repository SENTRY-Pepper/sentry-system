package com.sentry.app.core.organisation

private val nonIdentifierChars = Regex("[^A-Z0-9]+")
private val repeatedUnderscores = Regex("_+")

fun normaliseOrganisationId(value: String): String =
    value
        .trim()
        .uppercase()
        .replace(nonIdentifierChars, "_")
        .replace(repeatedUnderscores, "_")
        .trim('_')

