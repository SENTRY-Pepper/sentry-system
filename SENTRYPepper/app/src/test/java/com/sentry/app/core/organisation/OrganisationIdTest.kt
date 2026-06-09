package com.sentry.app.core.organisation

import org.junit.Assert.assertEquals
import org.junit.Test

class OrganisationIdTest {
    @Test
    fun normaliseOrganisationId_convertsDisplayNameToCanonicalId() {
        assertEquals("SENTRY_STUDY", normaliseOrganisationId(" Sentry Study "))
        assertEquals("HERITAGE_INSURANCE", normaliseOrganisationId("Heritage Insurance"))
        assertEquals("JKUAT_PILOT", normaliseOrganisationId("jkuat-pilot"))
    }

    @Test
    fun normaliseOrganisationId_removesRepeatedSeparators() {
        assertEquals("SENTRY_STUDY", normaliseOrganisationId("SENTRY   ---   STUDY"))
    }
}

