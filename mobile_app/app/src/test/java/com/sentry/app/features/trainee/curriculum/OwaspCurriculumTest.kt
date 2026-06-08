package com.sentry.app.features.trainee.curriculum

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OwaspCurriculumTest {
    @Test
    fun curriculum_has_one_module_for_each_owasp_top_10_category() {
        val expectedIds = (1..10).map { "A%02d".format(it) }

        assertEquals(expectedIds, OwaspCurriculum.modules.map { it.owaspId })
    }

    @Test
    fun every_module_has_four_labelled_answers_and_one_correct_answer() {
        OwaspCurriculum.modules.forEach { module ->
            assertEquals(4, module.options.size)
            assertEquals(listOf("A", "B", "C", "D"), module.options.map { it.label })
            assertEquals(1, module.options.count { it.isCorrect })
            assertTrue(module.options.any { it.id == module.correctAnswerId && it.isCorrect })
        }
    }
}
