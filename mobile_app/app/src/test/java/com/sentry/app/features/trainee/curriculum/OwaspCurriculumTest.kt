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
    fun every_module_has_two_or_three_questions_with_four_labelled_answers() {
        OwaspCurriculum.modules.forEach { module ->
            assertTrue(module.questions.size in 2..3)
            module.questions.forEach { question ->
                assertEquals(4, question.options.size)
                assertEquals(listOf("A", "B", "C", "D"), question.options.map { it.label })
                assertEquals(1, question.options.count { it.isCorrect })
                assertTrue(question.options.any { it.id == question.correctAnswerId && it.isCorrect })
            }
        }
    }

    @Test
    fun correct_answers_are_distributed_across_all_answer_letters() {
        val correctLabels = OwaspCurriculum.modules
            .flatMap { it.questions }
            .map { question -> question.options.first { it.isCorrect }.label }
            .toSet()

        assertEquals(setOf("A", "B", "C", "D"), correctLabels)
    }
}
