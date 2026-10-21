from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import LearningProgress, Lesson


class LessonPagesTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.lesson = Lesson.objects.get(slug="introduce-yourself-in-isl")
        cls.user = get_user_model().objects.create_user(username="learner", password="strong-test-password-42")

    def test_lesson_catalog_shows_beginner_lessons(self):
        response = self.client.get(reverse("learning:lesson_list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The ISL alphabet")
        self.assertContains(response, "/static/images/isl-alphabet-chart.jpg")
        self.assertContains(response, "Introduce yourself")
        self.assertContains(response, "Indian Sign Language")

    def test_catalog_can_filter_to_alphabet(self):
        response = self.client.get(reverse("learning:lesson_list"), {"category": Lesson.Category.ALPHABET})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The ISL alphabet")
        self.assertNotContains(response, "Introduce yourself")

    def test_lesson_view_saves_progress_for_signed_in_user(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse("learning:lesson_detail", args=(self.lesson.slug,)))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Open the original resource from ISLRTC")
        progress = LearningProgress.objects.get(user=self.user, lesson=self.lesson)
        self.assertEqual(progress.status, LearningProgress.Status.STARTED)

    def test_alphabet_lesson_displays_full_sign_chart(self):
        response = self.client.get(reverse("learning:lesson_detail", args=("isl-alphabet",)))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ISLRTC chart showing Indian Sign Language hand signs for English letters A to Z")
        self.assertContains(response, "Open full size")

    def test_learner_can_log_practice_and_complete_a_lesson(self):
        self.client.force_login(self.user)
        detail_url = reverse("learning:lesson_detail", args=(self.lesson.slug,))
        self.client.get(detail_url)

        practice_response = self.client.post(reverse("learning:practice", args=(self.lesson.slug,)))
        self.assertRedirects(practice_response, detail_url)
        self.client.post(reverse("learning:complete", args=(self.lesson.slug,)))

        progress = LearningProgress.objects.get(user=self.user, lesson=self.lesson)
        self.assertEqual(progress.practice_count, 1)
        self.assertEqual(progress.status, LearningProgress.Status.COMPLETED)
        self.assertIsNotNone(progress.completed_at)
