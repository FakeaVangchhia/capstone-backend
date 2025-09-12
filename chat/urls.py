from django.urls import path
from .views import ChatSessionListCreate, SessionMessagesList, MessageCreate

urlpatterns = [
    path("chat-sessions", ChatSessionListCreate.as_view()),
    path("chat-sessions/<uuid:session_id>/messages", SessionMessagesList.as_view()),
    path("messages", MessageCreate.as_view()),
]
