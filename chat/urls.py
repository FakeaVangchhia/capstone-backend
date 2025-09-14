from django.urls import path
from .views import (
    ChatSessionListCreate,
    SessionMessagesList,
    MessageCreate,
    RegisterView,
    LoginView,
    MeView,
    LogoutView,
    ChatSessionDetail,
)

urlpatterns = [
    # Auth
    path("auth/register", RegisterView.as_view()),
    path("auth/login", LoginView.as_view()),
    path("auth/me", MeView.as_view()),
    path("auth/logout", LogoutView.as_view()),
    # Chat
    path("chat-sessions", ChatSessionListCreate.as_view()),
    path("chat-sessions/<uuid:session_id>", ChatSessionDetail.as_view()),
    path("chat-sessions/<uuid:session_id>/messages", SessionMessagesList.as_view()),
    path("messages", MessageCreate.as_view()),
]
