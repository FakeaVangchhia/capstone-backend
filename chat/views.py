from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import ChatSession, Message, AppUser, AuthToken
from .serializers import (
    ChatSessionSerializer,
    MessageSerializer,
    AppUserSerializer,
    RegisterSerializer,
    LoginSerializer,
)
from .rag import rag
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import os
import time
from django.utils import timezone


def get_bearer_token(request) -> str | None:
    auth_header = request.headers.get("Authorization") or request.META.get("HTTP_AUTHORIZATION")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def get_auth_user(request) -> AppUser | None:
    token = get_bearer_token(request)
    if not token:
        return None
    try:
        t = AuthToken.objects.select_related("user").get(key=token)
        return t.user
    except AuthToken.DoesNotExist:
        return None


class RegisterView(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"message": "Invalid data", "issues": serializer.errors}, status=422)
        user = serializer.save()
        data = AppUserSerializer(user).data
        return Response(data, status=201)


class LoginView(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"message": "Invalid credentials"}, status=401)
        user: AppUser = serializer.validated_data["user_obj"]
        # issue token
        token = AuthToken.objects.create(user=user)
        return Response({
            "token": str(token.key),
            "user": AppUserSerializer(user).data,
        })


class MeView(APIView):
    def get(self, request):
        user = get_auth_user(request)
        if not user:
            return Response({"message": "Unauthorized"}, status=401)
        return Response(AppUserSerializer(user).data)


class LogoutView(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        token = get_bearer_token(request)
        if token:
            AuthToken.objects.filter(key=token).delete()
        return Response({"success": True})


class ChatSessionListCreate(APIView):
    def get(self, request):
        user_id = request.query_params.get("userId")
        qs = ChatSession.objects.all()
        if user_id:
            qs = qs.filter(user_id=user_id)
        else:
            # Default: require auth; only return that user's sessions
            user = get_auth_user(request)
            if not user:
                return Response({"message": "Unauthorized"}, status=401)
            qs = qs.filter(user_id=str(user.id))
        serializer = ChatSessionSerializer(qs, many=True)
        return Response(serializer.data)

    @method_decorator(csrf_exempt)
    def post(self, request):
        data = dict(request.data)
        # If authenticated and no explicit user_id was sent, associate automatically
        user = get_auth_user(request)
        if user and not data.get("user_id"):
            data["user_id"] = str(user.id)
        serializer = ChatSessionSerializer(data=data)
        if not serializer.is_valid():
            return Response({"message": "Invalid session data", "issues": serializer.errors}, status=422)
        session = serializer.save()
        return Response(ChatSessionSerializer(session).data)


class SessionMessagesList(APIView):
    def get(self, request, session_id):
        session = get_object_or_404(ChatSession, id=session_id)
        msgs = session.messages.all().order_by("created_at")
        out = []
        for m in msgs:
            out.append({
                "id": str(m.id),
                "sessionId": str(session.id),
                "content": m.content,
                "role": m.role,
                "created_at": m.created_at,
            })
        return Response(out)


class ChatSessionDetail(APIView):
    def _get_session(self, session_id: str) -> ChatSession:
        return get_object_or_404(ChatSession, id=session_id)

    def _authorized(self, request, session: ChatSession) -> bool:
        user = get_auth_user(request)
        if not user:
            return False
        return session.user_id == str(user.id)

    @method_decorator(csrf_exempt)
    def patch(self, request, session_id: str):
        session = self._get_session(session_id)
        if not self._authorized(request, session):
            return Response({"message": "Forbidden"}, status=403)
        title = request.data.get("title")
        if not title:
            return Response({"message": "title is required"}, status=422)
        session.title = title
        session.updated_at = timezone.now()
        session.save(update_fields=["title", "updated_at"])
        return Response(ChatSessionSerializer(session).data)

    @method_decorator(csrf_exempt)
    def delete(self, request, session_id: str):
        session = self._get_session(session_id)
        if not self._authorized(request, session):
            return Response({"message": "Forbidden"}, status=403)
        session.delete()
        return Response({"success": True})


class MessageCreate(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        # Authorization: only the owner of a session can post messages to it
        auth_user = get_auth_user(request)

        serializer = MessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"message": "Invalid message data", "issues": serializer.errors}, status=422)
        data = serializer.validated_data

        # Extract session and enforce ownership before creating the message
        session_info = data.get("session") or {}
        session_id = session_info.get("id")
        session = get_object_or_404(ChatSession, id=session_id)

        # If the session is associated with a user, require matching auth
        if session.user_id:
            if not auth_user or session.user_id != str(auth_user.id):
                return Response({"message": "Forbidden"}, status=403)

        message = serializer.save()

        # bump session updated_at
        ChatSession.objects.filter(id=message.session.id).update(updated_at=timezone.now())

        # If user message, generate assistant reply using RAG
        if message.role == "user":
            try:
                result = rag.answer(message.content)
                sources = result.get("sources", [])
                sources_text = "\n\nSources:\n" + "\n".join([f"- {s.get('preview')}" for s in sources]) if sources else ""
                # Simulate typing latency so the frontend can show a typing indicator
                answer_text = f"{result.get('answer','')}{sources_text}"
                # Allow override via env; otherwise compute based on length
                cfg_delay = os.getenv("ASSISTANT_TYPING_SEC")
                if cfg_delay is not None:
                    try:
                        delay = max(0.0, float(cfg_delay))
                    except Exception:
                        delay = 0.0
                else:
                    # 0.8s base + 12ms per word, clamped to 3.5s
                    word_count = len(answer_text.split()) or 1
                    delay = min(3.5, 0.8 + 0.012 * word_count)
                if delay > 0:
                    time.sleep(delay)
                assistant = Message.objects.create(
                    session=message.session,
                    content=answer_text,
                    role="assistant",
                )
            except Exception as e:
                # Log the error so we can diagnose issues in development/server logs
                try:
                    print(f"[MessageCreate] RAG error: {e}")
                except Exception:
                    pass
                Message.objects.create(
                    session=message.session,
                    content="Sorry, I had trouble searching the university data. Please try again.",
                    role="assistant",
                )

        # return the created user message
        out = {
            "id": str(message.id),
            "sessionId": str(message.session.id),
            "content": message.content,
            "role": message.role,
            "created_at": message.created_at,
        }
        return Response(out)
