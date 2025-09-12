from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import ChatSession, Message
from .serializers import ChatSessionSerializer, MessageSerializer
from .rag import rag
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


class ChatSessionListCreate(APIView):
    def get(self, request):
        user_id = request.query_params.get("userId")
        qs = ChatSession.objects.all()
        if user_id:
            qs = qs.filter(user_id=user_id)
        serializer = ChatSessionSerializer(qs, many=True)
        return Response(serializer.data)

    @method_decorator(csrf_exempt)
    def post(self, request):
        serializer = ChatSessionSerializer(data=request.data)
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


class MessageCreate(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request):
        serializer = MessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"message": "Invalid message data", "issues": serializer.errors}, status=422)
        data = serializer.validated_data
        message = serializer.save()

        # bump session updated_at
        ChatSession.objects.filter(id=message.session.id).update()

        # If user message, generate assistant reply using RAG
        if message.role == "user":
            try:
                result = rag.answer(message.content)
                sources = result.get("sources", [])
                sources_text = "\n\nSources:\n" + "\n".join([f"- {s.get('preview')}" for s in sources]) if sources else ""
                assistant = Message.objects.create(
                    session=message.session,
                    content=f"{result.get('answer','')}{sources_text}",
                    role="assistant",
                )
            except Exception:
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
