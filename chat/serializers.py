from rest_framework import serializers
from .models import ChatSession, Message

class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = ["id", "user_id", "title", "created_at", "updated_at"]

class MessageSerializer(serializers.ModelSerializer):
    sessionId = serializers.UUIDField(source="session.id", write_only=True, required=True)

    class Meta:
        model = Message
        fields = ["id", "sessionId", "content", "role", "created_at"]
        read_only_fields = ["id", "created_at"]

    def create(self, validated_data):
        # validated_data['session'] is a dict like {'id': UUID}
        session_info = validated_data.pop("session")
        session_id = session_info.get("id")
        session = ChatSession.objects.get(id=session_id)
        return Message.objects.create(session=session, **validated_data)
