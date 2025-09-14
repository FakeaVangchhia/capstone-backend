from rest_framework import serializers
from django.contrib.auth.hashers import make_password, check_password
from .models import ChatSession, Message, AppUser

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


class AppUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppUser
        fields = ["id", "username", "created_at"]
        read_only_fields = ["id", "created_at"]


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, min_length=4)

    def validate_username(self, value):
        if AppUser.objects.filter(username=value).exists():
            raise serializers.ValidationError("Username already exists")
        return value

    def create(self, validated_data):
        user = AppUser(username=validated_data["username"], password_hash=make_password(validated_data["password"]))
        user.save()
        return user


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        username = attrs.get("username")
        password = attrs.get("password")
        try:
            user = AppUser.objects.get(username=username)
        except AppUser.DoesNotExist:
            raise serializers.ValidationError({"message": "Invalid credentials"})
        if not check_password(password, user.password_hash):
            raise serializers.ValidationError({"message": "Invalid credentials"})
        attrs["user_obj"] = user
        return attrs
