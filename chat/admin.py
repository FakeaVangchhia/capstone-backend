from django.contrib import admin
from .models import AppUser, AuthToken, ChatSession, Message


@admin.register(AppUser)
class AppUserAdmin(admin.ModelAdmin):
    list_display = ("username", "is_admin", "created_at", "id")
    list_filter = ("is_admin", "created_at")
    search_fields = ("username", "id")
    ordering = ("-created_at",)


@admin.register(AuthToken)
class AuthTokenAdmin(admin.ModelAdmin):
    list_display = ("key", "user", "created_at")
    search_fields = ("key", "user__username")
    ordering = ("-created_at",)


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user_id", "title", "created_at", "updated_at")
    search_fields = ("id", "user_id", "title")
    ordering = ("-updated_at",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("id", "session__id", "content")
    ordering = ("created_at",)


