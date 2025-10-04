from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.views.decorators.http import require_http_methods
from .models import ChatSession, Message, AppUser, AuthToken
from .serializers import (
    ChatSessionSerializer,
    MessageSerializer,
    AppUserSerializer,
    RegisterSerializer,
    LoginSerializer,
)
from .rag import rag
import json
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import os
import time
from django.utils import timezone
from django.core.files.uploadedfile import UploadedFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import os
from django.conf import settings


def get_bearer_token(request) -> str | None:
    auth_header = request.headers.get("Authorization") or request.META.get("HTTP_AUTHORIZATION")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type"""
    filename = getattr(uploaded_file, "name", "").lower()
    content_bytes = uploaded_file.read()
    
    try:
        if filename.endswith('.txt'):
            # Plain text file
            return content_bytes.decode("utf-8", errors="ignore")
        
        elif filename.endswith('.pdf'):
            # PDF file
            import io
            pdf_file = io.BytesIO(content_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif filename.endswith(('.docx', '.doc')):
            # Word document
            import io
            doc_file = io.BytesIO(content_bytes)
            doc = docx.Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            # Try to decode as text for other file types
            return content_bytes.decode("utf-8", errors="ignore")
            
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        # Fallback to trying UTF-8 decode
        try:
            return content_bytes.decode("utf-8", errors="ignore")
        except:
            return ""


def get_auth_user(request) -> AppUser | None:
    token = get_bearer_token(request)
    if not token:
        return None
    try:
        # Validate that token looks like a UUID before querying
        import uuid
        uuid.UUID(token)  # This will raise ValueError if not a valid UUID
        t = AuthToken.objects.select_related("user").get(key=token)
        return t.user
    except (AuthToken.DoesNotExist, ValueError):
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


class AdminUploadView(View):
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    
    def options(self, request):
        # Handle CORS preflight request
        response = JsonResponse({})
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Accept-Encoding"
        response["Access-Control-Allow-Credentials"] = "true"
        response["Access-Control-Max-Age"] = "86400"
        print("OPTIONS request received - sending CORS headers")
        return response
    
    def post(self, request):
        # Check authentication using the same function
        user = get_auth_user(request)
        if not user or not getattr(user, "is_admin", False):
            return JsonResponse({"message": "Forbidden"}, status=403)

        print(f"=== UPLOAD DEBUG ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Content-Length: {request.META.get('CONTENT_LENGTH', 'Not set')}")
        print(f"FILES keys: {list(request.FILES.keys())}")
        print(f"POST keys: {list(request.POST.keys())}")
        print(f"Method: {request.method}")
        print(f"HTTP_CONTENT_TYPE: {request.META.get('HTTP_CONTENT_TYPE', 'Not set')}")
        print(f"REQUEST_METHOD: {request.META.get('REQUEST_METHOD', 'Not set')}")
        
        # Check if body exists and preview it
        try:
            body_size = len(request.body) if hasattr(request, 'body') else 0
            print(f"Body size: {body_size}")
            if body_size > 0:
                body_preview = request.body[:100]
                print(f"Body preview: {body_preview}")
        except Exception as e:
            print(f"Error reading body: {e}")
        
        # Try to get the uploaded file
        upload = None
        if request.FILES:
            upload = request.FILES.get("file")
            if not upload:
                # Get the first file regardless of key name
                upload = next(iter(request.FILES.values()))
        
        print(f"Upload object: {upload}")
        print(f"=== END DEBUG ===")
        
        if not upload:
            return JsonResponse({
                "message": "file is required",
                "received_fields": list(request.FILES.keys()),
                "debug_info": {
                    "content_type": request.content_type,
                    "content_length": request.META.get('CONTENT_LENGTH'),
                    "method": request.method
                }
            }, status=422)
        
        # Process the uploaded file
        try:
            filename = getattr(upload, "name", "uploaded")
            source = f"upload:{filename}"
            content_bytes = upload.read()
            
            try:
                text = content_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            
            if not text:
                return JsonResponse({"message": "Unsupported or empty file"}, status=400)
            
            # chunk and upsert
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_text(text)
            added = rag.upsert_texts(chunks, source=source)
            
            response = JsonResponse({"success": True, "chunks": added})
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Credentials"] = "true"
            return response
            
        except Exception as e:
            return JsonResponse({"message": f"Failed to process file: {str(e)}"}, status=400)


def test_upload_page(request):
    """Serve the test upload HTML page"""
    from django.http import FileResponse
    import os
    from django.conf import settings
    
    html_path = os.path.join(settings.BASE_DIR, 'test_upload.html')
    if os.path.exists(html_path):
        return FileResponse(open(html_path, 'rb'), content_type='text/html')
    else:
        return HttpResponse("Test upload page not found", status=404)


@csrf_exempt
def admin_upload_simple(request):
    """Simple function-based view for file upload"""
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = HttpResponse()
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
        response["Access-Control-Allow-Credentials"] = "true"
        response["Access-Control-Max-Age"] = "86400"
        print("Function view: OPTIONS request handled")
        return response
    
    if request.method != 'POST':
        return JsonResponse({"message": "Method not allowed"}, status=405)
    
    # Check authentication (flexible for testing)
    user = get_auth_user(request)
    if user and not getattr(user, "is_admin", False):
        # If user is logged in but not admin, deny access
        response = JsonResponse({"message": "Forbidden - Admin access required"}, status=403)
        response["Access-Control-Allow-Origin"] = "*"
        return response
    elif not user:
        # If no user is logged in, allow upload for testing (you can change this later)
        print("No authentication provided - allowing upload for testing")
    
    print(f"=== SIMPLE UPLOAD DEBUG ===")
    print(f"Content-Type: {request.content_type}")
    print(f"Content-Length: {request.META.get('CONTENT_LENGTH', 'Not set')}")
    print(f"FILES keys: {list(request.FILES.keys())}")
    print(f"POST keys: {list(request.POST.keys())}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    
    # Check if body exists
    try:
        body_size = len(request.body) if hasattr(request, 'body') else 0
        print(f"Body size: {body_size}")
        if body_size > 0 and body_size < 1000:  # Only preview small bodies
            print(f"Body preview: {request.body[:200]}")
    except Exception as e:
        print(f"Error reading body: {e}")
    
    # Try to get the uploaded file
    upload = request.FILES.get("file")
    if not upload and request.FILES:
        upload = next(iter(request.FILES.values()))
    
    print(f"Upload object: {upload}")
    print(f"=== END SIMPLE DEBUG ===")
    
    if not upload:
        response = JsonResponse({
            "message": "file is required",
            "received_fields": list(request.FILES.keys()),
        }, status=422)
        response["Access-Control-Allow-Origin"] = "*"
        return response
    
    # Process the file
    try:
        filename = getattr(upload, "name", "uploaded")
        source = f"upload:{filename}"
        
        # Save the uploaded file to uploads directory
        upload_dir = os.path.join(settings.BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create unique filename to avoid conflicts
        import time
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Save the file
        with open(file_path, 'wb') as f:
            for chunk in upload.chunks():
                f.write(chunk)
        
        print(f"File saved to: {file_path}")
        
        # Extract text based on file type
        upload.seek(0)  # Reset file pointer
        text = extract_text_from_file(upload)
        
        if not text or len(text.strip()) == 0:
            response = JsonResponse({
                "message": f"Could not extract text from {filename}. Supported formats: TXT, PDF, DOCX",
                "file_saved": file_path
            }, status=400)
            response["Access-Control-Allow-Origin"] = "*"
            return response
        
        print(f"Extracted text length: {len(text)} characters")
        
        # chunk and upsert
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)
        added = rag.upsert_texts(chunks, source=source)
        
        response = JsonResponse({
            "success": True, 
            "chunks": added,
            "filename": filename,
            "text_length": len(text),
            "file_saved": file_path
        })
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Credentials"] = "true"
        return response
        
    except Exception as e:
        response = JsonResponse({"message": f"Failed to process file: {str(e)}"}, status=400)
        response["Access-Control-Allow-Origin"] = "*"
        return response




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


class StudyBuddyView(APIView):
    """Quick study tips and motivation endpoint"""
    
    def get(self, request):
        tip_type = request.query_params.get("type", "tip")
        
        if tip_type == "motivation":
            message = rag.get_study_motivation()
            response_type = "motivation"
        else:
            message = rag.get_study_tip_of_day()
            response_type = "study_tip"
            
        return Response({
            "type": response_type,
            "message": message,
            "timestamp": timezone.now().isoformat()
        })


class StudyResourcesView(APIView):
    """Provide study resources and academic support info"""
    
    def get(self, request):
        resources = {
            "study_techniques": [
                {
                    "name": "Pomodoro Technique",
                    "description": "25 minutes focused study + 5 minute break",
                    "best_for": "Maintaining focus and avoiding burnout"
                },
                {
                    "name": "Active Recall",
                    "description": "Test yourself instead of just re-reading",
                    "best_for": "Better retention and understanding"
                },
                {
                    "name": "Spaced Repetition",
                    "description": "Review material at increasing intervals",
                    "best_for": "Long-term memory retention"
                },
                {
                    "name": "Feynman Technique",
                    "description": "Explain concepts in simple terms",
                    "best_for": "Deep understanding of complex topics"
                }
            ],
            "campus_study_spots": [
                {
                    "location": "Main Library",
                    "best_for": "Quiet individual study",
                    "features": ["Silent zones", "Research resources", "Group study rooms"]
                },
                {
                    "location": "Hostel Common Areas",
                    "best_for": "Group study sessions",
                    "features": ["High-speed Wi-Fi", "Collaborative environment", "Peer support"]
                },
                {
                    "location": "Lakeside Campus Outdoor Areas",
                    "best_for": "Fresh air study breaks",
                    "features": ["Natural environment", "Stress relief", "Creative thinking"]
                }
            ],
            "academic_support": [
                {
                    "resource": "Faculty Office Hours",
                    "description": "One-on-one help from professors",
                    "how_to_access": "Check with individual faculty for their office hours"
                },
                {
                    "resource": "Study Groups",
                    "description": "Collaborative learning with peers",
                    "how_to_access": "Form groups with classmates or join existing ones"
                },
                {
                    "resource": "LEAP Program",
                    "description": "Learning Engagement & Advancement Programme",
                    "how_to_access": "Contact Office of Student Affairs"
                }
            ],
            "exam_preparation": [
                "Start preparation at least 2 weeks before exams",
                "Create a study schedule and stick to it",
                "Practice past papers and sample questions",
                "Form study groups for discussion and clarification",
                "Take regular breaks to avoid burnout",
                "Get adequate sleep - don't pull all-nighters",
                "Stay hydrated and eat healthy during exam period"
            ]
        }
        
        return Response(resources)
